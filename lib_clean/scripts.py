import json
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from tensor_utils import TensorStorage
from simple_timer import CudaTimer as Timer
from models import AssistanceConfig, ModelType, get_basemodel_name, get_model, get_decoder_layers, get_token_embedding, load_assistant, set_decoder_layers

from pathlib import Path

import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import HfArgumentParser, PreTrainedTokenizer
from datasets import load_dataset, Dataset

@dataclass
class MainConfig:
    model_name: str
    time_execution: bool 
    collect_output: bool 
    benchmark: bool
    output_dir: Optional[str] = field(default=None)
    only_forward: Optional[bool] = False
    data_size: Literal['all', 'small', 'medium', 'large'] = 'all'


def get_data(n_examples: int):
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    return wikitext["train"].select(range(2 * n_examples))

@torch.no_grad
def run_once(data, model: ModelType, tokenizer: PreTrainedTokenizer,
             tokens_per_sample=50, tensor_storage_dir=None, only_forward_passes=True):
    assert torch.cuda.is_available()
    device = "cuda"
    model.eval()
    model = model.to(device)
    max_examples = len(data) // 2
    pbar = tqdm(total=max_examples, desc='Sample inference')
    idx = 0
    for x in data: 
        question: str = x["text"]
        if len(question) == 0:
            continue
        
        tensors = tokenizer(question, return_tensors='pt', return_attention_mask=True)
        input_ids = tensors.input_ids.to(device)
        attention_mask = tensors.attention_mask.to(device)

        if only_forward_passes:
            _ = model(input_ids, attention_mask=attention_mask)
        else:
            _ = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=tokens_per_sample + 1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        if tensor_storage_dir is not None:
            TensorStorage.commit_sample(tensor_storage_dir)

        idx += 1
        pbar.update(1)
        if idx == max_examples:
            break

def collect_output_hooks(model, collect_modules=None, save_uncached=False):
    if collect_modules is None:
        collect_modules = {'block'}

    def save_data(module_key: str, save_hidden_states=False, save_output=False, 
                  save_cached=True, save_uncached=False,
                  is_last_module=False):
        def save_data_hook(layer: nn.Module, args, kwargs, output):
            hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
            is_first_token = TensorStorage.token_idx == 0
            # if (is_first_token and not save_uncached) or (not is_first_token and not save_cached):
            #     return
            
            save_key = module_key
            if is_first_token:
                save_key += '_first'

            emb = None 
            if save_hidden_states:
                emb = hidden_states
            if save_output: 
                emb = output[0] if isinstance(output, tuple) else output

            TensorStorage.save_embedding(emb, save_key, last_module=is_last_module)

        return save_data_hook

    if 'token_embedding' in collect_modules:
        emb = get_token_embedding(model)
        emb.register_forward_hook(save_data('token_embedding', save_output=True, save_hidden_states=False, save_uncached=save_uncached), with_kwargs=True)

    layers = get_decoder_layers(model)
    for layer_idx, layer in enumerate(layers):
        if 'block' in collect_modules:
            layer.register_forward_hook(save_data(f'block{layer_idx}', save_hidden_states=True, save_uncached=save_uncached), with_kwargs=True)
            if layer_idx == len(layers) - 1:
                layer.register_forward_hook(save_data(f'block{layer_idx + 1}', save_output=True, save_uncached=save_uncached, is_last_module=True), with_kwargs=True)
        if 'attn' in collect_modules:
            layer_attn_in = layer.temporal_block if hasattr(layer, 'temporal_block') else layer.self_attn 
            layer_attn_out = layer_attn_in
            layer_attn_in.register_forward_hook(save_data(f'block{layer_idx}_attn_in', save_hidden_states=True, save_uncached=save_uncached), with_kwargs=True)
            layer_attn_out.register_forward_hook(save_data(f'block{layer_idx}_attn_out', save_output=True, save_uncached=save_uncached), with_kwargs=True)
        if 'mlp' in collect_modules:
            layer_mlp_in, layer_mlp_out = None, None
            if hasattr(layer, 'mlp_block'):
                # RecurrentGemma case 
                layer_mlp_in = layer.mlp_block
                layer_mlp_out = layer_mlp_in
            elif hasattr(layer, 'mlp'):
                # Gated models
                layer_mlp_in = layer.mlp 
                layer_mlp_out = layer_mlp_in
            else:
                # Opt case 
                layer_mlp_in = layer.fc1
                layer_mlp_out = layer.fc2
    
            layer_mlp_in.register_forward_hook(save_data(f'block{layer_idx}_mlp_in', save_hidden_states=True, save_uncached=save_uncached), with_kwargs=True)
            layer_mlp_out.register_forward_hook(save_data(f'block{layer_idx}_mlp_out', save_output=True, save_uncached=save_uncached), with_kwargs=True)

def collect_output(model_name: str, output_dir: str, collect_modules=None):
    data = get_data(100)
    model, tokenizer = get_model(model_name) 
    collect_output_hooks(model, collect_modules=collect_modules)

    run_once(data, model, tokenizer, tensor_storage_dir=Path(output_dir))

def time_execution(model_name: str, prompt_size: str, only_forward: bool):
    def time_execution_hooks(layer: nn.Module, timer_key: str):
        layer.register_forward_pre_hook(lambda *args: Timer.register(timer_key))
        layer.register_forward_hook(lambda *args: Timer.commit(timer_key))

    def count_tokens(tokenizer):
        def f(example):
            return {'num_tokens': len(tokenizer.tokenize(example['text']))}
        
        return f

    warmup_size = 10
    data_size = 100 
    if only_forward:
        warmup_size *= 10
        data_size *= 10

    # Benchmarks execution time of different parts of the model 
    warmup_data = get_data(warmup_size)

    model, tokenizer = get_model(model_name) 
    min_num_tokens = {
        'all': 1,
        'small': 50,
        'medium': 240,
        'large': 1000
    }
    max_num_tokens = {
        'all': 2048,
        'small': 60,
        'medium': 260,
        'large': 1100,
    }
    data = (
        load_dataset("Salesforce/wikitext", "wikitext-103-v1")['train']
        .map(count_tokens(tokenizer))
        .filter(lambda example: min_num_tokens[prompt_size] <= example['num_tokens'] <= max_num_tokens[prompt_size])
        .select(range(data_size))
    )
    # Warmup run
    run_once(warmup_data, model, tokenizer)

    # Add the hooks 
    for idx, layer in enumerate(get_decoder_layers(model)):
        time_execution_hooks(layer, f"Decoder Layer {idx}")
        if hasattr(layer, 'temporal_block'):
            # RecurrentGemma case 
            assert False
            time_execution_hooks(layer.temporal_block, 'Temporal Block')
        else:
            # Gated models + OPT 
            time_execution_hooks(layer.self_attn, "Self attention")
            time_execution_hooks(layer.self_attn.q_proj, "q_proj")
            time_execution_hooks(layer.self_attn.k_proj, "k_proj")
            time_execution_hooks(layer.self_attn.v_proj, "v_proj")

        if hasattr(layer, 'mlp_block'):
            # RecurrentGemma case 
            time_execution_hooks(layer.mlp_block, 'MLP')
        elif hasattr(layer, 'mlp'):
            # Gated models
            time_execution_hooks(layer.mlp, 'MLP')
        else:
            # Opt case 
            time_execution_hooks(layer.fc1, "fc1")
            time_execution_hooks(layer.fc2, "fc2")
            time_execution_hooks(layer.activation_fn, "MLP activation")

    # Real run
    run_once(data, model, tokenizer, only_forward_passes=only_forward)
    Timer.print()


def benchmark(model_name: str, assistance_config: AssistanceConfig, use_cache=True):
    # Benchmark model input ingestion and output generation speed
    assert torch.cuda.is_available()
    device = "cuda"
    model, tokenizer = get_model(model_name)
    # Sometimes this is needed to avoid warnings 
    model.generation_config.pad_token_id = tokenizer.eos_token_id 
    model.eval()
    model = model.to(device)
    if assistance_config.assistant_name is not None:
        load_assistant(assistance_config, model, get_basemodel_name(model_name), assistant_use_cache=use_cache)

    def count_tokens(tokenizer):
        def f(example):
            return {'num_tokens': len(tokenizer.tokenize(example['text']))}
        
        return f

    data = (
        load_dataset("Salesforce/wikitext", "wikitext-103-v1")['train']
        .map(count_tokens(tokenizer))
        .filter(lambda example: 240 <= example['num_tokens'] <= 260)
        .select(range(500))
    )

    n_burnin = 25
    input_speeds = []
    output_speeds = []
    output_buffer = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def timer_hook(start=False):
        def register_forward_pass(layer, *args):
            if start:
                # torch.cuda.synchronize()
                start_event.record()
            else:
                end_event.record()
                torch.cuda.synchronize()

                hidden_states = args[0][0]
                n_tokens = hidden_states.size(1) 
                forward_pass_time = start_event.elapsed_time(end_event) / 1000
                speed = n_tokens / forward_pass_time
                if n_tokens > 1:
                    input_speeds.append(speed)
                else:
                    output_buffer.append(speed)
        
        return register_forward_pass

    layers = get_decoder_layers(model)
    layers[0].register_forward_pre_hook(timer_hook(True))
    layers[-1].register_forward_hook(timer_hook(False))

    def compute_speed_metrics(speeds, n_burnin):
        speeds_tensor = torch.tensor(speeds)[n_burnin:]
        return speeds_tensor.mean(), speeds_tensor.std()

    with torch.no_grad():
        for x in tqdm(data):
            question: str = x["text"]

            # print(len(output_speeds), n_burnin)
            # if len(output_speeds) > n_burnin:
            #     print(len(output_speeds), compute_speed_metrics(output_speeds, n_burnin))

            tensors = tokenizer(question, return_tensors='pt', return_attention_mask=True)
            input_ids = tensors.input_ids.to(device)
            n_inputs = input_ids.size(1) 
            attention_mask = tensors.attention_mask.to(device)

            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=use_cache,
            )
            # Does this help give more stable results?
            torch.cuda.empty_cache()
            
            num_tokens_to_generate = 256
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens_to_generate,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=use_cache,
            )
            torch.cuda.empty_cache()
            tokens_generated = output.size(1) - n_inputs
            if tokens_generated == num_tokens_to_generate:
                output_speeds.append(sum(output_buffer) / len(output_buffer))
                output_buffer = []
    

    # with open('./data.json', 'w') as f:
    #     json.dump({
    #         'input_speeds': input_speeds,
    #         'output_speeds': output_speeds
    #     }, f)

    return (
        *compute_speed_metrics(input_speeds, n_burnin),
        *compute_speed_metrics(output_speeds, n_burnin)
    ) 

if __name__ == '__main__':
    config, assistance_config = HfArgumentParser((MainConfig, AssistanceConfig)).parse_args_into_dataclasses()
    if config.time_execution:
        time_execution(config.model_name, config.data_size, config.only_forward)
    if config.collect_output:
        assert config.output_dir is not None 
        collect_output(config.model_name, config.output_dir)
    if config.benchmark:
        input_speed, input_std, output_speed, output_std = benchmark(config.model_name, assistance_config)
        print(f'{input_speed:.2f}+-{input_std:.2f}, {output_speed:.2f}+-{output_std:.2f}')


import json
import time
from dataclasses import dataclass, field
from typing import Optional

from tensor_utils import TensorStorage
from simple_timer import CudaTimer as Timer
from models import ModelType, get_model, get_decoder_layers, set_decoder_layers

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


def get_data(n_examples: int):
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    return wikitext["train"].select(range(2 * n_examples))

@torch.no_grad
def run_once(data, model: ModelType, tokenizer: PreTrainedTokenizer, tokens_per_sample=50, tensor_storage_dir=None):
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

def collect_output_hooks(model, collect_modules=None):
    if collect_modules is None:
        collect_modules = {'block'}

    def save_data(module_key: str, save_hidden_states=False, save_output=False):
        def save_data_hook(layer: nn.Module, args, kwargs, output):
            hidden_states = kwargs.get('hidden_states', None)
            if hidden_states is None:
                hidden_states = args[0]
 
            num_tokens = hidden_states.size(hidden_states.dim() - 2)
            if num_tokens != 1:
                return

            # print(num_tokens, hidden_states.shape)
            if save_hidden_states:
                TensorStorage.save_embedding(hidden_states, module_key)
            if save_output: 
                if isinstance(output, tuple):
                    TensorStorage.save_embedding(output[0], module_key)
                else:
                    TensorStorage.save_embedding(output, module_key)

        return save_data_hook

    layers = get_decoder_layers(model)
    for layer_idx, layer in enumerate(layers):
        if 'block' in collect_modules:
            layer.register_forward_hook(save_data(f'block{layer_idx}', save_hidden_states=True), with_kwargs=True)
            if layer_idx == len(layers) - 1:
                layer.register_forward_hook(save_data(f'block{layer_idx + 1}', save_output=True), with_kwargs=True)
        if 'attn' in collect_modules:
            layer_attn_in = layer.temporal_block if hasattr(layer, 'temporal_block') else layer.self_attn 
            layer_attn_out = layer_attn_in
            layer_attn_in.register_forward_hook(save_data(f'block{layer_idx}_attn_in', save_hidden_states=True), with_kwargs=True)
            layer_attn_out.register_forward_hook(save_data(f'block{layer_idx}_attn_out', save_output=True), with_kwargs=True)
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
    
            layer_mlp_in.register_forward_hook(save_data(f'block{layer_idx}_mlp_in', save_hidden_states=True), with_kwargs=True)
            layer_mlp_out.register_forward_hook(save_data(f'block{layer_idx}_mlp_out', save_output=True), with_kwargs=True)

def collect_output(model_name: str, output_dir: str, collect_modules=None):
    data = get_data(100)
    model, tokenizer = get_model(model_name) 
    collect_output_hooks(model, collect_modules=collect_modules)

    run_once(data, model, tokenizer, tensor_storage_dir=Path(output_dir))

def time_execution(model_name: str):
    def time_execution_hooks(layer: nn.Module, timer_key: str):
        layer.register_forward_pre_hook(lambda *args: Timer.register(timer_key))
        layer.register_forward_hook(lambda *args: Timer.commit(timer_key))

    # Benchmarks execution time of different parts of the model 
    warmup_data = get_data(10)
    data = get_data(100)
    model, tokenizer = get_model(model_name) 
    # Warmup run
    run_once(warmup_data, model, tokenizer)

    # Add the hooks 
    for idx, layer in enumerate(get_decoder_layers(model)):
        time_execution_hooks(layer, f"Decoder Layer {idx}")
        if hasattr(layer, 'temporal_block'):
            # RecurrentGemma case 
            time_execution_hooks(layer.temporal_block, 'Temporal Block')
        else:
            # Gated models + OPT 
            time_execution_hooks(layer.self_attn, "Self attention")

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
    run_once(data, model, tokenizer)
    Timer.print()


def benchmark(model_name: str, use_cache=True):
    # Benchmark model input ingestion and output generation speed
    assert torch.cuda.is_available()
    device = "cuda"
    model, tokenizer = get_model(model_name)
    # Sometimes this is needed to avoid warnings 
    model.generation_config.pad_token_id = tokenizer.eos_token_id 
    model.eval()
    model = model.to(device)
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
    config: MainConfig = HfArgumentParser(MainConfig).parse_args_into_dataclasses()[0]
    if config.time_execution:
        time_execution(config.model_name)
    if config.collect_output:
        assert config.output_dir is not None 
        collect_output(config.model_name, config.output_dir)
    if config.benchmark:
        input_speed, input_std, output_speed, output_std = benchmark(config.model_name)
        print(f'{input_speed:.2f}+-{input_std:.2f}, {output_speed:.2f}+-{output_std:.2f}')


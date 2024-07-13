import time
from dataclasses import dataclass, field
from typing import Optional

from hooks import save_layer_io_hooks, time_execution_hooks
from tensor_utils import TensorStorage
from simple_timer import CudaTimer as Timer
from models import ModelType, get_model

from pathlib import Path

import torch
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

def get_decoder_layers(model: ModelType):
    base_model = model.model
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else base_model
    return decoder.layers

def set_decoder_layers(model: ModelType, layers):
    base_model = model.model
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else base_model
    decoder.layers = layers
    return model

def collect_output(model_name: str, output_dir: str):
    data = get_data(1000)
    model, tokenizer = get_model(model_name) 
    save_layer_io_hooks(get_decoder_layers(model))

    run_once(data, model, tokenizer, tensor_storage_dir=Path(output_dir))

def time_execution(model_name: str):
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

def benchmark(model_name: str):
    # Benchmark model input ingestion and output generation speed
    assert torch.cuda.is_available()
    device = "cuda"
    # A bigger number here because 100 gives very high stds
    data = get_data(10)
    model, tokenizer = get_model(model_name)
    model.eval()
    model = model.to(device)

    # Discard the first n examples to account for gpu warmup time 
    n_burnin = 1
    input_speeds = []
    output_speeds = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for x in tqdm(data):
            question: str = x["text"]
            if len(question) == 0:
                continue

            tensors = tokenizer(question, return_tensors='pt', return_attention_mask=True)
            input_ids = tensors.input_ids.to(device)
            attention_mask = tensors.attention_mask.to(device)
            n_inputs = input_ids.size(1)

            # start = time.perf_counter_ns()
            torch.cuda.synchronize()
            start_event.record()
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
            # time_ns = time.perf_counter_ns() - start
            # input_speed = n_inputs / (time_ns / (10**9))
            end_event.record()
            torch.cuda.synchronize()
            time_ms = start_event.elapsed_time(end_event)
            input_speed = n_inputs / (time_ms / 1000)
            
            input_speeds.append(input_speed)
            # print(f'Ingested {n_inputs} tokens with speed {} t/s')

            num_tokens_to_generate = 1024
            # start = time.perf_counter_ns()
            torch.cuda.synchronize()
            start_event.record()
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens_to_generate,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
            end_event.record()
            torch.cuda.synchronize()
            tokens_generated = output.size(1) - n_inputs
            # time_ns = time.perf_counter_ns() - start
            # output_speed = tokens_generated / (time_ns / (10**9))
            time_ms = start_event.elapsed_time(end_event) 
            output_speed = tokens_generated / (time_ms / 1000)
            output_speeds.append(output_speed)
    
    input_speeds = torch.tensor(input_speeds[n_burnin:]).detach().cpu()
    output_speeds = torch.tensor(output_speeds[n_burnin:]).detach().cpu()
    return (
        input_speeds.mean().item(), input_speeds.std().item(),
        output_speeds.mean().item(), output_speeds.std().item()
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


from dataclasses import dataclass, field
from typing import Optional
from hooks import save_layer_io_hooks, time_execution_hooks, SampleIds
from simple_timer import Timer
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
    output_dir: Optional[str] = field(default=None)

assert torch.cuda.is_available()
device = "cuda"

def get_data(n_examples: int):
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    return wikitext["train"].select(range(2 * n_examples))

@torch.no_grad
def run_once(data, model: ModelType, tokenizer: PreTrainedTokenizer, tokens_per_sample=50):
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

        SampleIds.cur_sample_id = idx
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

        idx += 1
        pbar.update(1)
        if idx == max_examples:
            break

def get_decoder_layers(model: ModelType):
    base_model = model.model
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else base_model
    return decoder.layers

def collect_output(model_name: str, output_dir: str):
    data = get_data(100)
    model, tokenizer = get_model(model_name) 
    save_layer_io_hooks(get_decoder_layers(model), Path(output_dir))

    run_once(data, model, tokenizer)

def time_execution(model_name: str):
    data = get_data(50)
    model, tokenizer = get_model(model_name) 

    for idx, layer in enumerate(get_decoder_layers(model)):
        time_execution_hooks(layer, f"Decoder Layer {idx}")
        time_execution_hooks(layer.self_attn, "Self attention")
        if hasattr(layer, 'mlp'):
            time_execution_hooks(layer.mlp, 'MLP')
        else:
            # Opt case 
            time_execution_hooks(layer.fc1, "fc1")
            time_execution_hooks(layer.fc2, "fc2")
            time_execution_hooks(layer.activation_fn, "MLP activation")

    run_once(data, model, tokenizer)
    Timer.print()

config: MainConfig = HfArgumentParser(MainConfig).parse_args_into_dataclasses()[0]
if config.time_execution:
    time_execution(config.model_name)
if config.collect_output:
    collect_output(config.model_name, config.output_dir)

# time_execution()
# collect_output()

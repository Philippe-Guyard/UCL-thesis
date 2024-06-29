import json
from pathlib import Path
from typing import Any, Tuple

from transformers import (
    OPTForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    PreTrainedTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    GemmaConfig,
    AutoModelForCausalLM,
    GemmaForCausalLM,
    RecurrentGemmaConfig,
    RecurrentGemmaForCausalLM
)

from accelerate import PartialState

def get_opt(model_name: str):
    device_string = PartialState().process_index

    model = OPTForCausalLM.from_pretrained(model_name, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_gated_model(model_name: str, with_relu=False):
    config = None 
    if with_relu:
        config = AutoConfig.from_pretrained(model_name)
        config: Qwen2Config
        config.hidden_act = 'relu'

    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_recurrent_gemma(model_name: str, with_relu=False):
    config = None 
    if with_relu:
        config = AutoConfig.from_pretrained(model_name)
        config: RecurrentGemmaConfig
        config.hidden_activation = 'relu'
    
    device_string = PartialState().process_index
    model = RecurrentGemmaForCausalLM.from_pretrained(model_name, config=config, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def model_getter(model_name: str):
    # NOTE: Order matters here, e.g recurrentgemma has to go before gemma 
    getters_map = [ 
        ('qwen2', get_gated_model),
        ('opt', get_opt),
        ('llama', get_gated_model),
        ('recurrentgemma', get_recurrent_gemma),
        ('gemma', get_gated_model),
    ] 

    for key, func in getters_map:
        if key in model_name.lower():
            return func
    
    assert False, 'Unkown model name'

ModelType = OPTForCausalLM | Qwen2ForCausalLM | GemmaForCausalLM | LlamaForCausalLM | RecurrentGemmaForCausalLM
def get_model(model_name: str, **model_kwargs) -> Tuple[ModelType, PreTrainedTokenizer]:
    model_is_local = model_name.startswith('./') or model_name.startswith('/')
    base_model_str = model_name
    if model_is_local:
        config_path = Path(model_name, 'config.json')
        with open(config_path) as config_file:
            cfg = json.load(config_file)
            base_model_str = cfg['_name_or_path']  

    # Need to fetch this getter separately as loading a llama model is different from loading a recgemma or opt model 
    getter = model_getter(base_model_str)
    return getter(model_name, **model_kwargs)

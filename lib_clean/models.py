from typing import Any, Tuple
from transformers import (
    OPTForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    PreTrainedTokenizer,
    LlamaConfig,
    LlamaForCausalLM
)

from accelerate import PartialState

def get_opt(model_name: str):
    device_string = PartialState().process_index

    model = OPTForCausalLM.from_pretrained(model_name, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_qwen2(model_name: str, with_relu=False):
    config = None 
    if with_relu:
        config = AutoConfig.from_pretrained(model_name)
        config: Qwen2Config
        config.hidden_act = 'relu'

    device_string = PartialState().process_index

    model = Qwen2ForCausalLM.from_pretrained(model_name, config=config, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_llama(model_name: str, with_relu=False):
    config = None 
    if with_relu:
        config = AutoConfig.from_pretrained(model_name)
        config: LlamaConfig 
        config.hidden_act = 'relu'

    device_string = PartialState().process_index

    model = LlamaForCausalLM.from_pretrained(model_name, config=config, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


ModelType = OPTForCausalLM | Qwen2ForCausalLM
def get_model(model_name: str, **model_kwargs) -> Tuple[ModelType, PreTrainedTokenizer]:
    getters_map = {
        'qwen2': get_qwen2,
        'opt': get_opt,
        'llama': get_llama,
    } 

    for key, func in getters_map.items():
        if key in model_name.lower():
            return func(model_name, **model_kwargs)
    
    assert False, 'Unkown model name'


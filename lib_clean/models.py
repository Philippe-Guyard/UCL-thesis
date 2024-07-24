import json
from pathlib import Path
from typing import Any, Tuple

from gpt import GPT2ForLayerPruning, GPTConfig

import torch
import torch.nn as nn
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

def is_local_model_name(model_name: str):
    return Path(model_name).exists() 

def get_basemodel_name(model_name: str, depth=0):
    assert depth <= 5, 'Maximum depth reached when finding base model'
    model_is_local = is_local_model_name(model_name)
    base_name = model_name
    if model_is_local:
        config_path = Path(model_name, 'config.json')
        with open(config_path) as config_file:
            cfg = json.load(config_file)
            base_name = cfg['_name_or_path']  

    if is_local_model_name(base_name):
        base_name = get_basemodel_name(base_name, depth + 1)

    return base_name

ModelType = OPTForCausalLM | Qwen2ForCausalLM | GemmaForCausalLM | LlamaForCausalLM | RecurrentGemmaForCausalLM
def get_model(model_name: str, **model_kwargs) -> Tuple[ModelType, PreTrainedTokenizer]:
    # Check by base_name, load model_name
    base_name = get_basemodel_name(model_name)
    getters_map = [ 
        ('qwen2', get_gated_model),
        ('opt', get_opt),
        ('llama', get_gated_model),
        ('recurrentgemma', get_recurrent_gemma),
        ('gemma', get_gated_model),
    ] 

    for key, func in getters_map:
        if key in base_name.lower():
            return func(model_name, **model_kwargs)
    
    assert False, 'Unkown base model'

class AssistantEvents:
    def __init__(self, model: GPT2ForLayerPruning) -> None:
        self.skip_layers = set()
        self.model = model
        # Perform the computation asynchronously 
        self.compute_stream = torch.cuda.Stream()
        self.compute_event = torch.cuda.Event()

    def compute_skip_layers(self, hidden_states):
        # Perform the computation on a separate CUDA stream
        with torch.cuda.stream(self.compute_stream):
            scores = self.model(hidden_states)
            self.skip_layers = torch.topk(scores, 2, largest=False).indices
            # Record the event to signal the end of computation
            self.compute_event.record(self.compute_stream)
    
class SkippableLayer(nn.Module):
    def __init__(self, layer: nn.Module, idx: int, assistant_events: AssistantEvents):
        super().__init__()
        self.layer = layer
        self.events = assistant_events
        self.layer_idx = idx
    
    def forward(self, *args, **kwargs):
        assert not kwargs['output_attentions']
        assert not kwargs['use_cache']
        if self.layer_idx == 0:
            # Wait for the computation to finish before processing the first layer
            self.events.compute_event.synchronize()
        
        if self.layer_idx in self.events.skip_layers:
            hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
            return (hidden_states,)

        return self.layer(*args, **kwargs)

class EnrichedEmbedding(nn.Module):
    def __init__(self, emb: nn.Module, assistant_events: AssistantEvents):
        super().__init__()
        self.emb = emb
        self.events = assistant_events

    def forward(self, *args, **kwargs):
        hidden_states = self.emb(*args, **kwargs)
        assert hidden_states.size(0) == 1, 'Batch size > 1 not supported yet'
        self.events.compute_skip_layers(hidden_states)
        
        return hidden_states 

def get_decoder_layers(model: ModelType):
    base_model = model.model
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else base_model
    return decoder.layers

def set_decoder_layers(model: ModelType, layers):
    base_model = model.model
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else base_model
    decoder.layers = layers
    return model

def load_assistant(assistant_path: Path, model: ModelType): 
    config: GPTConfig = None 
    teacher_hidden_size = None 
    output_size = None 
    with open(assistant_path.joinpath('assistant_config.json'), 'r') as cfg_file:
        data = json.load(cfg_file)
        config = GPTConfig(**data['model_cfg'])
        teacher_hidden_size = data['teacher_hidden_size']
        output_size = data['output_size']

    assistant_model = GPT2ForLayerPruning(config, teacher_hidden_size, output_size) 
    state_dict = torch.load(assistant_path.joinpath('assistant_state_dict.pt'))
    assistant_model.load_state_dict(state_dict)
    assistant_model.eval()
    # TODO: Proper device
    assistant_model = assistant_model.cuda()

    events = AssistantEvents(assistant_model)
    decoder = model.model
    if hasattr(decoder, 'decoder'): decoder = decoder.decoder
    embedding = decoder.embed_tokens 
    decoder.embed_tokens = EnrichedEmbedding(embedding, events)

    new_layers = []
    for idx, layer in enumerate(get_decoder_layers(model)):
        new_layers.append(SkippableLayer(layer, idx, events))
    
    set_decoder_layers(model, nn.ModuleList(new_layers))
        
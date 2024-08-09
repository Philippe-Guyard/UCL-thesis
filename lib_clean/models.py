from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Tuple
import warnings

from gpt import GPT2ForLayerPruning, GPTConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.cache_utils import DynamicCache

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
        name_field = '_name_or_path'
        if not config_path.exists():
            config_path = Path(model_name, 'adapter_config.json')
            name_field = 'base_model_name_or_path'

        with open(config_path) as config_file:
            cfg = json.load(config_file)
            base_name = cfg[name_field]  

    if is_local_model_name(base_name):
        base_name = get_basemodel_name(base_name, depth + 1)

    return base_name

ModelType = OPTForCausalLM | Qwen2ForCausalLM | GemmaForCausalLM | LlamaForCausalLM | RecurrentGemmaForCausalLM
def distil_prune(model: ModelType, target_sparsity: float, layers_root: Path):
    errors = None 
    with open(layers_root.joinpath('layer_errors.json'), 'r') as f:
        errors = json.load(f)
    
    model_orig_layers = get_decoder_layers(model)
    n_blocks = len(model_orig_layers)
    hidden_size = model.config.hidden_size
    layers = nn.ModuleList((nn.Linear(hidden_size, hidden_size) for _ in range(n_blocks)))
    layers.load_state_dict(torch.load(layers_root.joinpath('distilled_layers.pt')))

    n_to_prune = int(target_sparsity * len(model_orig_layers)) 

    # Add idx, Sort by error, slice
    n_most_linear = sorted(enumerate(errors), key=lambda x: x[1])[:n_to_prune]
    events = AssistantEvents(None, None, False)
    events.skip_layers = set()
    model_dtype = get_token_embedding(model).weight.dtype
    for idx, error in n_most_linear:
        events.skip_layers.add(idx)
        print(f'Replacing layer {idx} by a distilled linear layer')
        layer_cls = None
        if 'opt' in model.config.model_type:
            layer_cls = OPTSkippableLayer
        elif 'llama' in model.config.model_type:
            layer_cls = LlamaSkippableLayer
        elif 'gemma' in model.config.model_type or 'qwen2' in model.config.model_type:
            layer_cls = RoPEModelSkippableLayer
        else:
            assert False

        new_layer = layers[idx].to(dtype=model_dtype)
        model_orig_layers[idx] = layer_cls(new_layer, idx, events, replacement=layers[idx])
    
    set_decoder_layers(model, model_orig_layers)
    return model 

def _get_model(model_name: str, **model_kwargs):
    base_name = get_basemodel_name(model_name)
    getters_map = [ 
        ('qwen2', get_gated_model),
        ('opt', get_opt),
        ('llama', get_gated_model),
        ('recurrentgemma', get_recurrent_gemma),
        ('gemma', get_gated_model),
        ('smollm', get_gated_model),
    ] 

    for key, func in getters_map:
        if key in base_name.lower():
            return func(model_name, **model_kwargs)
    
    assert False, 'Unkown base model'

def get_model(model_name: str, **model_kwargs) -> Tuple[ModelType, PreTrainedTokenizer]:
    # Check by base_name, load model_name
    if model_name.endswith('.json'):
        # This is a prune config
        prune_config = None
        with open(model_name, 'r') as f:
            prune_config = json.load(f)
        
        model_name = prune_config['model_name']
        layers_root = Path(prune_config['layers_root'])
        sparsity_ratio = prune_config['target_sparsity']
        model, tokenizer = _get_model(model_name, **model_kwargs)
        return distil_prune(model, sparsity_ratio, layers_root), tokenizer

    return _get_model(model_name, **model_kwargs)

@dataclass
class AssistanceConfig:
    assistant_name: Optional[str] = None
    # Always prune n layers
    prune_nlayers: Optional[int] = None
    # Prune all layers that are less than distance
    prune_ndist: Optional[float] = None
    prune_maxlayers: Optional[int] = None 

class AssistantEvents:
    def __init__(self, model: GPT2ForLayerPruning, config: AssistanceConfig, use_cache: bool) -> None:
        self.skip_layers = set()
        self.model = model
        # Perform the computation asynchronously 
        self.compute_stream = torch.cuda.Stream()
        self.compute_event = torch.cuda.Event()
        self.config = config
        self.cache = DynamicCache() if use_cache else None

    def compute_skip_layers(self, hidden_states):
        # Perform the computation on a separate CUDA stream
        with torch.cuda.stream(self.compute_stream):
            # NOTE: self.cache is None when use_cache=False, so safe to pass
            scores = self.model(hidden_states, cache=self.cache).squeeze().abs()
            # Need to move them to cpu and convert to set for much faster access 
            # (to check layer_idx in skip_layers)
            if self.config.prune_nlayers is not None:
                skip_indices = torch.topk(scores, self.config.prune_nlayers, largest=False)
                skip_indices = skip_indices.indices
            else:
                skip_indices = torch.nonzero(scores < self.config.prune_ndist).view(-1)
                if self.config.prune_maxlayers is not None and skip_indices.size(0) > self.config.prune_maxlayers:
                    new_scores = scores[skip_indices]
                    best_indices = torch.topk(new_scores, self.config.prune_maxlayers, largest=False).indices
                    # NOTE: Best indices is in the space of [0, len(skip_indices)) and not [0, n_layers)
                    skip_indices = skip_indices[best_indices]

            self.skip_layers = set(skip_indices.cpu().numpy())
            
            # Record the event to signal the end of computation
            self.compute_event.record(self.compute_stream)
    
    def reset_cache(self):
        if self.cache is not None:
            del self.cache
            self.cache = DynamicCache()
        else:
            warnings.warn('Attempting to reset cache on assistant with use_cache=True')
    
class SkippableLayerBase(nn.Module):
    def __init__(self, layer: nn.Module, idx: int, assistant_events: AssistantEvents, replacement=None):
        super().__init__()
        self.layer = layer
        self.events = assistant_events
        self.layer_idx = idx
        self.replacement = replacement
    
    def skip_forward(self, *args, **kwargs):
        # hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
        # return (hidden_states,)
        raise NotImplementedError('Only subclasses of SkippableLayerBase may be skipped')
        
    def forward(self, *args, **kwargs):
        if self.layer_idx == 0:
            # Wait for the computation to finish before processing the first layer
            self.events.compute_event.synchronize()
        
        if self.layer_idx in self.events.skip_layers:
            # (hidden_states, past_key_values)
            skip_out = self.skip_forward(*args, **kwargs)
            # TODO: Ugly solution but whatever 
            if self.replacement is not None:
                skip_out = (self.replacement(skip_out[0]), *skip_out[1:])

            return skip_out
        else:
            return self.layer(*args, **kwargs)

class RoPEModelSkippableLayer(SkippableLayerBase):
    def eval_qkvproj(self, layer_attn, hidden_states):
        query_states = layer_attn.q_proj(hidden_states)
        key_states = layer_attn.k_proj(hidden_states)
        value_states = layer_attn.v_proj(hidden_states)
        return query_states, key_states, value_states

    def recompute_cache(self, layer_attn, hidden_states, past_key_value, cache_position, position_ids):
        # =====================================================
        # Copied from transformers/models/llama/modeling_llama.py with slight modifications 
        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.eval_qkvproj(layer_attn, hidden_states)

        query_states = query_states.view(bsz, q_len, layer_attn.num_heads, layer_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, layer_attn.num_key_value_heads, layer_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, layer_attn.num_key_value_heads, layer_attn.head_dim).transpose(1, 2)

        cos, sin = layer_attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        return past_key_value

    def skip_forward(self, *args, **kwargs):
        # TODO: Can these computations be done on a separate cuda stream?
        assert not kwargs['output_attentions']
        hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']  
        past_key_value = kwargs.get('past_key_value', None)
        outputs = (hidden_states,)
        if past_key_value is not None: 
            position_ids = kwargs['position_ids']
            cache_position = kwargs.get('cache_position', None)
            past_key_value = self.recompute_cache(self.layer.self_attn, hidden_states, past_key_value, cache_position, position_ids)
            outputs += (past_key_value,)
        
        return outputs

class LlamaSkippableLayer(RoPEModelSkippableLayer):
    def eval_qkvproj(self, layer_attn, hidden_states):
        # =====================================================
        # Copied from transformers/models/llama/modeling_llama.py with slight modifications 
        if layer_attn.config.pretraining_tp > 1:
            key_value_slicing = (layer_attn.num_key_value_heads * layer_attn.head_dim) // layer_attn.config.pretraining_tp
            query_slices = layer_attn.q_proj.weight.split(
                (layer_attn.num_heads * layer_attn.head_dim) // layer_attn.config.pretraining_tp, dim=0
            )
            key_slices = layer_attn.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = layer_attn.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(layer_attn.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(layer_attn.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(layer_attn.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
           query_states, key_states, value_states = super().eval_qkvproj(layer_attn, hidden_states) 
    
        return query_states, key_states, value_states


class OPTSkippableLayer(SkippableLayerBase):
    def recompute_cache(self, layer_attn, hidden_states, key_value_states, past_key_value):
        # =====================================================
        # Copied from transformers/models/llama/modeling_llama.py with slight modifications 
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = layer_attn._shape(layer_attn.k_proj(key_value_states), -1, bsz)
            value_states = layer_attn._shape(layer_attn.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = layer_attn._shape(layer_attn.k_proj(hidden_states), -1, bsz)
            value_states = layer_attn._shape(layer_attn.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = layer_attn._shape(layer_attn.k_proj(hidden_states), -1, bsz)
            value_states = layer_attn._shape(layer_attn.v_proj(hidden_states), -1, bsz)

        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        return (key_states, value_states)        

    def skip_forward(self, *args, **kwargs):
        assert not kwargs['output_attentions']
        hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
        use_cache = kwargs.get('use_cache', False)
        past_key_value = kwargs.get('past_key_value', None)
        outputs = (hidden_states,)
        if self.layer.self_attn.is_decoder and (use_cache or past_key_value): 
            key_value_states = kwargs.get('key_value_states', None)
            new_key_value = self.recompute_cache(self.layer.self_attn, hidden_states, key_value_states, past_key_value)
            outputs += (new_key_value,)
        
        return outputs 


class EnrichedEmbedding(nn.Module):
    def __init__(self, emb: nn.Embedding, assistant_events: AssistantEvents):
        super().__init__()
        self.emb = emb
        self.events = assistant_events

    def forward(self, *args, **kwargs):
        hidden_states = self.emb(*args, **kwargs)
        assert hidden_states.size(0) == 1, 'Batch size > 1 not supported yet'
        self.events.compute_skip_layers(hidden_states)
        
        return hidden_states 

    def __getattr__(self, name: str):
        if name in ('emb', 'events', 'forward'):
            return object.__getattribute__(self, name)
        return getattr(self.emb, name)

    def __setattr__(self, name: str, value):
        if name in ('emb', 'events', 'forward'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.emb, name, value)

def get_decoder_layers(model: ModelType):
    base_model = model.model
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else base_model
    return decoder.layers

def set_decoder_layers(model: ModelType, layers):
    base_model = model.model
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else base_model
    decoder.layers = layers
    return model

def get_decoder(model: ModelType):
    decoder = model.model
    if hasattr(decoder, 'decoder'): decoder = decoder.decoder
    return decoder

def get_token_embedding(model: ModelType) -> nn.Embedding:
    decoder = get_decoder(model)
    return decoder.embed_tokens 

def set_token_embedding(model: ModelType, emb: nn.Embedding):
    decoder = get_decoder(model)
    decoder.embed_tokens = emb

def load_assistant(assistant_config: AssistanceConfig, model: ModelType, model_basename: str, assistant_use_cache=True): 
    # It is unclear when to reset the assistant cache. For usage that just includes calling model.generate,
    # we can just overwrite the .generate method to reset the assistant cache together with the main model cache 
    # For libraries like lm-eval, the .forward method is sometimes called directly, and hence it is unclear when to use 
    # assistant cache and when to reset it. To avoid this any overly complicated solutions, we just leave it as a simple 
    # boolean flag here. The assistant overhead is in any case small enough for us to simply ignore it 
    assert assistant_config.prune_ndist or assistant_config.prune_nlayers

    assistant_path = Path(assistant_config.assistant_name)
    print(f'Loading assistant at {assistant_path}')
    model_basename = model_basename.lower()
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

    embedding = get_token_embedding(model)
    # In case we are doing mixed precision inference
    assistant_model = assistant_model.to(dtype=embedding.weight.dtype)
    events = AssistantEvents(assistant_model, assistant_config, use_cache=assistant_use_cache)

    def generate_decorator(f, events: AssistantEvents):
        def wrapper(*args, **kwargs):
            events.reset_cache()
            return f(*args, **kwargs)
        
        return wrapper
    
    model.generate = generate_decorator(model.generate, events)

    set_token_embedding(model, EnrichedEmbedding(embedding, events))

    new_layers = []
    for idx, layer in enumerate(get_decoder_layers(model)):
        if 'llama' in model_basename:
            new_layers.append(LlamaSkippableLayer(layer, idx, events))
        elif 'opt' in model_basename:
            new_layers.append(OPTSkippableLayer(layer, idx, events))
        elif 'gemma' in model_basename or 'qwen2' in model_basename:
            new_layers.append(RoPEModelSkippableLayer(layer, idx, events))
        else:
            assert False, 'unknown model basename'
    
    set_decoder_layers(model, nn.ModuleList(new_layers))
        
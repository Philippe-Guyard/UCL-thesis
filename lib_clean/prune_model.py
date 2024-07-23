import os 
import tempfile
import shutil
from pathlib import Path

from models import get_model, get_decoder_layers, set_decoder_layers
from tensor_utils import ConsecutiveAngularDistances, TensorStorage

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass 
class PruneConfig:
    model: str 
    target_sparsity: float 
    data_dir: str 
    model_out: str 
    strategy: str = field(default='simple')

def _simple_prune(model, n_layers: int, data_dir: str):
    layers = get_decoder_layers(model)
    per_sample_len = len(layers) - n_layers + 1
    per_block = [[] for _ in range(per_sample_len)]
    data_root = Path(data_dir) 
    n_samples = len(list(data_root.joinpath('block0').iterdir()))
    def get_embedding(block_idx, sample_idx, token_idx):
        return TensorStorage.get_embedding(data_root, f'block{block_idx}', sample_idx, token_idx)

    for block_idx in range(per_sample_len):
        for sample_idx in range(n_samples):
            n_tokens = TensorStorage.get_num_tokens(data_root, f'block{block_idx}', sample_idx)
            for token_idx in range(n_tokens):
                embedding = get_embedding(block_idx, sample_idx, token_idx)
                # embedding_plus_n = input to layer block_idx + target_n = output of layer block_idx + target_n - 1
                embedding_plus_n = get_embedding(block_idx + n_layers, sample_idx, token_idx)

                cos_sim = torch.clip(F.cosine_similarity(embedding, embedding_plus_n, dim=-1), -1, 1)
                per_block[block_idx].append(torch.arccos(cos_sim).view(-1) / torch.pi)

                token_idx += 1

    average_distances = torch.tensor([torch.cat(block_dists).mean() for block_dists in per_block])
    for idx, dist in enumerate(average_distances):
        # Make layers 1-indexed
        print(f'Dist(In(Layer {idx + 1}), Out(Layer {idx + n_layers})) = {dist.item():.2f}')

    best_start = torch.argmin(average_distances)
    # The input to layer best_start and the input to layer best_start + target_n is similar => 
    # The output of layer best_start - 1 and the input of layer best_start + target_n is similar => 
    # We can remove layers [best_start, best_start + target_n - 1]  
    print(f'Removing layers in range [{best_start}, {best_start + n_layers - 1}]')
    new_layers = nn.ModuleList(layers[:best_start] + layers[best_start + n_layers:])
    set_decoder_layers(model, new_layers)

    model.config.num_hidden_layers = len(new_layers)
    return model

def iterative_prune(model_name: str, target_sparsity: float):
    from scripts import collect_output

    model, tokenizer = get_model(model_name)
    layers = get_decoder_layers(model)
    target_n = int(len(layers) * target_sparsity)
    print(f'Seeking {target_n} layers to remove')

    def clear_content(path_str: str):
        path = Path(path_str)
        for file in path.iterdir():
            if file.is_file() or file.is_symlink():
                file.unlink()
            else:
                shutil.rmtree(file)

    # Need to init temp dirs at cur directory to avoid errors with unaccessible tmp dir 
    cur_dir = os.getcwd()
    with tempfile.TemporaryDirectory(dir=cur_dir) as model_dir:
        with tempfile.TemporaryDirectory(dir=cur_dir) as data_dir:
            for idx in range(target_n):
                # Make sure to clear the previous iteration 
                cur_model_name = model_name if idx == 0 else model_dir

                clear_content(data_dir)
                collect_output(cur_model_name, data_dir)
                TensorStorage.reset()

                model, _ = get_model(cur_model_name)
                model = _simple_prune(model, 1, data_dir)

                clear_content(model_dir)
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
            
            return model, tokenizer

def simple_prune(model_name: str, target_sparsity: float, data_dir: str):
    model, tokenizer = get_model(model_name)
    layers = get_decoder_layers(model)
    target_n = int(len(layers) * target_sparsity)
    print(f'Seeking {target_n} layers to remove')

    new_model = _simple_prune(model, target_n, data_dir)
    return new_model, tokenizer

    # # Layers go: 1 -> ... -> n -> out, so len(layers) + 1 is the real number of embeddings passing through the net 
    # # We will remove a block of length l, so the last layer to start removal would be n + 1 - l
    # per_sample_len = len(layers) - target_n + 1
    # per_block = [[] for _ in range(per_sample_len)]
    # data_root = Path(data_dir) 
    # n_samples = len(list(data_root.joinpath('block0').iterdir()))
    # for block_idx in range(per_sample_len):
    #     for sample_idx in range(n_samples):
    #         token_idx = 0  
    #         while (
    #             embedding := TensorStorage.get_embedding(data_root, sample_idx, block_idx, token_idx)
    #         ) is not None:
    #             # embedding_plus_n = input to layer block_idx + target_n = output of layer block_idx + target_n - 1
    #             embedding_plus_n = TensorStorage.get_embedding(data_root, sample_idx, block_idx + target_n, token_idx)
    #             if embedding_plus_n is None:
    #                 break

    #             cos_sim = torch.clip(F.cosine_similarity(embedding, embedding_plus_n, dim=-1), -1, 1)
    #             per_block[block_idx].append(torch.arccos(cos_sim).view(-1) / torch.pi)

    #             token_idx += 1

    # average_distances = torch.tensor([torch.cat(block_dists).mean() for block_dists in per_block])
    # for idx, dist in enumerate(average_distances):
    #     # Make layers 1-indexed
    #     print(f'Dist(In(Layer {idx + 1}), Out(Layer {idx + target_n})) = {dist.item():.2f}')

    # best_start = torch.argmin(average_distances)
    # # The input to layer best_start and the input to layer best_start + target_n is similar => 
    # # The output of layer best_start - 1 and the input of layer best_start + target_n is similar => 
    # # We can remove layers [best_start, best_start + target_n - 1]  
    # print(f'Removing layers in range [{best_start}, {best_start + target_n - 1}]')
    # new_layers = nn.ModuleList(layers[:best_start] + layers[best_start + target_n:])
    # set_decoder_layers(model, new_layers)

    # model.config.num_hidden_layers = len(new_layers)
    # return model, tokenizer

config: PruneConfig = HfArgumentParser(PruneConfig).parse_args_into_dataclasses()[0]
print('Got strategy:', config.strategy)
if config.strategy == 'simple':
    model, tokenizer = simple_prune(config.model, config.target_sparsity, config.data_dir) 
elif config.strategy == 'iterative':
    model, tokenizer = iterative_prune(config.model, config.target_sparsity)
else:
    assert False 

model.save_pretrained(config.model_out)
tokenizer.save_pretrained(config.model_out)
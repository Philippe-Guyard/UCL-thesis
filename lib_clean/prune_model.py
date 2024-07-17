from pathlib import Path

from models import get_model
from scripts import get_decoder_layers, set_decoder_layers
from tensor_utils import ConsecutiveAngularDistances, TensorStorage

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass 
class PruneConfig:
    model: str 
    target_sparsity: float 
    data_dir: str 
    model_out: str 

config: PruneConfig = HfArgumentParser(PruneConfig).parse_args_into_dataclasses()[0]
model, tokenizer = get_model(config.model)
layers = get_decoder_layers(model)
target_n = int(len(layers) * config.target_sparsity)
print(f'Seeking {target_n} layers to remove')

# Layers go: 1 -> ... -> n -> out, so len(layers) + 1 is the real number of embeddings passing through the net 
# We will remove a block of length l, so the last layer to start removal would be n + 1 - l
per_sample_len = len(layers) - target_n + 1
per_block = [[] for _ in range(per_sample_len)]
data_root = Path(config.data_dir) 
n_samples = len(list(data_root.joinpath('block0').iterdir()))
for block_idx in range(per_sample_len):
    for sample_idx in range(n_samples):
        token_idx = 0  
        while (
            embedding := TensorStorage.get_embedding(data_root, sample_idx, block_idx, token_idx)
        ) is not None:
            # embedding_plus_n = input to layer block_idx + target_n = output of layer block_idx + target_n - 1
            embedding_plus_n = TensorStorage.get_embedding(data_root, sample_idx, block_idx + target_n, token_idx)
            if embedding_plus_n is None:
                break

            cos_sim = torch.clip(F.cosine_similarity(embedding, embedding_plus_n, dim=-1), -1, 1)
            per_block[block_idx].append(torch.arccos(cos_sim).view(-1) / torch.pi)

            token_idx += 1

average_distances = torch.tensor([torch.cat(block_dists).mean() for block_dists in per_block])
for idx, dist in enumerate(average_distances):
    # Make layers 1-indexed
    print(f'Dist(In(Layer {idx + 1}), Out(Layer {idx + target_n})) = {dist.item():.2f}')

best_start = torch.argmin(average_distances)
# The input to layer best_start and the input to layer best_start + target_n is similar => 
# The output of layer best_start - 1 and the input of layer best_start + target_n is similar => 
# We can remove layers [best_start, best_start + target_n - 1]  
print(f'Removing layers in range [{best_start}, {best_start + target_n - 1}]')
new_layers = nn.ModuleList(layers[:best_start] + layers[best_start + target_n:])
set_decoder_layers(model, new_layers)

model.config.num_hidden_layers = len(new_layers)
model.save_pretrained(config.model_out)
tokenizer.save_pretrained(config.model_out)
model.config.save_pretrained(config.model_out)
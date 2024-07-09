from pathlib import Path

import torch
from models import get_model
from scripts import get_decoder_layers
from tensor_utils import ConsecutiveAngularDistances, TensorStorage

from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass 
class PruneConfig:
    model: str 
    target_sparsity: float 
    data_dir: str 

config: PruneConfig = HfArgumentParser(PruneConfig).parse_args_into_dataclasses()[0]
model, tokenizer = get_model(config.model_name)
layers = get_decoder_layers(model)
target_n = int(len(layers) * config.target_sparsity)

data_root = Path(config.data_dir)

for block_idx in len(range(layers) + 1):
    
# TODO 
TensorStorage.get_embedding()
data = ConsecutiveAngularDistances(Path(config.data_dir), target_n)
per_block = [[] for _ in range(data.per_sample_len)]
for _, dist, block in data:
    per_block[block].append(dist.item())

average_distances = torch.tensor([torch.tensor(block_dists).mean() for block_dists in per_block])
best_start = torch.argmin(average_distances)
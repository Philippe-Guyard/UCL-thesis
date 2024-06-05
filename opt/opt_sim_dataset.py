from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

@dataclass(frozen=True)
class TensorFile:
    block_idx: int
    sample_idx: int
    token_idx: int 
    data_root: Path 
    num_layers: int 

    def __hash__(self):
        return hash((self.block_idx, self.sample_idx, self.token_idx))

    @property
    def path(self):
        return (
            self.data_root 
            .joinpath(f'block{self.block_idx}')
            .joinpath(str(self.sample_idx))
            .joinpath(f'{self.token_idx}.pt')
        )

    @lru_cache(maxsize=300_000)
    def load(self) -> torch.Tensor:
        return torch.load(self.path)

    def advance(self, n_layers: int):
        assert self.block_idx + n_layers < self.num_layers

        return TensorFile(
            block_idx=self.block_idx + n_layers, 
            sample_idx=self.sample_idx,
            token_idx=self.token_idx,
            data_root=self.data_root,
            num_layers=self.num_layers
        )

class OptSimilaritiesDataset(Dataset):
    '''
    Given a dataset of an OptModel per-layer inputs, produce (x, y) pairs such that 
    For some layer index i:
        x is the input to layer i 
        y is the cosine similarity between input at i + 1 and input at i + n + 1  
    Hypothesis: if y is high, we can skip computing layers i + 1 -> i + n + 1
    '''
    def __init__(self, data_root: Path, num_layers: int, num_samples: int, n: int):
        self.data_root = data_root 
        self.num_layers = num_layers 
        self.num_samples = num_samples

        assert n >= 1
        self.n = n
        self.per_sample_len = self.num_layers - n - 1
        assert self.per_sample_len > 0

        sample_lengths = torch.zeros((self.per_sample_len, self.num_samples))
        self.total_len = 0
        for block_id in range(self.per_sample_len):
            for sample_id in range(self.num_samples):
                slen = self._compute_sample_len(block_id, sample_id)
                sample_lengths[block_id, sample_id] = slen
                self.total_len += slen
            
        block_lengths = sample_lengths.sum(dim=1)
        block_lengths: torch.Tensor

        self.block_prefix = block_lengths.cumsum(dim=0)
        self.sample_prefix = sample_lengths.cumsum(dim=1) 

    def _compute_sample_len(self, block_id: int, sample_id: int):
        return len(list(
            self.data_root
            .joinpath(f'block{block_id}')
            .joinpath(str(sample_id))
            .iterdir()
        ))
    
    def _find_in_prefix(self, prefix: torch.Tensor, value) -> Tuple[int, int]:
        idx = torch.searchsorted(prefix, value, right=True)
        prev_values = 0 if idx == 0 else prefix[idx - 1].item()
        return idx.item(), int(prev_values)

    def __len__(self):
        return self.total_len 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert idx <= len(self)

        start_layer_idx, block_offset = self._find_in_prefix(self.block_prefix, idx) 
        sample_idx, sample_offset = self._find_in_prefix(
            prefix=self.sample_prefix[start_layer_idx],
            value=idx - block_offset
        ) 
        token_idx = idx - block_offset - sample_offset 

        x = TensorFile(start_layer_idx, sample_idx, token_idx, self.data_root, self.num_layers)
        v1 = x.advance(1)
        v2 = x.advance(self.n + 1)
        cos_sim = F.cosine_similarity(
            v1.load(),
            v2.load(),
            dim=-1
        ).view(-1)

        return x.load().reshape(-1), cos_sim, torch.tensor([start_layer_idx], dtype=torch.long) 
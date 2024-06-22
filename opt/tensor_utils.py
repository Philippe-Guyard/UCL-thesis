from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset

@lru_cache(maxsize=500)
def _load_tensor(path: Path):
    return torch.load(path)

@dataclass(frozen=True)
class TensorFile:
    sample_idx: int
    token_idx: int

    data_root: Path 
    # (num_layers, embed_dim)
    _data: Optional[torch.Tensor] = field(default=None)
 
    @property
    def data(self) -> torch.Tensor:
        if self._data is None:
            self._data = _load_tensor(self.path)
        
        return self._data

    @property
    def path(self):
        return (
            self.data_root
            .joinpath(f'sample_{self.sample_idx}')
            .joinpath(f'token_{self.token_idx}.pt')
        )

    def block_input(self, block_idx: int) -> torch.Tensor:
        return self.data[block_idx]

class ConsecutiveOutputsDataset(Dataset):
    def __init__(self, data_root: Path, num_layers: int, num_samples: int, n: int):
        self.data_root = data_root 
        self.num_layers = num_layers 
        self.num_samples = num_samples

        assert n >= 0
        self.n = n
        self.per_sample_len = self.num_layers - n - 1
        assert self.per_sample_len > 0

        sample_lengths = 
        self.total_len = 0
        for sample_id in range(self.num_samples):
            slen = self._compute_sample_len(sample_id)
            sample_lengths[block_id, sample_id] = slen
            self.total_len += slen
            
        block_lengths = sample_lengths.sum(dim=1)
        block_lengths: torch.Tensor

        self.block_prefix = block_lengths.cumsum(dim=0)
        self.sample_prefix = sample_lengths.cumsum(dim=1) 

    def _compute_sample_len(self, sample_id: int):
        '''
        If there is a tensor file with sample id sample_id and token_id x 
        then the sample length is at least x 
        '''
        sample_len = 0
        while TensorFile(sample_id, sample_len, self.data_root).path.exists():
            sample_len += 1
        
        return sample_len

    def _find_in_prefix(self, prefix: torch.Tensor, value) -> Tuple[int, int]:
        idx = torch.searchsorted(prefix, value, right=True)
        prev_values = 0 if idx == 0 else prefix[idx - 1].item()
        return idx.item(), int(prev_values)

    def __len__(self):
        return self.total_len 

    def __getitem__(self, idx: int): 
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
        blocks = torch.tensor([start_layer_idx, start_layer_idx + 1, start_layer_idx + self.n + 1], dtype=torch.long) 
        return x.load(), v1.load(), v2.load(), blocks 


class OptSimilaritiesDataset(ConsecutiveOutputsDataset):
    '''
    Given a dataset of an OptModel per-layer inputs, produce (x, y) pairs such that 
    For some layer index i:
        x is the input to layer i 
        y is the cosine similarity between input at i + 1 and input at i + n + 1  
    Hypothesis: if y is high, we can skip computing layers i + 1 -> i + n + 1
    '''
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, v1, v2, blocks = super().__getitem__(idx)
        cos_sim = F.cosine_similarity(v1, v2, dim=-1).view(-1)
        return x.reshape(-1), cos_sim, blocks[0] 
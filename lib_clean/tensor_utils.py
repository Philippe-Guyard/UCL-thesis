from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TensorStorage:
    token_idx: int = 0
    sample_idx: int = 0
    # First dim = block idx, Second dim = token idx  
    _cur_sample: List[List[torch.Tensor]] = []

    @staticmethod
    @lru_cache(maxsize=500)
    def _load_tensor(path: Path):
        return torch.load(path)

    @staticmethod
    def _get_path(data_root: Path, sample_idx: int, block_idx: int) -> Path:
        return ( 
            data_root
            .joinpath(f'block{block_idx}')
            .joinpath(f's{sample_idx}.pt')
        )

    @staticmethod
    def get_embedding(data_root: Path, sample_idx: int, block_idx: int, token_idx: int) -> Optional[torch.Tensor]:
        path = TensorStorage._get_path(data_root, sample_idx, block_idx)
        if not path.exists():
            return None 

        data = TensorStorage._load_tensor(path)
        if token_idx >= data.size(0):
            return None 
        
        return data[token_idx]
    
    @staticmethod
    def commit_sample(data_root: Path):
        for block_idx, block in enumerate(TensorStorage._cur_sample):
            if len(block) == 0:
                continue

            path = TensorStorage._get_path(data_root, TensorStorage.sample_idx, block_idx)
            path.parent.mkdir(exist_ok=True, parents=True)
            # dim=0 here because we do unsqueeze(0) when appending 
            torch.save(torch.cat(block, dim=0), path)

        TensorStorage.token_idx = 0
        TensorStorage._cur_sample = []
        TensorStorage.sample_idx += 1

    @staticmethod
    def save_embedding(data: torch.Tensor, block_idx: int):
        '''
        Assumes that embeddings arrive sequentially 
        '''
        if block_idx == 0:
            TensorStorage.token_idx += 1

        if block_idx <= len(TensorStorage._cur_sample):
            TensorStorage._cur_sample.append([])

        # Do unsqueeze(0) to make sure that the returned tensors are of the same shape as the original hidden states  
        TensorStorage._cur_sample[block_idx].append(data.cpu().unsqueeze(0))
        
class ConsecutiveOutputsDataset(Dataset):
    def __init__(self, data_root: Path, n: int, num_layers: Optional[int]=None, num_samples: Optional[int]=None):
        self.data_root = data_root 

        if num_layers is None:
            num_layers = len(list(data_root.iterdir()))
        self.num_layers = num_layers 

        if num_samples is None:
            num_samples = len(list(data_root.joinpath('block0').iterdir()))
        self.num_samples = num_samples

        assert n >= 0
        self.n = n
        self.per_sample_len = self.num_layers - n - 1
        assert self.per_sample_len > 0

        sample_lengths = torch.zeros((self.per_sample_len, self.num_samples)) 
        self.total_len = 0
        for block_idx in range(self.per_sample_len):
            for sample_id in range(self.num_samples):
                slen = self._compute_sample_len(sample_id, block_idx)
                sample_lengths[block_idx, sample_id] = slen
                self.total_len += slen
            
        block_lengths = sample_lengths.sum(dim=1)
        block_lengths: torch.Tensor

        self.block_prefix = block_lengths.cumsum(dim=0)
        self.sample_prefix = sample_lengths.cumsum(dim=1) 

    def _compute_sample_len(self, sample_id: int, block_idx: int):
        '''
        If there is a tensor file with sample id sample_id and token_id x 
        then the sample length is at least x 
        '''
        sample_len = 0
        while TensorStorage.get_embedding(self.data_root, sample_id, block_idx, sample_len) is not None: 
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
 
        x  = TensorStorage.get_embedding(self.data_root, sample_idx, start_layer_idx + 0, token_idx)
        v1 = TensorStorage.get_embedding(self.data_root, sample_idx, start_layer_idx + 1, token_idx)
        v2 = TensorStorage.get_embedding(self.data_root, sample_idx, start_layer_idx + self.n + 1, token_idx)
        blocks = torch.tensor([start_layer_idx, start_layer_idx + 1, start_layer_idx + self.n + 1], dtype=torch.long) 
        return x, v1, v2, blocks 


class ConsecutiveCosineSimilarities(ConsecutiveOutputsDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, v1, v2, blocks = super().__getitem__(idx)
        cos_sim = F.cosine_similarity(v1, v2, dim=-1).view(-1)
        return x.reshape(-1), cos_sim, blocks[0] 
    
class ConsecutiveAngularDistances(ConsecutiveOutputsDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, v1, v2, blocks = super().__getitem__(idx)
        cos_sim = F.cosine_similarity(v1, v2, dim=-1).view(-1)
        return x.reshape(-1), torch.arccos(cos_sim), blocks[0] 

from pathlib import Path

from sample_ids import SampleIds

import torch
import torch.nn as nn 

# class OutputSaver(nn.Module):
#     def __init__(self, model: nn.Module, transformer_layers: nn.ModuleList, data_root: Path):
#         super().__init__()
#         self.model = model 
#         self.data_root = data_root

#         self.num_layers = len(transformer_layers)

#         for idx, layer in enumerate(transformer_layers):
#             layer.register_forward_hook(
#                 self.save_output_hook(idx),
#                 with_kwargs=True,
#                 always_call=True
#             )

#     def _make_path(self, layer_id: int) -> Path:
#         data_dir = (
#             self.data_root
#             .joinpath(f'block{layer_id}/')
#             .joinpath(SampleIds.cur_sample_id)
#         )
#         data_dir.mkdir(parents=True, exist_ok=True)
#         token_id = len(list(data_dir.iterdir())) 
#         return data_dir.joinpath(f'{token_id}.pt')

#     def save_output_hook(self, layer_id: int):
#         def save_output(layer: nn.Module, input, output):
#             torch.save(input['hidden_states'], self._make_path(layer_id))

#             if layer_id == self.num_layers - 1:
#                 torch.save(output[0], self._make_path(layer_id + 1)) 

#         return save_output

#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)

def add_hooks(layers: nn.ModuleList, data_root: Path):
    def make_path(layer_id: int) -> Path:
        data_dir = (
            data_root
            .joinpath(f'block{layer_id}/')
            .joinpath(str(SampleIds.cur_sample_id))
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        token_id = len(list(data_dir.iterdir())) 
        return data_dir.joinpath(f'{token_id}.pt')

    def save_output_hook(layer_id: int):
        def save_output(layer: nn.Module, args, kwargs, output):
            hidden_states = args[0]
            if hidden_states.size(1) != 1:
                return 

            torch.save(args[0], make_path(layer_id))

            if layer_id == len(layers) - 1:
                torch.save(output[0], make_path(layer_id + 1)) 
        
        return save_output

    for idx, layer in enumerate(layers):
        layer.register_forward_hook(save_output_hook(idx), with_kwargs=True)
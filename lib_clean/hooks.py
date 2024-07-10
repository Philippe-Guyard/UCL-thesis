from pathlib import Path

from simple_timer import Timer
from tensor_utils import TensorStorage

import torch.nn as nn

def save_layer_io_hooks(layers: nn.ModuleList):
    """
    Saves the inputs of all layers for later analysis. The output of the last layer is also saved
    """
    def save_output_hook(layer_id: int):
        def save_output(layer: nn.Module, args, kwargs, output):
            hidden_states = args[0]
            if hidden_states.size(1) != 1:
                return

            TensorStorage.save_embedding(hidden_states, layer_id)

            if layer_id == len(layers) - 1:
                TensorStorage.save_embedding(output[0], layer_id + 1)

        return save_output

    for idx, layer in enumerate(layers):
        layer.register_forward_hook(save_output_hook(idx), with_kwargs=True)


def time_execution_hooks(layer: nn.Module, timer_key: str):
    layer.register_forward_pre_hook(lambda *args: Timer.register(timer_key))
    layer.register_forward_hook(lambda *args: Timer.commit(timer_key))

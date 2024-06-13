from pathlib import Path

from simple_timer import Timer

import torch
import torch.nn as nn


class SampleIds:
    cur_sample_id = 0


def save_layer_io_hooks(layers: nn.ModuleList, data_root: Path):
    """
    Saves the inputs of all layers for later analysis. The output of the last layer is also saved
    """

    def make_path(layer_id: int) -> Path:
        data_dir = data_root.joinpath(f"block{layer_id}/").joinpath(
            str(SampleIds.cur_sample_id)
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        token_id = len(list(data_dir.iterdir()))
        return data_dir.joinpath(f"{token_id}.pt")

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


def time_execution_hooks(layer: nn.Module, timer_key: str):
    layer.register_forward_pre_hook(lambda *args: Timer.register(timer_key))
    layer.register_forward_hook(lambda *args: Timer.commit(timer_key))

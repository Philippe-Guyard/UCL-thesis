from dataclasses import dataclass
import json
from pathlib import Path
import torch
from scripts import collect_output_hooks
from tensor_utils import TensorStorage
from models import get_decoder_layers, get_model

from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import HfArgumentParser

def distil_layers(model_name: str, train_size: int, test_size: int, log_size: int):
    model, tokenizer = get_model(model_name)
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    n_blocks = len(get_decoder_layers(model))
    hidden_size = model.config.hidden_size
    collect_output_hooks(model, save_uncached=True, collect_modules={'block', 'token_embedding'})

    distilled_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(n_blocks))
    losses = [0 for _ in range(n_blocks)]

    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    def filter_nonempty_select(data, final_size):
        return data.select(range(2 * final_size)).filter(lambda x: len(x['text']) > 0).select(range(final_size))

    train_loader = DataLoader(filter_nonempty_select(wikitext['train'], train_size), batch_size=1, shuffle=True)
    test_loader  = DataLoader(filter_nonempty_select(wikitext['test'] , test_size) , batch_size=1, shuffle=False)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(distilled_layers.parameters())

    def get_all_block_inputs(block_idx):
        assert len(TensorStorage._cur_sample[f'block{block_idx}_first']) == 1
        first = TensorStorage._cur_sample[f'block{block_idx}_first'][0].squeeze()
        # If 0 tokens are generated cached_key is not in the sample
        cached_key = f'block{block_idx}'
        if cached_key in TensorStorage._cur_sample:
            cached = torch.cat([
                x.view(1, -1) for x in TensorStorage._cur_sample[cached_key]
            ], dim=0) 
            first = torch.cat([first, cached], dim=0) 
        
        return first

    def generate_data(batch):
        # Make sure we are operating with a clean dataset
        TensorStorage.reset()
        tokens = tokenizer(batch['text'], return_tensors='pt', max_length=2048, truncation=True, padding=False)
        output = model.generate(
            tokens.input_ids.cuda(),
            attention_mask=tokens.attention_mask.cuda(),
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )
        return [
            get_all_block_inputs(block_idx) for block_idx in range(n_blocks + 1)
        ]

    @torch.no_grad()
    def measure_val_loss():
        val_losses = [0 for _ in range(n_blocks)]
        total_num_tokens = 0
        for batch in test_loader:
            block_data = generate_data(batch)
            num_tokens = block_data[0].size(0)
            for block_idx in range(n_blocks):
                x = block_data[block_idx]
                y = block_data[block_idx + 1]
                preds = distilled_layers[block_idx](x)
                # This is just the sum of squared errors effectively 
                val_losses[block_idx] = criterion(preds, y).item() * num_tokens * hidden_size
            total_num_tokens += num_tokens
        
        # Don't divide back by hidden size as we want an idea of MSE per vector 
        return tuple(x / total_num_tokens for x in val_losses)

    for idx, batch in tqdm(enumerate(train_loader), total=train_size):
        if (idx + 1) % log_size == 0:
            per_layer = measure_val_loss()
            for batch_idx in range((n_blocks + 7) // 8):
                loss_batch = per_layer[batch_idx * 8: (batch_idx + 1) * 8]
                format_idx = lambda x: '0' + str(x) if x < 10 else x
                print(*[f'L{format_idx(batch_idx * 8 + idx)}: {x:.2e}' for idx, x in enumerate(loss_batch)])

        # No padding since we have a batch size of 1, but truncate inputs that are too long 
        block_data = generate_data(batch)
        for block_idx in range(n_blocks):
            x = block_data[block_idx]
            y = block_data[block_idx + 1]
            preds = distilled_layers[block_idx](x)
            losses[block_idx] = criterion(preds, y) 

        optim.zero_grad()
        for loss in losses:
            loss.backward()
        optim.step()
    
    # Also return the val loss ~= linearity of the layer
    return distilled_layers, measure_val_loss()

@dataclass
class DistilConfig:
    model_name: str
    layers_out: str
    train_size: int
    test_size: int 
    log_size: int

if __name__ == '__main__':
    config = HfArgumentParser(DistilConfig).parse_args_into_dataclasses()[0]

    layers, errors = distil_layers(config.model_name, config.train_size, config.test_size, config.log_size) 
    print('Layers sorted by linearity:')
    print(*sorted(enumerate(errors), key=lambda x: x[1]))

    layers_root = Path(config.layers_out)
    layers_root.mkdir(exist_ok=True, parents=True)
    with open(layers_root.joinpath('layer_errors.json'), 'w') as f:
        json.dump(errors, f)

    torch.save(layers.state_dict(), layers_root.joinpath('distilled_layers.pt'))
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from models import get_model, get_decoder_layers
from scripts import collect_output_hooks
from tensor_utils import TensorStorage
from gpt import GPT2ForLayerPruning, GPTConfig

import torch
torch.manual_seed(42)
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser

import wandb

@dataclass 
class AssistantConfig:
    run_name: str
    teacher_model: str
    assistant_out: str 
    n_layer: int 
    n_head: int 
    n_embd: int
    gradient_accumulation_steps: int = 16
    train_size: int = 500
    test_size: int = 50
    log_size: int = 100


config: AssistantConfig = HfArgumentParser(AssistantConfig).parse_args_into_dataclasses()[0]
wandb.init(project='UCL thesis', name=config.run_name)
teacher_model, teacher_tokenizer = get_model(config.teacher_model)
n_blocks = len(get_decoder_layers(teacher_model))
collect_output_hooks(teacher_model, save_uncached=True, collect_modules={'block', 'token_embedding'})
# teacher_model = teacher_model.cuda()
teacher_model.eval()

@torch.no_grad
def generate_synthetic_data(batch, device):
    # Make sure we are operating with a clean dataset
    TensorStorage.reset()
    tokens = teacher_tokenizer(batch['text'], return_tensors='pt', )
    output = teacher_model.generate(
        tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=teacher_tokenizer.eos_token_id,
    )
    tokens_generated = output.size(1) - tokens.input_ids.size(1)
    if tokens_generated < 2:
        return None, None 

    synthetic_inputs = []
    synthetic_outputs = []
    # Handle the first token separately as use_cache is False
    synthetic_inputs.append(
        # Squeeze the batch size of 1
        TensorStorage._cur_sample['token_embedding_first'][0].squeeze()
    )
    block_outputs = torch.stack([
        TensorStorage._cur_sample[f'block{block_idx}_first'][0].squeeze()
        for block_idx in range(0, n_blocks + 1)
    ])
    cos_similarities = F.cosine_similarity(block_outputs[:-1], block_outputs[1:], dim=-1).transpose(0, 1)
    cos_similarities = torch.clamp(cos_similarities, -1.0, 1.0)
    synthetic_outputs.append(torch.acos(cos_similarities) / torch.pi)
    for token_idx in range(tokens_generated - 1):
        synthetic_inputs.append(
            TensorStorage._cur_sample['token_embedding'][token_idx].view(1, -1)
        )

        block_outputs = torch.stack([
            TensorStorage._cur_sample[f'block{block_idx}'][token_idx].view(-1)
            for block_idx in range(0, n_blocks + 1)
        ])
        cos_similarities = F.cosine_similarity(block_outputs[:-1], block_outputs[1:])    
        cos_similarities = torch.clamp(cos_similarities, -1.0, 1.0) 
        angular_distances = torch.acos(cos_similarities) / torch.pi
        synthetic_outputs.append(angular_distances.unsqueeze(0))

    # Unsqueeze to simulate batch_size = 1 
    X = torch.cat(synthetic_inputs, dim=0).unsqueeze(0)
    y = torch.cat(synthetic_outputs, dim=0)
    return X.to(device), y.to(device)


wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
def filter_nonempty_select(data, final_size):
    return data.select(range(2 * final_size)).filter(lambda x: len(x['text']) > 0).select(range(final_size))

train_loader = DataLoader(filter_nonempty_select(wikitext['train'], config.train_size), batch_size=1, shuffle=True)
test_loader  = DataLoader(filter_nonempty_select(wikitext['test'] , config.test_size) , batch_size=1, shuffle=False)
# train_loader = DataLoader(wikitext['train'].select(range(config.train_size)), batch_size=1, shuffle=True)
# test_loader  = DataLoader(wikitext['test'].select(range(config.test_size)) , batch_size=1, shuffle=False)

dropout = 0.1
weight_decay = 1e-1 
lr = 6e-4
cfg = GPTConfig(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd, bias=False, dropout=dropout)
model = GPT2ForLayerPruning(cfg, teacher_model.config.hidden_size, teacher_model.config.num_hidden_layers).cuda()
optim = model.configure_optimizers(weight_decay, lr, 'cuda')
criterion = torch.nn.MSELoss()

from tqdm import tqdm

@torch.no_grad()
def measure_val_loss(model, test_loader, criterion):
    model.eval()
    val_loss = 0
    val_len = 0
    for batch in test_loader:
        X, y = generate_synthetic_data(batch, 'cuda')
        if X is None:
            continue

        num_tokens = y.size(0)
        val_len += num_tokens
        preds = model(X, training=True).squeeze()
        val_loss += criterion(preds, y).item() * num_tokens
    
    return val_loss / val_len

model.train()
running_train_loss = 0
running_train_len = 0
total_num_tokens = 0
optim.zero_grad()

for idx, batch in tqdm(enumerate(train_loader), total=config.train_size):
    if (idx + 1) % config.log_size == 0:
        wandb.log({
            'train_loss': running_train_loss / running_train_len,
            'val_loss': measure_val_loss(model, test_loader, criterion)
        }, step=idx)
        # print(f'Train loss: {running_train_loss / running_train_len:.2e}, Val loss: {measure_val_loss(model, test_loader, criterion):.2e}')
        model.train()
        running_train_loss = 0
        running_train_len = 0

    X, y = generate_synthetic_data(batch, 'cuda')
    if X is None:
        continue
 
    preds = model(X, training=True).squeeze()
    loss = criterion(y, preds) 

    num_tokens = y.size(0)
    running_train_len += num_tokens
    running_train_loss += loss.item() * num_tokens
    total_num_tokens += num_tokens

    loss.backward()
    wandb.log({
        'batch_loss': loss.item(),
        'total_tokens': total_num_tokens
    }, step=idx)
    if (idx + 1) % config.gradient_accumulation_steps == 0:
        optim.step()
        optim.zero_grad()

wandb.finish()

assistant_out = Path(config.assistant_out) 
assistant_out.mkdir(exist_ok=True, parents=True)

torch.save(model.state_dict(), assistant_out.joinpath('assistant_state_dict.pt'))
with open(assistant_out.joinpath('assistant_config.json'), 'w') as cfg_file:
    # Good for training, no sense in saving it for later
    cfg.dropout = 0
    json.dump({
        'teacher_model': config.teacher_model,
        'model_cfg': asdict(cfg),
        'teacher_hidden_size': teacher_model.config.hidden_size,
        'output_size': teacher_model.config.num_hidden_layers - 1
    }, cfg_file)
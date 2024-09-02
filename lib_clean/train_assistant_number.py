from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
from typing import Literal, Optional

from models import get_model, get_decoder_layers, set_decoder_layers
from gpt import GPT2ForLayerPruning, GPTConfig

import torch
import torch.nn as nn
torch.manual_seed(42)
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser

import wandb

@dataclass
class TrainingObjective:
    objective: Literal['regression', 'classification'] = 'regression'
    use_distil_target: bool = False
    angular_dist_threshold: Optional[float] = None
    cos_sim_threshold: Optional[float] = None 
    weight_estimation_steps: int = 50

@dataclass 
class AssistantConfig:
    run_name: str
    teacher_model: str
    n_layer: int 
    n_head: int 
    n_embd: int
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    dataset_name: str = 'wikitext'
    learning_rate: float = 6e-4
    train_size: int = 500
    test_size: int = 50
    eval_steps: int = 10
    save_steps: Optional[int] = None 

@torch.no_grad()
def get_targets(model, input_ids, orig_logits, pruning_order, attention_mask):
    """
    Iteratively prune layers in the order of increasing angular distance and check if the prediction changes per token.
    """
    orig_layers = get_decoder_layers(model) 
    num_layers = len(orig_layers)
    bsz, seqlen, _ = orig_logits.shape
    pruned_layers = []
    num_layers_pruned_per_token = torch.zeros((bsz, seqlen), dtype=torch.long, device=input_ids.device)
    checked = torch.zeros((bsz, seqlen), dtype=torch.bool, device=input_ids.device)
    num_checked = attention_mask.sum(dim=1)

    for layer_idx in pruning_order:
        pruned_layers.append(layer_idx)
        new_layers = nn.ModuleList([orig_layers[i] for i in range(num_layers) if i not in pruned_layers])
        set_decoder_layers(model, new_layers)  

        new_logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits

        for batch_idx in range(bsz):
            if num_checked[batch_idx] == seqlen:
                continue

            for token_idx in range(seqlen):
                if checked[batch_idx, token_idx] or attention_mask[batch_idx, token_idx] == 0:
                    continue  
                    
                if new_logits[batch_idx, token_idx].argmax() != orig_logits[batch_idx, token_idx].argmax():
                    checked[batch_idx, token_idx] = True
                    num_layers_pruned_per_token[batch_idx, token_idx] = len(pruned_layers) - 1
                    num_checked[batch_idx] += 1

        if num_checked.sum() == bsz * seqlen: 
            break

    set_decoder_layers(model, orig_layers)
    return num_layers_pruned_per_token  

config, objective_config = HfArgumentParser((AssistantConfig, TrainingObjective)).parse_args_into_dataclasses()
wandb.init(project='UCL thesis', name=config.run_name)
teacher_model, teacher_tokenizer = get_model(config.teacher_model)
# Needed to avoid warnings from qwen2
teacher_model.generation_config.pad_token_id = teacher_tokenizer.eos_token_id
teacher_tokenizer.padding_side = 'left'
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
n_blocks = len(get_decoder_layers(teacher_model))
teacher_model = teacher_model.cuda()
teacher_model.eval()

def select(t: torch.FloatTensor, attention_mask: torch.LongTensor):
    # Given a tensor of shape (bsz, seq_len, *)
    # Return a list of tensors of len bsz of shape defined by attention_mask
    return [t[i, attention_mask[i]] for i in range(t.size(0))]

@torch.no_grad()
def generate_synthetic_data(teacher_model, pruning_order, batch):
    # Make sure we are operating with a clean dataset
    tokens = teacher_tokenizer(batch['text'], return_tensors='pt', max_length=1024, truncation=True, padding=True)
    att_mask = tokens.attention_mask.cuda()
    inps = tokens.input_ids.cuda()
    logits = teacher_model(
        inps, 
        attention_mask=att_mask,
        use_cache=False, 
        past_key_values=None
    ).logits
    # (bsz, seqlen)
    targets = get_targets(teacher_model, inps, logits, pruning_order, att_mask) 
    att_mask = att_mask.bool()
    emb = teacher_model.get_input_embeddings()(inps)
    targets = F.one_hot(targets, num_classes=len(pruning_order)).float()
    return select(emb, att_mask), select(targets, att_mask) 

@torch.no_grad()
def estimate_angular_distances(model, dataloader, num_layers, device, num_samples=500):
    def compute_angular_distance(state1, state2):
        cosine_similarity = F.cosine_similarity(state1, state2, dim=-1, eps=1e-6)
        angular_distance = 1 - cosine_similarity
        return angular_distance.mean(dim=0)  # Mean distance over all tokens

    angular_distances = [[] for _ in range(num_layers - 1)]  # List to store distances for each layer

    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break

        tokens = teacher_tokenizer(batch['text'], return_tensors='pt', max_length=1024, truncation=True, padding='longest')
        input_ids = tokens.input_ids.to(device) 

        outputs = model(input_ids, use_cache=False, past_key_values=None, output_hidden_states=True)
        hidden_states = outputs.hidden_states  

        for i in range(num_layers - 1):
            dist = compute_angular_distance(hidden_states[i], hidden_states[i + 1])
            angular_distances[i].append(dist.cpu())  

    median_angular_distances = []
    for layer_distances in angular_distances:
        concatenated_distances = torch.cat(layer_distances, dim=0)  
        median_distance = torch.median(concatenated_distances, dim=0).values
        median_angular_distances.append(median_distance)

    return torch.stack(median_angular_distances)

class AugmentedLoader:
    '''
    Like a dataloader, but generates data in minibatches and accumulates them for gradient accumulation immediately 
    '''
    def __init__(self, teacher_model, pruning_order, loader: DataLoader, accumulation_steps: int, device='cuda'):
        self.teacher_model = teacher_model
        self.pruning_order = pruning_order
        self.loader = loader 
        self.steps = accumulation_steps
        self.device = device
        self._loader_iter = None

    def __len__(self):
        return (len(self.loader) + self.steps - 1) // self.steps
    
    def __iter__(self):
        self._loader_iter = iter(self.loader)
        return self
    
    def __next__(self):
        assert self._loader_iter is not None 

        X_lst, y_lst = [], []
        try:
            for _ in range(self.steps):
                batch = next(self._loader_iter)
                delta_X, delta_y = generate_synthetic_data(
                    self.teacher_model, 
                    self.pruning_order, 
                    batch
                )
                X_lst += delta_X
                y_lst += delta_y
        except StopIteration:
            pass 

        if len(X_lst) == 0:
            raise StopIteration

        return X_lst, y_lst

def get_loaders(dataset_name: str, train_size, test_size, batch_size):
    def filter_nonempty_select(data, final_size):
        return data.select(range(2 * final_size)).filter(lambda x: len(x['text']) > 0).select(range(final_size))
    if dataset_name == 'wikitext':
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
        train_dataset = filter_nonempty_select(wikitext['train'], train_size)
        test_dataset = filter_nonempty_select(wikitext['test'], test_size)
    elif dataset_name == 'tinystories':
        stories = load_dataset('roneneldan/TinyStories')
        train_dataset = stories['train'].select(range(train_size)) 
        test_dataset = stories['validation'].select(range(test_size))
    elif dataset_name == 'tinytextbooks':
        textbooks = load_dataset('nampdn-ai/tiny-textbooks')
        train_dataset = textbooks['train'].select(range(train_size)) 
        test_dataset = textbooks['test'].select(range(test_size))
    else:
        assert False 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = get_loaders(config.dataset_name, config.train_size, config.test_size, config.batch_size)

dropout = 0.0
weight_decay = 1e-1 
lr = config.learning_rate 
print(f'{lr=}')
cfg = GPTConfig(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd, bias=False, dropout=dropout)
print('Assistant config:')
print(cfg)
model = GPT2ForLayerPruning(cfg, teacher_model.config.hidden_size, teacher_model.config.num_hidden_layers - 1).cuda()
optim = model.configure_optimizers(weight_decay, lr, 'cuda')
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm

@torch.no_grad()
def compute_val_metrics(teacher_model, pruning_order, test_loader, criterion):
    model.eval()
    loss_sum = 0
    loss_cnt = 0
    for batch in test_loader:
        X_lst, y_lst = generate_synthetic_data(teacher_model, pruning_order, batch)
        for X, y in zip(X_lst, y_lst):
            X = X.unsqueeze(0)
            preds = model(X, training=True).squeeze()
            loss_sum += criterion(preds, y).item()
            loss_cnt += 1
    
    return loss_sum / loss_cnt

model.train()
total_num_tokens = 0
optim.zero_grad()

pruning_order = estimate_angular_distances(
    teacher_model, 
    train_loader,
    n_blocks, 
    'cuda', 
    num_samples=objective_config.weight_estimation_steps
).argsort()
print('Estimated pruning order:', pruning_order)

train_loader = AugmentedLoader(teacher_model, pruning_order, train_loader, config.gradient_accumulation_steps)

save_steps = config.save_steps or len(train_loader) + 1
run_root = Path('./runs').joinpath(config.run_name)
run_root.mkdir(exist_ok=True, parents=True)

checkpoints_root = run_root.joinpath('checkpoints')
assistant_out = run_root.joinpath('final') 

def dump_assistant(path: Path):
    path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), path.joinpath('assistant_state_dict.pt'))
    with open(path.joinpath('assistant_config.json'), 'w') as cfg_file:
        # Good for training, no sense in saving it for later
        cfg.dropout = 0
        json.dump({
            'teacher_model': config.teacher_model,
            'model_cfg': asdict(cfg),
            'teacher_hidden_size': teacher_model.config.hidden_size,
            'output_size': n_blocks 
        }, cfg_file)

loss_sum = 0
for step_idx, batch in enumerate(tqdm(train_loader)):
    X_lst, y_lst = batch 
    total_loss = 0
    for X, y in zip(X_lst, y_lst):
        preds = model(X.unsqueeze(0), training=True).squeeze()
        num_tokens = y.size(0)
        loss = criterion(preds, y) 
        total_num_tokens += num_tokens
        loss.backward()
        total_loss += loss.item()

    grad_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue

        param_norm = p.grad.data.norm(2)
        grad_norm += param_norm.item() ** 2

    grad_norm = grad_norm ** (1. / 2)

    wandb.log({
        # Normalize loss by batch size for logging
        'batch_loss': total_loss / len(X_lst), 
        'total_tokens': total_num_tokens,
        'grad_norm': grad_norm
    }, step=step_idx)

    optim.step()
    optim.zero_grad()

    loss_sum += total_loss

    if (step_idx + 1) % save_steps == 0:
        checkpoint = checkpoints_root.joinpath(f'checkpoint-{step_idx + 1}')
        tqdm.write(f'Saving {checkpoint.as_posix()}')
        dump_assistant(checkpoint)

    if (step_idx + 1) % config.eval_steps == 0:
        wandb.log({
            'train_loss': loss_sum / (config.eval_steps * config.batch_size),
            'val_loss': compute_val_metrics(teacher_model, pruning_order, test_loader, criterion) 
        })
        model.train()
        loss_sum = 0

wandb.finish()

dump_assistant(assistant_out)

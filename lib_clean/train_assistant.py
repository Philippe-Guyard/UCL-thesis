from dataclasses import dataclass, asdict
import json
from pathlib import Path

from models import get_model
from scripts import collect_output_hooks
from tensor_utils import TensorStorage
from gpt import GPT2ForLayerPruning, GPTConfig

import torch
torch.manual_seed(42)
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser

@dataclass 
class AssistantConfig:
    teacher_model: str
    assistant_out: str 
    n_layer: int 
    n_head: int 
    n_embd: int
    train_size: int = 250
    test_size: int = 50
    log_size: int = 50

config: AssistantConfig = HfArgumentParser(AssistantConfig).parse_args_into_dataclasses()[0]
teacher_model, teacher_tokenizer = get_model(config.teacher_model)
collect_output_hooks(teacher_model)
# teacher_model = teacher_model.cuda()
teacher_model.eval()

@torch.no_grad
def generate_synthetic_data(batch, device):
    tokens = teacher_tokenizer(batch['text'], return_tensors='pt')
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

    n_blocks = len(TensorStorage._cur_sample) - 1
    synthetic_inputs = []
    synthetic_outputs = []
    # The first token is not saved as use_cache is False 
    for token_idx in range(tokens_generated - 1):
        synthetic_inputs.append(
            TensorStorage._cur_sample['block0'][token_idx].view(-1)
        )

        block_outputs = torch.stack([
            TensorStorage._cur_sample[f'block{block_idx}'][token_idx].view(-1)
            for block_idx in range(1, n_blocks + 1)
        ])
        cos_similarities = F.cosine_similarity(block_outputs[:-1], block_outputs[1:])    
        cos_similarities = torch.clamp(cos_similarities, -1.0, 1.0) 
        angular_distances = torch.acos(cos_similarities) / torch.pi
        synthetic_outputs.append(angular_distances)

    return torch.stack(synthetic_inputs).to(device), torch.stack(synthetic_outputs).to(device)


wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
train_loader = DataLoader(wikitext['train'].select(range(config.train_size)), batch_size=1, shuffle=True)
test_loader = DataLoader(wikitext['test'].select(range(config.test_size)), batch_size=1, shuffle=False)

cfg = GPTConfig(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd)
model = GPT2ForLayerPruning(cfg, teacher_model.config.hidden_size, teacher_model.config.num_hidden_layers - 1).cuda()

optim = torch.optim.AdamW(model.parameters())
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
        preds = model(X).squeeze()
        val_loss += criterion(preds, y).item() * num_tokens
    
    return val_loss / val_len

for idx, batch in tqdm(enumerate(train_loader), total=250):
    if idx % 50 == 0:
        print(f'Val loss: {measure_val_loss(model, test_loader, criterion):.2e}')
        model.train()

    X, y = generate_synthetic_data(batch, 'cuda')
    if X is None:
        continue
 
    preds = model(X).squeeze()
    loss = criterion(y, preds) 
    optim.zero_grad()
    loss.backward()
    optim.step()

assistant_out = Path(config.assistant_out) 
assistant_out.mkdir(exist_ok=True, parents=True)

torch.save(model.state_dict(), assistant_out.joinpath('assistant_state_dict.pt'))
with open(assistant_out.joinpath('assistant_config.json'), 'w') as cfg_file:
    json.dump({
        'teacher_model': config.teacher_model,
        'model_cfg': asdict(cfg),
        'teacher_hidden_size': teacher_model.config.hidden_size,
        'output_size': teacher_model.config.num_hidden_layers - 1
    }, cfg_file)
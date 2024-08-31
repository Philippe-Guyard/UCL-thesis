from collections import defaultdict
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
from typing import Literal, Optional

from models import get_model, get_decoder_layers, set_decoder_layers
from scripts import collect_output_hooks
from tensor_utils import TensorStorage
from gpt import GPT2ForLayerPruning, GPTConfig 
from simple_histogram import SimpleHistogram

import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser

@dataclass 
class AssistantConfig:
    assistant_path: str
    dataset_name: str = 'wikitext'
    learning_rate: float = 1e-3 
    train_size: int = 500
    test_size: int = 50
    eval_steps: int = 10

config = HfArgumentParser(AssistantConfig).parse_args_into_dataclasses()[0]
assistant_root = Path(config.assistant_path)
assistant_cfg = json.loads(assistant_root.joinpath('assistant_config.json').read_text())
gpt_config = GPTConfig(**assistant_cfg['model_cfg'])
teacher_hidden_size = assistant_cfg['teacher_hidden_size']
output_size = assistant_cfg['output_size']

teacher_model, teacher_tokenizer = get_model(assistant_cfg['teacher_model'])
teacher_model.generation_config.pad_token_id = teacher_tokenizer.eos_token_id
teacher_tokenizer.padding_side = 'left'

assistant = GPT2ForLayerPruning(gpt_config, teacher_hidden_size, output_size)
state_dict = torch.load(assistant_root.joinpath('assistant_state_dict.pt'))
assistant.load_state_dict(state_dict)

n_blocks = len(get_decoder_layers(teacher_model))
collect_modules = {'token_embedding'}
collect_output_hooks(teacher_model, save_uncached=True, collect_modules=collect_modules)

teacher_model = teacher_model.cuda()
teacher_model.eval()

assistant = assistant.cuda()
assistant.eval()

def select(t: torch.FloatTensor, attention_mask: torch.LongTensor):
    # Given a tensor of shape (bsz, seq_len, *)
    # Return a list of tensors of len bsz of shape defined by attention_mask
    return [t[i, attention_mask[i]] for i in range(t.size(0))]

def _generate_synthetic_data_angles(batch, device):
    # Make sure we are operating with a clean dataset
    TensorStorage.reset()
    tokens = teacher_tokenizer(batch['text'], return_tensors='pt', max_length=1024, truncation=True, padding=False)
    outs = teacher_model(
        tokens.input_ids.cuda(), 
        use_cache=False, 
        past_key_values=None
    )
    # Handle the first token separately as use_cache is False
    # (bsz, seq_len, H)
    emb = TensorStorage._cur_sample['token_embedding_first'][0]
    assistant_out, assistant_embed = assistant(emb.to(device), training=True, return_block_embed=True)

    return tokens.input_ids.cuda(), outs.logits.cpu(), assistant_out, assistant_embed

@torch.no_grad()
def generate_synthetic_data(batch, device): 
    return _generate_synthetic_data_angles(batch, device)

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

train_loader, test_loader = get_loaders(config.dataset_name, config.train_size, config.test_size, 1)

classifier = nn.Linear(assistant_cfg['model_cfg']['n_embd'], n_blocks).cuda()
sigmoid = nn.Sigmoid()
total_tokens = 0

lr = config.learning_rate 
print(f'{lr=}')

optim = torch.optim.Adam(classifier.parameters(), lr=lr)

from tqdm import tqdm

classifier.train()
optim.zero_grad()

# def compute_loss(input_ids, orig_logits: torch.FloatTensor, pred: torch.LongTensor, assistant_out: torch.FloatTensor):
#     if pred.item() == 0:
#         return 0

#     indices_to_prune = torch.sort(assistant_out).indices[:pred]
#     orig_layers = get_decoder_layers(teacher_model)
#     new_layers = nn.ModuleList([x for idx, x in enumerate(orig_layers) if idx not in indices_to_prune])
#     set_decoder_layers(teacher_model, new_layers)

#     new_logits = teacher_model(
#         input_ids,
#         use_cache=False, 
#         past_key_values=None
#     ).logits[0, -1].cpu()

#     best_token_is_same = new_logits.argmax() == orig_logits.argmax()
#     reward = pred if best_token_is_same else -pred

#     set_decoder_layers(teacher_model, orig_layers)
#     # Maximize reward = minimize -reward for supervised learning 
#     return -reward

# for step_idx, batch in enumerate(tqdm(train_loader)):
#     tokens, orig_logits, assistant_outs, assistant_embeds = generate_synthetic_data(batch, 'cuda')
#     seq_len = tokens.size(1)

#     pred = sigmoid(classifier(assistant_embeds)).argmax(dim=-1)
#     for n_tokens in range(seq_len):
#         loss = compute_loss(tokens[:, :n_tokens], orig_logits[0, n_tokens - 1], pred[n_tokens], assistant_outs[0, n_tokens - 1])
#         print(loss)


@torch.no_grad()
def compute_loss_for_pruned_indices(input_ids, orig_logits, pred_indices, assistant_outs, token_indices):
    indices_to_prune = pred_indices
    orig_layers = get_decoder_layers(teacher_model)
    new_layers = nn.ModuleList([x for idx, x in enumerate(orig_layers) if idx not in indices_to_prune])
    set_decoder_layers(teacher_model, new_layers)

    # Run forward pass with pruned model without gradients
    new_logits = teacher_model(input_ids, use_cache=False, past_key_values=None).logits

    set_decoder_layers(teacher_model, orig_layers)  # Restore original model

    # Compute loss with gradients
    rewards = torch.zeros(len(token_indices)).cuda()
    for idx, token_idx in enumerate(token_indices):
        new_logit = new_logits[0, token_idx]
        orig_logit = orig_logits[0, token_idx]
        best_token_is_same = new_logit.argmax() == orig_logit.argmax()
        reward = len(pred_indices) if best_token_is_same else -len(pred_indices)
        rewards[idx] = reward

    return rewards

for step_idx, batch in enumerate(tqdm(train_loader)):
    input_ids, orig_logits, assistant_outs, assistant_embeds = generate_synthetic_data(batch, 'cuda')

    outputs = classifier(assistant_embeds.squeeze())
    probabilities = torch.softmax(outputs, dim=1) 
    log_probabilities = torch.log(probabilities)
    pred = torch.multinomial(probabilities, num_samples=1).squeeze()
    # pred = sigmoid(classifier(assistant_embeds)).argmax(dim=-1).squeeze()

    seq_len = input_ids.size(1)

    # Group by set of pruning indices
    pruning_groups = defaultdict(list)
    for token_idx in range(seq_len):
        pruning_indices = torch.sort(assistant_outs[:, token_idx]).indices[:pred[token_idx].item()].squeeze()
        pruning_indices_tuple = tuple(pruning_indices.cpu().numpy()) 
        pruning_groups[pruning_indices_tuple].append(token_idx)

    all_rewards = torch.zeros(seq_len).cuda()
    # Iterate over groups and compute loss
    for pruning_indices, token_indices in pruning_groups.items():
        rewards = compute_loss_for_pruned_indices(input_ids, orig_logits, pruning_indices, assistant_outs, token_indices)
        index_tensor = torch.tensor(token_indices, dtype=torch.long).cuda()
        all_rewards[index_tensor] = rewards

    print(f'Mean reward: {all_rewards.mean():.2f}')
    selected_log_probs = log_probabilities.gather(1, pred.unsqueeze(1)).squeeze() 
    loss = - (all_rewards * selected_log_probs).mean() 
    loss.backward()
    optim.step()
    optim.zero_grad()
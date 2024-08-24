import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from gpt import GPT2ForLayerPruning, GPTConfig
from scripts import collect_output_hooks
from simple_histogram import SimpleHistogram
from tensor_utils import TensorStorage
from models import AssistanceConfig, AssistantEvents, get_decoder_layers, get_model
from datasets import load_dataset


assistant_path = Path('./runs/gemma_assistant_p100_new/final')
model_path = 'google/gemma-2b'

model, tokenizer = get_model(model_path)
n_blocks = len(get_decoder_layers(model))

def get_loaders(dataset_name: str, train_size, test_size):
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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

train_data, _ = get_loaders('tinystories', 10, 0) 

with open(assistant_path.joinpath('assistant_config.json'), 'r') as cfg_file:
    data = json.load(cfg_file)
    config = GPTConfig(**data['model_cfg'])
    teacher_hidden_size = data['teacher_hidden_size']
    output_size = data['output_size']

histograms_dir = assistant_path.joinpath('histograms')
histograms = None
if histograms_dir.exists():
    n_layers = len(get_decoder_layers(model))
    histograms = [SimpleHistogram() for _ in range(n_layers)]
    for idx, hist in enumerate(histograms):
        hist.load_from_file(histograms_dir.joinpath(f'histogram_{idx}.npy'))

assistant_model = GPT2ForLayerPruning(config, teacher_hidden_size, output_size) 
state_dict = torch.load(assistant_path.joinpath('assistant_state_dict.pt'))
assistant_model.load_state_dict(state_dict)
assistant_model.eval()
assistant_model = assistant_model.cuda()

events = AssistantEvents(assistant_model, AssistanceConfig('', prune_nlayers=4), use_cache=False, histograms=histograms)

collect_output_hooks(model, save_uncached=True, collect_modules={'token_embedding', 'block'})

def print_metric(y_true, y_pred, k=3):
    top_k_true_values, top_k_true_indices = torch.topk(y_true, k, largest=False, sorted=True)
    top_k_pred_values, top_k_pred_indices = torch.topk(y_pred, k, largest=False, sorted=True)

    # Calculate the MSE for the top-k predicted values
    top_k_pred_at_true_indices = y_pred.gather(1, top_k_true_indices)
    top_k_mse = torch.mean((top_k_pred_at_true_indices - top_k_true_values) ** 2).item()
    
    mse = torch.mean((y_true[top_k_true_indices] - y_pred[top_k_pred_indices]) ** 2)

    # Print the results
    print(f'y_true: {[round(x, 3) for x in y_true.view(-1).tolist()]}')
    print(f'y_pred: {[round(x, 3) for x in y_pred.view(-1).tolist()]}')
    print(f'{k} best layers (true indices): {top_k_true_indices.tolist()}')
    print(f'{k} best predicted layers (predicted indices): {top_k_pred_indices.tolist()}')
    print(f'MSE at true indices: {mse}')
    print(f'Top-{k} MSE: {top_k_mse}')

with torch.no_grad():
    for idx, x in enumerate(train_data):
        print('=' * 30)
        print(f'Datapoint {idx}')
        tokens = tokenizer(x['text'], return_tensors='pt', max_length=2048, truncation=True)

        TensorStorage.reset()
        # No padding since we have a batch size of 1, but truncate inputs that are too long 
        output = model.generate(
            tokens.input_ids.cuda(),
            attention_mask=tokens.attention_mask.cuda(),
            max_new_tokens=5,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )
        tokens_generated = output.size(1) - tokens.input_ids.size(1)

        print('=' * 10)
        print('Token 0')
        X = TensorStorage._cur_sample['token_embedding_first'][0].squeeze(dim=0).cuda()
        block_outputs = torch.stack([
            TensorStorage._cur_sample[f'block{block_idx}_first'][0].squeeze()
            for block_idx in range(0, n_blocks + 1)
        ])
        cos_similarities = F.cosine_similarity(block_outputs[:-1], block_outputs[1:], dim=-1).transpose(0, 1)
        cos_similarities = torch.clamp(cos_similarities, -1.0, 1.0)
        y_true = (torch.acos(cos_similarities) / torch.pi)[-1].unsqueeze(0)
        events.compute_scores(X)
        events.compute_event.synchronize()
        y_pred = events.scores.cpu()
        # y_pred = assistant_model(X).cpu()
        print_metric(y_true, y_pred)
        skip_layers = events.compute_skip_layers() 
        print('Predicted skip layers:', skip_layers)
        for token_idx in range(tokens_generated - 1):
            print(f'Token {token_idx + 1}')
            delta_x = TensorStorage._cur_sample['token_embedding'][token_idx][0].cuda()
            X = torch.cat([X, delta_x], dim=1)

            block_outputs = torch.stack([
                TensorStorage._cur_sample[f'block{block_idx}'][token_idx].view(-1)
                for block_idx in range(0, n_blocks + 1)
            ])
            cos_similarities = F.cosine_similarity(block_outputs[:-1], block_outputs[1:])    
            cos_similarities = torch.clamp(cos_similarities, -1.0, 1.0) 
            angular_distances = torch.acos(cos_similarities) / torch.pi
            y_true = angular_distances.unsqueeze(0)

            y_pred = assistant_model(X).cpu()
            print_metric(y_true, y_pred)

        

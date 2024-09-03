import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from models import get_decoder_layers, set_decoder_layers, get_model
from scripts import get_data

import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

from contextlib import contextmanager
from pathlib import Path

ROOT_FOLDER = Path('./')

@contextmanager
def save_plot(filepath):
    """A context manager function that creates a plt figure,
    saves it under the provided filepath, and closes the figure."""
    
    fig = plt.figure()  # Create a new figure
    try:
        yield fig  # Yield the figure for plotting
    finally:
        plt.savefig(ROOT_FOLDER.joinpath(filepath))  # Save the figure
        plt.close(fig)  # Close the figure

data = get_data(75)
data = data.filter(lambda x: len(x['text']) > 0) 
model, tokenizer = get_model('facebook/opt-125m')  
model.cuda()
model.eval()

def collect_hidden_states(model, input_ids):
    """
    Perform a forward pass through the model and collect hidden states for each layer.
    """
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple of hidden states from all layers
    return hidden_states

def compute_angular_distance(state1, state2):
    """
    Compute the angular distance (1 - cosine similarity) between two hidden states.
    """
    cosine_similarity = F.cosine_similarity(state1, state2, dim=-1, eps=1e-6)
    angular_distance = 1 - cosine_similarity
    return angular_distance.mean().item()  # Mean distance over all tokens

def prune_layers_and_check_prediction_per_token(model, input_ids, orig_p, hidden_states, pruning_order):
    """
    Iteratively prune layers in the order of increasing angular distance and check if the prediction changes per token.
    """
    orig_layers = get_decoder_layers(model)  # Get original layers of the model
    num_layers = len(orig_layers)
    pruned_layers = []
    num_layers_pruned_per_token = torch.zeros(orig_p.size(1), dtype=torch.int32, device=input_ids.device)
    checked = torch.zeros(orig_p.size(1), dtype=torch.bool, device=input_ids.device)
    num_checked = 0

    orig_tokens = torch.multinomial(orig_p.squeeze(0), num_samples=1)

    # Iterate over layers in the order of increasing angular distance
    for layer_idx in pruning_order:
        pruned_layers.append(layer_idx)
        new_layers = nn.ModuleList([orig_layers[i] for i in range(num_layers) if i not in pruned_layers])
        set_decoder_layers(model, new_layers)  # Prune the model

        # Perform a forward pass with the pruned model
        new_logits = model(input_ids).logits
        # new_p = torch.softmax(new_logits, dim=-1).squeeze(0)

        # Check if the predicted token changes for each token position
        for token_idx in range(orig_p.size(1)):
            if checked[token_idx]:
                continue  
                
            # new_token = torch.multinomial(new_p[token_idx], num_samples=1) 
            # orig_token = orig_tokens[token_idx] 
            if new_logits[0, token_idx].argmax() != orig_p[0, token_idx].argmax():
                # num_layers_pruned_per_token[token_idx] = len(pruned_layers)  # Record number of layers pruned
            # if new_token != orig_token:
                checked[token_idx] = True
                num_layers_pruned_per_token[token_idx] = len(pruned_layers) - 1
                num_checked += 1


        # If all tokens have had their prediction changed, stop early
        if num_checked == orig_p.size(1):
            break

    set_decoder_layers(model, orig_layers)
    return num_layers_pruned_per_token  # Return the number of layers pruned until prediction change for each token

def evaluate_layer_pruning_strategy_per_token(model, prompts, tokenizer):
    """
    Given a list of prompts, evaluate the layer pruning strategy and record the number of layers pruned per token until prediction change.
    """
    results = []
    for prompt in prompts:
        # Tokenize the input prompt
        tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True, padding=False)
        input_ids = tokens.input_ids.cuda()

        # Collect original hidden states and prediction
        hidden_states = collect_hidden_states(model, input_ids)
        orig_logits = model(input_ids).logits

        # Compute angular distances between consecutive hidden states for each layer
        angular_distances = []
        for i in range(1, len(hidden_states)):
            dist = compute_angular_distance(hidden_states[i-1], hidden_states[i])
            angular_distances.append(dist)

        # Convert to a PyTorch tensor and determine the pruning order (ascending angular distances)
        angular_distances = torch.tensor(angular_distances, device=input_ids.device)
        pruning_order = torch.argsort(angular_distances)

        # Prune layers and check when the prediction changes for each token
        orig_p = torch.softmax(orig_logits, dim=-1)
        num_layers_pruned_per_token = prune_layers_and_check_prediction_per_token(
            model, input_ids, orig_p, hidden_states, pruning_order
        )

        results.append(num_layers_pruned_per_token.cpu())

        # # Record the result for each token
        # for token_idx, num_pruned in enumerate(num_layers_pruned_per_token):
        #     token_text = tokenizer.decode(input_ids[0, token_idx])
        #     results['prompt'].append(prompt)
        #     results['token'].append(token_text)
        #     results['layers_pruned'].append(num_pruned.item())

    return results

prompts = [x['text'] for x in data]
# Evaluate the pruning strategy per token
results = evaluate_layer_pruning_strategy_per_token(model, prompts, tokenizer)

df = pd.DataFrame({'Layers Removed': torch.cat(results)})

with save_plot('layers_removed.png'):
    sns.countplot(df, x='Layers Removed')
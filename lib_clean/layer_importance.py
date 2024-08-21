# %%
from scripts import get_data
from models import get_model, get_decoder_layers, set_decoder_layers
import torch
import torch.nn as nn 
import torch.nn.functional as F
from tqdm.notebook import tqdm
from collections import Counter
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

# %%
model, tokenizer = get_model('facebook/opt-125m')
data = get_data(100)
data = data.filter(lambda x: len(x['text']) > 0) 
# %%
@torch.no_grad()
def get_angular_dist(model, data, skip_layers=1): 
    dists = []
    total_tokens = 0
    for x in data:
        tokens = tokenizer(x['text'], return_tensors='pt')
        total_tokens += tokens.input_ids.size(1)
        hidden_states = model(tokens.input_ids.cuda(), output_hidden_states=True).hidden_states 
        # (n_blocks + 1, seq_len, hidden_dim)
        hidden_states = torch.cat(hidden_states, dim=0)
        # (n_blocks - skip_layers + 1, seq_len)
        cos_sim = F.cosine_similarity(hidden_states[:-skip_layers], hidden_states[skip_layers:], dim=-1)
        dist = torch.arccos(cos_sim) / torch.pi

        dists.append(dist.transpose(0, 1).cpu())
    
    return torch.cat(dists, dim=0)

# %%
all_losses = get_angular_dist(model, data)
print(f'{all_losses.shape=}')

# %%
means = torch.zeros(all_losses.size(1))
for idx in range(all_losses.size(1)):
    means[idx] = all_losses[:, idx].mean() 
    print(idx, means[idx])

best_idx = means.argmin()
worst_idx = means[1:-1].argmax() + 1
print(f'Best layer is {best_idx}, worst non-first non-last layer is {worst_idx}')

# %%
from collections import Counter
import pandas as pd 
import seaborn as sns 

def plot_topk_distribution(all_losses, k):
    c = Counter(torch.topk(all_losses, k, largest=False).indices.flatten().cpu().numpy())
    df = pd.DataFrame(list(c.items()), columns=['Element', 'Count'])
    sns.barplot(x='Element', y='Count', data=df)

# %%
top_3_values = torch.topk(all_losses, 5, largest=False).values
# with save_plot()
sns.displot(top_3_values.cpu().numpy())

# %%
sns.displot(top_3_values[:, 0].cpu().numpy())

# %%
with save_plot('best_layer_dist.png'):
    plot_topk_distribution(all_losses, 1)


# %%
with save_plot('best_3_layer_dist.png'):
    plot_topk_distribution(all_losses, 3)

# %%
from tensor_utils import TensorStorage

def collect_output_hooks(model, collect_modules=None, save_uncached=False):
    if collect_modules is None:
        collect_modules = {'block', 'mlp', 'attn'}

    def save_data(module_key: str, save_hidden_states=False, save_output=False, 
                  save_cached=True, save_uncached=False,
                  is_last_module=False):
        def save_data_hook(layer: nn.Module, args, kwargs, output):
            hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
            is_first_token = TensorStorage.token_idx == 0
            # if (is_first_token and not save_uncached) or (not is_first_token and not save_cached):
            #     return
            
            save_key = module_key
            if is_first_token:
                save_key += '_first'

            emb = None 
            if save_hidden_states:
                emb = hidden_states
            if save_output: 
                emb = output[0] if isinstance(output, tuple) else output

            TensorStorage.save_embedding(emb, save_key, last_module=is_last_module)

        return save_data_hook

    handles = []
    def reg(module, hook):
        handles.append(module.register_forward_hook(hook, with_kwargs=True))

    layers = get_decoder_layers(model)
    for layer_idx, layer in enumerate(layers):
        if 'block' in collect_modules:
            hook = save_data(f'block{layer_idx}', save_hidden_states=True, save_uncached=save_uncached)
            reg(layer, hook)
            if layer_idx == len(layers) - 1:
                hook = save_data(f'block{layer_idx + 1}', save_output=True, save_uncached=save_uncached, is_last_module=True)
                reg(layer, hook)
        if 'attn' in collect_modules:
            layer_attn_in = layer.temporal_block if hasattr(layer, 'temporal_block') else layer.self_attn 
            layer_attn_out = layer_attn_in
            hook_in = save_data(f'block{layer_idx}_attn_in', save_hidden_states=True, save_uncached=save_uncached)
            hook_out = save_data(f'block{layer_idx}_attn_out', save_output=True, save_uncached=save_uncached)
            reg(layer_attn_in, hook_in)
            reg(layer_attn_out, hook_out)
        if 'mlp' in collect_modules:
            layer_mlp_in, layer_mlp_out = None, None
            if hasattr(layer, 'mlp_block'):
                # RecurrentGemma case 
                layer_mlp_in = layer.mlp_block
                layer_mlp_out = layer_mlp_in
            elif hasattr(layer, 'mlp'):
                # Gated models
                layer_mlp_in = layer.mlp 
                layer_mlp_out = layer_mlp_in
            else:
                # Opt case 
                layer_mlp_in = layer.fc1
                layer_mlp_out = layer.fc2
    
            hook_in  = save_data(f'block{layer_idx}_mlp_in', save_hidden_states=True, save_uncached=save_uncached)
            hook_out = save_data(f'block{layer_idx}_mlp_out', save_output=True, save_uncached=save_uncached)
            reg(layer_mlp_in, hook_in)
            reg(layer_mlp_out, hook_out)

    return handles

# %%
@torch.no_grad()
def get_tokens(data, index):
    pref = 0
    result = None 
    assert index >= 0
    for x in data:
        tokens = tokenizer(x['text'], return_tensors='pt')
        n_tokens = tokens.input_ids.size(1)  
        if pref + n_tokens <= index:
            pref += n_tokens
            continue

        # index < pref + n_tokens 
        handles = collect_output_hooks(model, save_uncached=True)
        model(tokens.input_ids.cuda(), use_cache=False)
        
        token_idx = index - pref
        result = {
            # The token is always 0 since only 1 forward pass 
            # Take token 0, squeeze the bsz dimension, then take actual token_idx
            key: TensorStorage._cur_sample[key][0].squeeze(dim=0)[token_idx]
            for key in TensorStorage._cur_sample
            if key.endswith('_first') # This is more of an assert really
        } 
        TensorStorage.reset() 

        for handle in handles:
            handle.remove()
        
        # We found the index 
        break 
    
    return result

# %%
for key, item in get_tokens(data, 0).items():
    print(key, item.shape)
    break

# %%
for key, item in get_tokens(data, all_losses.size(0) - 1).items():
    print(key, item.shape)
    break

# %%
assert get_tokens(data, all_losses.size(0)) is None 

# %%
def compute_metric(tokens, key_in, key_out, residual=False, metric='norm_ratio'):
    x_in = tokens[key_in + '_first']
    x_out = tokens[key_out + '_first']
    x_in: torch.Tensor
    if residual:
        x_out = x_out + x_in
    
    if metric == 'norm_ratio':
        return x_out.norm() / x_in.norm()
    elif metric == 'angular_distance':
        return torch.arccos(F.cosine_similarity(x_out, x_in, dim=-1)) / torch.pi
    else:
        assert False, f'Unknown metric {metric}'

def get_metric(data, indices, key_in, key_out, residual=False, metric='norm_ratio'):
    metrics = torch.zeros_like(indices, dtype=torch.float)
    for i, index in enumerate(indices):
        tokens = get_tokens(data, index)    
        metrics[i] = compute_metric(tokens, key_in, key_out, residual=residual, metric=metric).cpu()

    return metrics.numpy()

def get_metrics_df(data, layer_idx, key_in, key_out, residual=False, metrics=None):
    if metrics is None:
        metrics = ['norm_ratio']

    cutoff = all_losses[:, layer_idx].quantile(0.25)
    good_indices = torch.nonzero(all_losses[:, layer_idx] < cutoff).view(-1)
    bad_indices = torch.nonzero(all_losses[:, layer_idx] >= cutoff).view(-1)[:good_indices.numel()]

    good_values = {metric: None for metric in metrics}
    bad_values = {metric: None for metric in metrics}
    for metric in metrics:
        good_values[metric] = get_metric(data, good_indices, key_in, key_out, residual=residual,metric=metric)
        bad_values[metric] = get_metric(data, bad_indices, key_in, key_out, residual=residual,metric=metric)

    good_df = pd.DataFrame({'category': 'good', **good_values})
    bad_df = pd.DataFrame({'category': 'bad', **bad_values})
    df = pd.concat((good_df, bad_df), ignore_index=True)

    return df 

def save_all(data, layer_idx, suff):
    df = get_metrics_df(data, layer_idx, f'block{layer_idx}', f'block{layer_idx + 1}', metrics=['norm_ratio', 'angular_distance'])

    with save_plot(f'block_norm_ratio_{suff}.png'):
        sns.displot(df, x='norm_ratio', hue='category')

    df = get_metrics_df(data, layer_idx, f'block{layer_idx}_mlp_in', f'block{layer_idx}_mlp_out', metrics=['norm_ratio', 'angular_distance'], residual=True)

    with save_plot(f'mlp_norm_ratio_{suff}.png'):
        sns.displot(df, x='norm_ratio', hue='category')
    with save_plot(f'mlp_ang_dist_{suff}.png'):
        sns.displot(df, x='angular_distance', hue='category')

    df = get_metrics_df(data, layer_idx, f'block{layer_idx}_attn_in', f'block{layer_idx}_attn_out', metrics=['norm_ratio', 'angular_distance'], residual=True)

    with save_plot(f'attn_norm_ratio_{suff}.png'):
        sns.displot(df, x='norm_ratio', hue='category')
    with save_plot(f'attn_ang_dist_{suff}.png'):
        sns.displot(df, x='angular_distance', hue='category')

save_all(data, best_idx, 'best')
save_all(data, worst_idx, 'worst')

def distillation_loss(student_logits, teacher_logits, per_batch=False, temperature=1.0):
    """
    Compute the distillation loss between student and teacher logits.

    Parameters:
    - student_logits: Logits from the student model (tensor of shape [batch_size, vocab_size])
    - teacher_logits: Logits from the teacher model (tensor of shape [batch_size, vocab_size])
    - temperature: Temperature for softening the probability distributions

    Returns:
    - Loss value (scalar tensor)
    """
    # Soften the probabilities with temperature
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # Compute the KL divergence loss
    reduction = 'none' if per_batch else 'batchmean'
    kldiv_loss = F.kl_div(student_probs, teacher_probs, reduction=reduction)
    if per_batch:
        kldiv_loss = kldiv_loss.sum(dim=1)
    
    kldiv_loss *= (temperature ** 2)
    return kldiv_loss

class PassthroughMLP(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
        return hidden_states

class PassthroughAttn(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
        return (hidden_states, None, None)

@torch.no_grad()
def get_distill_loss(model, data, sublayer_indices, prune_sublayer='mlp', show_tqdm=False, per_batch=False, cat=True, temperature=1.0):
    data_iter = tqdm(data) if show_tqdm else data
    losses = []
    orig_sublayers = [getattr(x, prune_sublayer) for x in get_decoder_layers(model)]
    for x in data_iter:
        tokens = tokenizer(x['text'], truncation=True, return_tensors='pt', max_length=2048)
        input_ids = tokens.input_ids.cuda()
        for layer in get_decoder_layers(model):
            new_val = PassthroughMLP() if prune_sublayer == 'mlp' else PassthroughAttn()
            setattr(layer, prune_sublayer, new_val)

        student_logits = model(input_ids).logits.cpu().squeeze(dim=0)

        for idx, layer in enumerate(get_decoder_layers(model)):
            setattr(layer, prune_sublayer, orig_sublayers[idx])

        teacher_logits = model(input_ids).logits.cpu().squeeze(dim=0)
        losses.append(distillation_loss(student_logits, teacher_logits, per_batch=per_batch, temperature=temperature))        
    
    if per_batch:
        if cat:
            losses = torch.cat(losses, dim=0)
    else:
        losses = torch.tensor(losses)
    
    return losses

def get_bestk_losses(all_losses, k):
    indices = torch.topk(all_losses, k, largest=False).indices
    list_of_sets = [frozenset(row.tolist()) for row in indices]
    c = Counter(list_of_sets)
    return c

# %%
c = get_bestk_losses(all_losses, 1)

print(c.most_common(10))

c = get_bestk_losses(all_losses, 3)

print(c.most_common(10))

def get_sublayer_losses(data, prune_sublayer='mlp'):
    sublayer_losses = torch.zeros(all_losses)
    for idx in range(all_losses.size(0)):
        tokens = get_tokens(data, idx)
        for block_idx in range(all_losses.size(1)):
            key_in = f'block{block_idx}_{prune_sublayer}_in'
            key_out = f'block{block_idx}_{prune_sublayer}_out'
            ang_dist = torch.arccos(F.cosine_similarity(tokens[key_in], tokens[key_out]), dim=-1)
            ang_dist = (ang_dist / torch.pi).item() 
            sublayer_losses[idx, block_idx] = ang_dist

    return sublayer_losses

def get_bestk_losses(all_losses, k, prune_sublayer='mlp'):
    indices = torch.topk(all_losses, k, largest=False).indices
    list_of_sets = [frozenset(row.tolist()) for row in indices]
    c = Counter(list_of_sets)
    indices_to_remove = c.most_common(1)[0][0]
    distil_losses = get_distill_loss(model, data, indices_to_remove, prune_sublayer=prune_sublayer, per_batch=True)
    removed_was_best = torch.tensor([x == indices_to_remove for x in list_of_sets])
    return distil_losses[removed_was_best]


mlp_losses = get_sublayer_losses(model, data, 'mlp')
attn_losses = get_sublayer_losses(model, data, 'attn')

def plot_losses(losses):
    # remove outliers
    sns.displot(losses[losses < losses.quantile(0.95)])

with save_plot('mlp_1_loss.png'):
    plot_losses(get_bestk_losses(mlp_losses, 1, prune_sublayer='mlp'))

with save_plot('mlp_3_loss.png'):
    plot_losses(get_bestk_losses(mlp_losses, 3, prune_sublayer='mlp'))

with save_plot('attn_1_loss.png'):
    plot_losses(get_bestk_losses(attn_losses, 1, prune_sublayer='self_attn'))

with save_plot('attn_3_loss.png'):
    plot_losses(get_bestk_losses(attn_losses, 3, prune_sublayer='self_attn'))
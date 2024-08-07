from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
from typing import Literal, Optional

from models import get_model, get_decoder_layers, set_decoder_layers
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
    assistant_out: str 
    n_layer: int 
    n_head: int 
    n_embd: int
    gradient_accumulation_steps: int = 16
    dataset_name: str = 'wikitext'
    learning_rate: float = 6e-4
    train_size: int = 500
    test_size: int = 50
    log_size: int = 50

class MetricTracker:
    def __init__(self, objective: Literal['classification', 'regression'], criterion, n_blocks: int):
        self.criterion = criterion
        self.objective = objective 
        self.n_blocks = n_blocks
        self.reset()

    def reset(self):
        self.running_loss = 0
        self.running_tokens = 0
        self.running_samples = 0
        if self.objective == 'classification':
            self.TP = torch.zeros(self.n_blocks).cuda()
            self.FP = torch.zeros(self.n_blocks).cuda()
            self.FN = torch.zeros(self.n_blocks).cuda()
            self.correct_samples = 0
        elif self.objective == 'regression':
            self.k_correct = {k: 0 for k in range(1, 6)}
            self.k_total_se = {k: 0 for k in range(1, 6)}

    @torch.no_grad
    def update(self, y_true, y_pred):
        """
        Update the metrics with new batch of data.

        Parameters:
        y_true (torch.Tensor): Ground truth labels, shape (batch_size, n_blocks)
        y_pred (torch.Tensor): Predicted labels, shape (batch_size, n_blocks)
        """
        # Ensure the tensors are of the same shape
        assert y_true.shape == y_pred.shape, "Shape of y_true and y_pred must match"
        
        num_tokens = y_true.size(0)
        self.running_tokens += num_tokens
        self.running_loss += self.criterion(y_pred, y_true).item() * num_tokens
        # Normally this is just running_tokens * n_blocks
        self.running_samples += y_true.numel()

        if self.objective == 'classification':
            # Binarize predictions (assuming predictions are probabilities or logits)
            y_pred = (y_pred >= 0.5).float()

            # Update counts for True Positives (TP), False Positives (FP), False Negatives (FN)
            self.TP += (y_true * y_pred).sum(dim=0)
            self.FP += ((1 - y_true) * y_pred).sum(dim=0)
            self.FN += (y_true * (1 - y_pred)).sum(dim=0)

            # Update counts for accuracy
            self.correct_samples += (y_true == y_pred).float().sum().item() 
        elif self.objective == 'regression':
            preds_sorted =  torch.topk(y_pred, 6, largest=False, sorted=True)
            targets_sorted = torch.topk(y_true, 6, largest=False, sorted=True) 
            for k in range(1, 6):
                top_k_pred_indices = preds_sorted.indices[:, :k] 
                top_k_target_indices = targets_sorted.indices[:, :k] 

                # Calculate the number of correct predictions
                correct = torch.sum(
                    torch.eq(
                        top_k_pred_indices.sort(dim=1).values, 
                        top_k_target_indices.sort(dim=1).values
                    )
                    .all(dim=1)
                ).item()

                # Calculate the squared errors for the top-k target layers
                top_k_target_values = targets_sorted.values[:, :k] 
                top_k_pred_values = y_pred.gather(1, top_k_target_indices)

                total_se = torch.sum((top_k_pred_values - top_k_target_values) ** 2).item()

                self.k_correct[k] += correct
                self.k_total_se[k] += total_se

    def _compute_metrics_classification(self):
        """
        Compute precision, recall, F1-score, and accuracy.

        Returns:
        dict: Dictionary containing precision, recall, F1-score, and accuracy
        """

        precision = self.TP / (self.TP + self.FP + 1e-8)
        recall = self.TP / (self.TP + self.FN + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = self.correct_samples / self.running_samples

        return {
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1_score": f1_score.mean().item(),
            "accuracy": accuracy
        }

    def _compute_metrics_regression(self):
        """
        Compute k-accuracy and top-k MSE for k in range(1, 6).

        Returns:
        dict: Dictionary containing k-accuracy and top-k MSE for k in range(1, 6)
        """
        k_metrics = {}
        
        for k in range(1, 6):
            k_accuracy = self.k_correct[k] / self.running_tokens
            # We take the SSE of k samples per token
            top_k_mse = self.k_total_se[k] / (k * self.running_tokens)

            k_metrics[f'{k}_accuracy'] = k_accuracy
            k_metrics[f'top_{k}_mse'] = top_k_mse

        return k_metrics

    def compute_metrics(self):
        metrics = {'loss': self.running_loss / self.running_tokens}
        if self.objective == 'classification':
            metrics.update(self._compute_metrics_classification())
        elif self.objective == 'regression':
            metrics.update(self._compute_metrics_regression())
        else: 
            assert False, 'Unkown objective'
        
        return metrics

config, objective_config = HfArgumentParser((AssistantConfig, TrainingObjective)).parse_args_into_dataclasses()
wandb.init(project='UCL thesis', name=config.run_name)
teacher_model, teacher_tokenizer = get_model(config.teacher_model)
# Needed to avoid warnings from qwen2
teacher_model.generation_config.pad_token_id = teacher_tokenizer.eos_token_id
n_blocks = len(get_decoder_layers(teacher_model))
collect_modules = {'token_embedding'}
if not objective_config.use_distil_target:
    collect_modules.add('block')
collect_output_hooks(teacher_model, save_uncached=True, collect_modules=collect_modules)
teacher_model = teacher_model.cuda()
teacher_model.eval()

def _generate_synthetic_data_angles(batch, device):
    # Make sure we are operating with a clean dataset
    TensorStorage.reset()
    # No padding since we have a batch size of 1, but truncate inputs that are too long 
    tokens = teacher_tokenizer(batch['text'], return_tensors='pt', max_length=2048, truncation=True, padding=False)
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
        outs = None 
        if objective_config.objective == 'regression':
            outs = angular_distances
        else:
            angular_thresh = objective_config.angular_dist_threshold
            if angular_thresh is None:
                cos_sim_thresh = torch.tensor(objective_config.cos_sim_threshold)
                angular_thresh = torch.acos(cos_sim_thresh) / torch.pi

            outs = angular_distances <= angular_thresh

        synthetic_outputs.append(outs.unsqueeze(0))

    # Unsqueeze to simulate batch_size = 1 
    X = torch.cat(synthetic_inputs, dim=0).unsqueeze(0)
    y = torch.cat(synthetic_outputs, dim=0)
    return X.to(device), y.to(device)

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

@torch.no_grad()
def _generate_synthetic_data_distil(batch, device): 
    # TODO: This can be faster 
    def get_logits():
        return teacher_model(input_ids, use_cache=False, past_key_values=None).logits.squeeze(dim=0)
    
    TensorStorage.reset()
    tokens = teacher_tokenizer(batch['text'], truncation=True, return_tensors='pt', max_length=2048)
    input_ids = tokens.input_ids.cuda()    
    layers = get_decoder_layers(teacher_model)
    orig_logits = get_logits() 
    seq_len = orig_logits.size(0)
    X = TensorStorage._cur_sample['token_embedding_first'][0].squeeze(dim=0)
    losses = torch.zeros((n_blocks, seq_len)).cuda()
    for layer_idx in range(n_blocks):
        new_layers = layers[:layer_idx] + layers[layer_idx + 1:]
        set_decoder_layers(teacher_model, new_layers)
        student_logits = get_logits() 
        losses[layer_idx] = distillation_loss(student_logits, orig_logits, per_batch=True)
        
    set_decoder_layers(teacher_model, layers)
    return X.to(device), losses.transpose(0, 1).to(device)

@torch.no_grad()
def generate_synthetic_data(batch, device): 
    if objective_config.use_distil_target:
        return _generate_synthetic_data_distil(batch, device)
    else:
        return _generate_synthetic_data_angles(batch, device)

def approximate_class_weight(loader, sample_size):
    assert objective_config.objective == 'classification'
    weights = torch.zeros(n_blocks).cuda()
    N = 0
    for batch in loader:
        _, y = generate_synthetic_data(batch, 'cuda')
        if y is None:
            continue

        batch_weights = y.mean(dim=0)
        # Simple averaging
        weights -= 1. / (N + 1) * (weights - batch_weights)
        N += 1
        if N >= sample_size:
            break

    # weights is the mean = proportion of ones
    # We need pos_weight * proportion of ones = proportion of zeros 
    # pos_weight * weights = (1 - weights) 
    # pos_weight = (1 - weights) / weights
    pos_weights = (1 - weights) / weights
    return pos_weights

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
    else:
        assert False 

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = get_loaders(config.dataset_name, config.train_size, config.test_size)

# train_loader = DataLoader(wikitext['train'].select(range(config.train_size)), batch_size=1, shuffle=True)
# test_loader  = DataLoader(wikitext['test'].select(range(config.test_size)) , batch_size=1, shuffle=False)

def get_criterion():
    if objective_config.objective == 'regression':
        return torch.nn.MSELoss()
    else:
        pos_weights = approximate_class_weight(train_loader, objective_config.weight_estimation_steps)
        print(f'Found approximate pos weights: {pos_weights}')
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        # criterions = [torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction='sum') for pw in pos_weights]
        # def new_bce_loss_with_logits(preds, targets):
        #     # Use sum here to avoid gradient gets propagated
        #     total_loss = sum( 
        #         criterions[i](preds[:, i], targets[:, i]) 
        #         for i in range(len(criterions))
        #     )

        #     return total_loss / targets.numel()

        # return new_bce_loss_with_logits

dropout = 0.0
weight_decay = 1e-1 
lr = config.learning_rate 
print(f'{lr=}')
cfg = GPTConfig(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd, bias=False, dropout=dropout)
print('Assistant config:')
print(cfg)
model = GPT2ForLayerPruning(cfg, teacher_model.config.hidden_size, teacher_model.config.num_hidden_layers).cuda()
optim = model.configure_optimizers(weight_decay, lr, 'cuda')
criterion = get_criterion()

from tqdm import tqdm

@torch.no_grad()
def compute_val_metrics(model, test_loader, criterion):
    model.eval()
    metric_tracker = MetricTracker(objective_config.objective, criterion, n_blocks) 
    for batch in test_loader:
        X, y = generate_synthetic_data(batch, 'cuda')
        if X is None:
            continue

        preds = model(X, training=True).squeeze()
        metric_tracker.update(y, preds)
    
    return metric_tracker.compute_metrics() 

model.train()
total_num_tokens = 0
optim.zero_grad()

train_tracker = MetricTracker(objective_config.objective, criterion, n_blocks)

for idx, batch in tqdm(enumerate(train_loader), total=config.train_size):
    if (idx + 1) % config.log_size == 0:
        train_metrics = {
            f'train_{metric}': value 
            for metric, value in train_tracker.compute_metrics().items() 
        }
        val_metrics = {
            f'val_{metric}': value 
            for metric, value in compute_val_metrics(model, test_loader, criterion).items()
        }
        wandb_message = train_metrics | val_metrics
        # NOTE: Maybe no need to require python 3.9 for just that line...  
        wandb.log(wandb_message, step=idx)
        model.train()
        train_tracker.reset()

    X, y = generate_synthetic_data(batch, 'cuda')
    if X is None:
        continue
 
    preds = model(X, training=True).squeeze()
    train_tracker.update(y, preds)
    num_tokens = y.size(0)
    loss = criterion(preds, y) * num_tokens
    total_num_tokens += num_tokens

    loss.backward()
    wandb.log({
        # Normalize loss by batch size for logging
        'batch_loss': (loss / num_tokens).item(),
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
        'output_size': n_blocks 
    }, cfg_file)
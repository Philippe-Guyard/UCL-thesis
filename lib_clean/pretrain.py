# from modeling_llama import LlamaForCausalLM
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from transformers import AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser, AutoConfig
from transformers.activations import ACT2FN
from transformers.models.llama import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from datasets import load_dataset, Dataset
import evaluate

from models import get_model

def get_trainable_model(model_name: str) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:  
    assert 'opt' not in model_name, 'OPT models already have relu'
    model, tokenizer = get_model(model_name, with_relu=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    assert tokenizer.pad_token_id is not None
    return model, tokenizer

def to_tokenized_dataset(dataset: Dataset, num_samples: int, shuffle_seed=42):
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=1024
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    return (
        dataset.shuffle(seed=shuffle_seed)
        .select(range(num_samples))
        .map(tokenize_function, remove_columns=["text"])
    )

def get_dataset(train_size: int, eval_size: int, seed: int):
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    train_dataset = to_tokenized_dataset(wikitext["train"], train_size, seed)
    test_dataset = to_tokenized_dataset(wikitext["test"], eval_size, seed)
    return train_dataset, test_dataset

perplexity_metric = evaluate.load("perplexity")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Shift the logits and labels to align predictions with targets
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    # Compute the perplexity
    perplexity = perplexity_metric.compute(
        predictions=shift_logits, references=shift_labels
    )
    return {"perplexity": perplexity}

@dataclass
class PretrainingConfig:
    model_name: str
    run_name: str
    train_batch_size: int
    eval_batch_size: int
    warmup_steps: int
    seed: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    train_size: int
    eval_size: int
    eval_accumulation_steps: int
    mixed_precision: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)
    eval_strategy: str = field(default='no')
    optim: str = field(default='adamw_torch')

config: PretrainingConfig = HfArgumentParser(PretrainingConfig).parse_args_into_dataclasses()[0]
model, tokenizer = get_trainable_model(config.model_name)

bf16, fp16 = False, False 
if config.mixed_precision:
    fp16 = not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_bf16_supported()

root_folder = Path(f'runs/{config.run_name}')
training_args = TrainingArguments(
    # Misc
    output_dir=root_folder.joinpath("checkpoints"),
    save_steps=config.save_steps,
    metric_for_best_model="perplexity",
    greater_is_better=False,
	bf16=bf16,
	fp16=fp16,
	gradient_checkpointing=config.gradient_checkpointing,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    optim=config.optim,
    # Training hypers
    num_train_epochs=1,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
	gradient_accumulation_steps=config.gradient_accumulation_steps,
    warmup_steps=config.warmup_steps,
    max_grad_norm=1,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    lr_scheduler_type="cosine",
    learning_rate=4e-4,
    eval_strategy=config.eval_strategy,
    eval_steps=config.eval_steps,
    eval_accumulation_steps=config.eval_accumulation_steps,
    # Logging
    logging_dir=root_folder.joinpath("logs"),
    logging_steps=10,
    report_to="wandb",
    run_name=config.run_name,
    # Seeds
    seed=config.seed,
    data_seed=config.seed,
)
print(training_args)

train_dataset, test_dataset = get_dataset(config.train_size, config.eval_size, config.seed)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

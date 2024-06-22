# from modeling_llama import LlamaForCausalLM
from pathlib import Path
from typing import Any, Callable, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from datasets import load_dataset, Dataset
import evaluate

# from trl import SFTTrainer
from accelerate import PartialState


@dataclass
class PretrainingConfig:
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


argparse = HfArgumentParser(PretrainingConfig)
config: PretrainingConfig = argparse.parse_args_into_dataclasses()[0]

root_folder = Path(config.run_name)


def replace_activation_function(
    model: LlamaForCausalLM, new_activation: Callable[[], nn.Module]
):
    config: LlamaConfig = model.config
    cur_actfn = type(ACT2FN[config.hidden_act])
    for module_name, module in model.named_modules():
        # need to iterate over child since setattr does not work with . notation
        for child_name, child in module.named_children():
            if isinstance(child, cur_actfn):
                setattr(module, child_name, new_activation())

    return model


device_string = PartialState().process_index
model_name = "TinyLlama/TinyLlama_v1.1"
model = LlamaForCausalLM.from_pretrained(model_name, device_map={"": device_string})
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

assert tokenizer.pad_token_id is not None


def to_tokenized_dataset(dataset: Dataset, num_samples: int, shuffle_seed=42):
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"], truncation=True, padding="longest", max_length=1024
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    return (
        dataset.shuffle(seed=shuffle_seed)
        .select(range(num_samples))
        .map(tokenize_function, batched=True, remove_columns=["text"])
    )


wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
train_dataset = to_tokenized_dataset(wikitext["train"], config.train_size, config.seed)
test_dataset = to_tokenized_dataset(wikitext["test"], config.eval_size, config.seed)
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


training_args = TrainingArguments(
    # Misc
    output_dir=root_folder.joinpath("checkpoints"),
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    save_steps=config.save_steps,
    metric_for_best_model="perplexity",
    greater_is_better=False,
    # Training hypers
    num_train_epochs=1,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    warmup_steps=config.warmup_steps,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    lr_scheduler_type="cosine",
    learning_rate=4e-4,
    eval_strategy="steps",
    eval_steps=500,
    # Logging
    logging_dir=root_folder.joinpath("logs"),
    logging_steps=10,
    report_to="wandb",
    run_name=config.run_name,
    # Seeds
    seed=config.seed,
    data_seed=config.seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

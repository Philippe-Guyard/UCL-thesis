from hooks import save_layer_io_hooks, time_execution_hooks, SampleIds
from simple_timer import Timer

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import OPTConfig, AutoTokenizer, OPTForCausalLM
from datasets import load_dataset, Dataset

assert torch.cuda.is_available()
device = "cuda"


def get_data(n_examples: int):
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    return wikitext["train"].select(range(n_examples))


def get_model():
    model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    return model, tokenizer


@torch.no_grad
def run_once(data, model, tokenizer, tokens_per_sample=50):
    model.eval()
    for idx, x in tqdm(enumerate(data), total=len(data)):
        question: str = x["text"]
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
        n_inputs = input_ids.size(1)

        SampleIds.cur_sample_id = idx
        # First token does not use cache, so we ignore it, hence the + 1
        max_length = n_inputs + tokens_per_sample + 1
        _ = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )


def collect_output():
    data = get_data(100)
    model, tokenizer = get_model()
    save_layer_io_hooks(model.model.decoder.layers, Path("./data"))

    run_once(data, model, tokenizer)


def time_execution():
    data = get_data(50)
    model, tokenizer = get_model()
    model: OPTForCausalLM

    for idx, layer in enumerate(model.model.decoder.layers):
        time_execution_hooks(layer, f"Decoder Layer {idx}")
        time_execution_hooks(layer.self_attn, "Self attention")
        time_execution_hooks(layer.fc1, "fc1")
        time_execution_hooks(layer.fc2, "fc2")
        time_execution_hooks(layer.activation_fn, "MLP activation")

    run_once(data, model, tokenizer)
    Timer.print()


# time_execution()
collect_output()

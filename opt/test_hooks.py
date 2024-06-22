import torch
from tqdm import tqdm 

from my_timer import Timer
from sample_ids import SampleIds
from outputs_extractor import add_hooks 
from pathlib import Path

from transformers import OPTConfig, AutoTokenizer, OPTForCausalLM
from datasets import load_dataset, Dataset

assert torch.cuda.is_available()
device = 'cuda'

model = OPTForCausalLM.from_pretrained('facebook/opt-125m') 
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
add_hooks(model.model.decoder.layers, Path('./data_longer'))
total_params = sum(p.numel() for p in model.parameters())
print(total_params // (10 ** 6))

n_examples = 25
wikitext = load_dataset('Salesforce/wikitext', 'wikitext-103-v1')
data = wikitext['train'].select(range(n_examples))
model.eval()

with torch.no_grad():
    for idx, x in tqdm(enumerate(data), total=n_examples):
        question: str = x['text'] 
        input_ids = tokenizer(question, return_tensors='pt').input_ids.to(device) 
        n_inputs = input_ids.size(1)

        SampleIds.cur_sample_id = idx
        num_tokens_to_generate = 1000 
        output = model.generate(input_ids, max_length=n_inputs + num_tokens_to_generate + 1, 
                            do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id) 
Timer.print()
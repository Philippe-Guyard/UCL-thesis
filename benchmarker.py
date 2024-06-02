'''
Compute average t/s for input and output tokens using random question from MMLU.  
'''
# TODO: How to measure model performance? MMLU good enough?

import time 

from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

assert torch.cuda.is_available()
device = 'cuda'

n_examples = 100
n_burnin = 1 
mmlu: Dataset = load_dataset('cais/mmlu', name='all')
data = mmlu['test'].select(range(n_examples))

# model = AutoModelForCausalLM.from_pretrained('/home/philippeg/UCL/thesis/papers/sparsegpt/sparse_opt/opt-125m')
# model = AutoModelForCausalLM.from_pretrained('/home/philippeg/UCL/thesis/papers/wanda/out/opt-125m')
# model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
model = AutoModelForCausalLM.from_pretrained('/home/philippeg/UCL/thesis/papers/PruneMe/out/opt-125m')
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

print(sum(p.numel() for p in model.parameters()) // (10 ** 6))

model = model.to(device)
model.eval()

input_speeds = []
output_speeds = []

with torch.no_grad():
    for x in tqdm(data):
        question: str = x['question']
        input_ids = tokenizer(question, return_tensors='pt').input_ids.to(device)         
        n_inputs = input_ids.size(1)

        num_tokens_to_generate = 1
        start = time.perf_counter_ns()
        output = model.generate(input_ids, max_length=n_inputs + num_tokens_to_generate, 
                            do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
        time_ns = time.perf_counter_ns() - start
        input_speed = n_inputs / (time_ns / (10 ** 9))
        input_speeds.append(input_speed)
        # print(f'Ingested {n_inputs} tokens with speed {} t/s')

        num_tokens_to_generate = 2 * n_inputs
        start = time.perf_counter_ns()
        output = model.generate(input_ids, max_length=n_inputs + num_tokens_to_generate, 
                            do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
        time_ns = time.perf_counter_ns() - start 
        output_speed = num_tokens_to_generate / (time_ns / (10 ** 9))
        output_speeds.append(output_speed)
        # print(f'Generated {num_tokens_to_generate} tokens with speed {} t/s')

average_input_speed = np.mean(input_speeds[n_burnin:])
average_output_speed = np.mean(output_speeds[n_burnin:])

print(f'{average_input_speed=} t/s, {average_output_speed=} t/s')
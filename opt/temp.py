import torch
from tqdm import tqdm 

from my_timer import Timer
from sample_ids import SampleIds

from transformers import OPTConfig, AutoTokenizer
from datasets import load_dataset, Dataset
from modeling_opt import OPTForCausalLM

assert torch.cuda.is_available()
device = 'cuda'

config = OPTConfig() # default value is for OPT-125M
print(config.activation_function)
model = OPTForCausalLM(config).to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

total_params = sum(p.numel() for p in model.parameters())
print(total_params // (10 ** 6))

n_examples = 500
n_burnin = 1 
mmlu: Dataset = load_dataset('cais/mmlu', name='all')
data = mmlu['test'].select(range(n_examples))
model.eval()

with torch.no_grad():
    for idx, x in tqdm(enumerate(data), total=n_examples):
        question: str = x['question'] 
        input_ids = tokenizer(question, return_tensors='pt').input_ids.to(device) 
        n_inputs = input_ids.size(1)

        SampleIds.cur_sample_id = idx
        num_tokens_to_generate = 50 
        output = model.generate(input_ids, max_length=n_inputs + num_tokens_to_generate, 
                            do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id) 
Timer.print()
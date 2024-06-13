import torch
from tqdm import tqdm 

from my_timer import Timer
from sample_ids import SampleIds

from transformers import OPTConfig, AutoTokenizer
from datasets import load_dataset, Dataset
from modeling_gemma import GemmaForCausalLM 

assert torch.cuda.is_available()
device = 'cuda'

model_name = 'google/gemma-2b'
model = GemmaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) 
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
print(total_params // (10 ** 6))

n_examples = 100
# mmlu: Dataset = load_dataset('cais/mmlu', name='all')
# data = mmlu['test'].select(range(n_examples))
wikitext = load_dataset('Salesforce/wikitext', 'wikitext-103-v1')
data = wikitext['train'].select(range(n_examples))
model.eval()

with torch.no_grad():
    for idx, x in tqdm(enumerate(data), total=n_examples):
        # print('===================================')
        question: str = x['text'] 
        input_ids = tokenizer(question, return_tensors='pt').input_ids.to(device) 
        n_inputs = input_ids.size(1)

        SampleIds.cur_sample_id = idx
        # print(idx, len(question), n_inputs)
        num_tokens_to_generate = 50 
        # +1 because we skip the first token which is prompt ingestion 
        output = model.generate(input_ids, max_length=n_inputs + num_tokens_to_generate + 1, 
                            do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id, use_cache=True) 

        # print(question)
        # print() 
        # print(tokenizer.batch_decode(output)[0][len(question):])
Timer.print()
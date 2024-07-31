from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from typing import List, Literal, Optional
import pandas as pd

from scripts import benchmark
from models import AssistanceConfig, get_basemodel_name, is_local_model_name, load_assistant

from lm_eval import evaluator, tasks, models
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoConfig

class AutoAssistedModel(AutoModelForCausalLM):
    def from_pretrained(*args, **kwargs):
        model_name = args[0]
        assistant_name = None 
        # We use @ as a special key to separate model_path and assistant_ath 
        if '@' in model_name:
            model_name, assistant_name = model_name.split('@')

        # Replace the first argument in case we had an assistant 
        model = AutoModelForCausalLM.from_pretrained(model_name, *args[1:], **kwargs) 
        if assistant_name is not None: 
            load_assistant(Path(assistant_name), model, get_basemodel_name(model_name))

        return model

class AssistedHFLM(models.huggingface.HFLM):
    def __init__(self, *args, **kwargs):
        self.assistant_name = kwargs.pop('assistant_name', None)
        self.pretrained=kwargs['pretrained']
        super().__init__(*args, **kwargs)
        
    def _create_model(self, *args, **kwargs):
        result = super()._create_model(*args, **kwargs)
        if self.assistant_name is not None:
            # No cache for assistant because lm_eval does not use model.generate, so it becomes unclear when to reset it 
            load_assistant(Path(self.assistant_name), self._model, model_basename=get_basemodel_name(self.pretrained), assistant_use_cache=False)

        return result

def get_metric_values(results, task):
    task_results = results['results'][task]
    metric_key = None 
    for key in task_results:
        if ',' not in key:
            continue 

        metric_name, filter = key.split(',')
        if metric_name.endswith('stderr'):
            metric_name = metric_name.split('_')[0]
            metric_key = f'{metric_name},{filter}'
            break

    assert metric_key is not None, 'Could not find metric key'
    metric_name, filter = metric_key.split(',')
    print(f'Using metric {metric_name} with filter {filter} for task {task}')
    mean = task_results[metric_key] 
    stderr = task_results[f'{metric_name}_stderr,{filter}']

    return mean, stderr

def format_number(mean, stderr):
    return f"{mean:.2f}±{stderr:.2f}"

def format_result(results, task):
    acc, stderr = get_metric_values(results, task) 
    return format_number(acc, stderr)

def evaluate_checkpoint(model_path: str, tasks: List[str]):
    if len(tasks) == 0:
        # Only benchmarking case 
        return None 

    # Why does batch_size='auto' not work??
    assistant_name = None
    if '@' in model_path:
        model_path, assistant_name = model_path.split('@')

    lm_obj = AssistedHFLM(
        pretrained=model_path,
        batch_size=1,
        assistant_name=assistant_name
    )
    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        num_fewshot=0,
        batch_size=1,
        device="cuda",
    )
    return results

def eval_all_checkpoints(run_name: str, tasks: List[str]):
    checkpoints_folder = Path(f'./runs/{run_name}/checkpoints')
    all_checkpoints = [f for f in checkpoints_folder.iterdir() if f.is_dir()]
    all_results = {task: [] for task in tasks}
    all_results["model"] = []
    all_results['model_path'] = []

    for checkpoint in all_checkpoints: 
        results = evaluate_checkpoint(checkpoint.as_posix(), tasks)
        
        model_str = f'{get_basemodel_name(checkpoint.as_posix())}-{checkpoint.name}'
        all_results["model"].append(model_str)
        all_results['model_path'] = checkpoint.as_posix()
        for task in tasks:
            all_results[task].append(format_result(results, task))

    df = pd.DataFrame(all_results)
    return df 

def eval_model(model_name: str, tasks: List[str]):
    all_results = {task: [] for task in tasks}
    basename = get_basemodel_name(model_name.split('@', 1)[0])
    all_results['model'] = [basename]
    all_results['model_path'] = [model_name]
    results = evaluate_checkpoint(model_name, tasks)
    for task in tasks:
        all_results[task].append(format_result(results, task))
    
    df = pd.DataFrame(all_results)
    return df 

@dataclass
class EvalConfig:
    tasks: Optional[str] = field(default=None) # Comma-separated list
    csv_out: str = field(default='eval_results.csv')
    model_name: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    benchmark: Optional[bool] = field(default=False)
    append: Optional[bool] = field(default=False)
    metadata: Optional[str] = field(default=None)

if __name__ == '__main__':
    config, assistant_config = HfArgumentParser((EvalConfig, AssistanceConfig)).parse_args_into_dataclasses()
    assert (config.model_name is None) ^ (config.run_name is None), 'Exactly one of --model_name or --run_name should be specified'
    csv_out = Path(config.csv_out)
    if not config.append:
        assert not csv_out.exists(), f'{csv_out} already exists'

    # Only benchmarking case, maybe should be done somewhere else 
    tasks = config.tasks.split(',') if config.tasks is not None else []

    if config.model_name:
        df = eval_model(config.model_name, tasks)
    else:
        df = eval_all_checkpoints(config.run_name, tasks)

    df = df.set_index('model_path').sort_index()
    
    if config.metadata:
        df['metadata'] = config.metadata
    elif config.model_name is not None:
        # if model is local then metadata is path
        df['metadata'] = config.model_name 
    else:
        df['metadata'] = 'Pretrained model'

    if config.benchmark:
        input_speeds, output_speeds = [], []
        for model_path in df.index:
            input_speed, input_std, output_speed, output_std = benchmark(model_path, assistant_name=assistant_name)
            input_speeds.append(format_number(input_speed, input_std))
            output_speeds.append(format_number(output_speed, output_std))
        
        df['input_speed'] = input_speeds
        df['output_speed'] = output_speeds

    if config.append:
        df_old = pd.read_csv(config.csv_out, index_col='model_path')
        df = pd.concat([df_old, df]) 

    df.to_csv(csv_out)

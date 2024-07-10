from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd

from scripts import benchmark
from models import get_basemodel_name

from lm_eval import evaluator, tasks
from transformers import HfArgumentParser

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
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=tasks,
        num_fewshot=0,
        batch_size=1,
        device="cuda",
    )
    return results

def eval_all_checkpoints(run_name: str, tasks: List[str], csv_out: Path):
    checkpoints_folder = Path(f'./runs/{run_name}/checkpoints')
    all_checkpoints = [f for f in checkpoints_folder.iterdir() if f.is_dir()]
    all_results = {task: [] for task in tasks}
    all_results["model"] = []

    for checkpoint in all_checkpoints: 
        results = evaluate_checkpoint(checkpoint.as_posix(), tasks)
        
        model_str = f'{get_basemodel_name(checkpoint.as_posix())}-{checkpoint.name}'
        all_results["model"].append(model_str)
        for task in tasks:
            all_results[task].append(format_result(results, task))

    df = pd.DataFrame(all_results)
    df = df.set_index('model').sort_index()
    return df 

def eval_model(model_name: str, tasks: List[str], csv_out: Path):
    all_results = {task: [] for task in tasks}
    all_results['model'] = [get_basemodel_name(model_name)]
    results = evaluate_checkpoint(model_name, tasks)
    for task in tasks:
        all_results[task].append(format_result(results, task))
    
    df = pd.DataFrame(all_results)
    df = df.set_index('model')
    return df 

@dataclass
class EvalConfig:
    tasks: str # Comma-separated list
    csv_out: str = field(default='eval_results.csv')
    model_name: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    benchmark: Optional[bool] = field(default=False)
    append: Optional[bool] = field(default=False)

if __name__ == '__main__':
    config: EvalConfig = HfArgumentParser(EvalConfig).parse_args_into_dataclasses()[0]
    assert (config.model_name is None) ^ (config.run_name is None), 'Exactly one of --model_name or --run_name should be specified'
    csv_out = Path(config.csv_out)
    tasks = config.tasks.split(',')
    assert not csv_out.exists(), f'{csv_out} already exists'

    if config.model_name:
        df = eval_model(config.model_name, tasks, csv_out)
    else:
        df = eval_all_checkpoints(config.run_name, tasks, csv_out)

    if config.benchmark:
        input_speeds, output_speeds = [], []
        for model_name in df.index:
           input_speed, input_std, output_speed, output_std = benchmark(model_name)
           input_speeds.append(format_number(input_speed, input_std))
           output_speeds.append(format_number(output_speed, output_std))
        
        df['input_speed'] = input_speeds
        df['output_speed'] = output_speeds

    if config.append:
        df_old = pd.read_csv(config.csv_out)
        df = df.merge(df_old, how='outer')

    df.to_csv(csv_out)

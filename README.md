Most of the relevant code is in the `lib_clean` folder.
Most experiments with contextual sparsity can be replicated by running `lib_clean/layer_importance.py` and `lib_clean/prune_count.py`. Note that one may need to uncomment some lines to see the required plots.

To train assistants, run `lib_clean/train_assistant.py`. Note that the file requires many hyperparameters, here is the script we used for training the OPT-1.3B assistant.
```
MODEL_NAME="facebook/opt-1.3b"
python3 train_assistant.py \
        --run_name "opt_assistant_histograms" \
        --teacher_model $MODEL_NAME --batch_size 8 \
        --gradient_accumulation_steps 2 \
        --n_layer 4 --n_head 8 --n_embd 128 \
        --train_size 399000 --test_size 500 \
        --eval_steps 1000 --save_steps 1000 \
        --dataset_name "tinytextbooks" \
```

Evaluation and benchmarking is done in `lib_clean/eval_model_fewshot.py`. Here is an example of how to eval an assisted model:
```
MODEL_PATH="Qwen/Qwen2-1.5B"
TASKS="mmlu@5,hellaswag@10,winogrande@5,openbookqa@0,arc_easy@25,arc_challenge@25,boolq@0,piqa@0"
CSV_OUT="assistants_results_clean.csv"
N_LAYERS=6

CHECKPOINT=19000
RUN_NAME=qwen_assistant_histograms
ASSISTANT_PATH="$PROJECT_HOME/project_main/lib_clean/runs/$RUN_NAME/checkpoints/checkpoint-$CHECKPOINT"
python3 eval_model_fewshot.py --model_name $MODEL_PATH --tasks $TASKS --csv_out $CSV_OUT --metadata "n_layers=$N_LAYERS" --benchmark --assistant_name $ASSISTANT_PATH --prune_nlayers $N_LAYERS
```

Alternatively, for evaluating non-assisted models, one can simply remove the `--assistant_name` argument.

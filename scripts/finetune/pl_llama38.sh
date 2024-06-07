#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --max_seq_length 4096 \
    --output_model_name rag-tge_pl_Llama-3-8B_LoRA \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps_denom 48 \
    --eval_steps_denom 48 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --epochs 2 \
    --data_collator completion \
    --completion_collator_needle "\nPytanie:" \
    --language pl \
    --dataset_name tdolega/rag-tge_finetuning-dataset_pl \
    --system_prompt_id 6 \
    --dataset_num_proc 1 \
    "$@"

#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --output_model_name rag-tge_Mistral_LoRA \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_steps_denom 32 \
    --eval_steps_denom 32 \
    --lora_rank 128 \
    --lora_alpha 64 \
    --epochs 2 \
    --data_collator completion \
    --completion_collator_needle "\nQuestion:" \
    --max_seq_length 4096 \
    --dataset_num_proc 1 \
    "$@"

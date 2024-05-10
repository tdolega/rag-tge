#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --output_model_name rag-tge_Mistral_LoRA \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 8192 \
    --save_steps_denom 12 \
    --eval_steps_denom 48 \
    --learning_rate 6e-4 \
    "$@"

    # --max_seq_length 32768 \
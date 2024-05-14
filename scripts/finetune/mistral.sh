#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --output_model_name rag-tge_Mistral_LoRA \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_steps_denom 40 \
    --eval_steps_denom 80 \
    --lora_rank 128 \
    --lora_alpha 128 \
    --epochs 3 \
    --data_collator completion \
    --max_seq_length 4096 \
    "$@"


    # --lr_scheduler_type constant_with_warmup \
    # --epochs 5 \
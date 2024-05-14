#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --output_model_name rag-tge_Llama-3-8B_LoRA \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps_denom 40 \
    --eval_steps_denom 80 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lr_scheduler_type constant_with_warmup \
    --epochs 5 \
    --data_collator completion \
    "$@"

# --lora_rank 64 \
# --lora_alpha 128 \

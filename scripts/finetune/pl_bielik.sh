#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name speakleash/Bielik-7B-Instruct-v0.1 \
    --max_seq_length 4096 \
    --output_model_name rag-tge_pl_Bielik_LoRA \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_steps_denom 48 \
    --eval_steps_denom 48 \
    --lora_rank 128 \
    --lora_alpha 64 \
    --epochs 2 \
    --data_collator completion \
    --language pl \
    --dataset_name tdolega/rag-tge_finetuning-dataset_pl \
    --system_prompt_id 5 \
    "$@"

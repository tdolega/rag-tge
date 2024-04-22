#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name meta-llama/Meta-Llama-3-8B \
    --output_model_name rag-tge_Llama-3-8B_LoRA \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 8192 \
    --use_unsloth True \
    "$@"

#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --output_model_name rag-tge_Mistral_LoRA \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    "$@"

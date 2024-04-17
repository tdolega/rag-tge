#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output_model_name rag-tge_TinyLlama_LoRA \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    "$@"

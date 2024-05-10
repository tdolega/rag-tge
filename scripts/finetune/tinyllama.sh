#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output_model_name rag-tge_TinyLlama_LoRA \
    --epochs 2 \
    --batch_size 16 \
    --max_seq_length 2048 \
    "$@"

#!/bin/bash

set -e

python src/finetune.py \
    --input_model_name tdolega/rag-tge_pl_Llama-3-8B \
    --max_seq_length 8192 \
    --output_model_name rag-tge_pwr_Llama-3-8B_LoRA \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_steps_denom 16 \
    --eval_steps_denom 16 \
    --lora_rank 128 \
    --lora_alpha 64 \
    --epochs 2 \
    --data_collator completion \
    --completion_collator_needle "\nPytanie:" \
    --language pl \
    --dataset_name tdolega/rag-tge_trl-pwr-dataset \
    --system_prompt_id 6 \
    --dataset_num_proc 1 \
    "$@"

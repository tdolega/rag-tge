#!/bin/bash

python ../src/serve_evaluator.py \
    --llm clarin_llama \
    --nli llm_clarin_llama \
    "$@"

# python ../src/serve_evaluator.py \
#     --llm hfia_meta-llama/Meta-Llama-3-8B-Instruct \
#     --nli llm_hfia_meta-llama/Meta-Llama-3-8B-Instruct \
#     "$@"

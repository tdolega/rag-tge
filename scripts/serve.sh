#!/bin/bash

python ../src/serve_evaluator.py \
    --llm groq_llama3-8b-8192 \
    --nli llm_groq_llama3-8b-8192 \
    "$@"

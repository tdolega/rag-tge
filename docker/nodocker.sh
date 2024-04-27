#!/bin/bash

source .env
python3 -m llama_cpp.server --port ${LLAMA_PORT} --model ./${MODEL_FILE} --n_ctx ${CTX_SIZE} --n_gpu_layers 99 --embedding TRUE

#!/bin/bash

set -e

input_path=$1

if [ -z "$input_path" ]; then
    echo "Usage: $0 <input-path>"
    exit 1
fi

if [ -f ./$input_path/ggml-model-f32.gguf ]; then
    echo "ggml-model-f32.gguf already exists in $input_path"
    exit 0
fi


# convert the model to gguf
docker run \
    --rm \
    -v ./$input_path:/input \
    ghcr.io/ggerganov/llama.cpp:full-cuda \
    -c /input

# quantize the model
docker run \
    --rm \
    -v ./$input_path:/input \
    ghcr.io/ggerganov/llama.cpp:full-cuda \
    -q /input/ggml-model-f32.gguf Q8_0

#!/bin/bash

set -e

input_path=$1
potential_names=(
    "ggml-model-f32.gguf"
    "input-8B-F32.gguf"
)

if [ -z "$input_path" ]; then
    echo "Usage: $0 <input-path>"
    exit 1
fi

for name in "${potential_names[@]}"; do
    if [ -f $input_path/$name ]; then
        echo "$name already exists in $input_path"
        exit 0
    fi
done

# convert the model to gguf
docker run \
    --rm \
    -v $input_path:/input \
    ghcr.io/ggerganov/llama.cpp:full-cuda \
    -c /input --vocab-type "spm,hfft,bpe"

# quantize the model
for name in "${potential_names[@]}"; do
    if [ -f $input_path/$name ]; then
        echo "quantizing $name"
        docker run \
            --rm \
            -v $input_path:/input \
            ghcr.io/ggerganov/llama.cpp:full-cuda \
            -q /input/$name Q8_0
        break
    fi
done

echo "done"

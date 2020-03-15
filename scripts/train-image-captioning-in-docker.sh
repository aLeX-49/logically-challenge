#!/bin/bash

set -e

docker build --rm -f Dockerfile . -t logically_challenge:1.0

docker run --rm -it \
    --gpus all \
    -v "$(pwd):/workspace/" \
    -w /workspace/ \
    logically_challenge:1.0 \
    $@

#!/bin/bash

# Define environment variables
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=false

# Define paths
INPUT_CSV=""
SAVE_PATH=""

# Run script on the full CSV file
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
    make_paired_data.py \
    --config "./make_data_config.py" \
    --data-path $INPUT_CSV \
    --save_path $SAVE_PATH
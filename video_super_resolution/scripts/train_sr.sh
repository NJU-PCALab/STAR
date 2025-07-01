#!/bin/bash

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  ./video_super_resolution/scripts/train_sr.py \
  --pretrained_model_path '' \
  --train_batch_size 1 \
  --max_train_steps 15000 \
  --checkpointing_steps 500 \
  --learning_rate 5e-5 \
  --train_data_dir '' \
  --num_frames 32 \
  --output_dir ''

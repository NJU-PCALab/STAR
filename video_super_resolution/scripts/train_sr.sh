#!/bin/bash

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  ./video_super_resolution/scripts/train_sr.py \
  --pretrained_model_path '/mnt/bn/videodataset-uswest/VSR/pretrained_models/venhancer/venhancer_paper.pt' \
  --train_batch_size 1 \
  --max_train_steps 15000 \
  --checkpointing_steps 500 \
  --learning_rate 5e-5 \
  --train_data_dir '/mnt/bn/videodataset-uswest/VSR/dataset/SRTraining/training_group_0' \
  --num_frames 32 \
  --output_dir '/mnt/bn/videodataset-uswest/VSR/exp/venhancer/debug'

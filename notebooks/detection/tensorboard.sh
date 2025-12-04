#!/bin/bash
# Load conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate env
conda activate bubbleseg

# Start TensorBoard
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

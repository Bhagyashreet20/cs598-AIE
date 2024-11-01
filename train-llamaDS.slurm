#!/bin/bash

#SBATCH --mem=200g 
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=train
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --time=01:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/code/logs/slurm-logs/output/train.%j.out
#SBATCH --error=/projects/bdof/code/logs/slurm-logs/error/train.%j.err

export HF_HOME="/projects/bdof/code/cs598-AIE/custom_hf_cache"
mkdir -p $HF_HOME
export TRITON_CACHE_DIR="/projects/bdof/code/cs598-AIE/custom_triton_cache"
mkdir -p $TRITON_CACHE_DIR

# Load necessary modules
module load gcc/11.4.0 
module load python/3.10.13
module load cuda/12.4.0

# Activate virtual environment
source /projects/bdof/code/cs598-AIE/myenv/bin/activate  

# Install requirements
#pip install --upgrade torch torchvision transformers accelerate bitsandbytes datasets deepspeed

export MASTER_ADDR="localhost"
export MASTER_PORT="29568"

nvidia-smi

# Launch DeepSpeed with the training script
deepspeed train-deepspeed.py

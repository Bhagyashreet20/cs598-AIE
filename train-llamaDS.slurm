#!/bin/bash

#SBATCH --mem=128g 
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=train
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --gpus-per-task=3
#SBATCH --time=03:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/nkanamarla/logs/output/train.%j.out
#SBATCH --error=/projects/bdof/nkanamarla/logs/error/train.%j.err

export HF_HOME="/projects/bdof/code/cs598-AIE/custom_hf_cache"
mkdir -p $HF_HOME
export TRITON_CACHE_DIR="/projects/bdof/code/cs598-AIE/custom_triton_cache"
mkdir -p $TRITON_CACHE_DIR

# Load necessary modules
module load gcc/11.4.0 
module load python/3.10.13
module load cuda/11.8.0   # Updated CUDA version

# Activate virtual environment
source /projects/bdof/code/cs598-AIE/myenv/bin/activate  

# Profiling 
export TRANSFORMERS_VERBOSITY=debug
export DEEPSPEED_LOG_LEVEL=debug
export WANDB_DISABLED=true

# Install requirements 
#pip install --upgrade torch torchvision transformers accelerate bitsandbytes datasets deepspeed

export MASTER_ADDR="localhost"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")

nvidia-smi

# Launch DeepSpeed with the training script
deepspeed train-deepspeed.py 


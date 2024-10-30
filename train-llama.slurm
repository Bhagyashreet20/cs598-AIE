#!/bin/bash

#SBATCH --mem=160g #cpu memory
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=train
#SBATCH --gpus=3
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --gpus-per-task=3
#SBATCH --time=01:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/code/logs/slurm-logs/output/btaleka-train.%j.out
#SBATCH --error=/projects/bdof/code/logs/slurm-logs/error/btaleka-train.%j.err

export HF_HOME="/projects/bdof/code/cs598-AIE/custom_hf_cache"
mkdir -p $HF_HOME

# Load necessary modules
module load gcc
module load python/3.10.13
module load cuda/12.4.0


python3 -m venv myenv     
source /projects/bdof/code/cs598-AIE/myenv/bin/activate  

pip install torch torchvision torchaudio
pip install transformers accelerate
pip install bitsandbytes datasets
pip install --upgrade torch torchvision transformers accelerate bitsandbytes datasets

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
# export RANK=0
# export WORLD_SIZE=2
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"  # Any free port on your system
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
torchrun --nproc_per_node=3 --master_port=$MASTER_PORT train.py

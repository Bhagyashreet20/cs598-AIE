#!/bin/bash

#SBATCH --mem=80g #cpu memory
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdoy-delta-gpu
#SBATCH --job-name=t5
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --time=00:20:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/leatherman/logs/t5.%j.out
echo STARTTIME `date`

# Load necessary modules
module load gcc
module load python/3.10.13
module load cuda/12.4.0
mkdir /projects
export HF_HOME="/projects/bdoy/leatherman/custom_hf_cache"

# python3 -m venv leatherman_env_llama3
# source /u/leatherman/.bashrc
# conda activate llama3
source /projects/bdof/leatherman/leatherman_env_llama3/bin/activate  
# pip install sentencepiece
# pip install accelerate
# export RANK=$SLURM_PROCID
# export WORLD_SIZE=$SLURM_NTASKS
# export RANK=0
# export WORLD_SIZE=2
MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_ADDR
export MASTER_PORT="12355 "  # Any free port on your system
#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=$((29500 + RANDOM % 100))
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
torchrun --nnodes 1 --nproc_per_node 2 train_t5.py
# CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train.py
# python train.py
# python -m torch.distributed.launch --nnodes=4 --nproc_per_node=1 train.py
echo ENDTIME `date`

#!/bin/bash

#SBATCH --mem=128g
#SBATCH --partition=gpuA100x8
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=train
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --time=02:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/code/cs598-AIE/accelerate_variants/examples/logs/output/train.%j.out
#SBATCH --error=/projects/bdof/code/cs598-AIE/accelerate_variants/examples/logs/error/train.%j.err

export HF_HOME="/work/hdd/bdof/custom_hf_cache"
export WANDB_API_KEY="aace6040e177861d2be40afc802b8f4e9cd0a351"
mkdir -p $HF_HOME
export TRANSFORMERS_VERBOSITY=info

# Load necessary modules
module load gcc
module load python/3.10.13
module load cuda/11.8.0   # Updated CUDA version

source /projects/bdof/code/cs598-AIE/myenv/bin/activate

# accelerate launch --num_processes 2 --config_file /projects/bdof/code/cs598-AIE/accelerate_variants/examples/default_config.yaml \
#     summarization-llama.py \
#     --mixed_precision fp16 \
#     --checkpointing_steps 100 \
#     --use_stateful_dataloader \
#     --with_tracking \
#     --output_dir ./checkpoints \
#     --project_dir ./training-logs


accelerate launch --num_processes 2 --config_file /projects/bdof/code/cs598-AIE/accelerate_variants/examples/default_config.yaml \
    summarization-llama.py \
    --checkpointing_steps 100 \
    --use_stateful_dataloader \
    --output_dir ./checkpoints \



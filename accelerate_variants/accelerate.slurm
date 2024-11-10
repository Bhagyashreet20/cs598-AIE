#!/bin/bash

#SBATCH --mem=160g #cpu memory
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=train
#SBATCH --gpus=2
#SBATCH --nodes=1                   # Number of nodes to use
#SBATCH --ntasks=1                  # Number of tasks per job (usually 1)
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=2          # Number of CPUs per task
#SBATCH --time=03:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/code/cs598-AIE/logs/output/train.%j.out
#SBATCH --error=/projects/bdof/code/cs598-AIE/logs/error/train.%j.err

echo "Starting job at $(date)" > /projects/bdof/code/cs598-AIE/logs/output/test_log.out
export HF_HOME="/projects/bdof/code/cs598-AIE/custom_hf_cache"
mkdir -p $HF_HOME
mkdir -p /projects/bdof/code/cs598-AIE/logs/output
mkdir -p /projects/bdof/code/cs598-AIE/logs/error

# Load necessary modules
module load gcc
module load python/3.10.13
module load cuda/12.4.0

# python3 -m venv myenv     
source /projects/bdof/code/cs598-AIE/myenv/bin/activate

# Run the fine-tuning script using `accelerate launch`
accelerate launch \
    --multi_gpu \
    FSDP.py  # Replace with the path to your fine-tuning script file

echo "Job completed at $(date)"
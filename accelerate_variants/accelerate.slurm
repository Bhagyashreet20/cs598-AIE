#!/bin/bash

#SBATCH --mem=160g #cpu memory
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=trainAccelerate
#SBATCH --gpus=2
#SBATCH --nodes=1                   # Number of nodes to use
#SBATCH --ntasks=1                  # Number of tasks per job (usually 1)
#SBATCH --cpus-per-task=2          # Number of CPUs per task
#SBATCH --time=03:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/code/logs/output/trainAccelerate.%j.out
#SBATCH --error=/projects/bdof/code/logs/error/trainAccelerate.%j.err

echo "Starting job at $(date)" 
nvidia-smi
export HF_HOME="/projects/bdof/code/cs598-AIE/custom_hf_cache"
mkdir -p $HF_HOME

# Load necessary modules
module load gcc
module load python/3.10.13
module load cuda/12.4.0

# python3 -m venv myenv     
source /projects/bdof/code/cs598-AIE/myenv/bin/activate

# Run the fine-tuning script using `accelerate launch`
accelerate launch \
    --multi_gpu \
    /projects/bdof/code/cs598-AIE/accelerate_variants/FSDP.py  # Replace with the path to your fine-tuning script file

echo "Job completed at $(date)"
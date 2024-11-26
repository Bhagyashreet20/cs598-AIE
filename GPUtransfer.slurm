#!/bin/bash

#SBATCH --mem=64g 
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=GPU-transfer
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --time=00:30:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/nkanamarla/logs/output/GPU-transfer.%j.out
#SBATCH --error=/projects/bdof/nkanamarla/logs/error/GPU-transfer.%j.err

# Load necessary modules
module load gcc/11.4.0 
module load cuda/11.8.0   # Updated CUDA version

# Print job details
echo "Running on host: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

# Compile the CUDA program
nvcc -o transfer_weights /projects/bdof/code/cs598-AIE/GPUtransfer.cu
if [ $? -ne 0 ]; then
    echo "CUDA program compilation failed."
    exit 1
fi
echo "CUDA program compiled successfully."

# Run the CUDA program
srun ./transfer_weights


#!/bin/bash

#SBATCH --mem=64g 
#SBATCH --partition=gpuA100x4             # Request resources from gpuA100x4 partition
#SBATCH --account=bdof-delta-gpu         # Specify account
#SBATCH --job-name=GPU-transfer          # Job name
#SBATCH --cpus-per-gpu=8                 # CPUs per GPU
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus=2                         # Total GPUs
#SBATCH --ntasks=1                       # Number of tasks (1 per node)
#SBATCH --gpus-per-node=2                # GPUs per node
#SBATCH --gpus-per-task=2                # GPUs per task
#SBATCH --time=00:10:00                  # Maximum runtime
#SBATCH --constraint="scratch"           # Additional constraints
#SBATCH --output=/projects/bdof/nkanamarla/logs/output/GPU-transfer.%j.out
#SBATCH --error=/projects/bdof/nkanamarla/logs/error/GPU-transfer.%j.err

# Load necessary modules
module load gcc/11.4.0 
module load cuda/11.8.0   # Load the correct CUDA module

# Set LD_LIBRARY_PATH if necessary
export LD_LIBRARY_PATH=/u/nkanamarla/nvcomp-linux-x86_64-4.1.0.6_cuda11-archive/lib:$LD_LIBRARY_PATH

# Print job details
echo "Running on host: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Compile the CUDA program
nvcc -o transfer_weights_compression /projects/bdof/code/cs598-AIE/GPUtransfer-compression.cu \
-I/u/nkanamarla/nvcomp-linux-x86_64-4.1.0.6_cuda11-archive/include \
-L/u/nkanamarla/nvcomp-linux-x86_64-4.1.0.6_cuda11-archive/lib -lnvcomp

if [ $? -ne 0 ]; then
    echo "CUDA program compilation failed."
    exit 1
fi
echo "CUDA program compiled successfully."

# Run the CUDA program
srun ./transfer_weights_compression

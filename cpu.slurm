#!/bin/bash
#SBATCH --mem=64g
#SBATCH --partition=cpu
#SBATCH --account=bdof-delta-cpu
#SBATCH --job-name=CPU-convert
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    
#SBATCH --time=00:15:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/nkanamarla/logs/output/cpu.%j.out
#SBATCH --error=/projects/bdof/nkanamarla/logs/error/cpu.%j.err

#load packages
module load gcc
module load python/3.10.13

# Activate virtual environment
source /projects/bdof/code/cs598-AIE/myenv/bin/activate 

python convertFile.py /work/hdd/bdof/nkanamarla/models/sample_model_checkpoint/model-00001-of-00002.safetensors /work/hdd/bdof/nkanamarla/models/sample_model_checkpoint/checkpoint.bin

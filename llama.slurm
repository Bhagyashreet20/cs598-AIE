#!/bin/bash

#SBATCH --mem=160g #cpu memory
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=train
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --time=03:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/code/cs598-AIE/logs/output/train.%j.out
#SBATCH --error=/projects/bdof/code/cs598-AIE/logs/error/train.%j.err

export HF_HOME="/work/hdd/bdof/custom_hf_cache"
mkdir -p $HF_HOME

# Load necessary modules
module load gcc
module load python/3.10.13
module load cuda/12.4.0

# python3 -m venv myenv     
source /projects/bdof/code/cs598-AIE/myenv/bin/activate

cd llama-recipes
pip install -U pip setuptools
pip install -e .
torchrun --nnodes 1 --nproc_per_node 4  ./llama-recepies/recipes/quickstart/finetuning/finetuning.py --enable_fsdp --model_name /work/hdd/bdof/custom_hf_cache/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062 --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --fsdp_config.pure_bf16 --use_fast_kernels

#pip install wandb
#!/bin/bash

#SBATCH --mem=128g
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdof-delta-gpu
#SBATCH --job-name=train
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --time=02:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/code/cs598-AIE/logs/output/train.%j.out
#SBATCH --error=/projects/bdof/code/cs598-AIE/logs/error/train.%j.err

export HF_HOME="/work/hdd/bdof/custom_hf_cache"
mkdir -p $HF_HOME

# Load necessary modules
module load gcc
module load python/3.10.13
module load cuda/12.4.0
module load gcc-runtime/11.4.0
#module load anaconda3_gpu

# Step 1: Download and Build SQLite Locally
# export PREFIX=$HOME/sqlite
# mkdir -p $PREFIX
# wget https://www.sqlite.org/2023/sqlite-autoconf-3410000.tar.gz
# tar xzf sqlite-autoconf-3410000.tar.gz
# cd sqlite-autoconf-3410000
# ./configure --prefix=$PREFIX
# make
# make install

# cd /projects/bdof/code/cs598-AIE/accelerate_variants/examples
# pwd

# Step 2: Update Environment Variables to Use Local SQLite Installation
# export LD_LIBRARY_PATH=$HOME/sqlite/lib:$LD_LIBRARY_PATH
# export PKG_CONFIG_PATH=$HOME/sqlite/lib/pkgconfig:$PKG_CONFIG_PATH

#conda create -n condaenv python=3.10
# eval "$(conda shell.bash hook)"
# conda create -y -n condaenv_1 python=3.10
# conda activate condaenv_1

source /projects/bdof/code/cs598-AIE/myenv/bin/activate
python --version
echo $PYTHONPATH
# pip install evaluate rouge-score nltk torchdata==0.8.0
pip install --upgrade accelerate

# python -c "import sqlite3; print(sqlite3.sqlite_version)"
#
accelerate launch --num_processes 2 --config_file /projects/bdof/code/cs598-AIE/accelerate_variants/examples/default_config.yaml \
    summarization-llama.py \
    --mixed_precision fp16 \
    --checkpointing_steps 100 \
    --use_stateful_dataloader \
    --with_tracking \
    --output_dir ./checkpoints \
    --project_dir ./training-logs

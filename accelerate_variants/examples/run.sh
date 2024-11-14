export CUDA_LAUNCH_BLOCKING=1
python summarization.py \
    --mixed_precision fp16 \
    --checkpointing_steps 2 \
    --use_stateful_dataloader \
    --with_tracking \
    --output_dir ./checkpoints \
    --project_dir ./logs



#     python train-complete.py \
#     --mixed_precision fp16 \
#     --checkpointing_steps epoch \
#     --use_stateful_dataloader \
#     --with_tracking \
#     --output_dir ./checkpoints \
#     --project_dir ./logs


# python train-complete.py --output_dir ./checkpoints --project_dir ./logs

#TODO:change this to slurm script




srun --mem=160g \
     --partition=gpuA100x4-interactive \
     --account=bdof-delta-gpu \
     --job-name=train \
     --gpus=2 \
     --ntasks=1 \
     --gpus-per-node=2 \
     --gpus-per-task=2 \
     --time=02:00:00 \
     --constraint="scratch" \
     --output=/projects/bdof/code/cs598-AIE/logs/output/train.%j.out \
     --error=/projects/bdof/code/cs598-AIE/logs/error/train.%j.err \
     apptainer exec --nv \
     --bind /projects/bdof/code/cs598-AIE:/projects/bdof/code/cs598-AIE,$HF_HOME:$HF_HOME \
     pytorch_24.09-py3.sif \
     bash -c "
     module load gcc;
     module load python/3.10.13;
     module load cuda/12.4.0;
     export HF_HOME='/work/hdd/bdof/custom_hf_cache';
     mkdir -p $HF_HOME;
     pip install nltk rouge-score evaluate;
     python /projects/bdof/code/cs598-AIE/summarization-llama.py \
         --mixed_precision fp16 \
         --checkpointing_steps 2 \
         --use_stateful_dataloader \
         --with_tracking \
         --output_dir ./checkpoints \
         --project_dir ./logs
     "


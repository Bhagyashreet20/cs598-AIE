# CS598-AIE Experiments

This repository contains scripts and configurations for running various experiments related to deep learning, model training, and GPU-based optimizations. Follow the instructions to set up the environment and execute each experiment.

---

## Table of Contents

- [Setup](#setup)
- [Experiments](#experiments)
  - [Experiment 1: Deepspeed](#experiment-1-deepspeed)
  - [Experiment 2: FSDP (Fully Sharded Data Parallel)](#experiment-2-fsdp-fully-sharded-data-parallel)
  - [Experiment 3: GPU to GPU Transfer](#experiment-3-gpu-to-gpu-transfer)
  - [Experiment 4: T5 Training with GPU Cache Checkpointing](#experiment-4-t5-training-with-gpu-cache-checkpointing)
- [Notes](#notes)
- [License](#license)

---

## Setup

1. **Create a Virtual Environment**
   - Ensure you have Python installed on your system. 
   - Create a virtual environment:
     ```bash
     python -m venv env
     ```
   - Activate the virtual environment:
     ```bash
     # On Linux/Mac
     source env/bin/activate
     
     # On Windows
     .\env\Scripts\activate
     ```
    - Disclaimer: There may be missing packages and the SLURM scripts may need to be adjusted.

2. **Install Dependencies**
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Setup Hugging Face**
   - Configure your Hugging Face account to use models:
     ```bash
     huggingface-cli login
     ```

4. **Setup WandB**
   - Log in to Weights & Biases:
     ```bash
     wandb login
     ```

---

## Experiments

### Experiment 1: Deepspeed
- **Objective**: Fine-tune LLAMA 3.2 3B model using DeepSpeed for optimized distributed training.
- **Script**: `code/cs598-AIE/train-llamaDS.slurm`
- **Output**: `code/cs598-AIE/experiment1.out`
- **Run Command**:
  ```bash
  sbatch train-llamaDS.slurm

### Experiment 2: FSDP
- **Objective**:  Perform summarization tasks using Fully Sharded Data Parallel (FSDP) for Fine-tune LLAMA 3.2 3B model.
- **Script**: `code/cs598-AIE/accelerate_variants/summarization.slurm`
- **Output**: `code/cs598-AIE/experiment2.out`
- **Run Command**:
  ```bash
  sbatch accelerate_variants/summarization.slurm


### Experiment 3: GPU to GPU transfer 
- **Objective**:  Test GPU-to-GPU data transfer to improve performance for asnyc checkpointing.
- **Script**: `code/cs598-AIE/GPUtransfer-basic.slurm`
- **Output**: `code/cs598-AIE/experiment3.out`
- **Run Command**:
  ```bash
  sbatch GPUtransfer-basic.slurm

### Experiment 4A: T5 training with GPU checkpointing
- **Objective**:  This experiment focuses on optimizing training processes by caching results on GPUs and efficiently managing checkpoints.
- **Script**: `code/cs598-AIE/train_t5.slurm`
- **Output**: `code/cs598-AIE/experiment4.out`
- **Run Command**:
  ```bash
  sbatch train_t5.slurm

### Experiment 4B: T5 training with GPU checkpointing
- **Objective**:  This experiment focuses on optimizing training processes by caching checkpoints on GPUs across nodes.
- **Script**: `code/cs598-AIE/two_nodes/train_t5_two_nodes0.slurm and code/cs598-AIE/two_nodes/train_t5_two_nodes1.slurm`
- **Output**: `code/cs598-AIE/two_nodes/t5_node0.5719534.out and code/cs598-AIE/two_nodes/t5_node0.5719535.out`
- **Run Command**:
  ```bash
  sbatch two_nodes/train_t5_two_nodes0.slurm
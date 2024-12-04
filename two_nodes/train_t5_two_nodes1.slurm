#!/bin/bash
#SBATCH --mem=20g #cpu memory
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdoy-delta-gpu
#SBATCH --job-name=t5
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bdof/leatherman/logs/t5_node1.%j.out
echo STARTTIME `date`
module load gcc
module load python/3.10.13
module load cuda/12.4.0
export HF_HOME="/projects/bdoy/leatherman/custom_hf_cache"
source /projects/bdof/leatherman/leatherman_env_llama3/bin/activate  

# Function to check basic network reachability
check_connectivity() {
    local addr=$1
    echo "=== Testing connectivity to $addr ==="
    
    echo "Current host info:"
    hostname
    hostname -I
    
    # Try ping
    if ping -c 1 -W 5 $addr > /dev/null 2>&1; then
        echo "✓ Can reach $addr"
        return 0
    else
        echo "✗ Cannot reach $addr"
        return 1
    fi
}



timeout=300  # 5 minutes
start_time=$(date +%s)
while [ ! -s /projects/bdof/leatherman/master_info.env ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo "Timeout waiting for master_info.env"
        exit 1
    fi
    sleep 5
    echo "Waiting for master node information... (${elapsed}s elapsed)"
done
source /projects/bdof/leatherman/master_info.env
export MASTER_PORT="12355"  # Any free port on your system
export WORLD_SIZE=2
export RANK=1
export LOCAL_RANK=0
export NODE_ROLE="job1"
export NODE_ROLE="worker"
echo "=== Environment Settings ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_ROLE: $NODE_ROLE"
echo "RANK: $RANK"
echo "=========================="

if ! check_connectivity $MASTER_ADDR $MASTER_PORT; then
    echo "Connectivity check failed! Exiting..."
    exit 1
fi

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_t5_two_nodes.py
echo ENDTIME `date`

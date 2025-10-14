#!/bin/bash
#SBATCH --time=600
#SBATCH --job-name=nus-cs4248-project-mt
#SBATCH --output=train_mt_%j.out
# SBATCH --gpus=2 
#SBATCH --gres=gpu:a100-80:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS visible GPU(s)"

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# Activate virtual environment (adjust path as needed)
# if [ -d "/home/n/<SoC_username>/envs/<venv_name>" ]; then
#     source /home/n/<SoC_username>/envs/<venv_name>/bin/activate
# fi

# Create directories
mkdir -p models

# conda activate
source ~/miniconda3/bin/activate
conda activate $ENV_NAME

echo "=== Training Script ==="
echo "Starting training at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

# Run the training script
echo "Training script started..."

# Launch distributed training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_mt.py \
    --config $HOME/cs4248-project-mt/configs/mT5-small-training.yaml

echo "Training script executed. Check the output file for details."
#!/bin/bash
#SBATCH --time=1200
#SBATCH --job-name=nus-cs4248-project-mt
#SBATCH --output=./logs/train_mt_single_gpu_%j.out
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# Create directories
mkdir -p models logs outputs

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

python train_mt.py --config ./configs/training.yaml

echo "Training script executed. Check the output file for details."


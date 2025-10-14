#!/bin/bash
#SBATCH --time=600
#SBATCH --job-name=nus-cs4248-project-mt
#SBATCH --output=train_mt_multi_gpu_%j.out
#SBATCH --gres=gpu:h100-96:2 # explicitly request 2 H100-96GB GPUs
## SBATCH --gpus=2 # implicitly requests 2 GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16

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

# --- Count visible MIG devices (PyTorch sees MIGs as normal CUDA devices) ---
NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

echo "Detected $NUM_GPUS visible CUDA device(s)"

# Run the training script
echo "Training script started..."

# Launch distributed training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_mt.py \
    --config $HOME/cs4248-project-mt/configs/training.yaml

echo "Training script executed. Check the output file for details."
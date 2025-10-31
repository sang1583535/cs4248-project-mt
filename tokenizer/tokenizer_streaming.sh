#!/bin/bash
#SBATCH --time=1200
#SBATCH --job-name=nus-cs4248-project-mt-tokenizer
#SBATCH --output=tokenizer_%j.out
#SBATCH --mem=32G
#SBATCH --tmp=100G 

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# conda activate
source ~/miniconda3/bin/activate
conda activate $ENV_NAME

echo "=== Tokenzing Script ==="
echo "Starting tokenzing at $(date)"
echo "Running on host: $(hostname)"
echo "Available memory: $SLURM_MEM_PER_NODE MB"

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the tokenizing script
python3 tokenizer_streaming.py \
    --src-path $HOME/wmt22-zhen/train.zho \
    --tgt-path $HOME/wmt22-zhen/train.eng \
    --save-dir $HOME/cs4248-project-mt/tokenized_dataset/WMT22_Train \
    --max-length 128 \
    --chunk-size 25000

echo "Tokenizing script executed. Check the output file for details."
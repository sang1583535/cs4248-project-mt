#!/bin/bash
#SBATCH --time=30              # 30 minutes should be enough for verification
#SBATCH --job-name=verify-subset
#SBATCH --output=./logs/verify_subset_%j.out
#SBATCH --error=./logs/verify_subset_%j.err
#SBATCH --mem=32G              # Memory to load datasets
#SBATCH --cpus-per-task=4      # Fewer CPUs needed for verification

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# Dataset paths - customize these as needed
ORIGINAL_DATASET="/home/n/ntasang/cs4248-project-mt/tokenized_dataset/WMT22_Train_Merged"
SUBSET_DATASET="/your/path/to/tokenized_dataset/WMT22_Train_Merged_5pct"

# Number of sample records to display
NUM_SAMPLES=3

# Create directories
mkdir -p logs

# conda activate
source ~/miniconda3/bin/activate
conda activate $ENV_NAME

echo "=== Dataset Subset Verification ==="
echo "Starting at $(date)"
echo "Running on host: $(hostname)"
echo "Original dataset: $ORIGINAL_DATASET"
echo "Subset dataset: $SUBSET_DATASET"
echo ""

# Run verification
python verify_subset.py \
    --original "$ORIGINAL_DATASET" \
    --subset "$SUBSET_DATASET" \
    --samples "$NUM_SAMPLES"

echo ""
echo "Verification completed at $(date)"


#!/bin/bash
#SBATCH --time=120        # 2 hours should be enough for subset creation
#SBATCH --job-name=create-wmt22-subset
#SBATCH --output=./logs/create_subset_%j.out
#SBATCH --error=./logs/create_subset_%j.err
#SBATCH --mem=64G         # Sufficient memory for loading large dataset
#SBATCH --cpus-per-task=8 # Multiple CPUs help with dataset operations
#SBATCH --tmp=50G         # Temporary space for intermediate operations

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# Dataset paths - customize these as needed
INPUT_DATASET="/home/n/ntasang/cs4248-project-mt/tokenized_dataset/WMT22_Train_Merged"
# Save to: /home/m/man0302/... (your user directory)
OUTPUT_SUBSET="/your/path/to/tokenized_dataset/WMT22_Train_Merged_5pct"

# Subset options - choose one:
# Option 1: Percentage (e.g., 0.05 for 5%)
SUBSET_PERCENTAGE=0.05
# Option 2: Fixed size (uncomment and use if preferred)
# SUBSET_SIZE=1000000

# Random seed for reproducibility
SEED=42

# Create directories
mkdir -p logs

# conda activate
source ~/miniconda3/bin/activate
conda activate $ENV_NAME

echo "=== Dataset Subset Creation Script ==="
echo "Starting at $(date)"
echo "Running on host: $(hostname)"
echo "Input dataset: $INPUT_DATASET"
echo "Output subset: $OUTPUT_SUBSET"

# Check if input dataset exists
if [ ! -d "$INPUT_DATASET" ]; then
    echo "ERROR: Input dataset directory does not exist: $INPUT_DATASET"
    exit 1
fi

# Create the subset
echo "Creating subset..."
if [ ! -z "$SUBSET_PERCENTAGE" ]; then
    python create_dataset_subset.py \
        --input "$INPUT_DATASET" \
        --output "$OUTPUT_SUBSET" \
        --percentage "$SUBSET_PERCENTAGE" \
        --seed "$SEED"
elif [ ! -z "$SUBSET_SIZE" ]; then
    python create_dataset_subset.py \
        --input "$INPUT_DATASET" \
        --output "$OUTPUT_SUBSET" \
        --size "$SUBSET_SIZE" \
        --seed "$SEED"
else
    echo "ERROR: Either SUBSET_PERCENTAGE or SUBSET_SIZE must be set"
    exit 1
fi

echo "Subset creation completed at $(date)"
echo "Output saved to: $OUTPUT_SUBSET"


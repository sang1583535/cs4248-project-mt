#!/bin/bash
#SBATCH --time=600
#SBATCH --job-name=nus-cs4248-project-mt
#SBATCH --output=./logs/evaluation_mt_%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# Create directories
mkdir -p logs

# conda activate
source ~/miniconda3/bin/activate
conda activate $ENV_NAME

echo "=== Evaluation Script ==="
echo "Starting evaluation at $(date)"
echo "Running on host: $(hostname)"

# Paths (change these paths as necessary)
TATOEBA_SRC="./dataset/tatoeba.zh"
TATOEBA_REF="./dataset/tatoeba.en"
TATOEBA_PRED="./outputs/tatoeba_mt5_large.en"
WMT_SRC="./dataset/wmttest2022.zh"
WMT_REF="./dataset/wmttest2022.AnnA.en"
WMT_PRED="./outputs/wmt_mt5_large.en"

# Calculate BLEU score
echo "Computing SACREBLEU score..."
sacrebleu -tok 13a -w 2 $TATOEBA_REF < $TATOEBA_PRED
sacrebleu -tok 13a -w 2 $WMT_REF < $WMT_PRED

# Calculate COMET score
echo "Computing COMET score..."
comet-score -s $TATOEBA_SRC \
    -t $TATOEBA_PRED \
    -r $TATOEBA_REF \
    --batch_size 256 \
    --gpus 1 \
    --num_workers 16 \
    --only_system

comet-score -s $WMT_SRC \
    -t $WMT_PRED \
    -r $WMT_REF \
    --batch_size 256 \
    --gpus 1 \
    --num_workers 16 \
    --only_system

echo "Inference complete at $(date)"

#!/bin/bash
#SBATCH --time=600
#SBATCH --job-name=nus-cs4248-project-mt
#SBATCH --output=evaluation_mt_%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# conda activate
source ~/miniconda3/bin/activate
conda activate $ENV_NAME

echo "=== Evaluation Script ==="
echo "Starting evaluation at $(date)"
echo "Running on host: $(hostname)"

# Paths (change these paths as necessary)
MODEL_PATH="$HOME/cs4248-project-mt/models/mt5-large-finetuned-single-gpu/checkpoint-19260"
TATOEBA_SRC="$HOME/cs4248-project-mt/dataset/tatoeba.zh"
TATOEBA_REF="$HOME/cs4248-project-mt/dataset/tatoeba.en"
TATOEBA_PRED="$HOME/cs4248-project-mt/outputs/tatoeba_mt5_large.en"
WMT_SRC="$HOME/cs4248-project-mt/dataset/wmttest2022.zh"
WMT_REF="$HOME/cs4248-project-mt/dataset/wmttest2022.AnnA.en"
WMT_PRED="$HOME/cs4248-project-mt/outputs/wmt_mt5_large.en"

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
    --model_storage_path $MODEL_PATH 

comet-score -s $WMT_SRC \
    -t $WMT_PRED \
    -r $WMT_REF \
    --batch_size 256 \
    --gpus 1 \
    --num_workers 16 \
    --model_storage_path $MODEL_PATH

echo "Inference complete at $(date)"

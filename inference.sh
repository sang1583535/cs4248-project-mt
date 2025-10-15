#!/bin/bash
#SBATCH --time=600
#SBATCH --job-name=nus-cs4248-project-mt
#SBATCH --output=inference_mt_%j.out
#SBATCH --gpus=1 # implicitly requests 1 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# conda activate
source ~/miniconda3/bin/activate
conda activate $ENV_NAME

echo "=== Inference Script ==="
echo "Starting Inference at $(date)"
echo "Running on host: $(hostname)"

# Create directories
mkdir -p outputs

# Example usage of inference.py
# Uncomment the following line to run inference with specified model and input text
# python3 inference.py \
# --model-path $HOME/cs4248-project-mt/models/mt5-large-finetuned/checkpoint-310 \
# --input-text "28岁厨师被发现死于旧金山一家商场" \

# Or run inference on a file
# Define paths. Adjust these paths as necessary.
MODEL_PATH="$HOME/cs4248-project-mt/models/mt5-large-finetuned-single-gpu/checkpoint-19260"
SRC_FILE="$HOME/cs4248-project-mt/dataset/tatoeba.zh"
OUTPUT_FILE="$HOME/cs4248-project-mt/outputs/tatoeba_mt5_large.en"
REF_FILE="$HOME/cs4248-project-mt/dataset/tatoeba.en"

python3 inference.py \
--model-path "$MODEL_PATH" \
--input-file "$SRC_FILE" \
--output-file "$OUTPUT_FILE" 

# BLEU score 
echo "Computing SACREBLEU score..."
sacrebleu -tok 13a -w 2 $REF_FILE < $OUTPUT_FILE

# COMET score
echo "Computing COMET score..."
comet-score -s $SRC_FILE \
    -t $OUTPUT_FILE \
    -r $REF_FILE \
    --batch_size 256 \
    --gpus 1 \
    --num_workers 16 \
    --model_storage_path $MODEL_PATH

echo "Inference complete at $(date)"
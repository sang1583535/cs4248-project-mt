#!/bin/bash
#SBATCH --time=600
#SBATCH --job-name=nus-cs4248-project-mt
#SBATCH --output=inference_mt_%j.out
#SBATCH --gpus=1 # implicitly requests 1 GPUs
#SBATCH --ntasks-per-node=1
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
python3 inference.py \
--model-path $HOME/cs4248-project-mt/models/mt5-large-finetuned/checkpoint-310 \
--input-file $HOME/cs4248-project-mt/dataset/tatoeba.zh \
--output-file $HOME/cs4248-project-mt/outputs/tatoeba_mt5_large.en

# BLEU score 
echo "Computing SACREBLEU score..."
sacrebleu -tok 13a -w 2 $HOME/cs4248-project-mt/dataset/tatoeba.en < $HOME/cs4248-project-mt/outputs/tatoeba_mt5_large.en

# COMET score
echo "Computing COMET score..."
comet-score -s $HOME/cs4248-project-mt/dataset/tatoeba.zh \
    -t $HOME/cs4248-project-mt/outputs/tatoeba_mt5_large.en \
    -r $HOME/cs4248-project-mt/dataset/tatoeba.en \
    --batch_size 256 \
    --gpus 1 \
    # --model $HOME/cs4248-project-mt/models/mt5-large-finetuned/checkpoint-310

echo "Inference complete at $(date)"
#!/bin/bash
#SBATCH --time=1200
#SBATCH --job-name=nus-cs4248-project-mt-merging
#SBATCH --output=tokenizer_merging_%j.out
#SBATCH --mem=32G
#SBATCH --tmp=100G 

# Define environment name
ENV_NAME="mt_env" # Ensure this matches the name in env-setup-miniconda.sh

# conda activate
source ~/miniconda3/bin/activate
conda activate $ENV_NAME

python3 chunk_merging.py \
--chunks-dir ../tokenized_dataset/WMT22_Train \
--output-dir ../tokenized_dataset/WMT22_Train_Merged

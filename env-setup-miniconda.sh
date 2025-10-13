#!/bin/bash
#SBATCH --time=600 
#SBATCH --job-name=nus-cs4248-project-mt

# Description: This script sets up a conda environment for the project.
# It checks if Miniconda is installed, installs it if not, creates a conda
# environment, and installs necessary packages.
# Usage: sbatch env-setup-miniconda.sh

# Check if Miniconda is installed
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Miniconda not found, installing..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
else
    echo "Miniconda already installed"
fi

# Define environment name
ENV_NAME="mt_env"
PYTHON_VERSION="3.10"

# Load conda
source ~/miniconda3/bin/activate

# Check if environment exists
if conda info --envs | grep -q $ENV_NAME; then
    echo "Conda environment $ENV_NAME already exists"
else
    echo "Creating conda environment $ENV_NAME"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y  # Adjust Python version as needed
fi

# Activate the environment
conda activate $ENV_NAME

# Install necessary packages
echo "Installing necessary packages..."
pip install -r requirements.txt

# Done
echo "Environment setup complete. Activate it using: conda activate $ENV_NAME"
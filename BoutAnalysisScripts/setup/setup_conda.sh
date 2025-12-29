#!/bin/bash
# Setup script for Conda environment
#
# This script creates a conda environment and installs all required dependencies
# for the behavior bout analysis pipeline.
#
# Usage:
#   bash BoutAnalysisScripts/setup/setup_conda.sh

set -e  # Exit on error

ENV_NAME="behavior_analysis"

echo "============================================================"
echo "Setting up Conda Environment"
echo "============================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda."
    echo "Visit: https://www.anaconda.com/products/distribution"
    exit 1
fi

echo "Found conda: $(conda --version)"
echo ""

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
fi

# Create new environment
echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install packages
echo ""
echo "Installing Python dependencies..."
pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo "============================================================"
echo "Conda environment setup complete!"
echo "============================================================"
echo ""
echo "To activate the conda environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""


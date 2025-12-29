#!/bin/bash
# Setup script for Python virtual environment
#
# This script creates a virtual environment and installs all required dependencies
# for the behavior bout analysis pipeline.
#
# Usage:
#   bash analysis_r/setup_venv.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "============================================================"
echo "Setting up Python Virtual Environment"
echo "============================================================"
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment in: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "============================================================"
echo "Virtual environment setup complete!"
echo "============================================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source analysis_r/venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "The virtual environment will be used automatically by the pipeline scripts."


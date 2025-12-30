#!/bin/bash
# Script to run analysis with environment check and fallback options

set -e

echo "Checking Python environment..."

# Test if numpy works
if python3 -c "import numpy; print('numpy OK')" 2>/dev/null; then
    echo "✓ Current Python environment is working"
    PYTHON_CMD="python3"
elif /usr/bin/python3 -c "import numpy; print('numpy OK')" 2>/dev/null 2>&1; then
    echo "✓ Using system Python"
    PYTHON_CMD="/usr/bin/python3"
elif [ -f "/opt/homebrew/bin/python3.11" ] && /opt/homebrew/bin/python3.11 -c "import numpy; print('numpy OK')" 2>/dev/null; then
    echo "✓ Using Homebrew Python"
    PYTHON_CMD="/opt/homebrew/bin/python3.11"
else
    echo "ERROR: No working Python environment found!"
    echo ""
    echo "Please fix your Python environment first:"
    echo "1. Create a new conda environment:"
    echo "   conda create -n behavior_analysis python=3.10 numpy h5py pandas -y"
    echo "   conda activate behavior_analysis"
    echo ""
    echo "2. Or see FIX_H5PY_INSTALLATION.md for detailed instructions"
    exit 1
fi

echo "Using: $PYTHON_CMD"
echo ""

# Run the analysis
cd "$(dirname "$0")"
$PYTHON_CMD run_complete_analysis.py \
    --behavior turn_left \
    --annotations-dir ../jabs/annotations \
    --features-dir ../jabs/features \
    --video-dir .. \
    --output-dir BoutResults \
    --workers 7 \
    "$@"


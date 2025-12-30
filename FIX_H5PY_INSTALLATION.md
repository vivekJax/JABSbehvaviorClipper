# Fixing h5py/numpy Segmentation Fault Issue

## Problem
The Anaconda Python environment has corrupted numpy/h5py installations causing segmentation faults when importing these libraries.

## Diagnosis
- **System**: macOS 15.7.2 (ARM64/Apple Silicon)
- **Python**: 3.10.9 (Anaconda)
- **Issue**: Segmentation fault when importing numpy, h5py, or pandas
- **Root cause**: Corrupted binary libraries in Anaconda base environment

## Solutions (Choose One)

### Option 1: Create a New Conda Environment (Recommended)

```bash
# Create a new environment with working packages
conda create -n behavior_analysis python=3.10 numpy h5py pandas -y

# Activate the environment
conda activate behavior_analysis

# Install additional dependencies
pip install multiprocessing-logging

# Run the analysis
cd JABSbehvaviorClipper
python3 run_complete_analysis.py --behavior turn_left \
    --annotations-dir ../jabs/annotations \
    --features-dir ../jabs/features \
    --video-dir .. \
    --output-dir BoutResults \
    --workers 7
```

### Option 2: Fix Base Anaconda Environment

**Note**: This requires write access to Anaconda base environment.

```bash
# Update conda first
conda update conda -y

# Reinstall numpy and dependencies
conda install --force-reinstall -y numpy

# Reinstall h5py
conda install --force-reinstall -y h5py

# Reinstall pandas
conda install --force-reinstall -y pandas

# Test
python3 -c "import numpy, h5py, pandas; print('All OK')"
```

### Option 3: Use System Python (if available)

```bash
# Check if system Python works
/usr/bin/python3 -c "import sys; print(sys.version)"

# If it works, install packages with pip
/usr/bin/python3 -m pip install --user numpy h5py pandas

# Update scripts to use system Python
# Or create a wrapper script
```

### Option 4: Use Homebrew Python

```bash
# Install Homebrew Python (if not already installed)
brew install python@3.11

# Use Homebrew Python
/opt/homebrew/bin/python3.11 -m pip install numpy h5py pandas

# Run analysis with Homebrew Python
/opt/homebrew/bin/python3.11 run_complete_analysis.py ...
```

## Quick Test

After fixing, test with:

```bash
python3 -c "import numpy; print('numpy:', numpy.__version__)"
python3 -c "import h5py; print('h5py:', h5py.__version__)"
python3 -c "import pandas; print('pandas:', pandas.__version__)"
```

All three should print versions without segmentation faults.

## Why This Happened

Common causes:
1. **Incomplete installation**: Package installation was interrupted
2. **Binary incompatibility**: Libraries compiled for wrong architecture
3. **Corrupted cache**: Anaconda package cache corruption
4. **System update**: macOS update broke binary compatibility

## Prevention

- Always use conda environments for projects (don't modify base)
- Keep conda updated: `conda update --all`
- Use `conda clean --all` periodically to clear cache


# Environment Setup Guide

This guide explains how to set up isolated Python and R environments for the behavior bout analysis pipeline.

## Why Use Virtual Environments?

Virtual environments provide:
- **Isolation**: Prevents conflicts with system packages
- **Reproducibility**: Ensures consistent package versions
- **Portability**: Easy to recreate on different machines
- **Cleanliness**: Keeps project dependencies separate

## Python Environment Setup

### Option 1: Python venv (Recommended)

**Pros:**
- Built into Python (no extra installation)
- Lightweight and fast
- Works well for most use cases

**Setup:**
```bash
# Create and setup virtual environment
bash analysis_r/setup_venv.sh

# Activate before running analysis
source analysis_r/venv/bin/activate

# Verify installation
python3 -c "import h5py, pandas, numpy; print('All packages OK!')"
```

**Usage:**
```bash
# Activate (do this in each new terminal)
source analysis_r/venv/bin/activate

# Run analysis
python3 analysis_r/extract_bout_features.py --behavior turn_left

# Deactivate when done
deactivate
```

### Option 2: Conda Environment

**Pros:**
- Better for complex dependencies
- Can manage both Python and system libraries
- Good if you already use Anaconda/Miniconda

**Setup:**
```bash
# Create and setup conda environment
bash analysis_r/setup_conda.sh

# Activate before running analysis
conda activate behavior_analysis

# Verify installation
python3 -c "import h5py, pandas, numpy; print('All packages OK!')"
```

**Usage:**
```bash
# Activate (do this in each new terminal)
conda activate behavior_analysis

# Run analysis
python3 analysis_r/extract_bout_features.py --behavior turn_left

# Deactivate when done
conda deactivate
```

### Option 3: Manual Installation

If you prefer to install packages globally (not recommended):

```bash
pip3 install -r analysis_r/requirements.txt
```

## R Environment Setup

R packages are installed globally, but you can use `renv` for project-specific R environments if needed.

### Standard Installation

```bash
# Install all required R packages
Rscript analysis_r/install_packages.R
```

### Using renv (Optional)

For project-specific R package management:

```bash
# Initialize renv
Rscript -e "renv::init()"

# Install packages
Rscript analysis_r/install_packages.R

# Restore packages (on other machines)
Rscript -e "renv::restore()"
```

## Automated Environment Activation

The `run_full_pipeline.sh` script automatically detects and activates virtual environments:

```bash
# The script will automatically use:
# 1. analysis_r/venv (if exists)
# 2. conda behavior_analysis (if exists)
# 3. System Python (fallback)

bash analysis_r/run_full_pipeline.sh --behavior turn_left
```

## Troubleshooting

### Virtual Environment Not Found

If you get "command not found" errors:
1. Make sure you've run the setup script: `bash analysis_r/setup_venv.sh`
2. Activate the environment: `source analysis_r/venv/bin/activate`
3. Verify: `which python3` should point to the venv

### Package Import Errors

If packages can't be imported:
1. Activate the environment
2. Reinstall: `pip install -r analysis_r/requirements.txt`
3. Check: `pip list` should show h5py, pandas, numpy

### Conda Issues

If conda commands don't work:
1. Initialize conda: `conda init bash` (then restart terminal)
2. Or use: `source $(conda info --base)/etc/profile.d/conda.sh`

## Best Practices

1. **Always activate** the virtual environment before running analysis
2. **Commit** `requirements.txt` to version control
3. **Don't commit** the `venv/` directory (it's in `.gitignore`)
4. **Document** any additional dependencies you add
5. **Test** the setup on a clean system to ensure reproducibility

## Quick Reference

```bash
# Setup (one time)
bash analysis_r/setup_venv.sh

# Activate (each session)
source analysis_r/venv/bin/activate

# Run analysis
python3 analysis_r/extract_bout_features.py --behavior turn_left

# Deactivate (when done)
deactivate
```


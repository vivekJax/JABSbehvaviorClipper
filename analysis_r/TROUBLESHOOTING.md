# Troubleshooting Guide

## Python Environment Issues

### Segmentation Fault with h5py

If you encounter a segmentation fault when running the Python feature extraction script:

**Symptoms:**
```
Segmentation fault: 11 python3 analysis_r/extract_bout_features.py
```

**Solutions:**

1. **Reinstall h5py:**
   ```bash
   pip3 uninstall h5py
   pip3 install h5py
   ```

2. **Use conda environment:**
   ```bash
   conda create -n behavior_analysis python=3.10
   conda activate behavior_analysis
   conda install h5py pandas numpy
   ```

3. **Check Python version compatibility:**
   ```bash
   python3 --version  # Should be 3.8+
   ```

4. **Try different Python interpreter:**
   ```bash
   which python3
   /usr/bin/python3 analysis_r/extract_bout_features.py --behavior turn_left
   ```

## Running the Analysis

### Step 1: Feature Extraction (Python)

Once Python environment is fixed:

```bash
python3 analysis_r/extract_bout_features.py \
  --behavior turn_left \
  --output bout_features.csv \
  --verbose
```

**Expected output:**
- `bout_features.csv` file with extracted features

### Step 2: Outlier Detection (R) - FIRST

```bash
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --output-dir results/ \
  --distance-metric mahalanobis \
  --use-pca
```

### Step 3: Filter Outliers (R)

```bash
Rscript analysis_r/filter_outliers.R \
  --features bout_features.csv \
  --explanations results/outlier_explanations.csv \
  --output bout_features_filtered.csv \
  --method consensus
```

### Step 4: Clustering on Filtered Data (R)

```bash
Rscript analysis_r/cluster_bouts.R \
  --input bout_features_filtered.csv \
  --output-dir results/ \
  --method all
```

### Step 5: Video Generation (R)

```bash
Rscript analysis_r/generate_outlier_videos.R \
  --outliers results/outlier_detection/outliers_mahalanobis.csv \
  --output results/outlier_detection/outliers_mahalanobis.mp4
```

## Alternative: Manual Feature Extraction

If Python continues to have issues, you can manually extract features using:

1. Python REPL:
   ```python
   import h5py
   import pandas as pd
   import numpy as np
   # ... manual extraction code
   ```

2. Or use existing feature files if available

## R Package Issues

If R packages are missing:

```bash
Rscript analysis_r/install_packages.R
```

## File Path Issues

If features are not found:

1. Check that `jabs/features` directory exists
2. Verify structure: `jabs/features/{video_basename}/{animal_id}/features.h5`
3. Use `--features-dir` to specify custom path:
   ```bash
   python3 analysis_r/extract_bout_features.py \
     --features-dir /path/to/features \
     --behavior turn_left
   ```


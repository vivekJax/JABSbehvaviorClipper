# Behavior Bout Clustering and Outlier Analysis

A comprehensive R-based analysis pipeline for clustering behavior bouts to identify sub-behaviors and detecting outliers using multiple statistical methods.

## Overview

This analysis pipeline processes behavior bouts to:

1. **Extract and aggregate features** from HDF5 files
2. **Detect outliers** using three distance-based methods (Mahalanobis, LOF, Isolation Forest)
3. **Filter outliers** from the dataset to focus on typical behavior
4. **Cluster remaining bouts** to identify sub-behaviors using multiple methods (K-means, hierarchical, DBSCAN)
5. **Generate video snippets** for each behavior subcluster (organized by method)
6. **Generate video snippets** of outlier bouts (optional)
7. **Create visualizations** for clustering and outlier analysis
8. **Create interpretable reports** explaining which bouts are outliers and why

## Quick Reference: Pipeline Summary

**Input**: 
- Behavior annotations: `jabs/annotations/*.json`
- Per-frame features: `jabs/features/{video}/{animal}/features.h5`

**Pipeline Steps**:
1. **Feature Extraction** (Python): Extract per-frame features → aggregate to bout-level → `bout_features.csv`
2. **Outlier Detection** (R): Identify atypical bouts → `outlier_explanations.csv`
3. **Filter Outliers** (R): Remove outliers → `bout_features_filtered.csv`
4. **Clustering** (R): Group similar bouts → `clustering/{method}/cluster_assignments_*.csv` (3 methods)
5. **Video Generation** (R→Python): Create cluster videos → `results/clustering/{method}/videos/`
6. **Visualizations** (R): Generate plots → `results/clustering/{method}/*.png`, `results/outlier_detection/*.png`

**Output**:
- **Data**: Feature tables, cluster assignments, outlier explanations
- **Videos**: Cluster videos (organized by method), outlier videos
- **Plots**: 9 clustering plots, 10 outlier detection plots

**Key Design Decision**: Outliers are detected **first**, then filtered out, then clustering is performed on the filtered data. This ensures clusters represent typical behavior patterns.

## Installation

### Prerequisites

- **Python 3.8+** (for feature extraction)
- **R** (version 4.0 or higher) (for clustering and outlier detection)
- **ffmpeg** (for video processing)

### Setup Python Virtual Environment

We recommend using a virtual environment to isolate dependencies. Choose one:

#### Option 1: Python venv (Recommended)

```bash
# Create and setup virtual environment
bash analysis_r/setup_venv.sh

# Activate virtual environment (do this before running analysis)
source analysis_r/venv/bin/activate
```

#### Option 2: Conda Environment

```bash
# Create and setup conda environment
bash analysis_r/setup_conda.sh

# Activate conda environment (do this before running analysis)
conda activate behavior_analysis
```

#### Option 3: Manual Installation

If you prefer to install packages globally:

```bash
pip3 install -r analysis_r/requirements.txt
```

**Python Dependencies:**
- `h5py>=3.10.0` (HDF5 file reading)
- `numpy>=1.21.0` (Numerical operations)
- `pandas>=1.3.0` (Data manipulation)

### Install R Packages

Run the installation script to install all required R packages:

```bash
Rscript analysis_r/install_packages.R
```

This will install:
- **CRAN**: `dplyr`, `jsonlite`, `optparse`, `ggplot2`, `factoextra`, `cluster`, `dbscan`, `MASS`, `isotree`, `DT`, `htmlwidgets`

**Note**: HDF5 file reading is done in Python using `h5py`, so `rhdf5` is not needed.

## Quick Start

### Full Pipeline

**Recommended**: Use the master script to run the entire pipeline:

```bash
# Auto-detects workers (CPU cores - 1)
bash analysis_r/run_full_pipeline.sh

# Or specify number of workers
bash analysis_r/run_full_pipeline.sh --workers 8
```

**Manual Step-by-Step**:

```bash
# 1. Extract features (Python)
python3 analysis_r/extract_bout_features.py --behavior turn_left

# 2. Find outliers FIRST (R)
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --output-dir results/ \
  --distance-metric mahalanobis \
  --use-pca

# 3. Filter outliers (R)
Rscript analysis_r/filter_outliers.R \
  --features bout_features.csv \
  --explanations results/outlier_detection/outlier_explanations.csv \
  --output bout_features_filtered.csv \
  --method consensus

# 4. Cluster filtered bouts (R)
Rscript analysis_r/cluster_bouts.R \
  --input bout_features_filtered.csv \
  --output-dir results/ \
  --method all

# 5. Generate cluster videos for all methods (R)
# K-means
Rscript analysis_r/generate_cluster_videos.R \
  --clusters results/cluster_assignments_kmeans.csv \
  --output-dir results/clustering \
  --behavior turn_left \
  --method kmeans

# Hierarchical
Rscript analysis_r/generate_cluster_videos.R \
  --clusters results/cluster_assignments_hierarchical.csv \
  --output-dir results/clustering \
  --behavior turn_left \
  --method hierarchical

# DBSCAN
Rscript analysis_r/generate_cluster_videos.R \
  --clusters results/cluster_assignments_dbscan.csv \
  --output-dir results/clustering \
  --behavior turn_left \
  --method dbscan

# 6. Generate outlier videos (optional) (R)
Rscript analysis_r/generate_outlier_videos.R \
  --outliers results/outlier_detection/outliers_mahalanobis.csv \
  --output results/outlier_detection/outliers_mahalanobis.mp4

# 7. Generate visualizations (R)
Rscript analysis_r/visualize_clusters.R \
  --input bout_features_filtered.csv \
  --output-dir results/clustering

Rscript analysis_r/visualize_outliers.R \
  --features bout_features.csv \
  --output-dir results/outlier_detection
```

## Detailed Usage

### Step 1: Feature Extraction (Python)

Extract per-frame features from HDF5 files and aggregate to bout-level statistics using Python (h5py).

**Important**: Activate your virtual environment first:
```bash
# For venv:
source analysis_r/venv/bin/activate

# OR for conda:
conda activate behavior_analysis
```

Then run:
```bash
python3 analysis_r/extract_bout_features.py \
  --behavior turn_left \
  --annotations-dir jabs/annotations \
  --features-dir jabs/features \
  --output bout_features.csv \
  --verbose
```

**Output**: `bout_features.csv` with columns:
- Metadata: `bout_id`, `video_name`, `animal_id`, `start_frame`, `end_frame`, `behavior`
- Features: Aggregated statistics (mean, std, min, max, median, IQR, first, last, duration) for each per-frame feature

### Step 2: Outlier Detection (FIRST)

Identify behavior bouts that are unusual or atypical using three methods. **This is done BEFORE clustering** to remove outliers and focus clustering on typical behavior.

```bash
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --output-dir results/ \
  --distance-metric mahalanobis \
  --use-pca \
  --pca-variance 0.95 \
  --percentile 0.95 \
  --workers 8
```

**Parallel Processing**:
- `--workers`: Number of parallel workers (default: CPU cores - 1)
- Outlier explanation generation is parallelized across workers
- Isolation Forest uses parallel threads internally
- Significantly speeds up processing for large datasets

**Three Outlier Detection Methods**:

1. **Mahalanobis Distance** (Required)
   - Accounts for feature correlations and covariance structure
   - Statistically principled for multivariate normal data
   - Formula: d² = (x - μ)ᵀ Σ⁻¹ (x - μ)

2. **Local Outlier Factor (LOF)**
   - Detects outliers based on local density
   - Identifies points isolated from their neighbors
   - Works well with non-uniform cluster densities

3. **Isolation Forest**
   - Tree-based method that isolates outliers by random feature splits
   - Fast and handles high-dimensional data
   - Non-parametric, handles complex distributions

**How Methods Are Prioritized**:

The outlier detection uses a **consensus-based approach** with the following prioritization:

1. **All Three Methods Always Run**: All three methods (Mahalanobis, LOF, Isolation Forest) are always executed independently. Each method identifies the top N outliers (default: top 5% or top 20, whichever is specified).

2. **Consensus Outliers (Highest Priority)**: 
   - A bout is flagged as a **consensus outlier** if it's identified by **2 or more methods**
   - Consensus outliers are considered the most reliable because multiple independent methods agree
   - These are saved to `outlier_detection/outliers_consensus.csv`
   - **Recommended**: Use consensus outliers for filtering (default: `--method consensus`)

3. **Method-Specific Outliers**:
   - Each method also produces its own outlier list:
     - `outlier_detection/outliers_mahalanobis.csv`: Top outliers by Mahalanobis distance
     - `outlier_detection/outliers_lof.csv`: Top outliers by LOF score
     - `outlier_detection/outliers_isolation_forest.csv`: Top outliers by Isolation Forest score
   - These can be used if you want to focus on a specific type of outlier

4. **Explanation Priority** (for feature contributions):
   - When generating explanations for why a bout is an outlier, the system prioritizes:
     1. **Mahalanobis** (if flagged by Mahalanobis)
     2. **LOF** (if flagged by LOF but not Mahalanobis)
     3. **Isolation Forest** (if flagged by Isolation Forest but not the others)
   - This ensures explanations use the most statistically principled method when available
   - If a bout is flagged by multiple methods, Mahalanobis explanation is used (most interpretable)

**Note**: The `--distance-metric` parameter doesn't change which outliers are selected (all three methods always run). It's currently not used for prioritization, but all three methods are treated equally for outlier selection.

**Output** (all saved to `results/outlier_detection/`):
- `outlier_explanations.csv`: **Comprehensive table explaining which bouts are outliers and why**
- `outlier_feature_contributions.csv`: **Feature-level contributions for each outlier**
- `outliers_mahalanobis.csv`: List of Mahalanobis outliers
- `outliers_lof.csv`: List of LOF outliers
- `outliers_isolation_forest.csv`: List of Isolation Forest outliers
- `outliers_consensus.csv`: Consensus outliers (flagged by multiple methods)

### Step 3: Filter Outliers

Remove outlier bouts from the dataset to focus clustering on typical behavior patterns.

```bash
Rscript analysis_r/filter_outliers.R \
  --features bout_features.csv \
  --explanations results/outlier_detection/outlier_explanations.csv \
  --output bout_features_filtered.csv \
  --method consensus
```

**Filtering Methods**:
- `consensus`: Remove bouts flagged by multiple methods (recommended)
- `mahalanobis`: Remove only Mahalanobis outliers
- `lof`: Remove only LOF outliers
- `isolation`: Remove only Isolation Forest outliers
- `all`: Remove if flagged by any method

**Output**: `bout_features_filtered.csv` with outliers removed

### Step 4: Clustering on Filtered Data

Cluster the remaining (non-outlier) bouts to identify sub-behaviors using K-means, hierarchical, and DBSCAN methods.

```bash
Rscript analysis_r/cluster_bouts.R \
  --input bout_features_filtered.csv \
  --output-dir results/ \
  --method all \
  --pca-variance 0.95 \
  --workers 8
```

**Parallel Processing**:
- `--workers`: Number of parallel workers (default: CPU cores - 1)
- K-means optimal k search is parallelized (testing different k values in parallel)
- K-means nstart is increased based on number of workers for better initialization
- Significantly speeds up clustering for large datasets

**Why PCA?**
- With ~400+ bouts and potentially hundreds of features, we have p >> n problem
- High-dimensional spaces suffer from curse of dimensionality
- Many features are correlated (e.g., angle and angle_cosine, angle_sine)
- PCA reduces to ~10-50 dimensions while preserving 95% variance

**Output** (all in `results/clustering/`):
- `kmeans/cluster_assignments_kmeans.csv`: K-means cluster assignments (includes metadata: video_name, animal_id, start_frame, end_frame)
- `hierarchical/cluster_assignments_hierarchical.csv`: Hierarchical cluster assignments
- `dbscan/cluster_assignments_dbscan.csv`: DBSCAN cluster assignments
- `pca_results.RData`: PCA transformation (shared across all methods, for later use)

### Step 5: Generate Cluster Videos

Create video montages for each behavior subcluster, organized by clustering method.

**Generate videos for all three methods:**

```bash
# K-means cluster videos
Rscript analysis_r/generate_cluster_videos.R \
  --clusters results/clustering/kmeans/cluster_assignments_kmeans.csv \
  --output-dir results/clustering \
  --behavior turn_left \
  --method kmeans

# Hierarchical cluster videos
Rscript analysis_r/generate_cluster_videos.R \
  --clusters results/clustering/hierarchical/cluster_assignments_hierarchical.csv \
  --output-dir results/clustering \
  --behavior turn_left \
  --method hierarchical

# DBSCAN cluster videos
Rscript analysis_r/generate_cluster_videos.R \
  --clusters results/clustering/dbscan/cluster_assignments_dbscan.csv \
  --output-dir results/clustering \
  --behavior turn_left \
  --method dbscan
```

**Output Organization**: 
- `results/clustering/kmeans/videos/cluster_*.mp4`: K-means cluster videos
- `results/clustering/hierarchical/videos/cluster_*.mp4`: Hierarchical cluster videos
- `results/clustering/dbscan/videos/cluster_*.mp4`: DBSCAN cluster videos
- Each video contains all bouts assigned to that cluster
- Videos use the same standards as `generate_bouts_video.py` (bounding boxes, text overlays)

### Step 6: Generate Outlier Videos (Optional)

Create video montages of outlier bouts using the existing video clipper. This is optional and can be done after clustering.

```bash
Rscript analysis_r/generate_outlier_videos.R \
  --outliers results/outlier_detection/outliers_mahalanobis.csv \
  --output results/outlier_detection/outliers_mahalanobis.mp4 \
  --behavior turn_left \
  --video-dir .
```

**Output**: `results/outlier_detection/outliers_{method}.mp4` - Video montage of outlier bouts

### Step 7: Generate Visualizations

Create comprehensive plots for clustering and outlier analysis.

```bash
# Clustering visualizations
Rscript analysis_r/visualize_clusters.R \
  --input bout_features_filtered.csv \
  --output-dir results/clustering

# Outlier detection visualizations
Rscript analysis_r/visualize_outliers.R \
  --features bout_features.csv \
  --output-dir results/outlier_detection
```

**Output**:
- `results/clustering/*.png`: 3 shared plots (PCA scree plot, PCA biplot, clustering comparison)
- `results/clustering/kmeans/*.png`: 2 K-means plots (clusters in PCA space, cluster sizes)
- `results/clustering/hierarchical/*.png`: 2 hierarchical plots (dendrogram, clusters in PCA space)
- `results/clustering/dbscan/*.png`: 2 DBSCAN plots (clusters in PCA space, cluster sizes)
- `results/outlier_detection/*.png`: 10 plots (score distributions, rankings, feature contributions, etc.)

## Pipeline Workflow: Detailed Step-by-Step Explanation

This section explains exactly how the analysis pipeline works, what each step does, and why it's performed in this order.

### Overview: Data Flow Through the Pipeline

```
Raw Data (HDF5 + JSON)
    ↓
[Step 1] Feature Extraction (Python)
    → bout_features.csv (406 bouts × 459 features)
    ↓
[Step 2] Outlier Detection (R)
    → outlier_detection/outlier_explanations.csv, outlier_detection/outliers_*.csv
    ↓
[Step 3] Filter Outliers (R)
    → bout_features_filtered.csv (384 bouts, outliers removed)
    ↓
[Step 4] Clustering (R)
    → clustering/{method}/cluster_assignments_*.csv (3 methods: kmeans, hierarchical, dbscan)
    ↓
[Step 5] Video Generation (R → Python)
    → Cluster videos organized by method
    → Outlier videos
    ↓
[Step 6] Visualizations (R)
    → 9 clustering plots + 10 outlier plots
```

---

### Step 1: Feature Extraction (Python)

**What it does:**
- Reads behavior annotations from JSON files (`jabs/annotations/*.json`)
- For each bout (behavior instance), locates the corresponding HDF5 feature file
- Extracts per-frame features from HDF5 files using `h5py`
- Aggregates per-frame features into bout-level statistics

**How it works:**

1. **Load Annotations**: Reads all JSON files in `jabs/annotations/` directory
   - Each JSON file contains labels for one video
   - Extracts bouts where `present = True` for the specified behavior

2. **Match Bouts to Features**: For each bout, constructs the feature file path:
   ```
   jabs/features/{video_basename}/{animal_id}/features.h5
   ```
   - Example: `jabs/features/video1/animal_1/features.h5`

3. **Extract Per-Frame Features**: Reads HDF5 file and extracts features for the bout's frame range
   - Opens HDF5 file using `h5py.File()`
   - Reads feature arrays for frames `[start_frame, end_frame]`
   - Handles missing files gracefully (logs warning, skips bout)

4. **Aggregate Features**: Computes statistics for each feature across the bout:
   - **Mean**: Average value during the bout
   - **Std**: Standard deviation (variability)
   - **Min/Max**: Extreme values
   - **Median**: Middle value (robust to outliers)
   - **IQR**: Interquartile range (spread)
   - **First/Last**: Initial and final values
   - **Duration**: Number of frames (bout length)

5. **Combine with Metadata**: Creates a CSV row with:
   - Bout identification: `bout_id`, `video_name`, `animal_id`, `start_frame`, `end_frame`
   - Behavior label: `behavior`
   - Aggregated features: 459 feature columns

**Why Python for this step:**
- `h5py` is the most reliable library for reading HDF5 files
- Better error handling and file I/O than R's `rhdf5`
- Faster for large-scale feature extraction

**Output:**
- `bout_features.csv`: One row per bout, columns = metadata + aggregated features
- Example: 406 bouts × 459 features = 406 rows × 465 columns (6 metadata + 459 features)

**Key Technical Details:**
- Handles missing feature files (skips bout, logs warning)
- Validates frame ranges (ensures `start_frame < end_frame`)
- Cleans feature names (removes special characters)
- Handles NaN values in features (replaces with 0)

---

### Step 2: Outlier Detection (R) - RUN FIRST

**What it does:**
- Identifies behavior bouts that are unusual or atypical
- Uses three independent methods to find outliers
- Generates comprehensive explanations for why each bout is an outlier

**Why this step is FIRST:**
- Outliers can distort cluster centers and boundaries
- Removing outliers before clustering focuses on typical behavior patterns
- Clustering on filtered data produces more interpretable sub-behaviors
- Outliers are analyzed separately to understand atypical patterns

**How it works:**

#### 2.1 Data Preprocessing

1. **Load Features**: Reads `bout_features.csv`
2. **Remove Metadata Columns**: Separates features from metadata (`bout_id`, `video_name`, etc.)
3. **Handle Missing Values**: 
   - Replaces `NA` with column mean (mean imputation)
   - Ensures all values are numeric
4. **Remove Constant Features**: Drops features with zero variance (no information)
5. **Standard Scaling**: Z-score normalization: `(x - mean) / std`
   - Centers data at 0, scales to unit variance
   - Makes features comparable across different units

#### 2.2 Dimensionality Reduction (Optional, Recommended)

**Why PCA?**
- **Curse of Dimensionality**: With 459 features and 406 bouts, we have p > n
- **Feature Correlation**: Many features are correlated (e.g., angle, angle_cosine, angle_sine)
- **Covariance Estimation**: Can't reliably estimate 459×459 covariance matrix with 406 samples
- **Distance Reliability**: In high dimensions, all points become equidistant

**How PCA works:**
1. Computes principal components that capture maximum variance
2. Retains components explaining 95% of variance (typically ~50-60 components)
3. Projects data onto reduced space: `X_reduced = X × PC_matrix`
4. **Important**: Distances calculated in PCA space, but explanations use original features

#### 2.3 Outlier Detection Methods

**Method 1: Mahalanobis Distance** (Required)

**What it measures:**
- Distance from a point to the distribution center, accounting for covariance
- Formula: `d² = (x - μ)ᵀ Σ⁻¹ (x - μ)`
- Unlike Euclidean distance, accounts for feature correlations

**How it works:**
1. Computes mean vector `μ` and covariance matrix `Σ` from all bouts
2. For each bout, calculates Mahalanobis distance to the mean
3. If covariance matrix is singular (not invertible):
   - Uses regularization: `Σ_reg = Σ + λI` (adds small value to diagonal)
   - Ensures matrix is invertible
4. Ranks bouts by distance (higher = more unusual)
5. Selects top N or top percentile as outliers

**Why it's effective:**
- Statistically principled for multivariate normal data
- Accounts for feature correlations (e.g., if angle is high, angle_cosine should also be high)
- Identifies points that deviate from the overall distribution pattern

**Method 2: Local Outlier Factor (LOF)**

**What it measures:**
- Local density-based outlier score
- Compares local density of a point to its neighbors
- LOF > 1 indicates lower density than neighbors (outlier)

**How it works:**
1. For each point, finds k nearest neighbors (k=5 by default)
2. Computes local reachability density (LRD) for each point
3. Compares point's LRD to average LRD of its neighbors
4. LOF score = average(neighbors' LRD) / point's LRD
5. Ranks by LOF score, selects top outliers

**Why it's effective:**
- Finds outliers based on local context, not global distribution
- Works well with non-uniform cluster densities
- Identifies points isolated from their neighbors

**Method 3: Isolation Forest**

**What it measures:**
- Anomaly score based on how easily a point can be isolated
- Uses random decision trees to isolate outliers

**How it works:**
1. Builds 100 random decision trees (forest)
2. For each tree, randomly selects features and split values
3. Counts average path length to isolate each point
4. Shorter path = easier to isolate = more anomalous
5. Anomaly score = 2^(-average_path_length / normalization_factor)
6. Ranks by score, selects top outliers

**Why it's effective:**
- Non-parametric (no distribution assumptions)
- Fast (O(n log n) complexity)
- Handles complex, non-linear distributions
- Works well in high dimensions

#### 2.4 Method Prioritization and Consensus Outliers

**How Methods Are Prioritized:**

1. **All Three Methods Always Run**: 
   - All three methods (Mahalanobis, LOF, Isolation Forest) are executed independently
   - Each method identifies the top N outliers (default: top 5% or top 20)
   - Methods are treated equally for outlier selection

2. **Consensus Outliers (Highest Priority)**:
   - A bout is flagged as a **consensus outlier** if it's identified by **2 or more methods**
   - Consensus outliers are considered the most reliable because multiple independent methods agree
   - Formula: `consensus_outlier = (is_outlier_mahalanobis + is_outlier_lof + is_outlier_isolation) >= 2`
   - **Recommended**: Use consensus outliers for filtering (default: `--method consensus`)

3. **Method-Specific Outliers**:
   - Each method also produces its own outlier list
   - These can be used if you want to focus on a specific type of outlier
   - Saved to separate CSV files in `outlier_detection/`: `outliers_mahalanobis.csv`, `outliers_lof.csv`, `outliers_isolation_forest.csv`

4. **Explanation Priority** (for feature contributions):
   - When generating explanations for why a bout is an outlier, the system uses a priority order:
     1. **Mahalanobis** (if flagged by Mahalanobis) - most statistically principled
     2. **LOF** (if flagged by LOF but not Mahalanobis) - local density perspective
     3. **Isolation Forest** (if flagged by Isolation Forest but not the others) - non-parametric perspective
   - This ensures explanations use the most interpretable method when available
   - If a bout is flagged by multiple methods, Mahalanobis explanation is used (most interpretable)

**Why Consensus Approach?**
- **Reliability**: If multiple independent methods agree, the outlier is more likely to be a true outlier
- **Reduces False Positives**: Single-method outliers might be false positives
- **Method Complementarity**: Different methods catch different types of outliers:
  - Mahalanobis: Global distribution outliers
  - LOF: Local density outliers
  - Isolation Forest: Non-linear pattern outliers
- **Robustness**: Consensus outliers are less sensitive to method-specific biases

#### 2.5 Outlier Explanations

**What it generates:**
- For each outlier, identifies which features contribute most to outlier status
- Calculates z-scores (standardized deviations) for each feature
- Creates human-readable explanations

**How it works:**
1. For each outlier bout:
   - Computes z-score for each feature: `z = (value - mean) / std`
   - Ranks features by absolute z-score
   - Selects top 5 contributing features
2. Generates explanation text:
   > "Outlier due to unusually high angular_velocity_mean (z-score: 3.24), low centroid_velocity_mag_mean (z-score: -2.87)..."
3. Saves to `outlier_explanations.csv` and `outlier_feature_contributions.csv`

**Output Files** (all saved to `{output-dir}/outlier_detection/`):
- `outlier_explanations.csv`: Summary table with scores, flags, and explanations
- `outlier_feature_contributions.csv`: Detailed feature-level analysis
- `outliers_mahalanobis.csv`: List of Mahalanobis outliers
- `outliers_lof.csv`: List of LOF outliers
- `outliers_isolation_forest.csv`: List of Isolation Forest outliers
- `outliers_consensus.csv`: Consensus outliers (flagged by multiple methods)

---

### Step 3: Filter Outliers (R)

**What it does:**
- Removes outlier bouts from the dataset
- Creates a clean dataset for clustering analysis

**How it works:**

1. **Load Data**: Reads `bout_features.csv` (all bouts) and `outlier_explanations.csv`
2. **Identify Outliers**: Based on filtering method:
   - `consensus`: Remove bouts flagged by 2+ methods (recommended)
   - `mahalanobis`: Remove only Mahalanobis outliers
   - `lof`: Remove only LOF outliers
   - `isolation`: Remove only Isolation Forest outliers
   - `all`: Remove if flagged by any method
3. **Filter Dataset**: Removes outlier rows from feature matrix
4. **Save Filtered Data**: Writes `bout_features_filtered.csv`

**Why filter before clustering:**
- Outliers can pull cluster centers away from typical patterns
- Clustering on filtered data produces more stable, interpretable clusters
- Focuses analysis on typical behavior subtypes
- Outliers can be studied separately

**Output:**
- `bout_features_filtered.csv`: Same structure as `bout_features.csv`, but with outliers removed
- Example: 406 bouts → 384 bouts (22 outliers removed)

---

### Step 4: Clustering Analysis (R)

**What it does:**
- Groups similar behavior bouts into clusters
- Identifies behavior subtypes using three different methods
- Each method provides a different perspective on the data structure

**How it works:**

#### 4.1 Data Preprocessing

1. **Load Filtered Features**: Reads `bout_features_filtered.csv`
2. **Preprocess**: Same steps as outlier detection (handle missing values, remove constants, scale)
3. **PCA Dimensionality Reduction**:
   - Computes principal components
   - Retains components explaining 95% variance (typically ~50-60 components)
   - Projects data onto reduced space
   - Saves PCA transformation for later use (visualizations, etc.)

#### 4.2 K-means Clustering

**What it does:**
- Partitions data into k clusters by minimizing within-cluster variance
- Assumes spherical clusters of similar size

**How it works:**

1. **Find Optimal k**:
   - Tests k values from 2 to 10
   - For each k, runs K-means 10 times (different random starts)
   - Computes silhouette score (measures cluster quality)
   - Selects k with highest silhouette score

2. **Run K-means**:
   - Initializes k cluster centers randomly
   - Iteratively:
     - Assigns each point to nearest center
     - Updates centers to mean of assigned points
   - Stops when assignments don't change

3. **Save Results**: `clustering/kmeans/cluster_assignments_kmeans.csv`
   - Columns: `bout_id`, `cluster`, `video_name`, `animal_id`, `start_frame`, `end_frame`

**Why K-means:**
- Fast and interpretable
- Works well when clusters are roughly spherical
- Provides clear cluster assignments

**Limitations:**
- Assumes clusters are spherical
- Requires specifying k (though we find optimal k)
- Sensitive to initialization

#### 4.3 Hierarchical Clustering

**What it does:**
- Builds a tree (dendrogram) showing relationships between bouts
- Reveals cluster hierarchy and structure

**How it works:**

1. **Compute Distance Matrix**: Calculates pairwise distances between all bouts
   - Uses Euclidean distance in PCA space

2. **Build Dendrogram**:
   - Starts with each bout as its own cluster
   - Iteratively merges closest clusters
   - Uses Ward's linkage (minimizes within-cluster variance)
   - Continues until all bouts in one cluster

3. **Cut Tree**: Selects k clusters by cutting dendrogram at specified height
   - Default: k = 5 (can be customized)
   - Creates cluster assignments

4. **Save Results**: `clustering/hierarchical/cluster_assignments_hierarchical.csv`

**Why Hierarchical:**
- Reveals cluster hierarchy (which clusters are similar)
- No assumptions about cluster shape
- Dendrogram provides visual representation of data structure

**Limitations:**
- Computationally expensive (O(n²) for distance matrix)
- Sensitive to linkage method
- Requires cutting tree to get final clusters

#### 4.4 DBSCAN Clustering

**What it does:**
- Density-based clustering that finds clusters of arbitrary shape
- Automatically identifies noise points (outliers)

**How it works:**

1. **Estimate Parameters**:
   - `eps`: Maximum distance for points to be neighbors
   - `minPts`: Minimum points to form a cluster
   - Uses k-distance graph to estimate `eps` automatically

2. **Cluster Assignment**:
   - Starts with random unvisited point
   - Finds all points within `eps` distance (neighbors)
   - If point has ≥ `minPts` neighbors, forms cluster
   - Expands cluster by adding neighbors of neighbors
   - Points not in any cluster = noise (cluster = -1 or 0)

3. **Save Results**: `clustering/dbscan/cluster_assignments_dbscan.csv`
   - Note: Noise points may be assigned cluster 0 or -1

**Why DBSCAN:**
- Handles irregular cluster shapes
- Automatically identifies noise
- No need to specify number of clusters

**Limitations:**
- Sensitive to `eps` and `minPts` parameters
- Struggles with varying density clusters
- May label many points as noise

#### 4.5 Why Multiple Methods?

Each method provides a different perspective:
- **K-means**: Fast, interpretable, good for spherical clusters
- **Hierarchical**: Reveals structure, no shape assumptions
- **DBSCAN**: Handles irregular shapes, identifies noise

Using multiple methods provides:
- **Robustness**: If methods agree, clusters are more reliable
- **Different Insights**: Each method may reveal different patterns
- **Validation**: Compare results across methods

**Output Files** (all in `{output-dir}/clustering/`):
- `kmeans/cluster_assignments_kmeans.csv`: K-means cluster assignments
- `hierarchical/cluster_assignments_hierarchical.csv`: Hierarchical cluster assignments
- `dbscan/cluster_assignments_dbscan.csv`: DBSCAN cluster assignments
- `pca_results.RData`: PCA transformation (shared, for visualizations)

---

### Step 5: Video Generation (R → Python)

**What it does:**
- Creates video montages for each cluster (organized by method)
- Creates video montages for outlier bouts
- Uses the same video standards as the main video clipper

**How it works:**

#### 5.1 Cluster Video Generation

**For each clustering method (K-means, Hierarchical, DBSCAN):**

1. **Load Cluster Assignments**: Reads `clustering/{method}/cluster_assignments_{method}.csv`

2. **Group by Cluster**: For each cluster:
   - Extracts all bouts assigned to that cluster
   - Groups by video and animal (maintains order: one animal at a time)

3. **Create Temporary Annotation Files**:
   - For each video, creates a JSON annotation file
   - Structure matches original annotation format:
     ```json
     {
       "version": 1,
       "file": "video.mp4",
       "labels": {
         "animal_1": {
           "turn_left": [
             {"start": 100, "end": 150, "present": true},
             ...
           ]
         }
       }
     }
     ```
   - Saves to temporary directory

4. **Call Python Video Clipper**:
   - Executes `generate_bouts_video.py` as subprocess
   - Passes arguments:
     - `--behavior turn_left`
     - `--annotations-dir {temp_dir}`
     - `--output {output_video}`
     - `--workers {n_workers}` (parallel processing)

5. **Organize Output**:
   - Creates method-specific directories:
     - `results/clustering/kmeans/videos/cluster_1.mp4`
     - `results/clustering/hierarchical/videos/cluster_1.mp4`
     - `results/clustering/dbscan/videos/cluster_0.mp4`

**Video Standards:**
- Same bounding boxes (drawn from pose data)
- Same text overlays (video name, mouse ID)
- Same frame extraction logic
- Same output format (MP4, H.264)

#### 5.2 Outlier Video Generation

**Similar process:**
1. Loads outlier list from `outlier_detection/outliers_{method}.csv`
2. Creates temporary annotation files for outlier bouts
3. Calls Python video clipper
4. Generates single montage video: `results/outlier_detection/outliers_{method}.mp4`

**Output Files:**
- `results/clustering/kmeans/videos/cluster_*.mp4`: K-means cluster videos
- `results/clustering/hierarchical/videos/cluster_*.mp4`: Hierarchical cluster videos
- `results/clustering/dbscan/videos/cluster_*.mp4`: DBSCAN cluster videos
- `results/outlier_detection/outliers_{method}.mp4`: Outlier video montage

---

### Step 6: Visualizations (R)

**What it does:**
- Creates comprehensive plots for clustering and outlier analysis
- Helps interpret results and understand data structure

#### 6.1 Clustering Visualizations

**Generates 9 plots (organized by method):**

**Shared plots** (in `results/clustering/`):
1. **PCA Scree Plot**: Shows variance explained by each principal component
2. **PCA Biplot**: Shows first two PCs with feature loadings
3. **Comparison of All Methods**: Side-by-side comparison

**K-means plots** (in `results/clustering/kmeans/`):
4. **K-means Clusters in PCA Space**: 2D projection colored by cluster
5. **K-means Cluster Sizes**: Bar chart of cluster sizes

**Hierarchical plots** (in `results/clustering/hierarchical/`):
6. **Hierarchical Dendrogram**: Tree showing cluster relationships
7. **Hierarchical Clusters in PCA Space**: 2D projection colored by cluster

**DBSCAN plots** (in `results/clustering/dbscan/`):
8. **DBSCAN Clusters in PCA Space**: 2D projection colored by cluster
9. **DBSCAN Cluster Sizes**: Bar chart (including noise points)

**How it works:**
1. Loads filtered features and cluster assignments
2. Loads PCA results (or recomputes if needed)
3. Projects data onto first two principal components
4. Colors points by cluster assignment
5. Generates plots using `ggplot2`

#### 6.2 Outlier Visualizations

**Generates 10 plots:**

1. **Outlier Score Distributions**: Histograms of scores for each method
2. **Top Outliers - Mahalanobis**: Bar chart of top 20 outliers
3. **Top Outliers - LOF**: Bar chart of top 20 outliers
4. **Top Outliers - Isolation Forest**: Bar chart of top 20 outliers
5. **Outliers in PCA Space**: 2D projection with outliers highlighted
6. **Outlier Method Comparison**: Comparison of methods
7. **Outlier Score Correlations**: Correlation matrix between methods
8. **Feature Contributions Heatmap**: Heatmap of top contributing features
9. **Outlier Score Box Plots**: Box plots by method
10. **Summary Statistics**: Summary table visualization

**How it works:**
1. Loads outlier explanations and full feature dataset
2. Computes PCA on full dataset (including outliers)
3. Projects all points onto first two PCs
4. Highlights outliers in different colors
5. Generates plots using `ggplot2` and `pheatmap`

**Output Files:**
- `results/clustering/*.png`: 3 shared plots + method-specific plots (9 total)
- `results/outlier_detection/*.png`: 10 outlier plots

---

## Workflow Rationale

### Why Detect Outliers First?

**Statistical Reasoning**:
- Outliers can distort cluster centers and boundaries
- Removing outliers before clustering focuses on typical behavior patterns
- Clustering on filtered data produces more interpretable sub-behaviors
- Outliers are analyzed separately to understand atypical patterns

**Practical Benefits**:
- Cleaner cluster separation
- More stable cluster assignments
- Better representation of typical behavior subtypes
- Outliers can be studied separately for insights into rare behaviors

## Outlier Explanation Reports

The pipeline generates comprehensive reports explaining which bouts are outliers and why.

### Outlier Explanations Table (`outlier_explanations.csv`)

Contains:
- **Bout identification**: `bout_id`, `video_name`, `animal_id`, `start_frame`, `end_frame`
- **Outlier scores**: Scores from all three methods
- **Outlier flags**: Boolean indicators for each method
- **Consensus flag**: Flagged by multiple methods
- **Top contributing features**: Top 5 features contributing to outlier status
- **Feature deviations**: Z-scores for top contributing features
- **Cluster assignment**: Which cluster (if any) the bout belongs to
- **Explanation**: Human-readable text explaining why it's an outlier

Example explanation:
> "Outlier due to unusually high angular_velocity_mean (z-score: 3.24), low centroid_velocity_mag_mean (z-score: -2.87), high angles_angle_BASE_NECK-CENTER_SPINE-BASE_TAIL_mean (z-score: 2.15)"

### Feature Contributions Table (`outlier_feature_contributions.csv`)

Detailed feature-level analysis:
- For each outlier, lists all features with:
  - Feature value
  - Population mean and standard deviation
  - Z-score (standardized deviation)
  - Contribution rank

### HTML Report (Future Enhancement)

An interactive HTML report with:
- Sortable table of all outliers
- Expandable rows showing feature contributions
- Links to video snippets
- Embedded visualizations

## Understanding the Results

This section helps you interpret the outputs of the analysis pipeline.

### Interpreting Cluster Assignments

**K-means Results** (`clustering/kmeans/cluster_assignments_kmeans.csv`):
- Each bout is assigned to exactly one cluster (1, 2, 3, ...)
- Clusters are numbered arbitrarily (not by size or importance)
- Optimal number of clusters is determined automatically using silhouette score
- Clusters represent behavior subtypes (e.g., fast turns vs. slow turns)

**Hierarchical Results** (`clustering/hierarchical/cluster_assignments_hierarchical.csv`):
- Each bout assigned to one cluster (1, 2, 3, ...)
- Clusters can be nested (see dendrogram visualization)
- Default: 5 clusters (can be customized)
- Reveals relationships between clusters (which are similar)

**DBSCAN Results** (`clustering/dbscan/cluster_assignments_dbscan.csv`):
- Each bout assigned to a cluster (0, 1, 2, ...) or noise (-1 or 0)
- Clusters can have irregular shapes
- Noise points are outliers that don't fit any cluster
- Number of clusters is determined automatically

**Comparing Methods**:
- If methods agree (same bouts in same clusters), clusters are more reliable
- Different methods may reveal different patterns
- Use visualizations to understand cluster structure

### Interpreting Outlier Explanations

**Outlier Explanations Table** (`outlier_explanations.csv`):
- **outlier_score_***: Raw scores from each method (higher = more unusual)
- **is_outlier_***: Boolean flag (TRUE = flagged as outlier)
- **consensus_outlier**: TRUE if flagged by multiple methods (more reliable)
- **top_features**: Top 5 features contributing to outlier status
- **explanation**: Human-readable text explaining why it's an outlier

**Example Explanation**:
> "Outlier due to unusually high angular_velocity_mean (z-score: 3.24), low centroid_velocity_mag_mean (z-score: -2.87)"

This means:
- `angular_velocity_mean` is 3.24 standard deviations above the mean (very high)
- `centroid_velocity_mag_mean` is 2.87 standard deviations below the mean (very low)
- These extreme values make this bout unusual

**Feature Contributions Table** (`outlier_feature_contributions.csv`):
- Detailed breakdown for each outlier
- Shows all features with their z-scores
- Ranked by contribution (highest absolute z-score first)
- Helps understand which features make a bout unusual

### Interpreting Visualizations

**Clustering Plots** (organized by method):
- **PCA Scree Plot**: Shows how many dimensions are needed (look for "elbow")
- **PCA Biplot**: First two PCs with feature loadings (arrows show feature directions)
- **Clusters in PCA Space**: 2D projection colored by cluster (points close together = similar)
- **Dendrogram**: Tree showing cluster relationships (height = distance)
- **Cluster Sizes**: Bar chart showing how many bouts per cluster

**Outlier Plots** (`results/outlier_detection/*.png`):
- **Score Distributions**: Histograms showing score ranges (outliers = high scores)
- **Top Outliers**: Bar charts of most unusual bouts
- **Outliers in PCA Space**: 2D projection with outliers highlighted in red
- **Feature Contributions Heatmap**: Shows which features contribute to outlier status
- **Method Comparison**: Compares agreement between methods

### Using Cluster Videos

**Video Organization**:
- Videos are organized by clustering method:
  - `results/clustering/kmeans/videos/cluster_1.mp4`
  - `results/clustering/hierarchical/videos/cluster_1.mp4`
  - `results/clustering/dbscan/videos/cluster_0.mp4`

**What to Look For**:
- **Within-cluster similarity**: Bouts in same cluster should look similar
- **Between-cluster differences**: Different clusters should show different patterns
- **Method comparison**: Compare videos from different methods to see different perspectives

**Interpreting Cluster Videos**:
- If K-means cluster 1 and Hierarchical cluster 2 have similar videos, methods agree
- If clusters have very different videos, they represent distinct behavior subtypes
- Use videos to validate that clusters make biological sense

### Next Steps After Analysis

1. **Review Cluster Videos**: Watch videos for each cluster to understand behavior subtypes
2. **Compare Methods**: Compare clusters from different methods to see which makes most sense
3. **Examine Outliers**: Review outlier videos to understand atypical behaviors
4. **Feature Analysis**: Use feature contributions to understand what makes clusters/outliers different
5. **Biological Interpretation**: Connect statistical results to biological understanding

## Statistical Rationale

### Why PCA for Dimensionality Reduction?

**Curse of Dimensionality**: In high dimensions, all points become equidistant, making distance-based methods unreliable.

**Feature Correlation**: Many features are derived from the same underlying measurements (e.g., angle, angle_cosine, angle_sine).

**Sample Size**: With 406 bouts, we can't reliably estimate covariance for hundreds of features.

**Solution**: PCA reduces to ~10-50 dimensions while preserving 95% variance, making clustering and outlier detection more reliable.

### Why Multiple Clustering Methods?

- **K-means**: Assumes spherical clusters, fast, interpretable
- **Hierarchical**: Reveals cluster hierarchy, no assumptions about shape
- **DBSCAN**: Handles irregular shapes, identifies noise naturally

Using multiple methods provides robustness and different perspectives on the data structure.

### Why Three Outlier Methods?

- **Mahalanobis**: Statistically rigorous, accounts for correlations
- **LOF**: Local perspective, finds density-based outliers
- **Isolation Forest**: Non-parametric, handles complex distributions

**Consensus Approach**: Bouts flagged by multiple methods are more reliable outliers.

## File Structure

```
analysis_r/
├── extract_bout_features.py     # Feature extraction from HDF5 (Python)
├── find_outliers.R              # Outlier detection (R) - RUN FIRST
├── filter_outliers.R            # Filter outliers from dataset (R)
├── cluster_bouts.R              # Clustering analysis on filtered data (R)
├── generate_cluster_videos.R    # Video generation for each cluster (R)
├── generate_outlier_videos.R    # Video generation for outliers (R)
├── visualize_clusters.R         # Clustering visualizations (R)
├── visualize_outliers.R         # Outlier detection visualizations (R)
├── run_full_pipeline.sh         # Master script to run entire pipeline
├── install_packages.R           # R package installation script
├── utils/
│   ├── data_preprocessing.R     # Data cleaning and scaling (R)
│   └── visualization.R          # Plotting functions (R)
└── README.md                    # This file
```

## Output Files

### Feature Extraction
- `bout_features.csv`: Extracted and aggregated features (all bouts)

### Outlier Detection (all in `results/outlier_detection/`)
- `outlier_explanations.csv`: **Comprehensive table explaining which bouts are outliers and why**
- `outlier_feature_contributions.csv`: **Feature-level contributions for each outlier**
- `outliers_mahalanobis.csv`: Mahalanobis outliers
- `outliers_lof.csv`: LOF outliers
- `outliers_isolation_forest.csv`: Isolation Forest outliers
- `outliers_consensus.csv`: Consensus outliers

### Outlier Filtering
- `bout_features_filtered.csv`: Features with outliers removed (for clustering)

### Clustering (all in `results/clustering/`)
- `kmeans/cluster_assignments_kmeans.csv`: K-means cluster assignments (includes metadata)
- `hierarchical/cluster_assignments_hierarchical.csv`: Hierarchical cluster assignments
- `dbscan/cluster_assignments_dbscan.csv`: DBSCAN cluster assignments
- `pca_results.RData`: PCA transformation (shared)

### Video Generation
- `results/clustering/kmeans/videos/cluster_*.mp4`: K-means cluster videos
- `results/clustering/hierarchical/videos/cluster_*.mp4`: Hierarchical cluster videos
- `results/clustering/dbscan/videos/cluster_*.mp4`: DBSCAN cluster videos
- `results/outlier_detection/outliers_{method}.mp4`: Video montage of outlier bouts (optional)

### Visualizations
- `results/clustering/*.png`: 3 shared plots (PCA, comparison)
- `results/clustering/kmeans/*.png`: 2 K-means plots
- `results/clustering/hierarchical/*.png`: 2 hierarchical plots
- `results/clustering/dbscan/*.png`: 2 DBSCAN plots
- `results/outlier_detection/*.png`: 10 outlier detection visualization plots

## Troubleshooting

### Feature Extraction Issues

**Problem**: "No features extracted"

**Solutions**:
1. Check that `--features-dir` points to the correct location
2. Verify feature file structure: `features_dir/{video_basename}/{animal_id}/features.h5`
3. Run with `--verbose` to see detailed path information
4. Check that video names match between annotations and feature files

### Clustering Issues

**Problem**: "more cluster centers than distinct data points"

**Solutions**:
1. Reduce number of clusters: `--n-clusters 3`
2. Check that features have sufficient variance
3. Verify data preprocessing completed successfully

### Outlier Detection Issues

**Problem**: "Covariance matrix is singular"

**Solutions**:
1. Use `--use-pca` to reduce dimensionality first
2. Check for constant features (should be removed in preprocessing)
3. Increase sample size if possible

## Advanced Options

### Custom PCA Variance Threshold

```bash
Rscript analysis_r/cluster_bouts.R \
  --input bout_features.csv \
  --pca-variance 0.90  # Retain 90% variance instead of 95%
```

### Custom Number of Clusters

```bash
Rscript analysis_r/cluster_bouts.R \
  --input bout_features.csv \
  --method kmeans \
  --n-clusters 5
```

### Custom Outlier Threshold

```bash
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --top-n 20  # Select top 20 outliers instead of percentile
```

### Parallel Processing (Multicore)

Both outlier detection and clustering support parallel processing to speed up computation:

**Outlier Detection**:
- Parallelizes outlier explanation generation (the most time-consuming step)
- Isolation Forest uses parallel threads internally
- Use `--workers N` to specify number of cores (default: CPU cores - 1)

**Clustering**:
- Parallelizes K-means optimal k search (testing different k values simultaneously)
- Increases K-means nstart based on number of workers for better initialization
- Use `--workers N` to specify number of cores (default: CPU cores - 1)

**Example**:
```bash
# Use 8 workers for faster processing
Rscript analysis_r/find_outliers.R \
  --features bout_features.csv \
  --workers 8

Rscript analysis_r/cluster_bouts.R \
  --input bout_features_filtered.csv \
  --workers 8
```

**Performance Benefits**:
- Outlier detection: 2-4x faster with 4-8 workers
- Clustering: 2-3x faster for optimal k search
- Scales well up to number of CPU cores

## Integration with Video Clipper

The outlier video generation uses the existing `generate_bouts_video.py` script to ensure consistency:
- Same video standards (bounding boxes, text overlays)
- Same frame extraction logic
- Same output format

This ensures that outlier videos are directly comparable to the main behavior video montages.

## Citation

If you use this analysis pipeline, please cite the statistical methods:
- Mahalanobis distance: Mahalanobis, P. C. (1936). "On the generalised distance in statistics"
- Local Outlier Factor: Breunig et al. (2000). "LOF: identifying density-based local outliers"
- Isolation Forest: Liu et al. (2008). "Isolation Forest"

## License

This analysis pipeline is part of the JABS BehaviorClipper project.


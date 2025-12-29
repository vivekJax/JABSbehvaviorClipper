#!/bin/bash
# Master script to run the full analysis pipeline
#
# This script runs:
# 1. Python feature extraction
# 2. R clustering analysis
# 3. R outlier detection
# 4. R video generation
#
# Usage:
#   bash BoutAnalysisScripts/setup/run_full_pipeline.sh --behavior turn_left
#
# The script will automatically use a virtual environment if available:
# - Python venv: BoutAnalysisScripts/venv (if using)
# - Conda: behavior_analysis

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Try to activate virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Activating Python virtual environment..."
    source "$VENV_DIR/bin/activate"
elif command -v conda &> /dev/null && conda env list | grep -q "^behavior_analysis "; then
    echo "Activating conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate behavior_analysis
fi

# Default values
BEHAVIOR="turn_left"
ANNOTATIONS_DIR="jabs/annotations"
FEATURES_DIR="jabs/features"
OUTPUT_DIR="results"
USE_PCA="--use-pca"
DISTANCE_METRIC="mahalanobis"
WORKERS=""  # Will be auto-detected if not specified

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --behavior)
            BEHAVIOR="$2"
            shift 2
            ;;
        --annotations-dir)
            ANNOTATIONS_DIR="$2"
            shift 2
            ;;
        --features-dir)
            FEATURES_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-pca)
            USE_PCA=""
            shift
            ;;
        --distance-metric)
            DISTANCE_METRIC="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect workers if not specified (CPU cores - 1)
if [ -z "$WORKERS" ]; then
    if command -v python3 &> /dev/null; then
        WORKERS=$(python3 -c "import multiprocessing; print(max(1, multiprocessing.cpu_count() - 1))" 2>/dev/null || echo "1")
    else
        WORKERS=1
    fi
fi

echo "============================================================"
echo "Behavior Bout Clustering and Outlier Analysis Pipeline"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Behavior: $BEHAVIOR"
echo "  Annotations: $ANNOTATIONS_DIR"
echo "  Features: $FEATURES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Workers: $WORKERS (parallel processing)"
echo ""

# Step 1: Feature Extraction (Python)
echo "============================================================"
echo "Step 1: Feature Extraction (Python)"
echo "============================================================"
# Ensure results directory exists
mkdir -p "$OUTPUT_DIR"

# Get project root (two levels up from setup/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$PROJECT_ROOT/scripts/extract_bout_features.py" \
    --behavior "$BEHAVIOR" \
    --annotations-dir "$ANNOTATIONS_DIR" \
    --features-dir "$FEATURES_DIR" \
    --output "$OUTPUT_DIR/bout_features.csv" \
    --verbose

if [ ! -f "$OUTPUT_DIR/bout_features.csv" ]; then
    echo "Error: Feature extraction failed. bout_features.csv not created."
    exit 1
fi

echo ""
echo "✓ Feature extraction complete"
echo ""

# Step 2: Outlier Detection (R) - FIRST
echo "============================================================"
echo "Step 2: Outlier Detection (R)"
echo "============================================================"
Rscript scripts/core/find_outliers.R \
    --features "$OUTPUT_DIR/bout_features.csv" \
    --output-dir "$OUTPUT_DIR" \
    --distance-metric "$DISTANCE_METRIC" \
    --workers "$WORKERS" \
    $USE_PCA

if [ ! -f "$OUTPUT_DIR/outlier_detection/outlier_explanations.csv" ]; then
    echo "Error: Outlier detection failed. Outlier explanations not created."
    exit 1
fi

echo ""
echo "✓ Outlier detection complete"
echo ""

# Step 3: Filter Outliers (R)
echo "============================================================"
echo "Step 3: Filter Outliers (R)"
echo "============================================================"
Rscript scripts/core/filter_outliers.R \
    --features "$OUTPUT_DIR/bout_features.csv" \
    --explanations "$OUTPUT_DIR/outlier_detection/outlier_explanations.csv" \
    --output "$OUTPUT_DIR/bout_features_filtered.csv" \
    --method consensus

if [ ! -f "$OUTPUT_DIR/bout_features_filtered.csv" ]; then
    echo "Error: Outlier filtering failed. Filtered features not created."
    exit 1
fi

echo ""
echo "✓ Outlier filtering complete"
echo ""

# Step 4: Clustering on Filtered Data (R)
echo "============================================================"
echo "Step 4: Clustering Analysis on Filtered Data (R)"
echo "============================================================"
Rscript scripts/core/cluster_bouts.R \
    --input "$OUTPUT_DIR/bout_features_filtered.csv" \
    --output-dir "$OUTPUT_DIR" \
    --method all \
    --workers "$WORKERS"

if [ ! -f "$OUTPUT_DIR/clustering/kmeans/cluster_assignments_kmeans.csv" ]; then
    echo "Error: Clustering failed. Cluster assignments not created."
    exit 1
fi

echo ""
echo "✓ Clustering complete"
echo ""

# Step 5: Generate Cluster Videos (R) - All Methods
echo "============================================================"
echo "Step 5: Generate Cluster Videos (R) - All Methods"
echo "============================================================"

# Generate videos for K-means
echo "Generating K-means cluster videos..."
Rscript scripts/video/generate_cluster_videos.R \
    --clusters "$OUTPUT_DIR/clustering/kmeans/cluster_assignments_kmeans.csv" \
    --output-dir "$OUTPUT_DIR/clustering" \
    --behavior "$BEHAVIOR" \
    --method kmeans

echo ""

# Generate videos for Hierarchical
echo "Generating Hierarchical cluster videos..."
Rscript scripts/video/generate_cluster_videos.R \
    --clusters "$OUTPUT_DIR/clustering/hierarchical/cluster_assignments_hierarchical.csv" \
    --output-dir "$OUTPUT_DIR/clustering" \
    --behavior "$BEHAVIOR" \
    --method hierarchical

echo ""

# Generate videos for DBSCAN
echo "Generating DBSCAN cluster videos..."
Rscript scripts/video/generate_cluster_videos.R \
    --clusters "$OUTPUT_DIR/clustering/dbscan/cluster_assignments_dbscan.csv" \
    --output-dir "$OUTPUT_DIR/clustering" \
    --behavior "$BEHAVIOR" \
    --method dbscan

echo ""
echo "✓ Cluster video generation complete for all methods"
echo ""

# Step 6: Generate Outlier Videos (R) - Optional
echo "============================================================"
echo "Step 6: Generate Outlier Videos (R) - Optional"
echo "============================================================"
Rscript scripts/video/generate_outlier_videos.R \
    --outliers "$OUTPUT_DIR/outlier_detection/outliers_${DISTANCE_METRIC}.csv" \
    --output "${OUTPUT_DIR}/outlier_detection/outliers_${DISTANCE_METRIC}.mp4" \
    --behavior "$BEHAVIOR"

# Step 7: Generate Visualizations (R)
echo ""
echo "============================================================"
echo "Step 7: Generate Visualizations (R)"
echo "============================================================"

echo "Generating clustering visualizations..."
Rscript scripts/visualization/visualize_clusters.R \
    --input "$OUTPUT_DIR/bout_features_filtered.csv" \
    --output-dir "$OUTPUT_DIR/clustering"

echo ""
echo "Generating outlier detection visualizations..."
Rscript scripts/visualization/visualize_outliers.R \
    --features "$OUTPUT_DIR/bout_features.csv" \
    --output-dir "$OUTPUT_DIR/outlier_detection" \
    --explanations "$OUTPUT_DIR/outlier_detection/outlier_explanations.csv" \
    --contributions "$OUTPUT_DIR/outlier_detection/outlier_feature_contributions.csv"

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/bout_features.csv (all bouts)"
echo "  - $OUTPUT_DIR/bout_features_filtered.csv (outliers removed)"
echo "  - $OUTPUT_DIR/clustering/kmeans/cluster_assignments_kmeans.csv"
echo "  - $OUTPUT_DIR/clustering/hierarchical/cluster_assignments_hierarchical.csv"
echo "  - $OUTPUT_DIR/clustering/dbscan/cluster_assignments_dbscan.csv"
echo "  - $OUTPUT_DIR/outlier_detection/outlier_explanations.csv"
echo "  - $OUTPUT_DIR/outlier_detection/outlier_feature_contributions.csv"
echo "  - $OUTPUT_DIR/outlier_detection/outliers_*.csv"
echo "  - $OUTPUT_DIR/outlier_detection/outliers_${DISTANCE_METRIC}.mp4 (outlier video)"
echo "  - $OUTPUT_DIR/clustering/kmeans/videos/cluster_*.mp4 (K-means cluster videos)"
echo "  - $OUTPUT_DIR/clustering/hierarchical/videos/cluster_*.mp4 (Hierarchical cluster videos)"
echo "  - $OUTPUT_DIR/clustering/dbscan/videos/cluster_*.mp4 (DBSCAN cluster videos)"
echo ""
echo "Visualizations:"
echo "  - $OUTPUT_DIR/clustering/*.png (9 plots)"
echo "  - $OUTPUT_DIR/outlier_detection/*.png (10 plots)"
echo ""


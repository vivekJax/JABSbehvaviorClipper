# Documentation Index

This directory contains comprehensive documentation for the behavior bout analysis pipeline. This guide helps you navigate the documentation based on your needs.

## Quick Navigation

### I want to...

**...run the complete analysis with one command**
→ [USAGE_GUIDE.md](USAGE_GUIDE.md) - Master script usage guide

**...get started quickly**
→ [QUICK_START.md](QUICK_START.md) - Quick reference commands

**...understand the statistical methods**
→ [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md) - Detailed statistical explanations

**...understand the complete pipeline**
→ [ANALYSIS_PIPELINE.md](ANALYSIS_PIPELINE.md) - Step-by-step pipeline overview

**...understand outlier detection**
→ [OUTLIER_EXPLANATION.md](OUTLIER_EXPLANATION.md) - Outlier detection and visualizations

**...understand distance metrics**
→ [DISTANCE_METRICS.md](DISTANCE_METRICS.md) - Comprehensive distance metrics guide

**...use the analysis tools**
→ [README.md](README.md) - Usage guide and examples

---

## Documentation Files

### 1. [USAGE_GUIDE.md](USAGE_GUIDE.md)

**Purpose**: Complete usage guide for the master analysis script

**Contents**:
- One-command workflow for complete analysis
- Master script options and examples
- Distance metrics selection guide
- Output structure
- Troubleshooting and performance tips

**Audience**: Users who want to run the complete analysis pipeline

**Key Sections**:
- Quick start: One command to run everything
- Distance metrics: When to use Mahalanobis, PCA, Euclidean
- Advanced options: Skip steps, customize parameters
- Examples: Common use cases

### 2. [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md)

**Purpose**: Comprehensive statistical and methodological explanations

**Contents**:
- Feature extraction rationale and aggregation strategies
- Data preprocessing (missing values, scaling, feature selection)
- Clustering algorithms (K-means, hierarchical, DBSCAN) with statistical justification
- Outlier detection methods and distance metrics
- Visualization strategies and dimensionality reduction
- Statistical assumptions, limitations, and best practices

**Audience**: Statisticians, data scientists, researchers who want to understand the methodology

**Key Sections**:
- Why each preprocessing step is necessary
- Statistical rationale for algorithm choices
- Assumptions and limitations
- Best practices and validation strategies

---

### 2. [ANALYSIS_PIPELINE.md](ANALYSIS_PIPELINE.md)

**Purpose**: Complete pipeline overview with step-by-step explanations

**Contents**:
- Overview of the entire analysis workflow
- Detailed explanation of each script and what it does
- Why each step is necessary
- How steps integrate together
- Typical workflow examples
- Data flow diagrams

**Audience**: Users who want to understand the complete analysis process

**Key Sections**:
- Feature extraction → Clustering → Visualization → Outlier detection → Video generation
- Input/output for each step
- Dependencies between steps
- Integration and workflow examples

---

### 3. [OUTLIER_EXPLANATION.md](OUTLIER_EXPLANATION.md)

**Purpose**: Detailed explanation of outlier detection and visualizations

**Contents**:
- How outlier detection works
- Distance metrics and aggregate distance calculations
- Outlier selection methods
- Explanation of each visualization type
- How to interpret the results
- Common reasons for outliers

**Audience**: Users who want to understand outlier detection results

**Key Sections**:
- Statistical methods for outlier detection
- Visualization explanations
- Interpretation guidelines
- Example interpretations

---

### 4. [README.md](README.md)

**Purpose**: Usage guide and reference

**Contents**:
- Installation instructions
- Usage examples for each script
- Command-line options
- Output file descriptions
- Troubleshooting

**Audience**: Users who want to run the analysis

**Key Sections**:
- Installation
- Quick workflow examples
- Detailed usage for each script
- Output files reference

---

### 7. [QUICK_START.md](QUICK_START.md)

**Purpose**: Quick reference for common commands

**Contents**:
- Installation commands
- Typical workflow commands
- Common use cases

**Audience**: Users familiar with the pipeline who need quick reference

---

## Documentation Structure

```
analysis_r/
├── README.md                    # Main usage guide
├── QUICK_START.md              # Quick reference
├── USAGE_GUIDE.md              # Master script usage guide
├── STATISTICAL_METHODOLOGY.md  # Statistical explanations
├── ANALYSIS_PIPELINE.md        # Pipeline overview
├── OUTLIER_EXPLANATION.md      # Outlier detection guide
├── DISTANCE_METRICS.md         # Distance metrics guide
└── DOCUMENTATION_INDEX.md      # This file
```

---

## Reading Order Recommendations

### For New Users

1. **README.md** - Get overview and installation
2. **QUICK_START.md** - Run a quick example
3. **ANALYSIS_PIPELINE.md** - Understand the workflow
4. **STATISTICAL_METHODOLOGY.md** - Deep dive into methods (optional)

### For Statisticians/Data Scientists

1. **STATISTICAL_METHODOLOGY.md** - Understand methods and rationale
2. **ANALYSIS_PIPELINE.md** - See how methods are applied
3. **OUTLIER_EXPLANATION.md** - Understand outlier detection
4. **README.md** - Reference for usage

### For Users Analyzing Results

1. **OUTLIER_EXPLANATION.md** - Understand outlier visualizations
2. **ANALYSIS_PIPELINE.md** - Understand what each step produces
3. **README.md** - Reference for output files

---

## Key Concepts Explained

### Feature Extraction
- **Why**: Temporal sequences need aggregation for statistical analysis
- **How**: Multiple statistics (mean, std, min, max, etc.) capture different aspects
- **Details**: [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md#feature-extraction)

### Data Preprocessing
- **Why**: Features have different scales, missing values need handling
- **How**: Scaling, imputation, constant feature removal
- **Details**: [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md#data-preprocessing)

### Clustering
- **Why**: Discover behavioral subtypes
- **How**: K-means, hierarchical, DBSCAN with validation metrics
- **Details**: [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md#clustering-analysis)

### Outlier Detection
- **Why**: Identify unusual behaviors
- **How**: Distance-based methods with aggregate metrics
- **Details**: [OUTLIER_EXPLANATION.md](OUTLIER_EXPLANATION.md) and [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md#outlier-detection)

### Visualization
- **Why**: Validate results, understand patterns, communicate findings
- **How**: PCA, t-SNE, feature distributions, heatmaps
- **Details**: [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md#visualization-strategy)

---

## Statistical Justification Summary

All methodological choices are statistically justified:

1. **Feature Aggregation**: Necessary for fixed-length vectors, preserves information through multiple statistics
2. **Mean Imputation**: Standard practice for MCAR missing data, preserves sample size
3. **Standard Scaling**: Required for distance-based methods, ensures equal feature importance
4. **K-Means**: Fast, interpretable, standard algorithm for spherical clusters
5. **Hierarchical Clustering**: Flexible, complementary to K-means, handles non-spherical clusters
6. **DBSCAN**: Automatic k, noise detection, flexible shapes
7. **Silhouette Score**: Standard internal validation metric
8. **Euclidean Distance**: Standard, interpretable, works well after scaling
9. **Mean Aggregate Distance**: Comprehensive, stable, interpretable
10. **Top 5% Threshold**: Standard outlier detection threshold

See [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md) for detailed explanations.

---

## Questions?

- **Methodology questions**: See [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md)
- **Usage questions**: See [README.md](README.md)
- **Pipeline questions**: See [ANALYSIS_PIPELINE.md](ANALYSIS_PIPELINE.md)
- **Outlier questions**: See [OUTLIER_EXPLANATION.md](OUTLIER_EXPLANATION.md)


#!/usr/bin/env Rscript
# Install required R packages for behavior bout clustering and outlier analysis
#
# This script installs:
# - Bioconductor packages (rhdf5)
# - CRAN packages (dplyr, jsonlite, optparse, ggplot2, etc.)
#
# Usage:
#   Rscript analysis_r/install_packages.R

cat("Installing required R packages for behavior bout analysis...\n")
cat("============================================================\n\n")

# Note: HDF5 file reading is done in Python (h5py), not R
# Install CRAN packages
cat("\n[1/1] Installing CRAN packages...\n")
cran_packages <- c(
  "dplyr",           # Data manipulation
  "jsonlite",        # JSON parsing
  "optparse",        # Command-line arguments
  "ggplot2",         # Plotting
  "gridExtra",       # Additional plotting utilities
  "factoextra",      # PCA and clustering visualization
  "cluster",         # Clustering algorithms
  "dbscan",          # DBSCAN and LOF
  "MASS",            # Mahalanobis distance
  "isotree",         # Isolation Forest
  "pheatmap",        # Heatmap visualization
  "tidyr",           # Data reshaping
  "tibble",          # Data frames
  "stringr",         # String manipulation
  "DT",              # HTML table generation
  "htmlwidgets",     # HTML widgets
  "Rtsne"            # t-SNE for visualization (optional)
)

for (pkg in cran_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cran.r-project.org", dependencies = TRUE)
  } else {
    cat(sprintf("  %s already installed\n", pkg))
  }
}

cat("\n============================================================\n")
cat("Package installation complete!\n")
cat("\nTo verify installation, run:\n")
cat("  Rscript -e 'library(rhdf5); library(dplyr); library(ggplot2)'\n")


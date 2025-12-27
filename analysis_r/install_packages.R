#!/usr/bin/env Rscript
# Installation script for R analysis dependencies
#
# Run this script to install all required packages:
#   Rscript analysis_r/install_packages.R

cat("Installing R packages for behavior bout analysis...\n\n")

# Install BiocManager if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  cat("Installing BiocManager...\n")
  install.packages("BiocManager", repos = "https://cran.r-project.org")
}

# Install Bioconductor packages
cat("Installing Bioconductor packages (rhdf5)...\n")
BiocManager::install("rhdf5", ask = FALSE, update = FALSE)

# Install CRAN packages
cran_packages <- c(
  "getopt",      # Dependency for optparse
  "optparse",    # Command-line argument parsing
  "dplyr",       # Data manipulation
  "jsonlite",    # JSON reading/writing
  "ggplot2",     # Plotting
  "Rtsne",       # t-SNE dimensionality reduction
  "factoextra",  # Cluster analysis
  "pheatmap",    # Heatmaps
  "cluster",      # Clustering algorithms
  "NbClust",     # Optimal cluster number
  "gridExtra",   # Plot arrangement
  "dbscan"       # DBSCAN clustering
)

cat("Installing CRAN packages...\n")
for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cran.r-project.org")
  } else {
    cat(sprintf("  %s already installed\n", pkg))
  }
}

# Verify installations
cat("\nVerifying installations...\n")
required_packages <- c("rhdf5", "optparse", "dplyr", "jsonlite", "ggplot2", 
                       "Rtsne", "factoextra", "pheatmap", "cluster", 
                       "NbClust", "gridExtra", "dbscan")

all_ok <- TRUE
for (pkg in required_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  ✓ %s\n", pkg))
  } else {
    cat(sprintf("  ✗ %s (FAILED)\n", pkg))
    all_ok <- FALSE
  }
}

if (all_ok) {
  cat("\n✓ All packages installed successfully!\n")
} else {
  cat("\n✗ Some packages failed to install. Please check error messages above.\n")
  cat("You may need to run this script with appropriate permissions.\n")
}


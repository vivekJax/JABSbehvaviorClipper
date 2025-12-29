#!/usr/bin/env Rscript
# Run complete behavior bout analysis pipeline.
#
# This script runs the entire analysis pipeline:
# 1. Feature extraction
# 2. Clustering (all methods)
# 3. Visualization
# 4. Cluster video generation
# 5. Outlier detection
#
# Usage:
#   cd JABSbehvaviorClipper
#   Rscript BoutAnalysisScripts/scripts/core/run_full_analysis.R --behavior turn_left

# Add default library paths (user library first)
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-b", "--behavior"), type="character", default="turn_left",
              help="Behavior name to analyze (default: turn_left)"),
  make_option(c("-a", "--annotations-dir"), type="character", default="../jabs/annotations",
              help="Directory containing annotation JSON files (default: ../jabs/annotations)"),
  make_option(c("-f", "--features-dir"), type="character", default=NULL,
              help="Base directory for feature HDF5 files (default: auto-detect from ../jabs/features)"),
  make_option(c("-v", "--video-dir"), type="character", default="..",
              help="Directory containing video files (default: parent directory)"),
  make_option(c("-o", "--output-dir"), type="character", default="BoutResults",
              help="Output directory for all results (default: BoutResults)"),
  make_option(c("--distance-metric"), type="character", default="mahalanobis",
              help="Distance metric for outlier detection: euclidean, mahalanobis, manhattan, cosine (default: mahalanobis)"),
  make_option(c("--use-pca"), action="store_true", default=FALSE,
              help="Use PCA reduction for outlier detection"),
  make_option(c("--pca-variance"), type="numeric", default=0.95,
              help="Proportion of variance to retain in PCA (default: 0.95)"),
  make_option(c("--skip-clustering"), action="store_true", default=FALSE,
              help="Skip clustering analysis (only extract features and find outliers)"),
  make_option(c("--skip-outliers"), action="store_true", default=FALSE,
              help="Skip outlier detection"),
  make_option(c("--skip-videos"), action="store_true", default=FALSE,
              help="Skip video generation"),
  make_option(c("--workers"), type="integer", default=NULL,
              help="Number of parallel workers for video clipping"),
  make_option(c("--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Get script directory to resolve relative paths
script_dir <- dirname(normalizePath(commandArgs()[4]))
base_dir <- dirname(dirname(script_dir))  # Go up from scripts/core to BoutAnalysisScripts

# Auto-detect features directory if not specified
if (is.null(opt$`features-dir`)) {
  # Try common locations relative to project root
  possible_paths <- c(
    file.path(base_dir, "..", "jabs", "features"),
    "../jabs/features",
    "jabs/features",
    "./jabs/features"
  )
  
  for (path in possible_paths) {
    if (dir.exists(path)) {
      opt$`features-dir` <- normalizePath(path)
      cat(sprintf("Auto-detected features directory: %s\n", opt$`features-dir`))
      break
    }
  }
  
  # If still not found, use default and let user know
  if (is.null(opt$`features-dir`)) {
    opt$`features-dir` <- "../jabs/features"  # Default
    cat(sprintf("Using default features directory: %s\n", opt$`features-dir`))
    cat("If this is incorrect, specify --features-dir explicitly\n")
  }
}

# Helper function to run R script with proper path resolution
run_script <- function(script_path, args = "", description = "") {
  if (description != "") {
    cat(sprintf("\n%s\n", description))
    cat(paste(rep("=", 60), collapse=""), "\n")
  }
  
  # Resolve script path relative to script directory
  if (!file.exists(script_path)) {
    # Try relative to script directory
    full_path <- file.path(script_dir, "..", script_path)
    if (file.exists(full_path)) {
      script_path <- normalizePath(full_path)
    } else {
      # Try relative to base directory
      full_path <- file.path(base_dir, script_path)
      if (file.exists(full_path)) {
        script_path <- normalizePath(full_path)
      }
    }
  }
  
  cmd <- sprintf("Rscript %s %s", script_path, args)
  
  if (opt$verbose) {
    cat(sprintf("Running: %s\n", cmd))
  }
  
  result <- system(cmd, intern=FALSE)
  
  if (result != 0) {
    stop(sprintf("Error running %s (exit code: %d)", script_path, result))
  }
  
  return(result)
}

# Main execution
main <- function() {
  cat("\n============================================================\n")
  cat("Behavior Bout Analysis Pipeline\n")
  cat("============================================================\n")
  cat(sprintf("Behavior: %s\n", opt$behavior))
  cat(sprintf("Output directory: %s\n", opt$`output-dir`))
  cat("============================================================\n\n")
  
  # Step 1: Extract features
  if (!opt$`skip-clustering` || !opt$`skip-outliers`) {
    args <- sprintf("--behavior %s --annotations-dir %s --features-dir %s --output %s/%s",
                    opt$behavior, opt$`annotations-dir`, opt$`features-dir`,
                    opt$`output-dir`, "bout_features.csv")
    if (opt$verbose) {
      args <- paste(args, "--verbose")
    }
    
    run_script("scripts/core/extract_bout_features.R", args,
              "Step 1: Extracting bout features from HDF5 files")
  }
  
  features_file <- file.path(opt$`output-dir`, "bout_features.csv")
  
  # Step 2: Clustering
  if (!opt$`skip-clustering`) {
    args <- sprintf("--input %s --method all --output-dir %s/clustering",
                    features_file, opt$`output-dir`)
    if (opt$verbose) {
      args <- paste(args, "--verbose")
    }
    
    run_script("scripts/core/cluster_bouts.R", args,
              "Step 2: Performing clustering analysis")
    
    # Step 3: Visualize clusters
    cluster_files <- c(
      file.path(opt$`output-dir`, "clustering", "cluster_assignments_kmeans.csv"),
      file.path(opt$`output-dir`, "clustering", "cluster_assignments_hierarchical.csv")
    )
    
    for (cluster_file in cluster_files) {
      if (file.exists(cluster_file)) {
        method <- gsub(".*cluster_assignments_(.*)\\.csv", "\\1", cluster_file)
        args <- sprintf("--features %s --clusters %s --output-dir %s/clustering",
                       features_file, cluster_file, opt$`output-dir`)
        if (opt$verbose) {
          args <- paste(args, "--verbose")
        }
        
        run_script("scripts/visualization/visualize_clusters.R", args,
                  sprintf("Step 3: Visualizing %s clusters", method))
        
        # Generate PDF report
        args_pdf <- sprintf("--features %s --clusters %s --output-dir %s/clustering",
                           features_file, cluster_file, opt$`output-dir`)
        run_script("scripts/visualization/visualize_clusters_pdf.R", args_pdf,
                  sprintf("Step 3b: Generating PDF report for %s", method))
      }
    }
    
    # Step 4: Generate cluster videos
    if (!opt$`skip-videos`) {
      for (cluster_file in cluster_files) {
        if (file.exists(cluster_file)) {
          method <- gsub(".*cluster_assignments_(.*)\\.csv", "\\1", cluster_file)
          args <- sprintf("--clusters %s --method %s --output-dir %s/videos --video-dir %s",
                         cluster_file, method, opt$`output-dir`, opt$`video-dir`)
          if (!is.null(opt$workers)) {
            args <- paste(args, sprintf("--workers %d", opt$workers))
          }
          if (opt$verbose) {
            args <- paste(args, "--verbose")
          }
          
          run_script("scripts/video/generate_cluster_videos.R", args,
                    sprintf("Step 4: Generating videos for %s clusters", method))
        }
      }
    }
  }
  
  # Step 5: Outlier detection
  if (!opt$`skip-outliers`) {
    args <- sprintf("--features %s --output-dir %s/outliers --distance-metric %s",
                    features_file, opt$`output-dir`, opt$`distance-metric`)
    if (opt$`use-pca`) {
      args <- paste(args, sprintf("--use-pca --pca-variance %.2f", opt$`pca-variance`))
    }
    if (opt$verbose) {
      args <- paste(args, "--verbose")
    }
    
    run_script("scripts/core/detect_outliers_consensus.R", args,
              "Step 5: Detecting outliers (multi-method consensus)")
  }
  
  cat("\n============================================================\n")
  cat("Analysis Pipeline Complete!\n")
  cat("============================================================\n")
  cat(sprintf("Results saved to: %s\n", opt$`output-dir`))
  cat("============================================================\n")
}

# Run main function
if (!interactive()) {
  main()
}

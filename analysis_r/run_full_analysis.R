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
#   Rscript analysis_r/run_full_analysis.R --behavior turn_left

# Add default library paths (user library first)
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-b", "--behavior"), type="character", default="turn_left",
              help="Behavior name to analyze (default: turn_left)"),
  make_option(c("-a", "--annotations-dir"), type="character", default="jabs/annotations",
              help="Directory containing annotation JSON files"),
  make_option(c("-f", "--features-dir"), type="character", default=NULL,
              help="Base directory for feature HDF5 files (default: auto-detect from jabs/features)"),
  make_option(c("-v", "--video-dir"), type="character", default=".",
              help="Directory containing video files"),
  make_option(c("-o", "--output-dir"), type="character", default="results",
              help="Output directory for all results (default: results)"),
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

# Auto-detect features directory if not specified
if (is.null(opt$`features-dir`)) {
  # Try common locations relative to current directory
  possible_paths <- c(
    "jabs/features",           # Same directory as script
    "../jabs/features",        # Parent directory
    "../../jabs/features",      # Two levels up
    "./jabs/features"          # Explicit current directory
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
    opt$`features-dir` <- "jabs/features"  # Default to most common location
    cat(sprintf("Using default features directory: %s\n", opt$`features-dir`))
    cat("If this is incorrect, specify --features-dir explicitly\n")
  }
}

# Helper function to run R script
run_script <- function(script_path, args = "", description = "") {
  if (description != "") {
    cat(sprintf("\n%s\n", description))
    cat(paste(rep("=", 60), collapse=""), "\n")
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
  cat("\n")
  cat("============================================================\n")
  cat("Complete Behavior Bout Analysis Pipeline\n")
  cat("============================================================\n")
  cat(sprintf("Behavior: %s\n", opt$behavior))
  cat(sprintf("Output directory: %s\n", opt$`output-dir`))
  cat(sprintf("Distance metric: %s\n", opt$`distance-metric`))
  if (opt$`use-pca`) {
    cat(sprintf("PCA reduction: Yes (%.0f%% variance)\n", opt$`pca-variance` * 100))
  }
  cat("\n")
  
  # Create output directory
  dir.create(opt$`output-dir`, showWarnings=FALSE, recursive=TRUE)
  
  # Step 1: Feature Extraction
  cat("\n[1/5] Feature Extraction\n")
  feature_file <- file.path(opt$`output-dir`, "bout_features.csv")
  
  # Quote paths that may contain spaces
  args <- sprintf("--behavior %s --annotations-dir '%s' --features-dir '%s' --output '%s'",
                  opt$behavior,
                  opt$`annotations-dir`,
                  opt$`features-dir`,
                  feature_file)
  
  if (opt$verbose) {
    args <- paste(args, "--verbose")
  }
  
  run_script("analysis_r/extract_bout_features.R", args, 
            "[1/5] Extracting features from HDF5 files...")
  
  if (!file.exists(feature_file)) {
    stop(sprintf("Feature extraction failed. File not found: %s", feature_file))
  }
  
  # Step 2: Clustering
  if (!opt$`skip-clustering`) {
    cat("\n[2/5] Clustering Analysis\n")
    cluster_dir <- file.path(opt$`output-dir`, "clustering")
    dir.create(cluster_dir, showWarnings=FALSE, recursive=TRUE)
    
    args <- sprintf("--input '%s' --method all --output-dir '%s'",
                   feature_file, cluster_dir)
    
    if (opt$verbose) {
      args <- paste(args, "--verbose")
    }
    
    run_script("analysis_r/cluster_bouts.R", args,
              "[2/5] Performing clustering analysis (K-means, hierarchical, DBSCAN)...")
    
    # Step 3: Visualization
    cat("\n[3/5] Visualization\n")
    
    # Visualize K-means clusters
    kmeans_clusters <- file.path(cluster_dir, "cluster_assignments_kmeans.csv")
    if (file.exists(kmeans_clusters)) {
      args <- sprintf("--features '%s' --clusters '%s' --output-dir '%s'",
                     feature_file, kmeans_clusters, cluster_dir)
      
      if (opt$verbose) {
        args <- paste(args, "--verbose")
      }
      
      run_script("analysis_r/visualize_clusters.R", args,
                "[3/5] Creating cluster visualizations...")
    }
    
    # Step 4: Generate Cluster Videos
    if (!opt$`skip-videos`) {
      cat("\n[4/5] Cluster Video Generation\n")
      
      # K-means videos
      if (file.exists(kmeans_clusters)) {
        video_dir <- file.path(opt$`output-dir`, "cluster_videos_kmeans")
        args <- sprintf("--clusters '%s' --method kmeans --output-dir '%s' --video-dir '%s'",
                       kmeans_clusters, video_dir, opt$`video-dir`)
        
        if (!is.null(opt$workers)) {
          args <- paste(args, sprintf("--workers %d", opt$workers))
        }
        
        if (opt$verbose) {
          args <- paste(args, "--verbose")
        }
        
        run_script("analysis_r/generate_cluster_videos.R", args,
                  "[4/5] Generating K-means cluster videos...")
      }
      
      # Hierarchical videos
      hier_clusters <- file.path(cluster_dir, "cluster_assignments_hierarchical.csv")
      if (file.exists(hier_clusters)) {
        video_dir <- file.path(opt$`output-dir`, "cluster_videos_hierarchical")
        args <- sprintf("--clusters '%s' --method hierarchical --output-dir '%s' --video-dir '%s'",
                       hier_clusters, video_dir, opt$`video-dir`)
        
        if (!is.null(opt$workers)) {
          args <- paste(args, sprintf("--workers %d", opt$workers))
        }
        
        if (opt$verbose) {
          args <- paste(args, "--verbose")
        }
        
        run_script("analysis_r/generate_cluster_videos.R", args,
                  "[4/5] Generating hierarchical cluster videos...")
      }
    }
  } else {
    cat("\n[2-4/5] Clustering, Visualization, and Videos (SKIPPED)\n")
  }
  
  # Step 5: Outlier Detection
  if (!opt$`skip-outliers`) {
    cat("\n[5/5] Outlier Detection\n")
    outlier_dir <- file.path(opt$`output-dir`, "outlier_videos")
    
    args <- sprintf("--features '%s' --output-dir '%s' --video-dir '%s' --distance-metric %s",
                   feature_file, outlier_dir, opt$`video-dir`, opt$`distance-metric`)
    
    if (opt$`use-pca`) {
      args <- paste(args, sprintf("--use-pca --pca-variance %.2f", opt$`pca-variance`))
    }
    
    if (!is.null(opt$workers)) {
      args <- paste(args, sprintf("--workers %d", opt$workers))
    }
    
    if (opt$verbose) {
      args <- paste(args, "--verbose")
    }
    
    run_script("analysis_r/find_outliers.R", args,
              sprintf("[5/5] Finding outliers using %s distance...", opt$`distance-metric`))
  } else {
    cat("\n[5/5] Outlier Detection (SKIPPED)\n")
  }
  
  # Summary
  cat("\n")
  cat("============================================================\n")
  cat("Analysis Complete!\n")
  cat("============================================================\n")
  cat(sprintf("Results saved to: %s\n", opt$`output-dir`))
  cat("\nGenerated files:\n")
  
  if (file.exists(feature_file)) {
    cat(sprintf("  ✓ Feature matrix: %s\n", feature_file))
  }
  
  if (!opt$`skip-clustering`) {
    cluster_dir <- file.path(opt$`output-dir`, "clustering")
    if (dir.exists(cluster_dir)) {
      cat(sprintf("  ✓ Clustering results: %s/\n", cluster_dir))
      cat(sprintf("    - Cluster assignments\n"))
      cat(sprintf("    - Cluster statistics\n"))
      cat(sprintf("    - Visualizations\n"))
    }
    
    if (!opt$`skip-videos`) {
      kmeans_videos <- file.path(opt$`output-dir`, "cluster_videos_kmeans")
      hier_videos <- file.path(opt$`output-dir`, "cluster_videos_hierarchical")
      
      if (dir.exists(kmeans_videos)) {
        n_videos <- length(list.files(kmeans_videos, pattern="\\.mp4$"))
        cat(sprintf("  ✓ K-means cluster videos: %s/ (%d videos)\n", kmeans_videos, n_videos))
      }
      
      if (dir.exists(hier_videos)) {
        n_videos <- length(list.files(hier_videos, pattern="\\.mp4$"))
        cat(sprintf("  ✓ Hierarchical cluster videos: %s/ (%d videos)\n", hier_videos, n_videos))
      }
    }
  }
  
  if (!opt$`skip-outliers`) {
    outlier_dir <- file.path(opt$`output-dir`, "outlier_videos")
    if (dir.exists(outlier_dir)) {
      cat(sprintf("  ✓ Outlier results: %s/\n", outlier_dir))
      cat(sprintf("    - Outlier bout information\n"))
      cat(sprintf("    - Outlier video\n"))
      cat(sprintf("    - Outlier visualizations\n"))
    }
  }
  
  cat("\nNext steps:\n")
  cat("  1. Review cluster visualizations in clustering/\n")
  cat("  2. Watch cluster videos to validate behavioral patterns\n")
  cat("  3. Review outlier visualizations to understand unusual behaviors\n")
  cat("  4. Use select_bouts.R to export specific clusters for further analysis\n")
  cat("\n")
}

# Run main function
if (!interactive()) {
  main()
}


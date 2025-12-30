#!/usr/bin/env Rscript
# Multi-method outlier detection with consensus
#
# Uses multiple outlier detection methods and consensus voting

# Fix R environment issues - set valid editor before loading packages
tryCatch({
  options(editor = "vim")
}, error = function(e) {
  # If that fails, try to unset it
  tryCatch({
    options(editor = NULL)
  }, error = function(e2) {
    # Ignore if both fail
  })
})

.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(ggplot2)
  library(MASS)
})

# Get script directory first (before modifying args)
cmd_args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args_full, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- sub("^--file=", "", file_arg)
  script_dir <- dirname(normalizePath(script_path))
} else {
  script_dir <- getwd()
}

# Include functions from find_outliers.R
# Source find_outliers.R from same directory
find_outliers_path <- file.path(script_dir, "find_outliers.R")
if (!file.exists(find_outliers_path)) {
  # Try relative to current working directory
  find_outliers_path <- file.path("BoutAnalysisScripts", "scripts", "core", "find_outliers.R")
  if (!file.exists(find_outliers_path)) {
    find_outliers_path <- "find_outliers.R"  # Last fallback
  }
}
source(find_outliers_path, local=TRUE)

# Parse arguments
option_list <- list(
  make_option(c("-f", "--features"), type="character", default=NULL,
              help="Input CSV file with bout features (required)"),
  make_option(c("-o", "--output-dir"), type="character", default="BoutResults/outliers",
              help="Output directory (default: BoutResults/outliers)"),
  make_option(c("--pca-variance"), type="numeric", default=0.95,
              help="Proportion of variance to retain in PCA (default: 0.95)"),
  make_option(c("--threshold"), type="character", default="auto",
              help="Outlier threshold: 'auto' (top 5%%), 'topN', or percentile"),
  make_option(c("--top-n"), type="integer", default=NULL,
              help="Number of top outliers (used with threshold='topN')"),
  make_option(c("--scale-method"), type="character", default="standard",
              help="Feature scaling method: standard, minmax, or robust (default: standard)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

# Remove --consensus-min from args before parsing (getopt limitation)
args <- commandArgs(trailingOnly = TRUE)
consensus_min <- 2  # default
consensus_idx <- grep("^--consensus-min$", args)
if (length(consensus_idx) > 0 && consensus_idx < length(args)) {
  consensus_min <- as.integer(args[consensus_idx + 1])
  # Remove from args so optparse doesn't see it
  args <- args[-c(consensus_idx, consensus_idx + 1)]
}

opt <- parse_args(OptionParser(option_list=option_list), args = args)

# Check required arguments
if (is.null(opt$features)) {
  stop("Error: --features is required. Use --help for usage information.")
}

# Functions from find_outliers.R are already sourced above

# Main execution
main <- function() {
  cat("Loading features...\n")
  df <- read.csv(opt$features, stringsAsFactors=FALSE)
  cat(sprintf("Loaded %d bouts\n", nrow(df)))
  
  # Prepare features
  prepared <- prepare_features(df)
  metadata_df <- prepared$metadata
  X_scaled <- prepared$features
  
  # Apply PCA
  cat(sprintf("Applying PCA (%.0f%% variance)...\n", opt$`pca-variance` * 100))
  pca_info <- apply_pca_reduction(X_scaled, opt$`pca-variance`)
  X_pca <- pca_info$X_reduced
  cat(sprintf("PCA: %d -> %d components (%.1f%% variance)\n",
             ncol(X_scaled), ncol(X_pca), pca_info$variance_explained * 100))
  
  # Multiple methods
  methods <- list(
    list(name="mean_mahalanobis", metric="mahalanobis", method="mean_distance"),
    list(name="median_mahalanobis", metric="mahalanobis", method="median_distance"),
    list(name="max_mahalanobis", metric="mahalanobis", method="max_distance"),
    list(name="mean_euclidean", metric="euclidean", method="mean_distance"),
    list(name="median_euclidean", metric="euclidean", method="median_distance")
  )
  
  outlier_results <- list()
  all_scores <- data.frame(bout_id = df$bout_id)
  
  for (m in methods) {
    cat(sprintf("\nMethod: %s...\n", m$name))
    dist_matrix <- calculate_distance_matrix(X_pca, metric=m$metric, 
                                            use_pca=FALSE, 
                                            pca_variance=opt$`pca-variance`)
    scores <- calculate_aggregate_distances(dist_matrix, method=m$method)
    all_scores[[m$name]] <- scores
    outliers <- identify_outliers(scores, threshold=opt$threshold, top_n=opt$`top-n`)
    outlier_results[[m$name]] <- outliers
    cat(sprintf("  Found %d outliers\n", length(outliers)))
  }
  
  # Consensus
  cat("\nComputing consensus...\n")
  consensus <- integer(0)
  votes <- integer(nrow(df))
  for (i in 1:nrow(df)) {
    vote_count <- sum(sapply(outlier_results, function(x) i %in% x))
    votes[i] <- vote_count
    # consensus_min is set above from command line args
    if (vote_count >= consensus_min) {
      consensus <- c(consensus, i)
    }
  }
  
  cat(sprintf("Consensus outliers (>=%d methods): %d\n", 
             consensus_min, length(consensus)))
  
  # Save results
  dir.create(opt$`output-dir`, showWarnings=FALSE, recursive=TRUE)
  
  outlier_df <- df[consensus, ]
  outlier_df$consensus_votes <- votes[consensus]
  for (m in methods) {
    outlier_df[[paste0(m$name, "_score")]] <- all_scores[[m$name]][consensus]
  }
  
  write.csv(outlier_df, file.path(opt$`output-dir`, "consensus_outliers.csv"), row.names=FALSE)
  write.csv(all_scores, file.path(opt$`output-dir`, "all_outlier_scores.csv"), row.names=FALSE)
  
  cat(sprintf("✓ Saved to %s\n", opt$`output-dir`))
  
  return(list(consensus=consensus, scores=all_scores, votes=votes))
}

if (!interactive()) {
  result <- main()
}


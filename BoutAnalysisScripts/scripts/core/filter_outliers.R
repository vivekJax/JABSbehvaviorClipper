#!/usr/bin/env Rscript
# Fix R environment issues
tryCatch({ options(editor = "vim") }, error = function(e) { tryCatch({ options(editor = NULL) }, error = function(e2) BoutAnalysisScripts/scripts/core/filter_outliers.R) })
# Fix R environment issues
options(editor = NULL)
options(defaultPackages = c("datasets", "utils", "grDevices", "graphics", "stats", "methods"))
# Filter out outlier bouts from feature data
#
# This script:
# 1. Loads outlier explanations
# 2. Identifies outlier bouts (by consensus or specific method)
# 3. Removes outliers from feature data
# 4. Saves filtered feature data for clustering
#
# Usage:
#   Rscript analysis_r/filter_outliers.R --features bout_features.csv --output bout_features_filtered.csv

# Add default library paths
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-f", "--features"), type="character", default="results/bout_features.csv",
              help="Input CSV file with bout features (default: results/bout_features.csv)"),
  make_option(c("-e", "--explanations"), type="character", default="results/outlier_detection/outlier_explanations.csv",
              help="Outlier explanations CSV file (default: results/outlier_detection/outlier_explanations.csv)"),
  make_option(c("-o", "--output"), type="character", default="results/bout_features_filtered.csv",
              help="Output CSV file with outliers removed (default: results/bout_features_filtered.csv)"),
  make_option(c("--method"), type="character", default="consensus",
              help="Outlier removal method: consensus, mahalanobis, lof, isolation, or all (default: consensus)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Main execution
main <- function() {
  cat("============================================================\n")
  cat("Filter Outliers from Feature Data\n")
  cat("============================================================\n\n")
  
  # Load features
  cat(sprintf("Loading features from: %s\n", opt$features))
  if (!file.exists(opt$features)) {
    stop(sprintf("Features file not found: %s", opt$features))
  }
  
  df <- read.csv(opt$features, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d bouts\n", nrow(df)))
  
  # Load outlier explanations
  cat(sprintf("Loading outlier explanations from: %s\n", opt$explanations))
  if (!file.exists(opt$explanations)) {
    stop(sprintf("Outlier explanations file not found: %s. Run outlier detection first.", opt$explanations))
  }
  
  explanations_df <- read.csv(opt$explanations, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d outlier records\n", nrow(explanations_df)))
  
  # Identify outlier bout IDs based on method
  if (opt$method == "consensus") {
    outlier_ids <- explanations_df$bout_id[explanations_df$consensus_outlier == TRUE]
    cat(sprintf("Removing %d consensus outliers\n", length(outlier_ids)))
  } else if (opt$method == "mahalanobis") {
    outlier_ids <- explanations_df$bout_id[explanations_df$is_outlier_mahalanobis == TRUE]
    cat(sprintf("Removing %d Mahalanobis outliers\n", length(outlier_ids)))
  } else if (opt$method == "lof") {
    outlier_ids <- explanations_df$bout_id[explanations_df$is_outlier_lof == TRUE]
    cat(sprintf("Removing %d LOF outliers\n", length(outlier_ids)))
  } else if (opt$method == "isolation") {
    outlier_ids <- explanations_df$bout_id[explanations_df$is_outlier_isolation == TRUE]
    cat(sprintf("Removing %d Isolation Forest outliers\n", length(outlier_ids)))
  } else if (opt$method == "all") {
    # Remove if flagged by any method
    outlier_ids <- explanations_df$bout_id[
      explanations_df$is_outlier_mahalanobis == TRUE |
      explanations_df$is_outlier_lof == TRUE |
      explanations_df$is_outlier_isolation == TRUE
    ]
    outlier_ids <- unique(outlier_ids)
    cat(sprintf("Removing %d outliers (flagged by any method)\n", length(outlier_ids)))
  } else {
    stop(sprintf("Unknown method: %s", opt$method))
  }
  
  # Filter out outliers
  df_filtered <- df[!df$bout_id %in% outlier_ids, ]
  
  cat(sprintf("\nFiltered data: %d bouts remaining (removed %d outliers)\n", 
             nrow(df_filtered), length(outlier_ids)))
  
  # Save filtered data
  write.csv(df_filtered, opt$output, row.names = FALSE)
  cat(sprintf("\nFiltered features saved to: %s\n", opt$output))
  
  cat("\n============================================================\n")
  cat("Outlier filtering complete!\n")
}

# Run main
main()


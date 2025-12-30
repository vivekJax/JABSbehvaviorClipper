#!/usr/bin/env Rscript
# Fix R environment issues
tryCatch({ options(editor = "vim") }, error = function(e) { tryCatch({ options(editor = NULL) }, error = function(e2) BoutAnalysisScripts/scripts/core/select_bouts.R) })
# Fix R environment issues
options(editor = NULL)
options(defaultPackages = c("datasets", "utils", "grDevices", "graphics", "stats", "methods"))
# Select behavior bouts based on cluster analysis.
#
# This script allows selection of bouts by:
# - Cluster ID(s)
# - Feature value ranges
# - Animal ID or video
# - Custom filters

# Add default library paths (user library first)
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(jsonlite)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-c", "--clusters"), type="character", default=NULL,
              help="Input CSV file with cluster assignments (required)"),
  make_option(c("-f", "--features"), type="character", default=NULL,
              help="Optional: Input CSV file with features for filtering"),
  make_option(c("--cluster-ids"), type="character", default=NULL,
              help="Select bouts from these cluster IDs (comma-separated)"),
  make_option(c("--animal-ids"), type="character", default=NULL,
              help="Select bouts from these animal IDs (comma-separated)"),
  make_option(c("--videos"), type="character", default=NULL,
              help="Select bouts from these video names (comma-separated)"),
  make_option(c("--feature-filter"), type="character", default=NULL,
              help="Filter by feature range: FEATURE_NAME,MIN,MAX"),
  make_option(c("--output-json"), type="character", default=NULL,
              help="Output JSON file path for selected bouts"),
  make_option(c("--output-csv"), type="character", default=NULL,
              help="Output CSV file path for selected bouts"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Load cluster results
load_cluster_results <- function(cluster_file, features_file = NULL) {
  df <- read.csv(cluster_file, stringsAsFactors=FALSE)
  
  if (!is.null(features_file) && file.exists(features_file)) {
    features_df <- read.csv(features_file, stringsAsFactors=FALSE)
    df <- merge(df, features_df, by="bout_id", all.x=TRUE, suffixes=c("", "_feat"))
    cat("Merged cluster assignments with features\n")
  }
  
  return(df)
}

# Select by clusters
select_by_clusters <- function(df, cluster_ids) {
  selected <- df[df$cluster_id %in% cluster_ids, , drop=FALSE]
  cat(sprintf("Selected %d bouts from clusters %s\n", nrow(selected), 
             paste(cluster_ids, collapse=", ")))
  return(selected)
}

# Select by feature range
select_by_feature_range <- function(df, feature_name, min_val = NULL, max_val = NULL) {
  if (!feature_name %in% colnames(df)) {
    cat(sprintf("Feature '%s' not found in dataframe\n", feature_name))
    return(data.frame())
  }
  
  mask <- rep(TRUE, nrow(df))
  
  if (!is.null(min_val) && min_val != "none") {
    mask <- mask & (df[[feature_name]] >= as.numeric(min_val))
  }
  if (!is.null(max_val) && max_val != "none") {
    mask <- mask & (df[[feature_name]] <= as.numeric(max_val))
  }
  
  selected <- df[mask, , drop=FALSE]
  cat(sprintf("Selected %d bouts with %s in range [%s, %s]\n", 
             nrow(selected), feature_name, 
             ifelse(is.null(min_val), "none", min_val),
             ifelse(is.null(max_val), "none", max_val)))
  return(selected)
}

# Select by animal
select_by_animal <- function(df, animal_ids) {
  selected <- df[df$animal_id %in% animal_ids, , drop=FALSE]
  cat(sprintf("Selected %d bouts from animals %s\n", nrow(selected),
             paste(animal_ids, collapse=", ")))
  return(selected)
}

# Select by video
select_by_video <- function(df, video_names) {
  selected <- df[df$video_name %in% video_names, , drop=FALSE]
  cat(sprintf("Selected %d bouts from videos %s\n", nrow(selected),
             paste(video_names, collapse=", ")))
  return(selected)
}

# Export to JSON
export_to_json <- function(selected_df, output_path) {
  bouts <- list()
  
  for (i in seq_len(nrow(selected_df))) {
    row <- selected_df[i, ]
    bout <- list(
      video_name = row$video_name,
      identity = as.character(row$animal_id),
      start_frame = as.integer(row$start_frame),
      end_frame = as.integer(row$end_frame),
      behavior = if ("behavior" %in% colnames(row)) row$behavior else "unknown"
    )
    
    if ("cluster_id" %in% colnames(row)) {
      bout$cluster_id <- as.integer(row$cluster_id)
    }
    
    bouts[[length(bouts) + 1]] <- bout
  }
  
  write_json(bouts, output_path, pretty=TRUE)
  cat(sprintf("Exported %d selected bouts to %s\n", length(bouts), output_path))
}

# Export to CSV
export_to_csv <- function(selected_df, output_path) {
  write.csv(selected_df, output_path, row.names=FALSE)
  cat(sprintf("Exported %d selected bouts to %s\n", nrow(selected_df), output_path))
}

# Print selection summary
print_selection_summary <- function(selected_df) {
  cat("\n============================================================\n")
  cat("Selection Summary\n")
  cat("============================================================\n")
  cat(sprintf("Total selected bouts: %d\n", nrow(selected_df)))
  
  if ("cluster_id" %in% colnames(selected_df)) {
    cluster_counts <- table(selected_df$cluster_id)
    cat(sprintf("\nClusters: %s\n", paste(names(cluster_counts), cluster_counts, sep="=", collapse=", ")))
  }
  
  if ("animal_id" %in% colnames(selected_df)) {
    animal_counts <- table(selected_df$animal_id)
    cat(sprintf("\nAnimals: %s\n", paste(names(animal_counts), animal_counts, sep="=", collapse=", ")))
  }
  
  if ("video_name" %in% colnames(selected_df)) {
    video_counts <- table(selected_df$video_name)
    cat(sprintf("\nVideos: %d unique videos\n", length(video_counts)))
    top_5 <- head(sort(video_counts, decreasing=TRUE), 5)
    cat(sprintf("Top 5 videos: %s\n", paste(names(top_5), top_5, sep="=", collapse=", ")))
  }
  
  if ("duration_frames" %in% colnames(selected_df)) {
    cat(sprintf("\nDuration statistics:\n"))
    cat(sprintf("  Mean: %.1f frames\n", mean(selected_df$duration_frames)))
    cat(sprintf("  Min: %.0f frames\n", min(selected_df$duration_frames)))
    cat(sprintf("  Max: %.0f frames\n", max(selected_df$duration_frames)))
  }
}

# Main execution
main <- function() {
  # Check required argument
  if (is.null(opt$clusters)) {
    stop("--clusters argument is required. Use --help for usage information.")
  }
  
  # Load data
  df <- load_cluster_results(opt$clusters, opt$features)
  
  if (nrow(df) == 0) {
    stop("No data loaded")
  }
  
  cat(sprintf("Loaded %d bouts\n", nrow(df)))
  
  # Apply filters
  selected_df <- df
  
  if (!is.null(opt$`cluster-ids`)) {
    cluster_ids <- as.integer(strsplit(opt$`cluster-ids`, ",")[[1]])
    selected_df <- select_by_clusters(selected_df, cluster_ids)
  }
  
  if (!is.null(opt$`animal-ids`)) {
    animal_ids <- strsplit(opt$`animal-ids`, ",")[[1]]
    selected_df <- select_by_animal(selected_df, animal_ids)
  }
  
  if (!is.null(opt$videos)) {
    video_names <- strsplit(opt$videos, ",")[[1]]
    selected_df <- select_by_video(selected_df, video_names)
  }
  
  if (!is.null(opt$`feature-filter`)) {
    parts <- strsplit(opt$`feature-filter`, ",")[[1]]
    if (length(parts) == 3) {
      feature_name <- parts[1]
      min_val <- if (parts[2] == "none") NULL else parts[2]
      max_val <- if (parts[3] == "none") NULL else parts[3]
      selected_df <- select_by_feature_range(selected_df, feature_name, min_val, max_val)
    }
  }
  
  if (nrow(selected_df) == 0) {
    cat("No bouts match the selection criteria\n")
    return
  }
  
  # Print summary
  print_selection_summary(selected_df)
  
  # Export results
  if (!is.null(opt$`output-json`)) {
    export_to_json(selected_df, opt$`output-json`)
  }
  
  if (!is.null(opt$`output-csv`)) {
    export_to_csv(selected_df, opt$`output-csv`)
  }
  
  if (is.null(opt$`output-json`) && is.null(opt$`output-csv`)) {
    cat("\nNo output file specified. Use --output-json or --output-csv to save results.\n")
  }
}

# Run main function
if (!interactive()) {
  main()
}


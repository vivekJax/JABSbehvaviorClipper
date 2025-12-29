#!/usr/bin/env Rscript
# Generate video montages for each cluster
#
# This script:
# 1. Loads cluster assignments
# 2. For each cluster, creates temporary annotation JSON files
# 3. Calls generate_bouts_video.py to create video montages for each cluster
#
# Usage:
#   Rscript analysis_r/generate_cluster_videos.R --clusters results/cluster_assignments_kmeans.csv --output-dir results/clustering/videos

# Add default library paths
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
  library(dplyr)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-c", "--clusters"), type="character", default=NULL,
              help="Cluster assignments CSV file (required)"),
  make_option(c("--output-dir"), type="character", default="results/clustering/videos",
              help="Output directory for cluster videos (default: results/clustering/videos)"),
  make_option(c("--method"), type="character", default=NULL,
              help="Clustering method name (kmeans, hierarchical, dbscan). Auto-detected from filename if not provided."),
  make_option(c("--annotations-dir"), type="character", default="jabs/annotations",
              help="Directory containing annotation JSON files"),
  make_option(c("--video-dir"), type="character", default=".",
              help="Directory containing video files"),
  make_option(c("--behavior"), type="character", default="turn_left",
              help="Behavior name (default: turn_left)"),
  make_option(c("--workers"), type="integer", default=NULL,
              help="Number of parallel workers for video clipping"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$clusters)) {
  stop("--clusters argument is required")
}

# Get default number of workers (CPU cores - 1)
get_default_workers <- function() {
  tryCatch({
    result <- system("python3 -c 'import multiprocessing; print(multiprocessing.cpu_count() - 1)'", 
                    intern = TRUE)
    as.integer(result[1])
  }, error = function(e) {
    return(1)
  })
}

# Create temporary annotation JSON files for cluster bouts
create_cluster_annotations <- function(cluster_bouts, behavior_name, output_dir) {
  # Group by video
  videos <- unique(cluster_bouts$video_name)
  
  annotation_files <- character(length(videos))
  
  for (i in seq_along(videos)) {
    video_name <- videos[i]
    video_bouts <- cluster_bouts[cluster_bouts$video_name == video_name, ]
    
    # Group by animal_id
    animals <- unique(video_bouts$animal_id)
    
    labels <- list()
    for (animal_id in animals) {
      animal_bouts <- video_bouts[video_bouts$animal_id == animal_id, ]
      
      # Create bout list for this animal
      bout_list <- lapply(seq_len(nrow(animal_bouts)), function(j) {
        list(
          start = animal_bouts$start_frame[j],
          end = animal_bouts$end_frame[j],
          present = TRUE
        )
      })
      
      # Initialize labels structure
      labels[[as.character(animal_id)]] <- list()
      labels[[as.character(animal_id)]][[behavior_name]] <- bout_list
    }
    
    # Create annotation structure
    annotation <- list(
      version = 1,
      file = as.character(video_name)[1],
      num_frames = NA,  # Will be determined by video clipper
      labels = labels
    )
    
    # Write to JSON file
    video_basename <- tools::file_path_sans_ext(video_name)
    annotation_file <- file.path(output_dir, sprintf("%s.json", video_basename))
    write_json(annotation, annotation_file, auto_unbox = TRUE, pretty = TRUE)
    annotation_files[i] <- annotation_file
  }
  
  return(annotation_files)
}

# Main execution
main <- function() {
  cat("============================================================\n")
  cat("Cluster Video Generation\n")
  cat("============================================================\n\n")
  
  # Detect method from filename if not provided
  method <- opt$method
  if (is.null(method)) {
    filename <- basename(opt$clusters)
    if (grepl("kmeans", filename, ignore.case = TRUE)) {
      method <- "kmeans"
    } else if (grepl("hierarchical", filename, ignore.case = TRUE)) {
      method <- "hierarchical"
    } else if (grepl("dbscan", filename, ignore.case = TRUE)) {
      method <- "dbscan"
    } else {
      method <- "unknown"
    }
    cat(sprintf("Auto-detected method: %s\n", method))
  }
  
  # Load cluster assignments
  cat(sprintf("Loading cluster assignments from: %s\n", opt$clusters))
  if (!file.exists(opt$clusters)) {
    stop(sprintf("Cluster assignments file not found: %s", opt$clusters))
  }
  
  clusters_df <- read.csv(opt$clusters, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d bout assignments\n", nrow(clusters_df)))
  
  # Check if metadata columns exist, if not load from filtered features
  required_cols <- c("video_name", "animal_id", "start_frame", "end_frame")
  if (!all(required_cols %in% names(clusters_df))) {
    cat("Cluster assignments missing metadata. Loading from filtered features...\n")
    # Try to find filtered features file in results directory
    features_file <- "results/bout_features_filtered.csv"
    if (!file.exists(features_file)) {
      # Fallback to current directory
      features_file <- "bout_features_filtered.csv"
    }
    if (!file.exists(features_file)) {
      stop(sprintf("Features file not found. Checked: results/bout_features_filtered.csv and bout_features_filtered.csv. Cannot merge metadata."))
    }
    features_df <- read.csv(features_file, stringsAsFactors = FALSE)
    clusters_df <- clusters_df %>%
      left_join(features_df[, c("bout_id", required_cols)], by = "bout_id")
  }
  
  # Get unique clusters
  unique_clusters <- sort(unique(clusters_df$cluster))
  cat(sprintf("Found %d clusters\n", length(unique_clusters)))
  
  # Create method-specific output directory
  method_output_dir <- file.path(opt$`output-dir`, method, "videos")
  dir.create(method_output_dir, showWarnings = FALSE, recursive = TRUE)
  cat(sprintf("Output directory: %s\n", method_output_dir))
  
  # Determine number of workers
  n_workers <- if (is.null(opt$workers)) get_default_workers() else opt$workers
  cat(sprintf("Using %d parallel workers\n", n_workers))
  
  # Python script path
  python_script <- "generate_bouts_video.py"
  if (!file.exists(python_script)) {
    stop(sprintf("Python script not found: %s", python_script))
  }
  
  video_files <- character(0)
  
  # Generate video for each cluster
  for (cluster_id in unique_clusters) {
    cat(sprintf("\nProcessing cluster %d (%d bouts)...\n", cluster_id, 
               sum(clusters_df$cluster == cluster_id)))
    
    # Get bouts for this cluster
    cluster_bouts <- clusters_df[clusters_df$cluster == cluster_id, ]
    
    # Create temporary annotation directory
    temp_annotation_dir <- tempfile(pattern = sprintf("cluster_%d_annotations_", cluster_id))
    dir.create(temp_annotation_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Create annotation files
    annotation_files <- create_cluster_annotations(cluster_bouts, opt$behavior, temp_annotation_dir)
    cat(sprintf("Created %d annotation file(s)\n", length(annotation_files)))
    
    # Output video filename (in method-specific directory)
    output_video <- file.path(method_output_dir, sprintf("cluster_%d.mp4", cluster_id))
    
    # Build Python command
    cmd_args <- c(
      python_script,
      "--behavior", opt$behavior,
      "--annotations-dir", temp_annotation_dir,
      "--video-dir", opt$`video-dir`,
      "--output", output_video,
      "--workers", as.character(n_workers)
    )
    
    if (opt$verbose) {
      cmd_args <- c(cmd_args, "--verbose")
    }
    
    # Run Python script
    cat("Calling Python video clipper...\n")
    result <- system2("python3", cmd_args, stdout = if (opt$verbose) "" else TRUE, 
                     stderr = if (opt$verbose) "" else TRUE)
    
    exit_code <- attr(result, "status")
    if (!is.null(exit_code) && exit_code != 0) {
      warning(sprintf("Video generation failed for cluster %d with exit code %d", cluster_id, exit_code))
    } else {
      # Check if output file was created
      if (file.exists(output_video)) {
        file_size <- file.info(output_video)$size
        cat(sprintf("✓ Video created: %s (%.1f MB)\n", output_video, file_size / 1e6))
        video_files <- c(video_files, output_video)
      } else {
        warning(sprintf("Output video file not found: %s", output_video))
      }
    }
    
    # Clean up temporary annotation directory
    if (dir.exists(temp_annotation_dir)) {
      unlink(temp_annotation_dir, recursive = TRUE)
    }
  }
  
  cat("\n============================================================\n")
  cat("Cluster video generation complete!\n")
  cat(sprintf("Method: %s\n", method))
  cat(sprintf("Videos successfully created: %d\n", length(video_files)))
  cat(sprintf("Output directory: %s\n", method_output_dir))
}

# Run main
main()


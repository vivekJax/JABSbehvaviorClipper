#!/usr/bin/env Rscript
# Fix R environment issues
tryCatch({ options(editor = "vim") }, error = function(e) { tryCatch({ options(editor = NULL) }, error = function(e2) BoutAnalysisScripts/scripts/video/generate_cluster_videos.R) })
# Fix R environment issues
options(editor = NULL)
options(defaultPackages = c("datasets", "utils", "grDevices", "graphics", "stats", "methods"))
# Generate videos for each cluster using the Python video clipper.
#
# This script:
# 1. Loads cluster assignments from all clustering methods
# 2. Groups bouts by cluster ID
# 3. Creates JSON files for each cluster
# 4. Calls the Python video clipper to generate videos for each cluster
#
# Video Standards (inherited from generate_bouts_video.py):
#   - Codec: libx264 + aac
#   - Frame rate: 30 fps
#   - Preset: fast
#   - Font: /System/Library/Fonts/Helvetica.ttc (fontsize=20)
#   - Bounding boxes: Yellow outline (t=3, color=yellow@1.0)
#   - Text overlay: Bottom center with black semi-transparent background
#   - Default workers: CPU cores - 1 (leaves one core free)

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
  make_option(c("-m", "--method"), type="character", default=NULL,
              help="Clustering method name (e.g., kmeans, hierarchical, dbscan)"),
  make_option(c("-a", "--annotations-dir"), type="character", default="jabs/annotations",
              help="Directory containing annotation JSON files"),
  make_option(c("-v", "--video-dir"), type="character", default="jabs/videos",
              help="Directory containing video files"),
  make_option(c("-o", "--output-dir"), type="character", default="cluster_videos",
              help="Output directory for cluster videos"),
  make_option(c("--video-clipper"), type="character", default="scripts/generate_bouts_video.py",
              help="Path to Python video clipper script"),
  make_option(c("--behavior"), type="character", default="turn_left",
              help="Behavior name for video clipping"),
  make_option(c("--workers"), type="integer", default=NULL,
              help="Number of parallel workers for video clipping"),
  make_option(c("--keep-temp"), action="store_true", default=FALSE,
              help="Keep temporary clip files"),
  make_option(c("--verbose"), action="store_true", default=FALSE,
              help="Enable verbose logging")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Load cluster assignments
load_cluster_assignments <- function(cluster_file) {
  df <- read.csv(cluster_file, stringsAsFactors=FALSE)
  return(df)
}

# Create annotation JSON files for a specific cluster (one per video)
create_cluster_annotations <- function(cluster_df, cluster_id, annotations_dir) {
  # Filter bouts for this cluster
  cluster_bouts <- cluster_df[cluster_df$cluster_id == cluster_id, , drop=FALSE]
  
  if (nrow(cluster_bouts) == 0) {
    cat(sprintf("No bouts found for cluster %d\n", cluster_id))
    return(character(0))
  }
  
  # Group by video
  videos <- unique(cluster_bouts$video_name)
  annotation_files <- character(0)
  
  for (video_name in videos) {
    video_bouts <- cluster_bouts[cluster_bouts$video_name == video_name, , drop=FALSE]
    
    # Group by animal
    animals <- unique(video_bouts$animal_id)
    labels <- list()
    
    for (animal_id in animals) {
      animal_bouts <- video_bouts[video_bouts$animal_id == animal_id, , drop=FALSE]
      
      # Create behavior bouts list
      behavior_bouts <- list()
      
      for (i in seq_len(nrow(animal_bouts))) {
        bout <- list(
          start = as.integer(animal_bouts$start_frame[i]),
          end = as.integer(animal_bouts$end_frame[i]),
          present = TRUE
        )
        behavior_bouts[[length(behavior_bouts) + 1]] <- bout
      }
      
      labels[[as.character(animal_id)]] <- list()
      labels[[as.character(animal_id)]][[opt$behavior]] <- behavior_bouts
    }
    
    # Create annotation structure (one file per video, matching expected format)
    annotation_data <- list(
      file = as.character(video_name)[1],  # Ensure scalar, not array
      labels = labels
    )
    
    # Create filename matching video name (without extension) + .json
    video_basename <- tools::file_path_sans_ext(video_name)[1]
    json_filename <- paste0(video_basename, ".json")
    json_file <- file.path(annotations_dir, json_filename)
    
    # Write JSON file (auto_unbox=TRUE ensures scalars are not arrays)
    write_json(annotation_data, json_file, pretty=TRUE, auto_unbox=TRUE)
    annotation_files <- c(annotation_files, json_file)
    
    cat(sprintf("  Created annotation: %s (%d bouts)\n", json_filename, nrow(video_bouts)))
  }
  
  cat(sprintf("Created %d annotation files for cluster %d (total %d bouts)\n", 
             length(annotation_files), cluster_id, nrow(cluster_bouts)))
  
  return(annotation_files)
}

# Generate video for a cluster
generate_cluster_video <- function(cluster_id, annotation_files, method_name) {
  output_filename <- sprintf("cluster_%s_%d.mp4", method_name, cluster_id)
  output_path <- file.path(opt$`output-dir`, output_filename)
  
  # Note: annotation_files are already in the temp_annotations_dir from create_cluster_annotations
  # So we can use that directory directly
  temp_annotations_dir <- dirname(annotation_files[1])
  
  # Build command - use same standards as generate_bouts_video.py
  # Default workers: n-1 cores (same as generate_bouts_video.py default)
  # This matches the Python script's default: max(1, multiprocessing.cpu_count() - 1)
  if (is.null(opt$workers)) {
    # Calculate default: CPU cores - 1 (leave one core free for system responsiveness)
    # Use Python to get CPU count (more reliable cross-platform)
    cpu_cores_str <- system("python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())'", intern=TRUE)
    cpu_cores <- as.integer(cpu_cores_str[1])
    default_workers <- max(1, cpu_cores - 1)
    workers_arg <- sprintf("--workers %d", default_workers)
  } else {
    workers_arg <- sprintf("--workers %d", opt$workers)
  }
  
  cmd <- sprintf("python3 %s --behavior %s --annotations-dir %s --video-dir %s --output %s %s",
                opt$`video-clipper`,
                opt$behavior,
                temp_annotations_dir,
                opt$`video-dir`,
                output_path,
                workers_arg)
  
  # Add optional arguments
  if (opt$`keep-temp`) {
    cmd <- paste(cmd, "--keep-temp")
  }
  
  if (opt$verbose) {
    cmd <- paste(cmd, "--verbose")
  }
  
  cat(sprintf("Generating video for cluster %d (%d annotation files)...\n", 
             cluster_id, length(annotation_files)))
  
  if (opt$verbose) {
    cat(sprintf("Command: %s\n", cmd))
  }
  
  # Execute command
  result <- system(cmd, intern=FALSE)
  
  if (result == 0) {
    cat(sprintf("✓ Successfully created video: %s\n", output_path))
    return(output_path)
  } else {
    cat(sprintf("✗ Failed to create video for cluster %d (exit code: %d)\n", cluster_id, result))
    return(NULL)
  }
}

# Main execution
main <- function() {
  if (is.null(opt$clusters)) {
    stop("--clusters argument is required. Use --help for usage information.")
  }
  
  if (is.null(opt$method)) {
    # Try to infer method from filename
    filename <- basename(opt$clusters)
    if (grepl("kmeans", filename, ignore.case=TRUE)) {
      opt$method <- "kmeans"
    } else if (grepl("hierarchical", filename, ignore.case=TRUE)) {
      opt$method <- "hierarchical"
    } else if (grepl("dbscan", filename, ignore.case=TRUE)) {
      opt$method <- "dbscan"
    } else if (grepl("bsoid", filename, ignore.case=TRUE)) {
      opt$method <- "bsoid"
    } else {
      opt$method <- "unknown"
    }
    cat(sprintf("Inferred clustering method: %s\n", opt$method))
  }
  
  cat(sprintf("Loading cluster assignments from %s\n", opt$clusters))
  cluster_df <- load_cluster_assignments(opt$clusters)
  
  if (nrow(cluster_df) == 0) {
    stop("No cluster assignments found")
  }
  
  # Get unique cluster IDs (excluding noise/0 if it's DBSCAN)
  unique_clusters <- sort(unique(cluster_df$cluster_id))
  unique_clusters <- unique_clusters[unique_clusters >= 0]  # Include 0 for DBSCAN noise
  
  cat(sprintf("Found %d clusters: %s\n", length(unique_clusters), 
             paste(unique_clusters, collapse=", ")))
  
  # Create output directory
  dir.create(opt$`output-dir`, showWarnings=FALSE, recursive=TRUE)
  dir.create(file.path(opt$`output-dir`, "temp_annotations"), showWarnings=FALSE, recursive=TRUE)
  
  # Process each cluster
  video_files <- list()
  
  for (cluster_id in unique_clusters) {
    cat(sprintf("\n=== Processing Cluster %d ===\n", cluster_id))
    
    # Create temp annotations directory
    temp_ann_dir <- file.path(opt$`output-dir`, "temp_annotations", 
                              sprintf("cluster_%s_%d", opt$method, cluster_id))
    dir.create(temp_ann_dir, showWarnings=FALSE, recursive=TRUE)
    
    # Create annotation files for this cluster
    annotation_files <- create_cluster_annotations(cluster_df, cluster_id, temp_ann_dir)
    
    if (length(annotation_files) == 0) {
      next
    }
    
    # Generate video
    video_path <- generate_cluster_video(cluster_id, annotation_files, opt$method)
    
    if (!is.null(video_path)) {
      video_files[[length(video_files) + 1]] <- list(
        cluster_id = cluster_id,
        video_path = video_path,
        n_bouts = nrow(cluster_df[cluster_df$cluster_id == cluster_id, ])
      )
    }
  }
  
  # Print summary
  cat("\n============================================================\n")
  cat("Cluster Video Generation Summary\n")
  cat("============================================================\n")
  cat(sprintf("Method: %s\n", opt$method))
  cat(sprintf("Total clusters processed: %d\n", length(unique_clusters)))
  cat(sprintf("Videos successfully created: %d\n", length(video_files)))
  cat("\nGenerated videos:\n")
  
  for (video_info in video_files) {
    cat(sprintf("  Cluster %d: %s (%d bouts)\n", 
               video_info$cluster_id, 
               basename(video_info$video_path),
               video_info$n_bouts))
  }
  
  cat(sprintf("\nAll videos saved to: %s\n", opt$`output-dir`))
}

# Run main function
if (!interactive()) {
  main()
}


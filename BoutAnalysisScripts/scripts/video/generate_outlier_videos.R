#!/usr/bin/env Rscript
# Generate video montages for outlier bouts
#
# This script:
# 1. Loads outlier bout information
# 2. Creates temporary annotation JSON files
# 3. Calls generate_bouts_video.py to create video montages
#
# Usage:
#   Rscript analysis_r/generate_outlier_videos.R --outliers outliers_mahalanobis.csv --output outliers_mahalanobis.mp4

# Add default library paths
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
  library(dplyr)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-o", "--outliers"), type="character", default=NULL,
              help="CSV file with outlier bout information (required)"),
  make_option(c("--output"), type="character", default=NULL,
              help="Output video filename (default: based on input file)"),
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

if (is.null(opt$outliers)) {
  stop("--outliers argument is required")
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

# Create temporary annotation JSON files for outlier bouts
create_outlier_annotations <- function(outliers_df, behavior_name, output_dir) {
  # Group by video
  videos <- unique(outliers_df$video_name)
  
  annotation_files <- character(length(videos))
  
  for (i in seq_along(videos)) {
    video_name <- videos[i]
    video_bouts <- outliers_df[outliers_df$video_name == video_name, ]
    
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
  cat("Outlier Video Generation\n")
  cat("============================================================\n\n")
  
  # Load outliers
  cat(sprintf("Loading outliers from: %s\n", opt$outliers))
  if (!file.exists(opt$outliers)) {
    stop(sprintf("Outliers file not found: %s", opt$outliers))
  }
  
  outliers_df <- read.csv(opt$outliers, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d outlier bouts\n", nrow(outliers_df)))
  
  if (nrow(outliers_df) == 0) {
    stop("No outliers found in file")
  }
  
  # Determine output filename
  if (is.null(opt$output)) {
    # Generate from input filename
    input_basename <- tools::file_path_sans_ext(basename(opt$outliers))
    opt$output <- sprintf("%s.mp4", input_basename)
  }
  
  # Create temporary annotation directory
  temp_annotation_dir <- tempfile(pattern = "outlier_annotations_")
  dir.create(temp_annotation_dir, showWarnings = FALSE, recursive = TRUE)
  
  annotation_files <- create_outlier_annotations(outliers_df, opt$behavior, temp_annotation_dir)
  cat(sprintf("Created %d temporary annotation file(s) in: %s\n", length(annotation_files), temp_annotation_dir))
  
  # Determine number of workers
  n_workers <- if (is.null(opt$workers)) get_default_workers() else opt$workers
  cat(sprintf("Using %d parallel workers\n", n_workers))
  
  # Build Python command
  python_script <- "generate_bouts_video.py"
  if (!file.exists(python_script)) {
    stop(sprintf("Python script not found: %s", python_script))
  }
  
  # Create command arguments
  cmd_args <- c(
    python_script,
    "--behavior", opt$behavior,
    "--annotations-dir", temp_annotation_dir,
    "--video-dir", opt$`video-dir`,
    "--output", opt$output,
    "--workers", as.character(n_workers)
  )
  
  if (opt$verbose) {
    cmd_args <- c(cmd_args, "--verbose")
  }
  
  # Run Python script
  cat("\nCalling Python video clipper...\n")
  cmd_str <- paste(c("python3", cmd_args), collapse = " ")
  if (opt$verbose) {
    cat(sprintf("Command: %s\n", cmd_str))
  }
  
  result <- system2("python3", cmd_args, stdout = if (opt$verbose) "" else TRUE, 
                   stderr = if (opt$verbose) "" else TRUE)
  
  exit_code <- attr(result, "status")
  if (!is.null(exit_code) && exit_code != 0) {
    stop(sprintf("Video generation failed with exit code %d", exit_code))
  }
  
  # Check if output file was created
  if (file.exists(opt$output)) {
    file_size <- file.info(opt$output)$size
    cat(sprintf("\n✓ Video successfully created: %s (%.1f MB)\n", opt$output, file_size / 1e6))
  } else {
    warning(sprintf("Output video file not found: %s", opt$output))
  }
  
  # Clean up temporary annotation directory
  if (dir.exists(temp_annotation_dir)) {
    unlink(temp_annotation_dir, recursive = TRUE)
  }
  
  cat("\n============================================================\n")
  cat("Video generation complete!\n")
}

# Run main
main()


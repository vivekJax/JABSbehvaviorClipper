#!/usr/bin/env Rscript
# Extract features from JABS HDF5 files for each behavior bout.
#
# This script:
# 1. Loads bouts from annotation files
# 2. Matches each bout to its feature file
# 3. Extracts per-frame features for the bout's frame range
# 4. Aggregates features to bout-level statistics
# 5. Saves results to CSV

# Add default library paths (user library first)
.libPaths(c("/Users/vkumar/Library/R/arm64/4.5/library", .libPaths()))

suppressPackageStartupMessages({
  library(optparse)
  library(rhdf5)
  library(dplyr)
  library(jsonlite)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-b", "--behavior"), type="character", default="turn_left",
              help="Behavior name to extract (default: turn_left)"),
  make_option(c("-a", "--annotations-dir"), type="character", default="jabs/annotations",
              help="Directory containing annotation JSON files"),
  make_option(c("-f", "--features-dir"), type="character", default=NULL,
              help="Base directory for feature HDF5 files (default: auto-detect from jabs/features)"),
  make_option(c("-o", "--output"), type="character", default="bout_features.csv",
              help="Output CSV file (default: bout_features.csv)"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
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
    "./jabs/features"            # Explicit current directory
  )
  
  for (path in possible_paths) {
    if (dir.exists(path)) {
      opt$`features-dir` <- normalizePath(path)
      if (opt$verbose) {
        cat(sprintf("Auto-detected features directory: %s\n", opt$`features-dir`))
      }
      break
    }
  }
  
  # If still not found, use default and let user know
  if (is.null(opt$`features-dir`)) {
    opt$`features-dir` <- "jabs/features"  # Default to most common location
    if (opt$verbose) {
      cat(sprintf("Using default features directory: %s\n", opt$`features-dir`))
    }
  }
}

# Helper function to get feature file path
get_feature_file_path <- function(video_name, animal_id, features_dir) {
  video_basename <- tools::file_path_sans_ext(video_name)
  feature_path <- file.path(features_dir, video_basename, animal_id, "features.h5")
  
  if (file.exists(feature_path)) {
    return(feature_path)
  } else {
    # Try alternative paths
    # Some feature directories might use different naming conventions
    alternatives <- c(
      feature_path,  # Original path
      file.path(features_dir, video_basename, paste0(animal_id, ".h5")),  # Direct h5 file
      file.path(features_dir, video_name, animal_id, "features.h5"),  # With extension
      file.path(features_dir, basename(video_basename), animal_id, "features.h5")  # Just basename
    )
    
    for (alt_path in alternatives) {
      if (file.exists(alt_path)) {
        if (opt$verbose) {
          cat(sprintf("Found feature file at alternative path: %s\n", alt_path))
        }
        return(alt_path)
      }
    }
    
    if (opt$verbose) {
      cat(sprintf("Feature file not found: %s (tried %d alternatives)\n", feature_path, length(alternatives)))
    }
    return(NULL)
  }
}

# Load all annotation files and extract bouts
load_annotations <- function(annotations_dir) {
  json_files <- list.files(annotations_dir, pattern="\\.json$", full.names=TRUE)
  
  cat(sprintf("Found %d annotation files\n", length(json_files)))
  
  all_bouts <- list()
  bout_counter <- 0
  
  for (json_path in json_files) {
    tryCatch({
      data <- fromJSON(json_path)
      video_name <- data$file
      
      if (is.null(video_name)) next
      
      # Use unfragmented_labels to match GUI counts (original bout boundaries)
      # unfragmented_labels contains the original start/end frames as labeled
      # labels contains fragmented bouts (broken up to exclude frames missing pose)
      labels <- data$unfragmented_labels
      
      # Fall back to labels if unfragmented_labels doesn't exist
      if (is.null(labels)) {
        labels <- data$labels
        if (opt$verbose) {
          cat(sprintf("Warning: unfragmented_labels not found in %s, using labels instead\n", basename(json_path)))
        }
      }
      
      for (identity_id in names(labels)) {
        behaviors <- labels[[identity_id]]
        
        for (behavior_name in names(behaviors)) {
          behavior_bouts <- behaviors[[behavior_name]]
          
          if (is.data.frame(behavior_bouts)) {
            # Load all bouts (filtering to present=True happens later in main())
            all_behavior_bouts <- behavior_bouts
            
            for (i in seq_len(nrow(all_behavior_bouts))) {
              bout_counter <- bout_counter + 1
              all_bouts[[bout_counter]] <- list(
                bout_id = bout_counter - 1,
                video_name = video_name,
                identity = identity_id,
                start_frame = all_behavior_bouts$start[i],
                end_frame = all_behavior_bouts$end[i],
                behavior = behavior_name,
                present = all_behavior_bouts$present[i],
                annotation_file = basename(json_path)
              )
            }
          }
        }
      }
    }, error = function(e) {
      if (opt$verbose) {
        cat(sprintf("Error reading annotation file %s: %s\n", json_path, e$message))
      }
    })
  }
  
  return(all_bouts)
}

# List all feature datasets in HDF5 file
list_feature_datasets <- function(h5_file, base_path = "/features/per_frame") {
  datasets <- c()
  
  tryCatch({
    # Use h5ls to list all objects recursively
    all_objects <- h5ls(h5_file, recursive=TRUE)
    
    # Filter for datasets under base_path
    base_path_clean <- gsub("^/", "", base_path)  # Remove leading slash
    relevant_objects <- all_objects[grepl(paste0("^", base_path_clean), all_objects$group), ]
    
    # Get only datasets (not groups)
    datasets_df <- relevant_objects[relevant_objects$otype == "H5I_DATASET", ]
    
    # Construct full paths
    for (i in seq_len(nrow(datasets_df))) {
      group_path <- datasets_df$group[i]
      name <- datasets_df$name[i]
      # Remove base_path prefix and construct relative path
      relative_path <- gsub(paste0("^", base_path_clean, "/"), "", paste0(group_path, "/", name))
      datasets <- c(datasets, relative_path)
    }
  }, error = function(e) {
    if (opt$verbose) {
      cat(sprintf("Error listing datasets: %s\n", e$message))
    }
  })
  
  return(datasets)
}

# Load feature dataset for frame range
load_feature_dataset <- function(h5_file_path, dataset_path, start_frame, end_frame) {
  tryCatch({
    # Construct full path
    full_path <- paste0("/features/per_frame/", dataset_path)
    
    # Read the dataset
    dataset <- h5read(h5_file_path, full_path)
    
    # Handle different data shapes (1D array or matrix)
    if (is.matrix(dataset)) {
      dataset <- as.vector(dataset)  # Flatten if needed
    }
    
    total_frames <- length(dataset)
    
    # Clamp frame range (R is 1-indexed, HDF5 frames are 0-indexed)
    start_idx <- max(1, min(start_frame + 1, total_frames))
    end_idx <- max(start_idx, min(end_frame + 1, total_frames))
    
    if (start_idx > end_idx) {
      return(NULL)
    }
    
    # Extract data for frame range
    data <- dataset[start_idx:end_idx]
    return(data)
  }, error = function(e) {
    if (opt$verbose) {
      cat(sprintf("Error loading dataset %s: %s\n", dataset_path, e$message))
    }
    return(NULL)
  })
}

# Extract all per-frame features for a bout
extract_bout_features <- function(feature_file_path, start_frame, end_frame) {
  features <- list()
  
  tryCatch({
    # Check if file exists and is readable
    if (!file.exists(feature_file_path)) {
      if (opt$verbose) {
        cat(sprintf("Feature file does not exist: %s\n", feature_file_path))
      }
      return(features)
    }
    
    # Get all feature datasets using h5ls
    all_objects <- tryCatch({
      h5ls(feature_file_path, recursive=TRUE)
    }, error = function(e) {
      if (opt$verbose) {
        cat(sprintf("Error reading HDF5 file %s: %s\n", feature_file_path, e$message))
      }
      return(data.frame())
    })
    
    if (nrow(all_objects) == 0) {
      if (opt$verbose) {
        cat(sprintf("No objects found in HDF5 file: %s\n", feature_file_path))
      }
      return(features)
    }
    
    # Filter for datasets under /features/per_frame
    feature_objects <- all_objects[
      grepl("^/features/per_frame", all_objects$group) & 
      all_objects$otype == "H5I_DATASET", 
    ]
    
    if (nrow(feature_objects) == 0) {
      if (opt$verbose) {
        cat(sprintf("No feature datasets found in /features/per_frame for: %s\n", feature_file_path))
        # List what groups exist
        groups <- unique(all_objects$group)
        cat(sprintf("Available groups: %s\n", paste(head(groups, 5), collapse=", ")))
      }
      return(features)
    }
    
    # Load each feature for the frame range
    for (i in seq_len(nrow(feature_objects))) {
      group_path <- feature_objects$group[i]
      name <- feature_objects$name[i]
      full_path <- paste0(group_path, "/", name)
      
      data <- load_feature_dataset(feature_file_path, 
                                   gsub("^/features/per_frame/", "", full_path),
                                   start_frame, end_frame)
      
      if (!is.null(data) && length(data) > 0) {
        # Use clean feature name
        feature_name <- gsub("/", "_", gsub("^/features/per_frame/", "", full_path))
        features[[feature_name]] <- data
      }
    }
    
  }, error = function(e) {
    if (opt$verbose) {
      cat(sprintf("Error extracting features from %s: %s\n", feature_file_path, e$message))
    }
  })
  
  return(features)
}

# Aggregate per-frame features to bout-level statistics
aggregate_bout_features <- function(per_frame_features) {
  aggregated <- list()
  
  for (feature_name in names(per_frame_features)) {
    values <- per_frame_features[[feature_name]]
    
    if (length(values) == 0) next
    
    # Convert to numeric if possible, skip if not numeric
    values_numeric <- tryCatch({
      as.numeric(values)
    }, warning = function(w) NULL, error = function(e) NULL)
    
    if (is.null(values_numeric) || any(is.na(values_numeric))) {
      # Skip non-numeric features
      next
    }
    
    # Compute statistics only for numeric values
    aggregated[[paste0(feature_name, "_mean")]] <- mean(values_numeric, na.rm=TRUE)
    aggregated[[paste0(feature_name, "_std")]] <- sd(values_numeric, na.rm=TRUE)
    aggregated[[paste0(feature_name, "_min")]] <- min(values_numeric, na.rm=TRUE)
    aggregated[[paste0(feature_name, "_max")]] <- max(values_numeric, na.rm=TRUE)
    aggregated[[paste0(feature_name, "_median")]] <- median(values_numeric, na.rm=TRUE)
    aggregated[[paste0(feature_name, "_first")]] <- values_numeric[1]
    aggregated[[paste0(feature_name, "_last")]] <- values_numeric[length(values_numeric)]
    aggregated[[paste0(feature_name, "_duration")]] <- length(values_numeric)
  }
  
  return(aggregated)
}

# Main extraction function
extract_features_for_bouts <- function(bouts, features_dir, behavior_name) {
  cat(sprintf("Extracting features for %d bouts of behavior '%s'\n", length(bouts), behavior_name))
  cat(sprintf("Features directory: %s\n", features_dir))
  
  # Check for placeholder paths
  placeholder_patterns <- c(
    "absolute/path/to", 
    "actual/path/to", 
    "path/to", 
    "your/path", 
    "your/jabs", 
    "example/path",
    "/path/to/your",
    "/actual/path/to/your"
  )
  is_placeholder <- any(sapply(placeholder_patterns, function(p) grepl(p, features_dir, ignore.case=TRUE)))
  
  if (is_placeholder) {
    cat("\n")
    cat("============================================================\n")
    cat("ERROR: You are using a placeholder path!\n")
    cat("============================================================\n")
    cat(sprintf("\nCurrent path: %s\n", features_dir))
    cat("\nThis is a placeholder - you need to replace it with your ACTUAL path!\n")
    cat("\n")
    cat("HOW TO FIX:\n")
    cat("  1. Find where your HDF5 feature files are actually stored\n")
    cat("  2. Replace '/actual/path/to/your/jabs/features' with that real path\n")
    cat("\n")
    cat("EXAMPLES:\n")
    cat("  If your features are at: /Users/vkumar/data/jabs/features\n")
    cat("  Then use: --features-dir /Users/vkumar/data/jabs/features\n")
    cat("\n")
    cat("  If your features are at: /data/experiments/study_428/features\n")
    cat("  Then use: --features-dir /data/experiments/study_428/features\n")
    cat("\n")
    cat("  If your features are relative: ../../data/jabs/features\n")
    cat("  Then use: --features-dir ../../data/jabs/features\n")
    cat("\n")
    cat("TO FIND YOUR FEATURES DIRECTORY:\n")
    cat("  Look for directories containing HDF5 files (.h5)\n")
    cat("  Structure should be: features_dir/{video_name}/{animal_id}/features.h5\n")
    cat("\n")
    cat("============================================================\n")
    stop("Invalid features directory path (placeholder detected - replace with actual path!)")
  }
  
  # Check if features directory exists
  if (!dir.exists(features_dir)) {
    cat("\n")
    cat(sprintf("ERROR: Features directory does not exist: %s\n", features_dir))
    cat("\n")
    cat("Please check the --features-dir path.\n")
    cat("Common issues:\n")
    cat("  1. Path is incorrect or misspelled\n")
    cat("  2. Path is relative but you're in a different directory\n")
    cat("  3. Use absolute path if unsure: --features-dir /full/path/to/features\n")
    cat("\n")
    stop(sprintf("Features directory does not exist: %s", features_dir))
  }
  
  results <- list()
  missing_files_count <- 0
  missing_files_examples <- list()
  
  for (i in seq_along(bouts)) {
    if (i %% 10 == 0) {
      cat(sprintf("Processing bout %d/%d\n", i, length(bouts)))
    }
    
    bout <- bouts[[i]]
    
    # Match bout to feature file
    feature_file <- get_feature_file_path(bout$video_name, bout$identity, features_dir)
    
    if (is.null(feature_file)) {
      # Count missing files for summary
      missing_files_count <- missing_files_count + 1
      
      # Store first few examples
      if (length(missing_files_examples) < 3) {
        video_basename <- tools::file_path_sans_ext(bout$video_name)
        expected_path <- file.path(features_dir, video_basename, bout$identity, "features.h5")
        missing_files_examples[[length(missing_files_examples) + 1]] <- list(
          video = bout$video_name,
          animal = bout$identity,
          expected = expected_path
        )
      }
      
      if (opt$verbose) {
        cat(sprintf("No feature file found for bout: %s animal %s\n", 
                   bout$video_name, bout$identity))
      }
      next
    }
    
    # Extract per-frame features
    per_frame_features <- extract_bout_features(
      feature_file,
      bout$start_frame,
      bout$end_frame
    )
    
    if (length(per_frame_features) == 0) {
      if (opt$verbose) {
        cat(sprintf("No features extracted for bout: %s frames %d-%d\n",
                   bout$video_name, bout$start_frame, bout$end_frame))
      }
      next
    }
    
    # Aggregate to bout-level
    aggregated_features <- aggregate_bout_features(per_frame_features)
    
    # Combine with bout metadata
    bout_data <- c(
      list(
        bout_id = bout$bout_id,
        video_name = bout$video_name,
        animal_id = bout$identity,
        start_frame = bout$start_frame,
        end_frame = bout$end_frame,
        behavior = bout$behavior,
        duration_frames = bout$end_frame - bout$start_frame + 1
      ),
      aggregated_features
    )
    
    results[[length(results) + 1]] <- bout_data
  }
  
  cat(sprintf("Successfully extracted features for %d/%d bouts\n", 
             length(results), length(bouts)))
  
  # Print summary of missing files if any
  if (missing_files_count > 0) {
    cat(sprintf("\nWarning: %d bouts had no feature files found\n", missing_files_count))
    if (length(missing_files_examples) > 0) {
      cat("Example missing feature file paths:\n")
      for (example in missing_files_examples) {
        cat(sprintf("  Video: %s, Animal: %s\n", example$video, example$animal))
        cat(sprintf("    Expected: %s\n", example$expected))
        
        # Check if directory exists
        expected_dir <- dirname(example$expected)
        if (dir.exists(expected_dir)) {
          cat(sprintf("    Directory exists, but features.h5 not found\n"))
          # List what's in the directory
          dir_contents <- list.files(expected_dir, full.names=FALSE)
          if (length(dir_contents) > 0) {
            cat(sprintf("    Directory contains: %s\n", paste(head(dir_contents, 5), collapse=", ")))
          }
        } else {
          cat(sprintf("    Directory does not exist\n"))
          # Check parent directory
          parent_dir <- dirname(expected_dir)
          if (dir.exists(parent_dir)) {
            parent_contents <- list.files(parent_dir, full.names=FALSE)
            cat(sprintf("    Parent directory contains: %s\n", paste(head(parent_contents, 5), collapse=", ")))
          }
        }
      }
      cat("\nTip: Check that feature directory structure matches:\n")
      cat("  features_dir/{video_basename}/{animal_id}/features.h5\n")
      cat("Or run with --verbose for more details.\n")
    }
  }
  
  return(results)
}

# Main execution
main <- function() {
  # Check if output file already exists (from Python pipeline)
  if (file.exists(opt$output)) {
    cat(sprintf("Output file already exists: %s\n", opt$output))
    cat("This file may have been generated by the Python pipeline (extract_bout_features.py).\n")
    cat("If you want to regenerate, delete the file or use a different --output path.\n")
    cat(sprintf("Skipping feature extraction. Using existing file: %s\n", opt$output))
    return(invisible(NULL))
  }
  
  cat(sprintf("Loading annotations from %s\n", opt$`annotations-dir`))
  all_bouts <- load_annotations(opt$`annotations-dir`)
  
  if (length(all_bouts) == 0) {
    stop("No bouts found in annotation files")
  }
  
  # Filter by behavior
  # Count behaviors before filtering for verification
  behavior_counts <- table(sapply(all_bouts, function(b) b$behavior))
  if (opt$verbose) {
    cat("\nBehaviors found in annotation files:\n")
    for (beh in names(behavior_counts)) {
      cat(sprintf("  %s: %d bouts\n", beh, behavior_counts[beh]))
    }
  }
  
  bouts <- Filter(function(b) b$behavior == opt$behavior, all_bouts)
  
  if (length(bouts) == 0) {
    stop(sprintf("No bouts found for behavior '%s'. Available behaviors: %s", 
                 opt$behavior, paste(names(behavior_counts), collapse=", ")))
  }
  
  # Count present vs not present for the selected behavior
  present_counts <- table(sapply(bouts, function(b) if(is.null(b$present)) "unknown" else if(b$present) "present=True" else "present=False"))
  cat(sprintf("\nFound %d bouts for behavior '%s':\n", length(bouts), opt$behavior))
  for (status in names(present_counts)) {
    cat(sprintf("  %s: %d\n", status, present_counts[status]))
  }
  
  # Filter to only include present=True bouts for feature extraction
  bouts <- Filter(function(b) !is.null(b$present) && b$present == TRUE, bouts)
  cat(sprintf("\nFiltering to present=True bouts only: %d bouts remaining\n", length(bouts)))
  
  if (length(bouts) == 0) {
    stop(sprintf("No bouts with present=True found for behavior '%s'", opt$behavior))
  }
  
  # Extract features
  results <- extract_features_for_bouts(bouts, opt$`features-dir`, opt$behavior)
  
  if (length(results) == 0) {
    cat("\nERROR: No features extracted for any bouts.\n")
    cat("Possible issues:\n")
    cat("  1. Feature file paths don't match video names\n")
    cat("  2. Feature directory structure is different\n")
    cat("  3. Animal IDs don't match\n")
    cat("\nTroubleshooting:\n")
    cat(sprintf("  - Features directory: %s\n", opt$`features-dir`))
    cat(sprintf("  - Number of bouts: %d\n", length(bouts)))
    cat(sprintf("  - Sample video name: %s\n", if(length(bouts) > 0) bouts[[1]]$video_name else "N/A"))
    cat(sprintf("  - Sample animal ID: %s\n", if(length(bouts) > 0) bouts[[1]]$identity else "N/A"))
    cat(sprintf("  - Expected feature path: %s\n", 
               if(length(bouts) > 0) {
                 video_basename <- tools::file_path_sans_ext(bouts[[1]]$video_name)
                 file.path(opt$`features-dir`, video_basename, bouts[[1]]$identity, "features.h5")
               } else "N/A"))
    cat("\nTry running with --verbose to see detailed path information.\n")
    stop("No features extracted. Check feature file paths and frame ranges.")
  }
  
  # Convert to data frame
  # First, ensure all lists have the same structure
  all_names <- unique(unlist(lapply(results, names)))
  
  # Fill missing values with NA
  results_filled <- lapply(results, function(x) {
    missing_names <- setdiff(all_names, names(x))
    for (name in missing_names) {
      x[[name]] <- NA
    }
    x[all_names]  # Reorder to match
  })
  
  # Convert to data frame
  df <- do.call(rbind, lapply(results_filled, function(x) {
    as.data.frame(x, stringsAsFactors=FALSE)
  }))
  
  # Save to CSV
  write.csv(df, opt$output, row.names=FALSE)
  cat(sprintf("Saved %d bout features to %s\n", nrow(df), opt$output))
  cat(sprintf("Feature matrix shape: %d rows, %d columns\n", nrow(df), ncol(df)))
  cat(sprintf("Number of feature columns: %d\n", ncol(df) - 7))  # Subtract metadata columns
  
  # Print summary
  cat("\nSummary:\n")
  cat(sprintf("  Total bouts processed: %d\n", length(bouts)))
  cat(sprintf("  Bouts with features: %d\n", length(results)))
  cat(sprintf("  Bouts without features: %d\n", length(bouts) - length(results)))
}

# Run main function
if (!interactive()) {
  main()
}


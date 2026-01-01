# Test Explanations

This document explains what each test does and why it's important for maintaining code quality.

## test_extract_bout_features.py

### TestComputeCacheKey

**Purpose**: Tests the caching mechanism that prevents recomputing features when annotation files haven't changed.

#### `test_cache_key_consistency`
- **What it does**: Verifies that calling `compute_cache_key()` with the same inputs produces the same hash key.
- **Why it matters**: Cache keys must be deterministic. If the same inputs produce different keys, caching breaks and we waste computation time re-extracting features unnecessarily.

#### `test_cache_key_different_behavior`
- **What it does**: Ensures that different behaviors produce different cache keys.
- **Why it matters**: If two behaviors share the same cache key, they would incorrectly share cached results, leading to wrong feature data being used for analysis.

### TestLoadAnnotations

**Purpose**: Tests the core function that reads annotation JSON files and extracts behavior bout information.

#### `test_load_annotations_basic`
- **What it does**: Tests loading annotations using `unfragmented_labels` (the preferred source that matches GUI counts).
- **Why it matters**: This is the foundation of the entire pipeline. If annotation loading fails or returns wrong data, all downstream analysis is incorrect. We verify:
  - Correct number of bouts extracted
  - All bouts have the correct behavior name
  - Only `present=True` bouts are included
  - Bout IDs are sequential starting from 0
  - Frame ranges are correctly extracted

#### `test_load_annotations_fallback_to_labels`
- **What it does**: Tests that the function falls back to `labels` when `unfragmented_labels` is missing.
- **Why it matters**: Older annotation files may not have `unfragmented_labels`. The code must handle this gracefully to maintain backward compatibility.

#### `test_load_annotations_filters_present_false`
- **What it does**: Verifies that bouts with `present=False` are excluded.
- **Why it matters**: Only bouts where the animal is actually present should be analyzed. Including absent bouts would contaminate the feature data with invalid measurements.

### TestGetFeatureFilePath

**Purpose**: Tests the function that locates HDF5 feature files for a given video and animal identity.

#### `test_get_feature_file_path_standard`
- **What it does**: Tests finding feature files using the standard directory structure.
- **Why it matters**: Feature extraction depends on correctly locating HDF5 files. If paths are wrong, features can't be extracted and the pipeline fails silently or with confusing errors.

#### `test_get_feature_file_path_not_found`
- **What it does**: Tests behavior when feature files don't exist.
- **Why it matters**: The function must return `None` gracefully rather than crashing, so the pipeline can skip missing files and continue processing other bouts.

### TestExtractBoutFeatures

**Purpose**: Tests extracting per-frame features from HDF5 files for a specific bout.

#### `test_extract_bout_features_basic`
- **What it does**: Verifies that features can be extracted from a valid HDF5 file for a valid frame range.
- **Why it matters**: This is the core data extraction step. If features aren't extracted correctly, all downstream analysis (clustering, outlier detection) will be wrong.

#### `test_extract_bout_features_nonexistent_file`
- **What it does**: Tests handling when the HDF5 file doesn't exist.
- **Why it matters**: Must return empty dict gracefully, not crash. Allows pipeline to continue with other bouts.

#### `test_extract_bout_features_invalid_frame_range`
- **What it does**: Tests behavior when frame range exceeds video length.
- **Why it matters**: Invalid frame ranges can occur due to annotation errors. The function must handle this gracefully without crashing.

### TestAggregateBoutFeatures

**Purpose**: Tests the function that converts per-frame features into bout-level statistics.

#### `test_aggregate_bout_features_basic`
- **What it does**: Verifies that per-frame features are correctly aggregated into statistics (mean, std, min, max, etc.).
- **Why it matters**: Clustering algorithms need bout-level features, not per-frame data. If aggregation is wrong, clusters will be meaningless. We verify:
  - All expected statistics are computed (mean, std, min, max)
  - Values are mathematically correct (e.g., mean of [1,2,3,4,5] = 3.0)

#### `test_aggregate_bout_features_empty`
- **What it does**: Tests aggregation with no input features.
- **Why it matters**: Edge case handling. Empty features should produce empty output, not crash.

---

## test_plot_trajectories.py

### TestGetPoseFile

**Purpose**: Tests finding pose estimation HDF5 files for videos.

#### `test_get_pose_file_found`
- **What it does**: Verifies that pose files are found using the standard naming convention (`<video>_pose_est_v8.h5`).
- **Why it matters**: Trajectory plotting requires pose files. If they can't be found, trajectories can't be plotted.

#### `test_get_pose_file_not_found`
- **What it does**: Tests behavior when pose file doesn't exist.
- **Why it matters**: Must return `None` gracefully so plotting can skip videos without pose data.

### TestGetCageDimensions

**Purpose**: Tests extracting cage dimensions from HDF5 files to set plot axis limits.

#### `test_get_cage_dimensions_basic`
- **What it does**: Verifies that maximum x and y coordinates are correctly extracted from bounding box data.
- **Why it matters**: Plot axes must match cage dimensions for accurate visualization. Wrong dimensions make trajectories look distorted or cut off.

#### `test_get_cage_dimensions_nonexistent_file`
- **What it does**: Tests handling when file doesn't exist.
- **Why it matters**: Must return `None` gracefully so plotting can use default dimensions.

### TestGetLixitLocation

**Purpose**: Tests extracting the lixit (water spout) location for plotting.

#### `test_get_lixit_location_basic`
- **What it does**: Verifies that lixit coordinates are extracted and correctly converted from [y,x] to (x,y) format.
- **Why it matters**: The lixit location is plotted as a reference point. If coordinates are wrong, it appears in the wrong place, misleading interpretation of behavior relative to the water source.

#### `test_get_lixit_location_not_found`
- **What it does**: Tests behavior when lixit data is missing.
- **Why it matters**: Lixit is optional. Plotting should continue without it.

### TestExtractKeypoint

**Purpose**: Tests extracting individual keypoint positions (e.g., nose, tail) from pose data.

#### `test_extract_keypoint_basic`
- **What it does**: Verifies that keypoints are extracted correctly for a valid frame range, with proper coordinate conversion from [y,x] to (x,y).
- **Why it matters**: Keypoint trajectories are plotted to show body orientation. Wrong coordinates make trajectories meaningless.

#### `test_extract_keypoint_invalid_range`
- **What it does**: Tests handling of frame ranges outside video bounds.
- **Why it matters**: Must handle gracefully without crashing.

### TestExtractBboxCentroids

**Purpose**: Tests calculating bounding box centroids for trajectory plotting.

#### `test_extract_bbox_centroids_basic`
- **What it does**: Verifies that centroids are calculated as (x1+x2)/2, (y1+y2)/2 and returned in correct format (frame, x, y).
- **Why it matters**: Centroid trajectories are the primary visualization. Wrong calculations produce misleading plots.

#### `test_extract_bbox_centroids_filters_invalid`
- **What it does**: Ensures that invalid bounding boxes (negative or zero coordinates) are filtered out.
- **Why it matters**: Invalid boxes would create artifacts in plots (points at 0,0 or negative coordinates), making trajectories uninterpretable.

### TestExtractNoseKeypoints

**Purpose**: Tests extracting nose keypoint specifically.

#### `test_extract_nose_keypoints_basic`
- **What it does**: Verifies nose keypoint extraction works correctly.
- **Why it matters**: Nose trajectories show head direction, important for understanding behavior orientation.

### TestFindCorrectIdentityIndex

**Purpose**: Tests the function that corrects identity mismatches between annotation files and HDF5 data.

#### `test_find_correct_identity_index_basic`
- **What it does**: Verifies that the function returns a valid identity index or None.
- **Why it matters**: Identity mismatches cause wrong keypoints to be extracted. This function fixes that by finding which HDF5 identity actually matches the annotated animal.

---

## test_generate_bouts_video.py

### TestValidateVideoFile

**Purpose**: Tests video file validation before processing.

#### `test_validate_video_file_exists`
- **What it does**: Verifies that existing video files are recognized as valid.
- **Why it matters**: Prevents processing errors later by catching missing files early.

#### `test_validate_video_file_not_exists`
- **What it does**: Tests that nonexistent files are rejected.
- **Why it matters**: Must fail fast rather than crash during video processing.

#### `test_validate_video_file_directory`
- **What it does**: Ensures directories are not mistaken for video files.
- **Why it matters**: Prevents confusing errors when a directory path is accidentally passed.

### TestValidateFrameRange

**Purpose**: Tests validation of frame ranges before video clipping.

#### `test_validate_frame_range_valid`
- **What it does**: Verifies that valid frame ranges pass validation.
- **Why it matters**: Valid ranges should not be rejected, or valid bouts would be skipped.

#### `test_validate_frame_range_negative_start`
- **What it does**: Tests that negative start frames are rejected.
- **Why it matters**: Negative frames are invalid and would cause video processing to fail.

#### `test_validate_frame_range_end_before_start`
- **What it does**: Tests that end frame before start frame is rejected.
- **Why it matters**: Invalid ranges would create zero-length or negative-length clips, wasting processing time.

#### `test_validate_frame_range_exceeds_video_length`
- **What it does**: Tests handling when frame range exceeds video length.
- **Why it matters**: Should clamp to video length rather than fail, so valid portions of bouts aren't lost.

### TestGetVideoFrameCount

**Purpose**: Tests extracting video metadata (frame count) using ffprobe.

#### `test_get_video_frame_count_success`
- **What it does**: Verifies that frame count is correctly extracted from video metadata.
- **Why it matters**: Frame count is needed for validation and video processing. Wrong counts cause incorrect clipping.

#### `test_get_video_frame_count_failure`
- **What it does**: Tests behavior when ffprobe fails.
- **Why it matters**: Must return None gracefully so processing can continue with other videos.

### TestGetBouts

**Purpose**: Tests extracting behavior bouts from annotation files.

#### `test_get_bouts_basic`
- **What it does**: Verifies that bouts are correctly extracted with all required fields.
- **Why it matters**: Video generation depends on correct bout extraction. Wrong bouts = wrong videos.

#### `test_get_bouts_no_annotations`
- **What it does**: Tests behavior when no annotation files exist.
- **Why it matters**: Must return empty list gracefully, not crash.

### TestGetPoseFile

**Purpose**: Tests finding pose files for video generation (same as trajectory plotting).

#### `test_get_pose_file_found` / `test_get_pose_file_not_found`
- **What it does**: Same as trajectory plotting tests - verifies pose file discovery.
- **Why it matters**: Bounding boxes for video overlays come from pose files.

### TestGetBboxes

**Purpose**: Tests extracting bounding boxes from HDF5 files using h5dump.

#### `test_get_bboxes_success`
- **What it does**: Verifies that bounding boxes are correctly parsed from h5dump output.
- **Why it matters**: Bounding boxes are drawn on video clips. Wrong boxes = misleading visualizations.

#### `test_get_bboxes_failure`
- **What it does**: Tests behavior when h5dump fails.
- **Why it matters**: Must return empty dict gracefully so videos can be generated without boxes if needed.

### TestSecToAssTime

**Purpose**: Tests time format conversion for ASS subtitle files.

#### `test_sec_to_ass_time_basic` / `test_sec_to_ass_time_zero` / `test_sec_to_ass_time_large`
- **What it does**: Verifies correct conversion from seconds to ASS time format (H:MM:SS.CC).
- **Why it matters**: Subtitles must be synchronized with video. Wrong time format = subtitles appear at wrong times.

### TestGenerateAss

**Purpose**: Tests ASS subtitle file generation for video overlays.

#### `test_generate_ass_basic`
- **What it does**: Verifies that ASS files are generated with correct format and bounding box data.
- **Why it matters**: ASS files control bounding box overlays in videos. Wrong format = boxes don't appear or appear incorrectly.

---

## test_run_complete_analysis.py

### TestGetPythonCmd

**Purpose**: Tests detection of the correct Python interpreter.

#### `test_get_python_cmd_system_python` / `test_get_python_cmd_fallback`
- **What it does**: Verifies that the function correctly detects system Python or falls back to default.
- **Why it matters**: Different Python installations may have different packages. Wrong Python = import errors or missing dependencies.

### TestFormatTime

**Purpose**: Tests time formatting for progress display.

#### `test_format_time_seconds` / `test_format_time_minutes` / `test_format_time_hours`
- **What it does**: Verifies correct formatting of elapsed time in human-readable format.
- **Why it matters**: Progress display helps users understand pipeline status. Wrong formatting is confusing.

### TestPrintProgressHeader

**Purpose**: Tests progress header display.

#### `test_print_progress_header_basic` / `test_print_progress_header_with_time`
- **What it does**: Verifies that progress headers display correctly with step numbers and percentages.
- **Why it matters**: Progress feedback is essential for long-running pipelines. Users need to know what's happening.

### TestRunCommand

**Purpose**: Tests command execution wrapper.

#### `test_run_command_success` / `test_run_command_failure` / `test_run_command_with_progress`
- **What it does**: Verifies that commands execute correctly and errors are handled properly.
- **Why it matters**: Pipeline orchestrates many external commands. If command execution fails silently or crashes, the entire pipeline fails.

---

## test_e2e_pipeline.py

### TestEndToEndPipeline

**Purpose**: Tests complete pipeline execution from start to finish.

#### `test_pipeline_creates_output_directory`
- **What it does**: Verifies that the pipeline creates the expected output directory structure.
- **Why it matters**: If output directories aren't created, results can't be saved and the pipeline appears to succeed but produces no output.

#### `test_feature_extraction_produces_csv`
- **What it does**: Verifies that feature extraction produces a valid CSV file.
- **Why it matters**: The CSV is the input for all downstream analysis. If it's missing or malformed, clustering and outlier detection fail.

### TestPipelineIntegration

**Purpose**: Tests integration between pipeline components.

#### `test_annotation_loading_integration`
- **What it does**: Tests annotation loading with real file structure.
- **Why it matters**: Unit tests use mocks. Integration tests verify real file I/O works correctly.

#### `test_h5_file_reading_integration`
- **What it does**: Tests that HDF5 files can be read correctly.
- **Why it matters**: HDF5 reading is complex. Integration tests catch issues that unit tests with mocks might miss.

### TestDataConsistency

**Purpose**: Tests data consistency across pipeline stages.

#### `test_bout_id_consistency`
- **What it does**: Verifies that bout IDs remain consistent and sequential across processing stages.
- **Why it matters**: Inconsistent IDs cause mismatches between features, clusters, and videos, making results uninterpretable.

#### `test_frame_range_validity`
- **What it does**: Ensures all frame ranges are valid (non-negative, end >= start).
- **Why it matters**: Invalid frame ranges cause video processing to fail or produce incorrect clips.

### TestFullPipelineExecution

**Purpose**: Slow tests that run the actual full pipeline.

#### `test_full_pipeline_with_mock_data`
- **What it does**: Runs the complete pipeline with mock data to verify end-to-end execution.
- **Why it matters**: Catches integration issues that only appear when components are connected. Verifies the entire workflow works correctly.

---

## Why These Tests Matter

1. **Prevent Regressions**: When code changes, tests catch if existing functionality breaks.
2. **Documentation**: Tests serve as executable documentation showing how functions should work.
3. **Confidence**: Passing tests give confidence that code works correctly.
4. **Refactoring Safety**: Tests allow safe code refactoring by catching breaking changes.
5. **Edge Case Coverage**: Tests verify handling of error conditions and edge cases.
6. **Integration Verification**: E2E tests verify that components work together correctly.


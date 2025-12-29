# Troubleshooting Guide

## Common Issues and Solutions

### Issue: "No features extracted" Error

**Error Message**:
```
Successfully extracted features for 0/406 bouts
Error: No features extracted. Check feature file paths and frame ranges.
```

**Possible Causes**:

1. **Feature directory path is incorrect**
   - Check that `--features-dir` points to the correct location
   - Default is `../jabs/features` (relative to annotations directory)
   - Use absolute path if unsure

2. **Feature file structure doesn't match expected pattern**
   - Expected: `features_dir/{video_basename}/{animal_id}/features.h5`
   - Example: `features/org-3-uploads-stage.study_410.cage_4480.2025-08-02.03.41/0/features.h5`

3. **Video names don't match between annotations and feature files**
   - Annotation file has: `"file": "video_name.mp4"`
   - Feature directory expects: `video_name` (without .mp4)
   - Check for naming mismatches

4. **Animal IDs don't match**
   - Annotation uses: `"0"`, `"1"`, `"2"` (as strings)
   - Feature directory expects same IDs
   - Check for ID mismatches

**Solutions**:

1. **Run with verbose mode** to see detailed path information:
   ```bash
   Rscript analysis_r/run_full_analysis.R \
     --behavior turn_left \
     --verbose
   ```

2. **Check feature directory structure**:
   ```bash
   # List what's in your features directory
   ls -la jabs/features/
   
   # Check a specific video's structure
   ls -la jabs/features/{video_basename}/
   ```

3. **Verify feature file exists**:
   ```bash
   # Check if expected path exists
   ls -la jabs/features/{video_basename}/{animal_id}/features.h5
   ```

4. **Check HDF5 file structure**:
   ```bash
   # Use h5dump to inspect file structure
   h5dump -n 1 jabs/features/{video_basename}/{animal_id}/features.h5 | head -20
   ```

5. **Use absolute paths**:
   ```bash
   Rscript analysis_r/run_full_analysis.R \
     --behavior turn_left \
     --features-dir /absolute/path/to/features
   ```

### Issue: Feature Files Found But No Features Extracted

**Symptoms**:
- Feature files exist
- But `Successfully extracted features for 0/N bouts`

**Possible Causes**:

1. **HDF5 file structure is different**
   - Expected: `/features/per_frame/...` group
   - Actual: Different group structure

2. **Frame range issues**
   - Start/end frames exceed video length
   - Invalid frame ranges

3. **All features are non-numeric**
   - Features are strings or other types
   - Script skips non-numeric features

**Solutions**:

1. **Check HDF5 structure**:
   ```bash
   h5dump -n 1 features.h5 | grep -A 10 "GROUP"
   ```

2. **Run with verbose mode** to see what's being found:
   ```bash
   Rscript analysis_r/extract_bout_features.R \
     --behavior turn_left \
     --verbose
   ```

3. **Check frame ranges**:
   - Verify start_frame < end_frame
   - Check that frames don't exceed video length

### Issue: Different Feature Directory Structure

If your feature files are organized differently, you may need to adjust the path matching.

**Common alternatives**:
- `features_dir/{video_name}.h5` (single file per video)
- `features_dir/{video_basename}_{animal_id}.h5` (file per animal)
- `features_dir/{video_basename}/features.h5` (single file per video)

**Solution**: The script now tries multiple path patterns automatically. If none match, run with `--verbose` to see what paths were tried.

### Issue: Memory Errors

**Symptoms**:
- R crashes or runs out of memory
- "Cannot allocate vector" errors

**Solutions**:

1. **Use PCA reduction** to reduce dimensionality:
   ```bash
   Rscript analysis_r/run_full_analysis.R \
     --behavior turn_left \
     --use-pca \
     --pca-variance 0.90  # Lower = fewer dimensions
   ```

2. **Process in batches**: Extract features for subset of bouts

3. **Increase R memory limit** (if possible):
   ```r
   memory.limit(size = 16000)  # Windows
   # macOS/Linux: Use ulimit or system settings
   ```

### Issue: Clustering Fails

**Error**: "more cluster centers than distinct data points"

**Cause**: Too few samples or too many features

**Solutions**:

1. **Reduce number of clusters**:
   ```bash
   Rscript analysis_r/cluster_bouts.R \
     --input bout_features.csv \
     --method kmeans \
     --n-clusters 3  # Lower number
   ```

2. **Use PCA reduction** before clustering:
   - Already handled in preprocessing
   - Check that features have sufficient variance

### Issue: Video Generation Fails

**Error**: "Video file not found"

**Solutions**:

1. **Check video directory path**:
   ```bash
   # Verify videos exist
   ls -la jabs/videos/*.mp4
   ```

2. **Use absolute path**:
   ```bash
   Rscript analysis_r/generate_cluster_videos.R \
     --video-dir /absolute/path/to/videos
   ```

3. **Check video file names match annotations**:
   - Annotation: `"file": "video_name.mp4"`
   - Video file: `video_name.mp4` must exist in video directory

## Diagnostic Commands

### Check Feature File Structure

```bash
# List feature directories
find jabs/features -type d -maxdepth 2

# Check a specific video's feature files
ls -la jabs/features/{video_basename}/

# Inspect HDF5 file structure
h5dump -n 1 jabs/features/{video_basename}/{animal_id}/features.h5 | head -30
```

### Check Annotation Structure

```bash
# View an annotation file
cat jabs/annotations/{annotation_file}.json | head -20

# Check video names in annotations
grep -h '"file"' jabs/annotations/*.json | head -5
```

### Verify Paths Match

```bash
# Extract video names from annotations
grep -h '"file"' jabs/annotations/*.json | sed 's/.*"file": "\(.*\)".*/\1/' | sed 's/\.mp4$//' | head -5

# Check if corresponding feature directories exist
for video in $(grep -h '"file"' jabs/annotations/*.json | sed 's/.*"file": "\(.*\)".*/\1/' | sed 's/\.mp4$//' | head -5); do
  echo "Checking: $video"
  ls -d jabs/features/$video 2>/dev/null || echo "  NOT FOUND"
done
```

## Getting Help

1. **Run with verbose mode**:
   ```bash
   Rscript analysis_r/run_full_analysis.R --behavior turn_left --verbose
   ```

2. **Check error messages carefully** - they now include:
   - Expected file paths
   - Directory contents when files not found
   - Available HDF5 groups if structure differs

3. **Verify your directory structure** matches expected pattern:
   ```
   jabs/
   ├── annotations/
   │   └── *.json
   ├── features/
   │   └── {video_basename}/
   │       └── {animal_id}/
   │           └── features.h5
   └── videos/
       └── *.mp4
   ```

4. **Check documentation**:
   - [README.md](README.md) - Usage guide
   - [STATISTICAL_METHODOLOGY.md](STATISTICAL_METHODOLOGY.md) - Method details
   - [ANALYSIS_PIPELINE.md](ANALYSIS_PIPELINE.md) - Pipeline overview


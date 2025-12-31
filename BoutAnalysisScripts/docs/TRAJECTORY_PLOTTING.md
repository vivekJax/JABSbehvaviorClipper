# Trajectory Plotting Documentation

**Last Updated:** December 29, 2025  
**Script:** `scripts/plot_bout_trajectories.py`

---

## Overview

The trajectory plotting functionality visualizes the movement of behavior bout centroids over time by extracting bounding box data from pose estimation HDF5 files and plotting the centroid positions in 2D space. This provides a spatial representation of how animals move during specific behaviors, organized by clustering results.

---

## How Centroid is Calculated

### Bounding Box Data Structure

Bounding boxes are stored in pose estimation HDF5 files (`*_pose_est_v8.h5`) under the dataset `/poseest/bbox` with the following structure:

- **Shape:** `(n_frames, n_identities, 2, 2)`
  - `n_frames`: Total number of frames in the video
  - `n_identities`: Number of tracked animals (typically 3)
  - `2`: Two points defining the bounding box (top-left and bottom-right corners)
  - `2`: Two coordinates (x, y) for each point

### Bounding Box Points

Each bounding box is defined by two points:
- **Point 0 (index 0):** Top-left corner `(x1, y1)`
- **Point 1 (index 1):** Bottom-right corner `(x2, y2)`

### Centroid Calculation

The centroid (center point) of each bounding box is calculated as:

```
centroid_x = (x1 + x2) / 2.0
centroid_y = (y1 + y2) / 2.0
```

This gives the geometric center of the bounding box, representing the approximate center of the animal's body position in each frame.

### Data Validation

Before calculating centroids, the code validates bounding boxes to exclude missing or invalid data:

1. **Negative values:** Missing bounding boxes are marked with `-1` in the HDF5 files
2. **All zeros:** `(0, 0, 0, 0)` indicates missing data
3. **Invalid geometry:** Ensures `x2 > x1` and `y2 > y1` (valid bounding box)
4. **Minimum size:** Bounding box must have at least 1 pixel width and height
5. **Positive coordinates:** All coordinates must be positive (no zeros, no negatives)

Only frames with valid bounding boxes are included in the trajectory.

---

## Coordinate System

### Image/Video Coordinates

The plots use **image coordinate system** where:
- **Origin (0, 0):** Top-left corner of the image
- **X-axis:** Increases from left to right (0 → max_width)
- **Y-axis:** Increases from top to bottom (0 → max_height) - **INVERTED** for display

### Y-Axis Inversion

The y-axis is **inverted** in the plots so that:
- **Y = 0** appears at the **top** of the plot
- **Y = max** appears at the **bottom** of the plot

This matches standard image/video coordinate conventions where (0,0) is at the top-left corner, making it easier to relate the plots to the actual video frames.

### Cage Dimensions

The axis limits are set to match the maximum cage dimensions found across all videos:
- **X-axis:** 0 to `max_cage_width` pixels
- **Y-axis:** `max_cage_height` to 0 pixels (inverted)

All plots use **identical scaling** to enable direct comparison between trajectories.

---

## How Trajectories are Plotted

### Time-Based Coloring

Each trajectory point is colored using the **viridis colormap** to indicate temporal progression:
- **Dark purple/blue:** Early frames in the bout
- **Bright yellow:** Late frames in the bout
- **Color gradient:** Smooth transition from early to late

The frame numbers are normalized to [0, 1] range:
```python
frame_norm = (frame - frame_min) / (frame_max - frame_min)
```

### Visualization Elements

1. **Line segments:** Connect consecutive frames, colored by time
2. **Points:** Scatter plot showing each frame position, colored by time
3. **Colorbar:** Shows the time scale (0 = early, 1 = late)
4. **Grid:** Light grid lines for easier reading
5. **Labels:** Video name, animal ID, bout number, frame range

### Plot Types

#### 1. Cluster Overlay Plots
- **File format:** PNG
- **Content:** All trajectories from a cluster overlaid on one plot
- **Filename:** `cluster_{cluster_id}_all_trajectories.png`
- **Use case:** Compare trajectory patterns within a cluster

#### 2. Individual Bout PDFs
- **File format:** PDF (multi-page)
- **Content:** One bout per page, organized by cluster
- **Filename:** `cluster_{cluster_id}_individual_bouts.pdf`
- **Use case:** Detailed examination of individual bouts with full labels

---

## Data Extraction Process

### Step 1: Load Cluster Assignments

The script reads cluster assignments from:
```
BoutResults/clustering/{method}/cluster_assignments_{method}.csv
```

This file contains:
- `bout_id`: Unique identifier for each bout
- `video_name`: Source video file
- `animal_id`: Identity ID (0, 1, or 2)
- `start_frame`: First frame of the bout
- `end_frame`: Last frame of the bout
- `cluster_id`: Assigned cluster from B-SOID or hierarchical clustering

### Step 2: Find Pose Estimation Files

For each video, the script locates the corresponding pose estimation HDF5 file:
- **Pattern:** `{video_name_stem}_pose_est_v8.h5`
- **Location:** Same directory as video files (typically parent directory)

### Step 3: Extract Bounding Boxes

For each bout, the script:
1. Opens the pose estimation HDF5 file
2. Extracts bounding box data for the frame range: `[start_frame:end_frame+1, animal_id, :, :]`
3. Validates each bounding box
4. Calculates centroids for valid frames only

### Step 4: Determine Cage Dimensions

The script finds the maximum cage dimensions across all videos:
- Scans all bounding boxes in each pose file
- Filters out invalid values (negative, zero)
- Finds maximum x and y coordinates
- Uses these as axis limits for consistent scaling

### Step 5: Generate Plots

Two types of plots are generated:
1. **Cluster overlay plots:** All trajectories from a cluster on one plot
2. **Individual bout PDFs:** One PDF per cluster, one bout per page

---

## Usage

### Basic Usage

```bash
# Generate all trajectory plots (cluster overlays + individual PDFs)
python3 scripts/plot_bout_trajectories.py \
    --behavior turn_left \
    --output-dir BoutResults \
    --video-dir .. \
    --faceted

# Generate only individual bout PDFs
python3 scripts/plot_bout_trajectories.py \
    --behavior turn_left \
    --output-dir BoutResults \
    --video-dir .. \
    --faceted-only

# Generate only cluster overlay plots
python3 scripts/plot_bout_trajectories.py \
    --behavior turn_left \
    --output-dir BoutResults \
    --video-dir ..
```

### Arguments

- `--behavior`: Behavior name (e.g., `turn_left`)
- `--output-dir`: Output directory (default: `BoutResults`)
- `--video-dir`: Directory containing video and pose files (default: `..`)
- `--cluster-method`: Clustering method to use (default: `bsoid`)
- `--faceted`: Generate both cluster overlays and individual PDFs
- `--faceted-only`: Generate only individual PDFs (skip cluster overlays)
- `--max-bouts`: Limit number of bouts for testing (optional)

---

## Output Files

### Cluster Overlay Plots

**Location:** `BoutResults/clustering/{method}/trajectories/`  
**Format:** PNG  
**Files:**
- `cluster_0_all_trajectories.png`
- `cluster_1_all_trajectories.png`
- ... (one per cluster)

**Content:** All trajectories from a cluster overlaid, colored by time, with consistent scaling.

### Individual Bout PDFs

**Location:** `BoutResults/clustering/{method}/trajectories/`  
**Format:** PDF (multi-page)  
**Files:**
- `cluster_0_individual_bouts.pdf`
- `cluster_1_individual_bouts.pdf`
- ... (one per cluster)

**Content:** One page per bout, each labeled with:
- Bout number
- Video filename
- Animal ID
- Frame range (start-end, total frames)
- Cluster ID

---

## Technical Details

### HDF5 Data Access

The script uses `h5py` to read bounding box data:

```python
with h5py.File(pose_file, 'r') as f:
    bbox_dataset = f['/poseest/bbox']
    # Extract frame range for specific identity
    bbox_data = bbox_dataset[start:end+1, identity_id, :, :]
    # Shape: (n_frames, 2, 2)
    # [frame, point_index, coordinate]
```

### Missing Data Handling

Missing bounding boxes are common in pose estimation data due to:
- Occlusion (animals blocking each other)
- Poor lighting conditions
- Animal out of frame
- Tracking failures

The script:
- **Skips** frames with missing bounding boxes (marked with -1)
- **Does not interpolate** missing frames
- **Plots only valid data points**

This means trajectories may have gaps if bounding boxes are missing for some frames within a bout.

### Coordinate System Consistency

All plots use the **same axis limits** based on maximum cage dimensions:
- Enables direct visual comparison between trajectories
- Maintains spatial relationships
- Preserves aspect ratio matching cage proportions

---

## Interpretation Guide

### Reading Trajectory Plots

1. **Position:** Each point shows where the animal's centroid was in that frame
2. **Time progression:** Color indicates when in the bout (dark = early, bright = late)
3. **Movement direction:** Follow the color gradient to see movement direction
4. **Speed:** Closer points = slower movement, farther points = faster movement
5. **Patterns:** Similar trajectory shapes within a cluster suggest similar movement patterns

### Common Patterns

- **Straight lines:** Linear movement
- **Curves:** Turning or curved paths
- **Loops:** Circular or returning movements
- **Clusters of points:** Stationary periods
- **Gaps:** Missing bounding box data (frames skipped)

### Comparing Clusters

- **Cluster overlay plots:** Show overall trajectory patterns for each cluster
- **Individual PDFs:** Allow detailed examination of specific bouts
- **Consistent scaling:** Enables direct comparison of trajectory sizes and positions

---

## Limitations

1. **Missing data:** Frames with missing bounding boxes are skipped (not interpolated)
2. **Centroid approximation:** Uses bounding box center, not actual body center
3. **2D projection:** Shows top-down view only (no depth information)
4. **Frame-based:** Discrete points at video frame rate (not continuous)

---

## References

- **Pose estimation files:** `*_pose_est_v8.h5` (HDF5 format)
- **Cluster assignments:** `cluster_assignments_{method}.csv`
- **Viridis colormap:** [Matplotlib documentation](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
- **Coordinate system:** Image coordinates (0,0 at top-left)

---

**Related Documentation:**
- `USER_GUIDE.md` - General usage guide
- `TECHNICAL_GUIDE.md` - Technical implementation details
- `ANALYSIS_REPORT.md` - Analysis results and interpretation


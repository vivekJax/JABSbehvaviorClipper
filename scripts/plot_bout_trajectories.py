#!/usr/bin/env python3
"""
Plot bounding box centroid trajectories for behavior bouts, colored by cluster.

This script:
1. Loads cluster assignments (B-SOID or hierarchical)
2. Extracts bounding box centroids from pose estimation HDF5 files
3. Calculates centroids as: (x1 + x2) / 2, (y1 + y2) / 2
4. Plots trajectories with time-based coloring (viridis colormap)
5. Generates cluster overlay plots and individual bout PDFs

Centroid Calculation:
- Bounding boxes stored in HDF5: /poseest/bbox
- Shape: (n_frames, n_identities, 2, 2)
- Point 0: top-left (x1, y1), Point 1: bottom-right (x2, y2)
- Centroid = ((x1+x2)/2, (y1+y2)/2)
- Invalid boxes (marked with -1) are filtered out

Coordinate System:
- Image coordinates: (0,0) at top-left
- X-axis: 0 (left) to max_width (right)
- Y-axis: 0 (top) to max_height (bottom) - INVERTED for display
- All plots use identical scaling for comparison

Usage:
    python3 scripts/plot_bout_trajectories.py --behavior turn_left --output-dir BoutResults
    
For detailed documentation, see: BoutAnalysisScripts/docs/TRAJECTORY_PLOTTING.md
"""

import json
import argparse
import os
import sys
import subprocess
import re
import pandas as pd
import numpy as np
# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_pose_file(video_name: str, video_dir: str) -> Optional[str]:
    """Find the pose estimation H5 file for a given video.
    
    Args:
        video_name: Name of the video file
        video_dir: Directory containing video and pose files
        
    Returns:
        Path to pose estimation file, or None if not found
    """
    # Expecting <video_name_stem>_pose_est_v8.h5
    basename = os.path.splitext(video_name)[0]
    pose_name = f"{basename}_pose_est_v8.h5"
    pose_path = os.path.join(video_dir, pose_name)
    
    if os.path.exists(pose_path):
        return pose_path
    
    # Check current directory if video_dir didn't have it
    if os.path.exists(pose_name):
        return pose_name
        
    return None


def get_cage_dimensions(pose_file: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Get cage dimensions (max x, max y) from pose estimation file.
    
    Args:
        pose_file: Path to pose estimation H5 file
        
    Returns:
        Tuple of (max_x, max_y) representing cage dimensions, or (None, None) if unavailable
    """
    if not pose_file or not os.path.exists(pose_file):
        return None, None
    
    try:
        with h5py.File(pose_file, 'r') as f:
            if '/poseest/bbox' not in f:
                return None, None
            
            bbox_dataset = f['/poseest/bbox']
            # Get max x and y from all bounding boxes (point 1 is bottom-right, has max coords)
            # Shape: (n_frames, n_identities, 2, 2) where last dim is [point, coord]
            # Point 1 (index 1) is bottom-right corner with max x and y
            # Filter out invalid values (typically -1 or negative values)
            x_coords = bbox_dataset[:, :, 1, 0][:]  # All x coordinates from bottom-right corners
            y_coords = bbox_dataset[:, :, 1, 1][:]  # All y coordinates from bottom-right corners
            
            # Filter out invalid values (negative or zero typically indicates missing data)
            valid_x = x_coords[x_coords > 0]
            valid_y = y_coords[y_coords > 0]
            
            if len(valid_x) == 0 or len(valid_y) == 0:
                return None, None
            
            max_x = float(np.max(valid_x))
            max_y = float(np.max(valid_y))
            
            return max_x, max_y
    except Exception as e:
        print(f"Warning: Could not determine cage dimensions from {pose_file}: {e}")
        return None, None


def get_lixit_location(video_name: str, video_dir: str) -> Optional[Tuple[float, float]]:
    """Get lixit location (x, y) from pose estimation HDF5 file.
    
    The lixit location is stored in the pose estimation file at /static_objects/lixit
    with shape (1, 3, 2) where we take the first prediction [0, 0, :].
    NOTE: Like keypoints, lixit is stored as [y, x] not [x, y]!
    
    Args:
        video_name: Name of video file
        video_dir: Directory containing video and pose files
        
    Returns:
        Tuple of (lixit_x, lixit_y) or None if not found
    """
    # Find pose estimation file
    pose_file = get_pose_file(video_name, video_dir)
    if not pose_file or not os.path.exists(pose_file):
        return None
    
    try:
        with h5py.File(pose_file, 'r') as f:
            if '/static_objects/lixit' in f:
                lixit_data = f['/static_objects/lixit'][:]  # Shape: (1, 3, 2)
                # Take first element, first prediction
                # NOTE: HDF5 stores as [y, x], so swap to get (x, y)
                if lixit_data.shape == (1, 3, 2):
                    y_raw = float(lixit_data[0, 0, 0])
                    x_raw = float(lixit_data[0, 0, 1])
                    x, y = x_raw, y_raw  # Swap to get correct (x, y)
                    # Validate coordinates are reasonable
                    if x > 0 and y > 0 and np.isfinite(x) and np.isfinite(y):
                        return (x, y)
                elif len(lixit_data.shape) >= 2:
                    # Try to extract from whatever shape we have
                    # Still need to swap y, x -> x, y
                    flat_data = lixit_data.flatten()
                    if len(flat_data) >= 2:
                        y_raw, x_raw = float(flat_data[0]), float(flat_data[1])
                        x, y = x_raw, y_raw  # Swap to get correct (x, y)
                        if x > 0 and y > 0 and np.isfinite(x) and np.isfinite(y):
                            return (x, y)
    except Exception as e:
        # Silently fail - lixit location is optional
        pass
    
    return None


def extract_keypoint(pose_file: Optional[str], identity_id: int,
                     start_frame: int, end_frame: int, 
                     keypoint_idx: int) -> List[Tuple[int, float, float]]:
    """Extract keypoint positions from pose estimation H5 file.
    
    Args:
        pose_file: Path to pose estimation H5 file, or None
        identity_id: Identity ID (integer index) to extract keypoints for
        start_frame: Starting frame number
        end_frame: Ending frame number
        keypoint_idx: Index of keypoint to extract
        
    Returns:
        List of tuples: [(frame, x, y), ...]
    """
    if not pose_file or not os.path.exists(pose_file):
        return []
    
    try:
        with h5py.File(pose_file, 'r') as f:
            if '/poseest/points' not in f:
                return []
            
            points_dataset = f['/poseest/points']
            
            # Points dataset shape: (n_frames, n_identities, n_keypoints, 2)
            # Where: [frame, identity, keypoint_index, coordinate]
            # coordinate 0 = x, 1 = y
            
            # Clamp frame range to valid indices
            max_frame = points_dataset.shape[0] - 1
            start = min(start_frame, max_frame)
            end = min(end_frame, max_frame)
            
            if start > end:
                return []
            
            # Ensure identity_id is an integer and within valid range
            identity_id = int(identity_id)
            if identity_id < 0 or identity_id >= points_dataset.shape[1]:
                return []
            
            # Check if keypoint index is valid
            if keypoint_idx >= points_dataset.shape[2]:
                return []
            
            # Get data for the frame range: shape (end-start+1, 1, 1, 2)
            points_data = points_dataset[start:end + 1, identity_id, keypoint_idx, :]
            
            # Extract keypoint positions
            # NOTE: HDF5 stores keypoints as [y, x] not [x, y]!
            keypoint_positions = []
            for i, frame_num in enumerate(range(start, end + 1)):
                # Keypoints stored as [y, x] in HDF5, swap to get (x, y)
                y_raw, x_raw = float(points_data[i, 0]), float(points_data[i, 1])
                x, y = x_raw, y_raw
                
                # Check for valid keypoint:
                # - Must be positive (0 or negative typically indicates missing data)
                # - Must be finite (not NaN or infinite)
                is_valid = (
                    np.isfinite(x) and np.isfinite(y) and
                    x > 0 and y > 0 and  # Both coordinates must be positive
                    not (x == 0 and y == 0)  # Not at origin (missing data)
                )
                
                if is_valid:
                    keypoint_positions.append((frame_num, x, y))
            
            return keypoint_positions
            
    except Exception as e:
        return []


def extract_nose_keypoints(pose_file: Optional[str], identity_id: int,
                           start_frame: int, end_frame: int, 
                           nose_keypoint_idx: int = 0) -> List[Tuple[int, float, float]]:
    """Extract nose keypoint positions from pose estimation H5 file.
    
    Args:
        pose_file: Path to pose estimation H5 file, or None
        identity_id: Identity ID (integer index) to extract keypoints for
        start_frame: Starting frame number
        end_frame: Ending frame number
        nose_keypoint_idx: Index of nose keypoint (default: 0)
        
    Returns:
        List of tuples: [(frame, nose_x, nose_y), ...]
    """
    return extract_keypoint(pose_file, identity_id, start_frame, end_frame, nose_keypoint_idx)


def find_correct_identity_index(pose_file: str, expected_identity: int, 
                                start_frame: int, end_frame: int) -> Optional[int]:
    """Find the correct HDF5 identity index by checking which identity's keypoints are inside the bbox.
    
    Args:
        pose_file: Path to pose estimation H5 file
        expected_identity: Expected identity ID from CSV
        start_frame: Starting frame number
        end_frame: Ending frame number
        
    Returns:
        The correct identity index, or expected_identity if no better match found
    """
    try:
        with h5py.File(pose_file, 'r') as f:
            if '/poseest/bbox' not in f or '/poseest/points' not in f:
                return expected_identity
            
            bbox_dataset = f['/poseest/bbox']
            points_dataset = f['/poseest/points']
            
            max_frame = min(bbox_dataset.shape[0] - 1, points_dataset.shape[0] - 1)
            start = min(start_frame, max_frame)
            end = min(end_frame, max_frame)
            
            if start > end or expected_identity >= bbox_dataset.shape[1]:
                return expected_identity
            
            # Check a few frames to find the best matching identity
            test_frames = [start, (start + end) // 2, end] if end > start else [start]
            best_match_count = 0
            best_identity = expected_identity
            
            for test_frame in test_frames:
                # Get bbox for expected identity
                bbox_data = bbox_dataset[test_frame, expected_identity, :, :]
                x1, y1 = float(bbox_data[0, 0]), float(bbox_data[0, 1])
                x2, y2 = float(bbox_data[1, 0]), float(bbox_data[1, 1])
                
                # Skip if bbox is invalid
                if x1 <= 0 or y1 <= 0 or x2 <= x1 or y2 <= y1:
                    continue
                
                # Check all identities to see which nose is inside the bbox
                for test_id in range(points_dataset.shape[1]):
                    nose_data = points_dataset[test_frame, test_id, 0, :]
                    nose_x, nose_y = float(nose_data[0]), float(nose_data[1])
                    
                    if nose_x > 0 and nose_y > 0:
                        nose_in_bbox = (x1 <= nose_x <= x2) and (y1 <= nose_y <= y2)
                        if nose_in_bbox:
                            # Count how many times this identity matches
                            match_count = sum(1 for f in test_frames 
                                            if f < points_dataset.shape[0] and
                                            f < bbox_dataset.shape[0])
                            if match_count > best_match_count:
                                best_match_count = match_count
                                best_identity = test_id
            
            # Only return different identity if we found a clear match
            if best_match_count > 0 and best_identity != expected_identity:
                return best_identity
            
            return expected_identity
            
    except Exception:
        return expected_identity


def extract_bbox_centroids(pose_file: Optional[str], identity_id: int, 
                          start_frame: int, end_frame: int) -> List[Tuple[int, float, float]]:
    """Extract bounding box centroids from pose estimation H5 file.
    
    Args:
        pose_file: Path to pose estimation H5 file, or None
        identity_id: Identity ID (integer index) to extract boxes for
        start_frame: Starting frame number
        end_frame: Ending frame number
        
    Returns:
        List of tuples: [(frame, centroid_x, centroid_y), ...]
    """
    if not pose_file or not os.path.exists(pose_file):
        return []
    
    try:
        # Try using h5py first (more portable)
        with h5py.File(pose_file, 'r') as f:
            if '/poseest/bbox' not in f:
                return []
            
            bbox_dataset = f['/poseest/bbox']
            
            # Bounding box dataset shape: (n_frames, n_identities, 2, 2)
            # Where: [frame, identity, point_index, coordinate]
            # point_index 0 = top-left (x1, y1), 1 = bottom-right (x2, y2)
            # coordinate 0 = x, 1 = y
            
            # Clamp frame range to valid indices
            max_frame = bbox_dataset.shape[0] - 1
            start = min(start_frame, max_frame)
            end = min(end_frame, max_frame)
            
            if start > end:
                return []
            
            # Ensure identity_id is an integer and within valid range
            identity_id = int(identity_id)
            if identity_id < 0 or identity_id >= bbox_dataset.shape[1]:
                return []
            
            # Identity correction is done at a higher level (in main()) to ensure
            # consistent identity usage across bbox, nose, and tail extractions
            # The identity_id passed here should already be the corrected identity
            
            # Get data for the frame range: shape (end-start+1, 1, 2, 2)
            bbox_data = bbox_dataset[start:end + 1, identity_id, :, :]
            
            # Calculate centroids
            centroids = []
            for i, frame_num in enumerate(range(start, end + 1)):
                # bbox_data[i] shape: (2, 2) - [point_index, coordinate]
                # point 0: (x1, y1), point 1: (x2, y2)
                x1, y1 = bbox_data[i, 0, 0], bbox_data[i, 0, 1]
                x2, y2 = bbox_data[i, 1, 0], bbox_data[i, 1, 1]
                
                # Check for invalid bounding boxes:
                # - Negative values (typically -1 indicates missing data)
                # - NaN or infinite values
                # - All zeros (missing bounding box)
                # - Invalid bounding box (x1 >= x2 or y1 >= y2 would be invalid)
                # - Very small bounding boxes (likely noise/missing data)
                is_valid = (
                    np.isfinite(x1) and np.isfinite(y1) and 
                    np.isfinite(x2) and np.isfinite(y2) and
                    x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and  # All coordinates must be positive (not 0, not negative)
                    not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0) and  # Not all zeros
                    x2 > x1 and y2 > y1 and  # Valid bounding box (bottom-right > top-left)
                    (x2 - x1) > 1 and (y2 - y1) > 1  # Bounding box must have some size (at least 1 pixel)
                )
                
                if is_valid:
                    centroid_x = (x1 + x2) / 2.0
                    centroid_y = (y1 + y2) / 2.0
                    # Additional check: centroid should be within reasonable bounds and not at origin
                    # Skip (0,0) points which indicate missing data
                    if centroid_x > 0 and centroid_y > 0 and not (centroid_x == 0 and centroid_y == 0):
                        centroids.append((frame_num, centroid_x, centroid_y))
            
            return centroids
            
    except Exception as e:
        # Fallback to h5dump if h5py fails
        try:
            count = end_frame - start_frame + 1
            start_idx = f"{start_frame},{identity_id},0,0"
            count_idx = f"{count},1,2,2"
            
            cmd = ['h5dump', '-d', '/poseest/bbox', '-s', start_idx, '-c', count_idx, pose_file]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                return []
            
            # Parse output
            bboxes = {}  # frame -> {p0: (x,y), p1: (x,y)}
            lines = result.stdout.splitlines()
            pattern = re.compile(r'^\s*\((\d+),\d+,(\d+),0\):\s*([-\d\.]+),\s*([-\d\.]+)')
            
            for line in lines:
                match = pattern.search(line)
                if match:
                    frame = int(match.group(1))
                    point = int(match.group(2))  # 0 or 1
                    x = float(match.group(3))
                    y = float(match.group(4))
                    
                    if frame not in bboxes:
                        bboxes[frame] = {}
                    
                    bboxes[frame][point] = (x, y)
            
            # Calculate centroids
            centroids = []
            for frame in sorted(bboxes.keys()):
                pts = bboxes[frame]
                if 0 in pts and 1 in pts:
                    x1, y1 = pts[0]
                    x2, y2 = pts[1]
                    
                    # Check for invalid bounding boxes (same checks as h5py version)
                    is_valid = (
                        np.isfinite(x1) and np.isfinite(y1) and 
                        np.isfinite(x2) and np.isfinite(y2) and
                        x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and  # All coordinates must be positive
                        not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0) and  # Not all zeros
                        x2 > x1 and y2 > y1 and  # Valid bounding box
                        (x2 - x1) > 1 and (y2 - y1) > 1  # Bounding box must have some size
                    )
                    
                    if is_valid:
                        centroid_x = (x1 + x2) / 2.0
                        centroid_y = (y1 + y2) / 2.0
                        # Skip (0,0) points which indicate missing data
                        if centroid_x > 0 and centroid_y > 0 and not (centroid_x == 0 and centroid_y == 0):
                            centroids.append((frame, centroid_x, centroid_y))
            
            return centroids
            
        except FileNotFoundError:
            print(f"Warning: h5dump command not found and h5py failed. Install HDF5 tools.")
            return []
        except Exception as e2:
            print(f"Warning: Error reading bboxes from {pose_file}: {e2}")
            return []


def get_cluster_color(cluster_id: int, n_clusters: int) -> Tuple[float, float, float]:
    """Get a color for a cluster ID.
    
    Args:
        cluster_id: Cluster ID (0 = noise, 1+ = clusters)
        n_clusters: Total number of clusters (excluding noise)
        
    Returns:
        RGB tuple (0-1 range)
    """
    if cluster_id == 0:  # Noise cluster
        return (0.5, 0.5, 0.5)  # Gray
    
    # Use a colormap for clusters
    try:
        cmap = cm.get_cmap('tab10')
    except AttributeError:
        # For newer matplotlib versions
        cmap = plt.cm.get_cmap('tab10')
    # Map cluster_id (1, 2, 3, ...) to colormap index (0, 1, 2, ...)
    color_idx = (cluster_id - 1) % 10
    return cmap(color_idx)[:3]


def plot_combined_trajectories_pdf(all_trajectories_by_cluster: Dict[int, List[Dict]],
                                   output_dir: str, cage_max_x: float, cage_max_y: float,
                                   video_dir: str = '..'):
    """Create individual PDF files with one bout per PDF, showing centroid, nose, and tail.
    
    Args:
        all_trajectories_by_cluster: Dict mapping cluster_id to list of trajectory dicts
        output_dir: Directory to save individual PDF files
        cage_max_x: Maximum x dimension (for consistent scaling)
        cage_max_y: Maximum y dimension (for consistent scaling)
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    centroid_cmap = plt.cm.get_cmap('viridis')
    nose_cmap = plt.cm.get_cmap('cool')
    tail_cmap = plt.cm.get_cmap('copper')
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_plots = 0
    
    for cluster_id in sorted(all_trajectories_by_cluster.keys()):
        cluster_trajectories = all_trajectories_by_cluster[cluster_id]
        
        if not cluster_trajectories:
            continue
        
        print(f"  Creating combined plots for cluster {cluster_id} ({len(cluster_trajectories)} bouts)...")
        
        for traj_data in cluster_trajectories:
            centroids = traj_data.get('centroids', [])
            nose_points = traj_data.get('nose_points', [])
            tail_points = traj_data.get('tail_points', [])
            bout_id = traj_data['bout_id']
            video_name = traj_data['video_name']
            animal_id = traj_data.get('animal_id', 'N/A')
            
            # Need at least one trajectory type
            if not centroids and not nose_points:
                continue
            
            # Create a new figure for each bout
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Determine frame range from all available trajectories
            all_frames = []
            if centroids:
                all_frames.extend([c[0] for c in centroids])
            if nose_points:
                all_frames.extend([n[0] for n in nose_points])
            if tail_points:
                all_frames.extend([t[0] for t in tail_points])
            
            if not all_frames:
                plt.close()
                continue
            
            frame_min = min(all_frames)
            frame_max = max(all_frames)
            
            # Plot centroid trajectory (viridis)
            if centroids:
                cent_frames = [c[0] for c in centroids]
                cent_x = [c[1] for c in centroids]
                cent_y = [c[2] for c in centroids]
                cent_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in cent_frames]
                
                # Plot centroid line segments
                for i in range(len(cent_x) - 1):
                    ax.plot([cent_x[i], cent_x[i+1]], [cent_y[i], cent_y[i+1]], 
                           color=centroid_cmap(cent_frame_norm[i]), linewidth=2, alpha=0.8, zorder=1, label='Centroid' if i == 0 else '')
                
                # Plot centroid points
                ax.scatter(cent_x, cent_y, c=cent_frame_norm, cmap='viridis', 
                          s=50, alpha=0.8, zorder=2, edgecolors='white', linewidths=0.5)
            
            # Plot nose trajectory (cool)
            if nose_points:
                nose_frames = [n[0] for n in nose_points]
                nose_x = [n[1] for n in nose_points]
                nose_y = [n[2] for n in nose_points]
                nose_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in nose_frames]
                
                # Plot nose line segments
                for i in range(len(nose_x) - 1):
                    ax.plot([nose_x[i], nose_x[i+1]], [nose_y[i], nose_y[i+1]], 
                           color=nose_cmap(nose_frame_norm[i]), linewidth=2, alpha=0.8, zorder=1, label='Nose' if i == 0 and not centroids else '')
                
                # Plot nose points
                ax.scatter(nose_x, nose_y, c=nose_frame_norm, cmap='cool', 
                          s=50, alpha=0.8, zorder=2, edgecolors='white', linewidths=0.5)
            
            # Plot tail trajectory (copper) and connect to nose
            if tail_points:
                tail_frames = [t[0] for t in tail_points]
                tail_x = [t[1] for t in tail_points]
                tail_y = [t[2] for t in tail_points]
                tail_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in tail_frames]
                
                # Plot tail line segments
                for i in range(len(tail_x) - 1):
                    ax.plot([tail_x[i], tail_x[i+1]], [tail_y[i], tail_y[i+1]], 
                           color=tail_cmap(tail_frame_norm[i]), linewidth=2, alpha=0.8, zorder=1, label='Tail' if i == 0 and not centroids and not nose_points else '')
                
                # Plot tail points
                ax.scatter(tail_x, tail_y, c=tail_frame_norm, cmap='copper', 
                          s=50, alpha=0.8, zorder=2, edgecolors='white', linewidths=0.5)
                
                # Draw grey lines connecting nose to tail for each frame
                if nose_points:
                    nose_dict = {n[0]: (n[1], n[2]) for n in nose_points}
                    tail_dict = {t[0]: (t[1], t[2]) for t in tail_points}
                    
                    # Find common frames
                    common_frames = set(nose_dict.keys()) & set(tail_dict.keys())
                    for frame_num in common_frames:
                        nose_x, nose_y = nose_dict[frame_num]
                        tail_x, tail_y = tail_dict[frame_num]
                        ax.plot([nose_x, tail_x], [nose_y, tail_y], 
                               color='gray', linewidth=1.0, alpha=0.5, zorder=0)
            
            # Set axis limits (consistent across all plots)
            ax.set_xlim(0, cage_max_x)
            ax.set_ylim(cage_max_y, 0)  # Inverted: max at bottom, 0 at top
            ax.set_aspect(cage_max_y / cage_max_x, adjustable='box')
            
            # Add colorbars (smaller size)
            # Centroid colorbar
            if centroids:
                sm_cent = plt.cm.ScalarMappable(cmap=centroid_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                sm_cent.set_array([])
                cbar_cent = plt.colorbar(sm_cent, ax=ax, label='Centroid Time', pad=0.02, location='right', shrink=0.4, aspect=20)
                cbar_cent.set_label('Centroid Time (0=early, 1=late)', fontsize=8)
                cbar_cent.ax.tick_params(labelsize=7)
            
            # Nose colorbar
            if nose_points:
                sm_nose = plt.cm.ScalarMappable(cmap=nose_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                sm_nose.set_array([])
                pad_nose = 0.15 if centroids else 0.02
                cbar_nose = plt.colorbar(sm_nose, ax=ax, label='Nose Time', pad=pad_nose, location='right', shrink=0.4, aspect=20)
                cbar_nose.set_label('Nose Time (0=early, 1=late)', fontsize=8)
                cbar_nose.ax.tick_params(labelsize=7)
            
            # Tail colorbar
            if tail_points:
                sm_tail = plt.cm.ScalarMappable(cmap=tail_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                sm_tail.set_array([])
                pad_tail = 0.28 if centroids and nose_points else (0.15 if (centroids or nose_points) else 0.02)
                cbar_tail = plt.colorbar(sm_tail, ax=ax, label='Tail Time', pad=pad_tail, location='right', shrink=0.4, aspect=20)
                cbar_tail.set_label('Tail Time (0=early, 1=late)', fontsize=8)
                cbar_tail.ax.tick_params(labelsize=7)
            
            # Add lixit location as red star (optional, skip if it causes issues)
            try:
                lixit_pos = get_lixit_location(video_name, video_dir)
                if lixit_pos:
                    lixit_x, lixit_y = lixit_pos
                    ax.plot(lixit_x, lixit_y, 'r*', markersize=10, markeredgecolor='black', markeredgewidth=1, zorder=10, label='Lixit Location')
            except Exception:
                # Skip lixit location if there's any error
                pass
            
            # Labels and title
            ax.set_xlabel('X Position (pixels)', fontsize=12, fontweight='medium')
            ax.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='medium')
            
            # Extract just the filename from video_name
            video_short = os.path.basename(video_name) if os.path.sep in video_name else video_name
            
            ax.set_title(f'Bout {bout_id} - Cluster {cluster_id}\n'
                        f'Video: {video_short}\n'
                        f'Animal ID: {animal_id} | Frames: {frame_min}-{frame_max} ({len(all_frames)} frames)',
                        fontsize=13, fontweight='bold', pad=15)
            
            # Styling
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.tick_params(labelsize=10)
            
            # Add explanation text below the plot (outside the axes)
            explanation = (
                f"Combined trajectory: Centroid (viridis), Nose (cool), Tail (copper). "
                f"Cage dimensions: {cage_max_x:.0f} x {cage_max_y:.0f} pixels. "
                f"Grey lines connect nose to tail for each frame. Red star indicates lixit location."
            )
            fig.text(0.5, 0.01, explanation, ha='center', va='bottom', 
                   fontsize=8, wrap=True,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
            
            # Save as individual PDF
            output_filename = f"bout_{bout_id}_cluster_{cluster_id}_combined.pdf"
            output_file = output_path / output_filename
            plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Leave space at bottom for explanation
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            total_plots += 1
    
    print(f"  Created {total_plots} combined trajectory PDFs in {output_dir}")


def plot_individual_bouts_pdf(all_trajectories_by_cluster: Dict[int, List[Dict]],
                               output_path: str, cage_max_x: float, cage_max_y: float,
                               colormap_name: str = 'viridis', point_type: str = 'centroid',
                               video_dir: str = '..'):
    """Create a multi-page PDF with one bout trajectory per page.
    
    Args:
        all_trajectories_by_cluster: Dict mapping cluster_id to list of trajectory dicts
        output_path: Path for output PDF file
        cage_max_x: Maximum x dimension (for consistent scaling)
        cage_max_y: Maximum y dimension (for consistent scaling)
        colormap_name: Name of colormap to use (default: 'viridis')
        point_type: Type of point being plotted ('centroid' or 'nose')
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    colormap = plt.cm.get_cmap(colormap_name)
    
    with PdfPages(output_path) as pdf:
        for cluster_id in sorted(all_trajectories_by_cluster.keys()):
            cluster_trajectories = all_trajectories_by_cluster[cluster_id]
            
            if not cluster_trajectories:
                continue
            
            print(f"  Creating PDF pages for cluster {cluster_id} ({len(cluster_trajectories)} bouts)...")
            
            for traj_data in cluster_trajectories:
                centroids = traj_data['centroids']
                bout_id = traj_data['bout_id']
                video_name = traj_data['video_name']
                animal_id = traj_data.get('animal_id', 'N/A')
                
                if not centroids or len(centroids) < 2:
                    continue
                
                # Create a new figure for each bout
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Extract data
                frames = [c[0] for c in centroids]
                x_coords = [c[1] for c in centroids]
                y_coords = [c[2] for c in centroids]
                
                # Normalize frame numbers to [0, 1] for colormap
                if len(frames) > 1:
                    frame_min = min(frames)
                    frame_max = max(frames)
                    frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in frames]
                else:
                    continue
                
                # For nose plots, also extract tail keypoints and draw connection lines
                if point_type == 'nose' and 'tail_points' in traj_data:
                    tail_points = traj_data['tail_points']
                    if tail_points:
                        # Create a dict for quick lookup: frame -> (x, y)
                        tail_dict = {tp[0]: (tp[1], tp[2]) for tp in tail_points}
                        
                        # Draw grey lines connecting nose to tail for each frame
                        for i, frame_num in enumerate(frames):
                            if frame_num in tail_dict:
                                nose_x, nose_y = x_coords[i], y_coords[i]
                                tail_x, tail_y = tail_dict[frame_num]
                                ax.plot([nose_x, tail_x], [nose_y, tail_y], 
                                       color='gray', linewidth=1.0, alpha=0.5, zorder=0)
                        
                        # Plot tail trajectory with copper colormap
                        tail_frames = [tp[0] for tp in tail_points]
                        tail_x_coords = [tp[1] for tp in tail_points]
                        tail_y_coords = [tp[2] for tp in tail_points]
                        
                        if len(tail_frames) > 1:
                            tail_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in tail_frames]
                            
                            # Plot tail line segments
                            tail_colormap = plt.cm.get_cmap('copper')
                            for i in range(len(tail_x_coords) - 1):
                                ax.plot([tail_x_coords[i], tail_x_coords[i+1]], 
                                       [tail_y_coords[i], tail_y_coords[i+1]], 
                                       color=tail_colormap(tail_frame_norm[i]), linewidth=2, alpha=0.8, zorder=1)
                            
                            # Plot tail points
                            ax.scatter(tail_x_coords, tail_y_coords, c=tail_frame_norm, cmap='copper', 
                                      s=40, alpha=0.8, zorder=2, edgecolors='white', linewidths=0.5)
                
                # Plot nose/centroid line segments with colors based on time
                for i in range(len(x_coords) - 1):
                    ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                           color=colormap(frame_norm[i]), linewidth=2, alpha=0.8, zorder=1)
                
                # Plot nose/centroid points colored by time
                scatter = ax.scatter(x_coords, y_coords, c=frame_norm, cmap=colormap_name, 
                                    s=40, alpha=0.8, zorder=2, edgecolors='white', linewidths=0.5)
                
                # Set axis limits (consistent across all plots)
                # Invert y-axis so 0 is at top (image coordinates)
                ax.set_xlim(0, cage_max_x)
                ax.set_ylim(cage_max_y, 0)  # Inverted: max at bottom, 0 at top
                ax.set_aspect(cage_max_y / cage_max_x, adjustable='box')
                
                # Add colorbar(s)
                if point_type == 'nose' and 'tail_points' in traj_data and traj_data['tail_points']:
                    # Add two colorbars for nose and tail
                    cbar_nose = plt.colorbar(scatter, ax=ax, label='Nose Time', pad=0.02, location='right', shrink=0.4, aspect=20)
                    cbar_nose.set_label('Nose Time (0=early, 1=late)', fontsize=9)
                    cbar_nose.ax.tick_params(labelsize=8)
                else:
                    # Single colorbar for centroid plots
                    cbar = plt.colorbar(scatter, ax=ax, label='Time (normalized)', pad=0.02, shrink=0.4, aspect=20)
                    cbar.set_label('Time (0=early, 1=late)', fontsize=10)
                    cbar.ax.tick_params(labelsize=9)
                
                # Add lixit location as red star (optional, skip if it causes issues)
                try:
                    lixit_pos = get_lixit_location(video_name, video_dir)
                    if lixit_pos:
                        lixit_x, lixit_y = lixit_pos
                        ax.plot(lixit_x, lixit_y, 'r*', markersize=10, markeredgecolor='black', markeredgewidth=1, zorder=10, label='Lixit Location')
                except Exception:
                    # Skip lixit location if there's any error
                    pass
                
                # Labels and title
                ax.set_xlabel('X Position (pixels)', fontsize=12, fontweight='medium')
                ax.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='medium')
                
                # Extract just the filename from video_name
                video_short = os.path.basename(video_name) if os.path.sep in video_name else video_name
                
                ax.set_title(f'Bout {bout_id} - Cluster {cluster_id}\n'
                            f'Video: {video_short}\n'
                            f'Animal ID: {animal_id} | Frames: {frame_min}-{frame_max} ({len(frames)} frames)',
                            fontsize=13, fontweight='bold', pad=15)
                
                # Styling
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.8)
                ax.spines['bottom'].set_linewidth(0.8)
                ax.tick_params(labelsize=10)
                
                # Add explanation text below the plot (outside the axes)
                if point_type == 'nose' and 'tail_points' in traj_data and traj_data['tail_points']:
                    explanation = (
                        f"Trajectory of nose and tail keypoints over time. "
                        f"Cage dimensions: {cage_max_x:.0f} x {cage_max_y:.0f} pixels. "
                        f"Nose: cool colormap (cyan→magenta). Tail: copper colormap (dark→bright). "
                        f"Grey lines connect nose to tail for each frame. Red star indicates lixit location."
                    )
                else:
                    point_desc = "bounding box centroid"
                    explanation = (
                        f"Trajectory of {point_desc} over time. "
                        f"Cage dimensions: {cage_max_x:.0f} x {cage_max_y:.0f} pixels. "
                        f"Color indicates time progression (early → late). Red star indicates lixit location."
                    )
                fig.text(0.5, 0.01, explanation, ha='center', va='bottom', 
                       fontsize=8, wrap=True,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Leave space at bottom for explanation
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    print(f"  Saved multi-page PDF: {output_path}")


def plot_combined_individual_bouts_pdf(all_trajectories_by_cluster: Dict[int, List[Dict]],
                                        output_path: str, cage_max_x: float, cage_max_y: float,
                                        video_dir: str = '..'):
    """Create a multi-page PDF with one bout per page, showing all three trajectories (centroid + nose + tail).
    
    Args:
        all_trajectories_by_cluster: Dict mapping cluster_id to list of trajectory dicts
        output_path: Path for output PDF file
        cage_max_x: Maximum x dimension (for consistent scaling)
        cage_max_y: Maximum y dimension (for consistent scaling)
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    centroid_cmap = plt.cm.get_cmap('viridis')
    nose_cmap = plt.cm.get_cmap('cool')
    tail_cmap = plt.cm.get_cmap('copper')
    
    with PdfPages(output_path) as pdf:
        for cluster_id in sorted(all_trajectories_by_cluster.keys()):
            cluster_trajectories = all_trajectories_by_cluster[cluster_id]
            
            if not cluster_trajectories:
                continue
            
            print(f"  Creating combined PDF pages for cluster {cluster_id} ({len(cluster_trajectories)} bouts)...")
            
            for traj_data in cluster_trajectories:
                centroids = traj_data.get('centroids', [])
                nose_points = traj_data.get('nose_points', [])
                tail_points = traj_data.get('tail_points', [])
                bout_id = traj_data['bout_id']
                video_name = traj_data['video_name']
                animal_id = traj_data.get('animal_id', 'N/A')
                start_frame = traj_data.get('start_frame', 'N/A')
                end_frame = traj_data.get('end_frame', 'N/A')
                
                # Need at least one trajectory type
                if not centroids and not nose_points:
                    continue
                
                # Create a new figure for each bout
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Determine frame range from all available trajectories
                all_frames = []
                if centroids:
                    all_frames.extend([c[0] for c in centroids])
                if nose_points:
                    all_frames.extend([n[0] for n in nose_points])
                if tail_points:
                    all_frames.extend([t[0] for t in tail_points])
                
                if not all_frames:
                    plt.close()
                    continue
                
                frame_min = min(all_frames)
                frame_max = max(all_frames)
                
                # Plot centroid trajectory (viridis)
                if centroids:
                    cent_frames = [c[0] for c in centroids]
                    cent_x = [c[1] for c in centroids]
                    cent_y = [c[2] for c in centroids]
                    cent_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in cent_frames]
                    
                    # Plot centroid line segments
                    for i in range(len(cent_x) - 1):
                        ax.plot([cent_x[i], cent_x[i+1]], [cent_y[i], cent_y[i+1]], 
                               color=centroid_cmap(cent_frame_norm[i]), linewidth=2, alpha=0.8, zorder=1)
                    
                    # Plot centroid points
                    ax.scatter(cent_x, cent_y, c=cent_frame_norm, cmap='viridis', 
                              s=40, alpha=0.8, zorder=2, edgecolors='white', linewidths=0.5)
                
                # Plot nose trajectory (cool)
                if nose_points:
                    nose_frames = [n[0] for n in nose_points]
                    nose_x = [n[1] for n in nose_points]
                    nose_y = [n[2] for n in nose_points]
                    nose_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in nose_frames]
                    
                    # Plot nose line segments
                    for i in range(len(nose_x) - 1):
                        ax.plot([nose_x[i], nose_x[i+1]], [nose_y[i], nose_y[i+1]], 
                               color=nose_cmap(nose_frame_norm[i]), linewidth=2, alpha=0.8, zorder=1)
                    
                    # Plot nose points
                    ax.scatter(nose_x, nose_y, c=nose_frame_norm, cmap='cool', 
                              s=40, alpha=0.8, zorder=2, edgecolors='white', linewidths=0.5)
                
                # Plot tail trajectory (copper) and connect to nose
                if tail_points:
                    tail_frames = [t[0] for t in tail_points]
                    tail_x = [t[1] for t in tail_points]
                    tail_y = [t[2] for t in tail_points]
                    tail_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in tail_frames]
                    
                    # Draw grey lines connecting nose to tail for each frame
                    if nose_points:
                        nose_dict = {n[0]: (n[1], n[2]) for n in nose_points}
                        tail_dict = {t[0]: (t[1], t[2]) for t in tail_points}
                        
                        # Find common frames
                        common_frames = set(nose_dict.keys()) & set(tail_dict.keys())
                        for frame_num in common_frames:
                            nose_x_coord, nose_y_coord = nose_dict[frame_num]
                            tail_x_coord, tail_y_coord = tail_dict[frame_num]
                            ax.plot([nose_x_coord, tail_x_coord], [nose_y_coord, tail_y_coord], 
                                   color='gray', linewidth=1.0, alpha=0.5, zorder=0)
                    
                    # Plot tail line segments
                    for i in range(len(tail_x) - 1):
                        ax.plot([tail_x[i], tail_x[i+1]], [tail_y[i], tail_y[i+1]], 
                               color=tail_cmap(tail_frame_norm[i]), linewidth=2, alpha=0.8, zorder=1)
                    
                    # Plot tail points
                    ax.scatter(tail_x, tail_y, c=tail_frame_norm, cmap='copper', 
                              s=40, alpha=0.8, zorder=2, edgecolors='white', linewidths=0.5)
                
                # Set axis limits (consistent across all plots)
                # Invert y-axis so 0 is at top (image coordinates)
                ax.set_xlim(0, cage_max_x)
                ax.set_ylim(cage_max_y, 0)  # Inverted: max at bottom, 0 at top
                ax.set_aspect(cage_max_y / cage_max_x, adjustable='box')
                
                # Add single compact legend instead of multiple colorbars
                # Create a tiny colorbar indicator in bottom-right corner
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                
                # Add mini colorbars stacked horizontally at bottom
                cbar_width = "2%"
                cbar_height = "25%"
                
                if centroids:
                    axins1 = inset_axes(ax, width=cbar_width, height=cbar_height, loc='lower right',
                                       bbox_to_anchor=(0.02, 0.02, 1, 1), bbox_transform=ax.transAxes)
                    sm_cent = plt.cm.ScalarMappable(cmap=centroid_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    sm_cent.set_array([])
                    cbar_cent = plt.colorbar(sm_cent, cax=axins1)
                    cbar_cent.ax.tick_params(labelsize=4, length=1, pad=1)
                    cbar_cent.ax.set_ylabel('C', fontsize=5, rotation=0, labelpad=2)
                
                if nose_points:
                    axins2 = inset_axes(ax, width=cbar_width, height=cbar_height, loc='lower right',
                                       bbox_to_anchor=(-0.03, 0.02, 1, 1), bbox_transform=ax.transAxes)
                    sm_nose = plt.cm.ScalarMappable(cmap=nose_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    sm_nose.set_array([])
                    cbar_nose = plt.colorbar(sm_nose, cax=axins2)
                    cbar_nose.ax.tick_params(labelsize=4, length=1, pad=1)
                    cbar_nose.ax.set_ylabel('N', fontsize=5, rotation=0, labelpad=2)
                
                if tail_points:
                    axins3 = inset_axes(ax, width=cbar_width, height=cbar_height, loc='lower right',
                                       bbox_to_anchor=(-0.08, 0.02, 1, 1), bbox_transform=ax.transAxes)
                    sm_tail = plt.cm.ScalarMappable(cmap=tail_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    sm_tail.set_array([])
                    cbar_tail = plt.colorbar(sm_tail, cax=axins3)
                    cbar_tail.ax.tick_params(labelsize=4, length=1, pad=1)
                    cbar_tail.ax.set_ylabel('T', fontsize=5, rotation=0, labelpad=2)
                
                # Add lixit location as red star (optional, skip if it causes issues)
                try:
                    lixit_pos = get_lixit_location(video_name, video_dir)
                    if lixit_pos:
                        lixit_x, lixit_y = lixit_pos
                        ax.plot(lixit_x, lixit_y, 'r*', markersize=10, markeredgecolor='black', markeredgewidth=1, zorder=10, label='Lixit Location')
                except Exception:
                    # Skip lixit location if there's any error
                    pass
                
                # Labels and title (compact sizes)
                ax.set_xlabel('X (pixels)', fontsize=8)
                ax.set_ylabel('Y (pixels)', fontsize=8)
                
                # Extract just the filename from video_name
                video_short = os.path.basename(video_name) if os.path.sep in video_name else video_name
                # Truncate long video names
                if len(video_short) > 50:
                    video_short = video_short[:47] + '...'
                
                ax.set_title(f'Bout {bout_id} - Cluster {cluster_id} | {video_short}\n'
                            f'Animal: {animal_id} | Frames: {start_frame}-{end_frame} ({len(all_frames)} pts)',
                            fontsize=8, fontweight='bold', pad=5)
                
                # Styling
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.5)
                ax.spines['bottom'].set_linewidth(0.5)
                ax.tick_params(labelsize=7)
                
                # Add compact explanation text below the plot
                explanation = f"Centroid(viridis) + Nose(cool) + Tail(copper) | Grey=nose-tail | ★=Lixit"
                fig.text(0.5, 0.01, explanation, ha='center', va='bottom', 
                       fontsize=6, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.3))
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.99])  # Maximize plot area
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    print(f"  Saved combined multi-page PDF: {output_path}")


def plot_faceted_trajectories(all_trajectories_by_cluster: Dict[int, List[Dict]],
                              output_path: str, cage_max_x: float, cage_max_y: float,
                              max_bouts_per_plot: int = 20):
    """Create faceted plots with one bout per subplot, organized by cluster.
    
    Args:
        all_trajectories_by_cluster: Dict mapping cluster_id to list of trajectory dicts
        output_path: Base path for output (will create multiple files)
        cage_max_x: Maximum x dimension (for consistent scaling)
        cage_max_y: Maximum y dimension (for consistent scaling)
        max_bouts_per_plot: Maximum number of subplots per figure (default: 20)
    """
    viridis = plt.cm.get_cmap('viridis')
    
    for cluster_id, cluster_trajectories in sorted(all_trajectories_by_cluster.items()):
        if not cluster_trajectories:
            continue
        
        n_trajectories = len(cluster_trajectories)
        n_plots = (n_trajectories + max_bouts_per_plot - 1) // max_bouts_per_plot  # Ceiling division
        
        for plot_idx in range(n_plots):
            start_idx = plot_idx * max_bouts_per_plot
            end_idx = min(start_idx + max_bouts_per_plot, n_trajectories)
            trajectories_subset = cluster_trajectories[start_idx:end_idx]
            
            # Calculate grid dimensions
            n_subplots = len(trajectories_subset)
            n_cols = min(5, max(2, int(np.ceil(np.sqrt(n_subplots)))))
            n_rows = int(np.ceil(n_subplots / n_cols))
            
            # Create figure with subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            fig.suptitle(f'Cluster {cluster_id} - Individual Bout Trajectories '
                        f'({start_idx+1}-{end_idx} of {n_trajectories})',
                        fontsize=16, fontweight='bold', y=0.995)
            
            # Flatten axes array for easier indexing
            # Handle different matplotlib return types
            if n_subplots == 1:
                axes_list = [axes]
            elif n_rows == 1:
                # Single row - axes is 1D array
                axes_list = axes.tolist() if isinstance(axes, np.ndarray) else list(axes)
            elif n_cols == 1:
                # Single column - axes is 1D array
                axes_list = axes.tolist() if isinstance(axes, np.ndarray) else list(axes)
            else:
                # 2D grid - flatten it
                axes_list = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [ax for row in axes for ax in row]
            
            axes = axes_list
            
            # Plot each trajectory in its own subplot
            for idx, traj_data in enumerate(trajectories_subset):
                ax = axes[idx]
                centroids = traj_data['centroids']
                bout_id = traj_data['bout_id']
                video_name = traj_data['video_name']
                
                if not centroids or len(centroids) < 2:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10)
                    ax.set_axis_off()
                    continue
                
                # Extract data
                frames = [c[0] for c in centroids]
                x_coords = [c[1] for c in centroids]
                y_coords = [c[2] for c in centroids]
                
                # Normalize frame numbers to [0, 1] for colormap
                if len(frames) > 1:
                    frame_min = min(frames)
                    frame_max = max(frames)
                    frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in frames]
                else:
                    continue
                
                # Plot line segments with colors based on time
                for i in range(len(x_coords) - 1):
                    ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                           color=viridis(frame_norm[i]), linewidth=1.5, alpha=0.8, zorder=1)
                
                # Plot points colored by time
                ax.scatter(x_coords, y_coords, c=frame_norm, cmap='viridis', 
                          s=20, alpha=0.7, zorder=2, edgecolors='none')
                
                # Set axis limits (consistent across all subplots)
                # Invert y-axis so 0 is at top (image coordinates)
                ax.set_xlim(0, cage_max_x)
                ax.set_ylim(cage_max_y, 0)  # Inverted: max at bottom, 0 at top
                ax.set_aspect(cage_max_y / cage_max_x, adjustable='box')
                
                # Labels and title for each subplot
                # Extract just the filename from video_name (remove path if present)
                video_short = os.path.basename(video_name) if os.path.sep in video_name else video_name
                # Truncate if too long
                if len(video_short) > 40:
                    video_short = video_short[:37] + '...'
                
                ax.set_title(f'Bout {bout_id}\n{video_short}', 
                            fontsize=9, fontweight='medium', pad=5)
                ax.set_xlabel('X (px)', fontsize=8)
                ax.set_ylabel('Y (px)', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            # Hide unused subplots
            for idx in range(n_subplots, len(axes)):
                axes[idx].set_axis_off()
            
            # Add colorbar for the whole figure
            sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, label='Time (0=early, 1=late)', 
                               orientation='horizontal', pad=0.02, shrink=0.6, aspect=30)
            cbar.set_label('Time (0=early, 1=late)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            
            # Save plot
            if n_plots > 1:
                output_file = output_path.replace('.png', f'_part{plot_idx+1}.png')
            else:
                output_file = output_path
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Leave space for colorbar
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Saved faceted plot: {output_file} ({n_subplots} bouts)")


def plot_combined_cluster_trajectories(cluster_trajectories: List[Dict], cluster_id: int,
                                       output_path: str, cage_max_x: float, cage_max_y: float,
                                       video_dir: str = '..'):
    """Plot all combined trajectories (centroid + nose + tail) for a cluster on one plot.
    
    Args:
        cluster_trajectories: List of dicts with keys: 'centroids', 'nose_points', 'tail_points', 'bout_id', 'video_name'
        cluster_id: Cluster ID
        output_path: Path to save the plot
        cage_max_x: Maximum x dimension (for consistent scaling)
        cage_max_y: Maximum y dimension (for consistent scaling)
    """
    if not cluster_trajectories:
        print(f"Warning: No trajectories for cluster {cluster_id}, skipping plot")
        return
    
    print(f"  Plotting {len(cluster_trajectories)} combined trajectories for cluster {cluster_id}...")
    
    # Create figure with better aesthetics
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use colormaps for each trajectory type
    centroid_cmap = plt.cm.get_cmap('viridis')
    nose_cmap = plt.cm.get_cmap('cool')
    tail_cmap = plt.cm.get_cmap('copper')
    
    # Plot all trajectories
    for traj_data in cluster_trajectories:
        centroids = traj_data.get('centroids', [])
        nose_points = traj_data.get('nose_points', [])
        tail_points = traj_data.get('tail_points', [])
        
        # Determine frame range from all available trajectories
        all_frames = []
        if centroids:
            all_frames.extend([c[0] for c in centroids])
        if nose_points:
            all_frames.extend([n[0] for n in nose_points])
        if tail_points:
            all_frames.extend([t[0] for t in tail_points])
        
        if not all_frames:
            continue
        
        frame_min = min(all_frames)
        frame_max = max(all_frames)
        
        # Plot centroid trajectory (viridis)
        if centroids:
            cent_frames = [c[0] for c in centroids]
            cent_x = [c[1] for c in centroids]
            cent_y = [c[2] for c in centroids]
            cent_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in cent_frames]
            
            # Plot centroid line segments
            for i in range(len(cent_x) - 1):
                ax.plot([cent_x[i], cent_x[i+1]], [cent_y[i], cent_y[i+1]], 
                       color=centroid_cmap(cent_frame_norm[i]), linewidth=1.5, alpha=0.5, zorder=1)
            
            # Plot centroid points
            ax.scatter(cent_x, cent_y, c=cent_frame_norm, cmap='viridis', 
                      s=12, alpha=0.4, zorder=2, edgecolors='none')
        
        # Plot nose trajectory (cool)
        if nose_points:
            nose_frames = [n[0] for n in nose_points]
            nose_x = [n[1] for n in nose_points]
            nose_y = [n[2] for n in nose_points]
            nose_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in nose_frames]
            
            # Plot nose line segments
            for i in range(len(nose_x) - 1):
                ax.plot([nose_x[i], nose_x[i+1]], [nose_y[i], nose_y[i+1]], 
                       color=nose_cmap(nose_frame_norm[i]), linewidth=1.5, alpha=0.5, zorder=1)
            
            # Plot nose points
            ax.scatter(nose_x, nose_y, c=nose_frame_norm, cmap='cool', 
                      s=12, alpha=0.4, zorder=2, edgecolors='none')
        
        # Plot tail trajectory (copper) and connect to nose
        if tail_points:
            tail_frames = [t[0] for t in tail_points]
            tail_x = [t[1] for t in tail_points]
            tail_y = [t[2] for t in tail_points]
            tail_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in tail_frames]
            
            # Draw grey lines connecting nose to tail for each frame
            if nose_points:
                nose_dict = {n[0]: (n[1], n[2]) for n in nose_points}
                tail_dict = {t[0]: (t[1], t[2]) for t in tail_points}
                
                # Find common frames
                common_frames = set(nose_dict.keys()) & set(tail_dict.keys())
                for frame_num in common_frames:
                    nose_x_coord, nose_y_coord = nose_dict[frame_num]
                    tail_x_coord, tail_y_coord = tail_dict[frame_num]
                    ax.plot([nose_x_coord, tail_x_coord], [nose_y_coord, tail_y_coord], 
                           color='gray', linewidth=0.6, alpha=0.3, zorder=0)
            
            # Plot tail line segments
            for i in range(len(tail_x) - 1):
                ax.plot([tail_x[i], tail_x[i+1]], [tail_y[i], tail_y[i+1]], 
                       color=tail_cmap(tail_frame_norm[i]), linewidth=1.5, alpha=0.5, zorder=1)
            
            # Plot tail points
            ax.scatter(tail_x, tail_y, c=tail_frame_norm, cmap='copper', 
                      s=12, alpha=0.4, zorder=2, edgecolors='none')
    
    # Set axis limits to cage dimensions (consistent across all plots)
    # Invert y-axis so 0 is at top (image coordinates)
    ax.set_xlim(0, cage_max_x)
    ax.set_ylim(cage_max_y, 0)  # Inverted: max at bottom, 0 at top
    # Set aspect ratio to match cage dimensions
    ax.set_aspect(cage_max_y / cage_max_x, adjustable='box')
    
    # Add colorbars to show time scale
    sm_cent = plt.cm.ScalarMappable(cmap=centroid_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm_cent.set_array([])
    cbar_cent = plt.colorbar(sm_cent, ax=ax, label='Centroid Time', pad=0.02, location='right', shrink=0.4, aspect=20)
    cbar_cent.set_label('Centroid Time (0=early, 1=late)', fontsize=9)
    cbar_cent.ax.tick_params(labelsize=8)
    
    sm_nose = plt.cm.ScalarMappable(cmap=nose_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm_nose.set_array([])
    cbar_nose = plt.colorbar(sm_nose, ax=ax, label='Nose Time', pad=0.15, location='right', shrink=0.4, aspect=20)
    cbar_nose.set_label('Nose Time (0=early, 1=late)', fontsize=9)
    cbar_nose.ax.tick_params(labelsize=8)
    
    sm_tail = plt.cm.ScalarMappable(cmap=tail_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm_tail.set_array([])
    cbar_tail = plt.colorbar(sm_tail, ax=ax, label='Tail Time', pad=0.28, location='right', shrink=0.4, aspect=20)
    cbar_tail.set_label('Tail Time (0=early, 1=late)', fontsize=9)
    cbar_tail.ax.tick_params(labelsize=8)
    
    # Add lixit location as red star (get from first trajectory's video, optional)
    if cluster_trajectories:
        try:
            first_video = cluster_trajectories[0].get('video_name', '')
            lixit_pos = get_lixit_location(first_video, video_dir)
            if lixit_pos:
                lixit_x, lixit_y = lixit_pos
                ax.plot(lixit_x, lixit_y, 'r*', markersize=10, markeredgecolor='black', markeredgewidth=1, zorder=10, label='Lixit Location')
        except Exception:
            # Skip lixit location if there's any error
            pass
    
    # Labels and title
    ax.set_xlabel('X Position (pixels)', fontsize=13, fontweight='medium')
    ax.set_ylabel('Y Position (pixels)', fontsize=13, fontweight='medium')
    ax.set_title(f'Cluster {cluster_id} - All Combined Trajectories ({len(cluster_trajectories)} bouts)', 
                fontsize=15, fontweight='bold', pad=15)
    
    # Styling
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(labelsize=10)
    
    # Add explanation text below the plot (outside the axes)
    explanation = (
        f"All combined trajectories from cluster {cluster_id} (centroid + nose + tail). "
        f"Cage dimensions: {cage_max_x:.0f} x {cage_max_y:.0f} pixels. "
        f"Centroid: viridis (purple→yellow). Nose: cool (cyan→magenta). Tail: copper (dark→bright). "
        f"Grey lines connect nose to tail for each frame. Red star indicates lixit location."
    )
    fig.text(0.5, 0.01, explanation, ha='center', va='bottom', 
           fontsize=9, wrap=True,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    # Save plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Leave space at bottom for explanation
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved combined cluster plot: {output_path}")


def plot_cluster_trajectories(cluster_trajectories: List[Dict], cluster_id: int,
                              output_path: str, cage_max_x: float, cage_max_y: float,
                              colormap_name: str = 'viridis', point_type: str = 'centroid',
                              video_dir: str = '..'):
    """Plot all trajectories for a cluster on one plot.
    
    Args:
        cluster_trajectories: List of dicts with keys: 'centroids', 'bout_id', 'video_name', optionally 'tail_points'
        cluster_id: Cluster ID
        output_path: Path to save the plot
        cage_max_x: Maximum x dimension (for consistent scaling)
        cage_max_y: Maximum y dimension (for consistent scaling)
        colormap_name: Name of colormap to use (default: 'viridis')
        point_type: Type of point being plotted ('centroid' or 'nose')
    """
    if not cluster_trajectories:
        print(f"Warning: No trajectories for cluster {cluster_id}, skipping plot")
        return
    
    print(f"  Plotting {len(cluster_trajectories)} trajectories for cluster {cluster_id}...")
    
    # Create figure with better aesthetics
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use specified colormap
    colormap = plt.cm.get_cmap(colormap_name)
    
    # For nose plots, also use copper for tail
    tail_colormap = plt.cm.get_cmap('copper') if point_type == 'nose' else None
    
    # Plot all trajectories
    for traj_data in cluster_trajectories:
        centroids = traj_data['centroids']
        if not centroids or len(centroids) < 2:
            continue
        
        # Extract data
        frames = [c[0] for c in centroids]
        x_coords = [c[1] for c in centroids]
        y_coords = [c[2] for c in centroids]
        
        # Normalize frame numbers to [0, 1] for colormap
        if len(frames) > 1:
            frame_min = min(frames)
            frame_max = max(frames)
            frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in frames]
        else:
            continue  # Skip single point trajectories
        
        # For nose plots, also extract tail keypoints and draw connection lines
        if point_type == 'nose' and 'tail_points' in traj_data:
            tail_points = traj_data['tail_points']
            if tail_points:
                # Create a dict for quick lookup: frame -> (x, y)
                tail_dict = {tp[0]: (tp[1], tp[2]) for tp in tail_points}
                
                # Draw grey lines connecting nose to tail for each frame
                for i, frame_num in enumerate(frames):
                    if frame_num in tail_dict:
                        nose_x, nose_y = x_coords[i], y_coords[i]
                        tail_x, tail_y = tail_dict[frame_num]
                        ax.plot([nose_x, tail_x], [nose_y, tail_y], 
                               color='gray', linewidth=0.8, alpha=0.4, zorder=0)
                
                # Plot tail trajectory with copper colormap
                tail_frames = [tp[0] for tp in tail_points]
                tail_x_coords = [tp[1] for tp in tail_points]
                tail_y_coords = [tp[2] for tp in tail_points]
                
                if len(tail_frames) > 1:
                    tail_frame_norm = [(f - frame_min) / (frame_max - frame_min) for f in tail_frames]
                    
                    # Plot tail line segments
                    for i in range(len(tail_x_coords) - 1):
                        ax.plot([tail_x_coords[i], tail_x_coords[i+1]], 
                               [tail_y_coords[i], tail_y_coords[i+1]], 
                               color=tail_colormap(tail_frame_norm[i]), linewidth=1.5, alpha=0.6, zorder=1)
                    
                    # Plot tail points
                    ax.scatter(tail_x_coords, tail_y_coords, c=tail_frame_norm, cmap='copper', 
                              s=15, alpha=0.5, zorder=2, edgecolors='none')
        
        # Plot nose/centroid line segments with colors based on time
        for i in range(len(x_coords) - 1):
            ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                   color=colormap(frame_norm[i]), linewidth=1.5, alpha=0.6, zorder=1)
        
        # Plot nose/centroid points colored by time (smaller and more transparent for overlay)
        ax.scatter(x_coords, y_coords, c=frame_norm, cmap=colormap_name, 
                  s=15, alpha=0.5, zorder=2, edgecolors='none')
    
    # Set axis limits to cage dimensions (consistent across all plots)
    # Invert y-axis so 0 is at top (image coordinates)
    ax.set_xlim(0, cage_max_x)
    ax.set_ylim(cage_max_y, 0)  # Inverted: max at bottom, 0 at top
    # Set aspect ratio to match cage dimensions
    ax.set_aspect(cage_max_y / cage_max_x, adjustable='box')
    
    # Add colorbar(s) to show time scale
    if point_type == 'nose':
        # Add two colorbars: one for nose (cool) and one for tail (copper)
        sm_nose = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        sm_nose.set_array([])
        cbar_nose = plt.colorbar(sm_nose, ax=ax, label='Nose Time', pad=0.02, location='right', shrink=0.4, aspect=20)
        cbar_nose.set_label('Nose Time (0=early, 1=late)', fontsize=10)
        cbar_nose.ax.tick_params(labelsize=8)
        
        sm_tail = plt.cm.ScalarMappable(cmap=tail_colormap, norm=plt.Normalize(vmin=0, vmax=1))
        sm_tail.set_array([])
        cbar_tail = plt.colorbar(sm_tail, ax=ax, label='Tail Time', pad=0.15, location='right', shrink=0.4, aspect=20)
        cbar_tail.set_label('Tail Time (0=early, 1=late)', fontsize=10)
        cbar_tail.ax.tick_params(labelsize=8)
    else:
        # Single colorbar for centroid plots
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Time (normalized)', pad=0.02, shrink=0.4, aspect=20)
        cbar.set_label('Time (normalized: 0=early, 1=late)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
    
    # Add lixit location as red star (get from first trajectory's video, optional)
    if cluster_trajectories:
        try:
            first_video = cluster_trajectories[0].get('video_name', '')
            lixit_pos = get_lixit_location(first_video, video_dir)
            if lixit_pos:
                lixit_x, lixit_y = lixit_pos
                ax.plot(lixit_x, lixit_y, 'r*', markersize=10, markeredgecolor='black', markeredgewidth=1, zorder=10, label='Lixit Location')
        except Exception:
            # Skip lixit location if there's any error
            pass
    
    # Labels and title
    ax.set_xlabel('X Position (pixels)', fontsize=13, fontweight='medium')
    ax.set_ylabel('Y Position (pixels)', fontsize=13, fontweight='medium')
    ax.set_title(f'Cluster {cluster_id} - All Trajectories ({len(cluster_trajectories)} bouts)', 
                fontsize=15, fontweight='bold', pad=15)
    
    # Styling
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(labelsize=10)
    
    # Add explanation text below the plot (outside the axes)
    if point_type == 'nose':
        explanation = (
            f"All trajectories from cluster {cluster_id} (nose + tail keypoints). "
            f"Cage dimensions: {cage_max_x:.0f} x {cage_max_y:.0f} pixels. "
            f"Nose: cool colormap (cyan→magenta). Tail: copper colormap (dark→bright). "
            f"Grey lines connect nose to tail for each frame. Red star indicates lixit location."
        )
    else:
        point_desc = "bounding box centroid"
        explanation = (
            f"All trajectories from cluster {cluster_id} ({point_desc}). "
            f"Cage dimensions: {cage_max_x:.0f} x {cage_max_y:.0f} pixels. "
            f"Color indicates time progression (early → late). Red star indicates lixit location."
        )
    fig.text(0.5, 0.01, explanation, ha='center', va='bottom', 
           fontsize=9, wrap=True,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    # Save plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Leave space at bottom for explanation
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved cluster plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot bounding box centroid trajectories for behavior bouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot trajectories for turn_left behavior
  python3 scripts/plot_bout_trajectories.py --behavior turn_left --output-dir BoutResults
  
  # Specify video directory
  python3 scripts/plot_bout_trajectories.py --behavior turn_left --video-dir .. --output-dir BoutResults
        """
    )
    
    parser.add_argument('--behavior', type=str, required=True,
                       help='Behavior name (e.g., turn_left)')
    parser.add_argument('--output-dir', type=str, default='BoutResults',
                       help='Output directory (default: BoutResults)')
    parser.add_argument('--video-dir', type=str, default='..',
                       help='Directory containing video and pose estimation files (default: ..)')
    parser.add_argument('--cluster-method', type=str, default='bsoid',
                       help='Clustering method to use (default: bsoid)')
    parser.add_argument('--max-bouts', type=int, default=None,
                       help='Maximum number of bouts to plot (for testing, default: all)')
    parser.add_argument('--faceted', action='store_true',
                       help='Create faceted plots (one bout per subplot) in addition to cluster plots')
    parser.add_argument('--faceted-only', action='store_true',
                       help='Only create faceted plots (skip cluster overlay plots)')
    
    args = parser.parse_args()
    
    # Set up paths
    output_dir = Path(args.output_dir)
    clustering_dir = output_dir / 'clustering' / args.cluster_method
    trajectories_dir = clustering_dir / 'trajectories'
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cluster assignments
    cluster_file = clustering_dir / f'cluster_assignments_{args.cluster_method}.csv'
    if not cluster_file.exists():
        print(f"Error: Cluster assignments file not found: {cluster_file}")
        sys.exit(1)
    
    print(f"Loading cluster assignments from: {cluster_file}")
    df_clusters = pd.read_csv(cluster_file)
    print(f"Loaded {len(df_clusters)} total cluster assignments")
    
    # Filter for the specified behavior
    df_clusters = df_clusters[df_clusters['behavior'] == args.behavior]
    df_clusters = df_clusters[df_clusters['cluster_method'] == args.cluster_method]
    
    if len(df_clusters) == 0:
        print(f"Error: No bouts found for behavior '{args.behavior}' with method '{args.cluster_method}'")
        sys.exit(1)
    
    print(f"Found {len(df_clusters)} bouts for behavior '{args.behavior}'")
    
    # Limit number of bouts if specified
    if args.max_bouts:
        df_clusters = df_clusters.head(args.max_bouts)
        print(f"Limiting to first {args.max_bouts} bouts for testing")
    
    # Get all unique clusters
    unique_clusters = sorted(df_clusters['cluster_id'].unique())
    
    print(f"Number of clusters: {len(unique_clusters)}")
    print(f"Clusters: {unique_clusters}")
    print(f"Saving cluster trajectory plots to: {trajectories_dir}")
    print()
    
    # First pass: Extract all trajectories and find max cage dimensions
    print("Step 1: Extracting trajectories and determining cage dimensions...")
    all_trajectories = {}  # cluster_id -> list of trajectory dicts
    max_cage_x = 0
    max_cage_y = 0
    video_cage_dims = {}  # Cache cage dimensions per video file
    
    for idx, row in df_clusters.iterrows():
        bout_id = int(row['bout_id'])
        video_name = row['video_name']
        animal_id = int(row['animal_id'])
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])
        cluster_id = int(row['cluster_id'])
        
        # Find pose estimation file
        pose_file = get_pose_file(video_name, args.video_dir)
        if not pose_file:
            continue
        
        # Get cage dimensions (cache per video file)
        if pose_file not in video_cage_dims:
            cage_max_x, cage_max_y = get_cage_dimensions(pose_file)
            if cage_max_x and cage_max_y:
                video_cage_dims[pose_file] = (cage_max_x, cage_max_y)
                max_cage_x = max(max_cage_x, cage_max_x)
                max_cage_y = max(max_cage_y, cage_max_y)
        else:
            cage_max_x, cage_max_y = video_cage_dims[pose_file]
        
        # Find the correct identity index first - this ensures bbox, nose, and tail all use the same identity
        corrected_identity = find_correct_identity_index(pose_file, animal_id, start_frame, end_frame)
        
        # Extract centroids using the corrected identity
        centroids = extract_bbox_centroids(pose_file, corrected_identity, start_frame, end_frame)
        if not centroids:
            continue
        
        # Store trajectory data
        if cluster_id not in all_trajectories:
            all_trajectories[cluster_id] = []
        
        all_trajectories[cluster_id].append({
            'centroids': centroids,
            'bout_id': bout_id,
            'video_name': video_name,
            'animal_id': animal_id,
            'start_frame': start_frame,
            'end_frame': end_frame
        })
    
    # Also extract nose and tail keypoints for second set of plots
    print("Step 1b: Extracting nose and tail keypoint trajectories...")
    all_nose_trajectories = {}  # cluster_id -> list of trajectory dicts
    all_combined_trajectories = {}  # cluster_id -> list of trajectory dicts with all three types
    BASE_TAIL_KEYPOINT_IDX = 9  # Base of tail keypoint index (most valid keypoint after nose)
    
    for idx, row in df_clusters.iterrows():
        bout_id = int(row['bout_id'])
        video_name = row['video_name']
        animal_id = int(row['animal_id'])
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])
        cluster_id = int(row['cluster_id'])
        
        # Find pose estimation file
        pose_file = get_pose_file(video_name, args.video_dir)
        if not pose_file:
            continue
        
        # Find the correct identity index by checking which identity's keypoints align with the bbox
        # This ensures we use the same identity for bbox, nose, and tail
        corrected_identity = find_correct_identity_index(pose_file, animal_id, start_frame, end_frame)
        
        # Extract nose keypoints using the corrected identity
        nose_points = extract_nose_keypoints(pose_file, corrected_identity, start_frame, end_frame)
        if not nose_points:
            continue
        
        # Extract tail keypoints (base of tail) using the corrected identity
        tail_points = extract_keypoint(pose_file, corrected_identity, start_frame, end_frame, BASE_TAIL_KEYPOINT_IDX)
        
        # Get centroid for this bout (from earlier extraction) - should use same corrected identity
        bout_centroids = None
        if cluster_id in all_trajectories:
            for traj in all_trajectories[cluster_id]:
                if traj['bout_id'] == bout_id:
                    bout_centroids = traj['centroids']
                    break
        
        # If we couldn't find centroids, try extracting with corrected identity
        if not bout_centroids:
            bout_centroids = extract_bbox_centroids(pose_file, corrected_identity, start_frame, end_frame)
        
        # Store trajectory data for nose plots
        if cluster_id not in all_nose_trajectories:
            all_nose_trajectories[cluster_id] = []
        
        all_nose_trajectories[cluster_id].append({
            'centroids': nose_points,  # Reuse 'centroids' key for consistency
            'tail_points': tail_points,  # Add tail keypoints
            'bout_id': bout_id,
            'video_name': video_name,
            'animal_id': animal_id,
            'start_frame': start_frame,
            'end_frame': end_frame
        })
        
        # Store combined trajectory data (centroid + nose + tail)
        if cluster_id not in all_combined_trajectories:
            all_combined_trajectories[cluster_id] = []
        
        all_combined_trajectories[cluster_id].append({
            'centroids': bout_centroids if bout_centroids else [],
            'nose_points': nose_points,
            'tail_points': tail_points,
            'bout_id': bout_id,
            'video_name': video_name,
            'animal_id': animal_id,
            'start_frame': start_frame,
            'end_frame': end_frame
        })
    
    print(f"  Found max cage dimensions: {max_cage_x:.1f} x {max_cage_y:.1f} pixels")
    print(f"  Extracted centroid trajectories for {len(all_trajectories)} clusters")
    for cluster_id, trajs in all_trajectories.items():
        print(f"    Cluster {cluster_id}: {len(trajs)} trajectories")
    print(f"  Extracted nose keypoint trajectories for {len(all_nose_trajectories)} clusters")
    for cluster_id, trajs in all_nose_trajectories.items():
        print(f"    Cluster {cluster_id}: {len(trajs)} trajectories")
    print()
    
    # Second pass: Plot each cluster (overlay plots) - CENTROID plots with viridis
    successful = 0
    failed = 0
    nose_successful = 0
    nose_failed = 0
    combined_successful = 0
    combined_failed = 0
    
    if not args.faceted_only:
        print("Step 2: Generating cluster overlay plots (centroid, viridis colormap)...")
        for cluster_id in unique_clusters:
            if cluster_id not in all_trajectories or len(all_trajectories[cluster_id]) == 0:
                print(f"  Skipping cluster {cluster_id}: no trajectories")
                continue
            
            # Create output filename
            output_filename = f"cluster_{cluster_id}_all_trajectories.png"
            output_path = trajectories_dir / output_filename
            
            try:
                plot_cluster_trajectories(
                    all_trajectories[cluster_id], 
                    cluster_id,
                    str(output_path),
                    max_cage_x,
                    max_cage_y,
                    colormap_name='viridis',
                    point_type='centroid',
                    video_dir=args.video_dir
                )
                successful += 1
            except Exception as e:
                import traceback
                print(f"  Error plotting cluster {cluster_id}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                failed += 1
                continue
        
        # Second pass (b): Plot each cluster (overlay plots) - NOSE plots with cool colormap
        print()
        print("Step 2b: Generating cluster overlay plots (nose keypoint, cool colormap)...")
        
        for cluster_id in unique_clusters:
            if cluster_id not in all_nose_trajectories or len(all_nose_trajectories[cluster_id]) == 0:
                print(f"  Skipping cluster {cluster_id}: no nose trajectories")
                continue
            
            # Create output filename with _nose suffix
            output_filename = f"cluster_{cluster_id}_nose_all_trajectories.png"
            output_path = trajectories_dir / output_filename
            
            try:
                plot_cluster_trajectories(
                    all_nose_trajectories[cluster_id], 
                    cluster_id,
                    str(output_path),
                    max_cage_x,
                    max_cage_y,
                    colormap_name='cool',
                    point_type='nose',
                    video_dir=args.video_dir
                )
                nose_successful += 1
            except Exception as e:
                import traceback
                print(f"  Error plotting nose trajectories for cluster {cluster_id}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                nose_failed += 1
                continue
        
        # Second pass (c): Plot each cluster (overlay plots) - COMBINED plots (centroid + nose + tail)
        print()
        print("Step 2c: Generating cluster overlay plots (combined: centroid + nose + tail)...")
        combined_successful = 0
        combined_failed = 0
        
        for cluster_id in unique_clusters:
            if cluster_id not in all_combined_trajectories or len(all_combined_trajectories[cluster_id]) == 0:
                print(f"  Skipping cluster {cluster_id}: no combined trajectories")
                continue
            
            # Create output filename with _combined suffix
            output_filename = f"cluster_{cluster_id}_combined_all_trajectories.png"
            output_path = trajectories_dir / output_filename
            
            try:
                plot_combined_cluster_trajectories(
                    all_combined_trajectories[cluster_id], 
                    cluster_id,
                    str(output_path),
                    max_cage_x,
                    max_cage_y,
                    video_dir=args.video_dir
                )
                combined_successful += 1
            except Exception as e:
                import traceback
                print(f"  Error plotting combined trajectories for cluster {cluster_id}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                combined_failed += 1
                continue
    
    # Third pass: Create multi-page PDFs (one PDF per cluster, one bout per page) - CENTROID
    if args.faceted or args.faceted_only:
        print()
        print("Step 3: Generating multi-page PDFs (centroid, viridis colormap)...")
        pdf_successful = 0
        pdf_failed = 0
        
        for cluster_id in sorted(unique_clusters):
            if cluster_id not in all_trajectories or len(all_trajectories[cluster_id]) == 0:
                continue
            
            try:
                # Create one PDF per cluster
                output_filename = f"cluster_{cluster_id}_individual_bouts.pdf"
                output_path = trajectories_dir / output_filename
                
                plot_individual_bouts_pdf(
                    {cluster_id: all_trajectories[cluster_id]},
                    str(output_path),
                    max_cage_x,
                    max_cage_y,
                    colormap_name='viridis',
                    point_type='centroid',
                    video_dir=args.video_dir
                )
                pdf_successful += 1
            except Exception as e:
                import traceback
                print(f"  Error creating PDF for cluster {cluster_id}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                pdf_failed += 1
                continue
        
        print(f"  Centroid PDFs created: {pdf_successful} successful, {pdf_failed} failed")
        
        # Third pass (b): Create multi-page PDFs - NOSE
        print()
        print("Step 3b: Generating multi-page PDFs (nose keypoint, cool colormap)...")
        nose_pdf_successful = 0
        nose_pdf_failed = 0
        
        for cluster_id in sorted(unique_clusters):
            if cluster_id not in all_nose_trajectories or len(all_nose_trajectories[cluster_id]) == 0:
                continue
            
            try:
                # Create one PDF per cluster with _nose suffix
                output_filename = f"cluster_{cluster_id}_nose_individual_bouts.pdf"
                output_path = trajectories_dir / output_filename
                
                plot_individual_bouts_pdf(
                    {cluster_id: all_nose_trajectories[cluster_id]},
                    str(output_path),
                    max_cage_x,
                    max_cage_y,
                    colormap_name='cool',
                    point_type='nose',
                    video_dir=args.video_dir
                )
                nose_pdf_successful += 1
            except Exception as e:
                import traceback
                print(f"  Error creating nose PDF for cluster {cluster_id}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                nose_pdf_failed += 1
                continue
        
        print(f"  Nose PDFs created: {nose_pdf_successful} successful, {nose_pdf_failed} failed")
        
        # Third pass (c): Create combined multi-page PDFs (one PDF per cluster, all three trajectories)
        print()
        print("Step 3c: Generating combined multi-page PDFs (centroid + nose + tail, one PDF per cluster)...")
        combined_pdf_successful = 0
        combined_pdf_failed = 0
        
        for cluster_id in sorted(unique_clusters):
            if cluster_id not in all_combined_trajectories or len(all_combined_trajectories[cluster_id]) == 0:
                continue
            
            try:
                # Create one PDF per cluster with _combined suffix
                output_filename = f"cluster_{cluster_id}_combined_individual_bouts.pdf"
                output_path = trajectories_dir / output_filename
                
                plot_combined_individual_bouts_pdf(
                    {cluster_id: all_combined_trajectories[cluster_id]},
                    str(output_path),
                    max_cage_x,
                    max_cage_y,
                    video_dir=args.video_dir
                )
                combined_pdf_successful += 1
            except Exception as e:
                import traceback
                print(f"  Error creating combined PDF for cluster {cluster_id}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                combined_pdf_failed += 1
                continue
        
        print(f"  Combined PDFs created: {combined_pdf_successful} successful, {combined_pdf_failed} failed")
        
        # Third pass (d): Create combined trajectory PDFs (one PDF per bout) - individual files
        print()
        print("Step 3d: Generating combined trajectory PDFs (centroid + nose + tail, one PDF per bout)...")
        combined_dir = trajectories_dir / 'combined'
        combined_dir.mkdir(exist_ok=True)
        
        try:
            plot_combined_trajectories_pdf(
                all_combined_trajectories,
                str(combined_dir),
                max_cage_x,
                max_cage_y,
                video_dir=args.video_dir
            )
            print(f"  Combined trajectory PDFs created successfully")
        except Exception as e:
            import traceback
            print(f"  Error creating combined trajectory PDFs: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
    
    print()
    if not args.faceted_only:
        print(f"Centroid overlay plots: {successful} successful, {failed} failed")
        print(f"Nose overlay plots: {nose_successful} successful, {nose_failed} failed")
        print(f"Combined overlay plots: {combined_successful} successful, {combined_failed} failed")
    print(f"All plots saved to: {trajectories_dir}")


if __name__ == '__main__':
    main()


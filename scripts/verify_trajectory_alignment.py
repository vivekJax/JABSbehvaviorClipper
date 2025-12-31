#!/usr/bin/env python3
"""
Diagnostic script to verify that bounding box centroids, nose, and tail keypoints
are correctly aligned for the same animal_id and bout.
"""

import json
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def get_pose_file(video_name: str, video_dir: str):
    """Find pose estimation file for a video."""
    video_stem = Path(video_name).stem
    pose_file = Path(video_dir) / f"{video_stem}_pose_est_v8.h5"
    if pose_file.exists():
        return str(pose_file)
    return None

def verify_bout_alignment(cluster_file, video_dir, behavior='turn_left', max_bouts=5):
    """Verify that centroids, nose, and tail are aligned for sample bouts."""
    
    df = pd.read_csv(cluster_file)
    df = df[df['behavior'] == behavior]
    
    print(f"Checking alignment for {min(max_bouts, len(df))} sample bouts...\n")
    
    for idx, row in df.head(max_bouts).iterrows():
        bout_id = int(row['bout_id'])
        video_name = row['video_name']
        animal_id = int(row['animal_id'])
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])
        
        pose_file = get_pose_file(video_name, video_dir)
        if not pose_file:
            print(f"Bout {bout_id}: No pose file found")
            continue
        
        print(f"Bout {bout_id}: Video={Path(video_name).name}, Animal={animal_id}, Frames={start_frame}-{end_frame}")
        
        try:
            centroid_x = None
            centroid_y = None
            x1 = x2 = y1 = y2 = None
            
            with h5py.File(pose_file, 'r') as f:
                # Check bbox
                if '/poseest/bbox' in f:
                    bbox_dataset = f['/poseest/bbox']
                    max_frame = bbox_dataset.shape[0] - 1
                    start = min(start_frame, max_frame)
                    end = min(end_frame, max_frame)
                    
                    if start <= end and animal_id < bbox_dataset.shape[1]:
                        # Get one frame for comparison
                        test_frame = start
                        bbox_data = bbox_dataset[test_frame, animal_id, :, :]
                        x1, y1 = float(bbox_data[0, 0]), float(bbox_data[0, 1])
                        x2, y2 = float(bbox_data[1, 0]), float(bbox_data[1, 1])
                        centroid_x = (x1 + x2) / 2.0
                        centroid_y = (y1 + y2) / 2.0
                        print(f"  Frame {test_frame} - BBox centroid (animal_id={animal_id}): ({centroid_x:.1f}, {centroid_y:.1f})")
                        print(f"    BBox: [{x1:.1f}, {y1:.1f}] to [{x2:.1f}, {y2:.1f}]")
                    else:
                        print(f"  BBox: Invalid frame range or animal_id (shape: {bbox_dataset.shape})")
                
                # Check nose keypoint - try all identity indices to find the correct one
                if '/poseest/points' in f and centroid_x is not None:
                    points_dataset = f['/poseest/points']
                    max_frame = points_dataset.shape[0] - 1
                    start = min(start_frame, max_frame)
                    end = min(end_frame, max_frame)
                    
                    if start <= end:
                        test_frame = start
                        print(f"  Checking nose keypoint for all identities (n_identities={points_dataset.shape[1]}):")
                        
                        best_match_id = None
                        best_distance = float('inf')
                        
                        for test_id in range(points_dataset.shape[1]):
                            nose_data = points_dataset[test_frame, test_id, 0, :]  # Keypoint 0 = nose
                            nose_x, nose_y = float(nose_data[0]), float(nose_data[1])
                            
                            # Check if valid
                            if nose_x > 0 and nose_y > 0:
                                distance = np.sqrt((nose_x - centroid_x)**2 + (nose_y - centroid_y)**2)
                                nose_in_bbox = (x1 <= nose_x <= x2) and (y1 <= nose_y <= y2)
                                print(f"    Identity {test_id}: Nose=({nose_x:.1f}, {nose_y:.1f}), "
                                      f"Distance={distance:.1f}px, InBBox={nose_in_bbox}")
                                
                                if distance < best_distance:
                                    best_distance = distance
                                    best_match_id = test_id
                        
                        if best_match_id is not None:
                            print(f"  Best match: Identity {best_match_id} (distance={best_distance:.1f}px)")
                            if best_match_id != animal_id:
                                print(f"  *** MISMATCH: CSV animal_id={animal_id} but best match is identity {best_match_id} ***")
                        else:
                            print(f"  No valid nose keypoint found")
                    else:
                        print(f"  Nose: Invalid frame range")
                
                # Check tail keypoint
                if '/poseest/points' in f:
                    points_dataset = f['/poseest/points']
                    max_frame = points_dataset.shape[0] - 1
                    start = min(start_frame, max_frame)
                    end = min(end_frame, max_frame)
                    
                    if start <= end and animal_id < points_dataset.shape[1] and points_dataset.shape[2] > 9:
                        test_frame = start
                        tail_data = points_dataset[test_frame, animal_id, 9, :]  # Keypoint 9 = tail base
                        tail_x, tail_y = float(tail_data[0]), float(tail_data[1])
                        print(f"  Frame {test_frame} - Tail keypoint: ({tail_x:.1f}, {tail_y:.1f})")
                        
                        # Check if tail is inside bbox
                        if 'centroid_x' in locals():
                            tail_in_bbox = (x1 <= tail_x <= x2) and (y1 <= tail_y <= y2)
                            print(f"  Tail inside bbox: {tail_in_bbox}")
                            print(f"  Distance from centroid: {np.sqrt((tail_x - centroid_x)**2 + (tail_y - centroid_y)**2):.1f} pixels")
                    else:
                        print(f"  Tail: Invalid frame range or animal_id")
        
        except Exception as e:
            print(f"  Error: {e}")
        
        print()

if __name__ == '__main__':
    cluster_file = 'BoutResults/clustering/bsoid/cluster_assignments_bsoid.csv'
    video_dir = '..'
    
    if len(sys.argv) > 1:
        cluster_file = sys.argv[1]
    if len(sys.argv) > 2:
        video_dir = sys.argv[2]
    
    verify_bout_alignment(cluster_file, video_dir)


#!/usr/bin/env python3
"""
Diagnostic script to investigate bounding box centroid vs keypoint mismatch.

This script will:
1. Read a sample bout from the cluster assignments
2. Extract bbox centroid and keypoints for all identities
3. Show which identity's keypoints are inside which identity's bbox
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import json
from typing import Optional, Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_pose_file(video_name: str, video_dir: str) -> Optional[str]:
    """Find pose estimation file for a video."""
    video_basename = os.path.splitext(video_name)[0]
    
    # Try different patterns
    patterns = [
        f"{video_basename}_pose_est_v8.h5",
        f"{video_basename}.h5",
    ]
    
    for pattern in patterns:
        pose_path = os.path.join(video_dir, pattern)
        if os.path.exists(pose_path):
            return pose_path
    
    # Try finding by partial match
    try:
        for f in os.listdir(video_dir):
            if f.endswith('_pose_est_v8.h5') and video_basename in f:
                return os.path.join(video_dir, f)
    except:
        pass
    
    return None


def diagnose_bout(pose_file: str, animal_id: int, start_frame: int, end_frame: int):
    """Diagnose identity alignment for a single bout."""
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC REPORT")
    print(f"{'='*70}")
    print(f"Pose file: {pose_file}")
    print(f"Animal ID from CSV: {animal_id}")
    print(f"Frame range: {start_frame} - {end_frame}")
    
    with h5py.File(pose_file, 'r') as f:
        # Get dataset shapes
        bbox_data = f['/poseest/bbox']
        points_data = f['/poseest/points']
        
        print(f"\nHDF5 Structure:")
        print(f"  bbox shape: {bbox_data.shape}")
        print(f"  points shape: {points_data.shape}")
        
        n_frames, n_identities, n_keypoints, _ = points_data.shape
        
        # Check external identity mapping
        if '/poseest/external_identity_mapping' in f:
            mapping = f['/poseest/external_identity_mapping'][:]
            print(f"\nExternal Identity Mapping:")
            for i, m in enumerate(mapping):
                print(f"  HDF5 index {i} -> external ID: {m}")
        
        # Test at middle frame
        test_frame = (start_frame + end_frame) // 2
        test_frame = min(test_frame, n_frames - 1)
        
        print(f"\n{'='*70}")
        print(f"FRAME {test_frame} ANALYSIS")
        print(f"{'='*70}")
        
        # Get all bboxes and keypoints at this frame
        for identity_idx in range(n_identities):
            bbox = bbox_data[test_frame, identity_idx, :, :]
            x1, y1 = float(bbox[0, 0]), float(bbox[0, 1])
            x2, y2 = float(bbox[1, 0]), float(bbox[1, 1])
            
            # Calculate centroid
            if x1 > 0 and y1 > 0 and x2 > x1 and y2 > y1:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_valid = True
            else:
                cx, cy = 0, 0
                bbox_valid = False
            
            print(f"\n--- Identity Index {identity_idx} ---")
            print(f"  BBox: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
            print(f"  BBox Centroid: ({cx:.1f}, {cy:.1f})")
            print(f"  BBox Valid: {bbox_valid}")
            
            # Get keypoints for this identity
            nose = points_data[test_frame, identity_idx, 0, :]  # Index 0 = nose
            tail = points_data[test_frame, identity_idx, 9, :]  # Index 9 = base_tail
            
            nose_x, nose_y = float(nose[0]), float(nose[1])
            tail_x, tail_y = float(tail[0]), float(tail[1])
            
            print(f"  Nose: ({nose_x:.1f}, {nose_y:.1f})")
            print(f"  Tail: ({tail_x:.1f}, {tail_y:.1f})")
            
            # Check if keypoints are inside the bbox of THIS identity
            if bbox_valid and nose_x > 0 and nose_y > 0:
                nose_in_own_bbox = (x1 <= nose_x <= x2) and (y1 <= nose_y <= y2)
                print(f"  Nose inside own bbox: {nose_in_own_bbox}")
            
            # Check if keypoints are inside the bbox of EACH identity
            print(f"\n  Keypoint-to-BBox matching (which bbox contains these keypoints):")
            for other_idx in range(n_identities):
                other_bbox = bbox_data[test_frame, other_idx, :, :]
                ox1, oy1 = float(other_bbox[0, 0]), float(other_bbox[0, 1])
                ox2, oy2 = float(other_bbox[1, 0]), float(other_bbox[1, 1])
                
                if ox1 > 0 and oy1 > 0 and ox2 > ox1 and oy2 > oy1:
                    if nose_x > 0 and nose_y > 0:
                        nose_in_bbox = (ox1 <= nose_x <= ox2) and (oy1 <= nose_y <= oy2)
                        if nose_in_bbox:
                            marker = " <-- MATCH!" if other_idx != identity_idx else ""
                            print(f"    Identity {identity_idx}'s nose is inside Identity {other_idx}'s bbox{marker}")
        
        # Summary: recommend correct identity
        print(f"\n{'='*70}")
        print(f"RECOMMENDATION")
        print(f"{'='*70}")
        
        # Find which identity's keypoints match the CSV animal_id's bbox
        csv_bbox = bbox_data[test_frame, animal_id, :, :]
        csv_x1, csv_y1 = float(csv_bbox[0, 0]), float(csv_bbox[0, 1])
        csv_x2, csv_y2 = float(csv_bbox[1, 0]), float(csv_bbox[1, 1])
        
        print(f"\nCSV animal_id={animal_id}'s bbox: ({csv_x1:.1f}, {csv_y1:.1f}) - ({csv_x2:.1f}, {csv_y2:.1f})")
        
        if csv_x1 > 0 and csv_y1 > 0 and csv_x2 > csv_x1 and csv_y2 > csv_y1:
            for check_idx in range(n_identities):
                check_nose = points_data[test_frame, check_idx, 0, :]
                check_x, check_y = float(check_nose[0]), float(check_nose[1])
                
                if check_x > 0 and check_y > 0:
                    in_bbox = (csv_x1 <= check_x <= csv_x2) and (csv_y1 <= check_y <= csv_y2)
                    if in_bbox:
                        if check_idx == animal_id:
                            print(f"\n✓ Identity {check_idx} (same as CSV) has keypoints inside the bbox")
                        else:
                            print(f"\n⚠ Identity {check_idx} (NOT {animal_id}) has keypoints inside animal_id={animal_id}'s bbox!")
                            print(f"  This suggests the keypoints should use identity index {check_idx}, not {animal_id}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose identity mismatch')
    parser.add_argument('--behavior', default='turn_left', help='Behavior name')
    parser.add_argument('--output-dir', default='BoutResults', help='Output directory with cluster assignments')
    parser.add_argument('--video-dir', default='..', help='Directory containing video/pose files')
    parser.add_argument('--bout-index', type=int, default=0, help='Which bout to analyze (0-indexed)')
    args = parser.parse_args()
    
    # Load cluster assignments to get bout info
    cluster_file = os.path.join(args.output_dir, 'clustering', 'bsoid', 'cluster_assignments_bsoid.csv')
    if not os.path.exists(cluster_file):
        print(f"Error: Cluster file not found: {cluster_file}")
        return
    
    df = pd.read_csv(cluster_file)
    print(f"Loaded {len(df)} bouts from {cluster_file}")
    print(f"Columns: {list(df.columns)}")
    
    # Get bout info
    if args.bout_index >= len(df):
        print(f"Error: bout_index {args.bout_index} is out of range (max: {len(df)-1})")
        return
    
    bout = df.iloc[args.bout_index]
    print(f"\nSelected bout {args.bout_index}:")
    print(bout)
    
    # Find pose file
    video_name = bout['video_name']
    pose_file = get_pose_file(video_name, args.video_dir)
    
    if not pose_file:
        print(f"Error: Could not find pose file for {video_name}")
        return
    
    # Get animal ID
    animal_id = int(bout.get('animal_id', bout.get('identity_id', 0)))
    start_frame = int(bout['start_frame'])
    end_frame = int(bout['end_frame'])
    
    # Run diagnosis
    diagnose_bout(pose_file, animal_id, start_frame, end_frame)


if __name__ == '__main__':
    main()


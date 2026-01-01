"""
Pytest configuration and shared fixtures
"""
import pytest
import os
import json
import tempfile
import shutil
import h5py
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_annotation_data():
    """Sample annotation JSON data structure."""
    return {
        "file": "test_video.mp4",
        "unfragmented_labels": {
            "0": {
                "turn_left": [
                    {
                        "start": 100,
                        "end": 150,
                        "present": True
                    },
                    {
                        "start": 200,
                        "end": 250,
                        "present": True
                    }
                ]
            },
            "1": {
                "turn_left": [
                    {
                        "start": 300,
                        "end": 350,
                        "present": True
                    }
                ]
            }
        },
        "labels": {
            "0": {
                "turn_left": [
                    {
                        "start": 100,
                        "end": 150,
                        "present": True
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_annotation_file(temp_dir, sample_annotation_data):
    """Create a sample annotation JSON file."""
    annotation_file = os.path.join(temp_dir, "test_annotation.json")
    with open(annotation_file, 'w') as f:
        json.dump(sample_annotation_data, f)
    return annotation_file


@pytest.fixture
def sample_h5_file(temp_dir):
    """Create a sample HDF5 file with pose estimation data."""
    h5_file = os.path.join(temp_dir, "test_video_pose_est_v8.h5")
    
    with h5py.File(h5_file, 'w') as f:
        # Create bbox dataset: (n_frames, n_identities, 2, 2)
        n_frames = 500
        n_identities = 2
        bbox_data = np.zeros((n_frames, n_identities, 2, 2))
        
        # Fill with sample bounding boxes
        for frame in range(n_frames):
            for identity in range(n_identities):
                # Top-left corner
                bbox_data[frame, identity, 0, 0] = 100.0 + frame * 0.1  # x1
                bbox_data[frame, identity, 0, 1] = 200.0 + frame * 0.1  # y1
                # Bottom-right corner
                bbox_data[frame, identity, 1, 0] = 200.0 + frame * 0.1  # x2
                bbox_data[frame, identity, 1, 1] = 300.0 + frame * 0.1  # y2
        
        f.create_dataset('/poseest/bbox', data=bbox_data)
        
        # Create points dataset: (n_frames, n_identities, n_keypoints, 2)
        n_keypoints = 10
        points_data = np.zeros((n_frames, n_identities, n_keypoints, 2))
        
        for frame in range(n_frames):
            for identity in range(n_identities):
                for kp in range(n_keypoints):
                    # HDF5 stores as [y, x]
                    points_data[frame, identity, kp, 0] = 150.0 + frame * 0.1  # y
                    points_data[frame, identity, kp, 1] = 150.0 + frame * 0.1  # x
        
        f.create_dataset('/poseest/points', data=points_data)
        
        # Create features/per_frame group
        per_frame = f.create_group('/features/per_frame')
        per_frame.create_dataset('velocity_x', data=np.random.randn(n_frames, n_identities))
        per_frame.create_dataset('velocity_y', data=np.random.randn(n_frames, n_identities))
        per_frame.create_dataset('speed', data=np.random.rand(n_frames, n_identities))
        
        # Create static_objects/lixit
        lixit_data = np.array([[[100.0, 200.0]]])  # [y, x] format
        f.create_dataset('/static_objects/lixit', data=lixit_data)
    
    return h5_file


@pytest.fixture
def sample_cluster_assignments():
    """Sample cluster assignment CSV data."""
    import pandas as pd
    return pd.DataFrame({
        'bout_id': [0, 1, 2, 3, 4],
        'cluster': [0, 0, 1, 1, 2],
        'video_name': ['test_video.mp4'] * 5,
        'identity': ['0', '0', '1', '1', '0'],
        'start_frame': [100, 200, 300, 400, 500],
        'end_frame': [150, 250, 350, 450, 550]
    })


@pytest.fixture
def mock_video_dir(temp_dir):
    """Create a mock video directory structure."""
    video_dir = os.path.join(temp_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Create a dummy video file (empty, just for path checking)
    video_file = os.path.join(video_dir, "test_video.mp4")
    with open(video_file, 'w') as f:
        f.write("dummy video content")
    
    return video_dir


@pytest.fixture
def mock_annotations_dir(temp_dir, sample_annotation_data):
    """Create a mock annotations directory with sample files."""
    annotations_dir = os.path.join(temp_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    annotation_file = os.path.join(annotations_dir, "test_annotation.json")
    with open(annotation_file, 'w') as f:
        json.dump(sample_annotation_data, f)
    
    return annotations_dir


@pytest.fixture
def mock_features_dir(temp_dir, sample_h5_file):
    """Create a mock features directory structure."""
    features_dir = os.path.join(temp_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    # Copy sample H5 file to features directory
    video_features_dir = os.path.join(features_dir, "test_video")
    os.makedirs(video_features_dir, exist_ok=True)
    
    for identity in ['0', '1']:
        identity_dir = os.path.join(video_features_dir, identity)
        os.makedirs(identity_dir, exist_ok=True)
        shutil.copy(sample_h5_file, os.path.join(identity_dir, "features.h5"))
    
    return features_dir


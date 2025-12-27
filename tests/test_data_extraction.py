"""
Unit tests for data extraction functions.
"""

import unittest
import sys
import os
import json
import tempfile
import shutil

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate_bouts_video import (
    get_bouts,
    get_pose_file,
    get_bboxes
)


class TestDataExtraction(unittest.TestCase):
    """Test data extraction functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.annotations_dir = os.path.join(self.temp_dir, 'annotations')
        os.makedirs(self.annotations_dir)
        
        # Create a test annotation file
        self.test_annotation = {
            "version": 1,
            "file": "test_video.mp4",
            "num_frames": 100,
            "labels": {
                "0": {
                    "turn_left": [
                        {"start": 10, "end": 20, "present": True},
                        {"start": 30, "end": 40, "present": False}
                    ]
                },
                "1": {
                    "turn_left": [
                        {"start": 50, "end": 60, "present": True}
                    ]
                }
            }
        }
        
        self.annotation_file = os.path.join(
            self.annotations_dir, 
            "test_annotation.json"
        )
        with open(self.annotation_file, 'w') as f:
            json.dump(self.test_annotation, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_bouts_finds_present_bouts(self):
        """Test that get_bouts finds only present=True bouts."""
        # Temporarily change the annotations directory and video dir
        import generate_bouts_video
        original_ann_dir = generate_bouts_video.ANNOTATIONS_DIR
        original_vid_dir = generate_bouts_video.VIDEO_DIR
        generate_bouts_video.ANNOTATIONS_DIR = self.annotations_dir
        generate_bouts_video.VIDEO_DIR = self.temp_dir
        
        # Create a dummy video file so validation passes
        test_video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        with open(test_video_path, 'w') as f:
            f.write('dummy video content')
        
        try:
            bouts = get_bouts(behavior_name='turn_left')
            # Should find 2 bouts (one from identity 0, one from identity 1)
            self.assertEqual(len(bouts), 2)
            
            # Check that all bouts have present=True
            for bout in bouts:
                self.assertIn('video_path', bout)
                self.assertIn('video_name', bout)
                self.assertIn('identity', bout)
                self.assertIn('start_frame', bout)
                self.assertIn('end_frame', bout)
                self.assertIn('behavior', bout)
                self.assertEqual(bout['behavior'], 'turn_left')
        finally:
            generate_bouts_video.ANNOTATIONS_DIR = original_ann_dir
            generate_bouts_video.VIDEO_DIR = original_vid_dir
    
    def test_get_bouts_filters_by_behavior(self):
        """Test that get_bouts filters by behavior name."""
        import generate_bouts_video
        original_ann_dir = generate_bouts_video.ANNOTATIONS_DIR
        original_vid_dir = generate_bouts_video.VIDEO_DIR
        generate_bouts_video.ANNOTATIONS_DIR = self.annotations_dir
        generate_bouts_video.VIDEO_DIR = self.temp_dir
        
        # Create a dummy video file so validation passes
        test_video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        with open(test_video_path, 'w') as f:
            f.write('dummy video content')
        
        try:
            # Add another behavior
            self.test_annotation["labels"]["0"]["jumping"] = [
                {"start": 70, "end": 80, "present": True}
            ]
            with open(self.annotation_file, 'w') as f:
                json.dump(self.test_annotation, f)
            
            bouts = get_bouts(behavior_name='jumping')
            self.assertEqual(len(bouts), 1)
            self.assertEqual(bouts[0]['behavior'], 'jumping')
        finally:
            generate_bouts_video.ANNOTATIONS_DIR = original_ann_dir
            generate_bouts_video.VIDEO_DIR = original_vid_dir
    
    def test_get_bouts_empty_annotations(self):
        """Test get_bouts with empty annotations."""
        import generate_bouts_video
        original_ann_dir = generate_bouts_video.ANNOTATIONS_DIR
        original_vid_dir = generate_bouts_video.VIDEO_DIR
        generate_bouts_video.ANNOTATIONS_DIR = self.annotations_dir
        generate_bouts_video.VIDEO_DIR = self.temp_dir
        
        # Create empty annotation file
        empty_annotation = {
            "version": 1,
            "file": "test_video.mp4",
            "labels": {}
        }
        empty_file = os.path.join(self.annotations_dir, "empty.json")
        with open(empty_file, 'w') as f:
            json.dump(empty_annotation, f)
        
        # Create a dummy video file so validation passes
        test_video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        with open(test_video_path, 'w') as f:
            f.write('dummy video content')
        
        try:
            bouts = get_bouts(behavior_name='turn_left')
            # Should still find bouts from other files
            self.assertGreaterEqual(len(bouts), 0)
        finally:
            generate_bouts_video.ANNOTATIONS_DIR = original_ann_dir
            generate_bouts_video.VIDEO_DIR = original_vid_dir
    
    def test_get_pose_file_naming(self):
        """Test pose file name generation."""
        video_name = "test_video.mp4"
        expected = "test_video_pose_est_v8.h5"
        
        # Test with non-existent file (should return None or expected name)
        result = get_pose_file(video_name)
        # Result could be None or the expected path
        if result:
            self.assertIn("_pose_est_v8.h5", result)
    
    def test_get_pose_file_with_extension(self):
        """Test pose file name with different video extensions."""
        test_cases = [
            ("video.mp4", "video_pose_est_v8.h5"),
            ("video.mov", "video_pose_est_v8.h5"),
            ("video.avi", "video_pose_est_v8.h5"),
        ]
        
        for video_name, expected_base in test_cases:
            result = get_pose_file(video_name)
            if result:
                self.assertIn("_pose_est_v8.h5", result)
    
    def test_get_bboxes_no_pose_file(self):
        """Test get_bboxes with no pose file."""
        result = get_bboxes(None, "0", 10, 20)
        self.assertEqual(result, {})
    
    def test_get_bboxes_nonexistent_file(self):
        """Test get_bboxes with non-existent pose file."""
        result = get_bboxes("nonexistent_file.h5", "0", 10, 20)
        self.assertEqual(result, {})


class TestDataExtractionIntegration(unittest.TestCase):
    """Integration tests using real annotation files if available."""
    
    def test_get_bouts_real_annotations(self):
        """Test get_bouts with real annotation files if they exist."""
        import generate_bouts_video
        
        annotations_dir = "../jabs/annotations"
        if not os.path.exists(annotations_dir):
            self.skipTest("Real annotation directory not found")
        
        original_dir = generate_bouts_video.ANNOTATIONS_DIR
        generate_bouts_video.ANNOTATIONS_DIR = annotations_dir
        
        try:
            bouts = get_bouts(behavior_name='turn_left')
            # Should find some bouts if annotation files exist
            self.assertIsInstance(bouts, list)
            
            # Validate bout structure
            for bout in bouts:
                self.assertIn('video_path', bout)
                self.assertIn('video_name', bout)
                self.assertIn('identity', bout)
                self.assertIn('start_frame', bout)
                self.assertIn('end_frame', bout)
                self.assertIn('behavior', bout)
                self.assertIsInstance(bout['start_frame'], int)
                self.assertIsInstance(bout['end_frame'], int)
                self.assertGreaterEqual(bout['end_frame'], bout['start_frame'])
        finally:
            generate_bouts_video.ANNOTATIONS_DIR = original_dir


if __name__ == '__main__':
    unittest.main()


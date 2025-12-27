"""
Unit tests for validation functions.
"""

import unittest
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate_bouts_video import (
    validate_video_file,
    validate_frame_range,
    get_video_frame_count
)


class TestValidation(unittest.TestCase):
    """Test validation functions."""
    
    def test_validate_video_file_exists(self):
        """Test validation of existing video file."""
        # Check if test video exists
        test_video = "../org-3-uploads-stage.study_428.cage_4507.2025-07-13.05.44.mp4"
        if os.path.exists(test_video):
            self.assertTrue(validate_video_file(test_video))
        else:
            # Skip if test file doesn't exist
            self.skipTest("Test video file not found")
    
    def test_validate_video_file_not_exists(self):
        """Test validation of non-existent video file."""
        self.assertFalse(validate_video_file("nonexistent_video.mp4"))
    
    def test_validate_video_file_directory(self):
        """Test validation fails for directory."""
        test_dir = "../jabs"
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            self.assertFalse(validate_video_file(test_dir))
    
    def test_validate_frame_range_valid(self):
        """Test validation of valid frame range."""
        is_valid, msg = validate_frame_range(10, 20)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")
    
    def test_validate_frame_range_negative_start(self):
        """Test validation fails for negative start frame."""
        is_valid, msg = validate_frame_range(-1, 20)
        self.assertFalse(is_valid)
        self.assertIn("must be >= 0", msg)
    
    def test_validate_frame_range_end_before_start(self):
        """Test validation fails when end < start."""
        is_valid, msg = validate_frame_range(20, 10)
        self.assertFalse(is_valid)
        self.assertIn("must be >=", msg)
    
    def test_validate_frame_range_with_video_length(self):
        """Test validation with video length constraint."""
        is_valid, msg = validate_frame_range(10, 20, num_frames=30)
        self.assertTrue(is_valid)
        
        is_valid, msg = validate_frame_range(10, 40, num_frames=30)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "clamped")
    
    def test_validate_frame_range_exceeds_video(self):
        """Test validation when start frame exceeds video length."""
        is_valid, msg = validate_frame_range(50, 60, num_frames=30)
        self.assertFalse(is_valid)
        self.assertIn("exceeds video length", msg)
    
    def test_validate_frame_range_single_frame(self):
        """Test validation of single frame range."""
        is_valid, msg = validate_frame_range(10, 10)
        self.assertTrue(is_valid)
    
    def test_get_video_frame_count_nonexistent(self):
        """Test frame count for non-existent video."""
        result = get_video_frame_count("nonexistent_video.mp4")
        self.assertIsNone(result)
    
    def test_get_video_frame_count_existing(self):
        """Test frame count for existing video (if ffprobe available)."""
        test_video = "../org-3-uploads-stage.study_428.cage_4507.2025-07-13.05.44.mp4"
        if os.path.exists(test_video):
            result = get_video_frame_count(test_video)
            # Result could be None if ffprobe not available, or an integer
            if result is not None:
                self.assertIsInstance(result, int)
                self.assertGreater(result, 0)
        else:
            self.skipTest("Test video file not found")


if __name__ == '__main__':
    unittest.main()


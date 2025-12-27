"""
Unit tests for video processing functions.
"""

import unittest
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate_bouts_video import (
    sec_to_ass_time
)


class TestVideoProcessing(unittest.TestCase):
    """Test video processing functions."""
    
    def test_sec_to_ass_time_zero(self):
        """Test time conversion for zero seconds."""
        result = sec_to_ass_time(0.0)
        self.assertEqual(result, "0:00:00.00")
    
    def test_sec_to_ass_time_seconds(self):
        """Test time conversion for seconds."""
        result = sec_to_ass_time(5.5)
        self.assertEqual(result, "0:00:05.50")
    
    def test_sec_to_ass_time_minutes(self):
        """Test time conversion for minutes."""
        result = sec_to_ass_time(65.25)
        self.assertEqual(result, "0:01:05.25")
    
    def test_sec_to_ass_time_hours(self):
        """Test time conversion for hours."""
        result = sec_to_ass_time(3661.75)
        self.assertEqual(result, "1:01:01.75")
    
    def test_sec_to_ass_time_fractional(self):
        """Test time conversion with fractional seconds."""
        result = sec_to_ass_time(1.23)
        self.assertEqual(result, "0:00:01.23")
    
    def test_sec_to_ass_time_large(self):
        """Test time conversion for large values."""
        result = sec_to_ass_time(12345.67)
        hours = 12345 // 3600
        minutes = (12345 % 3600) // 60
        seconds = 12345 % 60
        centiseconds = int((12345.67 * 100) % 100)
        expected = f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
        self.assertEqual(result, expected)
    


if __name__ == '__main__':
    unittest.main()


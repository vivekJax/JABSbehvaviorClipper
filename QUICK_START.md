# Quick Start Guide

## Running the Application

### Basic Usage
```bash
python3 generate_bouts_video.py
```

**Note:** By default, the application uses **multicore processing** with `n-1` CPU cores (leaves one core free for system responsiveness).

### With Options
```bash
# Extract specific behavior
python3 generate_bouts_video.py --behavior jumping

# Custom output file
python3 generate_bouts_video.py --output my_video.mp4

# Verbose logging (shows bounding box extraction details)
python3 generate_bouts_video.py --verbose
```

## Running Tests

### Quick Test Run
```bash
python3 -m unittest discover tests -v
```

### Run Specific Test File
```bash
python3 -m unittest tests.test_validation -v
python3 -m unittest tests.test_data_extraction -v
python3 -m unittest tests.test_video_processing -v
```

### Using pytest (if installed)
```bash
pip install pytest
pytest tests/ -v
```

## Test Results

All 25 tests should pass:
- ✅ 12 validation tests
- ✅ 10 data extraction tests  
- ✅ 7 video processing tests

## Documentation

- **README.md**: Full documentation and usage guide
- **TESTING.md**: Detailed testing instructions
- **This file**: Quick reference

## Common Commands

```bash
# Generate video
python3 generate_bouts_video.py --behavior turn_left

# Run all tests
python3 -m unittest discover tests -v

# Run with verbose output
python3 generate_bouts_video.py --verbose

# Keep temp files for debugging (useful for checking bounding boxes)
python3 generate_bouts_video.py --keep-temp --verbose

# Use custom number of workers (default is CPU cores - 1)
python3 generate_bouts_video.py --workers 4

# Use all CPU cores
python3 generate_bouts_video.py --workers $(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())")

# Disable parallel processing (sequential)
python3 generate_bouts_video.py --workers 1
```


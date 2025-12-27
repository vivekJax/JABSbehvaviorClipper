# Behavior Video Generator

A Python tool for generating video montages from labeled behavior bouts in annotation files. Extracts video clips where specific behaviors are marked as `present=True`, adds bounding boxes from pose estimation data, and concatenates them into a single output video.

## Features

- Extract behavior bouts from JSON annotation files
- Generate video clips with bounding box overlays
- **Parallel processing** for faster clip extraction (multiprocessing)
- Support for multiple behaviors (configurable)
- Automatic bounding box extraction from HDF5 pose estimation files
- Metadata overlays (video file, identity ID, frame ranges)
- Graceful handling of missing pose files
- Comprehensive logging and error handling

## Requirements

### Python Dependencies
- Python 3.6+
- Standard library only (no external Python packages required)

### External Tools
- **ffmpeg**: For video processing and encoding
  - Install: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)
- **h5dump**: For extracting bounding boxes from HDF5 files (optional)
  - Part of HDF5 tools: `brew install hdf5` (macOS) or `apt-get install hdf5-tools` (Linux)
  - If not installed, the script will continue without bounding boxes

### System Requirements
- macOS or Linux (font path is macOS-specific, adjust `FONT_FILE` for other systems)

## Installation

1. Clone or download this repository
2. Ensure ffmpeg is installed: `ffmpeg -version`
3. (Optional) Install HDF5 tools for bounding box extraction: `h5dump -V`

## Usage

### Basic Usage

Generate a video for the default behavior (`turn_left`):

```bash
python3 generate_bouts_video.py
```

### Command-Line Options

```bash
python3 generate_bouts_video.py [OPTIONS]
```

**Options:**

- `--behavior BEHAVIOR`: Behavior name to extract (default: `turn_left`)
- `--output FILENAME`: Output video filename (default: `all_bouts.mp4`)
- `--annotations-dir DIR`: Directory containing annotation JSON files (default: `jabs/annotations`)
- `--video-dir DIR`: Directory containing video files (default: `.`)
- `--keep-temp`: Keep temporary files after processing (for debugging)
- `--verbose`: Enable verbose logging (DEBUG level)
- `--workers N`: Number of parallel workers for clip extraction (default: number of CPU cores)
- `--help`: Show help message

### Examples

```bash
# Extract turn_left behavior bouts
python3 generate_bouts_video.py --behavior turn_left

# Extract jumping behavior with custom output
python3 generate_bouts_video.py --behavior jumping --output jumping_bouts.mp4

# Verbose mode for debugging
python3 generate_bouts_video.py --behavior turn_left --verbose

# Custom directories
python3 generate_bouts_video.py --annotations-dir /path/to/annotations --video-dir /path/to/videos

# Parallel processing with 4 workers
python3 generate_bouts_video.py --behavior turn_left --workers 4

# Disable parallel processing (use 1 worker)
python3 generate_bouts_video.py --behavior turn_left --workers 1
```

## Project Structure

```
.
├── generate_bouts_video.py    # Main script
├── jabs/
│   ├── annotations/             # JSON annotation files
│   │   └── *.json
│   └── project.json            # Project configuration
├── *.mp4                       # Source video files
├── *_pose_est_v8.h5           # Pose estimation files (optional)
├── all_bouts.mp4               # Output video
├── tests/                       # Test suite
│   ├── test_validation.py
│   ├── test_data_extraction.py
│   └── test_video_processing.py
└── README.md                    # This file
```

## Annotation File Format

Annotation files should be JSON files with the following structure:

```json
{
  "version": 1,
  "file": "video_filename.mp4",
  "num_frames": 1800,
  "labels": {
    "identity_id": {
      "behavior_name": [
        {
          "start": 12,
          "end": 32,
          "present": true
        }
      ]
    }
  }
}
```

**Key Fields:**
- `file`: Name of the video file (must exist in video directory)
- `num_frames`: Total number of frames in the video (optional, used for validation)
- `labels`: Dictionary mapping identity IDs to behaviors
- `behavior_name`: Name of the behavior (e.g., "turn_left", "jumping")
- `start`/`end`: Frame numbers for the behavior bout
- `present`: Boolean indicating if the behavior is present (only `true` bouts are extracted)

## Pose Estimation Files

Pose estimation files should be HDF5 files named: `<video_basename>_pose_est_v8.h5`

The script expects bounding box data at path `/poseest/bbox` with shape `(frame, identity, point, 0)` where:
- `point=0`: Top-left corner (x, y)
- `point=1`: Bottom-right corner (x, y)

**Bounding Box Rendering:**
- Boxes are drawn as **yellow rectangles** with 3-pixel thick outlines
- Each box is labeled with "Mouse {identity_id}" text above it
- Boxes are drawn only for frames where bounding box data exists
- Invalid boxes (negative coordinates, zero width/height) are automatically skipped
- Frame numbering uses absolute input stream frame numbers (works correctly with `-ss` seeking)

If pose files are missing, the script will continue without bounding boxes and log a debug message.

## Output

The script generates:
1. **Temporary clips**: Individual video clips in `temp_clips/` directory (deleted by default)
2. **Final video**: Concatenated video with all bouts (`all_bouts.mp4` by default)

Each clip includes:
- **Bounding boxes**: Yellow outline rectangles around tracked mice (when pose estimation data is available)
- **Identity labels**: Text labels showing which mouse (ID) is performing the behavior
- **Metadata overlay**: Source video file, identity ID, and frame ranges displayed in the top-left corner

## Logging

The script uses Python's logging module with two levels:

- **INFO** (default): Progress updates, warnings, and errors
- **DEBUG** (--verbose): Detailed information including:
  - Bounding box extraction counts per bout
  - Number of filters applied per clip
  - Example filter commands for debugging
  - Pose file lookup results

Log format: `YYYY-MM-DD HH:MM:SS - LEVEL - MESSAGE`

**Tip**: Use `--verbose` when troubleshooting bounding box issues to see detailed extraction and filter information.

## Error Handling

The script handles various error conditions gracefully:

- Missing video files: Logs warning and skips
- Missing pose files: Continues without bounding boxes
- Invalid frame ranges: Validates and clamps to video bounds
- Missing external tools: Provides helpful error messages
- Invalid JSON: Logs error and continues with other files

## Configuration

Default configuration (can be overridden via command-line):

```python
ANNOTATIONS_DIR = 'jabs/annotations'
VIDEO_DIR = '.'
OUTPUT_FILENAME = 'all_bouts.mp4'
TEMP_DIR = 'temp_clips'
FPS = 30.0
FONT_FILE = '/System/Library/Fonts/Helvetica.ttc'  # macOS-specific
DEFAULT_BEHAVIOR = 'turn_left'
```

## Performance

### Parallel Processing

The script supports parallel processing for clip extraction, which can significantly speed up processing when you have many clips:

- **Default**: Automatically uses all available CPU cores
- **Custom**: Specify number of workers with `--workers N`
- **Sequential**: Use `--workers 1` to disable parallel processing

**Example performance improvements:**
- 10 clips, 4 cores: ~2.5x faster than sequential
- 50 clips, 8 cores: ~6-7x faster than sequential

**When to use parallel processing:**
- ✅ Many clips to process (>5)
- ✅ Multiple CPU cores available
- ✅ Sufficient RAM (each worker uses memory)

**When to use sequential processing:**
- ⚠️ Limited RAM
- ⚠️ Very few clips (<3)
- ⚠️ Debugging (easier to track progress)

## Troubleshooting

### No bouts found
- Check that annotation files exist in `jabs/annotations/`
- Verify annotation files have `present: true` bouts for the specified behavior
- Use `--verbose` to see detailed processing information

### No bounding boxes appearing in video
- Verify pose estimation files exist (named `*_pose_est_v8.h5`)
- Check that `h5dump` is installed: `h5dump -V`
- Use `--verbose` to see bounding box extraction details:
  ```bash
  python3 generate_bouts_video.py --verbose --keep-temp
  ```
- Check debug output for:
  - "Extracted X bounding boxes" messages
  - "Adding X bounding box filters" messages
  - Any warnings about missing pose files or failed extractions
- Verify bounding box coordinates are valid (not negative or zero width/height)
- Bounding boxes are drawn as **yellow rectangles** with **3-pixel thick outlines**
- Each box includes a label showing the mouse identity ID above it

### Video encoding errors
- Ensure `ffmpeg` is installed and in PATH: `ffmpeg -version`
- Check that source video files are valid and accessible
- Verify font file exists (adjust `FONT_FILE` for your system)

### Performance issues
- Use `--keep-temp` to preserve intermediate files for debugging
- Check disk space (temporary clips can be large)
- Consider processing fewer bouts at a time

## Testing

See [TESTING.md](TESTING.md) for detailed testing instructions.

Quick test run:
```bash
python3 -m pytest tests/ -v
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]


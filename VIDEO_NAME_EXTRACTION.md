# Video Name Extraction Guide

Since text rendered in videos cannot be directly selected, we've implemented two methods to make video names easily accessible for copy/paste:

## Method 1: Video Name Mapping File (Recommended)

A text file is automatically created alongside your output video with the name format: `<output_video_name>_video_names.txt`

### Format
```
# Video Name Mapping - Copy/paste video names from this file
# Format: Clip_Index | Video_Name | Mouse_ID | Frame_Range
# ================================================================================

00000 | org-3-uploads-stage.study_410.cage_4480.2025-08-02.03.41.mp4 | Mouse 0 | Frames 59-73
00001 | org-3-uploads-stage.study_410.cage_4480.2025-08-02.03.41.mp4 | Mouse 0 | Frames 942-967
...
```

### Usage
1. Open the `.txt` file in any text editor
2. Select and copy the video name(s) you need
3. Paste wherever needed

### Location
The file is created in the same directory as your output video:
```bash
python3 generate_bouts_video.py --output my_video.mp4
# Creates: my_video_video_names.txt
```

## Method 2: Video Metadata

Each video clip has the video name embedded in its metadata, which can be extracted using various tools.

### Extract with ffprobe (Command Line)
```bash
# Extract title (video name)
ffprobe -v quiet -show_entries format_tags=title -of default=noprint_wrappers=1:nokey=1 video.mp4

# Extract all metadata
ffprobe -v quiet -show_entries format_tags -of default=noprint_wrappers=1 video.mp4
```

### Extract with exiftool (If Installed)
```bash
# Install: brew install exiftool (macOS) or apt-get install libimage-exiftool-perl (Linux)
exiftool -Title video.mp4
exiftool -Comment video.mp4
```

### Extract in Python
```python
import subprocess
import json

def get_video_metadata(video_path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    return data['format'].get('tags', {})

# Usage
metadata = get_video_metadata('temp_clips/clip_00000.mp4')
video_name = metadata.get('title', 'Unknown')
print(f"Video name: {video_name}")
```

### Metadata Fields
- **title**: Full video filename (e.g., `org-3-uploads-stage.study_428.cage_4507.2025-07-13.05.44.mp4`)
- **comment**: Additional info including source video, mouse ID, and frame range

## Quick Reference

### Get Video Name from Mapping File
```bash
# View all video names
cat output_video_names.txt

# Extract just video names (no clip index)
cat output_video_names.txt | grep -v "^#" | awk -F' | ' '{print $2}'

# Find video name for specific clip index
grep "^00042" output_video_names.txt | awk -F' | ' '{print $2}'
```

### Get Video Name from Metadata
```bash
# From temp clip
ffprobe -v quiet -show_entries format_tags=title -of default=noprint_wrappers=1:nokey=1 temp_clips/clip_00000.mp4

# From final output video (note: concatenation may not preserve all metadata)
ffprobe -v quiet -show_entries format_tags=title -of default=noprint_wrappers=1:nokey=1 all_bouts.mp4
```

## Tips

1. **Mapping File is Best**: The text file is the easiest way to copy/paste video names
2. **Keep Temp Files**: Use `--keep-temp` to preserve individual clips with their metadata
3. **Batch Extraction**: Use the mapping file with scripts to extract multiple names at once

## Example Workflow

```bash
# Generate video with mapping file
python3 generate_bouts_video.py --behavior turn_left --output my_video.mp4

# Open mapping file
open my_video_video_names.txt  # macOS
# or
cat my_video_video_names.txt   # Linux

# Copy video name from the file
# Paste into your document/script/etc.
```


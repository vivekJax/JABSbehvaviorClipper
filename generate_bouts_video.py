#!/usr/bin/env python3
"""
Behavior Video Generator

Generates a video montage from labeled behavior bouts where present=True.
Extracts video clips, adds bounding boxes from pose estimation data,
and concatenates them into a single output video.
"""

import json
import os
import subprocess
import glob
import shutil
import re
import math
import argparse
import logging
import multiprocessing
from typing import List, Dict, Optional, Tuple
from functools import partial

# ============================================================================
# Configuration
# ============================================================================

ANNOTATIONS_DIR = 'jabs/annotations'
VIDEO_DIR = '.'  # Root directory where videos are located
OUTPUT_FILENAME = 'all_bouts.mp4'
TEMP_DIR = 'temp_clips'
FPS = 30.0
# Font path for macOS. Adjust if running on a different OS or if font missing.
FONT_FILE = '/System/Library/Fonts/Helvetica.ttc'
DEFAULT_BEHAVIOR = 'turn_left'

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.
    
    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ============================================================================
# Validation Functions
# ============================================================================

def validate_video_file(video_path: str) -> bool:
    """Validate that a video file exists and is accessible.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if file exists, False otherwise
    """
    if not os.path.exists(video_path):
        logging.warning(f"Video file not found: {video_path}")
        return False
    if not os.path.isfile(video_path):
        logging.warning(f"Path is not a file: {video_path}")
        return False
    return True

def validate_frame_range(start_frame: int, end_frame: int, num_frames: Optional[int] = None) -> Tuple[bool, str]:
    """Validate that frame range is valid.
    
    Args:
        start_frame: Starting frame number
        end_frame: Ending frame number
        num_frames: Total number of frames in video (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if start_frame < 0:
        return False, f"Start frame must be >= 0, got {start_frame}"
    if end_frame < start_frame:
        return False, f"End frame ({end_frame}) must be >= start frame ({start_frame})"
    if num_frames is not None:
        if start_frame >= num_frames:
            return False, f"Start frame ({start_frame}) exceeds video length ({num_frames})"
        if end_frame >= num_frames:
            logging.warning(f"End frame ({end_frame}) exceeds video length ({num_frames}), clamping to {num_frames - 1}")
            return True, "clamped"
    return True, ""

def get_video_frame_count(video_path: str) -> Optional[int]:
    """Get the number of frames in a video file using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Number of frames, or None if unable to determine
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception as e:
        logging.debug(f"Could not determine frame count for {video_path}: {e}")
    return None

# ============================================================================
# Data Extraction Functions
# ============================================================================

def get_bouts(behavior_name: str = DEFAULT_BEHAVIOR) -> List[Dict]:
    """Extract behavior bouts from annotation files where present=True.
    
    Args:
        behavior_name: Name of the behavior to extract (default: 'turn_left')
        
    Returns:
        List of bout dictionaries, each containing:
            - video_path: Full path to video file
            - video_name: Video filename
            - identity: Identity ID
            - start_frame: Starting frame number
            - end_frame: Ending frame number
            - behavior: Behavior name
    """
    bouts = []
    json_files = glob.glob(os.path.join(ANNOTATIONS_DIR, '*.json'))
    
    logging.info(f"Found {len(json_files)} annotation files.")
    
    if not json_files:
        logging.warning(f"No annotation files found in {ANNOTATIONS_DIR}")

    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            video_filename = data.get('file')
            if not video_filename:
                logging.warning(f"No 'file' field in {json_path}, skipping")
                continue
            
            video_path = os.path.join(VIDEO_DIR, video_filename)
            if not validate_video_file(video_path):
                logging.warning(f"Skipping {json_path} due to missing video file")
                continue

            # Get video frame count for validation
            num_frames = data.get('num_frames')
            if num_frames is None:
                num_frames = get_video_frame_count(video_path)

            labels = data.get('labels', {})
            if not labels:
                logging.debug(f"No labels found in {json_path}")
                continue
                
            for identity_id, behaviors in labels.items():
                behavior_bouts = behaviors.get(behavior_name, [])
                if not behavior_bouts:
                    continue
                    
                for bout in behavior_bouts:
                    if bout.get('present') is True:
                        start_frame = bout.get('start')
                        end_frame = bout.get('end')
                        
                        if start_frame is None or end_frame is None:
                            logging.warning(f"Invalid bout in {json_path}, identity {identity_id}: missing start/end")
                            continue
                        
                        # Validate frame range
                        is_valid, msg = validate_frame_range(start_frame, end_frame, num_frames)
                        if not is_valid:
                            logging.warning(f"Invalid frame range in {json_path}, identity {identity_id}, "
                                         f"frames {start_frame}-{end_frame}: {msg}")
                            continue
                        
                        # Clamp end frame if needed
                        if msg == "clamped" and num_frames:
                            end_frame = num_frames - 1
                        
                        bouts.append({
                            'video_path': video_path,
                            'video_name': video_filename,
                            'identity': identity_id,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'behavior': behavior_name
                        })
                        
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON file {json_path}: {e}")
        except Exception as e:
            logging.error(f"Error processing {json_path}: {e}", exc_info=True)
            
    logging.info(f"Extracted {len(bouts)} bouts with present=True for behavior '{behavior_name}'")
    return bouts

def get_pose_file(video_name: str) -> Optional[str]:
    """Find the pose estimation H5 file for a given video.
    
    Args:
        video_name: Name of the video file
        
    Returns:
        Path to pose estimation file, or None if not found
    """
    # Expecting <video_name_stem>_pose_est_v8.h5
    basename = os.path.splitext(video_name)[0]
    pose_name = f"{basename}_pose_est_v8.h5"
    pose_path = os.path.join(VIDEO_DIR, pose_name)
    
    if os.path.exists(pose_path):
        return pose_path
    
    # Check current directory if VIDEO_DIR didn't have it (e.g. if files are flat)
    if os.path.exists(pose_name):
        return pose_name
        
    return None

def get_bboxes(pose_file: Optional[str], identity_id: str, start_frame: int, end_frame: int) -> Dict[int, Dict]:
    """Extract bounding boxes from pose estimation H5 file.
    
    Args:
        pose_file: Path to pose estimation H5 file, or None
        identity_id: Identity ID to extract boxes for
        start_frame: Starting frame number
        end_frame: Ending frame number
        
    Returns:
        Dictionary mapping frame numbers to bounding box coordinates:
            {frame: {'x': x1, 'y': y1, 'x2': x2, 'y2': y2}}
    """
    if not pose_file:
        return {}
    
    if not os.path.exists(pose_file):
        logging.debug(f"Pose file not found: {pose_file}")
        return {}
        
    # h5dump -d /poseest/bbox -s "start,id,0,0" -c "count,1,2,2" file
    # Count is end - start + 1
    count = end_frame - start_frame + 1
    start_idx = f"{start_frame},{identity_id},0,0"
    count_idx = f"{count},1,2,2"
    
    cmd = ['h5dump', '-d', '/poseest/bbox', '-s', start_idx, '-c', count_idx, pose_file]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logging.warning(f"h5dump failed for {pose_file}: {result.stderr}")
            return {}
            
        # Parse output
        # (1426,0,0,0): 1414.47, 310.807,
        # (1426,0,1,0): 1689.78, 611.693,
        
        bboxes = {}  # frame -> {p0: (x,y), p1: (x,y)}
        
        lines = result.stdout.splitlines()
        
        # Regex to match: (frame,id,point,0): x, y
        # We need to handle potential variable formatting
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
        
        # Consolidate into x1, y1, x2, y2
        final_bboxes = {}
        for f, pts in bboxes.items():
            if 0 in pts and 1 in pts:
                final_bboxes[f] = {
                    'x': pts[0][0], 'y': pts[0][1],
                    'x2': pts[1][0], 'y2': pts[1][1]
                }
        
        logging.debug(f"Extracted {len(final_bboxes)} bounding boxes from {pose_file}")
        return final_bboxes
        
    except FileNotFoundError:
        logging.warning(f"h5dump command not found. Install HDF5 tools to extract bounding boxes.")
        return {}
    except Exception as e:
        logging.warning(f"Error reading bboxes from {pose_file}: {e}")
        return {}

# ============================================================================
# Video Processing Functions
# ============================================================================

def sec_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS subtitle time format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Time string in format H:MM:SS.CC
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds * 100) % 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def extract_clip_worker(args_tuple: Tuple[Dict, int]) -> Optional[Tuple[int, str]]:
    """Worker function for parallel clip extraction.
    
    Args:
        args_tuple: Tuple of (bout, index)
        
    Returns:
        Tuple of (index, clip_name) or None if failed
    """
    bout, index = args_tuple
    clip_name = extract_clip(bout, index)
    if clip_name:
        return (index, clip_name)
    return None

def extract_clip(bout: Dict, index: int) -> Optional[str]:
    """Extract a video clip for a behavior bout with overlays.
    
    Args:
        bout: Bout dictionary with video and frame information
        index: Index of the bout for naming
        
    Returns:
        Filename of the generated clip, or None if extraction failed
    """
    start_time = bout['start_frame'] / FPS
    duration = (bout['end_frame'] - bout['start_frame'] + 1) / FPS
    
    if duration <= 0:
        logging.warning(f"Invalid duration for bout {index}: {duration}")
        return None

    clip_name = f"clip_{index:05d}.mp4"
    output_path = os.path.join(TEMP_DIR, clip_name)
    
    # Get Pose Data (gracefully handle missing files)
    pose_file = get_pose_file(bout['video_name'])
    bboxes = get_bboxes(pose_file, bout['identity'], bout['start_frame'], bout['end_frame'])
    
    if not pose_file:
        logging.debug(f"No pose file found for {bout['video_name']}, continuing without bounding boxes")
    elif not bboxes:
        logging.debug(f"No bounding boxes extracted for {bout['video_name']}, continuing without boxes")
    else:
        logging.debug(f"Extracted {len(bboxes)} bounding boxes for bout {index} (frames {bout['start_frame']}-{bout['end_frame']})")
    
    # Overlay text info (bottom center)
    # Display video name at bottom center
    video_name_short = os.path.splitext(bout['video_name'])[0]  # Remove .mp4 extension
    text_line1 = f"{video_name_short}"
    
    # Clean text for ffmpeg (escape special characters)
    text_line1_clean = text_line1.replace(":", "\\:").replace("'", "\\'")
    
    # Build filter chain
    # Video name at bottom center: x=(w-text_w)/2 centers horizontally, y=h-th-10 positions at bottom with 10px margin
    filters = [f"drawtext=fontfile={FONT_FILE}:text='{text_line1_clean}':fontcolor=white:fontsize=20:box=1:boxcolor=black@0.7:boxborderw=3:x=(w-text_w)/2:y=h-th-10"]
    
    # Add bounding boxes using drawbox filters
    if bboxes:
        start_f = bout['start_frame']
        end_f = bout['end_frame']
        
        # Create drawbox filters for each frame with bounding boxes
        # Use enable expressions to show boxes only for frames that have them
        # Note: When using -ss before -i, frame numbers in filters start from 0
        box_filters = []
        for frame_num in range(start_f, end_f + 1):
            if frame_num in bboxes:
                box = bboxes[frame_num]
                x1, y1, x2, y2 = box['x'], box['y'], box['x2'], box['y2']
                
                # Skip invalid boxes
                if x1 < 0 or y1 < 0:
                    continue
                
                w = x2 - x1
                h = y2 - y1
                
                # Skip if width or height is invalid
                if w <= 0 or h <= 0:
                    continue
                
                # When using -ss before -i, 'n' in filters refers to OUTPUT frame numbers (0-indexed from clip start)
                # So we need to use relative frame numbers: frame_num - start_f
                rel_frame = frame_num - start_f
                
                # Draw outline box (t=3 means thickness, outline mode)
                # Color yellow: yellow@1.0 (named color with full opacity)
                # Enable expression: only draw on the specific relative frame
                box_filter = f"drawbox=x={int(x1)}:y={int(y1)}:w={int(w)}:h={int(h)}:color=yellow@1.0:t=3:enable=eq(n\\,{rel_frame})"
                box_filters.append(box_filter)
                
                # Add label above box
                label_text = f"Mouse {bout['identity']}"
                label_text_clean = label_text.replace("'", "\\'")
                label_filter = f"drawtext=fontfile={FONT_FILE}:text='{label_text_clean}':fontcolor=white:fontsize=20:x={int(x1)}:y={int(max(0, y1-30))}:enable=eq(n\\,{rel_frame})"
                box_filters.append(label_filter)
        
        if box_filters:
            logging.debug(f"Adding {len(box_filters)} bounding box filters for bout {index} (frames {start_f}-{end_f})")
            # Log first filter as example
            if len(box_filters) > 0:
                logging.debug(f"Example box filter: {box_filters[0][:100]}...")
        filters.extend(box_filters)
    
    filter_chain = ",".join(filters)
    
    # Use -ss before -i for faster seeking
    # When -ss is before -i, 'n' in filters refers to OUTPUT frame numbers (0-indexed from clip start)
    # So we use relative frame numbers (frame_num - start_f) in enable expressions
    cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-i', bout['video_path'],
        '-t', str(duration),
        '-vf', filter_chain,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-r', '30',
        '-preset', 'fast',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='replace')
            logging.error(f"Error creating clip {index} from {bout['video_name']}: {error_msg}")
            return None
        logging.debug(f"Successfully created clip {index}: {clip_name}")
        return clip_name
    except FileNotFoundError:
        logging.error("ffmpeg command not found. Please install ffmpeg.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error creating clip {index}: {e}", exc_info=True)
        return None

# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Main entry point for the behavior video generator."""
    global ANNOTATIONS_DIR, VIDEO_DIR, OUTPUT_FILENAME
    
    parser = argparse.ArgumentParser(
        description='Generate a video montage from labeled behavior bouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s --behavior turn_left
  %(prog)s --behavior jumping --output jumping_bouts.mp4
  %(prog)s --behavior turn_left --verbose
        """
    )
    parser.add_argument(
        '--behavior',
        type=str,
        default=DEFAULT_BEHAVIOR,
        help=f'Behavior name to extract (default: {DEFAULT_BEHAVIOR})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_FILENAME,
        help=f'Output video filename (default: {OUTPUT_FILENAME})'
    )
    parser.add_argument(
        '--annotations-dir',
        type=str,
        default=ANNOTATIONS_DIR,
        help=f'Directory containing annotation JSON files (default: {ANNOTATIONS_DIR})'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default=VIDEO_DIR,
        help=f'Directory containing video files (default: {VIDEO_DIR})'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary files after processing'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers for clip extraction (default: CPU cores - 1)'
    )
    
    args = parser.parse_args()
    
    # Update global config from args
    ANNOTATIONS_DIR = args.annotations_dir
    VIDEO_DIR = args.video_dir
    OUTPUT_FILENAME = args.output
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    logging.info(f"Starting behavior video generation for behavior: '{args.behavior}'")
    logging.info(f"Annotations directory: {ANNOTATIONS_DIR}")
    logging.info(f"Video directory: {VIDEO_DIR}")
    logging.info(f"Output file: {OUTPUT_FILENAME}")
    
    # Create temp directory
    if os.path.exists(TEMP_DIR):
        logging.debug(f"Removing existing temp directory: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    logging.debug(f"Created temp directory: {TEMP_DIR}")
    
    # Extract bouts
    logging.info("Scanning for bouts...")
    bouts = get_bouts(behavior_name=args.behavior)
    
    if not bouts:
        logging.warning("No bouts found. Exiting.")
        if not args.keep_temp and os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        return

    # Sort bouts by video name and start frame
    bouts.sort(key=lambda x: (x['video_name'], x['start_frame']))
    
    concat_list_path = os.path.join(TEMP_DIR, 'concat_list.txt')
    created_clips = []
    
    # Extract clips (parallel or sequential)
    # Default to n-1 cores to leave one core free for system responsiveness
    default_workers = max(1, multiprocessing.cpu_count() - 1)
    num_workers = args.workers if args.workers is not None else default_workers
    use_parallel = num_workers > 1 and len(bouts) > 1
    
    if use_parallel:
        logging.info(f"Extracting {len(bouts)} clips using {num_workers} parallel workers...")
        
        # Prepare arguments for parallel processing
        clip_args = [(bout, i) for i, bout in enumerate(bouts)]
        
        # Use multiprocessing Pool
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(extract_clip_worker, clip_args)
        
        # Process results and maintain order
        created_clips_dict = {}
        for i, result in enumerate(results):
            if result is not None:
                idx, clip_name = result
                created_clips_dict[idx] = clip_name
                logging.info(f"Completed {idx+1}/{len(bouts)}: {bouts[idx]['video_name']} "
                           f"frames {bouts[idx]['start_frame']}-{bouts[idx]['end_frame']} (ID: {bouts[idx]['identity']})")
            else:
                logging.warning(f"Failed to create clip for bout {i+1}")
        
        # Sort by index to maintain order
        created_clips = [created_clips_dict[i] for i in sorted(created_clips_dict.keys())]
    else:
        logging.info(f"Extracting {len(bouts)} clips sequentially...")
        for i, bout in enumerate(bouts):
            logging.info(f"Processing {i+1}/{len(bouts)}: {bout['video_name']} "
                        f"frames {bout['start_frame']}-{bout['end_frame']} (ID: {bout['identity']})")
            clip_name = extract_clip(bout, i)
            if clip_name:
                created_clips.append(clip_name)
            else:
                logging.warning(f"Failed to create clip for bout {i+1}")
    
    if not created_clips:
        logging.error("No clips were successfully created. Exiting.")
        if not args.keep_temp and os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        return
    
    # Create concat list
    logging.info("Creating concat list...")
    with open(concat_list_path, 'w') as f:
        for clip in created_clips:
            # Use absolute path for concat list to avoid issues
            clip_path = os.path.join(TEMP_DIR, clip)
            f.write(f"file '{os.path.abspath(clip_path)}'\n")
    
    # Concatenate clips
    logging.info(f"Concatenating {len(created_clips)} clips...")
    concat_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',
        OUTPUT_FILENAME
    ]
    
    try:
        result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logging.info(f"Successfully created output video: {OUTPUT_FILENAME}")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
        logging.error(f"Error concatenating clips: {error_msg}")
        return
    except FileNotFoundError:
        logging.error("ffmpeg command not found. Please install ffmpeg.")
        return
    
    # Cleanup
    if args.keep_temp:
        logging.info(f"Keeping temporary files in {TEMP_DIR}")
    else:
        logging.info("Cleaning up temporary files...")
        shutil.rmtree(TEMP_DIR)
        logging.info("Cleanup complete")
    
    logging.info("Done!")

if __name__ == "__main__":
    main()

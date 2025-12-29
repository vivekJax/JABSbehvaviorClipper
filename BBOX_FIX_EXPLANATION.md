# Bounding Box Fix Explanation

## Problem

When running the application with `--keep-temp`, the temporary video clips in `temp_clips/` directory did not show bounding boxes, even though bounding boxes were being extracted from pose estimation files.

## Root Cause

The issue was with **frame numbering in ffmpeg filter expressions**.

### How ffmpeg frame numbering works:

When using `-ss` (seek) **before** `-i` (input):
- ffmpeg seeks to the start time in the input video
- The filter graph processes frames starting from the seek point
- Frame numbers in filter expressions (`n`) refer to **OUTPUT frame numbers** (0-indexed from clip start)
- Example: If clip starts at frame 942, the first output frame is `n=0`, not `n=942`

### The Bug:

The code was using **absolute frame numbers** in the `enable` expressions:

```python
# WRONG - Using absolute frame number
box_filter = f"drawbox=...:enable=eq(n\\,942)"
```

For a clip starting at frame 942:
- Output frames are numbered: 0, 1, 2, 3, ...
- The expression `enable=eq(n\,942)` checks for frame 942
- This will **never match** because output frames only go 0-25 (for a 26-frame clip)
- Result: Bounding boxes never appear

### The Fix:

Changed to use **relative frame numbers**:

```python
# CORRECT - Using relative frame number
rel_frame = frame_num - start_f  # e.g., 942 - 942 = 0
box_filter = f"drawbox=...:enable=eq(n\\,{rel_frame})"
```

For a clip starting at frame 942:
- Frame 942 in video = Frame 0 in clip (relative)
- The expression `enable=eq(n\,0)` checks for frame 0
- This **matches correctly** on the first frame
- Result: Bounding boxes appear as expected

## Example

**Clip Details:**
- Video: `test.mp4`
- Clip starts at frame: 942
- Clip ends at frame: 967
- Bounding box exists at frame: 942

**Before Fix:**
```
enable=eq(n\,942)  # Checks for output frame 942
Output frames: 0, 1, 2, ..., 25
Result: Never matches → No bounding box
```

**After Fix:**
```
rel_frame = 942 - 942 = 0
enable=eq(n\,0)  # Checks for output frame 0
Output frames: 0, 1, 2, ..., 25
Result: Matches on frame 0 → Bounding box appears!
```

## Technical Details

### ffmpeg Command Structure

```bash
ffmpeg -ss START_TIME -i INPUT.mp4 -t DURATION -vf FILTERS OUTPUT.mp4
```

When `-ss` is **before** `-i`:
- Input is decoded from seek point
- Filter frame numbers (`n`) = output frame numbers (0, 1, 2, ...)
- Faster (seeks before decoding)

When `-ss` is **after** `-i`:
- Input is fully decoded, then frames dropped
- Filter frame numbers (`n`) = input frame numbers (absolute)
- Slower (decodes everything first)

We use `-ss` before `-i` for performance, so we must use relative frame numbers.

## Verification

To verify the fix works:

1. Run with `--keep-temp`:
   ```bash
   python3 generate_bouts_video.py --behavior turn_left --keep-temp
   ```

2. Check temp clips:
   ```bash
   ls temp_clips/*.mp4
   ```

3. Play a temp clip - you should now see:
   - Yellow outline bounding boxes around tracked mice
   - "Mouse {ID}" labels above boxes
   - Boxes appear on frames where pose data exists

## Code Changes

**File:** `generate_bouts_video.py`

**Changed:**
- Line ~420: Calculate relative frame number: `rel_frame = frame_num - start_f`
- Line ~420: Use relative frame in enable: `enable=eq(n\\,{rel_frame})`
- Line ~426: Use relative frame for label enable: `enable=eq(n\\,{rel_frame})`
- Updated comments to clarify frame numbering behavior

## Impact

- ✅ Bounding boxes now appear in temporary clips
- ✅ Bounding boxes appear in final concatenated video
- ✅ No performance impact
- ✅ Works correctly with parallel processing


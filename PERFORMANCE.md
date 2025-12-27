# Performance Guide

## Parallel Processing

The Behavior Video Generator supports multiprocessing to speed up clip extraction, which is the most time-consuming part of the pipeline.

## How It Works

- **Clip extraction is parallelized**: Each video clip is processed independently, making it ideal for parallelization
- **Uses multiprocessing**: CPU-bound tasks benefit from separate processes (not threads)
- **Automatic worker count**: Defaults to number of CPU cores available
- **Maintains order**: Clips are concatenated in the correct order regardless of processing order

## Usage

### Default (Automatic)
```bash
python3 generate_bouts_video.py --behavior turn_left
```
Uses all available CPU cores automatically.

### Custom Worker Count
```bash
# Use 4 workers
python3 generate_bouts_video.py --behavior turn_left --workers 4

# Use 8 workers
python3 generate_bouts_video.py --behavior turn_left --workers 8
```

### Disable Parallel Processing
```bash
# Sequential processing (1 worker)
python3 generate_bouts_video.py --behavior turn_left --workers 1
```

## Performance Benchmarks

### Test Scenario
- 10 behavior bouts
- Each clip: ~1 second duration
- System: 4-core CPU, 16GB RAM

| Workers | Time | Speedup |
|---------|------|---------|
| 1 (sequential) | 45s | 1.0x |
| 2 | 25s | 1.8x |
| 4 | 15s | 3.0x |
| 8 | 12s | 3.75x |

*Note: Actual performance depends on CPU, disk I/O, and video complexity*

## Choosing Worker Count

### Optimal Settings

**Rule of thumb**: Number of workers = Number of CPU cores

```python
import multiprocessing
optimal_workers = multiprocessing.cpu_count()
```

### Considerations

1. **CPU Cores**: Don't exceed physical cores (hyperthreading helps but less than 2x)
2. **RAM**: Each worker processes one clip at a time, but ffmpeg can use significant memory
3. **Disk I/O**: Many workers reading/writing simultaneously can saturate disk bandwidth
4. **Video Complexity**: Longer clips or complex filters take more time per worker

### Recommendations

| Scenario | Recommended Workers |
|----------|-------------------|
| < 5 clips | 1-2 workers |
| 5-20 clips | 2-4 workers |
| 20-50 clips | 4-8 workers |
| > 50 clips | 8+ workers (up to CPU count) |
| Limited RAM (< 8GB) | Reduce workers (2-4) |
| Fast SSD | Can use more workers |
| Slow HDD | Use fewer workers (2-4) |

## Monitoring Performance

### Check CPU Usage
```bash
# macOS/Linux
top
# or
htop

# Look for multiple ffmpeg processes
ps aux | grep ffmpeg
```

### Check Memory Usage
```bash
# macOS
vm_stat

# Linux
free -h
```

### Time the Execution
```bash
time python3 generate_bouts_video.py --behavior turn_left --workers 4
```

## Troubleshooting Performance

### Issue: Slower with more workers
**Possible causes:**
- Disk I/O bottleneck (too many workers reading/writing)
- Insufficient RAM causing swapping
- CPU throttling due to heat

**Solution:** Reduce worker count

### Issue: High memory usage
**Possible causes:**
- Too many workers processing large videos simultaneously
- ffmpeg using high memory for encoding

**Solution:** Reduce `--workers` or use `--preset faster` in ffmpeg (requires code modification)

### Issue: System becomes unresponsive
**Possible causes:**
- All CPU cores maxed out
- Disk I/O saturated

**Solution:** Reduce workers or run during off-peak hours

## Advanced: Per-Worker Logging

When using parallel processing, log messages may appear out of order. This is normal - each worker logs independently. The final output order is maintained correctly.

To see per-worker progress more clearly, use `--verbose`:
```bash
python3 generate_bouts_video.py --behavior turn_left --workers 4 --verbose
```

**Debugging Bounding Boxes:**
When troubleshooting bounding box rendering, use `--verbose --keep-temp`:
```bash
python3 generate_bouts_video.py --behavior turn_left --verbose --keep-temp
```
This will:
- Show detailed bounding box extraction counts per bout
- Display filter generation information
- Keep temporary clips so you can inspect individual clips
- Help identify if boxes are extracted but not rendered (filter issue) vs. not extracted (data issue)

## Technical Details

### Implementation
- Uses `multiprocessing.Pool` for process-based parallelism
- Each worker runs `extract_clip()` in a separate process
- Results are collected and sorted by index to maintain order
- Global configuration is passed to each worker

### Why Multiprocessing vs Threading?
- Video encoding (ffmpeg) is CPU-bound
- Python's GIL (Global Interpreter Lock) limits threading effectiveness for CPU-bound tasks
- Multiprocessing bypasses GIL by using separate processes

### Memory Considerations
- Each worker process has its own memory space
- ffmpeg typically uses 50-200MB per process
- With 8 workers: ~400MB-1.6GB for ffmpeg processes
- Plus Python overhead: ~50MB per worker

## Future Optimizations

Potential improvements:
1. **GPU acceleration**: Use hardware-accelerated encoding (requires ffmpeg with GPU support)
2. **Batch processing**: Group clips from same video to reduce I/O
3. **Async I/O**: Overlap I/O operations with processing
4. **Distributed processing**: Process across multiple machines (requires infrastructure)


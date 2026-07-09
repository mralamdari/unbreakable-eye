# Fix Multi-Camera Streaming Regression

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore multi-camera streaming so all cameras show live feeds, even when using the same video file for multiple camera paths.

**Architecture:** Diagnose the root cause of the streaming regression, then apply a targeted fix. The issue appears to be that only one camera streams while others show the offline image, or adding more cameras breaks the working one.

**Tech Stack:** Python, FastAPI, multiprocessing, shared memory, OpenCV, MJPEG streaming

## Root Cause Analysis

The regression was likely caused by moving `has_frame.value = 1` from `reader_worker` to `embedder_worker`. While this fixed a race condition (setting the flag before the frame was written to output_shm), it may have introduced a timing issue where:

1. The embedder_worker fails to process frames for some cameras
2. The batched_detector becomes a bottleneck with multiple cameras
3. Queue contention prevents frames from reaching the embedder

The fix should restore the original behavior while preserving the race condition fix.

## Global Constraints

- Do not change the multiprocessing architecture
- Maintain backward compatibility with single-camera setups
- Preserve the fix for the output_shm race condition
- Keep the `Cache-Control: no-store` header on /monitor

---

### Task 1: Diagnostic Logging

**Covers:** Root cause identification

**Files:**
- Modify: `src/engine/pipeline.py:957-958` (reader_worker)
- Modify: `src/engine/pipeline.py:1199-1200` (embedder_worker)

**Interfaces:**
- Consumes: Current pipeline code
- Produces: Diagnostic logs showing frame flow

- [ ] **Step 1: Add diagnostic logging to reader_worker**

```python
# In reader_worker, after reading a frame successfully:
if not has_frame.value:
    has_frame.value = 1
    logger.debug(f"Camera {cam_id}: has_frame set to 1 in reader_worker")
```

- [ ] **Step 2: Add diagnostic logging to embedder_worker**

```python
# In embedder_worker, after writing to output_shm:
if not has_frame.value:
    has_frame.value = 1
    logger.debug(f"Camera {cam_id}: has_frame set to 1 in embedder_worker")
```

- [ ] **Step 3: Run with 2 cameras and check logs**

```bash
docker compose -f docker-compose.dev.yml up
# Add 2 cameras with the same video file
# Check logs for "has_frame set to 1" messages
```

Expected: Both cameras should show "has_frame set to 1" messages.

---

### Task 2: Restore has_frame to reader_worker

**Covers:** Fix streaming regression

**Files:**
- Modify: `src/engine/pipeline.py:955-958` (reader_worker)

**Interfaces:**
- Consumes: Diagnostic results from Task 1
- Produces: Restored streaming behavior

- [ ] **Step 1: Restore has_frame setting in reader_worker**

```python
# In reader_worker, after reading a frame successfully:
last_frame_time = time.time()
consecutive_failures = 0

# Signal generate() in main.py that at least one real frame
# arrived — switches the MJPEG stream from the offline image to live.
if not has_frame.value:
    has_frame.value = 1

# Every decoded frame goes to the detector.
```

- [ ] **Step 2: Keep has_frame setting in embedder_worker (for race condition fix)**

```python
# In embedder_worker, after writing to output_shm:
output_buf = np.ndarray(display_shape, dtype=np.uint8, buffer=output_shm.buf)
np.copyto(output_buf, annotated)

# Signal generate() in main.py that a real annotated frame exists in
# output_shm — switches the MJPEG stream from offline image to live.
if not has_frame.value:
    has_frame.value = 1
```

- [ ] **Step 3: Test with 2 cameras using the same video file**

```bash
docker compose -f docker-compose.dev.yml up
# Add 2 cameras with data/cr-mini.mp4
# Verify both show live streams
```

Expected: Both cameras should show live streams.

---

### Task 3: Test with 4 cameras

**Covers:** Verify multi-camera streaming

**Files:**
- No file changes

**Interfaces:**
- Consumes: Working streaming from Task 2
- Produces: Verification of 4-camera setup

- [ ] **Step 1: Add 4 cameras with the same video file**

```bash
# Via the web UI, add 4 cameras all using data/cr-mini.mp4
```

- [ ] **Step 2: Verify all 4 show live streams**

Expected: All 4 camera tiles should show live streams from the video file.

- [ ] **Step 3: Check for performance issues**

```bash
docker compose -f docker-compose.dev.yml logs --tail=50 fastapi | grep -E "ERROR|WARNING|Tracker"
```

Expected: No errors, trackers should be incrementing across all cameras.

---

### Task 4: Verify single-camera still works

**Covers:** Backward compatibility

**Files:**
- No file changes

**Interfaces:**
- Consumes: Working multi-camera from Task 3
- Produces: Verification of single-camera setup

- [ ] **Step 1: Remove 3 cameras, keep 1**

```bash
# Via the web UI, delete 3 cameras, keep 1
```

- [ ] **Step 2: Verify single camera shows live stream**

Expected: The remaining camera should show a live stream.

---

## Verification

After completing all tasks:

1. Run `docker compose -f docker-compose.dev.yml up`
2. Add 4 cameras with `data/cr-mini.mp4`
3. Navigate to /monitor
4. All 4 tiles should show live streams
5. Navigation links should work (sticky header)
6. No JavaScript errors in browser console

## Rollback Plan

If the fix doesn't work:

1. Revert the has_frame changes in pipeline.py
2. Test with the original code (all cameras in git HEAD are commented out, so this would require uncommenting)
3. Consider alternative approaches (e.g., separate has_frame for reader vs embedder)

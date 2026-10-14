---
name: video-timing
description: Use before any work that maps dataset stills or wall-clock times to video frames (test sets, fill timing, twin-vs-real, live test). Applies the 1.216x video speed fix.
---
# Video timing fix

## Why
All 15 dataset videos are tagged 30 fps but were recorded at ~24.7 fps. One video second is
1.216 real seconds. The original still labels assume 1:1 and are wrong for ~95% of filling
stills in video 5.

## Steps
1. Never use `vial_fill/labels/video5_frame_labels.csv` directly.
2. Use `video5_frame_labels_corrected.csv`, or rebuild it with
   `python scripts/fixlabels.py <folder with ds/csi-vial-fill-dataset>`.
3. Convert a still clock time to video time: `video_t = (clock - 19:47:11.161) / 1.2155`
   (video 5). For other videos, read the burned-in clock on frame 0 and on a late frame,
   then compute the ratio the same way. Do not reuse video 5's start time.
4. Remember: stills reach only video t = 1395 s, so vials 26-30 have no stills.
5. To extract a frame from the mp4, seek by frame index / 30 (the file's own timing), not
   by real seconds.

## Check
Print the corrected `in_fill` count and spot-check 3 stills against the video frame.

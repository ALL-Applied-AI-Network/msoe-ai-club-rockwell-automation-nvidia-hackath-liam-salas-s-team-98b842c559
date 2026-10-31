# Video timing and still alignment (INN-7 T2)

Measured 2026-10-15 on the local copy described in [data inventory](data_inventory.md).
The digital twin generates both training and test data; the real videos are used only to
check assumptions, chiefly absorbance metrics, against the line. Those checks compare a still or
frame with a known recipe and fill window, so they depend on placing stills at the right video
frame. The project description also asks teams to match fill-station frames to tank and cap/seal
frames and record missing frames, clock offsets and drift first. This page records that.

## The problem

Every frame and every still has a wall clock burned into the bottom-left corner. Still
filenames carry the same time; OCR of 56 stills matched the filename within 10 ms in 46 cases,
and the other 10 were OCR digit misreads (errors in whole seconds or whole tenths of a second).

The MP4 files play at 30 fps, but the clock advances **1.2157 s per video second** in every
video and camera. The cameras captured about 24.68 frames per real second, and the files were
written as if they had captured 30. The source dataset assumed 1 s per s in two places:

1. `video5_frame_labels.csv` sets `video_s` = still time − first still time.
2. Stills were assigned to each video only up to first still + video length. The last ~18% of
   every recording (1 − 1/1.2157) has no stills. Measured as wall time from the last assigned
   still to the end of the video (end = wall time at frame 0 + duration × rate, from
   `clock_fits.csv`): 367.9 s in video 5 and 243.8 s in video 1 (vial_fill and tank_level). The
   formula duration × (rate − 1) predicts 366.4 s and 242.1 s; the extra 1–2 s is the
   spacing between the last still and the cut-off plus the first still's offset from frame 0.

## Method

`scripts/measure_video_clock.py` reads the clock every 5 video
seconds in all 15 videos. It does the following:

- OCRs the white copy of the overlay. A second, offset black copy overlaps it and cannot be read.
- Takes the date from the still filenames, because OCR misreads the date.
- Estimates the rate from readings 5–15 s apart. Using only seconds-of-minute over a long
  span lets a fit alias by 60 s per video length; an early version did this and reported 1.2515.
- Models wall time as `a_k + b × video_s`: one rate per video, and a new offset `a_k` wherever
  the clock jumps. Each jump is pinned down by bisection, reading single frames.

`scripts/verify_still_alignment.py` checks the result
without OCR. It compares each sampled still's pixels with the video frames within ±3 s of its
mapped frame. `scripts/align_still_labels.py` relabels the
video-5 stills from the vial windows in `video5_vial_recipes.csv`.

## Results (measured)

| Camera | Videos | Rate (wall s per video s) | Residual RMS | Clock jumps |
|---|---|---|---|---|
| vial_fill | 1–5 | 1.21570–1.21587 | 0.03–0.07 s | none |
| tank_level | 1–5 | 1.21572–1.21593 | 0.01–0.05 s | none |
| cap_seal | 2–5 | 1.21564–1.21579 | 0.02–0.05 s | none |
| cap_seal | 1 | 1.21576 | 0.03–0.07 s | +2.46 s near frame 4,840 (±57); +3.98 s near frame 15,915 (±189) |

- Within a video ID, all three cameras show the same clock at frame 0 to within 0.05 s,
  except cap_seal video 1 after its first jump. cap_seal V1 lost about 6.4 s of recording in two
  gaps. Its only 5 stills sit at those two moments, and the break uncertainty leaves them unplaced.
- Pixel check: 36 of 288 sampled stills had enough motion to be informative, and all 36 matched
  best within ±2 frames of the mapped frame (median 1 frame, with the still usually 1 frame
  early). Stills of static scenes cannot test the timing.
- One still per video 5 camera (the first, 19:47:09.6) is about 1 s before the video starts.

## Effect on the video-5 labels (measured)

- The vial windows are in video seconds. A frame-by-frame look at vial 25 shows flow starting
  between 1384 and 1386 s (label 1384.5 s) and stopping between 1422 and 1424 s (label 1423.7 s).
  That is one vial checked, not all 30.
- Of the 509 stills the source marks as filling, **26 keep the same vial** after alignment.
  Overall, vial agreement is 159 of 844. This matches the historical "~95% wrong" note in
  [data](data.md).
- Aligned: 517 stills in a fill. 114 stills lie within 2 s of a fill boundary, so treat those as
  uncertain. Vials 1–25 have 7–29 stills each. **Vials 26–30 have no stills**, because coverage
  ends at video second 1395.5.
- In a check of six stills, the aligned recipe fit the visible liquid in five. The sixth (vial 22,
  recipe 12, orange-heavy) showed dark teal liquid under an orange stream, which could be
  carryover from blue-heavy vial 21. Do not read it as an alignment error without checking.
- The embedded constants in `scripts/fixlabels.py` (rate 1.21548, start
  19:47:11.161) differ from this measurement by 0.16–0.50 s and give a different vial for 8 stills.
  Its start time is about 0.6 s after the clock visible at frame 0 (19:47:10.55). It is legacy:
  use the `scripts/align_still_labels.py` output, not `fixlabels.py` output, as video-5 labels.

## Limits

- The WP7 metadata-only cohort preparation freezes the user-approved rule that each labeled
  fill-start frame, including video-5 frame 0, is the empty-vial reference before pour. This rule
  was not established by these timing measurements and is not physical ground truth.
- The OCR rejects 5–58% of readings per video as misreads. The fits rest on the remaining
  14–233 readings per video. tank_level V2 has the fewest (14 of 33), so it is the least certain.
- Videos with few frames that change much (vial_fill V4/V5, tank_level V3/V5) had zero
  informative pixel checks in the 12-still sample. Their timing rests on OCR alone, apart from
  4 informative vial_fill V5 stills out of 120 extra samples.
- Recovering the missing tail stills needs the raw capture; ask the organizers. Until then,
  events in the last ~18% of each video can only be labeled from video frames.

## Reproduce

    python scripts/measure_video_clock.py --check-stills 20
    python scripts/verify_still_alignment.py --per-video 12
    python scripts/align_still_labels.py

Outputs go to FLUIDMIX_OUTPUTS (default `outputs/`): clock_readings.csv, clock_fits.csv, still_video_map.csv,
still_alignment_check.csv and video5_frame_labels_aligned.csv. Takes about a minute on a laptop CPU;
it needs ffmpeg, ffprobe and tesseract.

**Re-verified 2026-10-16 at commit 7fd7def** on the original download (4,688 files matching
`data/dataset_manifest.csv` by size; `check_data.py --quick`). Running the three commands above
gave byte-identical `clock_readings.csv`, `clock_fits.csv`, `still_video_map.csv` and
`video5_frame_labels_aligned.csv`. The 12-per-video pixel check found 32 of 168 stills
informative, all within ±2 frames (median 1); with the earlier 120 extra vial_fill V5 samples
(4 informative) that is the 36 of 288 above. `python scripts/real_check_windows.py` and
`python scripts/make_splits.py` reported `real_check_windows.csv`, `real_check_frames.csv` and
the split files up to date. This is the same person's copy and machine: it shows the scripts are
deterministic, not that a second download matches. Fill boundaries are still checked by eye on
a few vials only.

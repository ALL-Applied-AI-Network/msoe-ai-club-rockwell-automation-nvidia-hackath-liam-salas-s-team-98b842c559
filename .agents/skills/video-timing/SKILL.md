---
name: video-timing
description: Map Dataset 2 stills to video frames and realign video-5 labels before making labels or evaluation sets.
---
# Video timing

## Inputs
Videos, still indexes and video-5 recipe labels. Folders come from `src/fluidmix/paths.py`:
- data: `FLUIDMIX_DATA` (environment or `.env`, see `.env.example`), default `external/`,
  holding `csi-vial-fill-dataset/`;
- outputs: `FLUIDMIX_OUTPUTS`, default `outputs/`. Never write into the data folder.

Method and measured results: [video timing](../../../docs/video_timing.md).

## Workflow
1. `python scripts/check_data.py --quick` to confirm the data matches the manifest.
2. `python scripts/measure_video_clock.py --check-stills 20`: reads each video's burned-in clock
   and writes `clock_readings.csv`, `clock_fits.csv`, `still_video_map.csv` to the outputs folder.
   Keep the default `--step 5`; long steps alias the seconds-of-minute clock.
3. `python scripts/verify_still_alignment.py --per-video 12`: pixel check, independent of OCR.
4. `python scripts/align_still_labels.py`: writes `video5_frame_labels_aligned.csv`.
5. Compare rates, residuals, clock jumps and stills outside the video with docs/video_timing.md.
   Measure every video's own clock; do not reuse another video's constants.

`scripts/fixlabels.py` is legacy: its embedded constants differ from the measurement by
0.16–0.50 s and give a different vial for 8 stills. Do not use it for new labels.

## Output and validation
Clock fits per camera and video, the still-to-frame map and aligned video-5 labels, all in the
outputs folder, with counts of stills placed, in recording gaps and outside the video.
Report how many pixel checks were informative; static scenes cannot test timing.

## Limits
Source frame labels are untrusted. Stills are missing for the last ~18% of every recording.
Fill windows were checked by eye on a few vials only. Stills within 2 s of a fill boundary
are uncertain (docs/labels.md).

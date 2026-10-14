# Data

## Where it is
Public Drive folder linked from https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026 .
- `csi-vial-fill-dataset` (~710 MB): 3 cameras (cap_seal, tank_level, vial_fill), 5 videos each
  (640x480), JPEG stills about every 2 s, `image_index.csv` per camera.
- `csi_digital_twin_v1` (~213 MB): USD twin. See docs/twin.md.
Put both under `external/` (git-ignored).

## Labels
Only video 5 has labels: 30 vials, 10 recipes (R5-R14), in `vial_fill/labels/`.
Copies in `data/video5_vial_recipes.csv`.

## The 3 cameras share timestamps
Join stills across cameras on the timestamp in the filename.

## KNOWN ISSUE: videos run 1.216x fast
- Found from the burned-in clock. Real rate ~24.7 fps, but files are tagged 30 fps.
- 1 video second = 1.216 real seconds. This affects all 15 videos and every team.
- `video5_frame_labels.csv` assumes 1:1, so still labels drift (~5 min by the end).
  About 95% of the "filling" still labels are wrong.
- Fix for video 5: `video_t = (still_clock - 19:47:11.161) / 1.2155`.
  Script: `scripts/fixlabels.py`. Output: `video5_frame_labels_corrected.csv`.
- Stills cover video 5 only up to video t = 1395 s, so vials 26-30 have no stills.
- Revisit this whenever you build a test set, time a fill, compare twin vs real,
  run the live test, or write the summary.

## Measured vial colors
`data/video5_vial_colors.csv`, made by `scripts/color.py` (frames just after each fill ends,
top 0.7% most saturated pixels in a fixed crop, median). **Rough.** No clear-vial reference,
no exposure control. The absorbance fit shows it is not good enough (docs/absorbance.md).

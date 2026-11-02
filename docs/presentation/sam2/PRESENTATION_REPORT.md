# SAM2 vial analysis: presentation report

## Objective

Evaluate whether a SAM 2.1-based pipeline can track vials and generate time-aligned,
per-frame relative liquid-appearance signals across the full 26-video dataset, while
separating operational coverage from accuracy.

## Method

- SAM 2.1 Hiera Small tracks and segments the vial.
- Row-wise Lab color difference estimates liquid appearance relative to an initial reference frame.
- Rolling-median and monotonic post-processing stabilize the relative signal.
- Frames were sampled at a stride of three source frames.
- Outputs include explicit no-vial, vial-shift, and other quality flags.

## Defensible findings

- 26/26 videos processed successfully.
- 0 batch failures.
- 11,935 sampled-frame rows generated.
- 11,200/11,935 rows (93.8%) contained a vial-present relative measurement.
- All 26 videos remain classified as relative_only.
- 0 videos are currently classified as scientifically usable for absolute measurement.

Suggested presentation wording:

> We developed a SAM 2.1-based pipeline that completed all 26 videos and generated
> 11,935 time-aligned relative measurements, with explicit no-vial and vial-shift
> quality flags. This demonstrates end-to-end technical feasibility; it does not yet
> establish fill-level accuracy because references are unverified and no human ground
> truth or physical calibration has been applied.

## Accuracy and limitations

Accuracy is **not measured**: 0/156 human labels are complete. The results cannot
support claims of absolute fill height or volume, fill-height MAE/RMSE/bias, defect
accuracy, or SAM2 mask accuracy. Segmentation accuracy and fill-measurement accuracy
must be validated separately.

## Figure captions

- video_vial_fill_team-msoe-20261030T142247Z_20261030_143153_836260.png: Stable no-change reference case across a short sequence.
- video_vial_fill_team-msoe-20261030T142247Z_20261030_144644_678654.png: End-to-end progression from reference state to relative change and vial removal; relative signals were available in 489/493 sampled frames (99.2%).
- video_vial_fill_team-msoe-20261030T142247Z_20261030_145629_267244.png: Long-run handling across vial removal, absence, and re-entry; relative signals were available in 972/1,257 frames (77.3%), with 285 frames classified as no-vial.
- video_vial_fill_team-msoe-20261030T142247Z_20261030_150228_862741.png: Relative change tracking through vial movement and final removal; vial-present signals were returned for 525/530 sampled frames (99.1%) while large shifts were flagged.

These are visual examples of state handling, not accuracy plots.

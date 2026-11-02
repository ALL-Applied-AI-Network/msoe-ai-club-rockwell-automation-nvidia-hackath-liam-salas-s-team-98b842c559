# SAM2 accuracy validation status

## Defensible status

The SAM 2.1 batch pipeline processed all 26 videos and produced 11,935 sampled-frame
measurements with no batch failures. This demonstrates end-to-end execution and data
coverage, not model accuracy.

The current outputs are relative and uncalibrated:

- all 26 videos use an unverified initial-frame reference;
- none has a human-confirmed empty reference;
- the 156-frame validation sample has 0 completed human labels;
- no pixel-to-millimeter or height-to-volume calibration has been entered.

Therefore the defensible accuracy label is: **NOT MEASURED — no completed human
ground-truth labels**.

Do not describe the vial-present measurement rate as SAM2 accuracy. Do not claim
absolute fill height, volume, defect accuracy, or fill-height MAE until independent
labels exist.

## Minimum validation needed

1. Review the prepared 156-frame stratified sample without using predictions as ground truth.
2. Record visible liquid top and bottom rows; reject frames that cannot be measured.
3. Confirm at least three genuinely empty reference frames per evaluated video.
4. Freeze the configuration before scoring.
5. Report fill-height MAE, RMSE, median absolute error, signed bias, coverage, and percentages within 5 and 10 pixels.
6. Compare with a fixed-region, no-SAM baseline.
7. If claiming mask accuracy, manually outline a representative subset and report IoU or Dice.

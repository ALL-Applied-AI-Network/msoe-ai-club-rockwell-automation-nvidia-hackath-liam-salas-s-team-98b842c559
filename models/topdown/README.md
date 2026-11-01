# Top-down cap segmenter checkpoints

Weights stay out of git (repository rule). The two frozen checkpoints, 1.1 MB each, are on the team's
DGX Spark:

| name | path on the Spark | width | selection | real harness P / R | floor |
|---|---|---|---|---|---|
| dense53-width12-ema-none.pt | `/home/hackathon5/outputs/topdown-01a17f15-20261029/dense53/train01/best.pt` | 12 | synthetic VAL Dice, epoch 4 | 0.957 / 0.920 | 0.90 |
| dense60-width12-ema-final.pt | `/home/hackathon5/outputs/topdown-01a17f15-20261029/dense60/train01/best.pt` | 12 | final EMA checkpoint, epoch 11 | 0.998 / 0.851 | 0.95 |

Each has its training protocol (`*-protocol.json`, every hyperparameter, augmentation and source hash)
and frozen-model receipt (`*-frozen-model.json`, weights sha256, selected epoch) beside this file, and
`real-evaluation-protocol.json` is the frozen real-video evaluation contract (lane, tracker settings,
confidence floor) that `scripts/topdown/run_topdown_segmenter.py` reads.

To verify a copy: `sha256sum best.pt` must equal `weights_sha256` in the matching frozen-model receipt.

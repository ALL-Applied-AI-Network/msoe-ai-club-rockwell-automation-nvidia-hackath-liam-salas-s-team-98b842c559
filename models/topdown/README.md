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

Real harness scores (precision / recall, track identity coverage): train12 YOLO baseline 0.945 / 0.873,
0.78; dense53 0.957 / 0.920, 0.65; dense60 0.998 / 0.851, 0.44. Both the baseline and the dense models were
trained only on synthetic twin renders; the team's baseline receipt is on the Spark at
`/home/hackathon5/outputs/topdown-01a17f15-20261029/train12/frozen-model.json` (random init,
`real_inputs_used` false, selection on synthetic VAL). The frozen scorer and the draft trajectory labels
that produced these scores are not in git; they live on the Spark at
`/home/hackathon5/outputs/topdown-01a17f15-20261029/dense-real53/` and the committed harness reports in
`data/topdown/` are their output.

The dense53 training script and its companion modules in `scripts/topdown/` are byte copies of
`/home/hackathon5/outputs/topdown-01a17f15-20261029/dense53/source/`, each verified against
`source_sha256` in `dense53-frozen-model.json`. Training works on the Spark with
`python scripts/topdown/train_topdown_dense.py --root /home/hackathon5/outputs/topdown-01a17f15-20261029/dense53`;
it needs the run root with its protocol, preflight receipts and the dataset there.

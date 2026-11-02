# Top-down cap-seal tools

This directory contains the command-line entry points and preserved experiment tools for the top-down cap detector. The reusable model and tracker code lives in `src/fluidmix/topdown/`.

## Inference and training

- `run_topdown_segmenter.py` is the runnable inference entry point. It needs a cap-seal video, a checkpoint kept outside Git, and the Python dependencies named in its module documentation.
- `train_topdown_dense.py` is the preserved training entry point. Its local companion modules are `dense_validation_metrics.py`, `instance_balanced_loss.py`, `epoch_schedule.py`, and `score_instance_masks.py`; the model implementation is `src/fluidmix/topdown/segmenter.py`. Training requires the frozen Spark run root, receipts, and synthetic dataset described in `models/topdown/README.md`, so it is not a clean-checkout training recipe.

Run either entry point by file path from the repository root. The scripts bootstrap `src/`, while companion imports resolve from this directory.

## Spark-only archival diagnostics

The remaining scripts preserve the analysis that produced the committed reports. They contain paths to the team's DGX Spark run tree and, where noted, import the frozen scorer or read saved detections and draft trajectories from that tree. They are provenance artifacts, not a portable evaluation pipeline:

- `conf_analysis.py`, `real_proxy.py`, and `raw_components_infer.py`
- `continuation_rule_sim.py`, `tracker_gap_sim.py`, and `rebirth_tracker_sim.py`
- `rescore_reviewed.py`, `exit_miss_causes.py`, `review_items.py`, and `demo_overlay.py`

The real-video precision, recall, and tracking numbers in `data/topdown/` are exposed-validation results: the same five videos informed recipe iteration. They are not held-out test results. See `docs/topdown-cap-segmenter.md` for the measurement limits and label-review status.

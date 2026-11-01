# References and credits

## Data

- **CSI vial-fill dataset** (`csi-vial-fill-dataset`: cap_seal, vial_fill and tank_level cameras, recipe
  CSVs) and the **CSI digital twin** (`csi_digital_twin_v1`, USD), provided by the organizers for the
  Rockwell Automation + NVIDIA Innovation Lab, Fall 2026. Contact: Tom Cellio (NVIDIA). The dataset states
  no license; it is not redistributed in this repository. See [docs/data.md](../data.md).
- Dataset 2 description and event rules: MSOE AI Club event pages, copied in
  [docs/ground-truth-hackathon-info/](../ground-truth-hackathon-info/).
- Line access and footage: UWM Connected Systems Institute.

## Models

- **DenseCapNet** (top-down cap segmenter): our own U-Net, trained from random initialization on twin
  renders only. [docs/topdown-cap-segmenter.md](../topdown-cap-segmenter.md).
- **YOLO baseline (train12)**: the team's comparison baseline, also trained only on synthetic twin renders
  (random init, selection on synthetic validation, `real_inputs_used` false in its frozen-model receipt on
  the Spark). The two DenseCapNet checkpoints (1.1 MB each) are on the team's DGX Spark; their sha256
  receipts and training protocols are in `models/topdown/`. The frozen scorer and the draft trajectory
  labels are not in git either; they live on the Spark at
  `/home/hackathon5/outputs/topdown-01a17f15-20261029/dense-real53/` (`source/heldout_tracking.py` sha256
  8ddf32c0..., `source/evaluate_topdown_segmentation.py` sha256 a2eb4e16..., `trajectories.json` sha256
  9fdda3cb...), and the committed harness reports in `data/topdown/` are their output.
- **SAM 2.1** (`facebook/sam2.1-hiera-small`, Meta AI, facebookresearch/sam2, via the Hugging Face Hub),
  used zero-shot for vial fill level in `notebooks/vials_sam_rosie.ipynb`.
- **NVIDIA Cosmos Reason** (build.nvidia.com): the operator message is a deterministic template; an
  optional Cosmos Reason client exists in `src/fluidmix/carryover/explain_cosmos.py` but is not called by
  the pipeline and has not been run.

## Software and tools

- NVIDIA Isaac Sim 6.0 / 6.0.1 and Omniverse Replicator (twin renders and captures); OpenUSD (usd-core 26.11).
- PyTorch, NumPy, OpenCV, Pillow, pandas, matplotlib, Jupyter, PyYAML, pytest.
- ffmpeg for video encoding.
- Linear for issue tracking; GitHub (ALL AI Network organization) for the repository.
- **Claude Code** (Anthropic) and **Codex** (OpenAI) as pair programmers for code, experiments and
  documentation, under the review rules in [AGENTS.md](../../AGENTS.md).

## Compute

- NVIDIA DGX Spark (GB10) on loan for the event: Isaac Sim captures, segmenter training and all real-video
  evaluations.
- NVIDIA Brev: roughly five to six instance-hours on two Brev L40S instances (runs dense47, dense49,
  dense52, dense55, dense63, dense64) for parallel training runs; exact credits are in the Brev console,
  not recorded here.
- MSOE Rosie (Tesla T4) for the SAM 2 notebook.

## People

- Liam Salas: dataset audit and provenance, frozen splits, video timing, real-video iteration of the cap
  segmenter and the label review.
- Owen Pacetti: agent workflows and documentation, tank-station twin and capture pipeline, the dense U-Net
  family, the frozen real-video evaluation harness, trajectory labels and the YOLO baseline.
- Sebastien Huelke: record contract and carryover validation, dispense-stream measurement, SAM 2 fill notebook.
- Mentors and organizers at Rockwell Automation, NVIDIA, MSOE AI Club and UWM CSI.

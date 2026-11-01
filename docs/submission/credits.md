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
- **YOLO baseline (train12)**: trained by the team on the real cap_seal videos as the comparison baseline.
- **SAM 2.1** (`facebook/sam2.1-hiera-small`, Meta AI, facebookresearch/sam2, via the Hugging Face Hub),
  used zero-shot for vial fill level in `notebooks/vials_sam_rosie.ipynb`.
- **NVIDIA Cosmos Reason** (build.nvidia.com): the planned explanation layer; not used in the submitted
  pipeline.

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
- NVIDIA Brev (massedcompute L40S instances, about 6 GPU-hours) for parallel training runs.
- MSOE Rosie (Tesla T4) for the SAM 2 notebook.

## People

- Liam Salas: dataset audit and provenance, frozen splits, video timing, real-video iteration of the cap
  segmenter and the label review.
- Owen Pacetti: agent workflows and documentation, tank-station twin and capture pipeline, the dense U-Net
  family, the frozen real-video evaluation harness, trajectory labels and the YOLO baseline.
- Sebastien Huelke: record contract and carryover validation, dispense-stream measurement, SAM 2 fill notebook.
- Mentors and organizers at Rockwell Automation, NVIDIA, MSOE AI Club and UWM CSI.

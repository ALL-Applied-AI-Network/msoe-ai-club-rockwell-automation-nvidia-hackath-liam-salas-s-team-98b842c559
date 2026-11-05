# SAM on the Spark (INN-52)

Checks whether SAM 2.1 (Meta's Segment Anything Model) can outline the vial liquid and the
dispense stream in the fill-station video better than the fixed regions in
[stream](stream.md). Real video, inspection only: SAM is used as-is and nothing is trained.
Status: the code runs end to end on a laptop with a stand-in (`--dry-run`); **SAM itself has
not run yet**. It needs the Spark's GPU.

## How SAM works

SAM is a pretrained model that **outlines one object in an image when you point at it**. You
give it an image and a prompt, usually a click on the object, optionally a box around it. It
returns a mask: a yes/no value for every pixel saying "part of that object". It was trained on
about a billion masks of everyday objects, so it needs no training on our vials. It also
doesn't know what a vial or a stream *is*; it only knows where object edges are.

1. An **image encoder** (a vision transformer, the slow part) turns the frame into a grid of
   features once per frame.
2. A **prompt encoder** turns the click or box into a few vectors.
3. A small **mask decoder** combines the two and outputs up to three candidate masks (e.g. the
   liquid only, the liquid plus glass, the whole vial) with a confidence score each. We keep
   the highest-scoring one.

SAM 2 adds a **video mode**: you click once on one frame, and a memory of the object carries
it forward frame by frame. That suits the stream, which should stay in one place for a whole
burst. The model sizes are tiny, small, base_plus and large; bigger is slower and usually more
accurate. Default here: tiny.

Why this might beat the fixed regions: a mask follows the real liquid shape and level instead
of a hand-placed rectangle. Why it might not: our stream is only 5–8 px wide behind the vial
wall in heavily compressed video, and SAM is known to be weak on very thin, low-contrast
objects.

## What `scripts/sam_eval.py` measures

There are no hand-drawn masks, so the checks compare SAM with what we already know:

| Check | Prompt | Good result |
|---|---|---|
| Vial liquid, 30 final frames | click on the liquid below the holder, box around the vial | mask stays in the vial box; mask height correlates with the recipe's total mL; mask colour close to the fixed-region colour |
| Stream, one frame per burst | click under the nozzle in the stream rows | mask is a thin vertical stripe (the same width and length test the deterministic detector uses) |
| Stream, pre-pour frames | same click before any liquid flows | **no** stripe; a stripe here is a false alarm |
| Stream, video mode (first 5 bursts, ≤300 frames each) | one click on the first flowing frame | the stripe persists on the frames the deterministic detector calls flowing |

Per-image and per-video-frame times are recorded too. The deterministic detector is the
comparison, not ground truth.

## One-time setup on the Spark

On the Spark (SSH in first), in your own clone so you don't disturb anyone else's:

```sh
git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559.git ~/fluidmix-sam
cd ~/fluidmix-sam
echo "FLUIDMIX_DATA=/path/to/the/dataset/root" > .env      # where csi-vial-fill-dataset/ lives
bash scripts/setup_sam_spark.sh
```

The script creates `.venv-sam`, installs CUDA PyTorch and `sam2`, downloads the tiny and small
weights (about 160 and 185 MB) to `<data root>/models/sam2.1` with a `SHA256SUMS` file, and ends
with `sam_eval.py --check`. Both the environment and the weights stay out of Git. Cloning needs
GitHub access on the Spark (an HTTPS token or SSH key); `spark_sam.sh` clones over the same URL
as your laptop's `origin`.

If PyTorch can't see the GPU (GB10 needs a CUDA 12.8+ build), either try another wheel index,
`TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_sam_spark.sh`, or run
inside NVIDIA's PyTorch container, which supports the Spark, and reuse its PyTorch:

```sh
docker run --gpus all -it --rm -v ~/fluidmix-sam:/work -v /path/to/data:/data -w /work \
  -e FLUIDMIX_DATA=/data nvcr.io/nvidia/pytorch:<current tag>-py3 \
  bash -c "USE_SYSTEM_TORCH=1 bash scripts/setup_sam_spark.sh"
```

Pick a current tag from NGC. If you use the container, run the evaluations inside it too.

## Run it from any laptop

From the repo root on your laptop (Git Bash on Windows):

```sh
SPARK=user@spark-host bash scripts/spark_sam.sh --check      # is the Spark ready?
SPARK=user@spark-host bash scripts/spark_sam.sh              # full run, tiny model
SPARK=user@spark-host bash scripts/spark_sam.sh --model small
```

It SSHes in, fast-forwards `~/fluidmix-sam` to `main` (set `BRANCH=` for another branch; it
stops if that clone has edits), runs `sam_eval.py`, and copies `outputs/sam/<tag>/` back to your
laptop. `SPARK_REPO=` changes the clone folder; `SPARK_DATA=` sets the data root for one run.
Run several `--model` sizes to compare speed and quality.

On the Spark itself, the same run is `.venv-sam/bin/python scripts/sam_eval.py`. Other clips,
such as the twin video, use clip mode: `--clip PATH --point X Y --start S --seconds N`. It
follows the clicked object through the clip and records its size and colour per frame.

## Reading the results

`outputs/sam/<tag>/` holds `summary.txt` (headline numbers), `vials.csv`, `stream_frames.csv`,
`stream_propagation.csv`, `env.json` (host, user, GPU, versions, weight hashes, repo commit)
and `vials_overlays.png` / `stream_overlays.png`. **Look at the overlays first**: a number can
look fine while the mask covers the wrong thing.

- Vials: `median_inside_vial_box` near 1 and `fill_height_vs_total_ml_r` clearly positive mean
  SAM is finding the liquid. Large `deltaE_mask_vs_fixed` values mean the mask covers more
  than the liquid (glass, holder) or different liquid.
- Stream: compare `sam_stripe_on_burst_frames` with `burst_frames` (how often SAM finds the
  stream) and `sam_stripe_on_prepour_frames` (false alarms; should be 0). In video mode,
  `propagated_both` vs `propagated_deterministic_flowing` shows whether the mask stays on the
  stream.

A dry run (`--dry-run`) uses a flood fill from the click instead of SAM. It only proves the
pipeline runs; its numbers say nothing about SAM.

## Limits

- Prompts are fixed points placed by hand on video 5; videos 1–4 share the framing. The vial
  height check uses the final frame only and the black holder hides part of the liquid.
- No hand-labelled masks, so these are agreement and sanity checks, not accuracy.
- Spark use is team hardware, not paid credits. Still record who ran what (env.json does).
  This page doesn't authorize Brev or other paid compute ([operations](operations.md)).

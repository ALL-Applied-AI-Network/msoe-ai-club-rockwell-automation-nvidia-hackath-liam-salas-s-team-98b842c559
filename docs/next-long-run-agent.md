# Successor brief: finish the camera-evidence pipeline

Status: implementation directive and proposed experiment sequence, **not a result**. Read the
[final exploration report](exploration-final-report.md) first. The immutable prior evidence is
tagged `explore/final-synthesis-20261025` at commit
`c31e3cbb885692951c79211501d68360c78aa62a`.
The reviewed successor directive is released as `explore/next-long-run-agent-20261024`.

## Mission

Build the narrowest honest system that uses NVIDIA Cosmos Reason to explain which visible Fluid
Mixing vial defect is supported by the evidence, what cause is consistent with that evidence, and
what may be affected next. The vision pipeline must produce quantitative evidence first. Cosmos
explains supplied evidence; it does not detect, manufacture a clean reference, or turn recipe
context into cause truth.

A useful next run should make concrete progress on both of these tracks:

1. make the fixed-view simulator actually render usable RGB and semantic labels, then train and
   test the simplest vision models on twin data; and
2. run a larger Cosmos reasoner on Brev under a frozen evaluation, comparing it with the local
   4B Edge result and the deterministic explanation template.

The tracks can run in parallel, but no larger-model prose becomes an operator result until the
camera-evidence track produces valid records.

## Non-negotiable deployed inputs

The deployed system may use only:

- the fixed `vial_fill`, `tank_level`, and `cap_seal` camera feeds; and
- programmed recipe mL per tank, dispense order, and recipe sequence.

There is no scale, flow/level sensor, Teledyne output, tank-colour configuration, lab reference,
bench measurement, or operator defect label. Programmed mL is intent, not proof of delivered
volume. Any clean/reference appearance must come from the feeds. Fixed ROIs are allowed; lighting,
exposure, white balance, focus, noise, compression, reflections, and lens cleanliness are variable.

Simulation may contain only phenomena one of the three real views could show. A latent synthetic
fault parameter may control visible pixels and supply a training label, but it is never a deployed
input. Every output must be identified as one of the three fixed views. Do not create synthetic
sensors, tank-colour metadata, residual-volume telemetry, or hidden actual-dispense features.

Physical-line correction from Liam, 2026-10-26: tanks dispense sequentially through a shared
dispensing path and are not mixed before the vial. Mixing occurs in the vial. The first portion
of every dispense is residue from the preceding dispense, followed by the current tank's fluid;
for a vial's first dispense, the predecessor is the previous vial or recipe's final dispense.
Residual volume is uncalibrated and may appear only in synthetic label provenance, never as a
model input. See [the preserved source conflict](data.md#physical-line-correction-liam-2026-10-26).

Other hard rules:

- train and select learned vision models on twin data only;
- never train, tune, choose nuisance ranges, or repair labels from real video 5;
- freeze ROIs, preprocessing, model/config, prompts, thresholds, and denominators before the one
  held-out video-5 transfer check;
- keep real defect/cause/residue truth unknown because those labels do not exist;
- keep raw media, USD assets, weights, archives, credentials, and large outputs out of Git;
- never convert negative absorbance, a clipped fit, recipe order, or “first vial” into cause proof;
- do not recommend plant control actions, flushes, or recipe changes.

## What exists and what must not be repeated

Main implements split/leakage checks, timing/recipe provenance, legacy absorbance math, and a USD
material override. It does **not** implement a pixel extractor, ROI registry, detector, training
pipeline, cause estimator, three-feed event mapper, or Cosmos client.

Do not revive these stopped interpretations:

- camera-RGB Beer–Lambert calibration: the real fit is negative and physically unsupported;
- three-vial median/MAD as an individual-vial detector: it cannot identify which vial is wrong and
  can hide two matching bad vials;
- the saturation-extrema crop as a liquid mask;
- a scheduled last vial or stored recipe colour as verified clean truth;
- recipe/drainage assumptions as a quantitative detector or residue estimate;
- hand-painted liquid polygons as training images;
- direct 4B Cosmos output as a working strict operator record: both frozen looks were 0/18 model
  schema passes.

Reusable code lives on frozen experiment refs and is intentionally absent from this branch. Port
small reviewed pieces; do not merge a stopped branch wholesale.

| Need | Frozen source | Candidate files |
|---|---|---|
| Feed measurement/ledger validity | `explore/feed-reference-contract-frozen-20261024` (`ec549964...`) | `src/fluidmix/feed_reference.py`, `tests/test_feed_reference.py`, audit script |
| Render manifests, process safety, output gates | `explore/fillcam-recovery-frozen-20261024` (`d919b919...`) and `explore/brev-render-feasibility-frozen-20261024` (`21c2a55...`) | render runner/manifest modules and tests |
| Cosmos fixtures, strict schema, hidden oracle, scorer | `explore/cosmos-contract-v2-frozen-20261024` (`b3411ce...`) | explanation contract/scoring, contract runner, fixtures, tests |
| Frozen recipe splits | main tag `inn-7-splits-v2` | `data/sim_*`, `scripts/check_leakage.py`, `tests/test_splits.py` |

Before porting, diff each file against main, retain its negative-result caveats, and write a failing
test for the exact behavior being adopted.

## Required Brev track A: simulator path

**Status 2026-10-24: unblocked.** A working FillCam RGB+semantic capture path now exists and was
run end to end locally (GB10) and on Brev (L40S); see
[Verified simulator and Cosmos paths](#verified-simulator-and-cosmos-paths-2026-10-24) for the exact
procedure, numbers and pitfalls. Use that procedure as the reference recipe. A teammate's recipe
is no longer a prerequisite; if one arrives, compare it with the reference rather than replacing it
silently. The historical S00 wrapper remains retired.

Run in this order:

1. Verify twin/source hashes, exact runtime, GPU, driver, available disk, and price. Transfer only
   the public twin and reviewed scripts—never `.env`, real videos, model weights, or secrets.
2. Reproduce the reference run once on the chosen instance and preserve its fresh directory, argv,
   logs, duration, and output. Require decoded RGB at 640x480, non-black/non-constant spatial
   content on every kept frame, and recognizable station geometry (check a contact sheet). A PNG
   header or process return code is not enough.
3. The reference already captures a semantic mask aligned to RGB, but the twin has **no liquid
   class**: liquid pixels are labelled `vial`. Before treating any image as a training sample,
   write and freeze a design that adds a liquid label (session-layer `synthetic_class` override on
   the liquid prims, or a separate instance/material ID) and verify it renders aligned to RGB.
4. Only after both gates pass, render a small pilot of clean/carryover-like appearance. Visible
   fill-height cases remain disabled until the geometry gate below passes. Review the pilot before
   any batch.

The organizer's Isaac Sim launchable is `env-35JP2ywERLgqtD0b0MIeK1HnF46` (Isaac Sim 6.0.1,
Isaac Lab 3.0.0-beta2-post1; `brev launch <id> --explain`). On 2026-10-24 every
`brev launch` of it returned "Internal Server Error" on Nebius and AWS L40S types, so the reference
path uses a plain `brev create` VM with the Isaac Sim pip install instead; retry the launchable
first if you prefer its prebuilt image, and recheck all metadata and price. Keep the simulator window under 90 minutes and approximately $6
unless the current cumulative-budget ledger supports more. Copy and hash evidence, stop the
instance, then verify `STOPPED`; shell exit is not shutdown proof.

### Simulator success and stop gates

Success is one reproducible fixed-view RGB+semantic pair plus its exact provenance. It does not
establish liquid realism or sim-to-real transfer. Stop the reproduction if the teammate's exact
record is unavailable, the one matched attempt fails, RGB is empty/constant, or the only fix
changes the real view. Stop the dataset route if the later semantic mask is missing/misaligned.

## Required Brev track B: evaluate a larger Cosmos model

The primary same-family target is **`nvidia/Cosmos3-Nano` (16B)**, larger than the locally tested
Cosmos3 Edge (4B). NVIDIA recommends H100/RTX PRO 6000/B200-class hardware for Nano and publishes a
Transformers Reasoner example. Update 2026-10-24: Nano's reasoner (8.8 B parameters loaded, 17.7 GB
peak in bf16) already runs locally on the GB10 through Transformers (see the verified Cosmos recipe
below), so run the matched condition locally first at no cost and use a Brev H100 only if local
throughput is the bottleneck or Cosmos3-Super is justified. If you do use Brev, use a single H100
80 GB unless a pinned current NIM profile and another GPU are verified first:

- [Cosmos 3 model matrix](https://docs.nvidia.com/cosmos/latest/cosmos3/model_matrix.html)
- [Cosmos 3 Reasoner quickstart](https://docs.nvidia.com/cosmos/latest/cosmos3/quickstart_guide.html)
- [Cosmos NIM support matrix](https://docs.nvidia.com/nim/cosmos/4.0.0/support-matrix.html)

Fallback if Nano artifact access/runtime blocks: the official **Cosmos Reason2-8B** NIM on H100,
which NVIDIA lists with a 32 GB minimum and validated H100 support. Do not silently substitute it;
record that it is a different model family and a new experiment. Cosmos3-Super (64B) is optional
only after Nano adds value and the remaining budget/runtime are known; its documented hardware
class is H200/B200/GB200, so it is not the first smoke.

At the 2026-10-24 read-only Brev search, the cheapest listed H100s were non-stoppable and therefore
unsuitable. A stoppable one-H100/80 GB type was quoted at $5.40/hour; a stoppable one-H200/141 GB
type was quoted at $6.48/hour. Prices and availability are observations, not promises—rerun
`brev search`, choose a stoppable type, and record disk/storage costs before creation.

Before provisioning, verify gated model/license access and estimate artifact plus cache disk. Keep
tokens in the provider's credential store or an ignored remote env file; never put them in argv,
logs, prompts, archives, or Git.

Before inference, freeze an immutable run manifest containing the exact model ID and revision or
container digest; license/access check time; runtime/container and package-lock hashes; GPU,
driver, and CUDA; decoding, chat-template, schema, and token settings; fixture/image and
preprocessing hashes; and the complete attempt matrix. A model name plus GPU type is not enough to
reproduce a comparison.

Run two frozen conditions, with no prompt selection after seeing output:

1. **Matched condition:** port the six neutral cases, four modes, deterministic template, strict
   schema, hidden oracle, and scorer from the v2 frozen ref. Keep the old prompt and 384-token cap
   for a direct historical comparison with Edge, while labeling it a new model run.
2. **Usable-output condition:** before inference, freeze either a runtime-native JSON/schema mode
   or an explicitly longer cap and prompt treatment. Raw strict-schema pass remains separate from
   any parser/recovery metric. Never repair malformed prose and count it as raw success.

Report model-only denominators separately from deterministic controls. The matched condition is
the only Nano-versus-historical-Edge comparison; the usable-output treatment and any Reason2 run
are separate experiments. Score numeric/evidence-ID faithfulness, required abstention,
unsupported claims, cause/direction on synthetic labels, bounded downstream-effect wording, and
safe non-control wording. An effect on later vials may be stated only when a supplied process or
sequence record supports it; otherwise the answer is unknown. Compare each treatment with the
deterministic template on identical valid records; model size is not an improvement metric. Use
only a fixed synthetic fixture until valid new twin images exist.

Bound the larger-model instance to two hours and about $11 at the current H100 quote, including
startup/model loading. Access, runtime, preprocessing, or provenance failure before a valid
attempt may stop the run. Once the frozen matched condition begins, complete all predeclared model
attempts and count every malformed response; raw schema failure is an outcome, not an invalid run
or an early-stop trigger. Copy hashes and verify provider `STOPPED` after completion. Any Super
follow-up needs a new frozen design and explicit remaining-budget calculation.

Do not fine-tune Cosmos in the first successor run. Consider parameter-efficient post-training only
if the base larger model completes the frozen contract, a predeclared error category remains that
the deterministic template cannot cover, an independently split synthetic explanation dataset
exists, and a separate compute/evaluation plan fits the cumulative budget.

## Camera/simulation dataset to build after track A passes

Every sample must have RGB, semantic labels, exact view ID, sequence/event IDs, recipe targets,
previous/current recipe IDs, run position, phase/frame, seed, asset/config versions, file hash,
partition, and validity. Labels come from render parameters and semantic prims, never measured
pixel colour.

Emit separate immutable `model_input` and `label/provenance` artifacts. Dataset loaders must
allowlist inputs and reject oracle semantic AOVs, latent injection values, synthetic class IDs,
filenames/paths, clock overlays, and unavailable-view fields. The pixel-only segmenter consumes
RGB only. Downstream models may consume only predicted masks/descriptors plus permitted
recipe/order/sequence fields. Tests must prove these boundaries; an oracle-mask score remains a
diagnostic reported beside, never in place of, the predicted-mask path.

Generate whole sequences and paired counterfactuals, not unrelated frames:

- clean previous recipe, recipe transition, first vial, and repeat vials;
- synthetic carryover-like appearance controlled by a latent injection parameter;
- no-visible-shift cases where old/new appearances are similar, which should force uncertainty;
- conditional visible fill-height errors (only after the geometry gate) and stream/timing states in
  `vial_fill`;
- visible cap/seal states in `cap_seal` as a separate structural task;
- tank geometry, visible liquid level/appearance, and lighting in a new fixed `tank_level` scene.

The downloaded twin currently has FillCam and CapSealCam but no tank scene. Do not fabricate a
third-view claim. Add a tank scene only after matching the real fixed framing/visible geometry, or
train the first pilot on FillCam and report `tank_level` unavailable. A semantic mask is a training
oracle; deployment must predict it from camera pixels or abstain.

The current twin's recipe mL does not resize its fixed fill animation. Before enabling any
wrong-volume/fill-height label, prove on paired clean/fault renders that the injected parameter
changes visible mesh geometry and independently measured mask height/area by a frozen tolerance.
Until this geometry gate passes, omit wrong-volume/fill-height labels and do not train that task;
time code, recipe mL, or manifest metadata alone is not rendered volume ground truth.

Cross each latent case with camera-visible nuisance randomization: illumination, exposure, white
balance, focus/blur, noise, compression, glass/reflection, lens dirt/occlusion, and small timing
offsets. Keep camera positions fixed. Use matched pairs (same latent case, different nuisance) and
nuisance-only negatives. Freeze distributions from a written design, not video-5 outcomes.

Keep complete sequences/counterfactuals in one split. Extend the existing leakage manifest so
recipe/ratio cell, previous-to-current transition, asset version, nuisance seed, and duplicate
latent tuple cannot cross train/val/test. Keep the ten video-5 recipes unrendered. Add a harder
challenge set with held-out transition families and nuisance combinations; current v2 test mainly
measures within-twin interpolation.

## Model ladder

Train vision models only after a reviewed pilot manifest passes RGB/mask/label/leakage gates.

1. **No-learning baselines:** fixed ROI/quality gate, robust colour summaries, the first-vial rule,
   a vision-only frozen threshold, and fill-height summaries only after the geometry gate. Do not
   revive a numeric drainage model without measured residual-volume/flush evidence.
2. **Liquid segmentation:** a small pixel-only binary segmenter (for example a lightweight
   U-Net/MobileNet-family model) trained from synthetic semantic masks. Report IoU/Dice,
   invalid/abstention coverage, and the gap between oracle-mask and predicted-mask downstream
   measurements.
3. **Visible fill level, conditional:** only after the paired-render geometry gate passes, estimate
   height/area from the predicted mask before training another network. Report
   pixel/normalized-height error on twin test; do not translate it to real mL without validated
   geometry.
4. **Appearance anomaly:** start with fixed descriptors plus logistic regression or gradient-
   boosted trees. Only then try a small temporal CNN/metric model. Synthetic labels may support
   `appearance_anomaly` or `carryover_injected`; real output must say `carryover-consistent`, not
   proven carryover.
5. **Cross-view fusion:** only after each view has a valid renderer and pixel-only extractor. Use
   per-view descriptors plus recipe targets/order/sequence and an explicit missing-view mask.
   Compare `vial_fill` alone, each added view, recipe-only, and all-view conditions.

Weights, manifests, predictions, and renders live under `FLUIDMIX_OUTPUTS`; Git receives only
small reviewed configs/schemas and evidence summaries.

## Validity and scoring

Choose thresholds/models on twin validation only, tag the freeze, and score twin test once. For
each visible fault type report counts plus false alarms, misses/recall, precision where meaningful,
and abstentions. Also report liquid-mask IoU/Dice, conditional fill-height error, paired nuisance
flip rate, and results for first-transition versus repeat vials. Keep deterministic baselines on
exactly the same eligible records.

Required falsifiers:

- nuisance-only changes cause flags;
- performance collapses on held-out lighting/WB/blur/noise/compression/lens conditions;
- oracle-mask performance does not survive the predicted-mask path;
- sequence, transition, recipe, nuisance, or byte leakage is found;
- gain disappears against the simplest vision baseline;
- the model relies on clock overlays, filenames, latent labels, or inputs unavailable at deploy;
- visually indistinguishable synthetic causes receive confident different cause outputs.

After everything is frozen, use video 5 once for measurement coverage, quality/abstention,
within-window stability, and descriptive transition-versus-repeat appearance. Test the existing
real lighting variation explicitly. Do **not** report real defect recall, cause accuracy, residue
fraction, or wrong-volume accuracy: no such real labels exist. A transfer failure is a result, not
permission to tune on video 5.

## Implementation order and concrete deliverables

1. Create a fresh worktree/branch from the reviewed handoff ref. Re-run tests, synthetic smoke,
   docs check, and the 4,688-file manifest check.
2. Freeze the verified simulator recipe (`scripts/twin_render_sequence.py`). Add render
   safety/manifest tests around it; reproduce its RGB+semantic output locally, then on Brev; add
   and verify a liquid semantic class before dataset work. Stop the instance after the bounded work.
3. Implement separate model-input and label/provenance manifests plus a tiny paired pilot. Run
   input-allowlist, leakage, RGB/mask, and visual-content gates; freeze before scaling.
4. Train the fixed-feature baseline and liquid segmenter; evaluate on disjoint twin test with
   explicit denominators and nuisance/oracle-gap tests.
5. Add carryover-like appearance cases. Add visible fill-level only after its paired geometry gate;
   add `tank_level` only through a real-view-matched scene and keep cap/seal structural evidence
   separate.
6. In parallel, run the frozen Cosmos3-Nano Brev comparison. Do not connect it to an operator UI
   until evidence faithfulness and template comparison pass.
7. Freeze the full vision/evidence/Cosmos configuration, then perform the single video-5 transfer
   audit and write the narrow result.

Suggested new interfaces/files (names may change before implementation):

- `src/fluidmix/evidence_records.py`: immutable source/quality/validity/partition records;
- `src/fluidmix/frame_quality.py`: camera-visible quality and abstention reasons;
- `scripts/render_twin_dataset.py`: fixed-view sequence manifest and external renderer wrapper;
- `scripts/check_render_dataset.py`: RGB/mask/hash/split/counterfactual gates;
- a dataset-loader boundary that rejects oracle/latent/provenance fields from deployable inputs;
- `scripts/train_vision.py` and `scripts/evaluate_vision.py`: frozen model/baseline runner;
- `scripts/run_cosmos_large_eval.py`: provider-neutral Nano/Reason2 adapter using the existing
  strict contract/scorer.

Implement test-first. Each experiment needs a written hypothesis, exact inputs, falsifier,
command, commit/tag, seed, denominators, output path/hash, and one result review. Use cheap scouts
for searches/sweeps; one adversarial review at each major approach/freeze/result decision is
enough—do not review every routine command.

## Verified simulator and Cosmos paths (2026-10-24)

These were run by a short path-paving session, not as experiments. They prove the plumbing works;
they say nothing about liquid realism, detector quality or Cosmos explanation quality. Full runs
(dataset rendering, Cosmos evaluations) are the successor's job under the gates in this brief.

### Workflow

Build and debug locally on the GB10 (free, ~20 s per render launch once warm), look at the frames,
then run full renders on Brev (L40S) for speed/fidelity. Never start a Brev run with a script that
has not produced a correct local contact sheet first.

### Isaac Sim reference recipe

Scripts (in this repo): `scripts/twin_render_sequence.py` (FillCam frame sequence, RGB + semantic
mask + `summary.json`), `scripts/twin_contact_sheet.py` (RGB-over-mask sheet for eyeballing),
`scripts/brev_isaac_setup.sh` (pip install on a plain Brev VM). Input: the public twin's
self-contained `stations/fill_station/` plus `cameras/` (9 MB tarball; the full line is 715 MB and
not needed for FillCam).

Install (both hosts): Isaac Sim **6.0.1 pip wheels, Python 3.12**, in a dedicated venv—not the shared
ML venv, because Isaac pins its own torch:
`uv venv --python-preference only-managed --python 3.12 .venv` then
`uv pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match`
(~18 GB on aarch64, ~24 GB on x86_64). Use a uv-managed Python: the system Python lacks headers
and the `psutil` build fails. NVIDIA's guide also recommends installing the cu130 torch first on
CUDA 13 hosts; that was not done here and rendering still worked—do it on the next install.
Set `OMNI_KIT_ACCEPT_EULA=YES` (documented). On the GB10 (aarch64/DGX Spark) also set
`LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` (documented; Kit refuses to start without it).

- Local GB10: venv at `/home/hackathon5/liamreporoot/isaacsim-env/.venv`. Use
  `--renderer PathTracing` (the script default). `RealTimePathTracing`, SimulationApp's default,
  segfaulted inside RTX/Hydra while opening the stage on the GB10 (observed once, undocumented).
- **Reuse first:** Liam kept the instance `claude-isaac-smoke-1024` (ID `m1vgz4tbs`, `g6e.xlarge`)
  **stopped** for the successor. It already has Isaac Sim 6.0.1 in `~/isaac/.venv`, the
  fill-station twin in `~/isaac/csi_digital_twin_v1`, the scripts, and a warm first-launch cache.
  `brev start claude-isaac-smoke-1024`, wait for `RUNNING`/`READY` in `brev ls`, `brev copy` the
  current repo scripts over the old copies, and run. Stop and verify `STOPPED` when done. Do not
  delete it without Liam's decision.
- New instance, if needed: `brev create <name> --type g6e.xlarge` (AWS L40S 48 GB, 4 vCPU, stoppable, $2.23/h quoted),
  `brev copy` the tarball and scripts, `brev exec <name> "bash ~/isaac/brev_isaac_setup.sh"`, then
  run the render with `brev exec` and copy results back with `brev copy`. Stop with `brev stop` and
  confirm `STOPPED` in `brev ls`; `brev stop` took several minutes to return.

Run: `python scripts/twin_render_sequence.py --twin <csi_digital_twin_v1> --frames 1:687:5
--subframes 64 --out <dir>` (time codes are the twin's 1–686 animation at 30 fps).

Measured:

| Host | Frames | Result | Timing |
|---|---|---|---|
| GB10 local | 3 (t=1, 301, 601) | 3/3 non-constant RGB + aligned semantic | ~20 s per launch incl. startup |
| Brev g6e.xlarge L40S | 138 (t=1..686 step 5), 64 subframes | 135/138 good; t=1,6,11 uniformly black (before the retry fix) | 654 s total: ~8 min first-launch wait with CPU busy/GPU idle, then median 1.5 s/frame |

The Brev run cost about $2 of compute (12:17–13:11 CDT at the quoted rate; not an invoice) plus
retained storage on the stopped instance `claude-isaac-smoke-1024`. Outputs and logs are at
`/home/hackathon5/liamreporoot/isaacsim-env/runs/brev-full/` (not in Git).

Pitfalls found (marked documented vs. observed):

1. **Clipping-range re-sync breaks FillCam (observed).** The twin's `tools/render_example.py`
   toggles the camera clipping range in the session layer as a CapSealCam workaround. On FillCam it
   left the rgb annotator empty (shape `(0,)`)—this is what the Codex Brev recovery run hit. The
   reference script applies it only to CapSealCam. CapSealCam itself is untested here.
2. **First captures black on a fresh machine (documented in spirit).** NVIDIA documents skipped
   first frames/missing annotator data and recommends warm-up `rep.orchestrator.step()` calls, and
   `rt_subframes >= 2` when materials load late. The reference script retries each frame until it
   is full-size and non-constant; never keep a frame without that check.
3. **Long first launch on Brev (observed).** About 8 min passed after "Startup Complete" with CPU
   busy and GPU idle before frame 1 (likely shader/material compilation; not confirmed in NVIDIA
   docs). Budget for it; use a larger-vCPU type if launches repeat, and do not kill early.
4. **Intermittent Kit startup stall (observed).** Two of about twelve local launches blocked inside
   `SimulationApp` startup (0% CPU, no log output, futex wait) and never progressed; identical
   reruns started in 8 s. If a launch shows no Kit log lines for several minutes, kill and relaunch.
   Run renders under a generous `timeout` (30 min+) and judge only from logs.
5. **`enable_extension("omni.replicator.core")` is unnecessary** in 6.0.1 (already loaded) but
   harmless (retested with semantics). An earlier "hang" attribution was a misread startup stall.
6. **Launchable 500s.** See Track A; plain `brev create` worked on the first `g6e.xlarge` attempt.
7. **No liquid class in the twin.** Semantic classes present: `vial`, `carrier`, `nozzle`, `label`,
   `track`, `background`, plus unlabelled prims. Liquid shares the `vial` label.

### Cosmos reference recipe (Hugging Face CLI)

`hf` (v2.3.0) is installed and logged in as Liam's account. Cosmos3-Edge and Cosmos3-Nano are **not
gated**; Cosmos-Reason2-8B reports `gated=auto` (accept its license on the model page first).
Download only reasoner files, pinned to a revision:

    hf download nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 \
      --include '*.json' --include '*.jinja' --include '*.txt' --include '*.md' \
      --include 'transformer/*' --include 'vision_encoder/*' --include 'text_tokenizer/*'

`--include`/`--exclude` take **one pattern per flag** in hf 2.x; listing several patterns after one
flag silently downloads those paths instead. Edge's chat template is `chat_template.jinja`; Nano's is
`chat_template.json`. Load from the HF cache with
`snapshot_download(..., allow_patterns=<same list>, local_files_only=True)`; a plain
`local_files_only` call rejects the intentionally partial snapshot. Cached revisions:
Edge `344d602b128d1bbdacb43b08d0a3626f46343e29` (7.3 GB), Nano
`e59a53c25979a090fa8706c9acc0c254a6e89b92` (32 GB).

Scripts: `scripts/cosmos_smoke.py` (one image, JSON result) and `scripts/cosmos_batch.py` (folder
of frames, batch-size sweep with GPU-utilisation sampling). Run them with the shared ML venv.

Measured on the GB10 (Brev frame t=686 / all 138 Brev frames):

| Model | Run | Load | Result |
|---|---|---|---|
| Cosmos3-Edge | 1 image, thinking on, 96-token cap | 27 s, 2.4 B params loaded, 5.0 GB peak | Hit the cap mid-"Okay, let's see…" reasoning; on-topic (blue liquid in left tube) |
| Cosmos3-Edge | 138 frames, `--no-thinking`, 64-token cap | — | Batch 1: 2.4 frames/s, 40 tok/s, 37 W. Batch 8: 8.7 frames/s. Batch 32: 12.1 frames/s, 257 tok/s, 57 W, 7.1 GB |
| Cosmos3-Nano | 1 image, default template, 256-token cap | 104 s, 8.8 B params loaded (reasoner subset of the 16B checkpoint), 17.7 GB peak | "A vial is being filled with blue liquid." (11 tokens, 1.85 s) |

Edge captions are not trustworthy: it called the nearly empty t=16 vial "filled to the rim".
Captions at batch 1 and batch 32 differed on 53/138 frames (padding/bf16 numerics), so freeze the
batch size in any evaluation. Nano was smoke-tested only; its full sweep is left to the successor.
Nano fits comfortably on the GB10, so a local Nano evaluation is possible before paying for an H100.

Use the GPU: batch-1 inference leaves most of the GB10 idle. Batch 32 gave 4.9x the frame
throughput of batch 1 for Edge at 7 GB peak memory and well under full power, so larger batches,
and running Cosmos alongside an Isaac render, are both open. Cosmos3-Edge's template supports
`enable_thinking=False`; without it the model spends its whole token budget reasoning aloud
(the 2026-10-24 Codex token-cap failures look consistent with this, unverified).

## Local ML environment (GB10)

Use the shared venv at `/home/hackathon5/liamreporoot/shared-ml-env/.venv` for local training,
evaluation and data analysis instead of creating per-worktree environments. It lives outside every
worktree, so do not commit it; record its lock hash in experiment manifests.

- Python 3.12.3 (system), aarch64. Locked, hashed pins: `shared-ml-env/requirements.lock`
  (sources in `requirements.in`). Torch matches the Cosmos Edge preflight: `torch==2.10.0+cu130`,
  `torchvision==0.25.0+cu130`, `transformers==5.17.0`, `accelerate==1.12.0`.
- Also included: peft, timm, segmentation-models-pytorch, datasets, huggingface_hub (hf_xet),
  opencv-headless, scikit-image, scikit-learn, scipy, statsmodels, pandas, polars, pyarrow,
  matplotlib, seaborn, optuna, mlflow, trackio, onnx, av, usd-core (`pxr`), jupyterlab, pytest.
  `uv` is installed inside the venv: `.venv/bin/uv pip install --python .venv/bin/python ...`.
  Add the cu130 extra index (`https://download.pytorch.org/whl/cu130`) for torch-family packages,
  and update `requirements.in`/`requirements.lock` whenever you add a package.
- Verified 2026-10-24 by `shared-ml-env/verify_env.py` (result in `verify-result.json`): CUDA
  available, `NVIDIA GB10`, compute capability 12.1, ~130 GB unified memory, finite bf16 matmul,
  and one finite bf16 timm training step. Torch warns that its max supported capability is 12.0;
  this is the same warning the Cosmos Edge preflight ran under and is not a failure. Rerun the
  script after any dependency change.
- Not available locally: Docker (user not in the `docker` group; no passwordless sudo), so
  Isaac Sim and NIM containers cannot run on the GB10 without Liam's fix. vLLM is not installed
  and is not validated on this aarch64/cu130 stack; prefer Transformers locally and NIM on Brev.
- The `hf` CLI is logged in (Liam's account) as of 2026-10-24; Cosmos3 Edge/Nano weights are in
  the default HF cache. No NGC credential is stored, so `nvcr.io` container pulls (NIM, Isaac Sim
  containers) will fail until Liam provides one; the pip-based Isaac path does not need it.

## Compute and stopping discipline

The $40 cap is cumulative; do not reset it for the successor. Before every create or start, record
an observable ledger satisfying:

`confirmed billed/committed spend + retained-storage liability + worst-case new compute/storage <= $40`.

If provider spend/storage cannot be obtained, or the retained resource has no owner-approved
lifecycle/disposition, paid tracks are blocked. The planning allowances are approximately $6 for
the simulator reproduction and $11 for the H100 larger-model run, but they are not proof of budget
remaining. Add to the ledger the 2026-10-24 path-paving run: instance `claude-isaac-smoke-1024`
(`g6e.xlarge`, created 12:17 CDT, verified `STOPPED` 13:11 CDT), about $2 compute at $2.23/hour
quoted, plus storage while it is retained. Liam chose (2026-10-24) to keep it stopped for reuse;
count its disk (~117 GB at about $0.12/GB-month quoted) as retained-storage liability. Never run multiple paid instances concurrently. Select stoppable types; record current
price, storage, stop deadline, and post-stop storage disposition; copy/hash outputs; stop
immediately; and verify `STOPPED`. Do not delete a teammate resource without its owner's decision.

Stop and report `insufficient_evidence` rather than patching when a renderer cannot pass RGB+mask,
a view is not represented, labels leak from outcomes, a detector does not beat simple baselines,
real reference cleanliness remains unverified, or Cosmos cannot beat the deterministic template.

## Human inputs to request once, if missing

Ask Liam for only the dependencies that cannot be reconstructed safely:

1. optionally, the teammate's Isaac/Brev command for comparison (no longer a blocker);
2. license acceptance for Cosmos-Reason2-8B if that fallback is needed (Edge and Nano are not
   gated and `hf` is already logged in);
3. the current cumulative Brev spend if the dashboard exposes it.

Do not ask for or store the credential value in chat or Git.

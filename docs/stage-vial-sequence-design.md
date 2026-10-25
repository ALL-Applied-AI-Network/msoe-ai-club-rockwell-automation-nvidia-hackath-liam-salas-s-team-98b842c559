# Frozen three-vial design with a separate reference bank

Status: **design frozen, unrendered, not training eligible**. The config and builder produce
three separate plan artifacts, not a receipted dataset or result. No real image or label is
training data.

## Source observations and claim boundary

Labeled video 5 has three vials per recipe. The observed fill totals for videos 1–4 are
19/3/6/6. Video 1's exact recipe grouping is unknown; this design assigns it no grouping
labels. A visual audit supports one three-fill run in video 2 and two in video 4, while video
3 remains unlabeled by recipe. The three-vial design is grounded in labeled video 5 and
organizer guidance, not invented video 1 labels.

The experiment asks whether a majority tight appearance mode, corroborated across a
separate same-recipe reference bank, can support stage-specific synthetic target decisions.
It cannot establish a real clean vial, defect, physical carryover extent/cause, calibrated
appearance, pure recipe generalization, sim-to-real validity, latency, fill/volume truth, or
plant action. Test scoring is explicitly **reference-assisted and transductive**: test data
never trains or tunes a model, but same-test-recipe reference measurements support decisions.

## Families, cohorts and matched nuisance

The four split-bound families remain train S0437->S0117, validation S0160->S0380, and test
S0177->S0431 plus S0147->S0074. Each family has two separate cohorts:

- four `reference_only` batches that can never be scored targets; and
- four `scored_target` batches that can never serve as external references.

Each cohort contains the four balanced three-vial profiles exactly once: `[0,0,0]`,
`[.30,0,0]`, `[.30,.12,0]`, and `[.30,.18,.08]`. Profile/alpha/onset are render-label
oracles, never reference or model inputs. Nonzero onset is after t200 for train, after t400
for validation, after t100 for test S0431 and after t400 for test S0074.

Within a family/cohort, all four counterfactual batches share the same per-vial seed vector.
Reference seeds are `base+[1,2,3]`; target seeds are `base+[101,102,103]`. Cohorts and
families have disjoint seed sets. These are separate matched-nuisance counterfactual batches,
not statistically independent observations. The pilot is base nuisance only.

The uncalibrated direct basis remains t200 `[.006,.2,.3]`, t400 `[.228,.16,.155]`, and t100
`[.45,.12,.01]`, opacity `.6`. At/after onset, current cumulative stage appearance blends
toward previous-recipe final appearance. This is renderer material appearance, not
Beer-Lambert physics, residue fraction, or validated real-liquid calibration.

## Cross-batch reference

For a scored target/stage, candidate sets contain later vials in that target batch plus
same-stage vials from selected `reference_only` batches of the exact same recipe. Every V1,
V2 and V3 reference-bank position is eligible. A terminal V3 comparison may be reported as
an optional organizer-guided sanity comparison, but terminal position has no selection
privilege and is never clean truth.

Six availability cases are frozen per target/stage:

- `same_batch_only`, diagnostic and always insufficient without a second batch;
- four `one_reference_batch` alternatives, one for every reference batch with no chosen
  winner; and
- `all_reference_batches`, the only primary disposition source, containing all 12 bank
  images plus any later target-batch vials.

A valid reference needs at least two peers across at least two batch IDs. The pipeline
enumerates all subsets with pairwise dE76 diameter <=2.0 and accepts only the unique largest.
It reports coordinate-median Lab center, peer IDs/origins/count/batch count, target distance,
median/max/MAD radius, pairwise diameter, raw values and fixed six-decimal strings. Threshold
is `max(2.0, max_radius + 1.0)` and outlier means strictly greater.

The explicit assumption is that the unique largest tight cross-batch cluster represents the
majority/reference appearance mode. Reference contamination that creates a wrong majority,
or equal-size competing tight modes, is a falsifier—not something patched with profile or
order information. Thin, tied, missing or mask-mismatched evidence abstains. The primary
pipeline discards a synthetic vial when any valid stage is an outlier, keeps it only when all
three are valid/inlier, and otherwise emits `insufficient_evidence`.

Measurement remains pinned to `skimage.color.rgb2lab` 0.26.0: uint8 sRGB divided by 255,
CIE Lab D65/2 degree, coordinate median under a fixed nonempty liquid mask, then Euclidean
dE76. Target/candidate same-stage masks must match shape and hash. Decisions are retrospective
after the complete three-vial batch; no latency claim is made.

## Separate artifacts and denominators

The plan has 4 families, 32 total batches (16 reference-only and 16 scored-target), 96 vial
launches, and 288 planned stage images. Only the 48 target vials are scored. Six reference
cases × 3 stages × 48 targets give 864 reference-case stage records. Cosmos remains 144 model
calls plus 48 template controls = 192 scored attempts, plus one unscored multi-image smoke.

The CLI atomically creates a new external directory and writes only:

- `render-plan.json`, including renderer settings and synthetic construction fields;
- `model-input-plan.json`, allowlisted to image IDs/placeholders, recipe mL/order/sequence,
  stage metadata, candidate origins/batch IDs, partitions and unrendered evidence; and
- `label-provenance-plan.json`, carrying synthetic render-label provenance.

The consumer receives no combined artifact. Model input recursively excludes profile names,
alpha, onset, clean/observed material RGB, semantic fields and other oracles. Opaque batch IDs
do not encode profiles.

## Runtime binding

New local renders are frozen to the watchdog contract in `src/fluidmix/render_process.py`:
profile `gb10-aarch64-isaac62`, host `spark5`/aarch64, Isaac Sim 6.2.0.0, Kit 110.4.0, and
Python `/home/hackathon5/liamreporoot/isaacsim62-env/.venv/bin/python`. Each logical launch
allows at most two identical process attempts: one retry after zero output at 180 s, missing
startup readiness at 420 s, or a fast crash within 180 s. Readiness is the exact
`Simulation App Startup Complete` marker. This plan only binds those immutable fields; the
watchdog owns orchestration.

The retained Brev Isaac Sim 6.0.1.0 environment at `~/isaac/.venv/bin/python` is a distinct
profile. One artifact set must use one host/runtime profile; local 6.2 and Brev 6.0.1
outputs may never be mixed.

### Frozen validation video cohort deviation

The complete validation family—eight batches, 24 vials—uses the retained Brev L40S and
Isaac Sim 6.0.1 only. This operator-directed runtime deviation keeps every validation target
and reference on one host/version. Validation RGB/Lab values are reported separately and
are never pooled with local 6.2 train output.

Each vial is one capture with the exact sorted union of `range(31, 687, 5)` and `{342}`:
133 frames per vial and 3,192 frames total. The three evidence stages 171/342/686 are
one-based capture ordinals 29/64/133, so all 72 validation stage images are subsets of those
same captures rather than separate renders. The renderer remains FillCam 640×480,
PathTracing, 64 subframes, 60 warm-up updates, at most 20 shared-capture tries per frame,
fixed per-vial seeds, and required nonempty aligned liquid semantics. One Isaac process is
allowed per vial, with no replacement process; the cohort stops at the first failed row.

Every successful capture is encoded at constant 6 fps as one MPEG-4 Part 2 `mp4v` MP4 using
the already installed OpenCV FFmpeg backend. A three-frame remote capability probe opened,
wrote and decoded `mp4v` at 6 fps; the same probe found H.264 unavailable. MP4 is a lossy
review artifact only. Colour measurements and Cosmos stage images use the original
hash-bound PNGs/masks, never decoded video frames. Adding exact time code 342 introduces a
one-code then four-code interval amid the usual five-code spacing, so constant-rate playback
does not reproduce exact source timing. USD interpolation between the three authored
appearance samples is uncalibrated synthetic visualization, not physical mixing dynamics.

Cosmos still explains deterministic evidence and cannot choose disposition or recommend
process changes. NVIDIA documents text/image/video input with a one-image Cosmos example;
Hugging Face documents multiple images for upstream Qwen3-VL. Exact six-image Cosmos3-Nano
support remains gated by one unscored smoke.

# Pipeline interfaces and units

The scaffold implements physics helpers and diagnostic scripts. It has no camera estimator,
threshold service, training pipeline or Cosmos client. The future stage handoffs below are
proposals for INN-11 to review, not existing APIs. See [plan](plan.md) and [proposal](proposal.md).

## Implemented contracts

All functions below live in src/fluidmix/absorbance.py. Vectors use R/G/B channel order.

| Function | Inputs | Output |
|---|---|---|
| srgb_to_linear | sRGB channels, 0–255 | Linear intensity, nominally 0–1 |
| linear_to_srgb | Linear intensity | sRGB floats clipped to 0–255 |
| volume_fractions | Tank volumes in mL | Fractions summing to one |
| mix_absorbance | Recipe and tank absorbance vectors | Dimensionless mixture absorbance |
| expected_color | Recipe, tanks, clear reference i0_lin | Linear intensity |
| carryover_absorbance | New/old recipes, tanks, old fraction 0–1 | Mixture absorbance |
| fit_tank_absorbance | List of (ml_dict, measured_linear_rgb) | tank_a, i0_lin, per-channel R² |
| fit_carryover_fraction | Measured intensity, recipes, tanks, reference | Old fraction and RMS absorbance residual |
| tank_absorbance_from_image | Tank/clear intensities, tank/vial path in mm | Absorbance rescaled to vial path |
| delta_e | Two sRGB 0–255 vectors | CIE76 color distance |

Recipes use t100, t200 and t400 keys. Absorbance is dimensionless; fraction 0.15 means 15%.
The API rejects unknown tanks, nonfinite/negative volumes and missing/nonfinite calibration
for used colored tanks. T-100 defaults to clear. Fitted residue fractions are still clipped
to [0,1]; matched references and physical validity require separate checks.
A clipped endpoint or small residual does not establish a causal diagnosis.
The forward carryover helper accepts one plain real fraction in [0,1]. It raises TypeError
for bools, arrays (including 0-d and NumPy masked values) and non-numbers, and ValueError
for NaN, infinity or values outside [0,1]. A per-channel fraction is not supported. See the
[acceptance audit](acceptance-inn-23-24.md#inn-11-follow-up-audit).

Calibration needs full-rank recipe fractions for the intercept and two colored tanks;
three vials alone do not guarantee identifiability. The fit rejects rank-deficient designs
and invalid intensity vectors, but can return negative absorbance for diagnostics.
See [physical limits](absorbance.md).

### Deterministic synthetic stage decision

`src/fluidmix/stage_decision.py` implements a decision-only boundary for three-stage,
three-vial synthetic batches. `decide_vial_stages` consumes pipeline-produced CIE Lab
statistics and explicit reference provenance; it does not read images, labels or Cosmos
output. Its frozen config supplies the peer-cluster and distance thresholds. A disposition
requires a unique tight peer mode containing at least two eligible reference-bank peers
from at least two batches. Later vials from the target batch may add context, but neither
same-batch evidence alone nor one independent reference batch can authorize `keep` or
`discard`.

The output is `keep`, `discard` or `insufficient_evidence`, with the target and peer evidence
IDs, per-stage delta E76, threshold, reference center and spread, and validity reason codes.
Target stages must belong to one vial and one partition, reference-bank rows cannot reuse
the target batch, and all references must match recipe, stage and split. The function has no
vial-position cleanliness rule and emits no operator-control advice. This is synthetic
colour-distance evidence only: the thresholds are not calibrated, and the result makes no
physical-carryover, defect, cause or sim-to-real claim.

### Verified stage-capture binding

`src/fluidmix/capture_manifest_adapter.py` binds one synthetic `vial_fill` batch of
three vials by three dispense stages to the existing measurement and explanation
contracts. A capture row is accepted only when its RGB and liquid-mask artifacts are
direct, hash-matched 640x480 PNG files and the manifest supplies the frozen twin split,
recipe, family, batch, stage, time and view provenance. Missing, duplicate, wrong-view,
wrong-time, unverified, empty-mask and cross-context rows fail closed with a named reason.

The per-batch `stage_capture_measurements` artifact is an intermediate record, not a
decision. `build_measured_evidence` requires the exact image set referenced by an
allowlisted model-input plan, binds every target and peer to its actual family, recipe,
partition, batch and stage, and then delegates to `vision_measurement.evaluate_model_plan`.
Its output is the canonical deterministic `measured_evidence` consumed by the stage
Cosmos boundary. Peer batches may differ, but recipe, family, partition and stage may
not; this prevents metadata from manufacturing the required independent reference
batches. Oracle/render parameters and labels are rejected from the capture manifest,
and pixel paths remain outside the deployable model-input rows.

This contract verifies file identity, image shape, mask usability and evidence routing.
It does **not** establish that a renderer passed the F2 particle/visible-volume gates,
that any reference is clean in the real plant, or that a synthetic decision transfers to
real footage. Until all three F2 volume stages pass, it is exercised only with constructed
test images and does not authorize case generation, training or a physical carryover claim.

## Proposed stage handoffs

| Stage | Input evidence | Proposed output |
|---|---|---|
| Recipe/context | Recipe ID, tank mL, previous recipe, vial ID, corrected time | Aligned record with source IDs |
| Camera measurement | Camera/frame ID, liquid mask, matched I0, exposure checks | Linear RGB, optional uncalibrated absorbance, validity and uncertainty |
| Drainage model | Recipe sequence, tank volumes, path/flush assumptions | Volume drawn per tank and predicted residue fraction; no camera input |
| Forward physics | Valid calibration, recipe/context, assumed residue fraction | Expected clean/carryover absorbance and intensity |
| Anomaly comparison | Clean-recipe expected value, measured value and frozen threshold | Score, threshold, flag and invalid-measurement status |
| Cause evidence | Valid intensity, current/previous recipes | Fitted fraction, residual and validity limits |
| Cosmos explanation | Quantitative evidence, recipe and uncertainty | Operator explanation linked to evidence IDs |

The tank-camera route needs measured path lengths and a suitable clear reference. The twin
has no tank images. Appearance estimates still need validation as transmission measurements.
Explanation must preserve invalid/missing evidence instead of advising plant changes from it.

### Proposed record contract for INN-11 review

These fields are a documentation proposal, not implemented classes, a serialized schema,
or team agreement. Revision 2 (2026-10-19) adapts it to the synthetic-only decision of
2026-10-16 in [evaluation](evaluation.md#partition-rules).

Every record carries `record_id`, source IDs, producer/config version, `validity`
(`valid`, `uncertain`, `invalid`), reason codes and an uncertainty description.
Use an explicit unavailable value/reason when uncertainty is not measured; do not imply zero.
Derived records link their upstream record IDs and calibration/model/threshold versions.

Every record also carries its partition, so evaluation code can refuse to mix sets:

| Field | Values | Rule |
|---|---|---|
| `data_origin` | `twin`, `real_check` | Twin renders vs real Dataset 2 video. Never inferred from file paths at scoring time |
| `split_role` | twin: `train`, `val`, `test`, `buffer`; real: `real_validation`, `calibration` | Copied from data/sim_recipes.csv or data/real_check_windows.csv, never assigned by a model or script at scoring time. `buffer` is never used |
| `split_version` | `inn-7-splits-v2` | The git tag of the frozen split. A record with another version is invalid for scoring |

A derived record inherits all three from its inputs. Inputs that disagree make it invalid
(reason `partition_mismatch`), because a mixed record could carry real or test information
into training or tuning. Twin test metrics and real-check results are reported apart.

| Record | Stage-specific fields and units |
|---|---|
| Recipe | Recipe ID (`sim_recipe_id` for twin), vial ID, tank volumes (mL), previous recipe ID or missing reason, video/frame IDs, corrected interval (s), timing-map version |
| Ground-truth label | Recipe ID, vial ID, label source, injected tank volumes (mL), injected residue fraction (0–1) and previous recipe, injected fault type or `none`, defect flags from the frozen definitions |
| Measurement | Camera/ROI/reference IDs, exposure/quality checks, linear RGB (required, nominal 0–1), absorbance (optional, dimensionless) with `absorbance_status` (`uncalibrated` until a calibration is agreed), reference/path provenance (mm), measurement version |
| Drainage prediction | Recipe sequence IDs, volume drawn per tank (mL), predicted residue fraction (0–1), path-volume/flush assumptions, drainage model and config version |
| Clean prediction | Recipe/measurement context, calibration ID, expected clean RGB and/or absorbance, prediction uncertainty and domain limits |
| Anomaly | Measurement and clean-prediction IDs, score definition/units, frozen threshold/config, decision or abstention reason |
| Cause | Anomaly ID, previous-recipe evidence, carryover fraction (0–1), RMS absorbance residual, clipping/identifiability status and alternative explanations |
| Explanation | Evidence IDs, reported status/limits and explanation-model version; no new quantitative or plant-action claim without supporting evidence |

Ground-truth labels are answers, not evidence. Twin labels come from the render parameters
passed to scripts/twin_liquid_variant.py, never from a colour score (label source
`render_parameters`); see [labels](labels.md). Video 5 has no defect labels, so real-check
records have none. Labels are stored apart from measurements; only scoring code reads them.
Models, thresholds and the drainage model never take a label as input.

Absorbance stays `uncalibrated` because real calibration is deferred and the only real fit
gives negative values (see [absorbance](absorbance.md)). An uncalibrated value can be shown
for diagnosis but cannot set a threshold, score a test set or support a cause claim.

The drainage prediction (INN-35) uses the recipe sequence only, so it is valid even when a
camera measurement is invalid. Its path-volume and flush values are assumptions until
checked against real transitions. The drainage-only baseline (INN-34) and the cause stage
read it; the anomaly stage does not.

**Circularity risk.** scripts/twin_liquid_variant.py sets each twin vial's material from
`mix_absorbance` with guessed tank values. If the twin's clean prediction also came from
`expected_color` or `mix_absorbance` with those values, a twin comparison would test the
formula against itself and could look accurate even when the physics is wrong. Proposed
rule: in the twin, the clean prediction for a recipe is the measured colour of a clean
render of that recipe (zero injected residue and fault), extracted by the same measurement
code, with `calibration ID` set to `twin_clean_render` and the render IDs as sources. The
twin then tests detection, measurement and the drainage model under a known colour
mapping, not that mapping. Twin results cannot show that `fluidmix.absorbance` or the
guessed tank values describe real fluid. Only `real_check` results are evidence for or
against that, and the write-up must say so next to every twin metric.

Anomaly scoring must compare with the clean recipe. It must not subtract an estimated
carryover contribution before deciding whether the vial is anomalous. Missing previous
recipe blocks that carryover attribution, but need not block a clean-recipe comparison
when its own recipe, measurement, calibration and threshold are valid. Invalid camera or
reference evidence blocks absorbance inference; downstream records carry the invalid
state and reasons. An explanation cannot turn missing or invalid evidence into a cause.
INN-26 and INN-27 own implementation of those decisions and abstention behavior.

## Script handoffs

- measure_video_clock.py -> verify_still_alignment.py -> align_still_labels.py: videos, still
  indexes and video-5 recipes -> clock fits, still-to-frame map and aligned video-5 labels in
  FLUIDMIX_OUTPUTS. See [video timing](video_timing.md). fixlabels.py is legacy.
- color.py: video/recipe endpoints -> rough colors.json using fresh temporary frames.
  It does not directly export data/video5_vial_colors.csv.
- fit_absorbance_video5.py: committed CSVs -> stdout diagnostic, no deployable calibration.
- twin_liquid_variant.py: external station, recipe, tank JSON -> override .usda.
  Rendering is a separate external command; see [twin](twin.md).

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
The forward carryover helper accepts one `numbers.Real` scalar in [0,1] (Python or NumPy
int/float, Fraction). It raises TypeError for bools, arrays (including 0-d and NumPy masked
values), NumPy time deltas and anything that is not a `numbers.Real` (for example Decimal,
complex, None or text), and ValueError for NaN, infinity or values outside [0,1], including
integers too large for a float. A per-channel fraction is not supported. See the
[acceptance audit](acceptance-inn-23-24.md#inn-11-follow-up-audit).

Calibration needs full-rank recipe fractions for the intercept and two colored tanks;
three vials alone do not guarantee identifiability. The fit rejects rank-deficient designs
and invalid intensity vectors, but can return negative absorbance for diagnostics.
See [physical limits](absorbance.md).

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
2026-10-16 in [evaluation](evaluation.md#partition-rules). Revision 3 (2026-10-19) applies
an independent agent review of revision 2 (PR #7); human review is still pending.

Every record carries `record_id`, source IDs, producer/config version, `validity`
(`valid`, `uncertain`, `invalid`), reason codes and an uncertainty description.
Use an explicit unavailable value/reason when uncertainty is not measured; do not imply zero.
Derived records link their upstream record IDs and calibration/model/threshold versions.

Every record also carries its partition, so evaluation code can refuse to mix sets:

| Field | Values | Rule |
|---|---|---|
| `data_origin` | `twin`, `real_check` (`real_calibration` reserved) | Twin renders vs real Dataset 2 video. Never inferred from file paths at scoring time |
| `split_role` | twin: `train`, `val`, `test`, `buffer`; real: `real_validation` | Copied from the `role` column of data/sim_recipes.csv or data/real_check_windows.csv, never assigned by a model or script at scoring time. `buffer` is never used |
| `split_version` | `inn-7-splits-v2` | The git tag of the frozen split (data/sim_split_spec.json calls it `v2`; the tag is authoritative). A record with another version is invalid for scoring |
| `eligible`, `exclusion_reason` | `1`/`0` and a reason code | Copied from data/real_check_windows.csv for real-check vials (vial 1 `previous_recipe_unknown`, vial 10 short window); twin vials follow the eligibility table in [evaluation](evaluation.md). An ineligible record is never scored |

A real vial used to fit anything leaves the real check entirely: it gets a separate origin
and role (`real_calibration`, `calibration`), as [evaluation](evaluation.md) requires. No such
vial exists today; the role is reserved, not populated.

Two kinds of input are treated differently:

- **Per-vial evidence** (recipe, measurement, clean prediction, drainage prediction, label)
  must share `data_origin`, `split_role` and `split_version`. A derived record inherits
  them; inputs that disagree make it invalid (reason `partition_mismatch`).
- **Frozen artifacts** (trained model, threshold, defect definitions, drainage parameters,
  config) record the split they were fitted or tuned on and the freeze tag. They may be
  applied only in one direction: `train` or `val` artifacts may be used on `test` and
  `real_validation` records, never the reverse. An artifact fitted on `test` or
  `real_validation`, or used before its freeze tag, makes the record invalid
  (reason `artifact_leak`).

Twin test metrics and real-check results are reported apart.

| Record | Stage-specific fields and units |
|---|---|
| Recipe | Recipe ID (`sim_recipe_id` for twin), vial ID, tank volumes (mL), previous recipe ID or missing reason, video/frame IDs, corrected interval (s), timing-map version |
| Ground-truth label | Recipe ID, vial ID, time code (frame), label source, injected tank volumes (mL), injected residue fraction (0–1) and previous recipe, injected fault type or `none`, defect flags from the frozen definitions |
| Measurement | Camera/ROI/reference IDs, exposure/quality checks, linear RGB (required, nominal 0–1), absorbance (optional, dimensionless) with `absorbance_status` (`uncalibrated` until a calibration is agreed), reference/path provenance (mm), measurement version |
| Drainage prediction | Recipe sequence IDs, volume drawn per tank (mL), predicted residue fraction (0–1), path-volume/flush assumptions, drainage model and config version |
| Clean prediction | Recipe/measurement context, `reference_source` (see below) and its calibration ID if any, expected clean RGB and/or absorbance, prediction uncertainty and domain limits |
| Anomaly | Measurement and clean-prediction IDs, score definition/units, frozen threshold/config, decision or abstention reason |
| Cause | Anomaly ID, previous-recipe evidence, carryover fraction (0–1), RMS absorbance residual, clipping/identifiability status and alternative explanations |
| Explanation | Evidence IDs, reported status/limits and explanation-model version; no new quantitative or plant-action claim without supporting evidence |

Ground-truth labels are answers, not evidence. Twin labels come from the render parameters
passed to scripts/twin_liquid_variant.py, never from a colour score (label source
`render_parameters`); see [labels](labels.md). Video 5 has recipe and fill-window labels
but no defect, cause or residue labels, so real-check records have no label record.
Labels are stored apart from measurements, and who may read them depends on the split:

| Split | Label readers |
|---|---|
| `train` | Training code, to fit models |
| `val` | Tuning code, to set thresholds and defect-definition numbers before the freeze |
| `test` | Scoring code only, after the freeze tag |

At prediction time no model, threshold or drainage model takes a label as input.

Absorbance stays `uncalibrated` because real calibration is deferred and the only real fit
gives negative values (see [absorbance](absorbance.md)). An uncalibrated value can be shown
for diagnosis but cannot set a threshold, score a test set or support a cause claim.

The drainage prediction (INN-35) uses the recipe sequence only, so it is valid even when a
camera measurement is invalid. Its path-volume and flush values are assumptions. If they
are tuned, that is done on twin `train`/`val` and frozen like any other artifact; tuning
them on real transitions would move those vials to `calibration` and out of the real check.
The drainage-only baseline (INN-34) and the cause stage read it; the anomaly stage does not.

**Clean prediction source.** `reference_source` names where the expected clean colour came
from. It is not a calibration: [evaluation](evaluation.md) calibrates nothing from renders.

| Data | `reference_source` | Status |
|---|---|---|
| Twin | `twin_clean_render`: measured colour of a clean render of the same recipe (zero injected residue and fault), extracted by the same measurement code, render IDs as sources. The clean render has the same `split_role` as the vial it serves | Proposed |
| Real check | **Open team decision.** The 10 real recipes are never rendered, so no clean render exists. Option A: `formula_uncalibrated` (`expected_color` with the guessed tank values), reported as a diagnostic only and never used to claim physical accuracy. Option B: the real check waits for an agreed calibration | Undecided |

**Circularity risk.** scripts/twin_liquid_variant.py sets each twin vial's material from
`mix_absorbance` with guessed tank values, for clean and defective renders alike. Two
consequences follow:

- If the twin's clean prediction came from `expected_color` or `mix_absorbance`, a twin
  comparison would test the formula against itself. The clean-render source avoids that
  for anomaly scoring, but only moves the circularity: the clean render's colour is still
  set by the same formula. The twin tests detection, measurement and the drainage model
  under a known colour mapping, not the mapping.
- The cause stage uses `carryover_absorbance` and `fit_carryover_fraction`, the same linear
  mixing the renders were built with. Twin residue-fraction error therefore measures
  self-consistency (render and measurement noise), not whether real residue mixes that way.

[twin](twin.md) proposes fitting the render map until renders match `expected_color()`.
That would make the clean render and the formula agree by construction, which strengthens
this circularity; it should be revisited with this contract. Twin results cannot show that
`fluidmix.absorbance` or the guessed tank values describe real fluid. Only `real_check`
results bear on that, and the write-up must say so next to every twin metric.

Anomaly scoring must compare with the clean recipe. It must not subtract an estimated
carryover contribution before deciding whether the vial is anomalous. Missing previous
recipe blocks that carryover attribution, but need not block a clean-recipe comparison
when its own recipe, measurement, calibration and threshold are valid and the record is
eligible. Real-check vial 1 is not eligible, so it is excluded entirely. Invalid camera or
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

## Phase 2 UI output boundary (implemented)

The UI's actual final-output adapter is the read-only Python materializer in
`scripts/materialize_ui_data.py`; it writes the schema-validated `ui-data/active-run.json`.
It joins the published development-history fixture's immutable SQLite evidence to its
development records and, for an approved future final source, reads only the final WP7
decision contract. SQLite is opened with `mode=ro&immutable=1`, `query_only`, and an
authorizer denial for `sim_metadata`; the writable WP5 `EvidenceStore` is not used. The
development source is the exact committed V3/V4 fixture/reel/manifest and has 18/18
`insufficient_evidence` decisions. This is measured source observation plus derived UI
materialization, not a plant or physical-validity claim. Final WP7 behavior is unverified:
no published final decision artifact is present.

The sole browser input switch is the one-line path/value in `ui-data/source.json`:
`{"source":"development"}`. Package preparation is centralized: `npm run prepare:data`
(from `ui/`) invokes the materializer, while `npm run dev`, `npm run build`, `npm run preview`
and the E2E web server prepare through that same command. The browser never edits source
fixtures. `?demo=1` enables muted autoplay and progressive cue selection for development or
future final sources; E2E advances the source-backed reel through stable cue boundaries.
Mock data and final-like keep/discard records are test-only fixtures. No sealed inputs, WP1
test/video-5/grounding qualification, oracle, or simulator truth are part of this boundary.

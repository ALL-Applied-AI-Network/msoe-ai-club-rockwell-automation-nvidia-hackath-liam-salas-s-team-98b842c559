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

Calibration needs full-rank recipe fractions for the intercept and two colored tanks;
three vials alone do not guarantee identifiability. The fit rejects rank-deficient designs
and invalid intensity vectors, but can return negative absorbance for diagnostics.
See [physical limits](absorbance.md).

## Proposed stage handoffs

| Stage | Input evidence | Proposed output |
|---|---|---|
| Recipe/context | Recipe ID, tank mL, previous recipe, vial ID, corrected time | Aligned record with source IDs |
| Camera measurement | Camera/frame ID, liquid mask, matched I0, exposure checks | Linear intensity, absorbance, validity and uncertainty |
| Forward physics | Valid calibration, recipe/context, assumed residue fraction | Expected clean/carryover absorbance and intensity |
| Anomaly comparison | Clean-recipe expected value, measured value and frozen threshold | Score, threshold, flag and invalid-measurement status |
| Cause evidence | Valid intensity, current/previous recipes | Fitted fraction, residual and validity limits |
| Cosmos explanation | Quantitative evidence, recipe and uncertainty | Operator explanation linked to evidence IDs |

The tank-camera route needs measured path lengths and a suitable clear reference. The twin
has no tank images. Appearance estimates still need validation as transmission measurements.
Explanation must preserve invalid/missing evidence instead of advising plant changes from it.

### Proposed record contract for INN-11 review

These fields are a documentation proposal, not implemented classes, a serialized schema,
or team agreement. Each record carries `record_id`, source IDs, producer/config version,
`validity` (`valid`, `uncertain`, `invalid`), reason codes and an uncertainty description.
Use an explicit unavailable value/reason when uncertainty is not measured; do not imply zero.
Derived records link their upstream record IDs and calibration/model/threshold versions.

| Record | Stage-specific fields and units |
|---|---|
| Recipe | Recipe ID, vial ID, tank volumes (mL), previous recipe ID or missing reason, video/frame IDs, corrected interval (s), timing-map version |
| Measurement | Camera/ROI/reference IDs, exposure/quality checks, linear RGB (nominal 0–1), optional absorbance (dimensionless), reference/path provenance (mm), measurement version |
| Clean prediction | Recipe/measurement context, calibration ID, expected clean RGB and/or absorbance, prediction uncertainty and domain limits |
| Anomaly | Measurement and clean-prediction IDs, score definition/units, frozen threshold/config, decision or abstention reason |
| Cause | Anomaly ID, previous-recipe evidence, carryover fraction (0–1), RMS absorbance residual, clipping/identifiability status and alternative explanations |
| Explanation | Evidence IDs, reported status/limits and explanation-model version; no new quantitative or plant-action claim without supporting evidence |

Anomaly scoring must compare with the clean recipe. It must not subtract an estimated
carryover contribution before deciding whether the vial is anomalous. Missing previous
recipe blocks that carryover attribution, but need not block a clean-recipe comparison
when its own recipe, measurement, calibration and threshold are valid. Invalid camera or
reference evidence blocks absorbance inference; downstream records carry the invalid
state and reasons. An explanation cannot turn missing or invalid evidence into a cause.
INN-26 and INN-27 own implementation of those decisions and abstention behavior.

## Script handoffs

- fixlabels.py: source labels/index -> corrected video-5 labels and last-still map.
  See [data](data.md) for the exact positional root convention.
- color.py: video/recipe endpoints -> rough colors.json using fresh temporary frames.
  It does not directly export data/video5_vial_colors.csv.
- fit_absorbance_video5.py: committed CSVs -> stdout diagnostic, no deployable calibration.
- twin_liquid_variant.py: external station, recipe, tank JSON -> override .usda.
  Rendering is a separate external command; see [twin](twin.md).

# Brev FillCam render feasibility protocol

Status: frozen proposal before provisioning. This protocol tests renderer compatibility and
camera-visible synthetic appearances only. It does not create defect labels, estimate carryover,
validate physical mixing, or measure sim-to-real transfer.

## Question and acceptance boundary

Can the organizer's Isaac Sim 6.0.1 launchable render the unmodified fill-station asset and a
small set of geometry-consistent liquid-material/animation-frame variants through the fixed
`FillCam` view? A passing run establishes only that the pinned twin and renderer executed and
produced inspectable 640 x 480 PNGs.

The run is rejected if the unchanged smoke render fails, is blank or unreadable, has the wrong
dimensions/view, or if source-asset hashes change. Phase B is rejected if a variant layer does not
reference the unchanged station, preserve 30 fps and time codes 1--686, author the requested
shader values, or if any render is visibly broken. A rejection stops the approach; it is not
patched during the same paid run.

## Synthetic appearance equation

For three unitless *linear diffuse-RGB renderer parameters*, define

`c(alpha) = (1 - alpha) c_A + alpha c_B`, for `0 <= alpha <= 1`,

where `c_A = (0.006, 0.2, 0.3)` is the twin's authored liquid colour and
`c_B = (0.45, 0.12, 0.01)` is an arbitrary orange-like endpoint. Opacity is fixed at 0.6.
The equation interpolates shader inputs only. `alpha` is not a residue fraction, the endpoints are
not configured tank colours, and neither endpoint comes from real video or a physical absorption
model. The separate decoy `c_D = (0.28, 0.02, 0.22)` checks another visible hue without assigning
a cause.

## Frozen render matrix

All jobs use `FillCam`, PathTracing, 640 x 480 output and 32 subframes. Neutral IDs and filenames
carry no defect answer.

| ID | Frame | Stage/material | Purpose |
|---|---:|---|---|
| S00 | 686 | unchanged station | Phase-A compatibility smoke |
| M00 | 686 | `c_A`, opacity 0.6 | direct-override positive control |
| M01 | 686 | `c_B`, opacity 0.6 | endpoint-B appearance |
| M02 | 686 | `(0.0615, 0.19, 0.26375)`, opacity 0.6 | alpha 0.125 appearance |
| M03 | 686 | `(0.117, 0.18, 0.2275)`, opacity 0.6 | alpha 0.25 appearance |
| M04 | 686 | `(0.228, 0.16, 0.155)`, opacity 0.6 | alpha 0.5 appearance |
| M05 | 686 | `c_D`, opacity 0.6 | unrelated-hue decoy appearance |
| F00 | 400 | unchanged station | stream-free animation-frame candidate |
| F01 | 500 | unchanged station | second stream-free animation-frame candidate |

Frames 400 and 500 are time codes in a fixed animation, not measured mL and not wrong-fill
labels. The rendered pixels must be inspected before either frame is even described as a useful
lower-level example.

## Execution, privacy and cost gates

- Use exactly one recommended `g6e.4xlarge` from organizer launchable
  `env-35JP2ywERLgqtD0b0MIeK1HnF46`; do not substitute a cheaper warned type or create a fallback.
- Transfer only the public canonical twin plus the frozen scripts/config. Transfer no real video,
  `.env`, API key, model weights, operator data or other sensor signal.
- Record source hashes locally and verify them remotely before rendering. Pass Isaac's documented
  anonymous-telemetry and structured-log disabling flags, record the actual launchable/runtime/GPU,
  and retain raw stdout/stderr in the ignored run directory.
- Run S00 first with a 20-minute process timeout and no retry. Continue to the eight Phase-B jobs
  only if S00 copies back and passes the mechanical and visual checks.
- Target at most two compute hours on the quoted $3.605088/hour instance (about $7.21). Copy back
  results and logs, verify hashes, then stop the instance. Liam will delete the stopped instance
  tomorrow; short-lived stopped storage is explicitly accepted, while recurring retention is not.

## Outputs and limits

Raw USD layers, PNGs and logs stay under ignored `outputs/`. The repository may retain only the
small frozen manifest, scripts, tests, hashes, timings and an evidence-labelled conclusion. No
Cosmos prompt, threshold, training, held-out real-video check or performance score is part of this
run. A separate preregistration and one material-result review are required before any rendered
scene is used to make an explanation-result claim.

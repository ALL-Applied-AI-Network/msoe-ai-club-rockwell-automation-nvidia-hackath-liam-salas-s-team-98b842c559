# Physical liquid simulation v1

Status: **O1b visually accepted, F1 passed 2/2 substantive runs, the F2 palette smoke failed
0/1 and the explicit global-blend fallback is in progress; no output is training eligible**.
This design supersedes the proposed diffuse-colour/flat-opacity v2 render path. Historical v1
renders remain immutable evidence but cannot be combined with any output from this design.

## Scope and claim boundary

The immediate sequence is: qualify transmissive optics, qualify GPU PBD particles on the
retained Brev host, then build one three-pour simulation. The stopped validation video cohort
does not resume until both gates pass and its complete source plan is regenerated.

This work does **not** fit absorbance, test an absorbance estimator, calibrate tank colour from
real images, train a detector, score a test split, or run Cosmos. Transmission colours and
stage-mixture colours are uncalibrated render parameters. Passing the gates would establish a
better-grounded synthetic renderer, not sim-to-real transfer or physical concentration accuracy.

## Grounded facts

Documented by NVIDIA:

- OmniSurface specular transmission models colour filtering with Beer's law. Transmission
  depth is scene-scale dependent; a non-thin-walled material represents a finite volume and
  refracts on entry and exit. Geometry opacity is a different cutout mechanism. Black
  refraction in path tracing may require more total and specular/transmission bounces.
  See <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/parameters/OmniSurface_Transmission.html>.
- The documented path-tracing settings are `/rtx/pathtracing/maxBounces` and
  `/rtx/pathtracing/maxSpecularAndTransmissionBounces`.
  See <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html>.
- PhysX particles are GPU-only PBD. Particle sets in different systems do not collide.
  Isosurface extraction is render-only and takes a render material on the particle system.
  The particle schemas are not final.
  See <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/particles/particles.html>.
- Isaac Sim 6.0 documents a particle-system wrapper and isosurface support, but still requires
  GPU simulation. Both Isaac Sim 6.0 and 6.2 document a possible isosurface memory leak leading
  to OOM. These pages support a bounded probe, not a claim that the retained Brev installation works.
  See <https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/deprecated/isaacsim.core.prims/docs/index.html>
  and <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/physics/physics_resources.html>.

Observed from the downloaded twin and local installation:

- `fill_station.usda` declares `metersPerUnit = 1`. At time 100 the vial-liquid bounds are
  about 26.4 mm wide and the fixed stream is about 3.0 mm wide. The current source material is
  flat `UsdPreviewSurface` diffuse colour plus opacity.
- The source stream is visible only at time codes 12--367 and 586--660. It therefore represents
  two pours, not the required three; time code 342 is inside a pour.
- The local Isaac Sim 6.2 package contains `PhysxParticleSystem`,
  `PhysxParticleIsosurfaceAPI`, demos and tests on aarch64. Package presence is not runtime
  compatibility evidence. Brev Isaac Sim 6.0.1 remains unverified for this feature.

## Gate O1: transmissive optics

Hypothesis: one shared, finite-volume OmniSurface material makes the approximately 3 mm stream
look paler than the approximately 26 mm vial column while retaining one hue family, without the
flat wall/rail split caused by opacity blending.

The first optical smoke uses the existing geometry only to isolate the material:

- one material and one shader bound to both the vial-liquid and stream meshes;
- `OmniSurface.mdl` / `OmniSurface`, transmission enabled, weight `1.0`, thin-walled `false`,
  diffuse weight `0.0`, IOR `1.333`, roughness `0.08`, `enable_opacity=false`;
- uncalibrated smoke colour `(0.18, 0.55, 0.75)` and scattering depth `0.03 m`;
- FillCam, 640x480, time code 100, PathTracing, 64 subframes, warm-up 60, seed 20261025;
- max total bounces `16` and max specular/transmission bounces `16`;
- method and the unchanged source-material control use identical camera/render settings on local
  Isaac Sim 6.2. A Brev render is unnecessary for this gate unless local rendering fails.

The `0.03 m` depth is a round uncalibrated smoke parameter near the observed vial width, not a
fit or measurement of a real liquid. It is interpreted as 0.03 m only after the layer inspection
proves `metersPerUnit=1`. Disabling cutout opacity must not be implemented as
`geometry_opacity=0`, which would hide the surface. The control and method each permit at most the existing one
Isaac 6.2 infrastructure relaunch. Every process attempt is retained.

Falsifiers and stop conditions:

- either composed mesh does not bind to the exact same inspected OmniSurface material;
- the rendered liquid or stream is black, missing, constant, or lacks the aligned liquid mask;
- the stream is not visibly paler than the vial in the same frame and hue family;
- the vial remains a flat two-tone wall/rail composite rather than a transmitted-light volume;
- source bytes drift, or the requested bounce settings are not present in the receipt.

The top-level visual decision uses a contact-sheet crop. A single frozen, qualitative comparison
against calibration-only real video 1 fill-phase stills may describe obvious geometry/appearance
gaps; it does not tune the material and does not score sim-to-real. Real videos 2--5 remain outside
this optical smoke, with video 5 still reserved for the final real check.

## Gate F1: Brev particle compatibility

Hypothesis: the retained Brev L40S with Isaac Sim 6.0.1 can run one GPU particle system, move a
fluid particle set under gravity, collide it with simple GPU-compatible primitives, extract an
isosurface and bind one render material.

One bounded probe starts from a new meter-scale stage and records Isaac, Kit, PhysX and extension
versions; CUDA/GPU-dynamics settings; schema/API availability; particle counts; finite positions
before and after stepping; movement under gravity; isosurface/API presence; material binding;
runtime; logs; command; commit/diff; seed; and output hashes. The probe uses one particle system.
It has one application attempt and a 30-minute bound after Kit startup; a startup infrastructure
failure is recorded and stops the probe rather than changing settings. No CPU fallback, second
particle system, Isaac upgrade or hidden feature substitution is allowed.

The gate fails with `unsupported` or `insufficient_evidence` if the feature/extension is missing,
GPU simulation cannot be proven active, positions do not change finitely, isosurface extraction
does not materialize, the process crashes, or the result cannot be receipted. A failure blocks the
three-pour build and is reported as a blocker.

The isosurface remains enabled for at most 30 minutes in F1. An outer monitor samples L40S memory;
the probe is terminated and retained as failed before the next step if used GPU memory reaches
40,000 MiB. This is a safety stop for the documented leak, not an isosurface correctness metric.

### Recorded O1 and F1 results

O1 at scattering depth `0.03 m` failed its visual gate because the stream was not visibly
present. The single preregistered O1b follow-up changed only scattering depth to `0.01 m`; it
rendered a faint cyan stream and a darker teal vial without the old flat-opacity two-tone and was
visually accepted as a synthetic optics smoke. It remains uncalibrated and has an obvious
qualitative gap from the much stronger stream and taller coloured column in the calibration-only
real still.

F1 passed `2/2` under Liam's evidence standard adopted after the run. An SSH disconnect caused the
long `brev exec` command to be replayed, so two sequential Isaac processes completed with identical
settings; both count. Each process observed GPU dynamics, finite downward particle motion,
containment, exactly one particle system, no MassAPI, material bindings, and a nonempty
514-vertex/1024-face isosurface. Both programmed receipts emitted `status=insufficient_evidence`
only because their PhysX metadata field was null; both saved Kit logs register
`omni.physx-110.1.13`, which supplies that bookkeeping value. The receipt-capture code now falls
back to the enabled versioned extension ID. The earlier protocol-only decision that the replay and
null metadata blocked F2 is retained in Git history but superseded by the operator's evidence
standard: substantive observed checks and saved artifacts decide the gate, infrastructure repeats
expand rather than invalidate the denominator, and recorded deviations do not block progress.
F2 is therefore authorized.

The ignored evidence record is `outputs/20261025-brev-physx-f1/result.json`. Future remote work
must launch a guarded detached process once and poll its files separately; a long-running command
must not be entrusted to an automatically replayed SSH session. Each concurrent workload must
have its own PID, process group, log, output directory and completion sentinel.

## Gate F2: one-system three-pour simulation

Only after F1 passes, build one new synthetic run with programmed order
`T-200 -> T-400 -> T-100` and volumes `10/10/40 mL`. All emitted particles belong to one
particle system. A provisional `1.5 mm` fluid rest offset implies a nominal particle volume
`8*r^3 = 0.027 mL` under NVIDIA's documented density-based particle-mass convention, giving
nearest-integer targets `370/370/1481` particles (59.967 mL nominal mass-convention equivalent).
The receipt proves that no higher-precedence particle Mass-component density or mass override is
present. It reports programmed volume, the nominal equivalent and error; it never calls this
physically displaced or exact dispensed volume. Fluid rest offset is not treated as a particle radius.

The F2 renderer is the existing twin, not a reconstructed scene. It opens the unchanged canonical
`stations/fill_station/fill_station.usda`, retains its lights, active-vial glass, holder,
background, materials and authored FillCam, and authors the temporary particle experiment only in
the session layer under `/FillStation/FluidmixF2`. The source `Fill_liquid` and
`Fill_dispensing_stream` roots are hidden by stronger session-layer visibility opinions. No
`/World` scene, replacement camera, light, backdrop or visible guide geometry is allowed. Physics
uses Z-up gravity, invisible primitive colliders fitted to the measured active-vial interior, and
the measured nozzle exit as its emission origin. Liquid render materials use the accepted O1b
OmniSurface parameter family: scattering depth `0.01 m`, non-thin-walled transmission weight `1`,
diffuse weight `0`, IOR `1.333`, roughness `0.08`, and cutout opacity disabled. T-400 retains the
accepted uncalibrated blue transmission colour `(0.18, 0.55, 0.75)`; T-200 orange and T-100
near-clear remain nominal uncalibrated render parameters. Organizer labels and existing real
frames support the orange/blue/clear identities, but the frames do not isolate every pure stream
and do not calibrate numeric colour.

Before emitting particles, the same Isaac process captures an unmodified twin reference and the
post-authoring empty F2 scene at source time code `1` through
`/FillStation/FillStation_LocalRoot/FillCam/FillCam`. Both use PathTracing at 640x480, warm-up 60,
64 subframes, seed 20261025, and 16 total and transmission/specular bounces. The whole-frame parity
gate, frozen before the run, requires mean absolute RGB difference at most `2.0`, 95th-percentile
absolute difference at most `8.0`, and at most `1%` of channel values differing by more than `16`.
Both frames, an absolute-difference image, metrics and hashes are required artifacts. Passing this
gate shows only that the non-liquid scene was preserved within path-tracing noise.

The first build uses only box, sphere or capsule GPU-native collision primitives at the measured
active-vial/nozzle location. A convex hull or SDF is allowed only after a separate compatibility
receipt. Cylinders, cones and the decorative vial mesh are not allowed as particle colliders. The full run
records emission windows, physics step size, seed, particle counts by pour, escaped particles,
GPU memory, render representation and stage-frame/video hashes.

The timing contract follows the measured real fill instead of waiting for synthetic settling. A
regression over the ten video-5 recipe-median fill durations models total duration as `8.16571903 s`
per active tank plus `0.07633141*t100_ml + 0.35935439*t200_ml + 0.19177068*t400_ml`. Its
in-sample R2 is `0.92595`, RMSE `2.371 s`, and leave-one-recipe-out RMSE `4.765 s`; this is a
real-like approximation, not measured tank-boundary timing. For F2's 10/10/40 mL recipe, T-200,
T-400 and T-100 run back to back for 706, 605 and 673 steps at 60 Hz. There are no inter-pour or
settling steps. Intermediate stage captures occur at exact pour-end steps 706 and 1311 while the
liquid may still move. Fill ends at step 1984; the primary final capture is step 2002, exactly
`+0.300 s`, which lies inside all 30 measured video-5 pre-cart windows. Optional `+1 s` and `+2 s`
diagnostics are not primary evidence because only 25/30 and 18/30 real windows extend that far.

F2 first runs one bounded palette feasibility smoke in the canonical twin. Official Omni Physics
supports a fluid particle system represented by `UsdGeom.PointInstancer`, and OpenUSD supports
per-instance prototype selection. The exact Isaac 6.0.1 package supplies
`add_physx_particleset_pointinstancer`, but RTX material switching through changing
`protoIndices` has not yet been observed. The smoke uses 12 particles and three transmissive sphere
prototypes, captures once, swaps orange/blue indices without advancing physics, and captures again.
It passes only if finite motion, three visibly distinct materials and the colour swap are all saved
without position or count corruption.

If that smoke passes, each physical particle begins with a one-hot T-200/T-400/T-100 composition.
PhysX PBD supplies advection. A conservative radius-neighbour exchange updates composition every
physics step with frozen effective diffusivity `1e-5 m2/s`; conservation, non-negativity and row
sums of one are substantive checks. A finite, frozen composition palette selects each particle's
transmissive sphere prototype. This is composition-aware but bead-like and quantized. A smooth
isosurface may be retained as hidden geometry evidence because its one system-level material cannot
display particle composition. PBD is not validated Navier-Stokes and the diffusivity is an
uncalibrated subgrid parameter, so motion and partial layering are physically motivated synthetic
behavior, not measured real mixing. If the smoke fails or the beads are visually unusable, F2
falls back to separately coloured stream rendering plus an explicit global vial blend and states
plainly that spatial colour mixing was not rendered; it must not claim particle-resolved mixing.

### Recorded palette feasibility result

The one frozen Brev application failed (`0/1`, exit `2`) and is retained under
`outputs/20261025-brev-physx-palette-smoke-a1/`. Isaac Sim 6.0.1 on the L40S authored GPU
dynamics, exactly one particle system, 12 finite particles, three sphere prototypes and three
bound transmissive materials. The requested 30 physics steps nevertheless produced maximum
movement `0.0 m`, and changing the PointInstancer `protoIndices` did not persist: both the readback
indices and complete before/after RGB files remained identical. The top-level contact-pair review
could see orange, blue and pale beads but no swap. Receipt SHA-256 is
`c34da1875d098cc791ce2237d40cf4b32fcac14acab98f247dc32c98b9993e17`; the judged contact pair is
`f997991cc6d0291ece3095ac3eba8f2e42cb45d72c3dad95578280c103f8a29d`.

This falsifies the conditional PointInstancer palette route as implemented; it is not silently
repaired or counted as a pass. F2 therefore uses the preregistered fallback: the one-system PBD
particles and isosurface provide geometry/motion, a render-only stream carries the active source
tank's uncalibrated transmissive material during its exact pour window, and the vial isosurface
uses an explicit cumulative global blend updated while fluid enters. That representation cannot
show spatial composition, diffusion or the real final-window layering; it may support a synthetic
global-colour baseline only and makes no physical-mixing or sim-to-real claim.

Fallback acceptance requires all three programmed pours in order and without settling gaps, one
interacting particle system, bounded nominal volume error, exact particle-count conservation,
finite positive PBD motion and containment at every stage capture, a non-empty visible isosurface,
a decodable full-sequence video, and stage frames that are subsets of that sequence. It also
requires the active tank's render-only stream for every step in that tank's pour, all 332 cumulative
global-colour updates during emission, three distinct hashed global-state artifacts, and the exact
336-frame 10 Hz timeline through step 2002. The PointInstancer palette failure hashes remain bound
into the receipt so the fallback cannot silently become a particle-mixing claim.

Every kept RGB must have a readable, aligned, non-empty `liquid` semantic mask at 640x480. The
generated isosurface must inherit the `liquid` label and the render-only streams must carry that
label directly; unsupported generated-isosurface semantics is a substantive failure. A top-level
visual gate judges saved stage/video crops for visible fluid geometry, correct source-stream colour
and usable alignment. The receipt must retain false fields for particle composition, diffusion,
spatial mixing/layering, physical validation and sim-to-real transfer. Distinct frame hashes and
metadata alone do not establish visible or physical mixing. Any failed denominator stays failed;
no row replacement is allowed.

### Recorded fallback A1 result

The first schema-4 fallback application is retained as a real failure (`0/1`). Canonical scene and
FillCam parity passed (mean absolute RGB `0.0973`, p95 `0`, fraction above 16 `0.000999`), but the
first T-200 activation failed before physics: the requested stream state was `inherited` while
`ComputeVisibility()` still returned `invisible`. It emitted zero particles and kept zero sequence
frames. Receipt SHA-256 is
`828a48860e5f7fe7d8a94f1d93bc37a4509a0a4494d6a77dc2a112c0206d2c27`; the copied transport
archive SHA-256 is `8d43be77e63135b7da6d23a9b6700a0af9398447d8cec5babe0c41588868f91e`.

The supported software cause is USD edit-target strength. Stream visibility, particle-enable and
material values were initially authored in the session layer, but live updates occurred after the
session edit context exited. A reduced OpenUSD probe reproduced that an `inherited` opinion in the
root layer cannot override `invisible` in the stronger session layer; setting the same value while
targeting the session layer changed computed visibility to `inherited`. The successor source wraps
all three live mutation classes—stream state, particle/material updates and physics writeback—in
that session edit target. Unit regression coverage passes, but renderer correctness remains
unproven until a new, separately identified application runs.

Three completed custom-scene applications preceded this twin integration and remain failed in the
denominator (`0/3`): A1 and A2 returned only constant RGB, and A3 saved a full-size uniform RGB 228
diagnostic. The supported cause was the custom camera's unauthored USD default 1 m near plane while
all custom geometry was only about 0.14--0.27 m away. A fourth custom-scene package was staged but
its guarded application was never launched because the twin-scene requirement superseded it; it is
an infrastructure cancellation, not a completed application. The twin-based run is a new design,
not a replacement for those outcomes. Its authored FillCam clipping range is consumed and attested,
never modified as a workaround.

F2 is one process with a 90-minute wall bound and at most 45 minutes of enabled isosurface time.
The same 40,000 MiB hard memory stop applies. In addition, stop after a timed stage capture if used
memory rises by at least 4,096 MiB from that stage's pre-pour sample and has not fallen. Preserve
that run as a memory-stop failure; do not shorten or skip the stage.

## Colour and mixing limitation

PhysX supplies particle motion, not dye concentration. The attempted project-side
composition/diffusion palette was not accepted because the frozen renderer smoke failed. The
implemented fallback therefore has no particle composition or diffusion state: its smooth
isosurface has one cumulative colour everywhere in the vial, while a separate render-only mesh
shows the active source stream. There is no absorbance fit and all tank transmission colours remain
uncalibrated. Consequently this version cannot reproduce or evaluate spatial layering after the
final T-100 addition. Calibration-only real frames may be used for a qualitative visual sanity
check, but not to tune on the real test split or claim physical mixing accuracy.

### Read-only real-stream visibility result

Measured after the freeze: a distinct coloured stream was visible in the top-ranked, visually
reviewed fill-phase candidate from each of videos 1--5 (`5/5`). Approximate median widths of the
thresholded saturated core in those five selected frames were `8, 10, 9, 4, 8` pixels. Frames were
sampled every 0.5 seconds, ranked in one fixed nozzle corridor, and the top candidate from each
video was reviewed. This selection favours coloured fluid: it does not establish that clear streams
are visible. The changing white-wall/transparent-vial background and 4--10 pixel width also make an
uncorrected stream-versus-vial colour comparison indefensible. This audit created no calibration
sample, label, defect evidence or sim-to-real score. The ignored evidence record is
`outputs/20261025-real-stream-visibility/review.json` (SHA-256
`b7823b2479a04bf601e671411e37d6344c776f81b2e62eb344eeb45cc21676d3`).

## Brev lifecycle

Use only retained stoppable instance `claude-isaac-smoke-1024`; never delete it. The cumulative
cap remains assumed under Liam's explicit directive. The original freeze prohibited overlapping
paid instances; Liam subsequently and explicitly superseded that restriction. Overlap is allowed
only on separate instances and in separate process trees, with independent PIDs, process groups,
logs, output directories and completion sentinels. Do not attach to, stop or modify a teammate's
instance or process. Stop this retained instance after each bounded probe/run and verify exact
`STOPPED` in `brev ls`. Record every start, attempt, result and stop in the ignored compute ledger.

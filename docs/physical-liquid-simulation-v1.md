# Physical liquid simulation v1

Status: **frozen compatibility and optics design; not yet executed; not training eligible**.
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

## Gate F2: one-system three-pour simulation

Only after F1 passes, build one new synthetic run with programmed order
`T-200 -> T-400 -> T-100` and volumes `10/10/40 mL`. All emitted particles belong to one
particle system. A provisional `1.5 mm` fluid rest offset implies a nominal particle volume
`8*r^3 = 0.027 mL` under NVIDIA's documented density-based particle-mass convention, giving
nearest-integer targets `370/370/1481` particles (59.967 mL nominal mass-convention equivalent).
The receipt proves that no higher-precedence particle Mass-component density or mass override is
present. It reports programmed volume, the nominal equivalent and error; it never calls this
physically displaced or exact dispensed volume. Fluid rest offset is not treated as a particle radius.

The first build uses only box, sphere or capsule GPU-native collision primitives at the measured
active-vial/nozzle location. A convex hull or SDF is allowed only after a separate compatibility
receipt. Cylinders, cones and the decorative vial mesh are not allowed as particle colliders. The full run
records emission windows, physics step size, seed, particle counts by pour, escaped particles,
settling windows, GPU memory, isosurface status and stage-frame/video hashes. Stage frames are the
first stable frames after each pour, not fixed source time codes 171/342/686.

Acceptance requires all three programmed pours in order, one system, bounded volume error,
particles retained in the vial at every settled stage, a nonempty isosurface, a decodable full
sequence video, and stage frames that are subsets of that sequence. Any failed denominator stays
failed; no row replacement is allowed.

F2 is one process with a 90-minute wall bound and at most 45 minutes of enabled isosurface time.
The same 40,000 MiB hard memory stop applies. In addition, stop after a stage if used memory rises
by at least 4,096 MiB from that stage's pre-pour sample and does not fall during its complete
settling window. Preserve that run as a memory-stop failure; do not shorten or skip the stage.

## Colour and mixing limitation

PhysX supplies motion, not dye concentration. V1 adopts only the allowed documented assumption:
after each pour settles, the whole isosurface is treated as fully mixed and receives one manually
authored, uncalibrated stage transmission colour. During a pour, colour is visualization-only and
must not be interpreted as local concentration or a tank reference. No absorbance arithmetic or
per-particle colour is introduced. If one particle-system material cannot express the required
stage changes, the run may remain neutral-colour geometry evidence; it must not fake independent
liquid colours.

Later work may add a concentration field and use visible streams as tank references. That is not
part of this freeze. The current real-video task is only to record whether a distinct stream is
visible in videos 1--5 and, when visible, its approximate pixel width and background limitations.

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

Use only retained stoppable instance `claude-isaac-smoke-1024`; never delete it. Before start,
verify no other paid instance is active. The cumulative cap remains assumed under Liam's explicit
directive. Stop after each bounded probe/run and verify exact `STOPPED` in `brev ls`. Never run a
second paid instance concurrently. Record every start, attempt, result and stop in the ignored
compute ledger.

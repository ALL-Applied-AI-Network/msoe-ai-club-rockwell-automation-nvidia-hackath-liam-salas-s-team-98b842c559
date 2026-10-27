# F2 programmed-volume acceptance gate

Status: frozen 2026-10-25 before inspecting the A27 target-stage image or mask. This
gate checks the synthetic simulator only. Passing it would not show that the flow or
appearance matches the real line.

## Geometry and programmed targets

The measurements below come from `fill_station.usda` at SHA-256
`5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9`
and its `geometry/fill_station.usdc` sublayer at SHA-256
`57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db`.
They were read with `pxr.Usd.Stage.Open` from the Isaac Sim 6.2 environment; no
SimulationApp was launched.

The authored straight glass wall has a 26.3652 mm inner diameter. The actual particle
containment boundary is the intersection of 12 collider half-spaces: a regular
dodecagon with 13.2029216 mm circumradius, 12.7530430 mm apothem and
522.9514163 mm2 area. The volume-height gate uses that dodecagon because it bounds
the simulated particles; the glass circle is 4.4% larger. The target heights are:

| Cumulative volume | Expected count | Target height |
|---:|---:|---:|
| 10 mL | 370 | 19.122235 mm |
| 20 mL | 740 | 38.244470 mm |
| 60 mL | 2,221 | 114.733411 mm |

The floor collider top is z=0.9762065645 m, exactly the authored source-liquid bottom.
It lies within the authored glass base rings. Source-liquid masks end within 1.4 pixels
of the projected front floor rim, so no additional opaque-base cutoff excuses a short
visible column.

## Required evidence and decisions

Each stream-hidden stage must save the emitted and active counts, exact-dodecagon
inside count, in-flight-above-mouth count, outside/escaped count, finite-state flag,
contained-particle z minimum, Q05, Q95 and maximum, and PointInstancer scale minimum
and maximum. A stage fails if its count differs from the programmed cumulative count,
any value is non-finite, or any particle is outside/escaped.

The particle bulk height is

`2 * 1.5 mm + (Q95(z_inside) - Q05(z_inside)) / 0.90`.

This trims the narrow final falling tail and removes the corresponding uniform-column
quantile bias. Raw minimum and maximum remain reported and are never substituted for
the bulk surface. Particle height passes within +/-6 mm of the programmed target. The
tolerance is two authored particle diameters and is fixed for all three stages.

For the visible gate, project the exact floor and mouth dodecagons through the authored
FillCam and rasterize their convex hull at 640x480. For every row, count liquid-semantic
pixels inside that geometry ROI. A row is dense when it covers at least one third of
the projected vial width. The surface is the first of two consecutive dense rows. This
rejects a narrow falling stream without requiring connectivity, because the fixed
holder can split a valid tall liquid mask. Invert the dodecagon top-contour projection
to metres and require the same +/-6 mm tolerance. A missing dense surface fails.

The RGB judgment uses the geometry-only overlay and crop `xyxy=(278,208,342,424)`.
It must show a visible bulk liquid column reaching the tolerance band; a semantic mask
over an optically faint smear does not pass. Save that crop, its source image/mask
hashes and the numeric gate JSON. One top-level visual judgment is sufficient unless
the image is ambiguous.

The sphere-instanced representation remains approximate. Its 1.5 mm sphere occupies
14.137 mm3 versus the 27 mm3 nominal spacing cell before smoothing and anisotropy.
Therefore state height and semantic/RGB visible height are separate required checks.

## Particle-count basis and A27 result

NVIDIA documents that a fluid particle's computed mass is
`8 * fluidRestOffset^3 * density`, and separately states that Fluid Rest Offset is not
a render radius. It also documents smoothing and anisotropy as render-only
post-processing that does not change particle dynamics. See the
[Omni Physics particle guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/particles/particles.html#particle-set-mass-computation).
At 1.5 mm and 1,000 kg/m3, 370 particles represent 9.99 mL under that documented
mass convention. The installed NVIDIA Paint Ball demo also uses a sphere prototype
radius equal to Fluid Rest Offset with unit initial scales. Therefore A27 does not
support changing the programmed particle count or prototype radius merely to make the
image taller.

Measured after the gate above was frozen, A27 emitted and retained 370 particles with
zero particles outside the older conservative inscribed-circle check. Its stream-hidden
semantic mask recovered a visible height of only 8.241994 mm, versus 19.122235 +/-6 mm,
and the single top-level RGB review showed a shallow bead/smear. A27 did not save the
exact dodecagon z quantiles or anisotropic scale range, so it cannot determine whether
the short result is in simulation state, render post-processing, or both. The next
diagnostic keeps count, radius, physics, material, timing, seed and render settings fixed
and adds those readbacks. A later material or rendering ablation is conditional on that
separation; A27 itself is rejected as a stage and training sample.

## A28 diagnostic result

A28 completed all 1,412 callbacks on local Isaac Sim 6.2 and retained exactly 370/370
particles inside the exact dodecagon, with none in flight, outside or escaped. Its raw
particle z min/Q05/Q95/max was 0.9790654/0.9790809/0.9895221/1.1730888 m. The frozen
bulk-height statistic was 14.6014 mm, so the state gate passed the 13.1222 mm lower
tolerance bound.

The visible gate still failed. Smoothing/anisotropy produced a 13.0196 mm quantile-based
render height, with PointInstancer scale components spanning 0.8000–1.9235, but the broad
semantic surface measured only 9.0743 mm. One top-level RGB review showed the same shallow
bottom cluster below the lower-tolerance contour. The floor top remained exactly at the
authored source-liquid bottom, so the opaque base does not explain the short column. A28 is
rejected `0/1` and is not training or benchmark data.

The prototype sphere radius was 1.5 mm, equal to the fluid rest offset. A sphere whose
volume equals the documented 3 mm particle-spacing cell would have radius 1.8611 mm, only
about 0.62 FillCam pixel larger. An offline mask-dilation diagnostic found that a one-pixel
increase still failed, while the first pass required four pixels and then jumped above the
target. This is not a rendered ablation, but it does not support a radius-only change as the
next experiment. An optical material change also cannot explain a semantic-mask failure.

A28's PBD material readback had density 1,000 kg/m3 and friction 0.1, but cohesion,
surface tension and viscosity were zero and unauthored. The next bounded experiment therefore
tests NVIDIA's named water PBD preset as a coherent-free-surface treatment. This is an explicit
new hypothesis: A28's state height passed, so the water treatment is not being presented as a
remedy for a failed state quantile. Count, radius, solver, timing, scene, renderer and gates stay
fixed. Passing would qualify the synthetic renderer only, not its physical accuracy.

## A29 water-treatment result and render fallback

A29 applied NVIDIA's named metre-scale water PBD preset and verified every requested
readback. It again retained 370/370 particles inside the exact dodecagon with no leakage.
The state-height statistic passed at 14.2325 mm, but the smoothed-render statistic fell to
12.5113 mm and the broad semantic surface fell to 8.2420 mm. The one-look RGB crop still
shows a shallow bottom cluster. Relative to A28, visible height decreased by 0.8323 mm.
The water treatment therefore fails `0/1` and does not authorize a full video or case cohort.

This result satisfies the previously documented fallback condition: the PointInstancer
spheres preserve the PBD state and per-particle composition, but do not render an acceptable
continuous bulk volume. The next bounded route keeps the same one-system PBD simulation,
particle identity/composition/diffusion, containment and numeric state-height gate, while
driving a copy of the twin's authored liquid mesh from the live post-callback particle rows
actually contained in the vial. The source mesh is already fitted to the same vial: its world
diameter is 26.4058 mm, its bottom is z=0.9762066 m, and its authored late-fill shape reaches
about 122.9 mm. A geometry audit found a 543.2554 mm2 effective mesh area versus the frozen
522.9514 mm2 dodecagon: retaining the original XY size and applying dodecagon height would
overstate volume by 3.883%. The bounded fallback must therefore apply the frozen 0.981134743
XY scale (or use the measured mesh area) and save its post-transform closed-volume residual.
It scales z about the fixed bottom only after a physics callback, excluding falling/outside
particles. This avoids inventing a new camera, vial, holder, light or backdrop.

The continuous body uses the existing transmissive global-blend material, while the falling
stream keeps the current tank-specific material. This is the explicitly allowed blending
fallback, not rendered spatial concentration: PBD still advects and diffuses composition for
quantitative provenance, but the body colour is a global uncalibrated blend and cannot show
particle-scale layering. Its visible-height gate remains required even though body height is
contained-count-driven, and its PBD state-height gate remains independent. The copied body's
points, extent, visibility and material state join the paired-capture hash, and a body/stream-
hidden semantic diagnostic must show that hidden sphere instances contribute no pixels.
Passing would establish only that the synthetic generator preserves programmed volume and its
stated render contract.

## Count-driven body qualification results

### A31: substantive empty-state failure

Measured run `20261025-local62-physx-f2-count-body-qual-a31` failed `0/1`.
Isaac Sim 6.2 started normally and exact pre-pour parity passed, but the first
pre-emission positions readback was a one-dimensional empty sequence. The new body
path required shape `(N,3)` and raised `RuntimeError: count-driven body received
invalid PBD positions` before any callback, physics step or particle emission. This
was a deterministic implementation failure, not an infrastructure retry. No
containment, height, semantic or liquid-appearance gate ran. The empty-state fix is
commit `ef24128441dd2f1bebd7b6278686164c48563c58`.

| A31 artifact | SHA-256 |
|---|---|
| frozen experiment | `838f381153bd6cbb70f0c5e71b3de528257967e9c28753c06ad00097ae13c6c3` |
| `execution-summary.json` | `36b81a611f452e6ed20b8cafc2a74c720378776d96df7be3089e42b8b2f104be` |
| `completion.json` | `3743117dc46783e27bae73a5c62a1fc7a7ca61660d280bbb7d8cebbcd6dc2432` |
| `app-output/receipt.json` | `edefb2296175548943d938c3695eb7d56f09d4ddd962bcb4781b44dc293935f9` |
| `isaac.log` | `0e26700bc6a64137d9f162e795093bff39bf2355331d3425fa7ea7f3f88f2753` |

### A32: accepted first-stage gate

Measured run `20261025-local62-physx-f2-count-body-emptyfix-qual-a32`, using runtime
commit `ef24128441dd2f1bebd7b6278686164c48563c58`, completed all 1,412 callbacks.
Under Liam's evidence standard it passes `1/1`: every substantive first-stage check
was observed and has a saved artifact. All 370 emitted particles were active and
inside the exact dodecagon; zero were in flight, outside or escaped, and all state was
finite. The Q05/Q95 particle bulk height was 14.37794 mm against the frozen
19.122235 +/-6 mm target.

The count-driven body represented 9.990000 mL. Its signed volume was
9.989999913 mL, an absolute residual of `8.716841640589012e-14 m3` against the
`2e-10 m3` limit. Its authored world height was 19.103113 mm from bottom
z=0.9762065645 m. The body-and-stream-hidden control contained zero liquid pixels;
restoring the body alone produced 2,179. The measured semantic height was
18.110658 mm, 1.0116 mm from the programmed target, so the visible gate passed.
Pre-pour parity, source/body/material and paired RGB/semantic/state hashes passed,
and the three-frame qualification AVI decoded `3/3`.

The required single top-level volume review passed: the saved crop shows a readable
orange bulk column above the lower-tolerance contour rather than the earlier shallow
smear. The crop hash is
`729d6f1fe688be148ebe0a0069c5718eccf099dc5f3ec16ee209e96db244dd62`;
the judgment hash is
`89e80941084479cc825ea92cde17c0aef26a583ebd383b7c5d0a34b8fcba58c1`.
The separate source-stream review also passed: step 706 shows only a continuous,
thin pale-orange T-200 stream entering the orange body. Its frame hash is
`18c26028cc94acbff71e9077ec1f50de740d1e5c7d13cc51dc657e4be1d5597a`
and judgment hash is
`4e054778d887ae7600f6c1b717b5272240c1bb5918e4388ed295cba4a71ad9e9`.

The raw runner result remains reported separately: exit code `2`, completion
`failed`, receipt `insufficient_evidence`, and validator failure
`count_driven_body_contract`. Its two predicates were bookkeeping defects, not
failed substantive checks:

1. The validator expected copied `displayColor` and `displayOpacity` values, but the
   frozen source contains unvalued declarations, so `copied_primvar_names=[]` is the
   exact source-preserving result.
2. The stage-row serializer omitted `volume_proof_coordinate_path`; the root body
   already recorded `local_points_vec3f_then_inherited_world_transform` and all
   numeric geometry evidence.

The original receipt was not edited. Revalidation after filling only that metadata
string in memory and correcting the exact-source predicate produced no failures.
Commit `83169d8439ef2acb16f6ed0aacea906e2bc49522` fixes future capture and validation.

| A32 artifact | SHA-256 |
|---|---|
| frozen experiment | `070dfcdf6a58978d517ff66a733c9b1f9b478941c0ad7429c58a69a75faa131a` |
| runtime source archive | `7ce698ae7d51e1b0ac7e67a5b888b2f8536ee50a0849adac67e04b848a6b3c94` |
| `result-record.json` | `bd9a299abef9fca3dd80998b19d1ab37a31d5b56dbeb1117889673a4616c55d6` |
| original `app-output/receipt.json` | `f5ec92f1b12194e8038ff4e8c0324013ced876b6b452f0bda6a93dc58fdac655` |
| `completion.json` | `1887f6858afe868592aea80ad8b0f398e7f1aa1b5818db457ffc37808dd3f9cb` |
| `isaac.log` | `ad6a26321bdb56f481ea27a9849501b05bb1308908a9728fff28197576cfe5e2` |
| `execution-summary.json` | `d54c7a76a2b89d299b643b43129eb79672edafbb04f0ebadb0f58fba71e7f4f5` |
| offline volume metrics | `eabe94016f5432387a290c0eba61ec254a1a3eafa5f513d1f1cc763de7853080` |
| bookkeeping correction | `e201ed10ee9174d3e3171f1cc7e45825686f51846a82285055cbb5d4c65ced72` |
| post-fix source archive | `03b1f7d02a866bd26a740766c302fa2c71f0c18b36c508eb4ef8967567d03fdd` |

A32 qualifies only the synthetic first-stage renderer and programmed-volume evidence
contract. It does not validate PBD as Navier-Stokes, the diffusion coefficient, tank
colours, spatial mixing or layering, real carryover, real defects or causes, or
sim-to-real transfer. The 20 mL and 60 mL stages still require independent evidence.

## Brev full-run infrastructure blockers

The retained stoppable L40S `claude-isaac-smoke-1024` (`m1vgz4tbs`) was exactly
`STOPPED` before launch. Both bounded, unchanged starts reached `FAILURE`,
`UNHEALTHY`, and `NOT READY`. No files were transferred and no Isaac/F2 application
ran, so the application denominator is zero. It was stopped and independently
verified exactly `STOPPED`; lifecycle-summary SHA-256 is
`41614e3e6e769dde532eb39798764071ac90d924d7253010b00dc57b5d79dc30`.

One fresh stoppable Crusoe L40S instance, `codex-isaac62-a33-1025`
(`mrg9dbd7m`), became ready and installed Torch 2.11 followed by Isaac Sim 6.2. Its
NVIDIA 565.57.01 driver exposed CUDA driver API 12070, which is incompatible with
the installed CUDA 13 Torch build; `torch.cuda.is_available()` was false. No source
or twin was transferred and no SimulationApp/F2 application ran, again leaving a
zero application denominator. The instance was stopped and verified `STOPPED`, not
deleted. Exact hashes are: lifecycle summary
`9d2fb1f394878d67e436d750547b80944ac290845611120c3a179f08d05410ef`,
install failure
`5aacac8f88dedab5fea47bb8bea4e6e6544925afe0f7e0f9056d30dbce898442`,
setup log `3dcc74094cab1819b31c3ca76e6e13b66ed6624ddd971c2c5f7f1cb5c43051e6`,
and final stopped receipt
`23f3f5f0586b5537f5486b449d38d164f5cab40fcbfeffb6ab9ff3150d1e0012`.
These are infrastructure failures only and give no evidence for or against the
three-stage renderer, physics, volume gates or video.

## A34 full-run result: rejected

Measured run `20261025-local62-physx-f2-count-body-full-a34` completed one
substantive application attempt and failed `0/1` (exit `2`, canonical receipt
`insufficient_evidence`). It saved 230 of the required 336 matched 640x480
frames and only the first two of three stage frames. The partial AVI independently
decoded all 230 saved frames at 640x480. Each saved stage PNG is byte-identical to
its video-source frame. The 60 mL stage at step 4,004 is absent.

The exact saved stage results are:

| Result | Stage 1: 10 mL, step 1,412 | Stage 2: 20 mL, step 2,622 |
|---|---:|---:|
| emitted / active / inside | 370 / 370 / 370 | 740 / 740 / 740 |
| in flight / outside / escaped | 0 / 0 / 0 | 0 / 0 / 0 |
| contained z min / Q05 / Q95 / max (m) | 0.9790360331535339 / 0.9790847837924957 / 0.9891894817352295 / 1.173088788986206 | 0.9789977073669434 / 0.979068922996521 / 1.0031452894210815 / 1.173088788986206 |
| frozen particle bulk height (mm) | 14.227442158593024 vs 19.12223523598308 +/-6: **pass** | 29.751518249511617 vs 38.24447047196616 +/-6: **fail** |
| smoothed particle bulk height (mm) | 12.5911588933733 | 27.960842397477988 |
| count-driven body height (mm) | 19.103113206204947 | 38.206224898451824 |
| body-volume residual (m3) | 8.716841640589012e-14 | 1.3434901384749495e-13 |
| visible semantic height (mm) | 18.11065774938775; error 1.011577486595333: **pass** | 37.10130873298978; error 1.1431617389763837: **pass** |

The required one-look top-level review passed stage 1: its crop shows a broad,
contiguous orange column seated at the apparent inner base, not a smear, with the
detected surface inside the frozen band. The stage/frame hash is
`7cd967aaf234f4cdb90244dc50f7b8818c6c57f72a9762f1f3850c25ad6a8d91`,
the judged crop hash is
`a5c50c6642147691de6bb6dc194b2783e74cc69b2863d51ec1c7c787fb2bde85`,
and the visual-judgment hash is
`5fdb83e6f2c641acb36413f0325922c802d329d7a9416902d85f144d3faeacd3`.
Stage 2's frame hash is
`c82d8f4526b803df7a18248f96e4a4f844246532c67e6e29e6e02dc618867bb5`.
Its constructed body and visible-mask checks passed, but they do not override the
independent frozen particle-height failure. No stage-2 visual pass is claimed: the
offline reviewer failed closed because the run did not save a same-step hidden-sphere
control. The run also did not save raw particle positions, velocities and scales, so
the evidence cannot distinguish estimator bias from true PBD compression. The frozen
failure is not rescored with a post-look estimator.

The application stopped at callback 2,746. Callback 2,745 had advanced normally to
22.874999999999343 s; callback 2,746 reported 0.008333333333333333 s instead of
22.883333333333333 s. The supported cause is a timeline wrap: the canonical twin ends
at `686/30 = 22.866666666666667` s, while the runner set neither an extended end nor
non-looping mode and called `timeline.play()`, whose looping argument defaults to true.
The paired capture at step 2,736 preserved callback and time exactly, followed by nine
normal callbacks, which falsifies that capture as the reset trigger. The diagnosis is
bound by SHA-256
`f6b55ef321e823e61fad00aff49fa531e11c5fd03f23ed15aea24f3bddc72a81`;
the relevant behavior is documented in NVIDIA's
[Omni Timeline API](https://docs.omniverse.nvidia.com/kit/docs/omni.timeline/latest/omni.timeline/omni.timeline.Timeline.html).

The canonical receipt remains byte-for-byte unchanged. Because the exception occurred
before success-only pair serialization, a separately labeled **noncanonical** receipt
copy recovered only the hash-bound stage-1/stage-2 paired-file metadata. It did not
change any gate number, status or decision, and cannot rescue A34. Its hash is
`5326d72ebd8c09e53eda0e4ae48c407b2a4fce1255311883304aeaa683cfa7e2`;
the recovery summary hash is
`171423478e9fc102876009f581d16672c0a1db37b97fce00600c47830405ba54`.

| A34 canonical artifact | SHA-256 |
|---|---|
| `completion.json` | `3ab45b028dfdf605aee2e595f4310f7bc4ded1d38cbe08c38b20f2ff33e39fbe` |
| unchanged `app-output/receipt.json` | `f20b9e27c1191480d772944c70e6b49a9a31ba7a8e6634e844a85957ea460446` |
| `app-output/timeline-callback-failure.json` | `c4df4c6e030898526740dc7cebc367f03b3a23216f1dca9b50dfd5a2ba733b11` |
| `isaac.log` | `0fed11df7066c4568b2b4aca3ca5e0a0e556eebdf99a150f432597fdefa8e15f` |
| partial `app-output/three-pour.avi` | `b5704b73609617c9239cf29bc644d7ac78fb06d8a3325e614f2322d347e41fb1` |

A34 therefore provides useful partial synthetic evidence but does not pass the full-run
gate, qualify a three-stage renderer, or authorize training/benchmark cases. It does not
validate PBD as Navier-Stokes, mixing or diffusion, colours, real carryover, defects,
causes, real volume, or sim-to-real transfer.

## A36 timeline-boundary probe: passed

The bounded no-render probe passed `1/1`. With the end set to
`(4004 + 1) / 120 = 33.375` s, looping read back false and playback explicitly started
non-looping, all 2,750 callbacks arrived exactly once per application update. The
measured callback dt was 0.008333333767950535 s, timing was strictly monotonic with a
maximum absolute index/120 error of `6.608047442568932e-13` s, and the former wrap
window continued from 22.858333333332677 s at callback 2,743 through
22.916666666666007 s at callback 2,750. Callback 2,746 was
22.883333333332676 s and did not wrap.

The probe authored no PBD or volume setting and performed no render capture.

Production commit `6e2edc5` applies the explicit timeline range and non-looping
playback; commit `4813146` updates its source-level regression assertion. The probe
receipt hash is
`c4a0ee0d96b03ca9b457f3bd5f5fb34e210371318c604f39eadde81382333588`,
the callback-trace hash is
`27336770e7666e40f2af78d4d4d9d0c7851bd6aef22fce068a4aed925e85b1e9`,
the log hash is
`7814b2d96df66f6e71b7ab68351c5057bf6b730ee31748842f229c44effc7fef`,
and the hash-manifest hash is
`aabea8f189d4ba2c3ba01653cc168f550c73cd65044c86cc83d150dc9cb4c3a5`.
This validates only the timeline-boundary correction. It does not validate a full
render, PBD state or volume/height, the final stage, case data, or sim-to-real transfer.

## A36 particle-array diagnostic: compact PBD bulk observed

A36 completed two application attempts with a cumulative `1/2` result. Attempt 1
started Isaac Sim but failed before its first callback because the diagnostic runner
omitted the authored-value flags required by the existing water-preset validator. It
is retained as a failed execution, not discarded as infrastructure. Its manifest and
receipt hashes are
`1d955dd7b063e82a5e549f3bd73866fb2035e4a3b425ed3b0d5c0b83f7abee7a`
and `705e370eaa672c89edde6af72052cc3f14ad06c0a3e2ee4e4259fc328d523ac4`.

Attempt 2 used the same frozen plan, seed, physics, emission schedule, material,
colliders and timing after adding only those missing readbacks. It passed the
diagnostic contract in 529.96 seconds: all 2,622 callbacks were exact and non-looping,
all 14 NPY arrays were readable, finite and hash-matched, and the run created no render
product, RGB image or video. The exact state was:

| Result | Step 1,412 / 10 mL | Step 2,622 / 20 mL |
|---|---:|---:|
| emitted / inside / outside / escaped | 370 / 370 / 0 / 0 | 740 / 740 / 0 / 0 |
| raw Q05 / Q95 z (m) | 0.979083103 / 0.989217153 | 0.979067913 / 1.003821266 |
| centre-support height (mm) | 11.2601 | 27.5037 |
| centre support + 2.9 mm rigid stand-off (mm) | 14.1601 | 30.4037 |
| raw z maximum (m) | 1.173088789 | 1.173088789 |
| rows in the high tail | 6 | 6 |

The separately frozen offline analysis independently verified every array hash. Median
three-dimensional nearest-neighbour distance was 1.01284 and 1.40487 mm, with 99.7297%
and 99.8649% of rows below 2.7 mm. Median wall clearance was 2.89849 and 2.89764 mm,
matching the configured 2.9 mm particle-to-rigid rest offset. NVIDIA documents that
fluid rest offset contributes to the target density rather than acting as a visible
particle radius, while the PhysX SDK describes a `2 * fluidRestOffset` target fluid
rest distance; see the [Omni Physics particle guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/particles/particles.html)
and [PhysX particle-system guide](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/ParticleSystem.html).
Together with the short centre support, the measured spacing supports a genuinely
compact/overlapping simulated-particle bulk rather than only an incorrect scalar
radius correction. It does not by itself identify why the density constraint produced
that state.

The canonical plan, receipt, manifest, callback trace and log hashes are
`105b6862c1a7bf97398652a4ffc78ca01433aaffb1f6bd3aa2789679623c7d10`,
`008fd317e8022caaefbb0af7874280fa5cfb252c48a1aec3b22efed468dc2a9b`,
`6dca605b4abedff7e8a361a0bbe8e918be644b82003740e8542d8dbeab0c168d`,
`1c4f5baa67ea909a10a823dd1deea2ae566e9a1b7e1ffcb15997f4172d7b9628`
and `96ccd4d81760841f808108d59794d04315dd73ecc4277cbc808aa5814141c330`.
The canonical manifest deliberately excludes the later `offline-analysis/` subtree;
its analysis and independent manifest hashes are
`b327ad5b86f3d31ae2cfe8422be102d58ea7603059a81e844a0e27dace4fc550`
and `ab4c84e19939d2bd664386286c5c40e61d5e58ea171ac94e26c014f358239b6a`.

A36 is a passed diagnostic, not a passed volume stage. It does not rescore A34,
authorize another full render or case cohort, validate the final 60 mL stage, establish
real fluid accuracy, or support any sim-to-real claim. The next bounded cause test uses
the documented per-particle mass formula explicitly during runtime emission; it changes
no render setting and must report a null result if the compact state remains.

## A37 explicit runtime-mass ablation: hypothesis falsified

A37 completed its single planned application attempt. The diagnostic execution passed
`1/1`: Isaac Sim 6.2 ran 2,622/2,622 exact, non-looping callbacks in 530.92 seconds;
all 14 primary NPY arrays and both nearest-neighbour arrays were readable, finite and
hash-matched; and no render product, RGB image or video was created. This execution pass
is separate from the experiment result below.

The run was identical to A36 attempt 2 except for the frozen mass treatment. It set the
documented density-equivalent per-particle mass to 0.000027 kg and, after every runtime
append, authored total `MassAPI` mass as active count times that value. All 219/219
transitions set and read back successfully. The stage readbacks were 0.009990000166 kg
for 370 particles and 0.019980000332 kg for 740 particles. Density remained
1,000 kg/m3.

| Result | Step 1,412 / 10 mL | Step 2,622 / 20 mL |
|---|---:|---:|
| emitted / inside / outside / escaped | 370 / 370 / 0 / 0 | 740 / 740 / 0 / 0 |
| raw Q05 / Q95 z (m) | 0.979082584 / 0.989350942 | 0.979069290 / 1.002800339 |
| centre-support height (mm) | 11.4093 | 26.3678 |
| centre support + 2.9 mm rigid stand-off (mm) | 14.3093 | 29.2678 |
| frozen acceptance band (mm) | 13.1222–25.1222: **pass** | 32.2445–44.2445: **fail** |

The frozen hypothesis is therefore **falsified**: explicit equivalent mass left stage 2
2.9766 mm below its lower bound and 8.9766 mm below its 38.2445 mm target. This was not
caused by leakage, missing identities, a callback drift or a failed mass write. Relative
to A36's 30.4037 mm stage-2 result, the measured support decreased by 1.1359 mm. A37
does not identify the remaining cause and does not justify tuning the estimator or
physics after inspection.

The frozen plan, receipt, artifact manifest, callback trace and launcher-log hashes are:

| A37 artifact | SHA-256 |
|---|---|
| `plan.json` | `556eeb93c8fcc9fab593b4762abc9fffe1f74a306c2d77102383a142d6d6099f` |
| `receipt.json` | `dcd8def930620684909c8b4a2ad74bb7c4c6aedd4824189f5149e7bceefa5501` |
| `artifact-manifest.json` | `e3e419338440879109b3eb111396d7cf2d96c6faa03f0fdb88aedb23e3adc7a5` |
| `callback-trace.json` | `1c4f5baa67ea909a10a823dd1deea2ae566e9a1b7e1ffcb15997f4172d7b9628` |
| `20261025-local62-a37-explicit-mass-attempt1.log` | `f7bc9195aae3cb5c826a2d65e893a120863dd6212fc93d456ad743222cdd4ac6` |

The receipt validator returned no failures, and no Isaac process remained after exit.
A37 does not rescore A34. It is a synthetic, no-render diagnostic and does not validate
physical-fluid accuracy or sim-to-real transfer. The programmed-volume state gate, the
three-stage renderer, and positive/negative case generation remain blocked.

## A39 pre-playback registration control: hypothesis falsified

The first shell launch command failed before Python or Isaac Sim because this fresh
worktree did not yet contain its ignored `outputs/` parent, so log redirection returned
exit 1. It is retained as a launcher setup failure with application denominator zero;
its record hash is
`5e4aafee3de932384ca4063d1fb38a2d5d516c92fc16121c3ba0df5a16da7986`.
Creating only that missing directory did not change any frozen simulation input or
setting.

The single application attempt then passed its diagnostic execution contract `1/1`.
It loaded A37's exact stage-2 compact state before the first post-authoring Kit update
and before timeline play: 740 raw positions, saved velocities, IDs, scales, palette
indices and compositions. The raw-position input hash was
`04b9751932d7ac438d4cbfa7737f21fc31262c929415225fc8ee9e2b7472b97f`;
the velocity input hash was
`e748d22fece8bfc3e7323bd4a9da8db4dfd8bd074039197304a913b28761c6b7`.
The helper received all 740 rows at particle-set creation and no runtime append occurred.
The pre-playback positions were byte-equivalent after float32 USD storage, and the live
readbacks were `fluid=true`, `selfCollision=true`, group 0, the expected particle-system
relationship, global self-collision effectively true, mass 0.019980000332 kg and density
1,000 kg/m3.

All 1,412/1,412 callbacks at 120 Hz were exact and non-looping. The run took 287.46
seconds, created no render product, RGB or video, retained exact IDs, and kept all
740/740 particles inside the vial with zero outside or escaped. The high falling tail
settled from six rows above z=1.0262065645 m to zero, so that expected gravitational
change is reported separately from density expansion.

| Diagnostic | Initial A37 state | After 1,412 callbacks | Frozen pass |
|---|---:|---:|---:|
| wall-to-centre support (mm) | 29.2678 | 31.5397 | 32.2445–44.2445: **fail** |
| median nearest-neighbour distance (mm) | 1.4566 | 1.5866 | >=2.7: **fail** |
| fraction with nearest neighbour below 2.7 mm | 99.7297% | 100.0000% | diagnostic |
| exact duplicate-position pairs | 10 | 10 | 0: **fail** |

The frozen parse-time-registration hypothesis is therefore **falsified**. A nonempty
particle set registered before playback did not restore the documented 3 mm target
spacing or move bulk support into the existing stage-2 band. This result rules out that
specific empty-at-parse explanation under the A39 settings; it does not diagnose the
remaining PhysX density/overlap cause and does not support a post-look estimator change.

| A39 artifact | SHA-256 |
|---|---|
| `plan.json` | `ca1ea3c85dfee21362a15edc6eaded755dcc0cfbf816d91b48e5f78174d098dc` |
| `receipt.json` | `ccbcc5f0455f2c16d3c8b3124993ca85ddbba61840e8e6d7cf5c9b26d057a938` |
| `artifact-manifest.json` | `fce5b89ceeb1d7ab609e03ad4bf68aafaa3c5e1c6ba45bf8157e123d16699760` |
| `callback-trace.json` | `0b0e7cb5c076df891a496955f6d9b99e8b874281e5ec09efc9f741492172efcd` |
| launcher log | `d0806f8de6ab50600a6de96c89c94ef63028801af8f3418c7edd07133dac6540` |

The independent receipt validator returned no failures and all 14 saved snapshot NPY
arrays were readable and finite. A39 is a passed diagnostic execution with a falsified
hypothesis, not a passed volume stage. It does not rescore A34 or A37, qualify the
three-stage renderer, authorize case generation, validate real fluid behavior, or support
sim-to-real transfer. The programmed-volume gate remains blocked.

## A40 documented-offset resolution probe: route exhausted on containment

A40 executed its only frozen wrapper invocation and application attempt, for a cumulative
`0/1` pass result. There was no wrapper or startup failure: Isaac Sim 6.2 reported startup
complete after 5.140 seconds, used GPU PhysX, and shut down cleanly. The application reached
the first frozen capture at step 1,412 after 284.96 seconds, then raised
`stage raw particle containment mismatch`. No Isaac process remained after exit.

The sole intervention was the NVIDIA-documented particle-offset resolution relationship:
fluid rest offset 1.0 mm; particle/contact offset 1.6835016835 mm; rest/solid-rest offset
1.6666666667 mm; 2.0 mm spawn spacing; density 1,000 kg/m3; and per-particle mass 8e-6 kg.
That implies 1,250 and 2,500 particles for the programmed 10 and 20 mL stages. The twin,
colliders, water material, 120 Hz timing, seed 20261025, 10 Hz emission cadence, CCD,
2.5 m/s velocity cap, one shared PointInstancer, and explicit runtime mass readback were
unchanged from A37. Diffusion and all rendering were disabled. The plan was frozen before
launch at commit `b48990b5d2484b7e42f464447da59f31c9761aba`; its SHA-256 is
`5016afa8945023c5d2c5cea7fc733304411d9e9dffcbfbf56e070fe1580f4fc1`.

The receipt preserves 118/118 first-pour mass transitions through step 1,412. Its final
transition reports 1,250 active particles, requested total mass 0.010000000000000002 kg,
read-back total mass 0.009999999776482582 kg, and a matching per-particle readback. The
stage-capture helper had already read arrays and computed containment before raising, so
at least one frozen raw-containment predicate was false. However, the then-current helper
raised before serializing the arrays, exact inside/outside/escaped counts, z range, or
nearest-neighbour diagnostics. `stages`, timing and emission records therefore remain
empty in the failed receipt and the manifest contains only the plan and command. Under the
evidence standard, these missing artifacts are themselves a substantive failure; no exact
containment split, support height, duplicate count, nearest-neighbour distribution, or
stage-2 result may be inferred.

The one-millimetre resolution route is therefore exhausted without tuning or another run.
It does not pass either programmed-volume stage, rescore A34, authorize a render or case
cohort, validate PBD as real fluid, or support sim-to-real claims. A test-first follow-up
changes future failed-stage capture ordering so arrays and a count diagnostic are written
before raising and added to the failure manifest; it does not repair or reinterpret A40.

| A40 artifact | SHA-256 |
|---|---|
| `plan.json` | `5016afa8945023c5d2c5cea7fc733304411d9e9dffcbfbf56e070fe1580f4fc1` |
| `command.json` | `bf46e476a822b2fdf0edf5261ab601c6a9cd64d6aa8609483eedce3455a4518e` |
| `receipt.json` | `e7aa9577022a0abbaffe44c3e30aacd1b26aa7224926eab5cbb7b75547358e88` |
| `artifact-manifest.json` | `8371037613aa3ac339ef2d81cc7672f43bc9de0aec34c1fe1f52b26f8cfb36bd` |
| launcher log | `dfe9f869861681c0d5f9f381ad8a35bb0448f1f1f622210f3ec450598a8746c7` |

## 2026-10-27 operator policy: rendered height is volume-driven

Liam ended further particle-packing tuning for the varied-run training cohort. This is an
explicit scope decision, not a reinterpretation of A34--A40: those particle-height failures
remain measured failures of their frozen physics hypotheses.

For new varied-run V3 images, videos and labels, the count-driven liquid-body height derived
from programmed cumulative volume is the height of record. A finite particle-height miss is
preserved with its measured height, frozen bounds and miss distance as
`diagnostic_warning_not_a_physics_pass`. It is a warning rather than a render-rejection gate.
Zero leakage and containment, readable finite outputs, aligned semantic masks, complete
renders and receipt/log consistency remain hard gates. A case that passes those checks may be
accepted as synthetic training-render data, but it is not a particle-physics pass and provides
no physical-fluid, chemistry, carryover-volume or sim-to-real validation.

Completed cases that failed only the former height gate may be accepted through fresh offline
artifact validation; their original failed attempts and exit evidence are not rewritten or
discarded. The merge records the old case-result, attempt and exit-code hashes together with
the warning. The final varied V3 merge accepted 18/18 render cases with zero hard failures;
15 cases contain 16 particle-height warning rows. The only warning-free cases are train indices
3--5, leaving no warning-free validation or evaluation subset. Merge receipt SHA-256 is
`23afda49f5a90a90ec70c118dd87f2fd46e4c5fddc5e5656927f9ca9ee816f11`.
That acceptance authorizes the bounded synthetic training-render handoff only; it does not
change any earlier physics result.

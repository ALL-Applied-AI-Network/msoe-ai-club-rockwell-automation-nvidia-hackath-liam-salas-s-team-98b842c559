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

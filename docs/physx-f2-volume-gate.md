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

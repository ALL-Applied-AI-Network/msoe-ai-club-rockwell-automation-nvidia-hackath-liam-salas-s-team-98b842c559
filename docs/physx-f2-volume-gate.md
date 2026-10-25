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

# Transport animation sequence and WP1b handoff

This is the durable handoff for INN-54 and later INN-60/WP1b stitching. It records the
approved authored-layer design; it does not claim that the GB10 pilot or visual gate has
passed. The preserved V3/V4 renders and the source twin remain read-only.

## Source motion reused

The reviewed source is
`external/external/csi_digital_twin_v1/stations/fill_station/fill_station.usda`
(SHA-256 `5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9`)
over `geometry/fill_station.usdc`
(SHA-256 `57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db`).
Its queued carrier-2 group and `Fill_vial_neighbour_2` have 37 motion keys at source time
codes 180 through 216. The approved compact JSON digest for the 13 used rows, represented
as `source_time_code` plus `local_y_delta`, is
`c340e573d8b225326176a2cd3523ea2b0c19895c7d1187e0a67954a8a485a317`.
The corresponding 37-key delta table hashes to
`621410b9ece7875fc0a0c7e5ef6feb3e7a34d521a6f6490bbf454f578702e528`.
This corrects the proposal's shorthand description of `c340…` as the 37-step digest:
source re-extraction proves it is the approved 13-row delta digest, while `6214…` is the
37-row delta digest.
The builder also binds richer canonical rows containing absolute XYZ and progress: all 37
hash to `58ac18e079d9c6f99280338046584b21772eea475eac976d9a83d75d15597625`,
while the 13-row absolute-Y/progress table below hashes to
`aed5b56bcd9443684f376db70272a25940b4715051d047b0a98cf64e06b2a613`.
The source-layer hash and exact values are checked together, so no digest is treated as a
self-authenticating substitute for the source bytes.

The renderer samples the authored curve at 10 fps without interpolation, spatial
mirroring, or temporal stretching. Its 13 endpoint-inclusive states span 1.2 seconds.
The canonical JSON of this table uses keys `time_code`, `local_y`, and `progress`, sorted
with compact separators and rows in table order.

| State | Source time | Local Y | Normalized progress |
|---:|---:|---:|---:|
| 0 | 180 | 10.680419921875 | 0 |
| 1 | 183 | 10.601715087890625 | 0.01967620849609375 |
| 2 | 186 | 10.384124755859375 | 0.07407379150390625 |
| 3 | 189 | 10.055419921875 | 0.15625 |
| 4 | 192 | 9.643386840820312 | 0.2592582702636719 |
| 5 | 195 | 9.175796508789062 | 0.3761558532714844 |
| 6 | 198 | 8.680419921875 | 0.5 |
| 7 | 201 | 8.185050964355469 | 0.6238422393798828 |
| 8 | 204 | 7.717460632324219 | 0.7407398223876953 |
| 9 | 207 | 7.305419921875 | 0.84375 |
| 10 | 210 | 6.976722717285156 | 0.9259243011474609 |
| 11 | 213 | 6.759117126464844 | 0.9803256988525391 |
| 12 | 216 | 6.680419921875 | 1 |

Every composed state evaluates the source at time code 1. The sampled curve supplies
only normalized progress. New default-valued state layers scale it along the existing
one-way conveyor direction; the source timeline never advances.

## Thirteen-state layout

Each segment contains `transport-stage.usda`, one receipt-bound
`transport-liquid.usda`, and `states/state-00.usda` through `state-12.usda`.
Every generated layer carries the bounded `ownerThreadId` custom-layer tag supplied to
the builder; the build and render receipts repeat the same value and reject a mismatch.
The common layer is stronger than the immutable source and weaker than a state layer.
A common or state layer preserves the source stage's 30 fps/time-code metadata, so source
time code 1 cannot be reinterpreted through USD's 24 fps root-layer default.
A state layer authors no time samples: only a default
`xformOp:translate:fluidmixTransport`, its complete default `xformOpOrder`, and the
mode-specific visibility values. The legacy animated liquid and dispensing stream are
hidden in the common layer. The track, source layers, accepted fill files, camera, and
lights receive no non-session transport specs.

The exact mover sets are 57 outgoing paths, 27 incoming paths, and 27 queued paths.
`IncomingLabels` contains 29 one-prim references for the active label, top rule, and 27
barcode pieces. `QueueTail` contains 27 one-prim references for the queued vial, carrier,
and fasteners. Treating a group as one reference is forbidden because it loses the
reviewed per-prim identity and material bindings.

All movement follows the same source ease and direction:

| Assembly | Final world-Y move | Purpose |
|---|---:|---|
| outgoing active vial/carrier/labels/liquid | +0.1900000000 m | fully leave frame |
| incoming neighbour 1 | +0.0848413319824885 m | reach the exact fill anchor |
| queued neighbour 2 | +0.186441330754633 m | reach neighbour-1 position |
| new queue tail | +0.1016000000 m | advance by the source's −4 local-Y motion |

The outgoing liquid is reconstructed once from the accepted final snapshot's complete
13-array closure and remains rigid relative to its carrier. Geometry and global blend use
those arrays; the exact accepted final stage-band evidence restores the receipt-bound
`displayColor` bytes when the saved diffusion array is later than the paired RGB capture.
The builder verifies both current and legacy liquid semantic bindings without requiring
SimulationApp. Transport invokes no PhysX step, callback, or simulation-time advance.

## Entry, between, and exit modes

- `entry`: the original active 57-path assembly is invisible in all 13 states. The
  incoming assembly and labels move to the fill anchor, the queued assembly advances,
  and the queue tail advances. The queued vial is hidden at state 0 and inherits source
  visibility from state 1 onward.
- `between`: the complete filled active assembly moves out while incoming, queued, label,
  and tail assemblies advance. The accepted outgoing fill supplies nuisance state and
  the reconstructed outgoing liquid; the next fill supplies the end seam.
- `exit`: only the outgoing assembly moves. No incoming-label or queue-tail namespace is
  authored, and incoming/queued transforms and visibility are untouched.

For a run of `N` accepted fills, the layout is one entry, `N−1` between segments, and one
exit: `N+1` segments total. A nine-fill WP1 run therefore has 10 segments. A three-fill
V3/V4 cohesive segment has four. Never place a between transport across a palette or
carryover-chain boundary; a reel uses a labelled cut card there.

## Endpoint substitution and seam rules

All 13 rendered states remain evidence, but shared endpoints are substituted during
packaging so the video has no duplicate hold:

- entry: rendered states 0–11, then the next fill's exact accepted first pre-pour frame;
- between: the outgoing fill's exact accepted final frame, rendered states 1–11, then
  the next fill's exact accepted first pre-pour frame;
- exit: the outgoing fill's exact accepted final frame, then rendered states 1–12.

Here “final frame” means the last frame decoded from the exact receipt-hashed accepted
fill video, because that is the frame actually adjacent to transport during composition.
The separately receipt-bound final RGB PNG is still rehashed as capture evidence, but it
is not substituted for the codec-decoded clip frame when measuring or composing a seam.

The checker evaluates the actual joins after substitution. Start and end seam mean
absolute differences must each be no greater than the 99th percentile of adjacent
in-fill frame differences from their corresponding accepted fill video. It must never
compare both ends with one video. The resulting 13-frame transition must be byte-unique,
the outgoing group must be absent at the exit endpoint, the incoming vial center must be
within ±3 px of the frozen fill position, and background sentinel channels may change by
at most two levels. The run compositor also decodes its newly encoded MP4 and rechecks
all 13 frames of every entry/between/exit transition for byte uniqueness; raw PNG
uniqueness alone is not accepted as proof of the packaged video. It also recomputes each
encoded start/end seam against the corresponding accepted fill's in-clip p99 threshold,
so the lossy MP4 step cannot silently invalidate a pre-encode seam pass.

## Evidence boundary and stop conditions

Before any output directory exists, the renderer attaches both annotators and freezes the
exact `/Render`, `/Render/Vars`, and `/Render/PostProcess/SDGPipeline` prim/property
partition. It rechecks that partition, camera/light effective-state hash, nuisance-session
hash, source time code 1, and zero-physics counters at every state. The camera/light hash
is identical immediately before annotator attachment and immediately after it, as well as
across every state and after the checker. The pre-output closure
binds both source layers, all 13 snapshot arrays, all 13 state layers, the accepted fill
receipt/video/snapshot, common and liquid layers, and manifest. The same bytes and
counters are reread after the independent checker and before atomic publication. The
render receipt carries the pre-output closure digest, an exact post-checker closure copy,
and the final runtime attestation; the independent checker rejects any difference among
those records, the build closure, and the frozen runtime partition.

The first GPU action is a three-state pilot (0, 6, 12), only after WP1's smoke and
concurrency benchmark. A reviewer must inspect its contact sheet for rigid motion,
attached labels, liquid contained in the outgoing vial, no ghosting or collision, no
legacy stream/liquid, and no background or lighting movement. Any allowlist, evidence,
camera/light, seam, mask, visual, or independent-review failure stops this approach and
returns INN-54 to Liam; it is not a reason to weaken a gate.

The full render command fails before importing or starting SimulationApp unless
`--pilot-visual-review` names an independently authored, hash-bound acceptance record.
That JSON record uses artifact type `fluidmix_transport_pilot_visual_review_v1`, status
`accepted`, the same `owner_thread_id`, the exact reviewed transport-implementation
SHA-256, an independent `reviewer_id`, UTC review time and decision reference, exact
path/SHA-256 bindings for the pilot render receipt and contact sheet, and six true checks named by `PILOT_VISUAL_REVIEW_CHECKS` in
`src/fluidmix/transport_stage.py`. The loader rereads the pilot receipt, its mechanical
checker gates, build binding and contact sheet. The one pilot must be the representative
`between` mode, which exercises all four mover groups; its approval can unlock entry,
between and exit builds only when their eight-file runtime implementation fingerprint is
identical. The full receipt embeds the resulting review binding, and both the full checker
and run compositor require it. A string saying “approved” without those artifact hashes
cannot unlock the full render.

## Lessons from the rejected approaches

### 2D compositor

The CPU compositor had only the accepted final RGB and liquid mask, not a clean 3D
background plate. Removing the full vial/carrier exposed occluded rail and wall regions;
same-row interpolation produced rectangular bands, duplicated rail texture, and hard
vertical seams. Its useful legacy is the initial transition acceptance vocabulary and the
need to inspect actual intermediate frames. Do not reuse its pixel inpainting or motion
frames.

### Anonymous-session renderer

The first Isaac fallback combined count-body reconstruction, moving transforms,
Replicator's live `/Render` authoring, and evidence in one anonymous session. Annotator
attachment changed the session partition after the code had assumed it was frozen, while
source/counter checks did not close every mutation window. Its useful pieces are the
reviewed 57/27/27 mover allowlists, world-distance measurements, time-code-1/zero-physics
guard, accepted count-body reconstruction, and instance-mask checks. The authored-layer
design moves all deterministic scene edits into inspectable immutable layers and reserves
the session for receipt-bound nuisance state and the measured runtime render partition.

### Frozen commit `86596b6`

That bounded repair correctly moved partition freezing until after annotator attachment
and improved final counter/source checks, but independent review found its evidence
closure bound the snapshot manifest without binding the 13 separately consumed `.npy`
arrays. A changed array could therefore affect reconstruction before partial-directory
creation or after the checker without detection. Do not patch or reuse the frozen commit.
Reuse its post-annotator partition concept, but bind and reread every consumed array,
layer, receipt, video, source file, and counter as this design does.

## Implementation map

- `src/fluidmix/transport_stage.py`: frozen curve, mover/mode plans, layer authoring,
  endpoint layout, runtime partition, and complete evidence closure.
- `scripts/build_transport_stage.py`: source inspection, accepted liquid reconstruction,
  13-layer materialization, endpoint bindings, and build receipt.
- `scripts/render_transport_stage.py`: pre-output post-annotator freeze, same-state Isaac
  captures, pilot-review gate before full SimulationApp startup, atomic candidate
  publication, and post-checker revalidation.
- `scripts/check_transport_stage.py`: independent masks, centering, background, seam, and
  endpoint-substitution gates.
- `scripts/compose_run_sequence_videos.py`: rendered-transition stitching; its old 2D
  `compose_case_video` entry point now fails closed.
- `scripts/package_run_sequence_dataset.py`: requires accepted rendered transport roots
  and packages original accepted fill clips plus endpoint-substituted run videos.

The CPU tests and source-layer parser cannot prove visual quality. Only the queued GB10
pilot and independent visual review can close that gate.

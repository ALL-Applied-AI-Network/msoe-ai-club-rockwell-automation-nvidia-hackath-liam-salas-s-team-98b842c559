# FillCam recovery smoke protocol

Status: frozen before execution at commit `d919b919`; the one permitted renderer process later
failed without an image. This protocol permits one unchanged FillCam smoke render and no synthetic
material variants.

## Question and boundary

Can the public CSI twin produce one inspectable, non-uniform `FillCam` image at frame 686 under
the retained Isaac Sim launchable when its documented asynchronous-throttling workaround is
applied? A pass establishes only this narrow render-feasibility fact for the recorded environment.
It does not establish root cause, a renderer fix, reliability, liquid appearance, carryover,
defect detection, camera robustness, physical validity or sim-to-real transfer.

The earlier attempt produced a decodable but uniformly black 640 x 480 PNG. Its log's asset-loading
warning appeared after the flushed `wrote` marker, so this second look does **not** change the
asset-loading timeout. NVIDIA lists skipped Replicator frames with a stopped timeline as a known
Isaac Sim 6.0.1 issue and gives
`--/exts/isaacsim.core.throttling/enable_async=false` as the workaround. This is support for one
compatibility hypothesis, not proof that it caused the earlier black frame:
[Isaac Sim 6.0.1 known issues](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/known_issues.html).

## Frozen request

| Field | Required value |
|---|---|
| Attempt | `fillcam-recovery-s00-20261024-a1` |
| View | public twin `FillCam` at frame 686 |
| Stage | public twin `stations/fill_station/fill_station.usda` |
| Renderer | `PathTracing`, 640 x 480, 32 real-time subframes |
| Timeout | 1,200 seconds, one renderer process spawn, no retry |
| Kit arguments | telemetry off; structured log off; async throttling Boolean `false` |
| Source lock | SHA-256 `2c9861f7...ed8a89`, five exact recovery files |

The schema-2 manifest rejects extra jobs, fields or settings. The reviewed renderer reads the live
Kit value after Replicator is enabled and before capture, then fails unless it is the Boolean value
`false`. It records that typed attestation in the renderer log. The canonical public renderer and
station are retained and hashed as inputs; the recovery renderer is a reviewed copy outside the
twin. The complete public twin tree is symlink-checked and hashed before and after the attempt.

The runner requires the externally pinned source-lock hash and verifies all five selected files
before and after the spawn. The freeze commit/tag and transfer-archive hash must be recorded before
the instance starts. After start but before spawn, the exact `/isaac-sim/python.sh` digest is pinned
on the command line and then checked before and after. Risky inherited code/config paths including
`PYTHONPATH`, `LD_LIBRARY_PATH`, `EXP_PATH` and `OV_*` are not forwarded; every remaining child
environment value is recorded. Remote inputs and outputs use fresh hash-named directories and must
not pre-exist.

The one renderer spawn starts the attempt. A timeout, nonzero exit, spawn error, missing or invalid
live attestation, missing flushed write marker, source/runtime/twin drift, output-copy/hash failure,
decoder failure, content-gate failure or visual-check failure is final. Copying already-produced
evidence may be retried; the renderer may not be respawned.

## Mechanical gate and visual check

The bounded standard-library decoder accepts only a regular, non-symlink, non-interlaced 8-bit
RGB PNG at exactly 640 x 480. It validates PNG chunk order and CRCs, reverses filters 0--4, bounds
compressed and decompressed input, and rejects trailing streams. Hash, metadata, decoding and
statistics come from the same no-follow file snapshot. Statistics use these fixed denominators:

- 307,200 spatial RGB pixels and 921,600 scalar channel samples;
- nonzero-pixel fraction over 307,200 pixels;
- per-channel mean and population standard deviation over 307,200 samples;
- spatial RGB population standard deviation over 921,600 scalar samples.

Uniform black and any spatially constant RGB image fail. Passing this weak gate means only that the
file is decodable and spatially non-uniform. It is not semantic validation.

Before seeing the new image, the visual checklist is frozen as follows. Primary reviewer: Codex.
Result-claim reviewer: a fresh-context agent that did not design the run.

1. The image shows the fixed FillCam viewpoint at the public twin's frame 686.
2. The fill-station and vial geometry are recognizable and have their expected spatial relation.
3. The output is not a uniform field, noise-only image, UI/error screen, grossly clipped frame or
   unrelated scene.
4. The check does not grade liquid colour, residue, defect class, physical realism or robustness.

## Compute and artifacts

Only reviewed source/config and the public twin may be used on the retained Brev instance. No
`.env`, key, real video, model weight, plant measurement or operator data may be transferred. The
retained stopped `g6e.4xlarge` is quoted at $3.605088/hour. The outer lifecycle is bounded to less
than one hour from start request through verified stop, and the instance is stopped even if render
or copy-back fails. It is never deleted by this run.

The ignored local evidence directory must contain the manifest, command, logs, source and twin
hashes, output inventory/statistics, copied PNG if available, timing, instance state and final
stop verification. Only a small reviewed summary may be committed.

## Post-freeze outcome

For attempt `fillcam-recovery-s00-20261024-a1`, the copied evidence records one renderer-process
invocation with the frozen FillCam/frame-686 request. The log records one live Boolean `false`
attestation for the async-throttling setting and reaches Simulation App startup complete. RGB
capture then returned shape `(0,)`; the process exited 1 after 114.959 seconds. There is no `wrote`
marker and the summary reports no output file, so no PNG decoder or visual result exists.

Fresh result review returned `VALID-WITH-NARROWER-WORDING`. This does not show that the workaround
fixed or caused anything, and it supplies no reliability, liquid, carryover, defect, physical,
camera-robustness or sim-to-real evidence. The raw summary and log remain ignored under
`outputs/20261024-fillcam-recovery/` with SHA-256 values `dfde173e...642dd5` and
`24c3af5e...ef154` respectively.

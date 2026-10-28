# Rigid Vial Transport Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the visually failed 2D transition with a receipt-bound,
no-physics Isaac render in which the filled vial/carrier exits and the next
empty vial/carrier enters under the unchanged FillCam scene.

**Architecture:** Pure Python code validates the source asset, accepted case,
saved final-stage snapshot, exact movable prim groups, rigid motion plan and
receipt. A standalone Isaac 6.2 CLI reconstructs the accepted count-driven
liquid body, freezes source time at 1, authors only anonymous-session-layer
transport transforms, and captures the same RaytracedLighting image plane
without advancing physics or timeline clocks. A separate checker owns the
transport-only acceptance gates; existing fill validators and accepted
receipts remain untouched.

**Tech Stack:** Python 3.12, pytest, NumPy, OpenCV, USD (`pxr`), Isaac Sim
6.2 / Kit 110.4 on the local NVIDIA GB10.

**Spec:** `/home/hackathon5/liamreporoot/next-agent-handoff-20261028/GOAL-PROMPT.md` §0, §1 step 3, WP1b and DP7; supporting evidence in
`/home/hackathon5/liamreporoot/outputs/pipeline-20261028/transport-reverify-v3/`.

## Global Constraints

- The source `fill_station.usda` and `geometry/fill_station.usdc` SHA-256
  values must remain `5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9`
  and `57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db`.
- Preserve camera, geometry, materials, lighting, physics, renderer, stream
  material, height bands, carryover model, fill validators and fill receipts.
  Vial transport is the only authorized scene-facing change.
- Read accepted case artifacts in place; write only to a new lead-owned output
  root. Fail before writer creation on every provenance mismatch.
- Use the accepted final stage snapshot (positions and compositions) and its
  receipt hashes. Do not replay or advance physics.
- Freeze the source stage at time code 1. Do not advance the authored
  neighbour-2 animation at time codes 180–216.
- Capture 640×480 RaytracedLighting at 10 fps with the receipt-pinned ACES /
  exposure state and same-state Replicator capture.
- The accepted receipt must remain byte-identical. The transport receipt is a
  separate artifact and is simulation/label provenance, excluded from Part B.
- No new label/barcode geometry may be cloned or synthesized. A label identity
  discontinuity fails the pilot and stops this route for a ruling.
- No paid compute. The local GB10 runtime is
  `/home/hackathon5/liamreporoot/isaacsim62-env/.venv/bin/python` and costs $0.

## Review Focus

- A source-layer or prim-set drift must fail closed, never broaden a wildcard
  group or silently omit a vial/carrier/fastener/label/liquid member.
- Every member of a rigid group must receive the identical world-Y delta while
  retaining its existing rotation, scale and authored transforms.
- The session-layer diff may contain only exact transport transform properties
  and exact count-body reconstruction; no source/root-layer edit is allowed.
- Snapshot, camera, light, material, renderer, tone and runtime mismatches must
  fail before an output writer or receipt exists.
- Physics callbacks, direct-step counters, simulation time and timeline time /
  playback must remain unchanged for every captured frame.
- RTX temporal ghosting, splice discontinuity, missing outgoing/incoming
  objects and label identity are runtime gates, not assumptions from unit tests.

---

### Task 1: Pure transport contract and motion plan

**Files:**
- Create: `src/fluidmix/rigid_vial_transport.py`
- Create: `tests/test_rigid_vial_transport.py`

**Interfaces:**
- Consumes: source layer records, resolved top-level stage prim paths, accepted
  `receipt.json`, final `particle-snapshots/.../snapshot.json` and its arrays.
- Produces: `RigidTransportSpec`, `resolve_transport_groups(...)`,
  `load_rigid_transport_inputs(...)`, `build_rigid_transport_plan(...)`,
  `validate_authored_transport_diff(...)`, and
  `validate_rigid_transport_receipt(...)`.

- [ ] **Step 1: Write failing exact-group and source-binding tests**

  Use a literal synthetic prim inventory and temporary receipt/snapshot files.
  Assert the two exact source hashes resolve three sorted groups: outgoing
  active vial + carrier-0 + fastener-0 + active labels/barcodes + count body;
  incoming neighbour-1 + carrier-1 + fastener-1; queued neighbour-2 +
  carrier-2 + fastener-2. Missing, extra, renamed or non-transform members and
  a wrong source hash must fail.

- [ ] **Step 2: Run the new tests RED**

  Run: `$PY -m pytest tests/test_rigid_vial_transport.py -q`

  Expected: import/API failures for the missing contract.

- [ ] **Step 3: Implement exact membership and provenance loading**

  `resolve_transport_groups(prim_paths: Collection[str], source_layer_hashes: Mapping[str, str]) -> Mapping[str, tuple[str, ...]]` must use committed exact path sets tied to the two reviewed source hashes; it may not accept prefix-derived extras. `load_rigid_transport_inputs(case_output: Path, stage_path: Path) -> dict[str, object]` verifies the accepted receipt/video/final snapshot manifests and every referenced array before returning paths/arrays.

- [ ] **Step 4: Write and verify RED motion/diff/receipt tests**

  Assert `RigidTransportSpec(frame_count=21, fps=10.0)` rejects invalid values;
  `build_rigid_transport_plan(...)` returns 21 smoothstep frames, frame 0 is
  unchanged, every member of a group has the same delta, incoming ends at the
  active anchor (`+0.0848413319824885 m` world Y), outgoing is monotone toward
  the off-frame bound, and queued remains unchanged. Crossed anchors,
  non-monotone rows, unauthorized session properties, wrong counters/hashes,
  malformed receipts and NaN values fail.

- [ ] **Step 5: Implement the minimal planner and validators**

  Keep the outgoing final delta explicit in the spec with a reviewed initial
  bound of `+0.19 m`; runtime instance-mask acceptance, not this number alone,
  proves it is offscreen. Receipt validation requires 21 unique frame hashes,
  exact bindings, exact group transforms, zero clock/physics advancement and
  all automatic acceptance gates.

- [ ] **Step 6: Run Task 1 GREEN and affected neighbors**

  Run: `$PY -m pytest tests/test_rigid_vial_transport.py tests/test_vial_transport.py tests/test_run_sequence_dataset.py -q`

- [ ] **Step 7: Commit Task 1**

  Commit the pure contract and tests as one reviewable unit.

### Task 2: Standalone no-physics Isaac renderer and checker

**Files:**
- Create: `scripts/render_rigid_vial_transport.py`
- Create: `scripts/check_rigid_vial_transport.py`
- Modify: `tests/test_rigid_vial_transport.py`
- Modify: `docs/architecture.md`

**Interfaces:**
- Consumes: Task 1's validated plan, the exact source stage, accepted case
  output and the existing count-driven-body / same-state helpers in
  `scripts/physx_three_pour_sequence.py`.
- Produces: `frame-0000.png` through `frame-0020.png`, optional MP4 after all
  frames pass, `contact-sheet.png`, and `transport-receipt.json` in a new
  output directory. `--pilot-frames 0,10,20` writes only a pilot receipt and
  three frames.

- [ ] **Step 1: Write failing CLI/preflight/counter tests**

  Import the CLIs without Isaac side effects. With injected fake stage,
  writer, timeline, physics/callback readers and capture function, assert that
  wrong source/snapshot/runtime/renderer/camera/light/material/tone readback,
  any clock/counter change, an unauthorized authored property, or existing
  output fails before publication. Assert the accepted receipt bytes do not
  change. The public CLI requires `--stage`, `--case-output`, `--output-dir`
  and either `--pilot-frames` or `--all-frames`.

- [ ] **Step 2: Run the Task 2 tests RED**

  Run: `$PY -m pytest tests/test_rigid_vial_transport.py -q`

  Expected: missing CLI/runtime interfaces.

- [ ] **Step 3: Implement the renderer orchestration**

  Launch SimulationApp only inside `main`, open the exact stage, set an
  anonymous session edit target, freeze time at 1, verify the resolved groups,
  reconstruct the step-2702 count body from hash-verified arrays with the
  existing body/material helpers, pin RaytracedLighting and tone/exposure,
  apply one common world-Y delta per group/frame, and capture through the
  existing checked same-state session wrapper. Never call a physics step or
  change a source layer.

- [ ] **Step 4: Implement fail-closed publication and checker**

  Write frames to a partial directory, compute transport/session/source hashes
  and automatic metrics, validate the complete receipt, then atomically rename
  the directory. The checker requires: source and accepted-receipt hashes
  unchanged; only allowlisted session edits; zero clock/counter advancement;
  outgoing instance absent at the final frame; incoming center within ±3 px;
  fixed-background sentinel delta ≤2; start and end splice MAD no greater than
  the accepted in-fill p99; 10 fps / 21 frames / unique hashes. Pilot mode
  records unavailable full-sequence gates as pending, never as passed.

- [ ] **Step 5: Document the boundary and run Task 2 GREEN**

  Document that this is a transport-only simulation artifact and that its
  receipt is excluded from model input. Run:

  `$PY -m pytest tests/test_rigid_vial_transport.py tests/test_physx_three_pour.py tests/test_vial_transport.py tests/test_run_sequence_dataset.py -q`

- [ ] **Step 6: Run the full CPU suite and commit Task 2**

  Run: `$PY -m pytest tests -q`, `scripts/smoke_absorbance.py`, and
  `scripts/check_agent_docs.py`; commit only after all pass.

### Task 3: Local GB10 pilot and 21-frame acceptance

**Files:**
- Create at runtime: lead-owned `outputs/pipeline-20261028/rigid-transport-pilot-v3/`
- Update only if measured evidence requires it: Task 1/2 files and tests

**Interfaces:**
- Consumes: committed Tasks 1–2, local Isaac 6.2, one accepted read-only V3
  case and the exact source stage.
- Produces: a three-frame pilot, then one complete accepted or failed
  21-frame transport segment with receipt/contact sheet and measured decision.

- [ ] **Step 1: Record the pilot configuration before launch**

  Add an owner marker, revision/diff/environment/source hashes, exact command,
  hypothesis, falsifiers and stop rule. Use local GB10 only; spend remains $0.

- [ ] **Step 2: Render start/middle/end and run the checker**

  Use `--pilot-frames 0,10,20`. Stop if source/session boundaries, zero-physics
  counters, reconstructed liquid, outgoing/incoming positions, RTX ghosting,
  splice seam or label identity fails. Inspect the three frames directly.

- [ ] **Step 3: Render all 21 frames only after pilot acceptance**

  Run the transport checker and direct contact-sheet review. Do not weaken a
  failed metric; diagnose under §0a and preserve negative evidence.

- [ ] **Step 4: Verify and review the measured result**

  Run the focused CPU tests and transport checker from a fresh state. Apply
  `review-change` to the complete Task 1–2 range. Record measured denominators,
  frame/receipt hashes, runtime, $0 spend, limitations and the route decision.

- [ ] **Step 5: Integrate or park**

  If accepted, remove INN-54's temporary Blocked label and use the renderer as
  WP1/WP1b's fallback. If the label identity or scene boundary requires new
  geometry/material/camera changes, stop and escalate; those exceed D10.

## Self-review

- Spec coverage: maps DP7's failed CPU route to the explicitly allowed Isaac
  fallback, retains all D10 and Part A/Part B boundaries, and includes the
  required local visual pilot before scale-out.
- Step scan: each task has one independent review boundary; each code change
  is preceded by a named RED test and followed by exact verification commands.
- Type consistency: Task 2 consumes only the named Task 1 public interfaces;
  Task 3 consumes Task 2's CLI artifacts and checker.
- Review focus: exact prim drift, source/session mutation, provenance failure,
  clock advancement, renderer seam and identity discontinuity each have an
  owning test or runtime gate.
- Proportion: no renderer implementation is transcribed; exact constants are
  limited to approved hashes, measured anchors and acceptance thresholds.

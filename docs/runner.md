# Production runner

The repository-native runner accepts file uploads, V4L2 devices, and RTSP streams, then
runs one frozen Varied-V4 profile: recipe 5, 640x480 RGB at 10 Hz, three consecutive
337-frame vial windows. It does not train, tune, download models, read evaluation/test
data, or adapt thresholds to the input. Content outside that profile produces an explicit
`insufficient_evidence` decision with a tracker reason code; transport and decode failures
remain terminal errors. From the repository root, after activating the documented ML
environment and making the repository command visible:

```bash
export PATH="$PWD:$PATH"
export PYTHONPATH="$PWD/src:/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.vlm-overlay${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/root/usr/include/python3.12:/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/root/usr/include/aarch64-linux-gnu/python3.12"
export TRITON_CACHE_DIR="/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/triton-cache"
FLUIDMIX_GPU_FREE=1 runner serve
```

The `CPATH` and `TRITON_CACHE_DIR` bindings point to the local, hash-recorded
`libpython3.12-dev` overlay used by the development end-to-end check. They do not
mutate the shared environment and are needed by the Triton CUDA helper.

The default listener is `127.0.0.1:8765`. This accepted development concat is the bounded
end-to-end check, not an input allowlist:

```bash
curl -sS -F source_kind=file -F file=@/home/hackathon5/liamreporoot/outputs/pipeline-20261028/runner-production-factory-01a18128/accepted-v4-train01-cohort.mp4\;type=video/mp4 \
  -F 'run_context={"operator_run_id":"RUN-V4-TRAIN-01","recipe_id":"5","tray_id":"tray-v4-train-01"}' \
  http://127.0.0.1:8765/api/sessions
```

Use the returned `session_id` at `/api/sessions/<session_id>/events`. SSE event IDs are
replay cursors; reconnect with `?after=<last-id>` or `Last-Event-ID`.

The `xgrammar==0.2.8` overlay is an already accepted local runtime dependency. This setup
does not install packages or change the shared environment. Before inference, the runner
validates exact CNN, tracker, topology, VLM-freeze, and pinned local model identities. Each
vial receipt binds the SHA-256 of the exact 337 contiguous RGB frames consumed in that
session. Frame zero of each window is its same-run empty reference. Short input abstains as
`incomplete_frozen_profile`; stream state missing/count/order/incomplete cases retain their
specific tracker reason. Extra frames are not consumed after the third vial closes.
`recipe_id` values other than `"5"` (including `null`) are admitted at the HTTP layer but
finish before media acquisition with `unsupported_recipe_profile`.

`GET /api/health` reports both the implementation commit and a machine-readable
`frozen_profile` object with geometry, timing, recipe, and artifact freeze hashes. PyAV
`rgb24` already matches the accepted tracker's RGB contract, so the adapter only validates
shape/type and contiguity. There is no channel shim, re-baseline, threshold change, or
color-model change. Raw media, weights, temporary crops, and runtime receipts remain outside
Git.

Set `FLUIDMIX_GPU_FREE=1` only after confirming the GB10 has no other owner's process.
Without that explicit attestation, uploaded file sessions fall back to CPU.

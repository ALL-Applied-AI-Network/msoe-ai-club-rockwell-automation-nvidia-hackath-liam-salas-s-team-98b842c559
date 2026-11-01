# Production runner

The repository-native runner accepts any decodable file upload, V4L2 device, or RTSP
stream and runs the same frozen recipe-5 pipeline. It does not identify media by path or
file hash. The source is decoded incrementally, vial boundaries come from observed stream
steps plus a stable empty dwell, and vial 2 and vial 3 become causal references for vial 1
within the operator-created session. The runner does not train, tune, download models,
read evaluation/test data, or adapt thresholds to the input. Unsupported evidence yields
the tracker's or measurement layer's specific abstention reason. From the repository root,
after activating the documented ML environment and making the repository command visible:

```bash
export PATH="$PWD:$PATH"
export PYTHONPATH="$PWD/src:/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.vlm-overlay${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/root/usr/include/python3.12:/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/root/usr/include/aarch64-linux-gnu/python3.12"
export TRITON_CACHE_DIR="/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/triton-cache"
FLUIDMIX_GPU_FREE=1 runner serve --build-attestation /run/fluidmix/build-attestation.json
```

The deployment creates that JSON after the service commit exists; it is not a source-tree
self-hash and must not be committed as a frozen artifact:

```json
{
  "schema_version": "fluidmix-runner-build-attestation-v1",
  "service_commit": "<40-hex commit for the deployed service>",
  "accepted_pipeline_commit": "9913b1a1cc5559cec9242072decabc3aaf2c4142"
}
```

Missing, malformed, or drifted attestation prevents startup. Public health, session, and
event `pipeline_commit` fields report `service_commit`; the accepted implementation pin
remains separate under `health.frozen_profile.accepted_pipeline_commit`.

The `CPATH` and `TRITON_CACHE_DIR` bindings point to the local, hash-recorded
`libpython3.12-dev` overlay used by the development end-to-end check. They do not
mutate the shared environment and are needed by the Triton CUDA helper.

The default listener is `127.0.0.1:8765`. This accepted development concat is the bounded
end-to-end regression input, not an input allowlist:

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
vial receipt binds the SHA-256 of the variable-length contiguous native RGB window actually
consumed in that session. Its first empty frame is the same-run CNN reference. Stream state
missing/count/order/incomplete cases retain their specific tracker reason, and the source is
not consumed after the third vial closes. The accepted frozen artifacts currently support
recipe `"5"`; another recipe fails closed as `unsupported_recipe_profile`, independently of
the source transport or media identity.

`GET /api/health` reports both the deployed service commit and a machine-readable
`frozen_profile` object with geometry, timing, recipe, and artifact freeze hashes. The
source frame rate comes from stream metadata, with PTS or measured-arrival fallback. The
frozen 12-frame check at 10 Hz is represented as 1.2 seconds, and the frozen three-frame
stable/hidden support is represented as 0.3 seconds; source-frame counts are their ceiling
at the measured rate. A proportional 4:3 FillCam input is accepted: its vial ROI is scaled
at native resolution and the analysis raster is normalized to 640x480. An invalid view
fails closed with a camera-geometry reason. PyAV `rgb24` matches the accepted RGB contract;
there is no channel shim, re-baseline, threshold change, or color-model change. Raw media,
weights, temporary crops, and runtime receipts remain outside Git.

Set `FLUIDMIX_GPU_FREE=1` only after confirming the GB10 has no other owner's process.
Without that explicit attestation, uploaded file sessions fall back to CPU.

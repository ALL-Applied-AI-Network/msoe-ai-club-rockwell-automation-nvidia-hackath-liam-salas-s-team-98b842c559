# Configuration, outputs and compute

## Configuration and credentials

There is no configuration loader, environment-variable contract or Cosmos client yet.
Data scripts take a positional root; the USD script uses flags. Its --tanks JSON contains
t200 and t400 three-channel absorbance arrays. T-100 is clear by default.
Keep local tank files under external/ with provenance; the negative real-color fit is invalid.

Use a provider's local credential mechanism or an ignored .env file when an integration
is added. Commit only placeholder variable names in .env.example.
Never put keys in prompts, screenshots, command arguments, reports or Git.
No credential is needed for the CPU quickstart. This guide does not authorize provisioning,
account access or compute spending.

## Locations

| Location | Contents |
|---|---|
| data/ | Reviewed small CSVs with provenance in data.md |
| external/ | Downloads, raw frames, local calibration JSON and data-tool scratch outputs |
| variants/ | Generated USD overrides |
| renders/ | Rendered images |
| outputs/ | Local logs, manifests, predictions and evaluation artifacts |
| .venv/ | Contributor Python environment |

Generated directories are ignored. Ignore rules do not protect already tracked files.
Inspect status and staging. Preserve provenance/checksums with artifacts, then commit only
a small reviewed evidence summary.

## Bounded compute and shutdown

Before authorized compute, record provider/instance ID, owner, inputs, job, time/credit cap
and output destination. Probe access with a small job first. GPU-hour estimates in the
proposal are requests, not measured runtime or approval.
After saving outputs, stop the instance through the provider and verify it is stopped.
A shell exit does not prove billing stopped. Record shutdown status or the specific blocker.
Do not delete teammates' resources or data as cleanup.

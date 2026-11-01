# Configuration, outputs and compute

## Configuration and credentials

The only shared path configuration is two folders: FLUIDMIX_DATA (inputs) and
FLUIDMIX_OUTPUTS (generated files), from the environment or an ignored .env, else external/
and outputs/ (see [data](data.md)). Local Cosmos evaluation clients now run the frozen
synthetic stage and V3 experiments; they are evidence-quality test runners, not a deployed
API or operator-facing service, and Cosmos does not make vial decisions. Data scripts take an
optional positional root; the USD script's --twin defaults to the twin under that root. Its
--tanks JSON contains t200 and t400 three-channel absorbance arrays. T-100 is clear by default.
Keep local tank files under external/ with provenance; the negative real-color fit is invalid.

Use a provider's local credential mechanism or an ignored .env file if a future provider
integration is added. Commit only placeholder variable names in .env.example.
Never put keys in prompts, screenshots, command arguments, reports or Git.
No credential is needed for the CPU quickstart. This guide does not authorize provisioning,
account access or compute spending.

## Locations

| Location | Contents |
|---|---|
| data/ | Reviewed small CSVs with provenance in data.md |
| external/ | Downloads and local calibration JSON; treat as read-only after the manifest check |
| variants/ | Generated USD overrides |
| renders/ | Rendered images |
| outputs/ | Local logs, manifests, predictions and evaluation artifacts |
| .venv/ | Contributor Python environment |

Generated directories are ignored. Ignore rules do not protect already tracked files.
Inspect status and staging. Preserve provenance/checksums with artifacts, then commit only
a small reviewed evidence summary.

The INN-67 reference-bank plan now consumes an explicit twin-published manifest instead of
importing twin builders into product memory. Both commands are CPU-only and no-clobber. First
publish the twin artifacts to a new file in an existing external directory:

    python cli/twin/cohorts/publish_reference_bank_manifest.py \
      --output /absolute/external/inn67-reference-bank-twin-manifest.json

The command prints JSON containing the manifest's `sha256`. Bind that exact value when building
the frozen plan; do not recompute or substitute a different manifest after review:

    python cli/fluidmix/memory/reference_bank/build_inn67_reference_bank_plan.py \
      --published-twin-manifest /absolute/external/inn67-reference-bank-twin-manifest.json \
      --published-twin-manifest-sha256 <sha256-from-publisher> \
      --output-dir /absolute/external/inn67-reference-bank-plan \
      --skip-live-resource-probe

Omit `--skip-live-resource-probe` only during an authorized local GB10 preflight. Publishing the
manifest or plan performs no render and does not authorize GPU, Brev, or test-data access.

For a completed frozen INN-67 development render root, produce the materializer's validation
inputs with:

    python scripts/produce_inn67_postrender_inputs.py \
      --plan-root /absolute/frozen-plan \
      --render-root /absolute/render-root \
      --output-dir /absolute/render-root/postrender-inputs

The output directory must not exist and must be a new child of the render root. The command is
CPU-only and publishes nothing unless all six subruns and all 18 planned cases pass the accepted
validator and exact hash bindings. It does not resume partial renders or create a store export.

## Sealed WP7 preparation

The WP7 development-history command has two CPU-only operations: build a fresh derived population
from the exact accepted synthetic tuple, then publish a separate sealed preparation preflight.
Both operations require caller-declared SHA-256 values, an owner ID, and fresh destinations. The
preflight additionally hash-verifies the accepted ordered 18-vial MP4 plus its composition and
final-review receipts. Its `publish-preflight` operation also requires paths and expected SHA-256
values for the reviewed INN-67 materialization receipt, development reference bank, and frozen
Delta-E config. The command verifies their exact hashes, development-only structure, shared
provenance, receipt output bindings, and CPU/no-WP1/no-video-5 access claims. The bank must carry
the exact accepted RGB-only CNN identity and predicted-record schema; the lineage must carry all
frozen plan/amendment identities, render and predicted-export hashes, and the accepted store tuple.
The command then records absolute paths and identities under `inn67_binding`. It does not decode the
MP4, accept a video-5 path, read frozen one-pass recipe/freeze inputs, acquire a claim, or publish a
one-pass output or journal.

The v2 preflight cannot be published without the accepted INN-67 bank/configuration, but it remains
sealed until the separate Liam access decision. The planned option-3 execution may proceed only
after that decision. The receipt retains the recorded option-1 all-insufficient-evidence fallback;
option 2 is not approved. A successful preparation command proves only deterministic assembly and
source integrity for synthetic development evidence; it is not an approved production-history
route or a one-pass result.

An authorized option-3 run supplies the prepared store twice: as the read-only
`--development-reference-store` with its preflight-bound SHA-256, and a distinct absent
`--evidence-out-store` for explained Part-B rows. Neither the accepted predicted store nor its
population receipt is an execution output. The controller opens the development-reference store
only after the claim, source binding and camera projection. External donor citations remain in
the terminal operator record rather than being substituted into the output database's local
`history_refs`. The explicit evaluation-only real-check skip opens neither loader and leaves the
evidence output path absent. Running or consuming that one-shot still requires Liam's separate
decision.

## Bounded compute and shutdown

Before authorized compute, record provider/instance ID, owner, inputs, job, time/credit cap
and output destination. Probe access with a small job first. GPU-hour estimates in the
proposal are requests, not measured runtime or approval.
After saving outputs, stop the instance through the provider and verify it is stopped.
A shell exit does not prove billing stopped. Record shutdown status or the specific blocker.
Do not delete teammates' resources or data as cleanup.

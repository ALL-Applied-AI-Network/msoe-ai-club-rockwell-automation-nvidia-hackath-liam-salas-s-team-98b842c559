# Queue dispatcher for the PhysX train cohort (scaffold)

Status: scaffold. The scheduler has been tested against simulated hosts only. It has not run
against Brev or Isaac. Base: `7042fbd` on `codex/successor-20261024`.

## Why this layout

Measured on the 7042fbd cohort, 2026-10-27 00:43–00:47 CDT:

- One vial took 15.1 minutes on a warm `g6e.xlarge`
  (case `CASE_407dfd2b02902483`, `d3odmnzyn`).
- Warm Isaac startup took 8.7 s per case. Relaunching Isaac for each case is
  therefore cheap.
- A cold shader-cache start took 384 s. This was shard 1, reported by the agent.
- Load average was about 1.5 out of 4 vCPU.
- The Isaac process used 0.3 to 1.4 cores.
- The GPU switched between 100 % bursts and idle.
- Peak VRAM was 18.8 to 20.8 GB out of 46 GB.

Inferred from those numbers:

- The runs are limited by a serial loop: a physics step on the CPU, then a
  path-traced render on the GPU. Extra vCPUs do not shorten that loop.
- A host with more GPUs costs more per GPU than 1-GPU hosts. AWS `g6e.12xlarge`
  costs $12.59/h, compared with $6.69/h for 3 × `g6e.xlarge`.
- The wins come from scheduling, not from bigger machines.

## What it changes and what it keeps

| Change | Why |
|---|---|
| Each case is a one-case shard (`--shard-count N --shard-index i`). Free host slots pull from a single queue. | With static modulo shards, the slowest shard sets the end time. A queue keeps the load balanced. |
| A failed case is retried once, into a fresh root (`-r1`), on whichever host is free next. | The retry no longer waits on the host that failed. Every attempt is still recorded in `state.json`. |
| Each passing root is copied back as soon as it finishes. | Hosts can stop early. |
| `--stop-idle` runs `brev stop` on each host once the queue is empty and the host has no running case. It never deletes a host. | No paid host sits idle while it waits for other shards. |
| Optional prewarm: push a tar of a warm Kit shader cache before a host's first case. | Saves about 6.4 minutes per new host. The cache path is not yet confirmed, so the setting is null by default. |
| `merge-cmd` prints the agent's own `merge_physx_train_residual_shards.py` command for all N passing roots. | Validation, receipts and the merge stay the agent's code. |

What it does not touch: the supervisor, renderer, plan, receipts, qualification contract,
camera and scene.

The current supervisor binds both the qualification receipt and the nuisance receipt.
Replace every uppercase placeholder in `hosts.example.json`, and pass both receipts to
`bootstrap_host.sh`; the bootstrap copies and hashes both before dispatch.

## Cost model

`python dispatch.py plan ...` computes wall time and cost for one queue feeding k 1-slot
hosts. The defaults are 18 cases, 15 minutes per case and 15 minutes of overhead per host.
The table is a lower bound: per-minute billing, no failures.

| Offer | Hosts | Wall | Cost |
|---|---|---|---|
| AWS g6e.xlarge, $2.23/h | 3 | 105 min | $11.71 |
| AWS g6e.xlarge, $2.23/h | 6 | 60 min | $13.38 |
| AWS g6e.xlarge, $2.23/h | 18 | 30 min | $20.07 |
| Nebius 1×L40S 8 vCPU, $1.86/h | 6 | 60 min | $11.16 |
| Massed Compute 1×L40S 12 vCPU, $1.06/h | 6 | 60 min | $6.36 |
| Massed Compute 1×L40S 12 vCPU, $1.06/h | 18 | 30 min | $9.54 |

How to read it:

- Adding hosts trades money for wall time. At 6 hosts the time is cut in half for about 15 % more cost.
- Cheaper 1-GPU providers cut the cost per case by roughly half. That holds only if Isaac 6.2
  qualifies on them. This must be checked once per provider before any cohort runs there:
  driver, image, and the agent's qualification receipt.

## Not done and open questions

- **Two cases per GPU (`slots: 2`):** the supervisor refuses to start while another Isaac
  process is running (`_find_other_isaac_processes`). Two processes peaking at about 21 GB each
  would also be close to the 46 GB VRAM limit. Using it would need an agent change and a VRAM
  qualification run. If the GPU idle gaps overlap, the gain could be up to about 1.6×; this is
  unmeasured.
- **Where the time goes in a case:** the receipt does not record a breakdown, so the split
  between physics and rendering is unknown. A per-capture timer in the renderer would show
  whether `--rt-subframes 64` and 64 spp at 640×480 are the main cost.
- **Done marker:** `<root>.done` holds the supervisor's exit code. A pass is still determined
  by the merge validator, not by this exit code.
- **Not exercised yet:** the Brev rate-limit backoff, and `brev copy` of a whole run directory.

## Usage

```bash
python dispatch.py plan --offer "aws=2.23" --offer "massed=1.06" --hosts 3,6,9,18
python dispatch.py run hosts.json --dry-run --stop-idle --fail-rate 0.1
python dispatch.py run hosts.json --stop-idle      # real Brev calls
python dispatch.py merge-cmd hosts.json
python -m pytest -q test_dispatch.py
```

## Live test, 2026-10-27 (2 vials, 2 fresh AWS g6e.xlarge hosts)

Measured. Inputs were hash-identical to the agent's 7042fbd cohort. Vials were shard indices
0 and 1 of 18, seeds 330001 and 330002.

Timeline:

| Step | Time |
|---|---|
| Host create to READY | about 6 min |
| Bootstrap: copy inputs, Isaac 6.2 setup, hash check | about 7 min |
| Vial 0 on host a, cold start, launch to done | 22.6 min |
| Vial 1 on host b, cold start, launch to done | 21.4 min |

- A warm vial on the agent's host took 15.1 min. The cold-start overhead is
  therefore about 6–7 min, which is consistent with the agent's 384 s shader-cache figure.
- Both supervisor receipts reported `status: pass`, with 19 captures and 393 decoded
  frames per vial.
- `--stop-idle` stopped both hosts right after copy-back. Each host ran about 37 min,
  which is about $2.75 in total.

The test found a bug: `brev copy host:dir local/` copies the directory's contents, so both
roots were merged locally and one `supervisor-attempts/attempt-0001.json` overwrote the
other. Both case directories survived because their names differ. This is now fixed: each
root copies into `local_results_dir/<root-name>/`, and there is a regression test for it.

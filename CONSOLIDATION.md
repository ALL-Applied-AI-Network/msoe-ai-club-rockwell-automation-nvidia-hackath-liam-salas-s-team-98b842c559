# Pipeline 2026-10-28 consolidation record

This branch starts from `origin/main` at `fdc53752274a28cfcd4c3928bdb6d319ed6e5b39` and is the only branch intended for the final push. The inventory was refreshed after `git fetch --prune origin` on 2026-10-28 CT. It covers all 53 local and remote-tracking refs then present (42 distinct commits). No source ref or other agent's worktree was deleted, rewritten or edited.

Evidence labels used below:

- **Measured:** Git ancestry, stable patch equivalence, file diff, test or hash observed in this worktree.
- **Documented:** disposition fixed by the approved goal prompt or checkpoint.
- **Inferred:** engineering judgment from the measured diff and current contract; the stated limit remains.

## Applied consolidation actions

| Source | Action on `codex/pipeline-20261028` | Evidence and boundary |
|---|---|---|
| `codex/sim-cohort2-20261027` (`7b904b2`) | **Merge** as `4751a1f` | Required by the approved prompt. One `docs/twin.md` conflict was resolved by retaining both Owen's tank-station note and the later CapSeal evidence. |
| Saved V4 compatibility patch | **Apply** as `27be7e7` | Exact source SHA-256 `cb3e0c7ceb0d1d6995e94b2e83c4f4aa4ee23e8c38d6c174944c7feb2427fe86`; the two saved concurrency files were copied byte-for-byte. Targeted checks: 92 passed. |
| `codex/run-sequence-20261027` (`4f40ff6`) | **Cherry-pick parts**: `0f4e733`→`6e1a4df`, `6c82c84`→`c151200`, `4f40ff6`→`d469cb0` | Preserves original author metadata. Manifest, supervisor and dispatcher commits were excluded. A hidden fixed-30-case dependency was corrected test-first in `fe63791`; targeted checks: 12 passed. |
| `codex/archive-model-bakeoff-wip-20261027` (`a18c7fb`) | **Cherry-pick parts** as authored extraction `62894ca` | Imported eight generic runner/parser/test paths. Answer-leaky benchmark/SFT/real-validation code, deleted-artifact bindings and unapproved training launchers were excluded. Focused checks: 110 passed. Runner remains a scaffold until WP3/WP6 hardening. |
| `origin/huelkes/inn-52-segment-the-liquid-stream-and-identify-the-active-tube-in` (`5c45883`) | **Merge** as `04a6a51` | Clean teammate code/docs/tests; both speedysebi commits and authorship retained. Results are real-video inspection evidence, never training or held-out evaluation. Targeted checks: 6 passed. |
| `origin/huelkes/inn-52-rescope` (`1e709a2`) | **Merge** as `c327b00` | Clean teammate scope documents retained alongside the implementation/result branch. It keeps plumbing uncertainty explicit. |
| `origin/codex/inn-11-carryover-review-fixes` (`673743d`) | **Merge** as `3ceccc0` | Self-contained scalar-input correctness fix with regressions for huge reals and `np.timedelta64`; original speedysebi authorship retained. Targeted checks: 48 passed. |
| Consolidated tank-station rebuild | **Fix** as `532ac27` | The first complete suite exposed platform-dependent final-bit output from USD's `SetLookAt/GetInverse`. Authoring the already reviewed fixed-camera matrix directly preserves camera behavior and restores byte-identical provenance across the tested Linux/macOS USD 26.11 environments. Exact rebuild: 1 passed; tank-station module: 33 passed. |

## Complete branch/ref inventory

Aliases are grouped only when they resolve to the same commit; every local and remote-tracking branch is named. “Already contained” means literal ancestry or a measured equivalent patch is already reachable from the consolidation history.

| Ref(s) | Tip | Decision | Reason |
|---|---:|---|---|
| `burn-codex`; `main`; `origin/burn-codex` | `40d31b9` | **Already contained** | Literal ancestor of both approved base and sim history. |
| `origin` (`origin/HEAD`); `origin/main` | `fdc5375` | **Already contained** | Approved base; first parent of this consolidation line. |
| `origin/codex/tank-station` | `44a255d` | **Already contained** | Literal ancestor of `fdc5375`; Owen's history is retained by the base merge. |
| `codex/sim-cohort2-20261027` | `7b904b2` | **Merge** | Required pipeline history; merged as `4751a1f`. |
| `claude/beer-mixing-20261026` | `b48e375` | **Already contained** | Literal ancestor of sim history. |
| `claude/sim-runner-20261027` | `4bc1e32` | **Already contained** | Both dispatcher commits have exact reachable equivalents `f37f47c` and `785b2b1`. |
| `codex/run-sequence-20261027` | `4f40ff6` | **Cherry-pick parts** | Only transport, packager and tests were selected; conflicting manifest/supervisor/dispatcher changes were excluded. |
| `codex/archive-model-bakeoff-wip-20261027` | `a18c7fb` | **Cherry-pick parts** | Only safe generic runner/parser paths were extracted; old evaluation/training artifacts disclose answers or depend on deleted inputs. |
| `origin/huelkes/inn-52-segment-the-liquid-stream-and-identify-the-active-tube-in` | `5c45883` | **Merge** | Clean, useful optional stage-timing evidence; merged with authorship. |
| `origin/huelkes/inn-52-rescope` | `1e709a2` | **Merge** | Clean scope clarification; merged with authorship. |
| `origin/codex/inn-11-carryover-review-fixes` | `673743d` | **Merge** | Uncontained correctness fix; merged with authorship. |
| `origin/codex/inn-11-contract-review-fixes` | `f8614fb` | **Cherry-pick parts** | Partition/artifact-leak and circularity principles remain relevant, but the changed proposed-record prose is obsolete. Carry the principles into the new INN-58 v1 contracts rather than merging stale text. |
| `origin/codex/inn-11-record-contract` | `4545a95` | **Already contained** | Literal ancestor of base and sim history. |
| `origin/codex/inn-11-carryover-validation` | `c5dd82a` | **Already contained** | Literal ancestor through merged PR #6. |
| `origin/codex/inn-22-24-agent-docs` | `ec25d38` | **Already contained** | Literal ancestor of base and sim history. |
| `origin/codex/final-synthesis-9-20261024` | `67e673d` | **Already contained** | Literal ancestor of sim history. |
| `origin/INN-7_Audit_Dataset2` | `6dbfdfc` | **Already contained** | Literal ancestor of base and sim history. |
| `origin/INN-9` | `8449bf8` | **Already contained** | Literal ancestor of base and sim history. |
| `origin/claude/scaffold-agents-absorbance` | `c2f88d4` | **Already contained** | Literal ancestor of base and sim history. |
| `origin/docs/add-proposal` | `ad04ddf` | **Already contained** | Literal ancestor of base and sim history. |
| `codex/a24-empty-path-diag-20261025` | `f2fad0e` | **Already contained** | Exact reachable patch equivalent `01ee1ec`. |
| `codex/a26-transparent-semantic-20261025` | `5164b2f` | **Already contained** | Exact reachable patch equivalent `34ee3dc`. |
| `codex/a27-two-pass-semantics-20261025` | `2f6dd3e` | **Already contained** | Exact reachable patch equivalent `0aa2a83`. |
| `codex/a36-particle-array-probe-20261025` | `050e40d` | **Already contained** | Useful probe code/results have exact or amended reachable equivalents in sim history. |
| `codex/a40-resolution-probe-20261026` | `12f68ba` | **Already contained** | Implementation `44075fc` and amended result `423c741` are reachable. |
| `codex/e2e-t4-runtime-20261026` | `04a8018` | **Already contained** | Runtime/light fixes have reachable equivalents; nuisance authoring was reworked and extended. |
| `codex/f2-manual-step-probe-20261025` | `2cd7afd` | **Already contained** | Both commits have exact reachable patch equivalents `f6a8e07` and `cf8b2c6`. |
| `codex/f2-mesh-fallback-20261025` | `445199e` | **Already contained** | Exact reachable patch equivalent `26d689d`. |
| `codex/paintball-launcher-plumbing-20261025` | `82b04e9` | **Already contained** | Exact reachable patch equivalent `cb66933`. |
| `codex/paintball-receipt-metadata-20261025` | `d107c5a` | **Already contained** | Exact reachable patch equivalent `60116e1`. |
| `codex/volume-review-tool-20261025` | `3857213` | **Already contained** | Exact reachable patch equivalent `c6634f6`. |
| `codex/residual-pilot-cli-20261025` | `3abbcc0` | **Abandon** | Uncontained 21-vial CLI is frozen to obsolete split v2; current rules require split v3 and later materializers supersede it. |
| `codex/stored-reference-8-20261024`; `origin/codex/stored-reference-8-20261024` | `4db38a0` | **Abandon** | Explicitly stopped reference-detector route; current product uses camera evidence and measured history. |
| `codex/cosmos-stress-20261027` | `4f0413b` | **Abandon** | Uncontained S0088 identity tweak; checkpoint marks S0088 superseded by V3. |
| `codex/vision-explore-6-20261024`; `origin/codex/vision-explore-6-20261024` | `594e5a5` | **Abandon** | Explicitly stopped saturation-mask preflight; superseded by implemented measurement paths. |
| `codex/twin-recovery-7-20261024`; `origin/codex/twin-recovery-7-20261024` | `d02be28` | **Abandon** | Failed FillCam recovery route; accepted renderer/cohort history supersedes it. |
| `codex/cosmos-explore-20261024`; `origin/codex/cosmos-explore-20261024` | `dbb38f0` | **Abandon** | Stopped initial Cosmos exploration; no current implementation to retain. |
| `codex/cosmos-explore-2-20261024`; `origin/codex/cosmos-explore-2-20261024` | `f45f785` | **Abandon** | Stopped local smoke/preview route; later model paths supersede it. |
| `codex/cosmos-explore-3-20261024`; `origin/codex/cosmos-explore-3-20261024` | `dcfab38` | **Abandon** | Stopped black-FillCam/Brev route; accepted local renderer/cohort history supersedes it. |
| `codex/cosmos-explore-4-20261024`; `origin/codex/cosmos-explore-4-20261024` | `b8ffa3c` | **Abandon** | Stopped Cosmos contract-v1 result; superseded by later explanation/evaluation code. |
| `codex/cosmos-explore-5-20261024`; `origin/codex/cosmos-explore-5-20261024` | `a193da8` | **Abandon** | Stopped frozen/no-thinking exploration; obsolete fixtures and contracts. |
| `codex/pipeline-20261028` | current | **Already contained** | Consolidation destination, not an import source. Current head is recorded by Git; this row is refreshed at the exit gate. |

## Excluded-content and provenance limits

- No branch or ref was deleted. “Abandon” means do not import its content into this branch.
- The archived bakeoff runner is not accepted for WP6 yet. It must record all ordered images, bind local model snapshots to immutable revisions, enforce an all-error completion gate, and consume RCA-bench v1 before use.
- INN-52's measured video-5 results are documented inspection evidence only. Its thresholds were chosen on those same vials, so the merged code/results are not training data or held-out evaluation.
- The run-sequence packager now validates structural provenance for variable run sizes. Dataset-specific count/balance gates remain explicit in WP1/WP1b acceptance and may not be inferred or weakened by the generic package validator.
- The exact `f8614fb` prose was not imported. Its relevant separation and circularity safeguards are inputs to the new lead-owned v1 contracts.

## Exit-gate checklist

- [x] Exact approved base and source histories recorded.
- [x] Every local and remote-tracking branch classified.
- [x] Required merges/selective imports preserve author metadata.
- [x] Repository hygiene scan: no tracked `outputs/`/`external/` files, prohibited media/weight/archive extensions in the consolidation diff, >1 MiB blobs, or common private-key/token patterns.
- [x] Full CPU suite passes: 2,036 passed in 169.22 seconds.
- [x] `scripts/smoke_absorbance.py` passes.
- [x] `scripts/check_agent_docs.py` passes: shared pointer, 11 skills and 51 document file-target checks.
- [ ] Independent consolidation review is clean.

The diagnostic `scripts/fit_absorbance_video5.py` may exit successfully while printing negative absorbance. That remains a documented non-physical diagnostic and is not a calibration or physical-validity gate.

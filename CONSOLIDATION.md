# Pipeline 2026-10-28 consolidation record

## Completion-audit refresh (2026-10-31)

The active local consolidation branch is now `codex/pipeline-20261028`. It was fast-forwarded
through the accepted WP7/video-5 source-preparation head
`f7e909dab42119eb0387d6f5fb80c8853e718b0e`, then received three bounded local follow-ups:

- `b507199`: reconcile the WP4 matrix with the accepted CNN-backed downstream path;
- `0717491`: repair only the stale final-wiring test fixture so it exercises the mandatory
  phase-capture boundary and source-hash inventory; and
- `4694824`: declare the `scikit-image` CPU dependency already used by the measurement path and
  verified test suite.

Fresh verification on `07174915c3027ea5b82d94f43498b2d64bf80977` passed **3,345 tests** with
**17 skipped** in `1,477.42 s`. The dependency-only child `4694824` then passed the 18-test final
wiring file, synthetic absorbance smoke, agent-doc checks, requirement parsing and a direct
`scikit-image 0.26.0` specifier check. `git diff --check` is clean. A fresh Brev inventory also
reported zero workspaces and zero nodes.

This is branch-readiness evidence, not a §6 completion claim. Accepted limitations remain explicit:
WP1 DP3 is parked as a measured detection limit; transport is 81/84 with one V4 development run
deferred; WP3 baselines and the full WP6 comparison were not run; and video-5 attempt-02 remains
prepared but must not create a claim or reopen the media without Liam's approval. Final independent
review passed after one bounded closure correction: INN-62 was returned to In Review + Blocked
because its indirect query-to-image chain does not literally satisfy WP5.5 / §6 image grounding.
Review SHA-256 is `205d34e104290baf1e5c766dc34c6be35fa56c671489999f41eb1ae743f738a5`.
The tracked README, plan and one-page deliverable were then refreshed through `c4bbcae`; their
independent final prose review passed after removing two stale claims about the already accepted
CNN path. Documentation-review SHA-256 is
`1b362b22329d1e53d5df6ee8bdf8835201090dffe96c9398ceef943048f96237`.
Linear closure and the one authorized push remain open.

## Final-integration staging (2026-10-30)

This staging branch is `codex/pipeline-final-integration-01a18128`, created from the exact
destination base `codex/pipeline-20261028` at
`60ed104a15072ac7b6fa43c797cc43d9ec25630d`. The destination worktree and branch were not
modified. This is an intermediate integration record, not a final-completion claim.

Three independently accepted tranches are currently integrated with complete Git ancestry:

| Tranche | Exact accepted commit | Integration merge | Independent review evidence |
|---|---|---|---|
| WP1/DP3 | `c8cbdaa1970838c88ed83b27b98c5325fa4c2bd6` | `8abad54aa2725f723e4a87f31afdd7124024bfda` | PASS; `FINAL-REVIEW.md` SHA-256 `d1bd581f82e7048689422ab5bad10f407c53b0ee90a1f14ff75c3fa9de466e70`; `FINAL-REVIEW.json` SHA-256 `98c1d3fe04b62cf60394e6964aed0c4d49db17934c11b740803ab84d71e280bb` |
| WP7/INN-67 | `7980cc82ea10dc1b33d14190a0cf226c9fb7ab8d` | `1755dcb41b4ae0405dea38fdc438c99bf40dc19d` | attempt 02 PASS; `FINAL-REVIEW.md` SHA-256 `d7112986d936ebf183113beb0b0667466d25f1ce5ab0b1926019cbd16a1ae5e0`; `FINAL-REVIEW.json` SHA-256 `c8287bf0e16fa6c9c57901f3e74ac9606e1e8fd43ec383fca8c73650bb1e3d2a` |
| WP5/WP6 | `a9c8644695c115c3d0f52e72bed7c9ec6c6728a7` | `4650738e6ccef5a16ad3fd0c771d69413102d00b` | attempt 02 PASS; `FINAL-REVIEW.md` SHA-256 `281f2366286c7d5f3cffdae000fae84e2ea4d1496af043126eb20b8536b2d821`; `FINAL-REVIEW.json` SHA-256 `adb7bd8108e79ba459e21ae0958750cd886ec0c5355db424c7d715790b7617e2` |

All three were explicit non-fast-forward merges. WP1/DP3 and WP7/INN-67 merged without textual
conflicts; Git cleanly auto-merged their shared `docs/architecture.md` and boundary-registry
changes. WP5/WP6 then produced two genuine ownership-registry conflicts in
`src/fluidmix/part_boundary_manifest.py` and `tests/test_part_boundary.py`; the integration
resolution unions every accepted WP1, WP5/WP6 and WP7/INN-67 registration while retaining the
destination's target-alias, carryover and palette protections.

The failed WP5/WP6 attempt 01 is excluded; only its exact attempt-02-reviewed corrective head is
merged. Still excluded are unreviewed controller/one-shot work and all generated outputs,
databases, media and model weights. This remains a source integration awaiting final independent
review, not a completion or deployment claim.

Fresh complete-branch verification after the ownership resolution:

- `python -m pytest tests -q`: `3090 passed, 18 skipped in 1155.58s`.
- `python -m pytest tests/test_part_boundary.py -q`: `53 passed in 15.03s`.
- `python scripts/smoke_absorbance.py`: synthetic smoke passed; old-fluid fraction `0.150000`,
  RMS absorbance residual `8.012e-18`.
- `python scripts/check_agent_docs.py`: shared pointer, 11 skills and 55 document file-target
  checks passed.
- `git diff --check`: passed with no output.

The first full integration run exposed only the duplicate ownership classification created by the
mechanical registry union: `cnn_predictor` and `empty_reference_rgb_predictor` appeared in both
production and evaluation categories. The integration correction keeps their accepted WP5/WP6
production ownership, removes only the duplicate evaluation entries, and then passed the complete
suite above.

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
| `codex/run-sequence-20261027` (`4f40ff6`) | **Cherry-pick parts**: `0f4e733`→`6e1a4df`, `6c82c84`→`c151200`, `4f40ff6`→`d469cb0` | Preserves original author metadata. Manifest, supervisor and dispatcher commits were excluded. The first hidden fixed-30-case dependency was corrected test-first in `fe63791`. Review round 1 then closed the remaining V3/V4 selection, generated-claim, warning-receipt, source-video-binding and model-input-allowlist seams; covering neighbor checks: 67 passed. |
| `codex/archive-model-bakeoff-wip-20261027` (`a18c7fb`) | **Cherry-pick parts** as authored extraction `62894ca` | Imported eight generic runner/parser/test paths. Answer-leaky benchmark/SFT/real-validation code, deleted-artifact bindings and unapproved training launchers were excluded. Focused checks: 110 passed. Runner remains a scaffold until WP3/WP6 hardening. |
| `origin/huelkes/inn-52-segment-the-liquid-stream-and-identify-the-active-tube-in` (`5c45883`) | **Merge** as `04a6a51` | Clean teammate code/docs/tests; both speedysebi commits and authorship retained. Results are real-video inspection evidence, never training or held-out evaluation. Targeted checks: 6 passed. |
| `origin/huelkes/inn-52-rescope` (`1e709a2`) | **Merge** as `c327b00` | Clean teammate scope documents retained alongside the implementation/result branch. It keeps plumbing uncertainty explicit. |
| `origin/codex/inn-11-carryover-review-fixes` (`673743d`) | **Merge** as `3ceccc0` | Self-contained scalar-input correctness fix with regressions for huge reals and `np.timedelta64`; original speedysebi authorship retained. Targeted checks: 48 passed. |
| Consolidated tank-station rebuild | **Fix** as `532ac27` | The first complete suite exposed platform-dependent final-bit output from USD's `SetLookAt/GetInverse`. Authoring the already reviewed fixed-camera matrix directly preserves camera behavior and restores byte-identical provenance across the tested Linux/macOS USD 26.11 environments. Exact rebuild: 1 passed; tank-station module: 33 passed. |
| INN-58 Tasks 4a/4b (`e82a338`, `e5480ce`, `9f6603d`) | **Retain/cherry-pick** through `673230c` | Independently accepted ownership/read-isolation boundary. Task 4a was already on the branch; Task 4b was replayed with authorship. |
| WP4 accepted chain (`415b1c8` through `dac3de7`) | **Cherry-pick** as `73b3730` through `a7a6d5d` | CPU clip/measurement producer, provenance fixes and the V3 42-row regression matrix; all source commits had passed their scoped review. |
| WP5 accepted chain (`30d822a`, `9a3ed46`, `5d9ba6a`) | **Cherry-pick** as `46067d4` through `a8ac6d4` | Append-only store and two leakage corrections, including strictly-earlier measured cross-run regime references. No real store population is implied by code integration. |
| WP3 accepted contract/scorer chain (`620f4510` through `3f58a435`) | **Cherry-pick** as `c3db0eb` through `f6c371c` | Frozen CPU RCA-bench contracts and reviewed real-input corrections. |
| WP3 accepted controller chain (`373257ed`, `5c406223`) | **Cherry-pick** as `db41359`, `ebcc1d1`; register boundary ownership as `2c1d0d3` | The original controller candidate is retained only with its independently accepted bounded remediation. The combined implementation validates actual freeze bytes before template/oracle use, separates raw/final scoring, derives method class, publishes atomically and exposes a standard-library-only fatal finalizer. Fresh WP3/boundary slice: 99 passed. |
| WP1 occlusion/portable-attestation chain (`16fbb85`, `765f045`, `24889ca`) | **Cherry-pick** as `de97ee3` through `30445a8` | Source-bound carrier occlusion gate plus Liam's exact-authored / `rtol=atol=1e-12` derived-matrix rule. This is the code used by the accepted L40S proof and fills. |
| WP1 aggregate acceptance chain (`23186a3`, `e595ef2`, `309812b3`) | **Cherry-pick** as `76d3e97` through `5f611dd` | Reviewed post-render validation, packaging and real-producer binding. It is integrated code only; final aggregate output still depends on all fills and transport. |
| Accepted transport chain (`29b8511` through `0208b94`) | **Cherry-pick** as `7d5232d` through `6c8a265`; register boundary ownership as `42b73a6` | Authored 13-state no-physics transport, evidence remediations, derived-matrix portability, exact accepted-`24889cad` readiness identity and real offline-revalidation schema. The final focused re-review passed against the actual proof log bytes; motion/camera/lights/stage/geometry/fills are unchanged. Fresh transport/boundary slice: 317 passed. |
| Consolidated ownership registry | **Fix** as `b204f9a` | The combined branch added three modules after INN-58's inventory freeze. The integration patch classifies `transform_attestation` and `wp1_postrender` as Part A and `rca_bench` as evaluation-only; boundary tests pass 52/52. |

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
- The run-sequence packager supports only the explicit `--cohort-version v3` and `--cohort-version v4` source-built manifests. It validates the selected manifest identity, accepts the V3 `shard-merge.json` warning provenance (plus the supported legacy receipt), derives package counts and README claims from validated rows, and rejects unbound source video or non-allowlisted model-input fields. These are synthetic integrity checks, not dataset-quality, physical-validity or sim-to-real evidence. Dataset-specific acceptance gates remain separate and may not be inferred or weakened by the generic package validator.
- The exact `f8614fb` prose was not imported. Its relevant separation and circularity safeguards are inputs to the new lead-owned v1 contracts.

## Exit-gate checklist

- [x] Exact approved base and source histories recorded.
- [x] Every local and remote-tracking branch classified.
- [x] Required merges/selective imports preserve author metadata.
- [x] Repository hygiene scan: the only tracked `outputs/` path is the intentional 12,575-byte reviewed WP7 `IMPLEMENTATION-REPORT.md` evidence summary permitted by `docs/operations.md`; there are no tracked `external/` files, prohibited media/weight/archive extensions in the consolidation diff, >1 MiB blobs, or common private-key/token patterns.
- [x] Full CPU suite passes after the accepted WP1/WP3/WP4/WP5 integration: 2,592 passed, 14 skipped in 315.46 seconds.
- [x] `scripts/smoke_absorbance.py` passes.
- [x] `scripts/check_agent_docs.py` passes: shared pointer, 11 skills and 55 document file-target checks.
- [x] Initial independent consolidation review is clean. Round 0 requested five Important changes; scoped round 1 verified all five addressed with no new Critical, Important or Minor breakage.
- [x] Final independent review of the refreshed complete branch passed after the INN-62 closure overstatement was corrected; review SHA-256 `205d34e104290baf1e5c766dc34c6be35fa56c671489999f41eb1ae743f738a5`. Submission-facing documentation review also passed at SHA-256 `1b362b22329d1e53d5df6ee8bdf8835201090dffe96c9398ceef943048f96237`. No review-failed intermediate is represented without its accepted correction.

The diagnostic `scripts/fit_absorbance_video5.py` may exit successfully while printing negative absorbance. That remains a documented non-physical diagnostic and is not a calibration or physical-validity gate.

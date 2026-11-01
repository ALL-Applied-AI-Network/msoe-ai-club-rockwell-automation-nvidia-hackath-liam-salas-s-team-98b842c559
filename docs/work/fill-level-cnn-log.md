# Fill-level CNN implementation log

Newest entries appear first. Evidence labels follow `AGENTS.md`.

## 2026-11-01 — Step 8 Revision 2: validation result and calibration preflight fix

### Result

- **Source-file observation:** before the revised-only retry, Codex reread the newest
  main-checkout review at SHA-256 `19691bc3...9b4c`; it kept Step 8 GO, required the frozen
  validation result even if its gates failed, and authorized the one calibration run after
  disclosure. Immediately before calibration, the review advanced to `4dac48f2...7366`:
  the new hidden result remains below the frozen 95% gate but is accepted by the user, whose
  practical-use bar is 90%. The hidden target for Step 9 remains 95%.
- The first rule attempt was preserved after seed `1101016` exposed a label-truth boundary:
  its 4.144 mL surface starts at full row 427, immediately below the frozen prediction rows
  `[83,427)`. Commit `b0da6f6` left prediction frozen and allowed label truth alone to fall
  back to the full trained crop, with an explicit fallback counter.
- Fresh `run-v2` completed the rule pass over all 360 validation frames. Its rule-summary
  SHA-256 is `850228d2...b27b`. The revised wrapper then failed in a four-worker PyTorch
  DataLoader with `unable to allocate shared memory(shm)`; the full failure was preserved,
  and no rule work was repeated.
- Fresh `run-v3` reused that exact rule summary and changed only DataLoader workers from four
  to zero. It ended `GATE_FAILED` normally at `2026-11-01T22:53:06Z`. The full revised
  summary SHA-256 is `04bb91db...e61d`: normal visible-surface p95 is 17.05 rows against the
  frozen 3-row limit; old Step 6 hidden success is 69/72 (95.83%, pass), and new exact-v4
  hidden success is 17/18 (94.44%, just below 95%, accepted by the user). A hidden success
  means the true level lies inside the reported range, whose validation median is about
  30.3 mL wide; it is a large-error flag, not a precise point estimate.
- Revised validation evidence was copied container -> fresh host -> fresh ignored local path
  `validation-evidence-v3-20261101T225306Z`. All eight files pass `SHA256SUMS`; that manifest
  is SHA-256 `60f63d23...ef55`. Inputs were hash-bound, the aggregate contained exactly 26
  train/validation directories and no test directory, and the evaluator records zero train,
  test and real-frame access.
- The approved 98 calibration frames were packaged without `_tmp_avg_label.png`: vial 1 has
  28 frames and vial 13 has 70, bound to index SHA-256 `73181d39...f9b2`. Linux-metadata
  input archive SHA-256 is `13990386...72ae`; its 105-file manifest is `c635a68b...b04e`.
  The runner pinned the selected weight, successful frozen rule, gate-failed revised summary
  and user authorization, then launched exactly one fresh attempt.
- That attempt failed before substantive calibration. The calibrator rejected the frozen
  rule during its first provenance gate: the approved 360-frame profile incorrectly expected
  ledger `680d6d38...04f4c`, which belongs to the full 3,480-train + 360-validation preflight,
  while the validation-only evaluator correctly recorded `cffb8c32...51bf` for 360 samples /
  720 files. It failed before opening the frame index, creating a calibration output root,
  importing/loading Torch or decoding any PNG, so zero real frames were scored and no ruler
  was fit. The preserved failure manifest is SHA-256 `a52cbf7d...ee66`.
- The minimal fix changes only that approved validation-only ledger and adds a regression
  that rejects the full-training ledger at this gate. The 28 ruler tests pass, the actual
  preserved frozen rule now passes the corrected guard as `holder_band_v2`, and Python
  compilation passes. No retry was launched: the review allows only one calibration run, so
  a corrected first substantive inference requires fresh user/review authorization.
- Hidden-surface work is implemented and measured as above. The approved calibration-only
  pour-timing estimate remains deferred until after the Step 10 checkpoint if time permits.
  The existing Brev instance remains running; no instance lifecycle action, push, PR or
  external message occurred.

### Commands run

- Package and checksum fresh revised-only inputs; preserve a local manifest-transcription
  failure and a Windows-directory-permission extraction failure; rebuild the successful
  package with Linux modes; verify it locally, on the host and in the container.
- Re-read/hash the review at both launch boundaries; run only the revised validation pass
  with `--workers 0`; preserve and hash the gate-failed evidence at all three locations.
- Package only the approved calibration index and 98 PNGs plus the frozen calibrator code;
  verify all 105 input hashes at each hop; launch one detached calibration attempt.
- Preserve the failed attempt container -> fresh host -> fresh ignored local path; compare
  every payload hash; trace the exception ordering to prove that no index/model/image access
  occurred; distinguish the train+validation and validation-only preflight ledgers.
- Update the constant and regression with `apply_patch`; run the 28 focused ruler tests,
  Python compilation and an actual-rule preflight against the preserved 360-frame summary.

### Problems

- A first local retry-package manifest line transcribed the revised evaluator hash
  incorrectly. Local `sha256sum -c` caught it before upload; the bad archive remains
  preserved, and a separately named corrected archive was used.
- The corrected Windows tar retained read-only OneDrive directory modes and could not fully
  extract in the container. The partial `input-v4` remains preserved. Fresh `input-v5` was
  staged with Linux directory modes, independently verified and extracted without error.
- `brev copy` flattened the validation evidence into its existing local parent. The fresh
  files were moved, without overwrite, into the intended ignored evidence directory and the
  host manifest then verified exactly.

## 2026-11-01 — Step 8 Revision 2: combined synthetic retraining

### Result

- **Source-file observation:** immediately before launch, Codex reread `AGENTS.md`, the
  fill-level plan and the newest main-checkout review at SHA-256
  `265ab61a9a1532328f72627d168e91b178fbb49d6ce52aac07b2eb4e413c4645`.
  The review contains no STOP and says to finish Revision 2 retraining, validation and one
  calibration run before the next checkpoint. The existing Brev instance remained running.
- Batch/aggregate checkpoint commit `fe545ae` supplied the exact combined summary
  `bfe6650e...d6b5`, exception review `e0efefd...9eea`, 3,480/360 eligible train/validation
  contract and zero exclusions. The fresh container aggregate reproduced the committed
  combined summary byte-for-byte, contained exactly 26 train/validation links and no test
  link, and rechecked 3,840 samples / 7,680 files against ordered ledger
  `680d6d38...04f4c`. Container preflight SHA-256 is `eff05a10...87f5`.
- All immutable training inputs passed the same checksum manifest locally, on the Brev host
  and in the container. Input archive SHA-256 is `dc360c86...fbe9`, input manifest is
  `26537b55...8e55`, runner is `a607e00e...6e37`, and run contract is
  `976eb596...8b09`. Training script/module hashes remain `9579c678...cbae` and
  `ed6e8fce...b47`; the Step 7 Python/CUDA environment is reused read-only.
- **Measured smoke:** fresh `smoke-attempt-01` completed two epochs in 7.70 seconds with
  64 train / 32 validation records, best epoch 2, validation loss 1.342374 and macro-IoU
  0.217297. It opened zero test manifests/images; its summary SHA-256 is
  `e1235256...d030`.
- **Measured full training:** fresh `full-attempt-01` completed all 30 epochs in 477.59
  seconds. Validation-only selection chose epoch 29 with validation loss 0.046605 and
  macro-IoU 0.948626. Per-class IoU is empty 0.981718, liquid 0.982377, foam 0.889917 and
  blocked 0.940490. The selected weight is `best-epoch-029.pt`, SHA-256
  `1949bdea...b094`; final epoch-30 weight is `f9514cfe...ea8c`; progress is
  `b6aedc2d...6f0e`. The committed training summary SHA-256 is
  `728991a8...0930`.
- Both runs record the frozen Revision 2 liquid-only recolour/darkening augmentation,
  combined population contract, CUDA/L40S environment, validation-only selection and zero
  test access. The full preflight inside the training summary exactly matches the independent
  aggregate ledger. No sealed-test or real frame was linked, opened or decoded.
- Container outputs were copied without the aggregate/raw frames to fresh host evidence
  `training-evidence-v1-20261101T214142Z`, then to the matching fresh ignored local path.
  All 57 payload files—including immutable inputs, link ledger, environment, smoke/full
  state/logs/summaries and every checkpoint—pass `SHA256SUMS`; that manifest is SHA-256
  `b9cd4878db946f5d0efc81c651497086ff596dbfeef9d2569fa69f7d351ea20f`.
  Weights remain ignored and outside Git.
- The Brev instance remains running and idle for the approved validation/calibration work.
  Codex did not create, reset, stop or delete an instance; did not push, open a PR or send an
  external message.

### Commands run

- Reread/hash the plan/review; verify fresh host/container roots, the preserved Step 7 venv,
  source component roots and idle GPU; build and hash `input-v1.tar.gz` locally.
- Copy the archive to a fresh host root, verify/extract it, copy it into the fresh container
  root and rerun `sha256sum -c` plus `bash -n` at every hop.
- Run `build_fill_level_revision2_aggregate.py` against the two preserved component roots;
  compare the generated report to the committed combined-summary hash; enumerate its 26
  links; run the full train/validation discovery and 7,680-file preflight.
- Re-read/hash the newest review, confirm fresh state/run/log paths and launch the detached
  smoke-then-full runner. Poll append-only state/progress, epoch metrics, processes and GPU;
  independently validate each summary, state sequence, population, holdout counters,
  scheduler, augmentation contract and artifact hashes.
- Copy only inputs, aggregate link/preflight/environment evidence, smoke/full outputs,
  weights, logs and state container -> fresh host evidence -> fresh ignored local evidence;
  create and verify the same `SHA256SUMS` manifest at both destinations.

### Problems

- After aggregate creation and exact report-hash verification, a trailing read-only shell
  check quoted its container path incorrectly. Brev retried the command; the aggregate
  builder correctly refused the existing root, so no evidence was changed. A corrected
  read-only structural/preflight check then passed.
- The first post-training read-only assertion used stale shorthand names for the augmentation
  keys. It failed before changing data and Brev repeated the same read-only failure. The
  corrected assertion used the recorded `mask_guided_liquid_recolour` and
  `mask_guided_liquid_darkening` keys and all training gates passed.

## 2026-11-01 — Step 8 Revision 2: exact-v4 batch completion and combined preflight

### Result

- **Source-file observation:** immediately before the recovery launch, Codex read the
  newest main-checkout review at SHA-256
  `265ab61a9a1532328f72627d168e91b178fbb49d6ce52aac07b2eb4e413c4645`.
  Its newest entry keeps Step 8 GO, confirms the hidden-rule design, defers the approved
  calibration-only pour-timing ledger until after Step 10 if time allows, and says to finish
  the batch, retrain, validate and run one calibration. No STOP was present.
- Validation chunk `1101000-1101059` attempt 01 rendered all 60 records, then the renderer
  returned 1 at `2026-11-01T20:37:47Z` after 434 wall seconds. Its summary proves the source
  station/camera unchanged, but its integrity gate and the independent checker rejected one
  frame: seed `1101007`, 68.254407 mL. The failed attempt was preserved container -> fresh
  host failure path -> fresh ignored local failure path; all copies match ordered tree
  SHA-256 `15bbf0403915da9cb9c2ae25f469f5c117875eb88c2f0f60b6ddc000330c9cec`.
- **Measured diagnosis:** seed `1101007` authored a two-by-eight-pixel foam projection at
  `[336,225,338,233]`. All 16 pixels are class `blocked` in the saved composed mask; the
  sticker bbox `[281,197,338,254]` covers that projection and blocked precedence is the
  intentional four-class contract. The exact-v4 renderer incorrectly recorded foam
  `visible_in_camera` from the pre-blocker foam pass, while both its final gate and the
  checker interpreted that field against the post-blocker saved mask. This was bookkeeping,
  not a bad render or missing output.
- A red regression reproduced the fully occluded case. The fix derives foam visibility,
  count and bbox from the final composed class mask; focused renderer/checker tests then
  passed 29/29. Materials, geometry, passes, camera, variation, specs, seeds and class
  precedence are unchanged. Fresh immutable `input-v2` uses renderer SHA-256
  `e5340bcaafe4a6d35be65cafe44556f6c72a923af7b046c1ddcb1dcdf967a83b`;
  the unchanged checker remains `a03442a...966`, variation remains `71aea9f8...a182`,
  checksum manifest is `42df1780...a027`, retry runner is `f04af18a...73e1`, retry contract
  is `07962623...23e`, and no-overwrite archive is `80cd1c8e...7a25`. Every input checksum
  passed locally, on the Brev host and inside the container.
- Fresh validation attempt 02 started at `20:54:38Z` and completed at `21:01:57Z` with
  renderer/checker return codes zero: 436.146564821 render seconds / 439 wall seconds, or
  495.246363 frames/hour. Its container, fresh host and fresh ignored local copies match
  ordered tree SHA-256
  `9994c9a45d92f4715fa8ad29c482b987451261c373566a8d09ff0f69a5a045b7`.
  A local checker rerun passes all 60 records. There are zero positive-level/zero-liquid and
  zero zero-empty frames. Seeds `1101007` and `1101028` have enabled but completely occluded
  foam, correctly recorded as not camera-visible; seed `1101007` still has 7,943 liquid,
  8,946 empty and 5,154 blocked pixels, so it is not a training exception.
- Sealed-test chunk `2101000-2101059` then completed at `21:09:18Z` with renderer/checker
  return codes zero: 437.015527203 render seconds / 441 wall seconds, or 494.261614
  frames/hour. Container, fresh host and fresh ignored local copies match ordered tree
  SHA-256 `a4ee2f1f29e2cc222835255436ce9ecdade3c99a54e45d315db2659b5b895ba4`.
  The local checker passes all 60 records with zero positive-level/zero-liquid and zero
  zero-empty frames; no sealed-test RGB was decoded or viewed.
- The completed exact-v4 component summary is integrity-passed for 10 chunks and
  `480/60/60` frames, records validation `attempt-02` and the per-chunk renderer policy,
  inventories zero new old-rule exceptions, and reports zero sealed-test RGB decodes. Its
  SHA-256 is `e224ca0fcf18c0f833d53d1f1a61422a5067470ff85eaa079514ef9da91774ab`.
- The combined summary hash-binds the approved Step 6 summary/review and exact-v4 summary,
  remaps new chunks to `20-29`, and creates exactly 26 train/validation links with no test
  link. It passes the `3,480/360/560` contract; SHA-256 is
  `bfe6650e2d60311411284e4abbf63b71592eb22904b88c50a2da02dfa3ebd6b5`.
  The fresh combined exception review opens only the one approved Step 6 exception,
  seed `100327`, accepts its physical occlusion explanation, excludes nothing and leaves
  `3,480/360` eligible; SHA-256 is
  `e0efefdcd3e603dbb6f88404d8301e20db3d431728b94b19e544cd1c9a189eea`.
- The local combined preflight validates 26 approved chunks, discovers exactly 3,480 train
  and 360 validation samples, hashes 3,840 samples / 7,680 files, and produces ordered ledger
  SHA-256 `680d6d382ed8d55f0d87eb76c5605a290242b95d6e1c2ba5acad32e634204f4c`.
  Its ignored report SHA-256 is
  `f65ddb74467215e81fd4ca48fc24cfd208c71bcc001ae68ed85bc0131a127fe7`.
- Final immutable batch inputs and supervisor ledgers were copied from the container to fresh
  host path `final-evidence-v2-20261101T212407Z`, then to the matching fresh ignored local
  path. Both copies pass the same file manifest; `SHA256SUMS` has SHA-256
  `4bfec8ad521b3857e78d8f641d76f07febea749bdada832d6ed74673314598e6`.
  This evidence contains only `input-v1`, `input-v2`, the initial train-state marker and the
  v1/v2 progress/state ledgers; it contains no rendered frames.
- Independent review found no actionable blocker in the post-occlusion fix, retry-aware
  component summary or combined aggregate builder; its focused verification passed 34
  tests. The complete tracked fill-level suite plus the new aggregate tests passed 251 tests
  with one skip in 202.20 seconds. Python compilation, strict parsing of all three new JSON
  artifacts and `git diff --check` passed. The existing Brev instance remains running for
  training. No instance lifecycle action, push, PR, external message, raw-render commit or
  real-check access occurred.

### Commands run

- Re-read/hash the main review; inspect `progress-v1.tsv`, failed state, manifest count,
  processes and GPU; copy the failed attempt container -> fresh host -> fresh ignored local
  paths and compare ordered tree hashes.
- Rerun the pinned checker on the failed manifest; decode the one failing mask crop; compare
  the raw foam/sticker bboxes and final class IDs; add a red regression, apply the
  post-occlusion metadata fix, and rerun focused tests and diff checks.
- Create fresh `input-v2` and a no-overwrite retry supervisor; run `sha256sum -c` and
  `bash -n`; create/hash `input-v2.tar.gz`; copy it to the existing Brev host, verify and
  extract it, copy it into the container, verify again, then launch the detached retry.
- Poll `state-v2.txt`, `progress-v2.tsv`, attempt manifest counts, processes and GPU. For
  each new COMPLETE attempt, verify stored state/checker, copy container -> fresh host ->
  fresh ignored local, compare ordered tree hashes, rerun the local checker and inventory
  zero-liquid/zero-empty cases.
- Copy `input-v1`, `input-v2`, `first-train-state-v1.txt`, both progress ledgers and both
  state files container -> fresh host final-evidence path -> fresh ignored local path; create
  and verify the same `SHA256SUMS` manifest at both destinations.
- Run `summarize_fill_level_revision2_v4_batch.py` against the ten collected chunks and
  `input-v2`; a second invocation correctly refused to overwrite its completed report.
- Run `build_fill_level_revision2_aggregate.py` with the pinned Step 6 summary/review and
  exact-v4 summary; run `review_fill_level_exceptions.py` on the train/validation-only
  aggregate; validate the combined audit, discover 3,840 train/validation samples, and hash
  all 7,680 selected files. No sealed-test sample was linked or decoded by these steps.

### Problems

- The first sealed-test host-to-local copy command outlived its 30-second client wait. A
  read-only inspection found the fresh destination complete (127 files, COMPLETE state),
  and its ordered tree hash matched container and host exactly, so no copy was repeated.

## 2026-11-01 — Step 8 Revision 2: augmentation contract and exact-v4 chunks 02-03

### Result

- **Source-file observation:** Codex read the newest main-checkout review at SHA-256
  `51e47f4d46d5c632fbde0d3eec42f11f31cd3129994f8130bf3c40f3edd4c4cc`.
  Claude independently checked the first 60-frame analysis and montage, found the visible
  liquid, boundary, labels, glass and holder consistent with v4, wrote **NO STOP**, and
  explicitly kept Step 8 GO. The batch therefore continued without a material change.
- **Later source-file observation:** after commit `a9ba0e5`, the main review advanced to
  SHA-256 `6f498dcfc7d158390e8b96a0a543cf0c32b468088197b3a3ae0610bce5afe9f9`.
  Claude confirmed that the liquid-only recolour, independent dark tail, transform order,
  measured dark-validation S/V evidence and combined `3,480/360` contract match the
  requested specification. Step 8 remains GO; the review also requires the hidden-surface
  and pour-timing status to be explicit in this log before calibration.
- Train chunk `104120-104179` completed at `2026-11-01T19:54:11Z` with renderer/checker
  return codes zero: 430.582916997 render seconds / 434 wall seconds, or 501.645540
  frames/hour. Its container, fresh host and fresh ignored local copies match tree SHA-256
  `79cfa97de6d4485587b1b8b9cc2c9ce897642bc7cc4fa801863cc22e5649d9fe`.
  The corrected checker independently passed all 60 local records; there are zero
  positive-level/zero-liquid and zero-empty exceptions.
- Train chunk `104180-104239` completed at `20:01:29Z` with renderer/checker return codes
  zero: 434.832493881 render seconds / 438 wall seconds, or 496.743006 frames/hour. Its
  container, fresh host and fresh ignored local copies match tree SHA-256
  `e3f97b3fdde26156951d7779c17de9133ec728f75325062541015beabef81756`.
  The corrected checker independently passed all 60 local records; there are again zero
  positive-level/zero-liquid and zero-empty exceptions.
- Train chunk `104240-104299` completed at `20:08:43Z` with renderer/checker return codes
  zero: 430.381178088 render seconds / 434 wall seconds, or 501.880684 frames/hour. Its
  container, fresh host and fresh ignored local copies match tree SHA-256
  `5ff9c3b407c1a7859f011cc38817892c56c47a06a1a72da13e8c1fe44b49c109`.
  The corrected checker independently passed all 60 local records with zero
  positive-level/zero-liquid and zero-empty exceptions.
- Train chunk `104300-104359` completed at `20:16:00Z` with renderer/checker return codes
  zero: 433.545964806 render seconds / 437 wall seconds, or 498.217069 frames/hour. Its
  container, fresh host and fresh ignored local copies match tree SHA-256
  `3c9c506235491d091d45f20e449b5b8154e23d335eb4cae7e9fe9d9b1b27b172`.
  The corrected checker independently passed all 60 local records with zero
  positive-level/zero-liquid and zero-empty exceptions.
- Train chunk `104360-104419` completed at `20:23:20Z` with renderer/checker return codes
  zero: 436.517074807 render seconds / 440 wall seconds, or 494.826004 frames/hour. Its
  container, fresh host and fresh ignored local copies match tree SHA-256
  `73f0ac0b5f99daddd9191f60adac8440cac5e170ae4f8ecd23c59ce7ac80f497`.
  The corrected checker independently passed all 60 local records with zero
  positive-level/zero-liquid and zero-empty exceptions.
- Train chunk `104420-104479` completed at `20:30:33Z` with renderer/checker return codes
  zero: 429.972172078 render seconds / 433 wall seconds, or 502.358092 frames/hour. Its
  container, fresh host and fresh ignored local copies match tree SHA-256
  `8b69521dd95e1d90303f9161350c7edad3f426b70a328b7bc56676cc71f21c21`.
  The corrected checker independently passed all 60 local records with zero
  positive-level/zero-liquid and zero-empty exceptions. The 60-frame validation chunk
  started automatically; the existing Brev instance remains running.
- The frozen training augmentation now recolours liquid-mask pixels on an independent
  probability-0.5 draw. It samples one measured reference hue family (red/brown, olive,
  green/teal or blue), requests S `0.30-0.85`, and carries every source pixel's V through
  the HSV8-to-RGB8 conversion. A separate probability-0.10 draw requests mean liquid V
  `0.05-0.09` while carrying source H/S and relative V texture. Both liquid-only transforms
  run before the existing global jitter, JPEG 4:2:0, blur and noise.
- Config and evidence explicitly distinguish requested floating-point targets from measured
  post-RGB8 HSV. Darkened-validation evaluation now records actual liquid S and V pixel
  histograms plus pixel and per-frame distributions. It also states that quantization and
  round trips can move measured values across requested boundaries.
- The legacy Step 6 role-count contract remains `3,000/300`. A hash-bound aggregate batch
  summary may explicitly declare the combined `3,480/360` train/validation contract; that
  expected/eligible count and its source are recorded in training config and summary and
  revalidated by the evaluator. The current Revision 1 comparison CLI still binds both
  artifacts to one audit, so it must not be used for a cross-population old-vs-new comparison
  without a later explicit design.
- **Pour-timing status:** the separate estimate remains safely blocked and no real frame or
  unapproved recipe data was opened. `scripts/stream_sequence.py` materializes all recipe
  rows and derives prior-tank data before applying `--vials`; the approved calibration
  ledger contains only settled post-pour windows; and vial 1 starts at frame zero without
  the detector's required pre-pour reference, so the current detector excludes it. The
  minimum safe unblock is an approved, hash-bound calibration-only ledger for vials 1 and
  13 with source hash, FPS/timebase, volumes, complete pour bounds and approved reference
  intervals, plus an explicit vial-1 baseline decision. The detector must then accept only
  that ledger rather than reading the full recipe CSV.
- **Hidden-surface status:** the approved validation-only derivation is implemented from
  the 300 Step 6 validation records only: holder band rows `[217,275]`, broad vial columns
  `[279,340)`, a three-liquid-pixel visibility threshold and a 17-row surface-oval margin
  (the 95%-coverage order statistic from 33 unobstructed validation frames). The rule calls
  hidden when liquid is present just below but not just above the holder, widens the holder
  interval by the oval margin, and flags visible liquid near the holder's top edge. A
  label-oracle sanity check—not model performance—improves actual in-band hidden calls from
  45/107 to 107/107 and range containment from 21/107 to 107/107. Focused surface/revision
  evaluator tests pass 27/27. No sealed-test image or real frame was opened; full model
  before/after results still require the retrained checkpoint and combined validation.
- Independent review caught and the implementation fixed two pre-commit rule defects. An
  exact holder-top or top-plus-oval surface now takes edge precedence instead of being
  misclassified hidden. Revised rule artifacts must now use `holder_band_v2` and bind the
  derivation source, validation role/count, validation hash ledger and all four copied
  constants to the same audited validation population; the original artifact path retains
  its legacy rule. Mutation and boundary regressions cover both fixes, and re-review found
  no remaining blocker.
- A dedicated revised-model-only validator now binds the combined audit, revised training
  artifact and frozen rule without requiring the old `3,000/300` model. It evaluates normal
  plus deterministic darkened validation, requires exact normal-rule reproduction, validates
  case counts, reports measured post-RGB8 liquid S/V, applies the normal visible-surface p95
  gate, refuses output reuse and records zero train/test/real access. Run-level regressions
  cover overwrite refusal, frozen-rule mismatch failure evidence and a gate-failed terminal
  result.
- The dedicated exact-v4 batch summarizer is implemented but was not run on the incomplete
  batch. It pins the ten chunks and `480/60/60` role/seed contract plus all immutable input
  hashes; verifies state/checker/stderr/log/spec/summary/manifest/source evidence and every
  declared RGB/mask byte without decoding sealed-test pixels; rejects missing, reused,
  tampered and extra unmanifested outputs; allows only the preserved chunk-00 direct-copy
  layout exception; and inventories old-rule exceptions for train/validation only.
- Independent reviews found no remaining blocker in the hidden rule, revised-only validator
  or summarizer. The final combined surface/training/revision/summarizer suite passes 85
  tests with one skip; Python compilation and `git diff --check` pass.
- Independent review found no blocker. Focused augmentation/evaluator checks passed 35
  tests with one skip; the broader combined implementation review passed 75 tests with one
  skip. The complete tracked fill-level suite passed 215 tests with one skip in 162.69
  seconds. Python compilation and `git diff --check` also passed. The separate untracked
  Step 9 scaffold was not opened, changed or staged.

### Commands run

- Poll `state-v1.txt`, `progress-v1.tsv`, manifest totals, the supervisor/renderer processes
  and `nvidia-smi`; validate each new attempt's `COMPLETE` state, stored checker JSON and
  empty checker stderr.
- For chunks 02-03, run fresh no-overwrite `docker cp` container-to-host transfers, compute
  ordered file-tree SHA-256 digests on both sides, run `brev copy --host` to fresh ignored
  local attempt paths, recompute the digest, rerun
  `check_fill_level_capture_r2_v4_batch.py --manifest .../manifest.jsonl`, and inventory
  every positive-level/zero-liquid and zero-empty record.
- Run `.venv\\Scripts\\python.exe -m pytest` over every tracked
  `tests/test_fill_level*.py`; run Python compilation over the six changed implementation
  and test files; run `git diff --check`. No sealed-test RGB, real frame, push, PR, external
  message or Brev lifecycle action occurred.

## 2026-11-01 — Step 8 Revision 2: exact-v4 600-frame batch and first-chunk checkpoint

### Result

- **Source-file observation:** immediately before staging, transfer and launch, Codex read
  the newest main-checkout review, SHA-256
  `d5e9ebefcda0341c29b24dc6f38de30fe93f3998461833b0293a2c64249d6522`.
  It stops material tuning, requires the exact accepted pilot-v4 look, moves colour and the
  darkest tail to mask-guided training augmentation, authorizes a fresh 480/60/60 batch and
  says to keep rendering unless Claude adds STOP. No newer STOP was present through this
  checkpoint.
- The immutable fresh contract contains 600 unique, disjoint records: train seeds
  `104000-104479`, validation seeds `1101000-1101059`, and sealed-test seeds
  `2101000-2101059`, in ten retry-safe 60-frame chunks. Every record uses
  `real-look-v2`; the spec-tree SHA-256 is
  `44442cd6529baf0445643bc09ac226ea924df4dc9868ea7716f6033aba76327f`.
- Inputs are the exact archived pilot-v4 renderer
  `8e23ece14ed5c58599981c59589d4c9f0484703fc629ad760333e0d6f09025cd`,
  variation module
  `71aea9f84e7b6c84e7ed807d3784891715e3ddbeac5dea8f17ed291d9f52a182`,
  corrected checker
  `a03442a7da4cde31dee084ffbe3375f0574b370a78829c01976d77f49246e966`,
  and camera match
  `aeee0c5ef7e5b8b88678356f30de4d14a3126da70e4eb5c84711c674bfafc2f8`.
  V4 does not author a replacement glass material, so the later explicit F0 contract is
  correctly recorded as inapplicable and the source-stage glass remains unchanged.
- All 18 staged files passed their checksum manifest locally, on the Brev host and inside
  the existing container. The no-overwrite input archive SHA-256 is
  `2fa3bf2fe8739cd6ad9ccdf33f007441677e6aad54b0c879ecaa153e1c89d531`;
  the runner SHA-256 is
  `6f746b34a70500f171916f6bfd1285dadb34ddeed5629a8e4632b8802ae86c4d`.
- The detached batch started at `2026-11-01T19:32:23Z`. First train chunk
  `104000-104059` completed at `19:39:41Z` with renderer/checker return codes zero:
  435.328930036 render seconds / 438 wall seconds, or 496.176535 frames/hour. Its complete
  attempt was copied container -> fresh host -> fresh ignored local evidence. All three
  copies match tree SHA-256
  `dbaff05ae916b0cd9b4276d39c537ee1d92c421a3239aeb002dcae9f5499a9dc`.
  A fresh local checker rerun passed all 60 records; there are zero positive-level frames
  with zero liquid pixels and zero frames with zero empty pixels.
- The explicit `v4-look-60` analyzer independently re-ran the pinned checker, verified the
  renderer/variation/spec/camera hashes, unchanged source hashes and unchanged-v4-glass
  contract, and opened synthetic train frames only. Of 60 eligible frames, rendered liquid
  V has min/median/p95/max `0.098528/0.131792/0.177033/0.256814`; 58/60 fall in the
  review's approximate v4 range `0.09-0.19`, with two above and none below. Rendered S has
  min/median/p95/max `0.020833/0.089790/0.273639/0.292743`; the absent `S=0.30-0.85`
  and `V=0.05-0.09` strata are explicitly non-gating because the active review moves them
  to training augmentation.
- **Qualitative observation:** the 20-frame synthetic montage matches the accepted v4
  traits: wrapped printed/barcode labels, black holder, light near-neutral backdrop, and
  grey/dark surface ovals. It also retains the accepted v4 glass limitation rather than
  silently claiming the later clear-glass contract. The compact report SHA-256 is
  `8a082175e83cdd535baa47131b4a6afda44688b8fede82ca755b667fa82772e3`;
  the montage SHA-256 is
  `cf03da3ea39dbefd11c781e1eca9772064d10b390547cfce54833712929aa2c9`.
- Chunk `104060-104119` then completed at `19:46:57Z`, again with renderer/checker return
  codes zero: 432.285067205 render seconds / 436 wall seconds, or 499.670279 frames/hour.
  Its container/host/local copies match tree SHA-256
  `0dae5615986f34d7d3ee4c9c066a0b24951ad3e840c3f9df290cd35e69e4156e`;
  its fresh local checker passes all 60 records with zero positive-level/zero-liquid and
  zero-empty exceptions. Chunk 02 started automatically and Brev remains running.
- In parallel, mask-guided training recolour/darkening is implemented but remains outside
  this first-chunk checkpoint while measured post-transform HSV reporting and combined
  3,480/360 population support are completed. The hidden-surface rule remains at its old
  heuristic pending synthetic-validation derivation. Pour timing remains safely blocked:
  the current script reads all recipe rows before vial filtering and the approved ledger
  lacks full-pour windows, so no unauthorized real data was accessed.

### Commands run

- Generate the ten specs with the archived v4 module; audit exact roles, seed ranges,
  uniqueness, bands and hue-family coverage; compute the deterministic spec-tree hash;
  run `sha256sum -c INPUT_SHA256SUMS-r2-v4-batch`, `bash -n` and Python compilation.
- Create `input-v1.tar.gz` without generated bytecode, hash it, run
  `brev copy --host ... isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step8-r2-v4-batch-20261101-codex-01/`,
  verify/extract it on the host, `docker cp` it into the fresh container root, and rerun
  the checksum manifest there.
- Re-read/hash the main review; confirm fresh state/progress/lock paths and an idle GPU;
  launch `docker exec -d vscode bash /tmp/fill-level-cnn-step8-r2-v4-batch-20261101-codex-01/input-v1/run-step8-r2-v4-batch-v1.sh`.
- Poll progress/state/manifest counts/processes/GPU; for each completed chunk, validate the
  stored checker, copy to a fresh host path, compare the container/host tree digest, copy
  to a fresh ignored local path, compare the local digest, rerun the checker, and inventory
  the zero-liquid/zero-empty exceptions.
- Run the new analyzer with `--analysis-mode v4-look-60` and the exact expected hashes,
  write fresh compact report/montage paths, inspect the montage, and run its focused tests
  and compilation/diff checks. No sealed-test RGB, real-check frame, push, PR, external
  message or instance lifecycle action occurred.

### Problems

- One remote command lost its SSH connection after host extraction and before creating the
  container root. Read-only inspection proved the host copy intact and the container root
  absent; the retry performed only the missing fresh container copy and then hash-verified
  it.
- The first host-to-local copy placed chunk-00 attempt contents directly in the requested
  fresh chunk directory rather than adding an `attempt-01` child. No file was overwritten;
  its complete tree still matches the container/host digest. Later copies use explicit
  fresh attempt directories.
- The first real v4 analyzer run failed before writing any artifact because its synthetic
  fixture had incorrectly included the newer `saturation_band`. The mode-specific decoder
  now requires that field to be absent for exact v4, records `not_authored_v4`, and leaves
  the default 100-frame/v5 contract strict. The corrected focused suite passes 11 tests,
  and the real 60-frame analysis then completed without overwriting any evidence.

## 2026-11-01 — Step 8 Revision 2 v5: batch stopped; glass contract fixed; pilot still fails

### Result

- **Source-file observation:** Codex read the newest main-checkout review, SHA-256
  `ceb71826...fd8e`, before stopping the batch, before launching v5, and again before this
  checkpoint. It says the first train chunk is unusable, orders the batch stopped, requests
  one fresh 20-frame v5 pilot, and requires work to stop for Claude after committing its
  code, montage and analysis.
- The batch supervisor was stopped at `2026-11-01T18:30:32Z`; the already-started chunk 03
  renderer was stopped at `18:31:09Z`. Chunks 00-02 had completed with renderer/checker
  return codes zero; chunk 02 finished 100 frames in 687.798045 render seconds / 692 wall
  seconds (523.409455 frames/hour). Chunk 03 stopped after seven RGB/manifest records. No
  later chunk started. All completed and partial batch frames are explicitly excluded from
  training, validation and test.
- Every diagnostic was preserved without overwrite. Chunk 02 was copied container -> fresh
  host -> fresh ignored local evidence and rechecked; its recorded tree digest is
  `ad037...`. The 17-file stopped chunk-03 partial has tree SHA-256
  `8a666285...fc33c5c`, and the stopped-control evidence has ordered digest `091e8...`.
- **Measured root cause:** the shell and liquid were still bound to the intended distinct
  prims. The broken shell instead authored `specularColor=(0.96,0.96,0.96)` while using the
  specular workflow. That field is F0, not transmission or IOR. For IOR 1.46, the dielectric
  relation `((ior - 1) / (ior + 1)) ** 2` gives `0.034965959415691715`; the old value was
  about 27 times too large. A regression test first failed against 0.96, then the renderer,
  checker and tests were updated to require the physical F0 and reject the old value.
- Fresh v5 input pinned renderer SHA-256 `a003cfc...cb75`, variation SHA-256
  `a2d27a3...dc1b`, specs SHA-256 `4619774...e46a8`, runner SHA-256
  `f44e4fc...398f7`, input-manifest SHA-256 `5cf71fd...66af2`, and archive SHA-256
  `6a5cb51...93a0`. The input was hash-checked locally, on the Brev host and inside the
  existing container before launch.
- The detached v5 pilot ran from `2026-11-01T18:46:49Z` to `18:49:37Z` and produced all 20
  planned seed/level pairs. Renderer and corrected checker both returned zero. The complete
  59-file evidence was copied container -> fresh host -> fresh ignored local paths; all
  three copies match tree SHA-256 `c58b6d21...1809b`. A fresh local checker rerun again
  passed all 20 records.
- **Measured v5 failure:** the analysis status is `failed_automated_gate`. All 19 non-empty
  liquids miss the required rendered saturation stratum: observed S is
  `0.016613-0.142905`, with 0/19 in inclusive `0.30-0.85` instead of 9-10. Rendered V is
  `0.179890-0.259315`; 0/19 fall in `0.05-0.09` instead of the required two. The explicit
  dark-liquid/holder relation passes only 5/7 cases (seeds 28002 and 28020 fail). Integrity,
  the source/hash contract, corrected F0 contract and independent capture checker pass.
- **Qualitative observation:** v5 keeps the black holder and wrapped barcode label. The
  corrected glass is less solid than the broken batch, but it still looks milky grey; the
  backdrop is not convincingly visible through the vial body, and liquid boundaries are
  only faint greyscale bands in close crops and generally invisible at scene scale. V5
  therefore also misses the review's visual glass/liquid requirement.
- The compact v5 report SHA-256 is `2b8838b9...32de`; its synthetic-only montage is
  `9f68bad8...cdd9`. The ignored real/synthetic review montage is
  `dae7361f...868b`. Raw renders and masks remain ignored and out of Git.
- The focused Revision 2 renderer/analyzer suite passed 107 tests in 138.50 seconds and
  compilation passed. No v5 retry, batch restart, training, test-render inspection or
  real-check inspection occurred. The existing Brev instance remains running and idle for
  Claude's checkpoint decision; no instance action, push, PR or external message occurred.

### Commands run

- Read and hash the main review; inspect the V3 progress/state, runner and renderer process
  trees; stop only those batch processes; preserve and hash the completed/partial evidence.
- Diff the v4 and batch materials, verify shell/liquid bindings from manifests and the
  exported session layer, calculate the dielectric F0, and run the new glass contract test
  red then green.
- Build a fresh no-overwrite v5 input, verify every staged hash locally/host/container,
  launch the 20-frame pilot detached, and monitor its state, manifest count, process and GPU.
- Copy completed v5 evidence to fresh host/local destinations, compare file counts and
  ordered hashes, rerun the corrected checker, generate the compact analyses/montages, and
  inspect both montages side by side.
- Run the focused 107-test Revision 2 suite and Python compilation checks, inspect the final
  patch boundary, and keep the pre-existing Step 9 scaffold excluded.

### Problems

- The F0 fix addressed a real glass-material defect but did not make the requested liquid
  colours or dark tail visible. The remaining transfer is not safe to infer from material
  inputs: physically varied inputs still collapse to pale, nearly achromatic rendered
  pixels. Work stops here for Claude rather than guessing another material change.
- A local staging command used unsupported `New-Item -LiteralPath`; it failed before copying
  code, leaving only newly generated specs. Staging then completed into that same fresh
  input without overwriting any file. A remote verification command also had a quoting
  fault after the host copy had completed; its retry correctly refused the existing path,
  and read-only count/hash verification proved the first copy intact.
- The first host-to-local v5 copy attempt hit WSL socket-buffer exhaustion before transfer.
  Codex identified 333 orphaned `wslhost` helpers, terminated only those orphans, waited for
  WSL recovery, and then completed the fresh copy. No Brev process or evidence was changed.

## 2026-11-01 — Step 8 Revision 2 batch: first train chunk exposes failed look tweaks

### Result

- **Source-file observation:** immediately before launch and again before collecting the
  first completed train chunk, Codex read the newest main-checkout review, SHA-256
  `5e449b7d...377f`. It marks pilot r2-v4 PASS, authorizes the 1,000-frame Revision 2 batch,
  requires the first train chunk to be summarized without pausing the batch, and says Claude
  will write STOP only if a requested tweak failed. No newer STOP entry is present yet.
- The immutable 10-chunk contract is 800 train seeds `103000-103799`, 100 validation seeds
  `1100300-1100399`, and 100 sealed-test seeds `2100500-2100599`. The ranges are disjoint
  from Step 6, contain 1,000 unique role/seed identities, and have exactly 500 saturated and
  500 subdued material requests. The specs-tree SHA-256 is `7da994c4...3e64`.
- The first V2 launch failed before rendering any frame because the new clear-glass material
  tried to fetch a `metallic` input that the material helper had not authored. Attempt 01,
  its logs and state remain untouched in the container and fresh host failure path; the host
  evidence tree hashes to `4fc86830...a763`. The exact fix creates the input before setting
  it and is covered by a static regression test.
- Corrected input V3 was hash-checked locally, on the Brev host and in the existing container.
  The archive hashes to `a51776be...734`; the renderer is `93fd1e6c...f8a`, the variation
  module is `a2d27a3b...dc1b`, and the retry-safe runner is `33f244d3...6d34c`. The corrected
  supervisor started at `2026-11-01T17:55:05Z`, preserving attempt numbering and all V2
  failure evidence.
- **Measured first-chunk render result:** train seeds `103000-103099` completed at
  `2026-11-01T18:06:41Z`. Renderer/checker return codes were both zero; 100 frames took
  691.365592 render seconds / 696 wall seconds, or 520.708586 frames/hour. The stored checker
  and a fresh local rerun of the corrected checker both passed all 100 records with zero
  errors.
- The complete successful attempt was copied container -> fresh host -> fresh ignored local
  path without overwriting any evidence. All 207 files match the tree SHA-256
  `b10dc8e2...eb9ba` at all three locations. The chunk has no positive-level frame with zero
  liquid pixels and no frame with zero empty pixels.
- The train-only analyzer independently re-ran the corrected capture checker, pinned the
  chunk spec/renderer/variation hashes, re-hashed every RGB/mask and the exported session
  layer, validated unchanged source hashes and the full clear-glass summary contract, and
  opened no sealed-test or real frame. Its focused analyzer/capture-checker suite passed 19
  tests and compilation passed.
- **Measured blocking look failure:** all 100 rendered liquids are nearly neutral and too
  bright. Mean per-frame liquid saturation spans only 0.0196-0.0416, so 0/100 frames land
  in the requested `S=0.30-0.85` band. Mean value spans 0.2035-0.2783, so 0/100 land in the
  requested `V=0.05-0.09` tail. This fails two of the three direct-batch tweaks despite the
  material inputs containing the intended saturated/dark strata.
- The synthetic-only montage visually confirms the failure: saturated and subdued requests
  alike appear as opaque grey tubes, and the bright backdrop does not show through the vial.
  The clear-glass tweak therefore also failed in appearance. The compact report SHA-256 is
  `87a17165...41d`; the montage SHA-256 is `2300db78...a14`.
- **Measured root cause:** the exported session layer binds the shell and liquid to distinct,
  correct prims, but `ClearVialGlass` uses the specular workflow with
  `specularColor=(0.96,0.96,0.96)`. That value is normal-incidence reflectance in this
  workflow; IOR 1.46 glass has dielectric F0 about 0.035. The 27-times-too-high F0 leaves a
  low-roughness near-mirror shell even when diffuse opacity is 0.14, visually washing out the
  colored liquid. Saturated/subdued rendered mean S is 0.03115/0.03117 and dark/extreme-dark
  rendered mean V is 0.2233/0.2289, independently ruling out the intended strata.
- Train chunk 01 also completed with renderer/checker return codes zero at 524.885320
  frames/hour. Its 207-file container/host/local copies match tree SHA-256
  `61ff290f...89f5`; the local checker again passed all 100 records, with no positive-level
  zero-liquid or zero-empty exceptions. Chunk 02 then started automatically.
- In accordance with the review's keep-going instruction, chunk 01 started automatically
  while the first chunk was copied and analyzed. Brev remains running. No instance action,
  push, PR, external message, sealed-test RGB inspection or real-frame access occurred.

### Commands run

- Read and hash the newest main review; audit the 1,000-record spec contract and all immutable
  V2/V3 input hashes locally, on the Brev host and inside the existing container.
- Launch the retry-safe V2 supervisor, preserve its pre-render failure, apply the one-line
  metallic-input fix, build a fresh V3 archive and launch only the failed first chunk as
  attempt 02 before continuing the untouched remaining chunks.
- Poll progress/state/process/GPU evidence; confirm the first-chunk stored checker; copy its
  completed attempt to fresh host/local paths; compare file counts and tree hashes; rerun the
  corrected checker locally.
- Run the hardened train-only first-chunk analyzer with exact seed and input hashes, write the
  compact JSON/montage without overwrite, inspect the montage, and run the focused tests and
  compilation checks.

### Problems

- The first launch's missing material input was a renderer integration bug, not a data or
  transport failure. Its zero-frame evidence is preserved and the corrected retry completed.
- More importantly, the requested saturation, darkest-tail and clear-glass appearance tweaks
  all failed in the first 100-frame train chunk. The batch remains in progress only because
  the active review explicitly says to keep going and reserves STOP for Claude's follow-up.
  No training will use these frames unless a later review accepts or replaces them.

## 2026-11-01 — Results-draft review additions while stopped

### Result

- **Source-file observation:** the newest main-checkout review, SHA-256
  `02cd79ec...f07ef`, approves commit `96f54d6` as accurate and well labelled. It keeps the
  project STOPped but permits two documentation-only additions: the measured darkness gap
  and a proposal section describing likely improvements.
- Added the review's measured brightness comparison to the results draft. In the vial-centre
  columns, real liquid mean HSV `V` was 0.155 for vial 1 and 0.088 for vial 13, versus 0.184
  for the holder. Rendered validation liquid had 5th-percentile/median `V` of 0.26/0.56 and
  none below 0.15. The note labels this as a Claude-review measurement rather than a new run.
- Added a clearly labelled proposal section covering physically dark translucent renders,
  realistic large labels, unlabelled-real appearance adaptation and a possible small
  hand-labelled real set. The last option explicitly requires a policy change and the user
  and Liam's approval.
- **Integrity:** this was a docs-only review fix. No model, rule, render or inference changed;
  no test-render manifest/image or real-check vial was opened; Step 9 remains unstarted; the
  existing untracked Step 9 scaffold remains untouched; and Brev stays running and idle.

### Commands run

- Read and hash the newest main review; inspect the existing commit and stopped Brev state.
- Apply only the two authorized prose additions, run link/diff/document checks, independently
  review the patch and commit only the results note and this newest-first log entry.

### Problems

- The measured real/rendered darkness gap explains the observed failure but does not itself
  validate any proposed remedy. All four future directions remain proposals.
- The next experimental action still requires the user's decision; no Step 9 work is
  authorized.

## 2026-11-01 — Steps 1-8 results-note draft while stopped

### Result

- **Source-file observation:** the newest main-checkout review, SHA-256
  `2edc16a4...eb6e7`, keeps the project STOPped for the user's decision. It permits one
  documentation-only action while stopped: draft the Step 12 results note through Steps 1-8
  with evidence labels, both calibration failures, the ignored qualitative overlays and
  disclosure of the single revision. It forbids starting Step 9 or opening test or
  real-check data.
- Added `docs/fill-level-cnn-results.md` as a draft through Step 8. It separates measured
  synthetic results, source-file observations, qualitative diagnostics, interpretations,
  limitations and unchecked holdouts. It states prominently that this is not a final result.
- The draft records the 3,800-frame checked synthetic batch, synthetic-only training,
  original/revised validation metrics, both failed real calibration attempts and the one
  disclosed revision informed by calibration vials 1 and 13.
- The corrected diagnostic interpretation is explicit: Revision 1 recovers some vial
  structure but still labels much of vial 13's dark liquid as blocked, vial 1's BRADY label
  as foam and stray upper vial-1 regions as liquid. The evidence therefore supports a CNN
  real-domain transfer failure, not merely a surface-rule failure.
- **Integrity:** this documentation pass opened no test-render manifest/image and no
  real-check vial, changed no model or rule, ran no inference or acceptance evaluation and
  left the three untracked Step 9 scaffold files untouched. Raw data, weights and diagnostic
  artifacts remain ignored. Brev remains running and idle.
- The note concludes that no ruler or physical-accuracy result exists, the 500 test renders
  and 27 real-check vials remain unscored, and work remains STOPped before Step 9.

### Commands run

- Read and hash the newest main review; reread `AGENTS.md`, the project-writing workflow,
  the Step 12 plan boundary and the existing public evidence-note convention.
- Read only committed compact summaries and the implementation log; calculate their
  SHA-256 values without opening raw test or real-check data.
- Inspect Step 6 completion, active processes and GPU on the existing Brev instance.
- Draft the evidence-labeled note, run documentation and diff checks, independently review
  the result and commit only the draft/log changes.

### Problems

- The six-frame overlays invalidate the earlier optimistic inference that a rule-only change
  would likely solve the transfer failure. They show material segmentation errors inside the
  vial. The draft records that correction rather than presenting synthetic metrics as proof
  of real accuracy.
- The next experimental action requires the user's decision. No Step 9 work is authorized.

## 2026-11-01 — Step 8 Revision 1: synthetic gate passed; one real calibration failed

### Result

- **Source-file observation:** Codex reread `AGENTS.md`, the build plan and the newest
  main-checkout review, SHA-256 `a7ce1da7...de70`. The review authorized exactly one
  revision after the initial calibration failure, then exactly one calibration run on the
  same 98 frames, with a mandatory stop at this Step 8 checkpoint whether it passed or
  failed.
- **Revision disclosure:** the model/rule was revised once after video-5 calibration vials 1
  and 13 exposed two transfer gaps: the old search included rows above the synthetic vial
  outline and selected the real dark nozzle, while the real liquids were darker than the
  rendered training distribution. No other real frames informed the revision.
- **Synthetic-only rule change:** the intersection of all 300 validation label outlines
  produced the full-frame vial-interior interval `[83,427)`. Its ordered derivation ledger is
  `8969e972...2066dd`. The frozen center columns remain `[298,316)`, with minimum liquid and
  blocked row counts 5 and 1.
- **Synthetic-only training change:** train-only mask-guided liquid recoloring was applied
  with probability 0.5, V uniform in `[0.03,0.35]`, full-range hue/saturation and retained
  local texture. Global brightness, contrast, gamma, saturation and white-balance jitter was
  added; JPEG 4:2:0, blur and noise remained. The Step 7 settings were retained with cosine
  learning-rate decay from 0.001 to 0.00001 over 30 epochs.
- Full Revision 1 training used exactly 3,000 synthetic train and 300 validation frames,
  selected on validation only and opened no test or real frame. Best epoch 27 reached
  validation macro IoU 0.965030; `best-epoch-027.pt` is ignored and hashes to
  `51f4e3cf...f0197`. The ignored training summary is `2c877eff...99336`.
- **Synthetic gate passed:** on normal validation, original/revised visible-surface p95 was
  2.0/2.55 rows and macro IoU was 0.970578/0.965033. The revised normal p95 therefore stayed
  below the required 3-row ceiling. On the fixed-seed darkened validation copy, p95 improved
  from 14.8 to 2.0 rows and macro IoU from 0.964944 to 0.978789. The compact comparison is
  `b5f546ad...60a0`.
- The revised validation rule returned 210 visible, 89 hidden and 1 abstain across all 300
  frames. Visible error was median 1, p95 2.55, maximum 59 and mean 2.052 rows. Hidden
  geometry fell inside 25 predicted ranges (8.33% of all frames; 28.09% conditional on a
  hidden prediction). These weak hidden results are recorded without qualification.
- **Exactly one permitted real calibration run failed.** Vial 1 returned 23 visible and 5
  hidden frames; its visible median was row 99 against target 329 (outside +/-4 rows). Vial
  13 returned 0 visible and 70 hidden frames, so the calibrator correctly stopped with
  `calibration needs at least 3 visible rows; found 0` before fitting. The required ordered,
  in-tolerance pair was unavailable; no ruler was created and no physical accuracy is
  claimed.
- **Evidence preservation:** the complete fresh input, smoke, training, validation,
  comparison, failed calibration, logs and all ignored weights were copied container -> a
  fresh host collected path -> a fresh ignored local path. Both checksum passes succeeded;
  the 69-file `SHA256SUMS` hashes to `55af0b50...dfff`. The failed state is
  `dacf43fb...be14` and the 98-frame ledger is `6ad4d5fb...ffe1`.
- **Integrity:** no second revision or calibration run was made. The 500 synthetic test
  renders and 27 real-check vials remain unopened; Step 9 was not started. Raw frames and
  weights remain ignored. Brev `isaac-launchable-d70fec` remains running, and no instance
  action, push, PR or external message was made.
- The focused Revision 1 suite passed 60 tests with one Torch-only skip. The full suite had
  seven known Windows-only fake-executable/symlink-privilege failures, 195 passes and two
  skips. Independent reviews of the row rule, training changes and
  comparison evaluator found no remaining blocking defect after fixes.
- Work stops at the Step 8 failure checkpoint pending a newer explicit GO. The committed
  compact record is `assets/fill-level/step8-revision1-failure-summary.json`; all large
  evidence remains ignored.

### Commands run

- Read and hash the newest main review; reread `AGENTS.md` and the Step 8/9 plan boundary.
- Derive the `[83,427)` vial-interior rows from synthetic validation labels and run the
  focused rule/training/evaluator tests.
- Build and hash a fresh exact-input archive; copy it to the existing Brev host/container.
- Run one smoke, one 30-epoch full training, one revised validation evaluation and one
  original/revised normal/darkened validation comparison.
- Confirm the revised normal-validation p95 gate before opening the permitted calibration
  frames.
- Run exactly one Revision 1 calibration attempt with automatic retry disabled; inspect its
  preserved state and 98-frame ledger without changing the model or rule.
- Copy all evidence and weights to fresh host/local paths and verify every payload against
  `SHA256SUMS` at both destinations.
- Run strict JSON, compilation, focused/full tests, diff, ignore and commit-scope checks.

### Problems

- The first real calibration showed that the original rule searched above the vial and that
  the real liquids were outside the rendered darkness distribution. Revision 1 addressed
  those two disclosed gaps once, as authorized.
- Revision 1 passed the synthetic p95 gate but still failed the real acceptance gate:
  vial 1 produced a grossly high visible surface and vial 13 produced no visible surface at
  all. The failure is substantive, not a transport or transient-run failure.
- The review forbids another revision or calibration run. The failure is preserved and work
  stops before Step 9.

## 2026-11-01 — Step 8: validation rule frozen; real calibration failed

### Result

- **Source-file observation:** before Step 8 and again before the calibration run, Codex read
  the newest main-checkout review, SHA-256 34b3e05e...60e9b. It authorizes Step 8, requires
  the surface rule to be frozen from synthetic validation before any real calibration frame,
  and orders the frozen CNN/rule to be run on vials 1 and 13 before Step 9.
- **Measured validation-only freeze:** all 300 approved validation renders were evaluated with
  zero train, test or real image access. The selected frozen rule uses crop
  (256,48,368,464), full-frame columns [298,316), at least 5 liquid pixels per row and at
  least 1 blocked pixel per row. The compact artifact is
  assets/fill-level/step8-validation-surface-summary.json, SHA-256
  797a2bb8...7e12; its ignored full summary is 5afe39c9...462d.
- The frozen validation result has 215 visible predictions: median absolute surface error
  1 row / 0.329 mL, p95 2 rows / 0.658 mL and maximum 60 rows / 19.737 mL. In the 72-frame
  true-hidden cohort, the rule predicted hidden for 38 (52.78%); the true geometry row was
  inside the predicted range for 21/72 (29.17%), or 21/38 (55.26%) conditional on predicting
  hidden. These weak hidden results are recorded without qualification.
- The threshold sweep remained synthetic-validation-only. Minimum-liquid threshold 5 was
  selected because it had the lowest p95 tier (2 rows) without the much larger maximum error
  at threshold 4; the column band and hidden rule were not changed after real frames opened.
- **Measured real-calibration input:** the exact source video SHA-256 is
  6f87cec1...5b9 and committed windows CSV SHA-256 is 19174a4b...229. Every frame in both
  settled inclusive windows was extracted losslessly at 640x480: vial 1 frames 816-843
  (28 frames, 30 mL) and vial 13 frames 19700-19769 (70 frames, 80 mL). The ignored
  index.json is SHA-256 73181d39...f9b2 and binds all 98 PNG hashes.
- Exact code, frozen rule, frame index and best-epoch-028.pt
  (SHA-256 86e89644...ddbeb) were copied to a fresh Brev host/container root and hash-checked
  before inference. The unchanged Step 7 CUDA environment and frozen settings were used.
- **Measured blocking failure:** all 28 vial-1 frames and all 70 vial-13 frames returned
  status visible at full-frame row 48, the first row of the crop. The affine-ruler guard
  correctly rejected equal medians with “30 mL row must be below the 80 mL row.” No ruler
  was created and no physical accuracy is claimed.
- **Measured diagnosis:** extraction, dimensions, frame windows, preprocessing, crop, code,
  checkpoint and hashes all match. On every one of the 98 frames, all 18 center-strip pixels
  at row 48 are classified liquid, so every validation-swept liquid threshold 1-18 still
  selects row 48. At the expected 30 mL row 329, vial-1 liquid counts range 0-18 with median
  9; at the expected 80 mL row 177, vial-13 has zero liquid pixels and median 18 blocked
  pixels. This is a real-domain model/rule transfer failure, not a transient run failure.
- Changing columns, adding a vertical ROI, changing thresholds, connected components or the
  top-row rule now would tune on the real calibration images and violate the explicit
  validation-only freeze. Step 9 was therefore not started and no test manifest or image was
  opened. The interrupted, uncommitted Step 9 scaffolding remains excluded from this commit.
- **Evidence preservation:** the failed RUNNING/FAILED state and all 98 per-frame records were
  copied container -> fresh host collected path -> fresh ignored local path. The local
  SHA256SUMS check passed; SHA256SUMS is d2bd8014...bb2c4,
  calibration-frames.jsonl is 7a049a78...bb4a and state.jsonl is 0dbcf04d...8340.
- **Verification:** the focused Step 7/8 suite passed 44 tests with one Torch-only local skip.
  Two independent reviews found no remaining P1/P2 in the validation, extraction and ruler
  code before the real run. The post-failure review concluded the guard and stop decision are
  correct.
- Work stops before Step 9 pending a newer review or explicit user decision. Brev instance
  isaac-launchable-d70fec remains running; no instance was created, reset, stopped or deleted,
  no evidence was deleted or overwritten, and no push, PR or external message was made.

### Commands run

- Read and hash the newest main-checkout review; inspect AGENTS.md and the plan.
- Run the focused validation/extraction/ruler/training test suite (44 passed, 1 skipped).
- Extract all committed calibration-window frames with the exact video SHA-256 gate.
- Build and verify fresh transfer bundles; copy them to a new Brev host/container root.
- Verify code, rule, index, windows and weight SHA-256 values in the container.
- Run the frozen calibration once in fresh calibration-attempt-01.
- Inspect the preserved failure ledger/per-frame output and run read-only class-row
  diagnostics; do not alter the frozen rule.
- Copy the failed attempt to fresh host/local evidence paths and verify SHA256SUMS.
- Run strict JSON, compilation, focused tests, diff and ignore checks before commit.

### Problems

- Earlier Step 8 validation setup preserved several non-data failures: a PowerShell reserved
  Host-variable collision sent input to a literal host-object path; one fresh root included
  its checksum manifest in itself; and one threshold-2 attempt exhausted shared memory before
  a successful workers=0 attempt. None opened test or real data.
- A quoting fault created /home/ubuntu/SHA256SUMS (SHA-256 1346e891...a9) while attempting
  an evidence checksum. It is preserved and was not used. The correct validation evidence
  was copied through a fresh path and verified.
- The local extraction completed correctly, but its wrapper then looked for the wrong filename
  calibration-frames-index.json instead of index.json. The fresh output was preserved and
  verified in place; extraction was not repeated or overwritten.
- Brev automatically retried the nonzero calibration command; the retry refused to reuse the
  already-created attempt directory. Both errors refer to the same preserved failed attempt.
- The first host-to-local failed-evidence copy could not create a missing local parent. After
  creating only that fresh parent, the same untouched host evidence copied and verified.
- The calibration failure is substantive: the frozen synthetic-validation rule cannot
  distinguish the two permitted real calibration levels. Repeating unchanged inference would
  not help, while retuning from these images would invalidate the freeze.


## 2026-11-01 — Step 7: synthetic-only four-class U-Net training checkpoint

### Result

- **Source-file observation:** before starting Step 7, Codex reread `AGENTS.md`, the plan
  and the newest main-checkout review, SHA-256 `6055619c...bca55`. The review authorizes
  Step 7, requires stopping work at this checkpoint, and says to keep
  `isaac-launchable-d70fec` running for later steps and fixes.
- **Measured pre-training exception review:** `scripts/review_fill_level_exceptions.py`
  reconciled every retired-rule exception in the Step 6 summary against the copied manifest,
  decoded mask and source hashes, then rechecked source immutability. Seed 100327 is the only
  exception: train, 0.065117 mL, `full_0_85`, with 0 liquid, 16,850 empty, 1,391 foam,
  3,411 blocked and 285,548 ignored pixels. The sub-mL liquid is far below the visible vial
  base edge, with visible foam and sticker adding occlusion, so its final disposition is
  `accepted_legitimate_occlusion`. No train or validation frame is excluded.
- The full-frame and vial-crop mask-on-RGB overlays were visually inspected under ignored
  `outputs/fill-level-cnn/step7-20261101-codex-01/exception-review/`; their SHA-256 values
  are `85e2515b...84568` and `2c32b58b...0d279`. The committed audit report
  `assets/fill-level/step7-exception-review.json` is `6ef12c4a...6130f`, has status
  `complete`, and declares 3,000 eligible train and 300 eligible validation records.
- Added a synthetic-only training pipeline with an audited-data gate. Before discovery it
  requires the completed exception review and exact Step 6 batch-summary hash. Discovery
  skips test chunk names before opening their state or manifest, pins the 17 train/validation
  COMPLETE attempts and manifest hashes from the Step 6 summary, rejects unsafe relative
  paths, and preflight-hashes every selected RGB and mask against its manifest before model
  loading. The full run verified 3,300 samples / 6,600 files; its ordered hash-ledger SHA-256
  is `f92a0d77...2acf8`. Both `test_manifests_opened` and `test_images_loaded` are zero.
- The fixed crop is `(256,48,368,464)`, or 112x416 pixels. It preserves the original row
  scale and is divisible by 16. Training-only photometric augmentation uses JPEG 4:2:0,
  blur and noise without changing mask geometry. Labels are empty 0, liquid 1, foam 2,
  blocked 3 and ignore 255. Class weights are derived only from cropped train masks using
  inverse square-root frequency.
- **Measured model/config:** four-level U-Net, base width 16, 1,942,628 parameters; AdamW,
  learning rate 0.001, weight decay 0.0001, batch 16, AMP, seed 20261101, at most 30 epochs
  and patience 6. Train-only class pixel counts were 32,644,475 empty, 21,075,348 liquid,
  2,070,757 foam and 8,998,620 blocked, producing weights
  `[0.492630, 0.613110, 1.955967, 0.938292]`.
- **Measured diagnostics:** the fresh 12-frame tiny overfit ran 40 epochs in 15.98 s, reduced
  train loss from 1.3643 to 0.1469 and selected diagnostic epoch 31 at macro-IoU 0.8582.
  The fresh two-epoch smoke used 64 train / 32 validation frames, completed in 7.18 s, and
  improved validation macro-IoU from 0.0818 to 0.3316. Both had zero test access and their
  state/progress ledgers and checkpoints are preserved.
- **Measured full training:** fresh `full-attempt-01` completed all 30 epochs in 303.49 s.
  Validation-only selection chose epoch 28 with validation loss 0.026975 and macro-IoU
  0.970575. Per-class IoU is empty 0.990214, liquid 0.988638, foam 0.935690 and blocked
  0.967756. The committed compact summary
  `assets/fill-level/step7-training-summary.json` is SHA-256 `799fc3af...5c88`.
- The selected ignored weight is `best-epoch-028.pt`, SHA-256
  `86e89644...ddbeb`; the final epoch-30 weight is `69b8057b...e36b`; and the append-only
  progress ledger is `b16ff20c...4ffb`. State is an append-only two-line RUNNING/COMPLETE
  ledger. New best checkpoints use unique epoch filenames, so no Brev evidence is overwritten.
- **Measured environment:** fresh isolated venv under container Step 7 root, Python 3.12.13,
  Torch 2.10.0+cu128, torchvision 0.25.0+cu128, CUDA 12.8, Pillow 12.3.0, NumPy 2.5.1,
  typing-extensions 4.12.2, NVIDIA L40S. A persisted CUDA 112x416 Conv2d
  forward/backward smoke passed before training.
- **Evidence preservation:** container outputs were copied to fresh host directory
  `/home/ubuntu/fill-level-cnn/step7-20261101-codex-01/collected/training-evidence-v1-20261101T083003Z`
  and then to the same-named ignored local directory. All 54 payload files, including every
  diagnostic/full checkpoint, log, input, state/progress ledger, package report and GPU
  smoke result, verify against `SHA256SUMS`; that manifest is SHA-256
  `4484861c...fa397`. Weights and diagnostic outputs remain outside Git.
- **Measured local checks:** 19 focused Step 7 tests passed and one Torch-only shape test
  skipped locally. The full suite reached 135 passed / 2 skipped, with only the already
  documented Windows fake-ffmpeg failure and six symlink-privilege setup errors. Python
  compilation, strict JSON parsing and `git diff --check` passed. Independent code review
  returned GO with no blocking finding; it confirmed holdout isolation, audit/provenance
  gates, train-only class weights, validation-only selection and append-only evidence.
- The Brev instance remains running and idle at this checkpoint. Codex did not create, reset,
  stop or delete an instance; did not delete/overwrite Brev evidence; and did not push,
  open a PR or send an external message.

### Commands run

```text
Get-Content AGENTS.md
Get-Content docs\work\fill-level-cnn-plan.md
Get-Content <main checkout>\docs\work\fill-level-cnn-review.md -TotalCount 70
Get-FileHash -Algorithm SHA256 <main checkout>\docs\work\fill-level-cnn-review.md

.venv\Scripts\python.exe scripts\review_fill_level_exceptions.py \
  --summary assets\fill-level\full-batch-summary.json \
  --collected-root outputs\fill-level-cnn\step6-20261031-codex-01\collected \
  --overlay-dir outputs\fill-level-cnn\step7-20261101-codex-01\exception-review \
  --report-out assets\fill-level\step7-exception-review.json

# Fresh Brev environment and exact-input transfer
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  '<create fresh Step 7 host/container roots; create --system-site-packages venv>'
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  '<pip install Pillow/Jinja2/MarkupSafe/packaging/typing-extensions into fresh venv>'
wsl -e /home/loducaj/.local/bin/brev copy \
  <training code and audited JSON> isaac-launchable-d70fec:<fresh host input paths>
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  '<create and verify INPUT_SHA256SUMS; docker cp input to fresh container; verify again>'

# Common CLI inputs: Step 6 runs, committed Step 6 summary, completed exception review,
# CUDA, seed 20261101, base width 16, lr 0.001 and weight decay 0.0001.
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  'docker exec vscode timeout 1800s <venv python> <train script> \
   --output-root <fresh tiny-attempt-01> --profile tiny --workers 0 <common inputs>'
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  'docker exec vscode timeout 1800s <venv python> <train script> \
   --output-root <fresh smoke-attempt-01> --profile smoke --workers 2 <common inputs>'
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  'docker exec -d vscode timeout 10800s <venv python> <train script> \
   --output-root <fresh full-attempt-01> --profile full --workers 4 \
   --epochs 30 --patience 6 --batch-size 16 <common inputs>'
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  '<poll append-only state/progress, process, log and nvidia-smi>'

# Fresh evidence copy, remote and local hash verification
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  '<docker cp input/runs/logs/install evidence to fresh host collected path; \
   create and verify SHA256SUMS>'
wsl -e /home/loducaj/.local/bin/brev copy --host \
  isaac-launchable-d70fec:<fresh training-evidence path>/ \
  <fresh ignored local training-evidence path>
wsl -e bash -lc '<cd ignored local training evidence; sha256sum -c SHA256SUMS>'

.venv\Scripts\python.exe -m pytest \
  tests\test_fill_level_exception_review.py tests\test_fill_level_training.py -q
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m py_compile <new Step 7 Python files>
.venv\Scripts\python.exe -c '<strictly parse Step 7 audit/training summaries>'
git check-ignore -v <overlays> <best weights> <training logs>
git diff --check
```

### Problems

- The Isaac base environment exposes unrelated optional-package gaps in `pip` diagnostics.
  Step 7 used a fresh venv, installed only the required packages, and used targeted
  Torch/CUDA/import and training checks instead of treating global `pip check` as a gate.
- The first dependency resolution selected typing-extensions 4.16.0, while installed
  IsaacLab metadata pins 4.12.2. The fresh venv was adjusted to 4.12.2 before the GPU smoke
  and every training run; both package-install reports are preserved.
- CUDA cross-entropy reports that its 2-D NLL kernel has no deterministic implementation.
  `warn_only=True` preserved the fixed seed and completed training, but bit-for-bit GPU
  reproduction is not claimed. Exact inputs, configuration, software versions, progress and
  weight hashes are recorded instead.
- One monitoring command accidentally allowed the remote host shell to expand a container
  path expression, so that read-only state check failed and Brev retried the SSH command.
  No file changed; the corrected literal-path check confirmed two append-only state entries,
  COMPLETE status, no trainer process and an idle GPU.

### Questions for Claude

- Does the Step 7 synthetic-only checkpoint pass given the completed occlusion audit,
  validation-only selection, zero test access and copied/hash-verified ignored weights?

## 2026-11-01 — Step 6: full 3,800-frame synthetic batch

### Result

- **Source-file observation:** the newest main-checkout review entry at about 00:51 CT,
  SHA-256 `6055619c...bca55`, keeps Step 6 GO, directs Codex to proceed through Step 7 and
  stop at that checkpoint, and says to leave `isaac-launchable-d70fec` running. Codex must
  not write the final Brev-completion line or stop the instance; Claude will stop it only
  after all work through Step 12 and any review fixes is complete.
- **Measured batch result:** all 20 chunks completed at `2026-11-01T07:43:33Z`: 3,000
  train, 300 validation and 500 test frames with exact, disjoint seed ranges
  `100000-102999`, `1100000-1100299` and `2100000-2100499`. Every stored checker passed,
  every successful attempt was copied container -> fresh remote-host path -> fresh ignored
  local path, and the corrected checker passed again locally for all 3,800 records.
- **Measured aggregate integrity:** `scripts/summarize_fill_level_batch.py` rediscovered the
  single COMPLETE attempt for every chunk, re-hashed manifests and specs, reran all 20
  offline checks, decoded every mask, and enforced exact roles/seeds, PathTracing, 32
  subframes, 640x480, frame 686, approved camera/twin hashes and source immutability.
  `assets/fill-level/full-batch-summary.json` reports 20/20 chunks, 3,800/3,800 frames,
  zero integrity errors and SHA-256 `e736d17e...d88c0`.
- **Measured throughput:** successful render time summed to 25,618.800 s, or 533.983
  frames/hour. The final-successful-attempt wall window was 27,754 s (492.90 frames/hour),
  and the full window including preserved failed attempts was 28,570 s (478.82
  frames/hour). Individual final chunk rates ranged from 502.827 to 541.221 frames/hour,
  except the final 100-frame chunk at 503.540 frames/hour.
- **Renderer provenance:** chunks 00-02 used renderer
  `e807f93a...57958`; chunks 03-19 used V4 `0d0f03c2...91717`. V4 changed only the
  acceptance gate: a positive level can be fully occluded, and nonzero frames need not show
  empty pixels. The final checker hash is `a03442a7...e966`, variation module
  `b8e404c5...79898`, runner `baf5a593...d3a68f` and deterministic spec tree
  `f30ada8d...79a7b`.
- **Measured old-rule inventory:** exactly one of 3,800 frames fails the retired visibility
  rule: train seed 100327, level 0.065117 mL, `full_0_85`, with 0 liquid, 16,850 empty,
  1,391 foam and 3,411 blocked pixels. Foam is visibly enabled, the visible sticker accounts
  for 672 blocked pixels, and the sub-mL liquid is below the visible base edge. The compact
  summary deliberately marks its disposition `pending_step7_review`; Step 7 will create
  ignored mask-on-RGB evidence and complete the required explanation before loading train
  or validation data. No frame has zero visible empty pixels.
- **Coverage caveat:** continuous level sampling produced no frame at exactly 0 mL, although
  the empty class is present in all 3,800 masks. This satisfies the approved batch contract
  but means later empty-vial behavior must not be inferred from an exact-zero training case.
- **Evidence preservation:** final supervisor inputs and progress ledgers were copied from
  container `/tmp/fill-level-cnn-step6-20261031-codex-01` to fresh host path
  `/home/ubuntu/fill-level-cnn/step6-20261031-codex-01/final-evidence-v4-20261101T074333Z`
  and then to the same-named ignored local output directory. All 63 payload files verify
  against `SHA256SUMS`; that manifest's SHA-256 is `a581a05e...1cf0d`, and copied
  `state-v4.txt` is `COMPLETE`. Raw RGB, masks, manifests, logs and specs remain ignored.
- Validation now treats `255` as ignore and requires scored vial pixels, requires visible
  empty pixels only at exactly 0 mL, and inventories rather than rejects legitimately hidden
  positive liquid. Direct regression tests cover hidden liquid, all-liquid/no-empty positive
  frames, invalid labels, manifest/mask count reconciliation and the renderer contract.
- The batch summarizer pins the approved camera, source camera and station-tree hashes,
  refuses an inconsistent or uniformly wrong source, requires the chunk-specific renderer
  policy, treats a silent `passed: false` offline check as failure, converts invalid timing
  to JSON `null`, refuses report overwrite and writes strict JSON.
- **Measured local checks:** 42 focused tests passed. The full suite reached 116 passed / 1
  skipped, with the same one Windows fake-ffmpeg failure and six Windows symlink-privilege
  setup errors documented in Step 5; there was no fill-level regression. Python compilation,
  strict JSON parsing and `git diff --check` passed. An independent code/evidence review found the approved-input
  pinning and silent-checker gaps; both are fixed with regressions. A second independent
  completion audit found no remaining Step 6 gate beyond this log, the committed summary and
  final review.
- The existing Brev instance remains running. No instance was created, reset, stopped or
  deleted; no Brev evidence was deleted or overwritten; nothing was pushed and no external
  message or PR was created.

### Commands run

```text
Get-Content <main checkout>\docs\work\fill-level-cnn-review.md -TotalCount 70
Get-FileHash -Algorithm SHA256 <main checkout>\docs\work\fill-level-cnn-review.md
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  '<inspect progress-v4.tsv, state-v4.txt, current attempt, processes and GPU>'

# Per newly complete chunk (fresh paths only; no overwrite)
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  '<confirm state/check; docker cp vscode:<attempt> <fresh host collected path>; verify hashes>'
wsl -e /home/loducaj/.local/bin/brev copy --host \
  isaac-launchable-d70fec:<fresh host collected path>/ <fresh ignored local chunk path>
.venv\Scripts\python.exe scripts\check_fill_level_capture.py --manifest <local manifest>

# Final supervisor evidence
wsl -e /home/loducaj/.local/bin/brev exec isaac-launchable-d70fec \
  '<copy input and progress/state ledgers to fresh final-evidence path; create and verify SHA256SUMS>'
wsl -e /home/loducaj/.local/bin/brev copy --host \
  isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step6-20261031-codex-01/final-evidence-v4-20261101T074333Z/ \
  <ignored local final-evidence-v4-20261101T074333Z>
wsl -e bash -lc '<cd ignored local final evidence; sha256sum -c SHA256SUMS>'

.venv\Scripts\python.exe scripts\summarize_fill_level_batch.py \
  --collected-root outputs\fill-level-cnn\step6-20261031-codex-01\collected \
  --specs-root outputs\fill-level-cnn\step6-20261031-codex-01\input\specs \
  --report-out assets\fill-level\full-batch-summary.json
.venv\Scripts\python.exe -m pytest \
  tests\test_fill_level_batch_summary.py tests\test_fill_level_capture.py \
  tests\test_fill_level_variation.py tests\test_fill_level_renderer_contract.py -q
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m py_compile <changed Python files>
.venv\Scripts\python.exe -c '<strictly parse full-batch-summary.json; reject NaN>'
git check-ignore -v <sample RGB> <sample mask> <final supervisor evidence>
git diff --check
```

### Problems

- Chunk 00 attempt 01 failed before useful work because the first helper called absent
  `/usr/bin/python3`. Attempt 02 stopped after 66 frames because the original checker
  incorrectly required empty pixels in every frame. Both attempts remain preserved.
- V3 chunk 03 attempt 01 rendered all 200 frames, then the old final gate rejected seed
  100327 even though its 0.065117 mL liquid is legitimately hidden by visible foam, sticker
  and the vial base edge. That attempt remains preserved; fresh V4 attempt 02 passed.
- A local WSL socket-resource failure followed hundreds of stale `wslhost` processes.
  `wsl --shutdown` cleared only the local client state; the Brev instance and remote render
  continued untouched. Subsequent serial checks and copies succeeded.
- The final aggregate scan took about two minutes because it re-hashed and decoded 7,600
  files from the ignored OneDrive-backed output tree. It completed with no integrity error.

### Questions for Claude

- None for Step 6. Proceeding directly to the Step 7 old-rule evidence review and synthetic
  training, then stopping at the Step 7 checkpoint as instructed.

## 2026-10-31 — Step 5: 20-frame pilot and checkpoint

### Result

- **Source-file observation:** Claude's newest review entry at about 17:42 CT, SHA-256
  `05fce574...c47ab`, says the Step 4 labels look right, confirms the twin-based holder
  sampling zones, requires matte/darker holder material variation, gives GO for Step 5 and
  says to stop at this checkpoint. The review watcher now uses this SHA as its baseline and
  remains on permanent five-minute checks.
- Added deterministic holder brightness and roughness to every frame spec and manifest row.
  **Measured on the unchanged twin:** the active vial and `carrier_0` share the same XY
  centre; `V3_carrier_0_clamp_21/Mesh_041` is the upper plate at z=1.09435 m, while
  `cart_*` is the lower base near z=0.93281 m. The source `Fill_black_acetal` shader has
  linear diffuse RGB `(0.014, 0.018, 0.018)` and roughness `0.35`. The final session-layer
  range is blue/green brightness `0.002-0.05` with roughness `0.20-0.85`, including matte
  cases and values clearly darker than the default without generating pale holders.
- Added `scripts/summarize_fill_level_pilot.py`. It cryptographically links the render
  summary to the exact manifest, reuses the offline capture checker, aggregates class and
  case coverage, measures held-out rows and horizontal vial position, writes an ignored
  RGB/mask montage, and reports pass-only, startup-adjusted and end-to-end timing separately.
  The capture checker now also reconciles every manifest class count and vial-pixel total
  against the decoded mask.
- **Measured final pilot on Brev attempt 07:** 20/20 frames passed at 640x480 using
  PathTracing and 32 subframes. Organizer station and camera hashes were unchanged. Exact
  hashes: renderer `e807f93a...57958`, variation module `24de6f73...388ab`, specs
  `5015aaa2...afa24`, manifest `2c289a1a...f1bec`, render summary
  `c833a8d5...76121a` and session layer `9d22cd2a...4e5db`. The summary's manifest hash
  matches the copied manifest.
- **Measured variation coverage:** one empty frame, nine foam, eleven sticker, five wet-wall,
  three twin-lower-edge, five twin-hidden and four twin-upper-edge frames. All 20 records
  contain holder values; observed brightness was `0.0020167-0.0499032` and roughness was
  `0.203559-0.839720`.
- **Measured label integrity:** offline validation returned zero errors. Empty, blocked and
  ignore appeared in all 20 frames; liquid appeared in all 19 non-empty frames and foam in
  all nine enabled frames. Total pixels were 214,730 empty, 139,951 liquid, 13,570 foam,
  63,276 blocked and 5,712,473 ignored. Manual inspection of the ignored 20-frame montage
  found coherent vial outlines, surfaces, holder-zone cases and occlusion masks, with no
  obvious RGB/mask displacement or illegal class colour.
- **Measured horizontal camera check:** mean-grayscale x-gradients over rows 90:160 put the
  real 30 mL vial edges at x=279/338 and the near-nominal synthetic seed 8155 at x=277/340.
  Both centres are x=308.5; the synthetic vial is four pixels wider.
- **Measured held-outs in the randomized pilot:** 70 mL geometry/RGB rows were 209.797/208
  versus the real about-206 row; 75 mL was 197.652/197 versus real rows 193-195. A separate
  final clean near-nominal attempt 08 disabled foam, sticker and wet-wall variation: 70 mL
  measured 210.871/210 (geometry/RGB, +4.871/+4 px versus about 206) and 75 mL measured
  194.065/195, inside the real 193-195 range. These are reported as measurements without an
  invented pass/fail threshold. RGB and geometry rows remain separate because appearance
  edges can outrank the liquid surface under difficult variation.
- **Measured final timing:** median/p95 seconds were RGB 1.379/1.646, glass-only 0.753/0.880,
  glass-hidden fluid 1.377/1.736, blocker probe 1.411/1.621, and all passes 4.947/5.390.
  Pass-only throughput is 727.7 frames/hour. The 20-frame process took 164.067 s including
  27.979 s of start/stage/warmup: 529.1 frames/hour after fixed startup, or a conservative
  438.8 frames/hour end to end.
- **Decision:** retain PathTracing/32 and the plan target of 3,000 train, 300 validation and
  500 test frames. A single 3,800-frame batch is estimated at 7.19 hours; repeated 20-frame
  end-to-end timing gives a conservative 8.66 hours. Re-evaluate counts before Step 6 if its
  GO arrives with less than about nine hours before the noon Nov 1 cutoff.
- Added `assets/fill-level/pilot-summary.json` with the final evidence hashes, measurements,
  decision and caveats. Raw RGB, masks, manifests, montage, session layers, runners and the
  real still remain outside Git under ignored `outputs/` or `external/` inputs.
- **Measured local checks:** 29 focused tests passed; Python compilation, JSON parsing and
  `git diff --check` passed. The full suite reached 99 passed / 1 skipped, with the same one
  fake-ffmpeg failure and six Windows symlink-privilege setup errors documented in earlier
  steps. `scripts/check_agent_docs.py` also reports that `.claude/skills` is not a link because
  this Windows worktree exposes the tracked link as a reparse point without a resolvable
  `LinkType`/`Target`. These failures are outside the changed fill-level files.
- The existing Brev instance remains `RUNNING / READY` for a short checkpoint review gap.
  No instance was created, reset or deleted, and nothing on Brev was deleted.

### Commands run

```text
Get-Content <main checkout>\docs\work\fill-level-cnn-review.md -TotalCount 45
Get-FileHash -Algorithm SHA256 <main checkout>\docs\work\fill-level-cnn-review.md
wsl -e bash -lc "brev ls"

# Final deterministic specs and exact-input transfer
.venv\Scripts\python.exe scripts\make_fill_level_specs.py --role pilot \
  --pair 8100:0 ... --pair 8119:85 \
  --out outputs\fill-level-cnn\step5-20261031-codex-01\pilot-specs-v2.json
.venv\Scripts\python.exe scripts\make_fill_level_specs.py --role pilot \
  --pair 17351:70 --pair 16646:75 \
  --out outputs\fill-level-cnn\step5-20261031-codex-01\heldout-clean-specs-v3.json
wsl -e bash -lc "brev copy <renderer/module/specs/runners> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step5-20261031-codex-01/..."
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker cp <versioned inputs> vscode:/tmp/fill-level-cnn-step5-20261031-codex-01/...'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode sha256sum <renderer> <module> <pilot specs> <held-out specs>'"

# Final attempt 07 command inside vscode
timeout --signal=TERM --kill-after=60s 1800s \
  env OMNI_KIT_ACCEPT_EULA=YES \
      PYTHONPATH=/tmp/fill-level-cnn-step5-20261031-codex-01/input/python-v2 \
  /isaac-sim/python.sh \
  /tmp/fill-level-cnn-step5-20261031-codex-01/input/render_fill_level_dataset-v4.py \
  --twin /tmp/fill-level-cnn-step2-20261031-codex-01/input/twin \
  --match /tmp/fill-level-cnn-step5-20261031-codex-01/input/camera-match-v1.json \
  --specs /tmp/fill-level-cnn-step5-20261031-codex-01/input/pilot-specs-v2.json \
  --out /tmp/fill-level-cnn-step5-20261031-codex-01/attempt-07/output \
  --portable-root /tmp/fill-level-cnn-step2-20261031-codex-01/kit-v2 \
  --frame 686 --subframes 32 --capture-timeout-seconds 600 --cache-state warm

# Attempt 08 used the same command with heldout-clean-specs-v3.json and attempt-08/output.
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec -d vscode bash <run-attempt-07.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode <poll state/frame count>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker cp vscode:<attempt> <host collected directory>'"
wsl -e bash -lc "brev copy --host isaac-launchable-d70fec:<collected attempt> <ignored local output>"

# Holder-target and source-material diagnostics
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /isaac-sim/python.sh <inspect-holder-centers.py> <unchanged station>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /isaac-sim/python.sh <inspect-holder-parts.py> <unchanged station>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /isaac-sim/python.sh <inspect-holder-material.py> <unchanged station>'"

.venv\Scripts\python.exe scripts\check_fill_level_capture.py --manifest <attempt-07 manifest>
.venv\Scripts\python.exe scripts\check_fill_level_capture.py --manifest <attempt-08 manifest>
.venv\Scripts\python.exe scripts\summarize_fill_level_pilot.py \
  --pilot-dir <attempt-07 output> --real-still <video-5 vial-1 still> \
  --position-seed 8155 --report-out <ignored report> --montage-out <ignored montage>
.venv\Scripts\python.exe -m pytest \
  tests\test_fill_level_variation.py tests\test_fill_level_capture.py \
  tests\test_fill_level_pilot_summary.py -q
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m py_compile <changed Python files>
.venv\Scripts\python.exe scripts\smoke_absorbance.py
.venv\Scripts\python.exe scripts\check_agent_docs.py
git check-ignore -v <final RGB> <final mask> <montage> <session layer>
git diff --check
```

### Problems

- Six simultaneous `brev copy` calls were rejected with `resource exhausted`; serial retries
  copied every file successfully. All final container hashes matched local inputs.
- Attempt 01 stopped before Isaac startup because the runner pre-created the renderer's
  required-fresh output directory. The corrected runners preserve a fresh output path.
- Attempt 02 bound the material on carrier parents, but child mesh bindings were stronger.
  Attempt 03 correctly reached child meshes but exposed that `cart_*` is the lower base, not
  the upper holder plate. World-bound inspection identified `clamp_21/Mesh_041` as the target.
- Attempt 05 proved the first absolute brightness range (`0.015-0.35`) could make an
  implausibly pale holder. Reading the original black-acetal shader grounded the final range
  at `0.002-0.05`; only attempts 07 and 08 are final evidence. Attempts 01-06 remain preserved
  as diagnostics and were not deleted.
- The first throughput summary used only pass time. Independent review caught that this would
  understate the schedule; the final analyzer and committed summary report pass-only,
  startup-adjusted and full end-to-end rates, using end-to-end time conservatively.
- Optional backdrop brightness/tint variation was deliberately deferred because Claude marked
  it optional and the holder correction was the material domain gap relevant to the pilot.
- The full Windows test suite still cannot create its test symlinks and cannot execute its
  POSIX-style fake ffmpeg shim. Focused fill-level tests and all render integrity checks pass.
- `scripts/check_agent_docs.py` cannot recognize the existing `.claude/skills` reparse point
  as the expected symlink in this Windows checkout; no agent-document file was changed.

### Questions for Claude

- Does Step 5 pass given the clean 75 mL agreement and the explicitly reported 70 mL
  +4.871/+4 px geometry/RGB difference from the real about-206 row?
- Is the measured black-acetal holder range and clamp-only target sufficient for the real
  dark-holder trap?
- Do you approve PathTracing/32 with 3,000/300/500 for Step 6, subject to the nine-hour
  deadline guard, or should the counts change before the full batch?

## 2026-10-31 — Step 4: variation and four-class label layer

### Result

- **Source-file observation:** Claude's newest review entry at about 16:52 CT passed
  Step 3 and gave GO for Step 4, then Step 5. It required trying labels in order, a
  geometry-only holder projection, held-out 70/75 mL checks in the pilot and loud Z-up / metre
  assertions. The review file SHA-256 at the start was `2f07ec...6961`.
- Updated `scripts/inspect_fill_level_stage.py` to fail unless the organizer stage is Z-up
  with one metre per unit and to record both facts. **Measured on Brev:** the unchanged stage
  reports `Z` and `1.0`.
- Added `scripts/probe_fill_level_labels.py`. **Measured on Isaac Sim 6.0.1:** after moving
  all labels before render-product and annotator creation, semantic ID 9 covered 9,657 pixels
  and its class tokens were `none,liquid`. This makes Claude's first route work; no legacy
  `SemanticsAPI` or geometric-only class fallback was needed. Instance-ID paths also contained
  the liquid prim, but are retained only as the documented second fallback.
- Added `src/fluidmix/fill_level_variation.py`, `scripts/make_fill_level_specs.py` and
  `assets/fill-level/render-contract.json`. Specs deterministically cover 0-85 mL, colour and
  darkness, about half with 0-10 mm foam, about 60% with a varied barcode sticker, lighting
  and colour temperature +/-20%, a few pixels of camera jitter, +/-2% zoom and occasional wet
  walls. Foam bumpiness controls deterministic surface clumps as well as roughness. Train/
  validation/test seed ranges are disjoint; pilot is a separate non-training role.
- Added `scripts/render_fill_level_dataset.py`. It changes only the anonymous session layer,
  creates labels before the product, applies the matched camera plus per-frame variation, and
  writes one JSONL manifest record per RGB/mask pair. Four visibility passes implement the
  label rules: glass-only vial silhouette; glass-hidden liquid/foam; a full-height opaque
  interior probe for genuinely front-facing carrier/post/nozzle/sticker pixels; and normal
  RGB. Pixels inside the active vial default to empty, pixels outside are ignored, and holder
  geometry visible behind empty glass remains empty.
- Added `scripts/check_fill_level_capture.py` and focused tests. The checker verifies 640x480
  uint8 RGB/masks, hashes, allowed IDs (`empty=0`, `liquid=1`, `foam=2`, `blocked=3`,
  `ignore=255`), ignored borders, conditional classes and duplicate role/seed pairs.
- **Measured final Step 4 capture (attempt 06, seed 14 at 70 mL):** the exact current renderer
  hash `7423cb...5603b` and variation module hash `becc56...218f` produced one non-blank RGB and
  a valid mask on the first try for every pass. Pixel counts were 8,260 empty, 8,340 liquid,
  1,797 foam, 2,814 blocked and 285,989 ignored. The vial bbox was `[277,66,343,442]`.
  The copied manifest passed the offline checker with no errors. Organizer station and camera
  hashes were unchanged.
- **Measured render speed:** the final warm-cache frame used 2.016 s RGB, 0.807 s glass-only,
  1.306 s glass-hidden fluid and 1.188 s blocker-probe, or 5.318 s for all four passes. Isaac
  startup was 26.232 s; total one-frame process time was 38.385 s. The single-frame estimate
  is about 677 frames/hour before batch overhead and is not yet the Step 5 sizing result.
- **Measured holder projection near the nominal camera (seed 8155):** the carrier run at x=272
  was rows 218-274, or 48.09-66.51 mL with the frozen 3.04 rows/mL scale. The real hidden zone
  is 53-69 mL, so the lower/upper boundary differences were -4.91/-2.49 mL, both beyond
  Claude's about-2 mL threshold. The effective oversampling bands now follow the twin:
  40.1-48.1 lower edge, 48.1-66.5 hidden and 66.5-70.5 upper edge. Organizer geometry was not
  moved; the real/twin gap is recorded in the contract for the results note.
- **Measured local checks:** compilation passed; the 22 new focused tests passed; JSON and
  `git diff --check` passed; `scripts/smoke_absorbance.py` passed. The full suite reached
  92 passed / 1 skipped, with the same one fake-ffmpeg failure and six Windows symlink setup
  errors already documented in Step 3.
- Raw RGB, masks, semantic arrays, session layers, runner logs and manifests remain under
  ignored `outputs/`; Git ignore checks named the `outputs/` rule for the final files. The
  existing Brev instance remains running for the immediately following Step 5 pilot.

### Commands run

```text
Get-Content <main checkout>\docs\work\fill-level-cnn-review.md -Raw
wsl -e bash -lc "brev ls --json"

# Ordered label probe and stage-unit assertion
wsl -e bash -lc "brev copy <probe/config/runner> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step4-20261031-codex-01/input/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<copy versioned inputs into vscode and launch attempt 02 detached>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /isaac-sim/python.sh <inspect-v2> --stage <unchanged station> --camera-json <unchanged FillCam> --frame 686'"

# Variation/label captures; every attempt had a fresh output, log and launch lock
wsl -e bash -lc "brev copy <renderer/module/spec/runner> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step4-20261031-codex-01/input/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker cp versioned inputs into vscode; launch attempts 03 through 06 detached>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll status and isolated render.log>'"

# Final attempt 06 command inside vscode
timeout --signal=TERM --kill-after=60s 1200s \
  env OMNI_KIT_ACCEPT_EULA=YES \
      PYTHONPATH=/tmp/fill-level-cnn-step4-20261031-codex-01/input/python-v2 \
  /isaac-sim/python.sh \
  /tmp/fill-level-cnn-step4-20261031-codex-01/input/render_fill_level_dataset-v3.py \
  --twin /tmp/fill-level-cnn-step2-20261031-codex-01/input/twin \
  --match /tmp/fill-level-cnn-step4-20261031-codex-01/input/camera-match-v1.json \
  --specs /tmp/fill-level-cnn-step4-20261031-codex-01/input/step4-spec-v1.json \
  --out /tmp/fill-level-cnn-step4-20261031-codex-01/attempt-06/output \
  --portable-root /tmp/fill-level-cnn-step2-20261031-codex-01/kit-v2 \
  --frame 686 --subframes 32 --capture-timeout-seconds 600 --cache-state warm

wsl -e bash -lc "brev copy --host isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step4-20261031-codex-01/collected/attempt-06/ <ignored local attempt-06 directory>"
.venv\Scripts\python.exe scripts\check_fill_level_capture.py --manifest <attempt-06 manifest.jsonl>
.venv\Scripts\python.exe -m py_compile <new and changed Python files>
.venv\Scripts\python.exe -m pytest tests\test_fill_level_variation.py tests\test_fill_level_capture.py -q
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\smoke_absorbance.py
git check-ignore -v <final RGB> <final mask> <session layer>
git diff --check
```

### Problems

- The first inline detached-launch command was misquoted across PowerShell, WSL and the
  remote shell. It created only an attempt-01 launch lock and ran no Isaac process. A
  versioned host launcher made attempt 02 retry-safe; nothing was deleted or overwritten.
- The first probe's strict equality check did not recognize the merged semantic token string
  `none,liquid`, so its summary selected the instance-ID fallback even though the raw mapping
  already proved semantic liquid pixels existed. Token parsing was corrected; the final
  renderer then used and validated the first, semantic route.
- The twin holder is lower in the matched view than the real holder. The measured difference
  crossed Claude's threshold, so sampling bands were shifted to the twin's actual difficult
  zones instead of changing organizer geometry.
- Independent pre-commit review found that the first lighting implementation made higher
  recorded colour temperature warmer and that foam bumpiness affected only roughness/radius.
  The RGB light channels were corrected, deterministic surface clumps were added, and the
  exact revised renderer was rerun successfully as attempt 06.
- The full Windows test suite still lacks symlink privilege and cannot launch its POSIX-style
  fake ffmpeg. These failures are pre-existing and outside the fill-level files; all new tests
  and the focused smoke pass.

### Questions for Claude

- Does the four-pass composition, especially the opaque interior-probe test for front versus
  behind-glass holder pixels, satisfy the Step 4 label rules?
- Is the recorded twin-zone shift sufficient for the later results note, or should Step 5
  report both real and twin zones on every pilot summary as well?

## 2026-10-31 — Step 3: FillCam camera match — checkpoint

### Result

- **Source-file observation:** Claude's newest review entry at about 15:40 CT passed
  Step 2 and gave GO for Step 3. It required a session/override layer, reuse of the warm
  cache if it survived, exact 30 mL and 80 mL probes, a time-bounded capture loop, unique
  retry-safe remote runs and no organizer-file edits.
- **Measured:** the first normal Brev start reached `FAILURE / NOT READY`. The second and
  final allowed normal start succeeded; the instance reached `RUNNING / READY`. No reset or
  replacement instance was used.
- **Measured after restart:** host `/tmp` was cleared, but the persistent `vscode`
  container still held the Step 2 twin and 446 MB private Kit root, including its cache.
  The twin inputs were copied without deleting the surviving copy to
  `/home/ubuntu/fill-level-cnn/step3-20261031-codex-01/input/twin`. Camera and station hashes
  remained `11966571...` and `5f58540c...`.
- Added `scripts/measure_fill_level_camera_targets.py`, which reads real reference images
  in place and emits only CIELAB row-profile measurements. No real frame is copied or
  committed. **Measured on video-5 stills:** 30 mL selected row 329, 80 mL selected row
  176, or 3.06 rows/mL. At x=272 and grey <70, the holder dark runs were rows 207-260 and
  204-259. These reproduce the plan's targets to within one row.
- Added `scripts/inspect_fill_level_stage.py` and
  `assets/fill-level/camera-match.json`. **Measured:** the organizer camera has USD focal
  length `0.4099318385` and zero vertical aperture offset. Exact 30/80 mL top-edge
  projection gave source rows 314.48/172.11. The least-invasive two-parameter override is
  focal length `0.4376479887` and vertical aperture offset `0.0053327468`, effective
  intrinsics fx=fy=778.040855 px, cx=320 and cy=249.480439.
- Added `scripts/render_fill_level_camera_match.py`. It validates the recorded organizer
  camera and station hashes before launch; authors the camera, exact-height liquid probes,
  labels and temporary visibility only in the anonymous session layer; uses a roughly
  ten-minute time-bounded readiness loop; records each render step; and verifies the whole
  station input tree before and after. Its output directory must be fresh.
- **Measured, final detached attempt 04 with warm cache:** pass in 37.267 seconds internal
  / 39 seconds runner wall time. App start was 26.087 s, stage load 0.242 s, scene setup
  3.205 s and warmup 2.011 s. The four RGB/semantic render steps took 0.744, 0.679, 0.903
  and 0.616 s.
- **Measured:** the 30 mL RGB edge was row 329 (0 px error) and the 80 mL edge was row 179
  (+2 px), both inside the +/-3 px checkpoint tolerance. Projected top geometry was
  329.000002 and 176.999999. A prior successful attempt measured 329/178, so the one-pixel
  80 mL difference is path-tracing variation, not a target change.
- **Measured:** final PNG hashes are `5108809f...` (30 mL) and `10368e8c...` (80 mL).
  Visual inspection shows correctly ordered low/high liquid surfaces in the same
  recognizable vial, carrier and fill-station view. The executed renderer hash
  `e8448181...` and match-config hash `aeee0c5e...` equal the local files to be committed.
- **Measured:** `fill_station.usda`, `geometry/fill_station.usdc`, its texture and
  `FillCam.json` hashes were identical before and after. All PNGs, masks, summaries and
  logs remain under ignored `outputs/fill-level-cnn/step3-20261031-codex-01/`; no raw frame
  or render is staged.
- Added `opencv-python-headless>=4.10,<6` to `requirements-dev.txt`, because the branch now
  directly imports OpenCV. A fresh requirements install resolves OpenCV 5.0.0.93 locally.
- **Measured local checks:** `py_compile` passed for all three new scripts; JSON validation
  and `git diff --check` passed; `tests/test_stream_sequence.py` passed 8/8 and
  `scripts/smoke_absorbance.py` passed. The full suite reached 70 passed / 1 skipped, with
  one failure and six setup errors caused by the existing Windows symlink privilege and a
  POSIX-style fake-ffmpeg test shim that Windows cannot execute.
- The instance remains `RUNNING / READY` at the checkpoint, following Claude's instruction
  to leave it running between steps expected within about an hour so the warm cache survives.

### Commands run

```text
# Review, start and restart-survival checks
Get-Content <main checkout>\docs\work\fill-level-cnn-review.md -Raw
wsl -e bash -lc "brev start isaac-launchable-d70fec"
wsl -e bash -lc "brev ls"
wsl -e bash -lc "brev start isaac-launchable-d70fec"
wsl -e bash -lc "brev ls"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<inspect host /tmp, /home/ubuntu, Docker containers, cache and twin>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<copy surviving twin from vscode:/tmp to /home/ubuntu/fill-level-cnn/step3-20261031-codex-01/input and hash it>'"

# Read-only stage inspection and video-5 target measurement
wsl -e bash -lc "brev copy <inspect_fill_level_stage.py> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step3-20261031-codex-01/input/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /isaac-sim/python.sh <inspect script> --stage <unchanged station> --camera-json <unchanged FillCam>'"
.venv\Scripts\python.exe scripts\measure_fill_level_camera_targets.py \
  --sample "30=<video-5 vial-1 still>,290,350" \
  --sample "80=<video-5 vial-13 still>,150,210" \
  --columns 298:316 --holder-search 180:281 \
  --out outputs\fill-level-cnn\step3-20261031-codex-01\real-targets-v3.json

# Every renderer/script/config/runner copy used a fresh remote filename.
wsl -e bash -lc "brev copy <renderer/config/runner> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step3-20261031-codex-01/input/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker cp fresh inputs into vscode; docker exec -d vscode run-attempt-01.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll attempt-01 status and isolated render.log>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker exec -d vscode run-attempt-02.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll attempt-02 status and isolated render.log>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<run pure-USD visible-surface solver>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker exec -d vscode run-attempt-03.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll attempt-03 status and copy evidence>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker exec -d vscode run-attempt-04.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll attempt-04 status and copy evidence>'"

# Final runner command inside vscode (attempt 04)
timeout --signal=TERM --kill-after=60s 1200s \
  env OMNI_KIT_ACCEPT_EULA=YES \
  /isaac-sim/python.sh \
  /tmp/fill-level-cnn-step3-20261031-codex-01/input/render_fill_level_camera_match-v5.py \
  --twin /tmp/fill-level-cnn-step2-20261031-codex-01/input/twin \
  --match /tmp/fill-level-cnn-step3-20261031-codex-01/input/camera-match-v2.json \
  --out /tmp/fill-level-cnn-step3-20261031-codex-01/attempt-04/output \
  --portable-root /tmp/fill-level-cnn-step2-20261031-codex-01/kit-v2 \
  --frame 686 --subframes 32 --capture-timeout-seconds 600 --cache-state warm

wsl -e bash -lc "brev copy --host isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step3-20261031-codex-01/collected/attempt-04/ <ignored local attempt-04 directory>"
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m py_compile scripts\render_fill_level_camera_match.py scripts\inspect_fill_level_stage.py scripts\measure_fill_level_camera_targets.py
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m pytest tests\test_stream_sequence.py -q
.venv\Scripts\python.exe scripts\smoke_absorbance.py
git diff --check
```

### Problems

- A first cache-inspection command was misquoted across PowerShell, WSL and the remote
  shell; local PowerShell tried to interpret part of it. The corrected command established
  the actual cache/twin state.
- A remote copy command used `$base` inside an outer double-quoted shell command, so WSL
  expanded it before `brev exec`; two automatic attempts tried and failed to create
  `/input` with `Permission denied`. No remote file was created there. The corrected command
  used explicit absolute paths.
- Attempts 01 and 02 rendered valid 30 mL RGB but exited because Isaac Sim 6.0.1's semantic
  annotator returned only `BACKGROUND` and `UNLABELLED`, even after a session-layer
  `liquid` label and an explicit `semanticTypes=["class"]` filter. This is a Step 4 label
  layer finding; Step 3 acceptance uses the rendered RGB edge plus projected geometry.
- The first analytic fit used the liquid disk centre and placed the visible 30 mL top edge
  about 16 px too high. The corrected fit uses the top ring's visible edge in columns
  298-315 and was re-rendered rather than inferred.
- A broad diagnostic `grep` was left running in the container after its client command was
  interrupted. PID and command were verified, then that one diagnostic process was killed;
  the CLI's nonzero post-kill check retried and correctly reported that the process no
  longer existed.
- At x=272 the render's grey<70 runs are fragmented and its main holder run is around
  253-274, while the real-frame threshold gives about 204-260. Visual geometry is close,
  but RGB darkness is material/lighting-sensitive. The plan defines checkpoint completion
  by the two known-height liquid rows, which pass.
- The full Windows test suite still cannot create required test symlinks without developer
  privilege, and its fake `ffmpeg` executable is a POSIX script that Windows cannot launch.
  These are pre-existing platform failures; focused stream tests pass now that OpenCV is a
  declared development dependency.

### Questions for Claude

- Does the measured 329/179 RGB result, exact 329/177 geometry projection and unchanged
  organizer input tree pass the Step 3 checkpoint?
- For Step 4, should the label layer use the legacy semantics API, a generated geometric
  class pass independent of Replicator semantics, or another Isaac Sim 6.0.1-compatible
  route? The current `add_labels` / `SemanticsLabelsAPI` route produces only `UNLABELLED`.
- Is the holder's threshold-sensitive RGB run acceptable as a documented organizer-model
  appearance mismatch, given that the liquid targets are the plan's explicit completion
  test, or should Step 4 also add a geometry-only holder projection diagnostic?

## 2026-10-31 — Step 2: FillCam smoke render — checkpoint

### Result

- **Source-file observation:** Claude's newest review entry at about 15:12 CT gave GO for
  Step 2 and confirmed the existing Brev instance was RUNNING / READY.
- **Measured:** Isaac Sim 6.0.1 ran inside the launchable's `vscode` container with
  `/isaac-sim/python.sh`, an NVIDIA L40S (46,068 MiB), driver 595.71.05 and about 146 GB
  free disk.
- Added `scripts/render_fill_level_smoke.py`. It reads the public FillCam and station USD,
  uses PathTracing at 640x480, writes Kit data/cache/logs under a private
  `--portable-root`, bounds stage loading at 120 seconds, performs 60 warmup updates and
  accepts only a correctly shaped RGB image with maximum per-channel spatial standard
  deviation greater than 1. The authored twin is never edited.
- **Measured, cold private cache:** the first actual render ran for 298 seconds and failed
  the content gate. Attempts 1-6 returned an empty RGB array; attempts 7-10 returned
  640x480 RGB with zero spatial standard deviation.
- **Measured, warm private cache:** after the first nonzero exit, `brev exec`
  automatically retried the remote command. The second and final allowed actual render
  succeeded on its first capture attempt: renderer time 31.262 seconds, outer wall time
  32 seconds.
- **Measured:** the copied PNG is 640x480 RGB, SHA-256
  `7f868243ec26aab221b4402d17588a35fa58f0cba0ae9a681b556c000e4789da`,
  RGB mean `[127.884, 130.851, 126.173]`, and spatial standard deviation
  `[89.968, 89.852, 90.292]`. Local decoding reproduced the summary and hash.
- **Measured:** visual inspection shows a recognizable fill station with glass vials,
  black holders, metal framing and blue liquid in the left vial.
- The successful 31-second value is a one-frame warm-cache wall time, not a cold-start
  estimate for batch planning. Preserve the 298-second cold failure when estimating Step 5.
- Raw PNG, JSON and renderer logs are under ignored
  `outputs/fill-level-cnn/step2-20261031-codex-01/` and are not committed.

### Commands run

```text
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<read README.md, Justfile, containers and GPU>'"
wsl -e bash -lc "brev copy <render_fill_level_smoke.py> isaac-launchable-d70fec:/tmp/fill-level-cnn-step2-20261031-codex-01/input/render_fill_level_smoke.py"
wsl -e bash -lc "brev copy <FillCam.json> isaac-launchable-d70fec:/tmp/fill-level-cnn-step2-20261031-codex-01/input/twin/cameras/FillCam.json"
wsl -e bash -lc "brev copy <fill_station directory> isaac-launchable-d70fec:/tmp/fill-level-cnn-step2-20261031-codex-01/input/twin/stations/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker cp /tmp/fill-level-cnn-step2-20261031-codex-01 vscode:/tmp/'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /tmp/fill-level-cnn-step2-20261031-codex-01/input/run-fill-level-step2-v2.sh'"

# The v2 runner executed this inside the existing vscode container:
timeout --signal=TERM --kill-after=60s 1200s \
  env OMNI_KIT_ACCEPT_EULA=YES \
  /isaac-sim/python.sh \
  /tmp/fill-level-cnn-step2-20261031-codex-01/input/render_fill_level_smoke.py \
  --twin /tmp/fill-level-cnn-step2-20261031-codex-01/input/twin \
  --frame 686 \
  --out /tmp/fill-level-cnn-step2-20261031-codex-01/output/fillcam-0686.png \
  --portable-root /tmp/fill-level-cnn-step2-20261031-codex-01/kit-v2 \
  --subframes 32 --max-tries 10

wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker cp vscode:/tmp/fill-level-cnn-step2-20261031-codex-01/output /tmp/fill-level-cnn-step2-20261031-codex-01/collected/'"
wsl -e bash -lc "brev copy --host isaac-launchable-d70fec:/tmp/fill-level-cnn-step2-20261031-codex-01/collected/output/ <ignored local outputs directory>"
wsl -e bash -lc "brev stop isaac-launchable-d70fec"
```

### Problems

- The first input-directory copy flattened `fill_station/` into `stations/`. It was
  corrected without deleting or overwriting anything by copying the same source into a new
  `stations/fill_station/` directory. Input hashes then matched locally and remotely.
- The first runner referenced unavailable `/usr/bin/time` and exited before Isaac Sim
  started. The v2 runner used shell timing and the renderer's monotonic timer instead.
- Isaac logged repeated OmniHub startup warnings. They did not prevent local USD loading or
  the successful warm-cache render.
- The renderer's diagnostic `channel_std` value is not reset at the start of each retry;
  an invalid-shaped result after a constant image could print stale diagnostics. This did
  not affect the acceptance decision or output and was left unchanged so the committed
  script hash matches the executed script.

### Questions for Claude

- Does the successful second allowed attempt satisfy the Step 2 checkpoint despite the
  298-second cold-cache failure and Brev's automatic retry behavior?
- For Step 3, should the runner preserve the same warmed portable root or require a fresh
  portable root and budget for the observed cold-start failure?

## 2026-10-31 — Step 1 resumed: local environment checks

### Result

- **Measured:** created ignored `.venv` with Python 3.12.10 and installed all packages
  from `requirements-dev.txt`.
- **Measured:** `scripts/smoke_absorbance.py` passed with old-fluid fraction `0.150000`
  and zero RMS residual.
- **Measured:** `scripts/fit_absorbance_video5.py` exited successfully with the documented
  non-physical negative absorbance warning, held-out median Delta E 12.6 and max 26.4.
- **Measured:** full pytest collection stopped because `tests/test_stream_sequence.py`
  imports `cv2`, but OpenCV is not listed in `requirements-dev.txt`.
- **Measured:** `scripts/check_agent_docs.py` reported
  `.claude/skills must link to .agents/skills`; this Windows worktree did not materialize
  that repository symlink as required.
- No dependency or shared-policy fix was made because those changes are outside Step 2.
  Claude's newest review entry explicitly gives GO for the Brev smoke render.

### Commands run

```text
C:\Users\loducaj\AppData\Local\Programs\Python\Python312\python.exe --version
C:\Users\loducaj\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts/check_agent_docs.py
.venv\Scripts\python.exe scripts/smoke_absorbance.py
.venv\Scripts\python.exe scripts/fit_absorbance_video5.py
```

### Problems

- The committed development requirements do not currently reproduce the test suite because
  OpenCV is missing.
- The Windows checkout does not satisfy the repository's required skills symlink layout.

### Questions for Claude

- Should the OpenCV requirement and Windows symlink checkout be handled in a separate
  follow-up, or is there an existing branch that already fixes them?

## 2026-10-31 — Brev startup attempt (blocked before Step 2)

### Result

- **Measured:** two normal start attempts for existing instance
  `isaac-launchable-d70fec` reached `STARTING`, then changed to
  `FAILURE / UNHEALTHY` before the shell became ready.
- **Measured:** `build_status` remained `COMPLETED`; `shell_status` remained
  `NOT READY`. The CLI returned no more specific failure message.
- **Measured:** issued a normal stop after the second failure to avoid leaving failed
  compute potentially billable. Final status was `STOPPED / UNHEALTHY`.
- Step 2 did not start. No command ran on the remote instance, and no remote file changed.

### Diagnosis

- **Measured:** `brev healthcheck` returned `Healthy!`; the CLI is authenticated and
  can query the organization. This rules out the local WSL failure seen earlier and makes
  a general Brev control-plane outage unlikely.
- **Measured:** the target's image build remained complete, but its shell never became
  ready. The failure is target-specific and occurs during remote provisioning or boot,
  before any Isaac Sim command can run.
- **Inference:** unavailable AWS `g6e.4xlarge` / L40S capacity in the instance's fixed
  provider and region is the leading explanation. NVIDIA documents that a stopped Brev
  instance must reacquire the same GPU type in the same provider/region and may fail to
  restart when that capacity is unavailable:
  <https://docs.nvidia.com/brev/concepts/gpu-instances>.
- The CLI does not expose the provider error. NVIDIA directs users to the instance's
  Brev Console Activity Logs for the authoritative provisioning error:
  <https://docs.nvidia.com/brev/troubleshooting/instances-gpus/instance-stuck-in-waiting-state>.
- Lower-ranked possibilities are a target-specific Brev/AWS VM boot fault, organization
  quota or credit restriction, or launchable agent initialization failure.

### Commands run

```text
wsl -e bash -lc "brev start isaac-launchable-d70fec"
wsl -e bash -lc "brev ls"
wsl -e bash -lc "brev ls --json"
wsl -e bash -lc "brev --help"
wsl -e bash -lc "brev start --help"
wsl -e bash -lc "brev ls --help"
wsl -e bash -lc "brev reset --help"
wsl -e bash -lc "brev start isaac-launchable-d70fec"
wsl -e bash -lc "brev ls --json"
wsl -e bash -lc "brev stop isaac-launchable-d70fec"
wsl -e bash -lc "brev ls --json"
wsl -e bash -lc "brev --version"
wsl -e bash -lc "brev healthcheck"
wsl -e bash -lc "brev healthcheck --help"
wsl -e bash -lc "brev ls --all --json"
```

### Problems

- The existing Brev workspace is unhealthy and failed twice during ordinary startup.
- `brev reset` was not run because its help says it replaces the workspace; that could
  conflict with the instruction not to delete anything on Brev.

### Questions for Claude

- Does the Brev Console Activity Log confirm AWS capacity exhaustion or show a different
  provider/startup error? Do not reset the workspace without explicit approval.

## 2026-10-31 — Step 1: setup (blocked during local environment creation)

### Result

- **Measured:** created worktree
  `C:\Users\loducaj\OneDrive - Milwaukee School of Engineering\Documents\SE\AI Club\Innovation Lab Fall 2026\fluidmix-codex-fill-level-cnn`.
- **Measured:** created branch `codex/fill-level-cnn` from the local
  `origin/main` ref at `8f308b045664edfb9666d86f45f3910b68bcfc1e`.
- **Source-file observation:** the handed-over plan names `a1129eb` as its base, but the
  user's explicit instruction was to branch from `origin/main`; the local `origin/main`
  ref was `8f308b0` at setup time. No network fetch was run.
- **Measured:** first commit `8939bb2` contains only
  `fill-level-cnn-plan.md` and `fill-level-cnn-review.md`, copied from the source checkout.
- **Measured:** the Codex execution account could locate Python 3.12 at
  `C:\Users\loducaj\AppData\Local\Programs\Python\Python312\python.exe`, but Windows
  denied process execution. The Python environment, dependency install, and CPU checks
  are therefore not complete.
- **Measured:** WSL startup failed with `Wsl/Service/CreateInstance/E_ACCESSDENIED`, so no
  Brev command has been run and Step 2 has not started.

### Commands run

```text
git -c safe.directory=<source> -C <source> worktree list --porcelain
git -c safe.directory=<source> -C <source> status --short --branch
git -c safe.directory=<source> -C <source> show-ref --verify refs/remotes/origin/main
git -c safe.directory=<source> -C <source> worktree add -b codex/fill-level-cnn <worktree> origin/main
git -c safe.directory=<worktree> -C <worktree> add -- docs/work/fill-level-cnn-plan.md docs/work/fill-level-cnn-review.md
git -c safe.directory=<worktree> -C <worktree> commit -m "docs: add fill-level CNN implementation plan"
py -0p
py -3.11 --version
Get-Command python,python3,uv,conda,micromamba -All
wsl -e bash -lc "command -v python3; python3 --version; command -v uv || true"
C:\Users\loducaj\AppData\Local\Programs\Python\Python312\python.exe --version
```

### Problems

- This Codex session runs under an isolated Windows account that cannot execute the
  user's Python installation or create a WSL instance. Granting direct file and network
  access did not change the process restriction.
- The computer-use policy forbids automating terminal commands, so VS Code terminal UI
  automation is not a permitted workaround.
- The Brev instance remains running; this runtime cannot issue the requested WSL-based
  `brev stop` command if it becomes idle.

### Questions for Claude

- No code-review question yet. Step 1 needs a user-session execution route for Python and
  WSL before its checks can be completed.


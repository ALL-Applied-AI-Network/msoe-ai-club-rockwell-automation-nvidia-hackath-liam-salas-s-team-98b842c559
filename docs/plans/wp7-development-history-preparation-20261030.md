# WP7 development-history preparation plan

Spec authority: Liam's delegated INN-64 decision relayed 2026-10-30: prepare option 1 only;
do not open the frozen WP1 pass or video 5. Evaluation-only / section-6 waiver is not approved.

Owner thread: `01a18128-8dc4-7780-ace1-4d816b49d176`

Base: accepted WP7 template integration
`6a04994a9ac8646d088470a09edeb834f0ef7501`.

## Global constraints

- The accepted CNN source tuple is exact:
  - receipt
    `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp5-cnn-swap-sixth-dev-blocker-fix-ff0fab6-01a18128-attempt01/population-receipt.json`,
    SHA-256 `836001f775d20e4835ed0cdee4f9197992b7a79def75ae1e06c9863c8e207810`;
  - SQLite
    `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp5-cnn-swap-sixth-dev-blocker-fix-ff0fab6-01a18128-attempt01/evidence.sqlite3`,
    SHA-256 `b83deef57e2c17d4b8cc068440c763729f9ebe3a1c078711516f57bc732dc793`;
  - independent rereview
    `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp5-cnn-swap-review-01a18128/independent-rereview-ff0fab6.md`,
    SHA-256 `0c7f843764179e0533d151320cca461ba9d5cff3b6b96965493754704dbdb63a`.
- The current receipt's nested `base_tuple` points to an older store. It is provenance only and
  must never replace the current exact receipt/database pair above.
- Source is synthetic twin development evidence: six runs, 18 vials, 45 predicted-mask stage
  records (39 train, six validation), not real production history. Never label it real, camera
  production, test, or plant evidence.
- Do not mutate the accepted receipt, database, source records, locators, images, masks or clips.
  Verify their hashes before and after. Create a separately owned derived bundle/store only.
- Decisions are `insufficient_evidence` only. The required limit codes are
  `frozen_decision_config_unavailable` and `matched_peer_batches_unavailable`; add
  `prior_reference_unavailable` when a strict-earlier query is empty. Never invent a threshold,
  peer batch, prior, keep/discard result, cause, defect or plant action.
- Use only exact predicted-mask Part-B records. Reject oracle/evaluation/test rows, `sim_metadata`,
  non-predicted masks, hash drift and accepted-tuple drift.
- History members are same-run, strictly earlier, and train-role only. Every cited member must
  pass `assert_history_citation`; no empty query may produce a citation.
- Explanations use a deterministic guarded template, status `UNKNOWN`, abstention, and
  collect-evidence wording. They send no images and repeat the exact limit codes. No learned model
  is loaded and no GPU is used.
- The fresh preparation preflight binds the exact accepted tuple and exact derived population but
  also binds the already accepted ordered 18-vial video (MP4 SHA-256
  `1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b`), its composition receipt
  (`0a88c47515af9fb0e389900f32504994914fe48bcc1e06bc392185382116b037`) and final review JSON
  (`ae4f9b2fdc3c6341a77dfa25044b673d7357155a98795ffa0f2df8b93ddec683`). It remains access-
  denied/sealed pending the accepted INN-67 reference bank/config. It must be structurally unusable
  as the execution preflight accepted by `run_test` or `run_real_check`.
- Liam chose option 3: run the one-shot only after INN-67's development reference bank and frozen
  decision config are accepted. If they are not accepted by about Saturday 2026-10-31 18:00 CT,
  option 1 may run as-is with honest abstentions. Option 2 / section-6 waiver is not approved.
- Never read the frozen WP1 one-pass video/recipe/freeze inputs or video 5. Never create a one-pass
  claim, output or journal. No push.
- Reviews block on real-input correctness, vacuous gates, leakage, provenance loss, or accidental
  one-shot access. Adversarial mid-run mutation/forgery hardening is a recorded limitation only.

## Task 1 — implement the derived development population and sealed preparation preflight

Implement test-first production boundaries in a new focused module and CLI(s).

1. Add an append-only `EvidenceStore` boundary that can add fully validated explained-phase
   decision/explanation rows to a separately copied measured store without rewriting stages.
   It must reconstruct and verify the null measured predecessor and its normalized locator hash
   against the stored `record_sha256`, validate full vial sequences, verify every history citation
   against stored query membership, reject collisions, and append only decisions/explanations.
2. Add a builder that:
   - hash-verifies the exact accepted tuple and source record/locator artifacts;
   - copies the accepted SQLite to a new no-clobber owner-tagged root and proves the source hash is
     unchanged;
   - validates SQLite integrity and the exact 6/18/45, 39/6, zero-decision/zero-explanation source
     denominators;
   - creates one factory-produced `vial_history` query per vial using the first stage as target,
     same run, strictly earlier cursor and `allowed_roles={"train"}`;
   - derives 45 explained records / 18 unique vial decisions and explanations, all
     `insufficient_evidence`, with the exact limit-code rules in Global constraints;
   - appends them to the copied store, verifies 45 decision and 45 explanation rows, verifies zero
     oracle/evaluation/test/sim-metadata exposure, and publishes deterministic population records,
     query receipts, explanation records, a receipt and SHA inventory without clobber.
3. Add a preparation-preflight boundary that accepts only the exact current source tuple plus an
   exact derived population receipt and the exact accepted ordered-video/review tuple. It
   publishes a canonical-hash receipt with synthetic
   development provenance, 18/18 insufficient-evidence denominators, exact limit-code counts,
   `one_shot_inputs_opened=false`, `access_status=sealed_pending_liam_decision`, and no route-
   decision artifact. Existing `_validate_execution_preflight`, `run_test` and `run_real_check`
   must reject it before any input/claim/output access.
4. Add CLI entry points for population build and preflight publication. They require expected
   SHA-256 values, owner ID and fresh destinations. They expose no WP1/video-5 argument.
5. Tests must demonstrate red/green behavior at public boundaries and cover exact positive build,
   deterministic content in two fresh roots, source immutability, stale tuple rejection,
   nested-base-tuple non-substitution, non-predicted/oracle/test/evaluation rejection, predecessor
   drift, empty/nonempty strict-earlier query citation rules, decision/explanation collisions,
   exact limit codes, 45/18 denominators, no-clobber, and execution-preflight rejection before
   `_read_production_bytes` or claim publication. Run the affected files, full suite, synthetic
   smoke and agent-doc check.
6. Update `docs/architecture.md` and `docs/operations.md` to describe this as preparation only,
   synthetic development evidence, not an approved production-history route and not a one-pass
   execution.

Commit the clean result locally. Do not materialize the canonical output in Task 1.

## Task 2 — materialize, verify and document the exact preparation bundle

After Task 1 passes its scoped code review:

1. Run the reviewed builder exactly once against the accepted tuple into
   `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-development-history-preparation-01a18128-attempt01`.
2. Publish the sealed preparation preflight into the same owner-tagged root. Do not invoke any
   execution phase.
3. Independently rehash and query the derived store read-only. Confirm exact source hashes are
   unchanged, all output inventory hashes pass, the store has 45 decisions/explanations but zero
   keep/discard decisions, and all citations are valid.
4. Send the exact code commit and materialized bundle through one scoped sol-high final review.
   A blocking finding gets one bounded correction/re-review; nonblocking adversarial hardening goes
   to Known limitations.
5. On PASS, record exact hashes and denominators in `PROGRESS.md`, `FINAL-REPORT.md`, and INN-64;
   move INN-64 to In Review + Blocked on Liam's irreversible access decision. Preserve the frozen
   WP1/video-5 seal.

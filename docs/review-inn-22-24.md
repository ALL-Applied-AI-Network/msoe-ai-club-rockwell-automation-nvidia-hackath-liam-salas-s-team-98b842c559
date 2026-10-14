# INN-22–24 review evidence

Reviewed 2026-10-14. Owen authorized publishing and merging after checks pass.
Scope: complete delivery against main, including the unmerged PR #2 scaffold and the
shared instructions, nine skills, contributor docs and review fixes.

## Delivered behavior

- AGENTS.md owns shared policy; CLAUDE.md imports it. Codex/Claude share one skill source.
- Six project skills plus plan-change, review-change and experiment-cycle are concise.
  Planning covers any feature/refactor and ends at handoff unless implementation is requested.
- Pytest is the main test command. The docs checker discovers new skills without a fixed
  inventory, parses YAML, rejects duplicate/invalid metadata, and checks local file targets.
  Heading fragments and mandatory index membership are outside that check.
- Contributor docs separate implemented APIs, proposals, measured evidence and unchecked claims.
- Recipe prediction rejects unknown/nonfinite/negative volumes and missing/nonfinite
  calibration for used colored tanks. Calibration rejects rank-deficient designs and
  invalid intensity vectors. Negative fitted absorbance remains allowed for diagnostics.
- Color extraction requires fresh successful frames and preserves prior JSON on failure.
  Empty video-5 input does not truncate prior labels. Outputs are prepared in temporary
  locations, then each destination is replaced. Valid nonempty subsets are supported.
- The synthetic smoke rejects nonfinite results.

These review fixes change physics input validation and data-tool failure behavior.
Original measurement values are unchanged; inherited color-CSV CRLF line endings were
normalized. This is a prototype scaffold, not a validated physical detector.

## Review and verification

Three fresh-context reviewers covered spec conformance, correctness/data loss and blast
radius. All three reproduced malformed YAML passing the original checker. Other confirmed
findings were stale-frame success, destructive empty-label failure, unsafe recipe inputs,
unidentifiable calibration, a NaN smoke pass and stale planning routing.

A fresh verifier attacked the repairs through real scripts/exported APIs and found the
code fixes held. It caught stale review-document claims; this current summary replaces them.
The review was independent of the author but used the same GPT model family.

| Check | Result |
|---|---|
| Full pytest suite | 24 passed |
| Review regressions on disposable pre-fix code | 14 failed, 5 baseline/positive checks passed |
| Shared pointer/YAML/file targets | 9 skills, 24 documents passed |
| Skill-creator frontmatter validator | All nine passed |
| Synthetic smoke | Fraction 0.150000, RMS residual 0 |
| Video-5 diagnostic | Negative absorbance; median/max Delta E 12.6/26.4 reproduced |
| Complete diff whitespace check | Passed |
| Fresh Codex local discovery | All nine enabled with repository scope, no errors |
| Fresh Claude workflow | Nine discovered; physics smoke passed and planning skill loaded |

Environment: macOS arm64, Python 3.14.5, NumPy 2.5.4, pytest 9.1.1, PyYAML 6.0.3,
Pillow 12.3.0. Regression fixtures simulate ffmpeg; they are not real-video measurements.
Local review/verification receipts stay under /private/tmp/inn-adversarial.

## Limits and issue acceptance

| Issue | Delivered | Remaining evidence |
|---|---|---|
| INN-22 | Shared contract, routing, tested commands and fresh Claude workflow | None for this instruction change |
| INN-23 | Shared skill tree, reviewed workflows and fresh host discovery | Model-backed Codex workflow blocked by HTTP 429; local discovery is not a substitute |
| INN-24 | Contributor setup, interfaces, provenance, partitions and operations | Required teammate clean-checkout trial |

The lead's clean-source checks do not certify the teammate trial. Record teammate,
checkout, environment, commands and corrections before marking that criterion complete.
Source video clocks, original extraction provenance, downloaded twin, GPU rendering,
Teledyne outputs, CSI plant and provider shutdown were not exercised.

Clear-reference physical validity, fitted-fraction clipping and nonphysical calibration
remain documented limits. No inference or plant advice may treat these diagnostics as
valid calibration. No new detector/training/Cosmos component was added.

## Publication scope

The branch includes scaffold PR #2; publishing the complete tree must include canonical
skills, the Claude symlink and the proposal already merged through PR #1. Validate the
committed tree, not only working-tree files, before pushing. The final PR and merge status
are verified separately in the task handoff.


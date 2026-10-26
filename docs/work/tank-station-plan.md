# Tank station design and acceptance

Outcome: a portable tank assembly and verified synthetic RGB/label capture pipeline
for the existing twin runtime. Base: `40d31b9`; task branch: `codex/tank-station`.
The user authorized implementation, runtime use, commit/push and a PR. Merge requires
separate approval. Work stays in the isolated task worktree and owned Spark directory.

## Design

- Four vessels with separate shells/liquids, small geometry labels and an inactive T300.
- Analytic cone-plus-cylinder volume mapped to a fixed-topology 64-sided liquid mesh.
- Integer-frame samples interpolated in volume; no inferred flow/carryover dynamics.
- Texture-free USD materials; original organizer assets referenced without modification.
- Fixed-view camera registration from approximate reference landmarks. Absolute scale,
  optics, dimensions and tank placement remain unmeasured.
- Native RGB capture with a separate geometric liquid-label pass through hidden shells.
  Plumbing, labels and level strips remain occluders. Stable per-tank integer IDs.
- Browser preview uses exported USD geometry and samples. It is not a training renderer.

## Acceptance and falsifiers

| Criterion | Evidence required |
|---|---|
| Portable asset | Checked-in compact USDA rebuilds exactly; no missing dependencies |
| Correct level geometry | Analytic inverse and signed mesh volume checks; invalid volumes rejected |
| T300 inactive | Always zero volume; zero mask pixels in every native capture |
| Native capture integrity | Nonconstant RGB; masks/counts/boxes agree with exact authored volumes |
| Boundary states | Nine-state empty/low/full/independent-tank sweep passes audit |
| Visual inspection | Empty, intermediate and full RGB/label images inspected |
| Source preservation | All 16 original twin files retain their hashes; composed stage resolves |
| No duplicated work | Current PRs/branches and Linear ownership checked before publication |
| Reviewable delivery | Tests, provenance, limits, reproducible commands and published PR |

The sweep must fail for stale/empty images, bad IDs, empty/nonempty disagreement,
non-growing masks during filling, or altered source assets. Capture is bounded by an
outer timeout on the existing Spark runtime; no new provisioned resources or spend.
All canary frames belong to one diagnostic sequence, not separate dataset partitions.

## Ownership and limits

INN-33 was unassigned at the ownership check. INN-35 drainage (Liam), INN-36 future
vision estimation, INN-45 fill animation and INN-52 stream segmentation remain separate.
PR #8 edits architecture docs. PR #9 edits carryover logic/tests; the only shared file
has disjoint changes: this PR extends the docs-copy fixture with assets.

The pipeline verifies synthetic data integrity. It cannot prove physical accuracy,
real absolute-volume estimation or model transfer. Those need independent real evidence
and an agreed estimator evaluation. The geometry reference is explicitly not held out.
See [usage and limits](../tank-station.md) and [runtime evidence](../tank-station-evidence.md).
Self-review is used; no independent reviewer was delegated.

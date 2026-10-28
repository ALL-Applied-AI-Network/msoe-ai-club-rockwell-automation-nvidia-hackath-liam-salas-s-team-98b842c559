# Cosmos usable-output freeze

This condition loosens presentation syntax, not evidence policy. Cosmos returns three labeled
prose fields. The deterministic parser extracts wording, an abstention marker and a suggested-action
marker. Quantitative values, units, evidence IDs and limit codes are copied only from the supplied
validated evidence record. A lenient parse is never counted as a raw strict-schema pass.

The parser is deliberately frozen by hand-written messy-output tests before inference. Do not use
the prior matched-run completions to add aliases or tune the parser. The fresh condition keeps the
same six fixtures, three model modes, deterministic control, seed, greedy decoding, batch size one
and 384-token cap. Its raw text and every error remain in the declared denominator.

## Config and input-audit bootstrap

The processor input audit depends on the final prompt and model/runtime facts but does not contain
the config file or its `input_audit_sha256` field. Freeze without a hash cycle as follows:

1. Commit the reviewed source, create the configured freeze tag at that clean commit, and compute
   `source_sha256` for exactly `SOURCE_PATHS`.
2. Create a provisional config with every final value except `input_audit_sha256`, using a temporary
   nonzero 64-hex value. Keep it outside Git in the new ignored output directory.
3. Run `scripts/audit_cosmos_nano_usable_inputs.py` once. It loads only the pinned offline processor,
   writes the 18 model-input fingerprints, and does not load model weights.
4. Hash that audit and replace the provisional config value with the observed digest. Do not change
   any other field. Revalidate the config and run the tracked-audit verifier; it recomputes every
   processor input and requires byte equality with the audit.
5. Make both final files read-only, hash them, record the hypothesis/falsifiers/denominators, and only
   then launch `scripts/run_cosmos_nano_usable_eval.py`. The runner requires the clean tagged commit,
   final config hash, absent raw-output path and exact tracked audit before loading weights.

Scoring uses `scripts/score_cosmos_nano_usable.py`. Report raw strict schema, lenient parse, content
policy, required abstention and image invariance separately for the 18 model attempts and six
deterministic controls. A parsed response still fails content if it adds numbers or IDs, asserts a
defect/cause/next-vial effect, or gives equipment-control advice.

## Measured usable-output result (2026-10-24)

The Cosmos3-Nano model completed 18/18 attempts with 0 inference errors. It passed raw
strict schema 0/18, lenient parse 0/18, content policy 0/18, required abstention 0/12,
and reached the generation cap 0/18. Image invariance was 0/6 and is not established.
Presentation leniency therefore did not rescue Nano: raw and parsed results remain separate
measures. The deterministic control passed lenient parse and content policy 6/6, and required
abstention 3/3. These are synthetic explanation-interface measurements only; they establish no
detection, defect, cause, physical, or real-camera claim.

Artifact hashes (SHA-256): `raw-results.json` / `input_sha256.raw_results`
`5fa112ba1646ccb7e9b9fc47e5800d3610994a0a719a00549db09381a59cd79e`; `score.json`
`7775faf5ffe935dbff53dea3075eac89e78b2f74839e91816f873e72af141f3d`; config
`feedfc73d42e6d1ebe8d7cb82a2cd813af28550dc2a29d991d6f484715240ca1`; fixtures
`e88f6f13476ae89a49d53e0ddfa6cd2be051530c073e946f5e985c16d2f0ee9f`; model-view manifest
`6a79afb5ff8aa352256b1ec029259edcb137fdadd43a1530494d778f17745027`; sealed scoring data
`cb2d45ca1fce48e7603a6cde52ff0cbb1bdad11e7b2285ad01b80cbd232faab9`.

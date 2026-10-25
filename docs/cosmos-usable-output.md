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

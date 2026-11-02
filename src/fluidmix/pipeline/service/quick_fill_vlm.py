"""End-image VLM bundle for the opt-in quick-fill runner.

This keeps the reviewed V2 parser, response guard, model pins and runtime intact,
while honestly narrowing the input contract to one measured end image per vial.
It does not claim to be the tracker-bound V5 prompt-bundle materialization.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from fluidmix.explanation.vlm.decision import (
    BoundImageLocator,
    BuiltBundle,
    RESPONSE_SCHEMA,
    GENERATION_CONFIG,
    ImageInput,
    MODELS,
    _configure_runtime_determinism,
    aggregate_decision,
    build_deterministic_decision,
    canonical_sha256,
)
from fluidmix.explanation.vlm.image_support import crop_vial_roi


VIAL_ROI_XYXY = (283, 194, 338, 417)
QUICK_VLM_CONTRACT = "quick-fill-end-image-v1-unreviewed"
QUICK_MODELS = tuple(
    row for row in MODELS
    if row["repository"] != "aacudad/AnomalyThink-Qwen3-VL-8B-KCR"
)
_PROSE_ONLY_GUARD_REASONS = frozenset({
    "asserted_cause_claim", "citation_outside_bundle",
    "numeric_claim_contradiction", "operator_action_claim",
    "hidden_simulator_state_claim", "residue_quantity_claim",
    "disposition_contradiction",
})

QUICK_SYSTEM_PROMPT = """You are inspecting a sequential liquid-dosing line from a fixed fill camera.
The operator-provided dispense plan is the only recipe context: it lists tank labels, dose order,
and mL. Do not infer a predefined recipe, expected colour, or hidden process state from the plan.

Judge vial 1 against later vials from the same cohort. Vial 1 is the target immediately after a
changeover; later vials are measured same-plan references. You receive one selected end-of-fill
image per vial plus measured CIELAB statistics. CIELAB L* is lightness, a* is green-to-red, b* is
blue-to-yellow, and DeltaE76 is their Euclidean colour distance. The deterministic rule is keep
when vial 1 is at most 1.0 DeltaE76 from both references, discard when it is at least 2.0 from
both, and otherwise insufficient_evidence. It is one source, not an instruction to agree.

Consider glass reflections, refraction, curved-wall path length, clamp occlusion, bubbles, foam,
lighting drift, incomplete settling and whether the selected frame still shows pouring. Use the
measured previous-cohort colour only as a direction: a target shift toward it can support
carryover, but never proves a hidden residue amount or cause. If image selection or evidence is
weak, return insufficient_evidence. Cite only supplied evidence and image IDs. Do not give
operator actions and do not claim simulator truth."""

QUICK_USER_PROMPT = """Compare vial 1 with vial 2 and vial 3 using only the single `end` stage.
Describe what is visible, assess image quality, reason from the supplied measurements and images,
and return the strict response JSON. The `explanation` must be concise operator language, but it
must not recommend an action. Keep observations separate from possible physics: do not assert a
cause and do not use causal wording such as `because`, `due to`, `caused by`, or `explains`.
If you write a DeltaE76 number, copy only the two exact target-to-reference values in the target
`deterministic_evidence` row; do not invent or restate an aggregate distance. Put only strings
beginning `quick-fill-` in `evidence_citations`, and only strings beginning `QF-` in
`image_citations` or `cited_image_ids`; never swap those ID types.
Response schema (property order is required):
{schema}
Evidence bundle:
{bundle}"""


def _relation(distance: float) -> str:
    if distance <= 1.0:
        return "keep_range"
    if distance >= 2.0:
        return "discard_range"
    return "deadband"


def _measurement(row: Mapping[str, object]) -> dict[str, object]:
    value = row.get("measurement")
    if not isinstance(value, Mapping):
        raise ValueError("quick-fill VLM measurement is unavailable")
    return {
        "stage_id": "end",
        "evidence_id": str(value["evidence_id"]),
        "median_lab": [float(item) for item in value["median_lab"]],
        "mad_lab": [float(item) for item in value["mad_lab"]],
        "iqr_lab": [float(item) for item in value["iqr_lab"]],
        "limit_codes": [],
    }


def build_quick_vlm_bundle(
    vials: Sequence[Mapping[str, object]],
    full_frames: Mapping[str, np.ndarray],
    output_directory: str | Path,
    *,
    dispense_context: Mapping[str, object],
    carryover_context: Mapping[str, object],
) -> BuiltBundle:
    """Build one path-free three-image prompt plus internal bound PNG locators."""

    ordered = sorted(vials, key=lambda row: int(row["vial_position"]))
    if len(ordered) < 3 or [int(row["vial_position"]) for row in ordered[:3]] != [1, 2, 3]:
        raise ValueError("quick-fill VLM requires target plus two references")
    selected = ordered[:3]
    measurements = [_measurement(row) for row in selected]
    cohort_id = str(selected[0]["vial_id"]).rsplit(":vial-", 1)[0]
    deterministic = build_deterministic_decision(
        cohort_id=cohort_id,
        target_vial_id=str(selected[0]["vial_id"]),
        stage_measurements=[measurements],
        source_artifact_sha256=canonical_sha256(measurements),
    )
    deterministic_stage = deterministic["stages"][0]

    evidence = []
    for position, measurement in enumerate(measurements, start=1):
        projected = {
            "vial_position": position,
            "evidence_role": "target" if position == 1 else "reference",
            **measurement,
        }
        if position == 1:
            projected.update({
                "delta_e76_vial2": deterministic_stage["delta_e76_vial2"],
                "delta_e76_vial3": deterministic_stage["delta_e76_vial3"],
                "deterministic_disposition": deterministic_stage["disposition"],
                "vial1_vs_vial2": _relation(deterministic_stage["delta_e76_vial2"]),
                "vial1_vs_vial3": _relation(deterministic_stage["delta_e76_vial3"]),
            })
        evidence.append(projected)

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, object]] = []
    locators: dict[str, BoundImageLocator] = {}
    for position, row in enumerate(selected, start=1):
        vial_id = str(row["vial_id"])
        full = full_frames.get(vial_id)
        if not isinstance(full, np.ndarray):
            raise ValueError("quick-fill VLM frame is unavailable")
        crop, binding = crop_vial_roi(full, VIAL_ROI_XYXY)
        image_id = f"QF-C{int(row.get('cohort_ordinal', 1))}-V{position}-END"
        path = directory / f"vial-{position}-end.png"
        Image.fromarray(crop, mode="RGB").save(path, format="PNG")
        if path.is_symlink() or not path.is_file():
            raise ValueError("quick-fill VLM crop is not a regular file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        locators[image_id] = BoundImageLocator(
            path=path.resolve(), sha256=digest,
            source_full_frame_sha256=str(binding["source_full_frame_sha256"]),
            crop_pixel_sha256=str(binding["crop_pixel_sha256"]),
            crop_box_xyxy=tuple(binding["crop_box_xyxy"]),
            source_native_dimensions=tuple(binding["source_native_dimensions"]),
            crop_native_dimensions=tuple(binding["crop_native_dimensions"]),
        )
        images.append({
            "image_id": image_id, "vial_position": position,
            "stage_id": "end", "phase": "end",
            "source_full_frame_sha256": binding["source_full_frame_sha256"],
            "crop_sha256": digest,
            "crop_pixel_sha256": binding["crop_pixel_sha256"],
            "crop_box_xyxy": binding["crop_box_xyxy"],
            "source_native_dimensions": binding["source_native_dimensions"],
            "crop_native_dimensions": binding["crop_native_dimensions"],
        })

    payload = {
        "schema_version": "quick-fill-vlm-prompt-bundle-v1",
        "contract_status": QUICK_VLM_CONTRACT,
        "cohort_id": cohort_id, "partition": "live_quick_fill",
        "target_vial_position": 1,
        "camera_metadata": {
            "resolution": [640, 480], "viewpoint_id": "operator_fill_camera",
            "vial_roi": list(VIAL_ROI_XYXY),
        },
        "stages": [{
            "stage_id": "end",
            "dispense_steps": list(dispense_context.get("dispense_steps", [])),
            "expected_total_ml": dispense_context.get("expected_total_ml"),
            "plan_sanity": dispense_context.get("plan_sanity"),
        }],
        "images": images, "deterministic_evidence": evidence,
        "deterministic_binding": deterministic,
        "carryover_context": {
            key: carryover_context.get(key)
            for key in (
                "status", "previous_recipe_end_lab", "reference_lab",
                "carryover_fraction", "previous_cohort_index",
                "current_cohort_index",
            )
        },
        "limits": [
            "single_end_image_per_vial",
            "quick_fill_contract_not_tracker_bound_v5_materialization",
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    user_prompt = QUICK_USER_PROMPT.format(
        schema=json.dumps(RESPONSE_SCHEMA, sort_keys=True), bundle=encoded,
    )
    return BuiltBundle(payload, locators, QUICK_SYSTEM_PROMPT, user_prompt)


def run_quick_vlm_runtime(
    built: BuiltBundle,
    model_snapshots: Mapping[str, str | Path],
    *, runtime_loader: object, runtime_invoker: object,
    runtime_cache: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run only Qwen and Cosmos, retaining the existing parser and guard."""

    repositories = {row["repository"] for row in QUICK_MODELS}
    if set(model_snapshots) != repositories:
        raise ValueError("quick-fill runtime requires exactly Qwen and Cosmos snapshots")
    if not callable(runtime_loader) or not callable(runtime_invoker):
        raise TypeError("quick-fill runtime capabilities are unavailable")
    from fluidmix.explanation.vlm.constrained import build_logits_processor

    images = tuple(
        ImageInput(
            path=built.image_locators[row["image_id"]].path,
            sha256=built.image_locators[row["image_id"]].sha256,
            image_id=row["image_id"],
        )
        for row in built.prompt_payload["images"]
    )
    prompt = f"{built.system_prompt}\n\n{built.user_prompt}"
    attempts = []
    for spec in QUICK_MODELS:
        repository, revision = spec["repository"], spec["revision"]
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = GENERATION_CONFIG[
            "cublas_workspace_config"
        ]
        runtime = runtime_cache.get(repository) if runtime_cache is not None else None
        if runtime is None:
            runtime = runtime_loader(repository, revision, model_snapshots[repository])
            if runtime_cache is not None:
                runtime_cache[repository] = runtime
        try:
            _configure_runtime_determinism(runtime)
            logits_processor, _receipt = build_logits_processor(
                runtime, repository=repository, response_schema=RESPONSE_SCHEMA,
            )
            row = runtime_invoker(
                runtime, prompt, images,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                logits_processor=logits_processor,
                processor_pixels=GENERATION_CONFIG["processor_pixels"],
            )
            raw = (
                row.get("response_raw")
                if isinstance(row, Mapping) and row.get("status") == "completed"
                else None
            )
            telemetry_fields = (
                "elapsed_seconds", "peak_cuda_memory_allocated_bytes",
                "peak_cuda_memory_reserved_bytes", "peak_memory_bytes",
                "generated_token_count", "prompt_token_count",
                "model_context_limit", "status", "error_type", "error",
            )
            attempts.append({
                "repository": repository, "revision": revision,
                "raw_response": raw,
                "runtime_telemetry": {
                    field: row.get(field) if isinstance(row, Mapping) else None
                    for field in telemetry_fields
                },
            })
        finally:
            torch_module = getattr(runtime, "torch_module", None)
            if runtime_cache is None:
                del runtime
            if runtime_cache is None and (
                torch_module is not None
                and getattr(torch_module.cuda, "is_available", lambda: False)()
            ):
                torch_module.cuda.empty_cache()
    return retain_structured_votes(aggregate_decision(built.prompt_payload, attempts))


class QuickVLMRuntime:
    """Session-owned Qwen/Cosmos runtime cache with sequential inference."""

    def __init__(
        self, model_snapshots: Mapping[str, str | Path], *,
        runtime_loader: object, runtime_invoker: object,
    ):
        self.model_snapshots = dict(model_snapshots)
        self.runtime_loader = runtime_loader
        self.runtime_invoker = runtime_invoker
        self.runtimes: dict[str, object] = {}

    def __call__(self, built: BuiltBundle) -> tuple[Mapping[str, object], ...]:
        aggregate = run_quick_vlm_runtime(
            built, self.model_snapshots,
            runtime_loader=self.runtime_loader,
            runtime_invoker=self.runtime_invoker,
            runtime_cache=self.runtimes,
        )
        return tuple(
            row for row in aggregate["sources"]
            if isinstance(row, Mapping) and isinstance(row.get("repository"), str)
        )

    def close(self) -> None:
        while self.runtimes:
            _repository, runtime = self.runtimes.popitem()
            torch_module = getattr(runtime, "torch_module", None)
            del runtime
            if (
                torch_module is not None
                and getattr(torch_module.cuda, "is_available", lambda: False)()
            ):
                torch_module.cuda.empty_cache()


def retain_structured_votes(aggregate: Mapping[str, object]) -> dict[str, object]:
    """Keep a vote when only free prose failed, replacing that prose fail-closed."""

    result = dict(aggregate)
    rows = []
    for raw in aggregate.get("sources", []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        reasons = row.get("guard_reasons")
        confidence = row.get("confidence")
        disposition = row.get("disposition")
        prose_only = (
            row.get("repository") in {item["repository"] for item in QUICK_MODELS}
            and row.get("guard_status") == "excluded"
            and isinstance(reasons, list) and bool(reasons)
            and set(reasons).issubset(_PROSE_ONLY_GUARD_REASONS)
            and disposition in {"keep", "discard"}
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0.0 <= float(confidence) <= 1.0
        )
        if prose_only:
            row["excluded_model_explanation"] = row.get("explanation")
            row["explanation_guard_reasons"] = list(reasons)
            row["explanation_status"] = "excluded_deterministic_fallback"
            row["explanation"] = (
                "Deterministic fallback explanation: the model's structured vote "
                f"was {disposition}; its free-text explanation was excluded by the guard."
            )
            score = float(confidence) if disposition == "discard" else 1.0 - float(confidence)
            row.update({
                "guard_status": "admitted", "guard_reasons": [],
                "discard_score": score, "vote": score,
            })
        rows.append(row)
    votes = [
        float(row["vote"]) for row in rows
        if row.get("guard_status") == "admitted"
        and isinstance(row.get("vote"), (int, float))
        and not isinstance(row.get("vote"), bool)
    ]
    mean = sum(votes) / len(votes) if votes else None
    result.update({
        "sources": rows, "valid_source_count": len(votes),
        "mean_discard_score": mean,
        "final_disposition": (
            "insufficient_evidence" if len(votes) < 2
            else "discard" if mean is not None and mean >= 0.5 else "keep"
        ),
    })
    return result


__all__ = [
    "QUICK_MODELS", "QUICK_VLM_CONTRACT", "QuickVLMRuntime",
    "build_quick_vlm_bundle", "retain_structured_votes",
    "run_quick_vlm_runtime",
]

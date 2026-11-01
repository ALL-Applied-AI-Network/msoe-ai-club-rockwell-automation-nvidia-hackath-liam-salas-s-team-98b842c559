"""Outer-ring runtime adapters for the frozen WP7 VLM contract.

The deterministic bundle, response guard, and aggregation policy remain in
``fluidmix.decision.vlm``.  Model loading/invocation, constrained decoding, process
runtime configuration, and receipt publication live here because they are effects.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from fluidmix.decision import vlm as decision


def run_sequential_vlm_runtime(
    built: decision.BuiltBundle,
    model_snapshots: Mapping[str, str | Path],
    *,
    runtime_loader: Any,
    runtime_invoker: Any,
    constrained_decoder_builder: Any = None,
) -> dict[str, Any]:
    """Deliver every ordered image to each exact pinned model, one at a time."""
    if not callable(runtime_loader) or not callable(runtime_invoker):
        raise decision.ContractError(
            "runtime loader and invoker are required injected capabilities"
        )
    if not isinstance(built, decision.BuiltBundle) or set(model_snapshots) != {
        row["repository"] for row in decision.MODELS
    }:
        raise decision.ContractError(
            "runtime requires snapshots for exactly the pinned models"
        )
    if constrained_decoder_builder is None:
        from fluidmix.explanation.vlm.constrained import build_logits_processor

        constrained_decoder_builder = build_logits_processor
    images = tuple(
        decision.ImageInput(
            path=built.image_locators[row["image_id"]].path,
            sha256=built.image_locators[row["image_id"]].sha256,
            image_id=row["image_id"],
        )
        for row in built.prompt_payload["images"]
    )
    prompt = f"{built.system_prompt}\n\n{built.user_prompt}"
    attempts = []
    for spec in decision.MODELS:
        repository, revision = spec["repository"], spec["revision"]
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = decision.GENERATION_CONFIG[
            "cublas_workspace_config"
        ]
        runtime = runtime_loader(repository, revision, model_snapshots[repository])
        try:
            decision._configure_runtime_determinism(runtime)
            logits_processor, _decoder_receipt = constrained_decoder_builder(
                runtime,
                repository=repository,
                response_schema=decision.RESPONSE_SCHEMA,
            )
            row = runtime_invoker(
                runtime,
                prompt,
                images,
                max_new_tokens=decision.GENERATION_CONFIG["max_new_tokens"],
                logits_processor=logits_processor,
                processor_pixels=decision.GENERATION_CONFIG["processor_pixels"],
            )
            raw = (
                row.get("response_raw")
                if isinstance(row, Mapping) and row.get("status") == "completed"
                else None
            )
            attempts.append(
                {
                    "repository": repository,
                    "revision": revision,
                    "raw_response": raw,
                    "runtime_telemetry": decision._runtime_telemetry(row),
                }
            )
        finally:
            del runtime
    return decision.aggregate_decision(built.prompt_payload, attempts)


def run_one_model_attempt_from_files(
    cohort_path: str | Path,
    image_locator_path: str | Path,
    tracker_receipt_paths: Sequence[str | Path],
    *,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    repository: str,
    revision: str,
    snapshot_path: str | Path,
    output_path: str | Path,
    prerequisite_qwen_receipt_path: str | Path | None = None,
    production: bool = False,
    runtime_loader: Any,
    runtime_invoker: Any,
    constrained_decoder_builder: Any = None,
) -> dict[str, Any]:
    """Run exactly one pinned stock model and exclusively publish its raw attempt."""
    if not callable(runtime_loader) or not callable(runtime_invoker):
        raise decision.ContractError(
            "runtime loader and invoker are required injected capabilities"
        )
    decision._assert_output_absent(output_path)
    context = decision.load_runtime_context(
        cohort_path,
        image_locator_path,
        tracker_receipt_paths,
        freeze_path=freeze_path,
        expected_freeze_sha256=expected_freeze_sha256,
        production=production,
    )
    pins = {row["repository"]: row["revision"] for row in decision.MODELS}
    if pins.get(repository) != revision:
        raise decision.ContractError(
            "runtime model is not one exact pinned model/revision"
        )
    policy = next(
        row
        for row in decision._MODEL_RUNTIME_POLICIES
        if row.repository == repository
    )
    if not policy.stock_runtime or policy.trust_remote_code:
        raise decision.ContractError(
            "runtime requires the stock trust_remote_code=False adapter"
        )
    qwen_repository = decision.MODELS[0]["repository"]
    prerequisite_sha = None
    if repository != qwen_repository:
        if prerequisite_qwen_receipt_path is None:
            raise decision.ContractError(
                "Cosmos/Anomaly runtime requires a valid Qwen attempt first"
            )
        prerequisite_value, _ = decision._regular_json(
            prerequisite_qwen_receipt_path, "Qwen prerequisite receipt"
        )
        prerequisite = decision._validate_attempt_receipt(
            prerequisite_value, context
        )
        if prerequisite["model"]["repository"] != qwen_repository:
            raise decision.ContractError("prerequisite attempt is not Qwen")
        try:
            parsed = decision.parse_model_response(
                prerequisite["invoke"]["response_raw"],
                repository=qwen_repository,
            )
        except decision.ContractError as exc:
            raise decision.ContractError(
                "Qwen prerequisite output did not parse"
            ) from exc
        if (
            decision.guard_model_response(
                parsed, context.built.prompt_payload
            )["guard_status"]
            != "admitted"
        ):
            raise decision.ContractError(
                "Qwen prerequisite output failed the response guard"
            )
        prerequisite_sha = prerequisite["receipt_sha256"]
    elif prerequisite_qwen_receipt_path is not None:
        raise decision.ContractError(
            "Qwen runtime must not declare a Qwen prerequisite"
        )

    prompt, images = decision._prompt_and_images(context, repository)
    if constrained_decoder_builder is None:
        from fluidmix.explanation.vlm.constrained import build_logits_processor

        constrained_decoder_builder = build_logits_processor
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = decision.GENERATION_CONFIG[
        "cublas_workspace_config"
    ]
    runtime = runtime_loader(repository, revision, snapshot_path)
    try:
        inventory = getattr(runtime, "inventory", None)
        if (
            not isinstance(inventory, Mapping)
            or decision._SHA.fullmatch(str(inventory.get("sha256"))) is None
        ):
            raise decision.ContractError(
                "stock runtime lacks snapshot inventory provenance"
            )
        decision._configure_runtime_determinism(runtime)
        logits_processor, decoder_receipt = constrained_decoder_builder(
            runtime,
            repository=repository,
            response_schema=decision.RESPONSE_SCHEMA,
        )
        invoke = decision._invoke_projection(
            runtime_invoker(
                runtime,
                prompt,
                images,
                max_new_tokens=context.binding["generation_config"][
                    "max_new_tokens"
                ],
                logits_processor=logits_processor,
                processor_pixels=context.binding["generation_config"][
                    "processor_pixels"
                ],
            )
        )
    finally:
        del runtime
    receipt: dict[str, Any] = {
        "schema_version": "wp7-vlm-model-attempt-receipt-v2",
        "artifact_type": "wp7_vlm_model_attempt_receipt",
        "cohort_id": context.built.prompt_payload["cohort_id"],
        "binding": decision._attempt_expected_binding(context, repository),
        "prompt": {
            "system_prompt_sha256": hashlib.sha256(
                context.built.system_prompt.encode("utf-8")
            ).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(
                context.built.user_prompt.encode("utf-8")
            ).hexdigest(),
            "combined_prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
        },
        "images": [
            {
                "image_id": image.image_id,
                "crop_sha256": image.sha256,
                "source_full_frame_sha256": metadata[
                    "source_full_frame_sha256"
                ],
                "crop_pixel_sha256": metadata["crop_pixel_sha256"],
                "crop_box_xyxy": metadata["crop_box_xyxy"],
                "source_native_dimensions": metadata[
                    "source_native_dimensions"
                ],
                "crop_native_dimensions": metadata["crop_native_dimensions"],
            }
            for metadata, image in zip(
                context.built.prompt_payload["images"], images, strict=True
            )
        ],
        "model": {
            "repository": repository,
            "revision": revision,
            "snapshot_inventory_sha256": inventory["sha256"],
            "trust_remote_code": False,
            "local_files_only": True,
        },
        "invoke": invoke,
        "constrained_decoding": decoder_receipt,
        "determinism": {
            "seed": decision.GENERATION_CONFIG["seed"],
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        },
        "token_budget": {
            "requested_max_new_tokens": 10000,
            "effective_max_new_tokens": 10000,
            "fallback_used": False,
            "fallback_proof_receipt_sha256": None,
            "prompt_token_count": invoke["prompt_token_count"],
            "model_context_limit": invoke["model_context_limit"],
            "compatibility": (
                "prompt_plus_10000_lte_context"
                if invoke["prompt_token_count"] is not None
                and invoke["model_context_limit"] is not None
                and invoke["prompt_token_count"] + 10000
                <= invoke["model_context_limit"]
                else "not_proven"
            ),
        },
        "prerequisite_qwen_receipt_sha256": prerequisite_sha,
    }
    receipt["receipt_sha256"] = decision._digest(receipt)
    decision._write_json_exclusive(output_path, receipt)
    return receipt

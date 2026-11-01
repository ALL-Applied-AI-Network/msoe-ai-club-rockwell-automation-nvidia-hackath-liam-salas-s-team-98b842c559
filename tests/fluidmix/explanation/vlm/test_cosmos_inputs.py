"""Processor audit for dynamic two/three-stage V3 Cosmos inputs."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from tests.fluidmix.explanation.vlm.test_cosmos_runtime import _plan  # noqa: E402


class _FakeBatch(dict):
    def to(self, *args, **kwargs):
        return self


class _FakeTokenizer:
    pass


class _FakeImageProcessor:
    def to_dict(self):
        return {"size": {"height": 480, "width": 640}, "rescale": True}


class _FakeProcessor:
    def __init__(self):
        self.chat_template = "native-template"
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeImageProcessor()

    def apply_chat_template(self, messages, **kwargs):
        batched = bool(messages and isinstance(messages[0], list))
        conversations = messages if batched else [messages]
        if kwargs["tokenize"] is False:
            values = [
                "rendered-user-message<|im_start|>assistant\n"
                for _ in conversations
            ]
            return values if batched else values[0]
        image_counts = [
            sum(item["type"] == "image" for item in row[0]["content"])
            for row in conversations
        ]
        result = _FakeBatch(
            input_ids=torch.tensor([[11, 12, 13]] * len(conversations)),
            attention_mask=torch.tensor([[1, 1, 1]] * len(conversations)),
        )
        image_count = sum(image_counts)
        if image_count:
            result["pixel_values"] = torch.full((image_count, 2), 0.25)
            result["image_grid_thw"] = torch.tensor([[1, 2, 3]] * image_count)
        return result

    def batch_decode(self, values, **kwargs):
        return [f"raw {index}" for index in range(values.shape[0])]


def _dynamic_plan():
    rows = _plan()
    # Match the frozen Varied V3 contract: peers are same-run comparison
    # images, not independently verified references.
    for row in rows:
        row["image_captions"] = [
            caption.replace("reference image", "same-run peer image")
            for caption in row["image_captions"]
        ]
    # Every third decision exercises three stages (six images); the rest use four.
    for decision in range(0, 18, 3):
        for offset in (1, 2):
            row = rows[decision * 4 + offset]
            row["image_ids"] += [
                f"IMG-{decision:02d}-STAGE-3-TARGET",
                f"IMG-{decision:02d}-STAGE-3-REFERENCE",
            ]
            row["image_captions"] += [
                "Stage 3 target image.",
                "Stage 3 same-run peer image.",
            ]
            row["image_count"] = 6
    return rows


def _config(processor):
    image_json = json.dumps(
        processor.image_processor.to_dict(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "experiment_id": "v3-cosmos-nano-v1",
        "model_revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
        "model_view_manifest_sha256": "a" * 64,
        "prompt_template_sha256": "b" * 64,
        "runtime_versions": {"transformers": "5.17.0"},
        "expected_processor_class": type(processor).__name__,
        "expected_tokenizer_class": type(processor.tokenizer).__name__,
        "chat_template_text_sha256": hashlib.sha256(
            processor.chat_template.encode()
        ).hexdigest(),
        "image_preprocessor_sha256": hashlib.sha256(image_json.encode()).hexdigest(),
        "chat_template_generation_suffix": "<|im_start|>assistant\n",
        "max_images_per_attempt": 6,
        "image_width": 640,
        "image_height": 480,
        "batch_size": 6,
    }


def _locators(tmp_path, plan):
    ids = sorted({image for row in plan for image in row["image_ids"]})
    locators = {}
    for index, image_id in enumerate(ids):
        rgb = tmp_path / f"rgb-{index:03d}.png"
        mask = tmp_path / f"mask-{index:03d}.png"
        Image.new("RGB", (640, 480), (index % 255, 20, 30)).save(rgb)
        Image.new("L", (640, 480), 255).save(mask)
        locators[image_id] = {
            "batch_id": "BATCH-TEST",
            "rgb_path": str(rgb),
            "rgb_sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
            "mask_path": str(mask),
            "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
            "recipe_id": "6",
            "run_id": "RUN-TEST",
            "source_manifest_sha256": "c" * 64,
            "split_role": "train",
            "split_version": "whole-run-varied-v3",
            "stage_id": "dispense-1-t200",
            "stage_ordinal": 1,
            "tank_id": "t200",
            "vial_id": "V1",
        }
    return locators


def test_audit_binds_all_54_dynamic_stage_inputs_and_nine_batches(tmp_path):
    from fluidmix.explanation.vlm.cosmos_inputs import (
        assert_v3_batch_matches_audit,
        audit_v3_model_inputs,
        build_v3_model_input_batch,
    )

    plan = _dynamic_plan()
    processor = _FakeProcessor()
    config = _config(processor)
    locators = _locators(tmp_path, plan)
    audit = audit_v3_model_inputs(
        plan,
        processor,
        image_locators=locators,
        config=config,
        observed_transformers_version="5.17.0",
    )

    assert audit["model_attempt_count"] == 54
    assert audit["condition_counts"] == {
        "numbers_only": 18,
        "ordered_stage_visual_pairs": 18,
        "irrelevant_shuffled_images": 18,
    }
    assert audit["batch_size"] == 6
    assert audit["batch_count"] == 9
    assert len(audit["batches"]) == 9
    assert {len(row["attempt_ids"]) for row in audit["batches"]} == {6}
    assert any(6 in row["image_counts"] for row in audit["batches"])
    assert any(4 in row["image_counts"] for row in audit["batches"])

    specs = [
        row
        for row in plan
        if row["kind"] == "model" and row["condition_id"] == "ordered_stage_visual_pairs"
    ][:6]
    built = build_v3_model_input_batch(
        processor, specs, image_locators=locators, config=config
    )
    assert_v3_batch_matches_audit(specs, built, audit)
    built["tensors"]["input_ids"][1, 0] = 99
    with pytest.raises(ValueError, match="differs"):
        assert_v3_batch_matches_audit(specs, built, audit)


def test_audit_rejects_locator_hash_drift_before_model_loading(tmp_path):
    from fluidmix.explanation.vlm.cosmos_inputs import audit_v3_model_inputs

    plan = _dynamic_plan()
    processor = _FakeProcessor()
    config = _config(processor)
    locators = _locators(tmp_path, plan)
    first = next(iter(locators.values()))
    first["rgb_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="RGB locator file or hash drifted"):
        audit_v3_model_inputs(
            plan,
            processor,
            image_locators=locators,
            config=config,
            observed_transformers_version="5.17.0",
        )


def test_batched_generation_rechecks_audit_and_retains_each_raw_suffix(tmp_path):
    from fluidmix.explanation.vlm.cosmos_inputs import (
        audit_v3_model_inputs,
        build_v3_model_input_batch,
        generate_from_audited_v3_input_batch,
    )

    plan = _dynamic_plan()
    processor = _FakeProcessor()
    config = {
        **_config(processor),
        "do_sample": False,
        "max_new_tokens": 3,
        "native_eos_token_ids": [151643, 151645],
    }
    locators = _locators(tmp_path, plan)
    audit = audit_v3_model_inputs(
        plan,
        processor,
        image_locators=locators,
        config=config,
        observed_transformers_version="5.17.0",
    )
    specs = [
        row
        for row in plan
        if row["kind"] == "model" and row["condition_id"] == "numbers_only"
    ][:6]
    built = build_v3_model_input_batch(
        processor, specs, image_locators=locators, config=config
    )

    class _Model:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            suffix = torch.tensor(
                [[index + 30, 151643, 151643] for index in range(len(specs))]
            )
            return torch.cat((kwargs["input_ids"], suffix), dim=1)

    model = _Model()
    results = generate_from_audited_v3_input_batch(
        model,
        processor,
        specs,
        built,
        audit,
        config,
        inference_context=nullcontext,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 3
    assert len(results) == 6
    assert [row["generated_suffix_token_ids"] for row in results] == [
        [index + 30, 151643] for index in range(6)
    ]

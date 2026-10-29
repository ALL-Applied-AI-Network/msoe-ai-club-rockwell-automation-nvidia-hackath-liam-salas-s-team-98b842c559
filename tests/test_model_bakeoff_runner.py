"""Contract and fake-runtime tests for the model bake-off runner."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.model_bakeoff_runner import (
    NEMOTRON_3_REVISION,
    MODEL_SPECS,
    ContractError,
    build_messages,
    load_adapter_model,
    load_plan,
    preflight,
    run_attempts,
    run_model,
    validate_model_selection,
    write_json_exclusive,
    validate_adapter_dir,
)


def _plan(
    tmp_path: Path,
    *,
    count: int = 1,
    schema_version: str = "real_validation_bakeoff.v1",
) -> Path:
    image = tmp_path / "frame.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(image)
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    cases = [
        {
            "case_id": f"case-{index}",
            "prompt": "Explain only the supplied quantitative evidence.",
            "images": [{"path": str(image), "sha256": image_sha}],
        }
        for index in range(count)
    ]
    path = tmp_path / "frozen-plan.json"
    path.write_text(
        json.dumps({"schema_version": schema_version, "cases": cases})
    )
    return path


def test_registry_contains_only_exact_pinned_repositories():
    assert set(MODEL_SPECS) == {
        "nvidia/Cosmos3-Nano",
        "nvidia/Cosmos-Reason1-7B",
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "Qwen/Qwen3-VL-8B-Instruct",
        "aacudad/AnomalyThink-Qwen3-VL-8B-KCR",
        "nvidia/Cosmos-Reason2-8B",
    }
    assert MODEL_SPECS["nvidia/Cosmos3-Nano"].revision == (
        "e59a53c25979a090fa8706c9acc0c254a6e89b92"
    )
    assert MODEL_SPECS["nvidia/Cosmos-Reason1-7B"].revision == (
        "375e24000b24baed78f4618d3dd779e47cd96323"
    )
    assert MODEL_SPECS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"].revision == (
        "ca9543b126e8bf3176916d3d305ccc415f89fd4d"
    )
    assert NEMOTRON_3_REVISION == "e5e9932441de940c9a62185c870ea5bcd4cd24e2"
    assert MODEL_SPECS[
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    ].expected_model_types == ("NemotronH_Nano_VL_V2",)
    assert MODEL_SPECS[
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    ].expected_model_types == ("NemotronH_Nano_Omni_Reasoning_V3",)
    expected_stock = {
        "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "aacudad/AnomalyThink-Qwen3-VL-8B-KCR": "8331bc79ecf01e3466f0e6a27cec8c77f182f430",
        "nvidia/Cosmos-Reason2-8B": "a9fae2cf89dc64db96b12860417f0eb403013bb9",
    }
    for repository, revision in expected_stock.items():
        spec = MODEL_SPECS[repository]
        assert spec.revision == revision
        assert spec.expected_model_types == ("qwen3_vl",)
        assert spec.stock_runtime is True
        assert spec.trust_remote_code is False


def test_unknown_or_unpinned_model_is_rejected():
    with pytest.raises(ContractError, match="exact pinned"):
        validate_model_selection("nvidia/Cosmos3-Nano", "main")
    with pytest.raises(ContractError, match="exact pinned"):
        validate_model_selection("someone/Cosmos3-Nano", None)


def test_plan_requires_existing_hashed_image_and_preserves_case_order(tmp_path):
    path = _plan(tmp_path, count=2)
    plan = load_plan(path)
    assert [case.case_id for case in plan.cases] == ["case-0", "case-1"]
    assert plan.cases[0].images[0].sha256
    assert plan.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("image_count", [4, 6])
def test_plan_and_attempt_metadata_preserve_every_ordered_image(tmp_path, image_count):
    from fluidmix.model_bakeoff_runner import _bind_attempt_metadata

    cases = []
    images = []
    for index in range(image_count):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (1, 1), (index, 2, 3)).save(path)
        images.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "image_id": f"image-{index}",
        })
    cases.append({"case_id": "case", "prompt": "prompt", "images": images})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema_version": "real_validation_bakeoff.v1", "cases": cases}))
    plan = load_plan(plan_path)
    row = _bind_attempt_metadata([{"status": "completed"}], plan, "repo")[0]
    assert row["ordered_image_ids"] == [f"image-{index}" for index in range(image_count)]
    assert row["ordered_image_sha256"] == [image["sha256"] for image in images]
    assert row["image_id"] == "image-0"


def test_plan_accepts_synthetic_eval_schema_without_relaxing_image_validation(tmp_path):
    plan = load_plan(_plan(tmp_path, schema_version="inn51_synthetic_eval_plan.v1"))
    assert len(plan.cases) == 1


def test_attempt_plan_collapses_identical_rows_across_models(tmp_path):
    base = json.loads(_plan(tmp_path).read_text())
    case = base["cases"][0]
    attempt = {
        "case_id": case["case_id"],
        "model_id": "model-a",
        "prompt": case["prompt"],
        "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        "image_path": case["images"][0]["path"],
        "image_sha256": case["images"][0]["sha256"],
    }
    path = tmp_path / "attempt-plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "real_validation_attempt_plan.v1",
                "attempts": [attempt, {**attempt, "model_id": "model-b"}],
            }
        )
    )
    assert len(load_plan(path).cases) == 1

    list_path = tmp_path / "attempt-plan-list.json"
    list_path.write_text(json.dumps([attempt, {**attempt, "model_id": "model-b"}]))
    assert len(load_plan(list_path).cases) == 1


def test_messages_use_one_common_text_then_images(tmp_path):
    plan = load_plan(_plan(tmp_path))
    messages = build_messages(plan.cases[0])
    assert messages[0]["role"] == "user"
    assert [part["type"] for part in messages[0]["content"]] == ["image", "text"]
    assert messages[0]["content"][0]["image"].size == (1, 1)


class _FakeBatch(dict):
    def to(self, *_args, **_kwargs):
        return self


class _FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("tokenize"):
            return _FakeBatch(input_ids=[[10, 11]])
        return "rendered"

    def batch_decode(self, suffix, **_kwargs):
        assert suffix == [[12, 13]]
        return ["raw model answer"]


class _ExtraFieldProcessor(_FakeProcessor):
    def apply_chat_template(self, messages, **kwargs):
        return _FakeBatch(input_ids=[[10, 11]], num_patches=[1])


class _FakeModel:
    device = "cuda:0"

    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 17
        return [[10, 11, 12, 13]]


class _StrictForwardModel(_FakeModel):
    def forward(self, input_ids):
        return input_ids


def test_run_attempts_filters_processor_only_fields_for_strict_custom_forward(tmp_path):
    plan = load_plan(_plan(tmp_path))
    result = run_attempts(
        plan,
        _ExtraFieldProcessor(),
        _StrictForwardModel(),
        max_new_tokens=17,
        torch_module=None,
    )
    assert result[0]["status"] == "completed"


def test_run_attempts_is_greedy_and_records_raw_output_and_latency(tmp_path):
    plan = load_plan(_plan(tmp_path))
    result = run_attempts(
        plan,
        _FakeProcessor(),
        _FakeModel(),
        max_new_tokens=17,
        torch_module=None,
        clock=lambda: 100.0,
    )
    assert result[0]["status"] == "completed"
    assert result[0]["attempt_status"] == "completed"
    assert result[0]["response_raw"] == "raw model answer"
    assert result[0]["response_utf8_sha256"] == hashlib.sha256(
        b"raw model answer"
    ).hexdigest()
    assert result[0]["generated_token_count"] == 2
    assert result[0]["elapsed_seconds"] == 0.0
    assert result[0]["peak_cuda_memory_allocated_bytes"] is None


def test_run_attempts_retains_inference_error(tmp_path):
    plan = load_plan(_plan(tmp_path))

    class BrokenModel:
        device = "cuda:0"

        def generate(self, **_kwargs):
            raise RuntimeError("synthetic inference failure")

    result = run_attempts(
        plan,
        _FakeProcessor(),
        BrokenModel(),
        max_new_tokens=17,
        torch_module=None,
    )
    assert result[0]["status"] == "error"
    assert result[0]["error_type"] == "RuntimeError"
    assert result[0]["error"] == "synthetic inference failure"


def test_run_attempts_records_processor_image_telemetry_on_success(tmp_path):
    plan = load_plan(_plan(tmp_path))

    class Processor(_FakeProcessor):
        def apply_chat_template(self, messages, **kwargs):
            return _FakeBatch(
                input_ids=[[10, 11]],
                pixel_values=np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2),
            )

    result = run_attempts(
        plan, Processor(), _FakeModel(), max_new_tokens=17, torch_module=None
    )[0]
    telemetry = result["processor_telemetry"]
    assert telemetry["image_count"] == 1
    assert telemetry["original_rgb_sizes"] == [[1, 1]]
    assert telemetry["processor_class"] == "Processor"
    assert telemetry["pixel_values_shape"] == [1, 3, 2, 2]
    assert telemetry["pixel_values_dtype"] == "float32"
    assert telemetry["pixel_values_sha256"] == hashlib.sha256(
        np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2).tobytes()
    ).hexdigest()


def test_run_attempts_retains_processor_telemetry_on_generation_error(tmp_path):
    plan = load_plan(_plan(tmp_path))

    class BrokenModel(_FakeModel):
        def generate(self, **_kwargs):
            raise RuntimeError("generation failed")

    class Processor(_FakeProcessor):
        def apply_chat_template(self, messages, **kwargs):
            return _FakeBatch(
                input_ids=[[10, 11]],
                pixel_values=np.ones((1, 3, 2, 2), dtype=np.float16),
            )

    result = run_attempts(
        plan, Processor(), BrokenModel(), max_new_tokens=17, torch_module=None
    )[0]
    assert result["status"] == "error"
    assert result["processor_telemetry"]["pixel_values_shape"] == [1, 3, 2, 2]
    assert result["processor_telemetry"]["pixel_values_dtype"] == "float16"


def test_run_model_preserves_load_error_for_every_case_without_downloading(tmp_path):
    plan = load_plan(_plan(tmp_path, count=2))
    result = run_model(
        "nvidia/Cosmos3-Nano",
        MODEL_SPECS["nvidia/Cosmos3-Nano"].revision,
        plan,
        model_dir=tmp_path / "missing-snapshot",
    )
    assert result["model_load_error"]["type"] in {"ContractError", "ModuleNotFoundError"}
    assert [row["attempt_index"] for row in result["attempts"]] == [0, 1]
    assert all(row["model_id"] == "nvidia/Cosmos3-Nano" for row in result["attempts"])
    assert all(row["attempt_status"] == "error" for row in result["attempts"])


def test_preflight_uses_case_sensitive_cached_nemotron_config_type(tmp_path):
    plan = load_plan(_plan(tmp_path))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "README.md").write_text("official card")
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "NemotronH_Nano_VL_V2", "architectures": []})
    )
    result = preflight(
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        MODEL_SPECS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"].revision,
        plan,
        snapshot,
    )
    assert result["status"] == "ready"


def test_stock_runtime_rejects_remote_code_metadata_and_loads_with_false_policy(tmp_path):
    from fluidmix.model_bakeoff_runner import load_stock_runtime, snapshot_inventory

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
    }))
    (snapshot / "weights.safetensors").write_bytes(b"synthetic")
    calls = []

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((cls.__name__, Path(path), kwargs))
            return cls()

        def eval(self):
            return self

    class Processor(Loader):
        pass

    class Model(Loader):
        device = "cpu"

    runtime = load_stock_runtime(
        "Qwen/Qwen3-VL-8B-Instruct",
        MODEL_SPECS["Qwen/Qwen3-VL-8B-Instruct"].revision,
        snapshot,
        processor_class=Processor,
        model_class=Model,
        torch_module=None,
    )
    assert runtime.inventory["sha256"] == snapshot_inventory(snapshot)["sha256"]
    assert all(call[2]["trust_remote_code"] is False for call in calls)
    assert all(call[2]["local_files_only"] is True for call in calls)

    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "auto_map": {"AutoModel": "remote.Model"},
    }))
    with pytest.raises(ContractError, match="auto_map|remote"):
        load_stock_runtime(
            "Qwen/Qwen3-VL-8B-Instruct",
            MODEL_SPECS["Qwen/Qwen3-VL-8B-Instruct"].revision,
            snapshot,
            processor_class=Processor,
            model_class=Model,
            torch_module=None,
        )

    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
    }))
    with pytest.raises(ContractError, match="type"):
        load_stock_runtime(
            "Qwen/Qwen3-VL-8B-Instruct",
            MODEL_SPECS["Qwen/Qwen3-VL-8B-Instruct"].revision,
            snapshot,
            processor_class=Processor,
            model_class=Model,
            torch_module=None,
        )


def test_snapshot_inventory_is_deterministic_and_rejects_symlinks(tmp_path):
    from fluidmix.model_bakeoff_runner import snapshot_inventory

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "b").write_bytes(b"bb")
    (snapshot / "a").write_bytes(b"a")
    inventory = snapshot_inventory(snapshot)
    assert [row["path"] for row in inventory["files"]] == ["a", "b"]
    assert inventory["files"][0]["bytes"] == 1
    (snapshot / "link").symlink_to(snapshot / "a")
    with pytest.raises(ContractError, match="symlink"):
        snapshot_inventory(snapshot)
    (snapshot / "link").unlink()
    snapshot_link = tmp_path / "snapshot-link"
    snapshot_link.symlink_to(snapshot, target_is_directory=True)
    with pytest.raises(ContractError, match="symlink"):
        snapshot_inventory(snapshot_link)


def test_run_attempts_synchronizes_cuda_around_generation(tmp_path):
    plan = load_plan(_plan(tmp_path))
    events = []

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def reset_peak_memory_stats():
            pass

        @staticmethod
        def max_memory_allocated():
            return 10

        @staticmethod
        def max_memory_reserved():
            return 20

        @staticmethod
        def synchronize():
            events.append("sync")

    class Torch:
        cuda = Cuda()

    class Model(_FakeModel):
        def generate(self, **kwargs):
            events.append("generate")
            return super().generate(**kwargs)

    row = run_attempts(plan, _FakeProcessor(), Model(), max_new_tokens=17, torch_module=Torch())[0]
    assert row["status"] == "completed"
    assert events == ["sync", "generate", "sync"]


def test_preflight_binds_adapter_metadata_only_to_pinned_nemotron(tmp_path):
    plan = load_plan(_plan(tmp_path))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "README.md").write_text("official card")
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "NemotronH_Nano_VL_V2", "architectures": []})
    )
    revision = MODEL_SPECS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"].revision
    result = preflight(
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        revision,
        plan,
        snapshot,
        _adapter(tmp_path),
    )
    assert result["adapter"]["base_model_repository"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"

    with pytest.raises(ContractError, match="pinned 12B"):
        run_model(
            "nvidia/Cosmos3-Nano",
            MODEL_SPECS["nvidia/Cosmos3-Nano"].revision,
            plan,
            model_dir=snapshot,
            adapter_dir=_adapter(tmp_path / "other"),
        )


def test_result_publication_is_atomic_and_rejects_overwrite(tmp_path):
    output = tmp_path / "raw-results.json"
    write_json_exclusive(output, {"value": 1})
    assert json.loads(output.read_text()) == {"value": 1}
    with pytest.raises(ContractError, match="overwrite"):
        write_json_exclusive(output, {"value": 2})


def _adapter(tmp_path: Path, *, model: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", revision: str | None = None, evaluated: bool = False) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({
            "peft_type": "LORA",
            "base_model_name_or_path": model,
            "revision": revision or MODEL_SPECS[model].revision if model in MODEL_SPECS else revision,
        })
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
    (adapter / "training-receipt.json").write_text(
        json.dumps({
            "schema_version": "inn51_lora_training_receipt.v1",
            "model_id": model,
            "model_revision": revision or MODEL_SPECS[model].revision if model in MODEL_SPECS else revision,
            "evaluation_dataset_used": evaluated,
        })
    )
    return adapter


def test_adapter_validation_hashes_receipt_and_rejects_wrong_base_or_eval(tmp_path):
    adapter = _adapter(tmp_path)
    metadata = validate_adapter_dir(
        adapter,
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        MODEL_SPECS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"].revision,
    )
    assert metadata["schema_version"] == "inn51_lora_adapter.v1"
    assert metadata["files"]["adapter_config.json"]["sha256"]
    assert metadata["files"]["training-receipt.json"]["sha256"]

    with pytest.raises(ContractError, match="pinned 12B"):
        validate_adapter_dir(
            _adapter(tmp_path / "wrong", model="nvidia/Cosmos3-Nano"),
            "nvidia/Cosmos3-Nano",
            MODEL_SPECS["nvidia/Cosmos3-Nano"].revision,
        )
    with pytest.raises(ContractError, match="evaluation"):
        validate_adapter_dir(
            _adapter(tmp_path / "eval", evaluated=True),
            "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
            MODEL_SPECS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"].revision,
        )


def test_adapter_load_wraps_base_and_sets_eval():
    calls = {}

    class FakeAdapter:
        def eval(self):
            calls["eval"] = True
            return self

    class FakePeft:
        @classmethod
        def from_pretrained(cls, model, adapter_dir, **kwargs):
            calls["load"] = (model, adapter_dir, kwargs)
            return FakeAdapter()

    result = load_adapter_model("base", "/tmp/adapter", peft_model_class=FakePeft)
    assert isinstance(result, FakeAdapter)
    assert calls["load"] == ("base", "/tmp/adapter", {"is_trainable": False})
    assert calls["eval"] is True

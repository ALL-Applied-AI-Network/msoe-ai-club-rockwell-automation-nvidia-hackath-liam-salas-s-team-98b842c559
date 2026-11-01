from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_production_stock_runtime_owns_pinned_local_loader(tmp_path):
    from fluidmix.explanation.vlm.stock_runtime import (
        PINNED_MODELS, load_stock_runtime, snapshot_inventory,
    )

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
            calls.append((Path(path), kwargs))
            return cls()

        def eval(self):
            return self

    runtime = load_stock_runtime(
        "Qwen/Qwen3-VL-8B-Instruct",
        PINNED_MODELS["Qwen/Qwen3-VL-8B-Instruct"], snapshot,
        processor_class=Loader, model_class=Loader,
    )
    assert runtime.inventory["sha256"] == snapshot_inventory(snapshot)["sha256"]
    assert all(row[1] == {"trust_remote_code": False, "local_files_only": True}
               for row in calls)


def test_evaluation_module_reexports_production_stock_kernel():
    import evaluation.rca_bench.model_bakeoff_runner as evaluation
    import fluidmix.explanation.vlm.stock_runtime as production

    assert evaluation.snapshot_inventory is production.snapshot_inventory
    assert evaluation.load_stock_runtime is production.load_stock_runtime
    assert evaluation.invoke_stock_runtime is production.invoke_stock_runtime
    assert evaluation.ContractError is production.ContractError


def test_stock_invoker_tolerates_torch_without_cuda_or_inference_mode():
    from fluidmix.explanation.vlm.stock_runtime import StockRuntime, invoke_stock_runtime

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        image_processor = SimpleNamespace(min_pixels=1, max_pixels=1)

        def apply_chat_template(self, *args, **kwargs):
            return Batch(input_ids=[[1, 2]])

        def batch_decode(self, value, **kwargs):
            return ["ok"]

    class Model:
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=100)

        def forward(self, input_ids):
            pass

        def generate(self, input_ids, **kwargs):
            return [[1, 2, 3]]

    runtime = StockRuntime(
        repository="repo", revision="rev", snapshot=Path("."),
        processor=Processor(), model=Model(), torch_module=object(),
        processor_class="Processor", model_class="Model", inventory={},
    )
    row = invoke_stock_runtime(
        runtime, "prompt", (), max_new_tokens=1,
        processor_pixels={"min_pixels": 1, "max_pixels": 2},
    )
    assert row["status"] == "completed"
    assert row["response_raw"] == "ok"
    assert row["peak_cuda_memory_allocated_bytes"] is None


def test_stock_invoker_reports_bounded_path_free_failure_reason():
    from fluidmix.explanation.vlm.stock_runtime import StockRuntime, invoke_stock_runtime

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        image_processor = SimpleNamespace(min_pixels=1, max_pixels=1)

        def apply_chat_template(self, *args, **kwargs):
            return Batch(input_ids=[[1, 2]])

    class Model:
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=100)

        def forward(self, input_ids):
            pass

        def generate(self, input_ids, **kwargs):
            raise RuntimeError(
                "/tmp/private-build/cuda_utils.c: fatal error: Python.h: No such file"
            )

    runtime = StockRuntime(
        repository="repo", revision="rev", snapshot=Path("."),
        processor=Processor(), model=Model(), torch_module=None,
        processor_class="Processor", model_class="Model", inventory={},
    )
    row = invoke_stock_runtime(runtime, "prompt", (), max_new_tokens=1)
    assert row["status"] == "error"
    assert row["error_type"] == "RuntimeError"
    assert row["error"] == "RuntimeError: model invocation failed"
    assert "/tmp" not in row["error"]
    assert len(row["error"]) <= 160

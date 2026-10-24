from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_contract_runner import build_attempt_plan  # noqa: E402
from fluidmix.cosmos_nano_input_audit import (  # noqa: E402
    assert_attempt_matches_audit,
    audit_json_bytes,
    audit_model_inputs,
    build_model_inputs,
)
from fluidmix.cosmos_nano_runner import (  # noqa: E402
    FatalRunError,
    decode_generated_suffix,
    execute_nano_attempt_plan,
    generate_from_audited_inputs,
    load_nano_model,
)
from fluidmix.explanation_contract import ContractError  # noqa: E402
from test_cosmos_nano_preflight import valid_config  # noqa: E402


class FakeBatch(dict):
    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs
        return self


class FakeTokenizer:
    pass


class FakeImageProcessor:
    def to_dict(self):
        return {"size": {"height": 480, "width": 640}, "rescale": True}


class FakeProcessor:
    def __init__(self):
        self.chat_template = "native-template"
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()
        self.calls = []
        self.decode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if kwargs["tokenize"] is False:
            return "rendered-user-message<|im_start|>assistant\n"
        batch = FakeBatch(
            input_ids=torch.tensor([[11, 12, 13]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.int64),
        )
        if messages[0]["content"][0]["type"] == "image":
            batch["pixel_values"] = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
            batch["image_grid_thw"] = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        return batch

    def batch_decode(self, value, **kwargs):
        self.decode_calls.append((value, kwargs))
        return ["raw completion"]


def _audit_config(processor):
    config = valid_config()
    config["expected_processor_class"] = type(processor).__name__
    config["expected_tokenizer_class"] = type(processor.tokenizer).__name__
    config["chat_template_text_sha256"] = hashlib.sha256(
        processor.chat_template.encode()
    ).hexdigest()
    config["image_preprocessor_sha256"] = hashlib.sha256(
        json.dumps(
            processor.image_processor.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return config


def _plan():
    fixtures = json.loads((ROOT / "data/cosmos_contract_fixtures_v1.json").read_text())
    prompt = (ROOT / "data/cosmos_operator_prompt_v1.txt").read_text()
    return build_attempt_plan(fixtures, valid_config(), prompt)


def test_native_builder_omits_thinking_and_all_unreviewed_template_kwargs(tmp_path):
    processor = FakeProcessor()
    spec = next(row for row in _plan() if row["mode"] == "image_evidence_recipe")

    built = build_model_inputs(
        processor,
        spec,
        image_path=tmp_path / "fixed.png",
        config=_audit_config(processor),
    )

    assert len(processor.calls) == 2
    assert processor.calls[0][1] == {"tokenize": False, "add_generation_prompt": True}
    assert processor.calls[1][1] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert all("enable_thinking" not in kwargs for _, kwargs in processor.calls)
    assert built["content_order"] == ["image", "text"]
    assert set(built["tensor_fingerprints"]) == {
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "pixel_values",
    }


def test_processor_audit_covers_exactly_eighteen_model_attempts(tmp_path):
    processor = FakeProcessor()
    config = _audit_config(processor)

    audit = audit_model_inputs(
        _plan(),
        processor,
        image_path=tmp_path / "fixed.png",
        config=config,
        observed_transformers_version="5.17.0",
    )

    assert audit["model_attempt_count"] == 18
    assert len(audit["attempts"]) == 18
    assert {row["mode"] for row in audit["attempts"]} == {
        "image_evidence_recipe",
        "evidence_recipe",
        "image_recipe",
    }
    assert audit_json_bytes(audit).endswith(b"\n")


def test_audit_comparison_fails_if_any_tensor_byte_changes(tmp_path):
    processor = FakeProcessor()
    config = _audit_config(processor)
    plan = _plan()
    audit = audit_model_inputs(
        plan,
        processor,
        image_path=tmp_path / "fixed.png",
        config=config,
        observed_transformers_version="5.17.0",
    )
    spec = next(row for row in plan if row["mode"] == "evidence_recipe")
    built = build_model_inputs(
        processor, spec, image_path=tmp_path / "fixed.png", config=config
    )

    assert_attempt_matches_audit(spec, built, audit)
    built["tensors"]["input_ids"][0, 0] = 99
    with pytest.raises(ContractError):
        assert_attempt_matches_audit(spec, built, audit)


def test_suffix_decoding_uses_prompt_width_native_eos_and_exact_cap_semantics():
    processor = FakeProcessor()
    config = valid_config()
    prompt_ids = torch.tensor([[1, 2, 3]])
    generated = torch.tensor([[1, 2, 3] + [7] * 383 + [151645]])

    result = decode_generated_suffix(processor, generated, prompt_ids, config)

    assert result["generated_token_count"] == 384
    assert result["generated_suffix_token_ids"][-1] == 151645
    assert result["terminal_token_is_eos"] is True
    assert result["length_cap_reached"] is True
    assert result["truncated_without_eos"] is False
    assert processor.decode_calls[0][1] == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    generated[0, -1] = 42
    result = decode_generated_suffix(processor, generated, prompt_ids, config)
    assert result["length_cap_reached"] is True
    assert result["truncated_without_eos"] is True


def test_generation_rechecks_audited_tensors_immediately_before_one_call(tmp_path):
    processor = FakeProcessor()
    config = _audit_config(processor)
    plan = _plan()
    spec = next(row for row in plan if row["mode"] == "evidence_recipe")
    audit = audit_model_inputs(
        plan,
        processor,
        image_path=tmp_path / "fixed.png",
        config=config,
        observed_transformers_version="5.17.0",
    )
    built = build_model_inputs(
        processor, spec, image_path=tmp_path / "fixed.png", config=config
    )

    class Model:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[11, 12, 13, 151643]])

    model = Model()
    result = generate_from_audited_inputs(
        model,
        processor,
        spec,
        built,
        audit,
        config,
        inference_context=nullcontext,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 384
    for forbidden in ("eos_token_id", "pad_token_id", "stopping_criteria", "stop_strings"):
        assert forbidden not in model.calls[0]
    assert result["generated_suffix_token_ids"] == [151643]


def test_executor_keeps_all_denominators_and_model_load_errors():
    plan = _plan()
    calls = []

    def infer(spec):
        calls.append(spec["attempt_index"])
        if len(calls) == 2:
            raise RuntimeError("retained ordinary failure")
        return {
            "response_raw": "{}",
            "response_utf8_sha256": hashlib.sha256(b"{}").hexdigest(),
            "generated_suffix_token_ids": [1],
            "generated_suffix_sha256": hashlib.sha256(b"[1]").hexdigest(),
            "generated_token_count": 1,
            "terminal_token_id": 1,
            "terminal_token_is_eos": False,
            "length_cap_reached": False,
            "truncated_without_eos": False,
        }

    rows = execute_nano_attempt_plan(plan, infer=infer)
    load_failed = execute_nano_attempt_plan(
        plan,
        infer=None,
        model_load_error={"type": "OSError", "message": "load failed"},
    )

    assert len(calls) == 18
    assert len(rows) == len(load_failed) == 24
    assert sum(row["attempt_status"] == "error" for row in rows) == 1
    assert sum(row["attempt_status"] == "error" for row in load_failed) == 18
    assert sum(row["mode"] == "template_reference" for row in load_failed) == 6


def test_executor_does_not_publish_a_fatal_memory_condition():
    def infer(spec):
        raise MemoryError("fatal allocation")

    with pytest.raises(FatalRunError):
        execute_nano_attempt_plan(_plan(), infer=infer)


def test_executor_treats_input_integrity_failure_as_incomplete_not_model_error():
    def infer(spec):
        raise ContractError("live tensors differ from audit")

    with pytest.raises(FatalRunError):
        execute_nano_attempt_plan(_plan(), infer=infer)


def test_model_loader_uses_exact_local_bf16_cuda_contract():
    config = valid_config()
    config["expected_model_class"] = "Model"
    captured = {}

    class Parameter:
        device = "cuda:0"

    class Model:
        def eval(self):
            captured["eval"] = True
            return self

        def parameters(self):
            return [Parameter()]

    class AutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return Model()

    class Torch:
        bfloat16 = "bf16"

    model = load_nano_model(AutoModel, "/offline/view", Torch, config)

    assert type(model).__name__ == "Model"
    assert captured["kwargs"] == {
        "dtype": "bf16",
        "device_map": "cuda",
        "local_files_only": True,
    }
    assert captured["eval"] is True


@pytest.mark.parametrize(
    ("script", "options"),
    [
        (
            "stage_cosmos_nano_reasoner.py",
            ("--snapshot", "--model-dir", "--model-manifest"),
        ),
        (
            "audit_cosmos_nano_inputs.py",
            (
                "--config",
                "--model-dir",
                "--model-manifest",
                "--runtime-lock",
                "--image",
                "--out",
            ),
        ),
        (
            "run_cosmos_large_eval.py",
            (
                "--config",
                "--model-dir",
                "--model-manifest",
                "--runtime-lock",
                "--image",
                "--input-audit",
            ),
        ),
        (
            "score_cosmos_nano_matched.py",
            (
                "--raw-results",
                "--config",
                "--oracle",
                "--model-dir",
                "--model-manifest",
                "--out",
            ),
        ),
    ],
)
def test_cli_help_paths_do_not_load_models(script, options):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    for option in options:
        assert option in result.stdout


def test_model_runner_cli_has_no_scoring_data_or_output_override():
    script = ROOT / "scripts/run_cosmos_large_eval.py"
    source = script.read_text()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--oracle" not in result.stdout
    assert "--out" not in result.stdout
    assert "oracle" not in source.lower()
    assert "http://" not in source and "https://" not in source

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from ...core.policy import validate_deploy_input_fields


SOURCE_KINDS = frozenset({"video", "file", "v4l2", "rtsp"})


def _reject_truth_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in {"truth", "ground_truth"}:
                raise ValueError(f"forbidden_deploy_input_field: {key}")
            _reject_truth_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_truth_fields(child)


def validate_runtime_payload(value: object) -> None:
    """Reject simulation answers at every public runtime boundary."""
    validate_deploy_input_fields(value, require_allowlist=False)
    _reject_truth_fields(value)


def validate_create_request(value: object, *, validate_video_path: bool = True) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("request must be a JSON object")
    if not set(value).issubset({"source", "run_context", "device"}):
        raise ValueError("request contains unsupported fields")
    source = value.get("source")
    context = value.get("run_context")
    device = value.get("device", "cpu")
    if not isinstance(source, Mapping) or source.get("kind") not in SOURCE_KINDS:
        raise ValueError("source kind must be file, video, v4l2, or rtsp")
    if not isinstance(context, Mapping):
        raise ValueError("run_context must be an object")
    validate_runtime_payload(context)
    if set(context) != {"operator_run_id", "recipe_id", "tray_id"}:
        raise ValueError("operator run_context fields are not closed")
    if not isinstance(context.get("operator_run_id"), str) or not context["operator_run_id"]:
        raise ValueError("operator_run_id must be nonempty text")
    if context.get("recipe_id") is not None and (
        not isinstance(context["recipe_id"], str) or not context["recipe_id"]
    ):
        raise ValueError("recipe_id must be null or nonempty text")
    if not isinstance(context.get("tray_id"), str) or not context["tray_id"]:
        raise ValueError("tray_id must be nonempty text")
    if device not in {"cpu", "gpu", "auto"}:
        raise ValueError("device must be cpu, gpu, or auto")
    kind = str(source["kind"])
    expected_source_fields = {
        "video": {"kind", "path"},
        "file": {"kind", "path", "label"},
        "v4l2": {"kind", "device"},
        "rtsp": {"kind", "url"},
    }[kind]
    if set(source) != expected_source_fields:
        raise ValueError(f"{kind} source fields are not closed")
    normalized: dict[str, object] = {"kind": kind}
    if kind in {"video", "file"}:
        raw = source.get("path")
        if not isinstance(raw, str) or not raw:
            raise ValueError("video path must be nonempty text")
        path = Path(raw).expanduser()
        if validate_video_path and (path.is_symlink() or not path.is_file()):
            raise ValueError("video path must identify a regular file")
        normalized["path"] = str(path.resolve(strict=False))
        if kind == "file":
            label = source.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError("uploaded file label must be nonempty text")
            normalized["label"] = label
    elif kind == "v4l2":
        raw = source.get("device")
        if not isinstance(raw, str) or not raw.startswith("/dev/video"):
            raise ValueError("V4L2 device must be under /dev/videoN")
        suffix = raw.removeprefix("/dev/video")
        if not suffix.isdigit():
            raise ValueError("V4L2 device must be under /dev/videoN")
        normalized["device"] = raw
    else:
        raw = source.get("url")
        parsed = urlparse(raw) if isinstance(raw, str) else None
        if parsed is None or parsed.scheme not in {"rtsp", "rtsps"} or not parsed.netloc:
            raise ValueError("RTSP source must use an rtsp:// or rtsps:// URL")
        normalized["url"] = raw
    return {"source": normalized, "run_context": dict(context), "device": device}

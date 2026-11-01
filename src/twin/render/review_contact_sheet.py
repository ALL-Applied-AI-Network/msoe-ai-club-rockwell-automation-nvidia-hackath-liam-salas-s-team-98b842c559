"""Publish receipt-bound PhysX RGB and liquid-mask review sheets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from twin.render.render_process import hash_regular_file
from twin.render.review_video import _read_rgb, _validated_source_frames
from twin.render.twin_video import _atomic_rename_noreplace, _write_private_json


_CAPTION_HEIGHT = 24


def _regular_hash(path: Path, *, role: str) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"{role} must be a regular non-symlink file")
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{role} is missing or unreadable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} must be a regular non-symlink file")
    return hash_regular_file(path)


def _resolve_bound_artifact(
    root: Path,
    relative: object,
    expected_hash: object,
    *,
    role: str,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{role} path must be a non-empty relative path")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError(f"{role} hash must be a lowercase SHA-256 value")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"{role} is a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{role} is missing or unreadable: {relative}") from error
    if not resolved.is_relative_to(root):
        raise ValueError(f"{role} escapes the receipt directory: {relative}")
    snapshot = _regular_hash(resolved, role=role)
    if snapshot["sha256"] != expected_hash:
        raise ValueError(f"{role} hash does not match receipt: {relative}")
    return resolved, snapshot


def _load_mask(path: Path, *, width: int, height: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"liquid mask is not a PNG: {path}")
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"liquid mask does not decode: {path}") from error
    if mask.shape != (height, width):
        raise ValueError(
            f"liquid mask has raster {mask.shape}, expected {(height, width)}: {path}"
        )
    if not set(int(value) for value in np.unique(mask)).issubset({0, 255}):
        raise ValueError(f"liquid mask is not binary: {path}")
    return mask


def _tint_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    selected = mask == 255
    overlay[selected] = np.array([255, 0, 255], dtype=np.uint8)
    original = rgb[selected]
    if original.size:
        already_magenta = np.all(original == [255, 0, 255], axis=1)
        overlay[selected] = np.where(
            already_magenta[:, None],
            np.array([0, 255, 255], dtype=np.uint8),
            overlay[selected],
        )
    return overlay


def create_physx_review_contact_sheet(
    receipt_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    physics_steps: tuple[int, ...],
    columns: int = 3,
    crop_xyxy: tuple[int, int, int, int] | None = None,
) -> Path:
    """Build a compact RGB/semantic overlay grid for selected receipt frames."""

    if (
        not isinstance(physics_steps, tuple)
        or not physics_steps
        or any(type(step) is not int or step < 0 for step in physics_steps)
        or len(set(physics_steps)) != len(physics_steps)
    ):
        raise ValueError("physics_steps must be unique nonnegative integers")
    if type(columns) is not int or columns <= 0:
        raise ValueError("columns must be a positive integer")
    if crop_xyxy is not None and (
        not isinstance(crop_xyxy, tuple)
        or len(crop_xyxy) != 4
        or any(type(value) is not int for value in crop_xyxy)
    ):
        raise ValueError("crop_xyxy must be four integers")

    receipt_input = Path(receipt_path).expanduser().absolute()
    receipt_snapshot = _regular_hash(receipt_input, role="receipt")
    receipt = receipt_input.resolve(strict=True)
    receipt_root = receipt.parent.resolve(strict=True)
    try:
        receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("receipt is not readable JSON") from error
    if not isinstance(receipt_value, dict):
        raise ValueError("receipt root must be an object")
    frames, width, height = _validated_source_frames(receipt_value, receipt_root)
    if crop_xyxy is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = crop_xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("crop_xyxy must be inside the receipt raster")
    view_width = x2 - x1
    view_height = y2 - y1
    frames_by_step = {record["physics_step"]: record for record in frames}
    missing = [step for step in physics_steps if step not in frames_by_step]
    if missing:
        raise ValueError(f"selected physics steps are absent from receipt: {missing!r}")

    semantics = receipt_value.get("semantics")
    if not isinstance(semantics, dict) or not isinstance(semantics.get("pairs"), list):
        raise ValueError("receipt semantics.pairs must be an array")
    pair_rows = semantics["pairs"]
    pair_by_step: dict[int, dict[str, object]] = {}
    for raw_pair in pair_rows:
        if not isinstance(raw_pair, dict):
            raise ValueError("receipt semantic-pair records are malformed")
        step = raw_pair.get("physics_step")
        if type(step) is not int or step < 0 or step in pair_by_step:
            raise ValueError("receipt semantic-pair physics steps must be unique integers")
        pair_by_step[step] = raw_pair

    selected: list[dict[str, object]] = []
    for step in physics_steps:
        frame = frames_by_step[step]
        pair = pair_by_step.get(step)
        if pair is None:
            raise ValueError(f"selected step has no semantic pair: {step}")
        if (
            pair.get("rgb_path") != frame["path"]
            or pair.get("rgb_sha256") != frame["sha256"]
        ):
            raise ValueError(f"semantic pair disagrees with render frame at step {step}")
        mask_path, mask_snapshot = _resolve_bound_artifact(
            receipt_root,
            pair.get("mask_path"),
            pair.get("mask_sha256"),
            role="liquid mask",
        )
        rgb = _read_rgb(frame["resolved_path"], width=width, height=height)
        mask = _load_mask(mask_path, width=width, height=height)
        pixel_count = int(np.count_nonzero(mask == 255))
        if pair.get("liquid_pixel_count") != pixel_count:
            raise ValueError(f"liquid-pixel count disagrees with mask at step {step}")
        selected.append(
            {
                "ordinal": frame["ordinal"],
                "physics_step": step,
                "phase": pair.get("phase"),
                "rgb_path": frame["path"],
                "rgb_sha256": frame["sha256"],
                "mask_path": pair["mask_path"],
                "mask_sha256": mask_snapshot["sha256"],
                "liquid_pixel_count": pixel_count,
                "rgb": rgb,
                "mask": mask,
            }
        )

    output_candidate = Path(output_dir).expanduser()
    output_parent = output_candidate.parent.resolve(strict=True)
    if not output_parent.is_dir():
        raise ValueError("contact-sheet output parent must be a directory")
    if not output_candidate.name or output_candidate.name in {".", ".."}:
        raise ValueError("contact-sheet output must name one directory")
    output = output_parent / output_candidate.name
    if os.path.lexists(output):
        raise FileExistsError(output)
    if output.is_relative_to(receipt_root) or receipt_root.is_relative_to(output):
        raise ValueError("contact-sheet output must be outside the receipt tree")

    rows = math.ceil(len(selected) / columns)
    tile_height = _CAPTION_HEIGHT + 2 * view_height
    sheet = Image.new("RGB", (columns * view_width, rows * tile_height), "white")
    for index, record in enumerate(selected):
        tile = Image.new("RGB", (view_width, tile_height), "white")
        label = (
            f"step {record['physics_step']} | {record['phase']} | "
            f"liquid pixels {record['liquid_pixel_count']}"
        )
        ImageDraw.Draw(tile).text((2, 2), label, fill="black")
        rgb_crop = record["rgb"][y1:y2, x1:x2]
        mask_crop = record["mask"][y1:y2, x1:x2]
        rgb_image = Image.fromarray(rgb_crop, mode="RGB")
        overlay = Image.fromarray(
            _tint_mask(rgb_crop, mask_crop), mode="RGB"
        )
        tile.paste(rgb_image, (0, _CAPTION_HEIGHT))
        tile.paste(overlay, (0, _CAPTION_HEIGHT + view_height))
        x = (index % columns) * view_width
        y = (index // columns) * tile_height
        sheet.paste(tile, (x, y))

    partial: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output_parent)
    )
    os.chmod(partial, 0o700)
    try:
        image_path = partial / "contact-sheet.png"
        sheet.save(image_path, format="PNG")
        os.chmod(image_path, 0o600)
        with Image.open(image_path) as decoded:
            decoded.load()
            if decoded.format != "PNG" or decoded.mode != "RGB" or decoded.size != sheet.size:
                raise RuntimeError("published contact sheet did not decode as authored")
        image_snapshot = _regular_hash(image_path, role="contact sheet")

        receipt_after = _regular_hash(receipt, role="receipt")
        if receipt_after != receipt_snapshot:
            raise RuntimeError("source receipt changed while contact sheet was built")
        source_records = [
            {
                key: record[key]
                for key in (
                    "ordinal",
                    "physics_step",
                    "phase",
                    "rgb_path",
                    "rgb_sha256",
                    "mask_path",
                    "mask_sha256",
                    "liquid_pixel_count",
                )
            }
            for record in selected
        ]
        source_order_sha256 = hashlib.sha256(
            json.dumps(source_records, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        selection: dict[str, Any] = {
            "physics_steps": list(physics_steps),
            "columns": columns,
            "rows": rows,
        }
        if crop_xyxy is not None:
            selection["crop_xyxy_exclusive"] = list(crop_xyxy)
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "contact_sheet_verified",
            "source_receipt": {"path": str(receipt), **receipt_snapshot},
            "selection": selection,
            "source_frames": source_records,
            "source_order_sha256": source_order_sha256,
            "contact_sheet": {
                "path": "contact-sheet.png",
                "width": sheet.width,
                "height": sheet.height,
                **image_snapshot,
            },
            "presentation": {
                "upper_panel": "receipt-bound RGB PNG",
                "lower_panel": "same RGB with receipt-bound liquid mask tinted magenta",
                "caption_height_px": _CAPTION_HEIGHT,
            },
            "limits": {
                "role": "qualitative visual review only",
                "physical_claim": "none; this artifact does not validate simulation physics",
            },
        }
        _write_private_json(partial / "contact-sheet.json", record)
        directory_fd = os.open(partial, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _atomic_rename_noreplace(partial, output)
        partial = None
        return output / "contact-sheet.json"
    finally:
        if partial is not None and partial.exists():
            metadata = os.lstat(partial)
            if stat.S_ISDIR(metadata.st_mode) and partial.parent == output_parent:
                shutil.rmtree(partial)

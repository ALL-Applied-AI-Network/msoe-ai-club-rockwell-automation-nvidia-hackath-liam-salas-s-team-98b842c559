"""Build review reels separated by explicit per-run palette cut cards.

This encoder never synthesizes transport. It decodes already accepted run
videos and places a visible, fixed-duration card between each pair of runs.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from twin.render.render_process import hash_regular_file
from twin.render.twin_video import _atomic_rename_noreplace, _write_private_json


CUT_TEXT_LINES = (
    "NEW RUN — TANK COLOURS CHANGE",
    "Transport does not cross this palette boundary.",
)
CUT_DURATION_SECONDS = 2
_VIDEO_NAME = "palette-cut-reel.mp4"
_RECEIPT_NAME = "palette-cut-reel.json"
_INVENTORY_NAME = "SHA256SUMS"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_SOURCE_FRAME_MAE = 12.0
_MIN_SOURCE_IDENTITY_MARGIN = 2.0
_LITERAL_IDENTITY_MAGIC = b"FLUIDMIX-CUT-TEXT\x01"
PALETTE_CUT_REEL_ARTIFACT_TYPE = "fluidmix_palette_cut_reel_v2"
PART_A_REEL_ROOT_RELATIVE_PATH = "outputs/part-a/palette-cut-reels"


def validate_part_a_artifact_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    artifact_type: str,
) -> tuple[Path, Path]:
    """Require a palette reel below the twin package's classified output root."""

    if artifact_type != PALETTE_CUT_REEL_ARTIFACT_TYPE:
        raise ValueError(f"unregistered Part A artifact type: {artifact_type}")
    repository = Path(repo_root).expanduser().resolve(strict=True)
    root = (repository / PART_A_REEL_ROOT_RELATIVE_PATH).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("classified Part A output root must be a directory")
    output = Path(output_path).expanduser()
    parent = output.parent.resolve(strict=True)
    candidate = parent / output.name
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("reel output must be below its classified Part A output root")
    return candidate, root


def _load_av() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise RuntimeError("PyAV is required to encode a palette-cut reel") from error
    return av


def _regular_file(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.lstat(resolved)
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    return resolved


def _read_request(path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    request_path = _regular_file(path, "reel request")
    request_snapshot = hash_regular_file(request_path)
    try:
        value = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("reel request is not readable JSON") from error
    if not isinstance(value, dict):
        raise ValueError("reel request must contain one JSON object")
    return request_path, value, request_snapshot


def _bound_path(
    raw: object, *, relative_to: Path, label: str, must_be_child: bool = False
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path must be a non-empty string")
    supplied = Path(raw)
    if not supplied.is_absolute():
        supplied = relative_to / supplied
    path = _regular_file(supplied, label)
    if must_be_child and not path.is_relative_to(relative_to):
        raise ValueError(f"{label} escapes its accepted package root")
    return path


def _read_bound_json(path: Path, expected_hash: object, label: str) -> dict[str, object]:
    if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
        raise ValueError(f"{label} SHA-256 must be exact lowercase hex")
    if hash_regular_file(path)["sha256"] != expected_hash:
        raise ValueError(f"{label} SHA-256 drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _accepted_package(
    receipt_path: Path, expected_hash: object
) -> dict[str, object]:
    package_root = receipt_path.parent.resolve(strict=True)
    if receipt_path != package_root / "package-receipt.json":
        raise ValueError("accepted package receipt must be package-receipt.json at package root")
    receipt = _read_bound_json(receipt_path, expected_hash, "accepted package receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type")
        != "synthetic_run_sequence_dataset_package_v1"
        or receipt.get("status") != "pass"
        or receipt.get("cohort_version") not in {"v3", "v4"}
        or receipt.get("output_root") != os.fspath(package_root)
        or receipt.get("qualified_fill_receipts_changed") is not False
        or receipt.get("transport_mode") != "immutable_13_state_usd_overlay"
    ):
        raise ValueError("accepted package receipt is not a passing V3/V4 run package")
    sums = _bound_path(
        receipt.get("sha256sums_path"),
        relative_to=package_root,
        label="accepted package SHA inventory",
        must_be_child=True,
    )
    if sums != package_root / "SHA256SUMS":
        raise ValueError("accepted package SHA inventory path is noncanonical")
    if hash_regular_file(sums)["sha256"] != receipt.get("sha256sums_sha256"):
        raise ValueError("accepted package SHA inventory drifted")
    inventory: dict[str, str] = {}
    try:
        lines = sums.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("accepted package SHA inventory is unreadable") from error
    for line in lines:
        digest, separator, relative = line.partition("  ")
        relative_path = Path(relative)
        if (
            separator != "  "
            or _SHA256.fullmatch(digest) is None
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in inventory
        ):
            raise ValueError("accepted package SHA inventory is malformed")
        artifact = _regular_file(package_root / relative_path, "accepted package payload")
        if not artifact.is_relative_to(package_root):
            raise ValueError("accepted package payload escapes its package root")
        if hash_regular_file(artifact)["sha256"] != digest:
            raise ValueError(f"accepted package payload hash drifted: {relative}")
        inventory[relative_path.as_posix()] = digest
    if (
        len(inventory) != receipt.get("hashed_payload_file_count")
        or not inventory
    ):
        raise ValueError("accepted package SHA inventory count is invalid")
    index = _regular_file(package_root / "videos/INDEX.csv", "accepted package index")
    index_relative = index.relative_to(package_root).as_posix()
    if inventory.get(index_relative) != hash_regular_file(index)["sha256"]:
        raise ValueError("accepted package index is absent from its SHA inventory")
    run_receipts = receipt.get("run_video_receipts")
    if not isinstance(run_receipts, list) or not run_receipts:
        raise ValueError("accepted package lacks embedded run receipts")
    return {
        "root": package_root,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": expected_hash,
        "inventory_path": sums,
        "inventory_sha256": receipt["sha256sums_sha256"],
        "inventory": inventory,
        "index_path": index,
        "embedded_run_receipts": run_receipts,
    }


def _canonical_palette_sha256(transmission: Mapping[str, object]) -> str:
    encoded = json.dumps(
        transmission, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _accepted_run_evidence(
    row: Mapping[str, object],
    *,
    request_root: Path,
    package_cache: dict[tuple[Path, str], dict[str, object]],
) -> dict[str, object]:
    package_receipt = _bound_path(
        row.get("package_receipt_path"),
        relative_to=request_root,
        label="accepted package receipt",
    )
    package_hash = row.get("package_receipt_sha256")
    if not isinstance(package_hash, str) or _SHA256.fullmatch(package_hash) is None:
        raise ValueError("accepted package receipt SHA-256 must be exact lowercase hex")
    cache_key = (package_receipt, package_hash)
    package = package_cache.get(cache_key)
    if package is None:
        package = _accepted_package(package_receipt, package_hash)
        package_cache[cache_key] = package
    package_root = package["root"]
    if not isinstance(package_root, Path):  # pragma: no cover - internal invariant
        raise RuntimeError("accepted package root invariant failed")

    run_id = str(row["run_id"])
    run_receipt_path = _bound_path(
        row.get("run_receipt_path"),
        relative_to=request_root,
        label=f"accepted run receipt {run_id}",
        must_be_child=False,
    )
    if not run_receipt_path.is_relative_to(package_root):
        raise ValueError("accepted run receipt escapes its package root")
    run_receipt_hash = row.get("run_receipt_sha256")
    run_receipt = _read_bound_json(
        run_receipt_path, run_receipt_hash, f"accepted run receipt {run_id}"
    )
    relative_receipt = run_receipt_path.relative_to(package_root).as_posix()
    inventory = package["inventory"]
    if not isinstance(inventory, dict) or inventory.get(relative_receipt) != run_receipt_hash:
        raise ValueError("accepted run receipt is absent from package SHA inventory")
    if (
        run_receipt.get("schema_version") != 2
        or run_receipt.get("artifact_type") != "rendered_transport_run_sequence_v2"
        or run_receipt.get("status") != "pass"
        or run_receipt.get("run_id") != run_id
        or run_receipt.get("qualified_fill_receipts_changed") is not False
        or run_receipt.get("camera_and_lighting_changed") is not False
    ):
        raise ValueError(f"accepted run receipt is not a passing immutable run: {run_id}")
    embedded = package["embedded_run_receipts"]
    matching = [
        candidate
        for candidate in embedded
        if isinstance(candidate, dict) and candidate.get("run_id") == run_id
    ]
    if len(matching) != 1 or matching[0] != run_receipt:
        raise ValueError("accepted run receipt differs from package-embedded evidence")

    video = Path(str(row["video_path"]))
    if not video.is_relative_to(package_root):
        raise ValueError("accepted run video escapes its package root")
    relative_video = video.relative_to(package_root).as_posix()
    if (
        run_receipt.get("output_video") != os.fspath(video)
        or run_receipt.get("output_video_sha256") != row["video_sha256"]
        or inventory.get(relative_video) != row["video_sha256"]
    ):
        raise ValueError("accepted run receipt/package inventory video binding differs")

    index_path = package["index_path"]
    if not isinstance(index_path, Path):  # pragma: no cover - internal invariant
        raise RuntimeError("accepted package index invariant failed")
    try:
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            run_rows = [item for item in csv.DictReader(handle) if item.get("run_id") == run_id]
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError("accepted package index is unreadable") from error
    if len(run_rows) != 3 or [item.get("vial_ordinal") for item in run_rows] != ["1", "2", "3"]:
        raise ValueError("accepted package index lacks the exact three ordered run rows")
    if {item.get("run_video_path") for item in run_rows} != {relative_video}:
        raise ValueError("accepted package index run-video binding differs")
    palette_hashes = {item.get("palette_sha256") for item in run_rows}
    if len(palette_hashes) != 1:
        raise ValueError("accepted package index has inconsistent run palette identity")
    palette_hash = next(iter(palette_hashes))
    if not isinstance(palette_hash, str) or _SHA256.fullmatch(palette_hash) is None:
        raise ValueError("accepted package index palette SHA-256 is invalid")
    palette_fields = {
        "t100": "palette_t100_rgb",
        "t200": "palette_t200_rgb",
        "t400": "palette_t400_rgb",
    }
    transmission: dict[str, object] = {}
    for tank, field in palette_fields.items():
        values = {item.get(field) for item in run_rows}
        if len(values) != 1:
            raise ValueError("accepted package index has inconsistent palette RGB values")
        try:
            parsed = json.loads(next(iter(values)))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("accepted package index palette RGB is malformed") from error
        if (
            not isinstance(parsed, list)
            or len(parsed) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                for value in parsed
            )
        ):
            raise ValueError("accepted package index palette RGB is invalid")
        transmission[tank] = parsed
    if _canonical_palette_sha256(transmission) != palette_hash:
        raise ValueError("accepted package index palette SHA-256 differs from its RGB values")
    return {
        "package_receipt_path": os.fspath(package_receipt),
        "package_receipt_sha256": package_hash,
        "package_sha256sums_path": os.fspath(package["inventory_path"]),
        "package_sha256sums_sha256": package["inventory_sha256"],
        "run_receipt_path": os.fspath(run_receipt_path),
        "run_receipt_sha256": run_receipt_hash,
        "palette_sha256": palette_hash,
        "palette_transmission_1cm": transmission,
    }


def _validate_request(
    request: Mapping[str, object], request_root: Path
) -> list[dict[str, object]]:
    if (
        request.get("schema_version") != 1
        or request.get("artifact_type")
        != "fluidmix_palette_cut_reel_request_v1"
    ):
        raise ValueError("reel request schema/artifact type is invalid")
    rows = request.get("runs")
    run_order = request.get("run_order")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("reel request requires at least two ordered run inputs")
    if not isinstance(run_order, list) or len(run_order) != len(rows):
        raise ValueError("run_order must list every run exactly once")

    validated: list[dict[str, object]] = []
    run_ids: list[str] = []
    resolved_paths: list[Path] = []
    hashes: list[str] = []
    for expected_ordinal, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("every run input must be an object")
        if row.get("ordinal") != expected_ordinal:
            raise ValueError("run input ordinals must be consecutive and match list order")
        run_id = row.get("run_id")
        raw_path = row.get("video_path")
        expected_hash = row.get("video_sha256")
        if (
            not isinstance(run_id, str)
            or not run_id
            or any(character.isspace() for character in run_id)
        ):
            raise ValueError("run_id must be a non-empty token")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("video_path must be a non-empty string")
        if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
            raise ValueError("video_sha256 must be an exact lowercase SHA-256")
        supplied = Path(raw_path)
        if not supplied.is_absolute():
            supplied = request_root / supplied
        resolved = _regular_file(supplied, f"run video {run_id}")
        snapshot = hash_regular_file(resolved)
        if snapshot["sha256"] != expected_hash:
            raise ValueError(f"run video SHA-256 drifted for {run_id}")
        run_ids.append(run_id)
        resolved_paths.append(resolved)
        hashes.append(expected_hash)
        validated.append(
            {
                "ordinal": expected_ordinal,
                "run_id": run_id,
                "video_path": os.fspath(resolved),
                "video_sha256": expected_hash,
                "video_bytes": snapshot["bytes"],
                "package_receipt_path": row.get("package_receipt_path"),
                "package_receipt_sha256": row.get("package_receipt_sha256"),
                "run_receipt_path": row.get("run_receipt_path"),
                "run_receipt_sha256": row.get("run_receipt_sha256"),
            }
        )

    if run_order != run_ids:
        raise ValueError("run_order does not match the ordered run inputs")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run_id inputs are not allowed")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("duplicate run video paths are not allowed")
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate run video content hashes are not allowed")
    package_cache: dict[tuple[Path, str], dict[str, object]] = {}
    for record in validated:
        record.update(
            _accepted_run_evidence(
                record,
                request_root=request_root,
                package_cache=package_cache,
            )
        )
    palettes = [str(record["palette_sha256"]) for record in validated]
    if any(left == right for left, right in zip(palettes, palettes[1:])):
        raise ValueError("adjacent accepted run palettes must differ")
    return validated


def _rate(stream: Any, run_id: str) -> Fraction:
    raw = stream.average_rate
    if raw is None:
        raise ValueError(f"run video has no declared average frame rate: {run_id}")
    rate = Fraction(raw)
    if rate <= 0:
        raise ValueError(f"run video has a non-positive frame rate: {run_id}")
    if (rate * CUT_DURATION_SECONDS).denominator != 1:
        raise ValueError(
            f"run video fps cannot represent the fixed two-second cut card: {run_id}"
        )
    return rate


def _known_media_profile(stream: Any, run_id: str) -> dict[str, str]:
    values = {
        "codec_name": stream.codec_context.codec.name,
        "codec_profile": stream.codec_context.profile,
        "pixel_format": stream.codec_context.pix_fmt,
    }
    for field, value in values.items():
        if (
            not isinstance(value, str)
            or not value.strip()
            or value.strip().lower() in {"none", "unknown", "unspecified"}
        ):
            raise ValueError(f"run video has missing or unknown {field}: {run_id}")
    return values


def _validated_cfr_timing(
    stream: Any, decoded: list[tuple[int | None, Fraction | None, int | None]],
    *,
    rate: Fraction,
    run_id: str,
) -> dict[str, object]:
    if not decoded:
        raise ValueError(f"run video contains no decodable frames: {run_id}")
    normalized: list[Fraction] = []
    for pts, time_base, duration in decoded:
        if pts is None or time_base is None:
            raise ValueError(
                f"run video lacks contiguous constant-frame-rate timestamps: {run_id}"
            )
        position = Fraction(pts) * time_base * rate
        if position.denominator != 1:
            raise ValueError(
                f"run video lacks contiguous constant-frame-rate timestamps: {run_id}"
            )
        if duration is not None and Fraction(duration) * time_base * rate != 1:
            raise ValueError(
                f"run video lacks contiguous constant-frame-rate timestamps: {run_id}"
            )
        normalized.append(position)
    start = normalized[0]
    if normalized != [start + index for index in range(len(decoded))]:
        raise ValueError(
            f"run video lacks contiguous constant-frame-rate timestamps: {run_id}"
        )
    if (
        stream.duration is None
        or stream.time_base is None
        or Fraction(stream.duration) * stream.time_base * rate != len(decoded)
    ):
        raise ValueError(
            f"run video lacks exact constant-frame-rate duration: {run_id}"
        )
    return {
        "first_pts_at_rate": str(start),
        "duration_seconds": str(Fraction(len(decoded), 1) / rate),
        "stream_time_base": str(stream.time_base),
        "contiguous_cfr": True,
    }


def _probe(av: Any, record: Mapping[str, object]) -> dict[str, object]:
    path = str(record["video_path"])
    run_id = str(record["run_id"])
    try:
        with av.open(path, mode="r") as container:
            if len(container.streams.video) != 1:
                raise ValueError(f"run video must have exactly one video stream: {run_id}")
            if len(container.streams.audio) != 0:
                raise ValueError(f"run video must be silent: {run_id}")
            stream = container.streams.video[0]
            rate = _rate(stream, run_id)
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            profile = _known_media_profile(stream, run_id)
            count = 0
            decoded_timing: list[tuple[int | None, Fraction | None, int | None]] = []
            for frame in container.decode(stream):
                if frame.width != width or frame.height != height:
                    raise ValueError(f"run video geometry changes while decoding: {run_id}")
                frame.to_ndarray(format="rgb24")
                decoded_timing.append((frame.pts, frame.time_base, frame.duration))
                count += 1
            timing = _validated_cfr_timing(
                stream, decoded_timing, rate=rate, run_id=run_id
            )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"run video is not fully readable: {run_id}") from error
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"run video geometry must be positive and even: {run_id}")
    if count <= 0:
        raise ValueError(f"run video contains no decodable frames: {run_id}")
    return {
        "fps": str(rate),
        "width": width,
        "height": height,
        "profile": profile,
        "frame_count": count,
        "timing": timing,
    }


def _font_that_fits(draw: ImageDraw.ImageDraw, text: str, width: int, maximum: int):
    for size in range(maximum, 9, -1):
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        bounds = draw.textbbox((0, 0), text, font=font)
        if bounds[2] - bounds[0] <= int(width * 0.9):
            return font
    raise RuntimeError("video geometry is too small for the required cut-card text")


def _render_cut_card(
    width: int, height: int, text_lines: tuple[str, str]
) -> np.ndarray:
    image = Image.new("RGB", (width, height), color=(8, 18, 34))
    draw = ImageDraw.Draw(image)
    maximum = max(10, min(height // 5, width // 9))
    fonts = [
        _font_that_fits(draw, text, width, maximum if index == 0 else maximum - 3)
        for index, text in enumerate(text_lines)
    ]
    bounds = [
        draw.textbbox((0, 0), text, font=font)
        for text, font in zip(text_lines, fonts, strict=True)
    ]
    heights = [bottom - top for _, top, _, bottom in bounds]
    gap = max(8, height // 14)
    total = sum(heights) + gap
    y = (height - total) // 2
    stripe = max(4, height // 22)
    draw.rectangle((0, 0, width - 1, stripe), fill=(255, 174, 0))
    draw.rectangle((0, height - stripe, width - 1, height - 1), fill=(255, 174, 0))
    for index, (text, font, bound, line_height) in enumerate(
        zip(text_lines, fonts, bounds, heights, strict=True)
    ):
        line_width = bound[2] - bound[0]
        color = (255, 204, 64) if index == 0 else (245, 248, 252)
        draw.text(((width - line_width) // 2, y - bound[1]), text, font=font, fill=color)
        y += line_height + gap
    _draw_literal_identity(draw, width, height, stripe, text_lines)
    return np.asarray(image, dtype=np.uint8)


def _cut_card(width: int, height: int) -> np.ndarray:
    return _render_cut_card(width, height, CUT_TEXT_LINES)


def _literal_identity_payload(text_lines: tuple[str, str]) -> bytes:
    """Return an injective byte representation of the two visible literals."""

    encoded_lines = [line.encode("utf-8", errors="strict") for line in text_lines]
    if any(len(line) > 65535 for line in encoded_lines):
        raise RuntimeError("cut-card literal is too long for its identity raster")
    return b"".join(
        [
            _LITERAL_IDENTITY_MAGIC,
            len(encoded_lines).to_bytes(1, "big"),
            *(
                len(line).to_bytes(2, "big") + line
                for line in encoded_lines
            ),
        ]
    )


def _draw_literal_identity(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    stripe: int,
    text_lines: tuple[str, str],
) -> None:
    """Draw every literal byte, including lengths, as a visible binary raster.

    This is not a hash: the complete UTF-8 payload is rendered.  Therefore two
    supported literal tuples cannot share the same identity raster, even when
    their font glyphs happen to look alike (for example lowercase ``l`` and
    uppercase ``I`` at a small video resolution).
    """

    payload = _literal_identity_payload(text_lines)
    bits = [
        (byte >> shift) & 1
        for byte in payload
        for shift in range(7, -1, -1)
    ]
    cell = 2
    margin = 8
    columns = (width - 2 * margin) // cell
    if columns <= 0:
        raise RuntimeError("video geometry is too small for cut-card literal identity")
    rows = (len(bits) + columns - 1) // columns
    band_height = rows * cell
    bottom = height - stripe - 2
    top = bottom - band_height
    if top <= stripe + 1:
        raise RuntimeError("video geometry is too small for cut-card literal identity")
    draw.rectangle(
        (margin - 1, top - 1, margin + columns * cell, bottom),
        fill=(4, 10, 20),
    )
    for index, bit in enumerate(bits):
        row, column = divmod(index, columns)
        left = margin + column * cell
        upper = top + row * cell
        color = (250, 250, 250) if bit else (38, 64, 92)
        draw.rectangle(
            (left, upper, left + cell - 1, upper + cell - 1),
            fill=color,
        )


_LOCAL_BEHAVIOR_DEPENDENCIES = (
    "cli/twin/render/encode_palette_cut_reel.py",
    "src/twin/render/palette_cut_reel.py",
    "src/twin/render/render_process.py",
    "src/twin/render/twin_video.py",
)


def _git_identity(repo_root: Path) -> tuple[str, str, list[str]]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        tree = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{tree}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("repo_root does not identify a readable Git worktree") from error
    head_value = head.stdout.strip()
    tree_value = tree.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", head_value) is None:
        raise ValueError("Git HEAD is not a full lowercase commit hash")
    if re.fullmatch(r"[0-9a-f]{40}", tree_value) is None:
        raise ValueError("Git HEAD tree is not a full lowercase object hash")
    dirty_rows = status.stdout.splitlines()
    return head_value, tree_value, dirty_rows


def _source_provenance(repo_root: Path) -> dict[str, object]:
    head_value, tree_value, dirty_rows = _git_identity(repo_root)
    if dirty_rows:
        raise ValueError(
            "palette-cut reel execution requires a clean Git worktree; dirty: "
            + "; ".join(dirty_rows)
        )
    dependency_closure = []
    for relative in _LOCAL_BEHAVIOR_DEPENDENCIES:
        path = _regular_file(repo_root / relative, "local behavior dependency")
        if not path.is_relative_to(repo_root):
            raise ValueError("local behavior dependency escapes repo_root")
        dependency_closure.append(
            {"path": relative, **hash_regular_file(path)}
        )
    return {
        "commit": head_value,
        "tree": tree_value,
        "worktree_dirty": False,
        "dirty_status_rows": dirty_rows,
        "local_dependency_closure": dependency_closure,
    }


def _encode_reel_stream(
    av: Any,
    records: list[dict[str, object]],
    output: Path,
    *,
    rate: Fraction,
    width: int,
    height: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    card_pixels = _cut_card(width, height)
    cut_frame_count = int(rate * CUT_DURATION_SECONDS)
    cuts: list[dict[str, object]] = []
    output_index = 0
    with av.open(
        os.fspath(output), mode="w", format="mp4", options={"movflags": "+faststart"}
    ) as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(rate.denominator, rate.numerator)
        stream.codec_context.time_base = stream.time_base
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {"crf": "12", "preset": "medium"}

        def write_frame(pixels: np.ndarray) -> None:
            nonlocal output_index
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = output_index
            frame.time_base = stream.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
            output_index += 1

        for run_index, record in enumerate(records):
            with av.open(str(record["video_path"]), mode="r") as source:
                source_stream = source.streams.video[0]
                for decoded in source.decode(source_stream):
                    write_frame(decoded.to_ndarray(format="rgb24"))
            if run_index < len(records) - 1:
                start = output_index
                for _ in range(cut_frame_count):
                    write_frame(card_pixels)
                cuts.append(
                    {
                        "boundary_ordinal": run_index + 1,
                        "after_run_id": records[run_index]["run_id"],
                        "before_run_id": records[run_index + 1]["run_id"],
                        "start_frame_index": start,
                        "stop_frame_index": output_index,
                        "frame_count": cut_frame_count,
                        "text_lines": list(CUT_TEXT_LINES),
                        "transport_crosses_boundary": False,
                    }
                )
        for packet in stream.encode():
            container.mux(packet)
        encoded_profile = {
            "codec_name": stream.codec_context.codec.name,
            "codec_profile": stream.codec_context.profile,
            "pixel_format": stream.codec_context.pix_fmt,
        }
    return cuts, {"frame_count": output_index, "profile": encoded_profile}


def _encode(
    av: Any,
    records: list[dict[str, object]],
    output: Path,
    *,
    rate: Fraction,
    width: int,
    height: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return _encode_reel_stream(
        av,
        records,
        output,
        rate=rate,
        width=width,
        height=height,
    )


def _canonical_decoded_cut_cards(
    av: Any,
    records: list[dict[str, object]],
    cuts: list[dict[str, object]],
    *,
    rate: Fraction,
    width: int,
    height: int,
    temporary_parent: Path,
) -> list[list[np.ndarray]]:
    """Independently encode and decode the exact declared reel card sequence."""

    with tempfile.TemporaryDirectory(
        prefix=".canonical-cut-card-", dir=temporary_parent
    ) as temporary:
        reference_path = Path(temporary) / "reference.mp4"
        reference_cuts, _encoding = _encode_reel_stream(
            av,
            records,
            reference_path,
            rate=rate,
            width=width,
            height=height,
        )
        if reference_cuts != cuts:
            raise RuntimeError("canonical cut-card sequence differs from declaration")
        decoded_cards: list[list[np.ndarray]] = [[] for _ in cuts]
        with av.open(os.fspath(reference_path), mode="r") as container:
            if len(container.streams.video) != 1 or len(container.streams.audio) != 0:
                raise RuntimeError("canonical cut-card reference profile is invalid")
            stream = container.streams.video[0]
            profile = _known_media_profile(stream, "canonical cut-card reference")
            if (
                profile["codec_name"] != "h264"
                or profile["pixel_format"] != "yuv420p"
                or Fraction(stream.average_rate) != rate
            ):
                raise RuntimeError("canonical cut-card reference is not H.264/yuv420p CFR")
            cut_by_frame = {
                frame_index: cut_index
                for cut_index, cut in enumerate(cuts)
                for frame_index in range(
                    int(cut["start_frame_index"]), int(cut["stop_frame_index"])
                )
            }
            decoded_count = 0
            for decoded_count, frame in enumerate(container.decode(stream), start=1):
                frame_index = decoded_count - 1
                cut_index = cut_by_frame.get(frame_index)
                if cut_index is not None:
                    decoded_cards[cut_index].append(
                        frame.to_ndarray(format="rgb24")
                    )
        expected_count = sum(int(record["frame_count"]) for record in records) + sum(
            int(cut["frame_count"]) for cut in cuts
        )
        if decoded_count != expected_count:
            raise RuntimeError("canonical cut-card reference frame count drifted")
        if any(
            len(frames) != int(cut["frame_count"])
            for frames, cut in zip(decoded_cards, cuts, strict=True)
        ):
            raise RuntimeError("canonical cut-card reference boundary drifted")
        return decoded_cards


def _verify_output(
    av: Any,
    path: Path,
    *,
    records: list[dict[str, object]],
    cuts: list[dict[str, object]],
    expected_count: int,
    rate: Fraction,
    width: int,
    height: int,
) -> dict[str, object]:
    canonical_cut_cards = _canonical_decoded_cut_cards(
        av,
        records,
        cuts,
        rate=rate,
        width=width,
        height=height,
        temporary_parent=path.parent,
    )
    try:
        with av.open(os.fspath(path), mode="r") as container:
            if len(container.streams.video) != 1 or len(container.streams.audio) != 0:
                raise RuntimeError("encoded reel must contain one silent video stream")
            stream = container.streams.video[0]
            if Fraction(stream.average_rate) != rate:
                raise RuntimeError("encoded reel frame rate differs from its inputs")
            profile = _known_media_profile(stream, "encoded reel")
            if profile["codec_name"] != "h264" or profile["pixel_format"] != "yuv420p":
                raise RuntimeError("encoded reel is not required H.264/yuv420p")
            output_frames = iter(container.decode(stream))
            timing_rows: list[tuple[int | None, Fraction | None, int | None]] = []
            source_checks: list[dict[str, object]] = []
            cut_checks: list[dict[str, object]] = []
            count = 0

            def next_output() -> np.ndarray:
                nonlocal count
                try:
                    frame = next(output_frames)
                except StopIteration as error:
                    raise RuntimeError("encoded reel ended before its declared sequence") from error
                if frame.width != width or frame.height != height:
                    raise RuntimeError("encoded reel geometry differs from its inputs")
                timing_rows.append((frame.pts, frame.time_base, frame.duration))
                count += 1
                return frame.to_ndarray(format="rgb24")

            expected_card = _cut_card(width, height)
            expected_card_hash = hashlib.sha256(
                np.ascontiguousarray(expected_card).tobytes()
            ).hexdigest()
            if float(np.std(expected_card)) < 20.0:
                raise RuntimeError("required cut card lacks visible contrast")
            for run_index, record in enumerate(records):
                errors: list[float] = []
                actual_segment: list[np.ndarray] = []
                with av.open(str(record["video_path"]), mode="r") as source:
                    source_stream = source.streams.video[0]
                    for expected in source.decode(source_stream):
                        actual = next_output().astype(np.int16)
                        actual_segment.append(actual)
                        reference = expected.to_ndarray(format="rgb24").astype(np.int16)
                        errors.append(float(np.mean(np.abs(actual - reference))))
                if len(errors) != record["frame_count"] or not errors:
                    raise RuntimeError("encoded reel source segment frame count drifted")
                if max(errors) > _MAX_SOURCE_FRAME_MAE:
                    raise RuntimeError(
                        f"encoded reel source order/content verification failed: {record['run_id']}"
                    )
                candidate_distances: list[tuple[str, float]] = []
                for candidate in records:
                    if candidate["frame_count"] != len(actual_segment):
                        continue
                    candidate_errors: list[float] = []
                    with av.open(str(candidate["video_path"]), mode="r") as source:
                        source_stream = source.streams.video[0]
                        for actual, expected in zip(
                            actual_segment, source.decode(source_stream), strict=True
                        ):
                            reference = expected.to_ndarray(format="rgb24").astype(np.int16)
                            candidate_errors.append(
                                float(np.mean(np.abs(actual - reference)))
                            )
                    candidate_distances.append(
                        (str(candidate["run_id"]), float(np.mean(candidate_errors)))
                    )
                candidate_distances.sort(key=lambda item: (item[1], item[0]))
                nearest_id, nearest_distance = candidate_distances[0]
                second_distance = (
                    candidate_distances[1][1]
                    if len(candidate_distances) > 1
                    else float("inf")
                )
                identity_margin = second_distance - nearest_distance
                if nearest_id != record["run_id"]:
                    raise RuntimeError(
                        f"encoded reel source identity/order differs: {record['run_id']}"
                    )
                if identity_margin < _MIN_SOURCE_IDENTITY_MARGIN:
                    raise RuntimeError(
                        f"encoded reel source runs are not separable enough to prove order: {record['run_id']}"
                    )
                source_checks.append(
                    {
                        "run_id": record["run_id"],
                        "start_frame_index": count - len(errors),
                        "stop_frame_index": count,
                        "frame_count": len(errors),
                        "maximum_mean_absolute_error": max(errors),
                        "mean_mean_absolute_error": float(np.mean(errors)),
                        "maximum_allowed_mean_absolute_error": _MAX_SOURCE_FRAME_MAE,
                        "candidate_sequence_mean_absolute_error": {
                            run_id: distance for run_id, distance in candidate_distances
                        },
                        "nearest_run_id": nearest_id,
                        "nearest_distance": nearest_distance,
                        "identity_margin_to_second_nearest": identity_margin,
                        "minimum_required_identity_margin": _MIN_SOURCE_IDENTITY_MARGIN,
                    }
                )
                if run_index < len(cuts):
                    cut = cuts[run_index]
                    if count != cut["start_frame_index"]:
                        raise RuntimeError("encoded reel cut boundary start drifted")
                    reference_frames = canonical_cut_cards[run_index]
                    decoded_reference_hashes: list[str] = []
                    for reference in reference_frames:
                        actual = next_output()
                        decoded_reference_hashes.append(
                            hashlib.sha256(
                                np.ascontiguousarray(reference).tobytes()
                            ).hexdigest()
                        )
                        if not np.array_equal(actual, reference):
                            raise RuntimeError(
                                "encoded reel cut-card content differs from exact canonical decoded text raster"
                            )
                    if count != cut["stop_frame_index"] or not reference_frames:
                        raise RuntimeError("encoded reel cut boundary stop drifted")
                    cut_checks.append(
                        {
                            "boundary_ordinal": cut["boundary_ordinal"],
                            "start_frame_index": cut["start_frame_index"],
                            "stop_frame_index": cut["stop_frame_index"],
                            "text_lines": list(CUT_TEXT_LINES),
                            "expected_card_rgb_sha256": expected_card_hash,
                            "literal_identity_payload_sha256": hashlib.sha256(
                                _literal_identity_payload(CUT_TEXT_LINES)
                            ).hexdigest(),
                            "canonical_decoded_frame_rgb_sha256": decoded_reference_hashes,
                            "exact_canonical_decoded_frames_match": True,
                            "comparison_tolerance_pixels": 0,
                            "visible_contrast_standard_deviation": float(
                                np.std(expected_card)
                            ),
                        }
                    )
            try:
                next(output_frames)
            except StopIteration:
                pass
            else:
                raise RuntimeError("encoded reel contains frames after its declared sequence")
            timing = _validated_cfr_timing(
                stream, timing_rows, rate=rate, run_id="encoded reel"
            )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("encoded reel could not be fully decoded") from error
    if count != expected_count:
        raise RuntimeError(f"encoded reel decoded {count} frames, expected {expected_count}")
    return {
        "decoded_frame_count": count,
        "profile": profile,
        "timing": timing,
        "source_segment_content_checks": source_checks,
        "cut_card_content_checks": cut_checks,
        "source_segments_in_declared_order": True,
        "cut_cards_match_exact_rendered_text": True,
    }


def _write_inventory(path: Path, rows: list[tuple[str, Path]]) -> None:
    content = "".join(
        f"{hash_regular_file(file)['sha256']}  {name}\n" for name, file in rows
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        data = content.encode("utf-8")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def encode_palette_cut_reel(
    request_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    owner_thread_id: str,
    repo_root: str | os.PathLike[str],
) -> Path:
    """Encode and atomically publish one hash-bound palette-change review reel.

    The request asserts that each source video was already accepted upstream.
    This function checks the declared file identities and media compatibility;
    it does not re-run or weaken upstream transport/acceptance gates.
    """

    if (
        not isinstance(owner_thread_id, str)
        or not owner_thread_id.strip()
        or any(ord(character) < 32 for character in owner_thread_id)
    ):
        raise ValueError("owner_thread_id must be a non-empty printable string")
    request_file, request, request_before = _read_request(Path(request_path))
    records = _validate_request(request, request_file.parent)
    repository = Path(repo_root).expanduser().resolve(strict=True)
    source_provenance = _source_provenance(repository)

    candidate = Path(output_dir).expanduser()
    output, classified_root = validate_part_a_artifact_output(
        candidate,
        repo_root=repository,
        artifact_type=PALETTE_CUT_REEL_ARTIFACT_TYPE,
    )
    parent = output.parent
    name = candidate.name
    if not name or name in {".", ".."}:
        raise ValueError("reel output must name one directory")
    if os.path.lexists(output):
        raise FileExistsError(output)
    input_paths = [Path(str(row["video_path"])) for row in records]
    if any(path.is_relative_to(output) for path in input_paths):
        raise ValueError("reel output must be outside every source-video tree")
    accepted_package_roots = {
        Path(str(row["package_receipt_path"])).parent for row in records
    }
    if any(output.is_relative_to(root) for root in accepted_package_roots):
        raise ValueError("reel output must be outside every bound accepted package root")

    av = _load_av()
    probes = [_probe(av, row) for row in records]
    first = probes[0]
    compatibility = (first["fps"], first["width"], first["height"], first["profile"])
    if any(
        (probe["fps"], probe["width"], probe["height"], probe["profile"])
        != compatibility
        for probe in probes[1:]
    ):
        raise ValueError("run videos must have identical geometry, fps, and codec profile")
    for record, probe in zip(records, probes, strict=True):
        record.update(probe)

    rate = Fraction(str(first["fps"]))
    width = int(first["width"])
    height = int(first["height"])
    expected_count = sum(int(row["frame_count"]) for row in records) + (
        len(records) - 1
    ) * int(rate * CUT_DURATION_SECONDS)

    partial: Path | None = Path(tempfile.mkdtemp(prefix=f".{name}.partial-", dir=parent))
    os.chmod(partial, 0o700)
    try:
        video_path = partial / _VIDEO_NAME
        try:
            cuts, encoding = _encode(
                av, records, video_path, rate=rate, width=width, height=height
            )
        except Exception as error:
            raise RuntimeError("palette-cut reel encoding failed") from error
        if not video_path.exists() or video_path.is_symlink():
            raise RuntimeError("palette-cut reel encoder did not create a regular MP4")
        os.chmod(video_path, 0o600)
        video_snapshot = hash_regular_file(video_path)
        if video_snapshot["bytes"] <= 0:
            raise RuntimeError("palette-cut reel MP4 is empty")
        verification = _verify_output(
            av,
            video_path,
            records=records,
            cuts=cuts,
            expected_count=expected_count,
            rate=rate,
            width=width,
            height=height,
        )

        if hash_regular_file(request_file) != request_before:
            raise RuntimeError("reel request changed while encoding")
        for row in records:
            snapshot = hash_regular_file(Path(str(row["video_path"])))
            if snapshot["sha256"] != row["video_sha256"] or snapshot["bytes"] != row["video_bytes"]:
                raise RuntimeError(f"run video changed while encoding: {row['run_id']}")
        try:
            accepted_after = _validate_request(request, request_file.parent)
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(
                "accepted package/run evidence changed while encoding"
            ) from error
        acceptance_fields = (
            "run_id",
            "video_path",
            "video_sha256",
            "package_receipt_path",
            "package_receipt_sha256",
            "package_sha256sums_path",
            "package_sha256sums_sha256",
            "run_receipt_path",
            "run_receipt_sha256",
            "palette_sha256",
            "palette_transmission_1cm",
        )
        if [
            tuple(row[field] for field in acceptance_fields) for row in accepted_after
        ] != [tuple(row[field] for field in acceptance_fields) for row in records]:
            raise RuntimeError("accepted package/run evidence changed while encoding")
        if _source_provenance(repository) != source_provenance:
            raise RuntimeError("executed reel source provenance changed while encoding")

        ordered_inputs = [
            {
                "ordinal": row["ordinal"],
                "run_id": row["run_id"],
                "video_path": row["video_path"],
                "video_sha256": row["video_sha256"],
                "video_bytes": row["video_bytes"],
                "frame_count": row["frame_count"],
                "fps": row["fps"],
                "geometry": {"width": row["width"], "height": row["height"]},
                "profile": row["profile"],
                "timing": row["timing"],
                "accepted_package": {
                    "receipt_path": row["package_receipt_path"],
                    "receipt_sha256": row["package_receipt_sha256"],
                    "sha256sums_path": row["package_sha256sums_path"],
                    "sha256sums_sha256": row["package_sha256sums_sha256"],
                },
                "accepted_run_receipt": {
                    "path": row["run_receipt_path"],
                    "sha256": row["run_receipt_sha256"],
                },
                "palette_sha256": row["palette_sha256"],
                "palette_transmission_1cm": row["palette_transmission_1cm"],
            }
            for row in records
        ]
        receipt = {
            "schema_version": 1,
            "artifact_type": PALETTE_CUT_REEL_ARTIFACT_TYPE,
            "status": "encoded_verified",
            "owner_thread_id": owner_thread_id,
            "source_commit": source_provenance["commit"],
            "source_provenance": source_provenance,
            "part_a_artifact_boundary": {
                "artifact_type": PALETTE_CUT_REEL_ARTIFACT_TYPE,
                "classified_output_root": os.fspath(classified_root),
                "part_b_exact_read_policy": "forbid classified reel root",
            },
            "request": {
                "path": os.fspath(request_file),
                "sha256": request_before["sha256"],
                "bytes": request_before["bytes"],
            },
            "ordered_inputs": ordered_inputs,
            "input_profile": first["profile"],
            "cut_card_duration_seconds": CUT_DURATION_SECONDS,
            "cut_cards": cuts,
            "output": {
                "path": _VIDEO_NAME,
                "sha256": video_snapshot["sha256"],
                "bytes": video_snapshot["bytes"],
                "frame_count": encoding["frame_count"],
                "fps": str(rate),
                "geometry": {"width": width, "height": height},
                "profile": verification["profile"],
            },
            "verification": verification,
            "sha_inventory": {
                "path": _INVENTORY_NAME,
                "entries": [_VIDEO_NAME, _RECEIPT_NAME],
            },
            "transport_boundary_policy": {
                "transport_frames_added_by_reel": False,
                "transport_crosses_palette_boundaries": False,
                "boundary_mechanism": "fixed-duration visible cut card only",
            },
            "limits": {
                "video_role": "lossy review reel only",
                "upstream_acceptance": "asserted by request; not re-evaluated by this encoder",
                "physical_claim": "none",
                "audio": "source videos must be silent; output is silent",
            },
        }
        receipt_path = partial / _RECEIPT_NAME
        _write_private_json(receipt_path, receipt)
        _write_inventory(
            partial / _INVENTORY_NAME,
            [(_VIDEO_NAME, video_path), (_RECEIPT_NAME, receipt_path)],
        )
        directory_fd = os.open(partial, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _atomic_rename_noreplace(partial, output)
        partial = None
        return output / _RECEIPT_NAME
    finally:
        if partial is not None and partial.exists():
            metadata = os.lstat(partial)
            if stat.S_ISDIR(metadata.st_mode) and partial.parent == parent:
                shutil.rmtree(partial)

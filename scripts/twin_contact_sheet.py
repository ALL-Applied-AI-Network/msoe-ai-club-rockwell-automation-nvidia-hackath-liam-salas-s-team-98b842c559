#!/usr/bin/env python3
"""Build review-only RGB/liquid-mask contact sheets from Task 2A captures."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import secrets
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix.render_process import verify_capture_summary  # noqa: E402

CAPTION_HEIGHT = 24


def tint_liquid_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB copy tinted at exactly the emitted binary mask pixels."""

    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3:
        raise ValueError("RGB raster must be a uint8 HxWx3 array")
    if rgb.shape[2] != 3:
        raise ValueError("RGB raster must have exactly three channels")
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.uint8
        or mask.shape != rgb.shape[:2]
    ):
        raise ValueError("liquid mask must be a uint8 raster aligned to RGB")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError("liquid mask must contain only binary 0/255 values")
    selected = mask == 255
    overlay = rgb.copy()
    tint = np.array([255, 0, 255], dtype=np.uint8)
    alternate = np.array([0, 255, 255], dtype=np.uint8)
    original_selected = rgb[selected]
    overlay[selected] = tint
    if original_selected.size:
        already_tint = np.all(original_selected == tint, axis=1)
        overlay[selected] = np.where(already_tint[:, None], alternate, tint)
    return overlay


def _decode_png(path: Path, *, role: str) -> Image.Image:
    try:
        with Image.open(path) as candidate:
            if candidate.format != "PNG":
                raise ValueError(f"{role} artifact is not a PNG: {path}")
            candidate.verify()
        with Image.open(path) as decoded:
            decoded.load()
            return decoded.copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"{role} artifact does not decode: {path}") from error


def _select_frames(
    frames: list[object],
    *,
    limit: int | None,
    time_codes: tuple[int, ...] | None,
) -> list[Mapping[str, object]]:
    records = [frame for frame in frames if isinstance(frame, Mapping)]
    if len(records) != len(frames):
        raise ValueError("capture frame records are malformed")
    if time_codes is not None:
        if (
            not isinstance(time_codes, tuple)
            or not time_codes
            or any(isinstance(code, bool) or not isinstance(code, int) or code < 0 for code in time_codes)
            or len(set(time_codes)) != len(time_codes)
        ):
            raise ValueError("frame selection must be unique nonnegative time codes")
        by_time = {frame["time_code"]: frame for frame in records}
        missing = [code for code in time_codes if code not in by_time]
        if missing:
            raise ValueError(f"selected time codes are absent from capture: {missing!r}")
        records = [by_time[code] for code in time_codes]
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("LIMIT must be a positive integer")
        records = records[:limit]
    if not records:
        raise ValueError("contact sheet selection contains no frames")
    return records


def _publish_png_exclusive(image: Image.Image, output: Path) -> Path:
    if os.path.lexists(output):
        raise FileExistsError(f"contact sheet output already exists: {output}")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError(f"contact sheet output parent is not a directory: {parent}")
    temporary = parent / f".{output.name}.tmp-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        decoded = _decode_png(temporary, role="contact sheet")
        if decoded.size != image.size or decoded.mode != "RGB":
            raise OSError("contact sheet write verification failed")
        os.link(temporary, parent / output.name, follow_symlinks=False)
        published = True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    final = parent / output.name
    if not published:
        raise OSError(f"contact sheet was not published: {output}")
    decoded = _decode_png(final, role="contact sheet")
    if decoded.size != image.size or decoded.mode != "RGB":
        raise OSError("published contact sheet verification failed")
    return final


def create_contact_sheet(
    run_dir: str | Path,
    output: str | Path,
    *,
    limit: int | None = None,
    time_codes: tuple[int, ...] | None = None,
) -> Path:
    output_candidate = Path(output).expanduser()
    if os.path.lexists(output_candidate):
        raise FileExistsError(
            f"contact sheet output already exists: {output_candidate}"
        )
    run = Path(run_dir).expanduser()
    if not (run / "summary.json").is_file() and (run / "capture/summary.json").is_file():
        run = run / "capture"
    verified = verify_capture_summary(run, None)
    summary = verified["summary"]
    assert isinstance(summary, dict)
    frames = summary["frames"]
    assert isinstance(frames, list)
    selected = _select_frames(frames, limit=limit, time_codes=time_codes)

    tiles: list[Image.Image] = []
    raster_size: tuple[int, int] | None = None
    for frame in selected:
        artifacts = frame["artifacts"]
        assert isinstance(artifacts, Mapping)
        rgb_record = artifacts["rgb"]
        liquid_record = artifacts["liquid"]
        assert isinstance(rgb_record, Mapping) and isinstance(liquid_record, Mapping)
        rgb_image = _decode_png(run / str(rgb_record["path"]), role="RGB")
        liquid_image = _decode_png(run / str(liquid_record["path"]), role="liquid mask")
        if rgb_image.mode != "RGB" or liquid_image.mode not in {"L", "1"}:
            raise ValueError("capture RGB or liquid-mask PNG has the wrong raster mode")
        rgb = np.asarray(rgb_image, dtype=np.uint8)
        mask = np.asarray(liquid_image.convert("L"), dtype=np.uint8)
        shape_record = frame.get("rgb")
        if not isinstance(shape_record, Mapping) or shape_record.get("shape") != [
            rgb.shape[0],
            rgb.shape[1],
            3,
        ]:
            raise ValueError("capture RGB dimensions disagree with the summary")
        resolution = summary["render_settings"]["resolution"]
        if list(rgb_image.size) != resolution or mask.shape != rgb.shape[:2]:
            raise ValueError("capture RGB and liquid-mask raster dimensions are not aligned")
        if not set(int(value) for value in np.unique(mask)).issubset({0, 255}):
            raise ValueError("capture liquid mask is not binary")
        liquid_pixels = int(np.count_nonzero(mask == 255))
        counts = frame.get("class_pixel_counts")
        if not isinstance(counts, Mapping) or counts.get("liquid") != liquid_pixels:
            raise ValueError("capture liquid-pixel count disagrees with the emitted mask")
        overlay = tint_liquid_mask(rgb, mask)
        current_size = rgb_image.size
        if raster_size is None:
            raster_size = current_size
        elif raster_size != current_size:
            raise ValueError("selected capture frames do not share one raster size")
        width, height = current_size
        tile = Image.new("RGB", (width, CAPTION_HEIGHT + height * 2), "white")
        label = (
            f"frame {frame['capture_ordinal']} | time code {frame['time_code']} | "
            f"liquid pixels {liquid_pixels}"
        )
        ImageDraw.Draw(tile).text((2, 2), label, fill="black")
        tile.paste(rgb_image, (0, CAPTION_HEIGHT))
        tile.paste(Image.fromarray(overlay, mode="RGB"), (0, CAPTION_HEIGHT + height))
        tiles.append(tile)

    assert raster_size is not None
    tile_width, tile_height = tiles[0].size
    sheet = Image.new("RGB", (tile_width * len(tiles), tile_height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * tile_width, 0))
    return _publish_png_exclusive(sheet, output_candidate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", metavar="RUN_DIR")
    parser.add_argument("output", metavar="OUTPUT")
    parser.add_argument("limit", metavar="LIMIT", nargs="?", type=int)
    parser.add_argument("--frames", help="comma-separated exact time-code selection")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        time_codes = parse_frame_selection(args.frames) if args.frames else None
        result = create_contact_sheet(
            args.run_dir,
            args.output,
            limit=args.limit,
            time_codes=time_codes,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote review-only contact sheet: {result}")
    return 0


def parse_frame_selection(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--frames must be a comma-separated time-code list")
    try:
        result = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError("--frames contains a non-integer time code") from error
    if any(code < 0 for code in result) or len(set(result)) != len(result):
        raise ValueError("--frames time codes must be unique and nonnegative")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

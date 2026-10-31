"""Measure liquid-surface row candidates from real vial-fill reference images.

This diagnostic never writes or copies a source frame. Each --sample is
``VOLUME_ML=IMAGE_PATH,SEARCH_FIRST_ROW,SEARCH_LAST_ROW``. The row score is
the CIELAB distance between robust profiles immediately above and below the
candidate boundary. Search bounds are inclusive; column bounds use Python's
half-open ``FIRST:LAST`` convention.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_columns(value):
    try:
        first, last = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("columns must be FIRST:LAST") from exc
    if first < 0 or last <= first:
        raise argparse.ArgumentTypeError("columns must be a positive half-open range")
    return first, last


def parse_sample(value):
    try:
        volume_text, remainder = value.split("=", 1)
        path_text, first_text, last_text = remainder.rsplit(",", 2)
        volume = float(volume_text)
        first = int(first_text)
        last = int(last_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "sample must be VOLUME=IMAGE_PATH,SEARCH_FIRST,SEARCH_LAST"
        ) from exc
    if volume <= 0 or first < 0 or last <= first:
        raise argparse.ArgumentTypeError("sample volume and search range must be positive")
    return volume, Path(path_text), first, last


def separated_candidates(scores, count, separation):
    chosen = []
    for row, score in sorted(scores, key=lambda item: item[1], reverse=True):
        if all(abs(row - prior[0]) >= separation for prior in chosen):
            chosen.append((row, score))
        if len(chosen) == count:
            break
    return chosen


def true_runs(values, row_offset):
    runs = []
    start = None
    for index, value in enumerate(values):
        if bool(value):
            if start is None:
                start = index
        elif start is not None:
            runs.append([start + row_offset, index - 1 + row_offset])
            start = None
    if start is not None:
        runs.append([start + row_offset, len(values) - 1 + row_offset])
    return runs


def measure_sample(
    image_path,
    columns,
    search,
    half_window,
    candidate_count,
    holder_column,
    holder_search,
    dark_threshold,
):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {image_path}")
    height, width = image.shape[:2]
    x_first, x_last = columns
    y_first, y_last = search
    if x_last > width or y_last >= height:
        raise ValueError(
            f"measurement range exceeds {width}x{height} image: {image_path}"
        )
    if y_first < half_window or y_last + half_window >= height:
        raise ValueError("search rows need a full comparison window above and below")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    column_strip = lab[:, x_first:x_last]
    scores = []
    for row in range(y_first, y_last + 1):
        above = np.median(column_strip[row - half_window : row], axis=(0, 1))
        below = np.median(column_strip[row : row + half_window], axis=(0, 1))
        scores.append((row, float(np.linalg.norm(below - above))))
    candidates = separated_candidates(
        scores, count=candidate_count, separation=max(2, half_window)
    )
    selected_row, selected_score = candidates[0]
    holder_first, holder_last = holder_search
    if holder_column >= width or holder_last > height:
        raise ValueError("holder diagnostic range exceeds the image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    holder_dark = gray[holder_first:holder_last, holder_column] < dark_threshold
    return {
        "image": str(image_path.resolve()),
        "image_shape": [height, width, int(image.shape[2])],
        "columns_half_open": [x_first, x_last],
        "search_rows_inclusive": [y_first, y_last],
        "half_window_rows": half_window,
        "selected_row": selected_row,
        "selected_score_cielab": selected_score,
        "candidates": [
            {"row": row, "score_cielab": score} for row, score in candidates
        ],
        "holder_diagnostic": {
            "column_x": holder_column,
            "search_rows_half_open": [holder_first, holder_last],
            "dark_threshold": dark_threshold,
            "dark_runs": true_runs(holder_dark, holder_first),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=parse_sample, action="append", required=True)
    parser.add_argument("--columns", type=parse_columns, default=(298, 316))
    parser.add_argument("--half-window", type=int, default=6)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--holder-column", type=int, default=272)
    parser.add_argument("--holder-search", type=parse_columns, default=(180, 281))
    parser.add_argument("--dark-threshold", type=int, default=70)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.half_window < 1 or args.candidate_count < 1:
        parser.error("window and candidate count must be positive")
    if args.holder_column < 0 or not 0 <= args.dark_threshold <= 255:
        parser.error("holder column and dark threshold are out of range")
    volumes = [sample[0] for sample in args.sample]
    if len(volumes) != len(set(volumes)):
        parser.error("sample volumes must be unique")
    if args.out and args.out.exists():
        parser.error("--out must be a fresh path")

    records = []
    for volume, image_path, first, last in sorted(args.sample):
        if not image_path.is_file():
            parser.error(f"sample image does not exist: {image_path}")
        record = measure_sample(
            image_path,
            columns=args.columns,
            search=(first, last),
            half_window=args.half_window,
            candidate_count=args.candidate_count,
            holder_column=args.holder_column,
            holder_search=args.holder_search,
            dark_threshold=args.dark_threshold,
        )
        record["volume_ml"] = volume
        records.append(record)

    result = {
        "schema_version": 1,
        "purpose": "real video-5 camera-target measurement; never training data",
        "method": "strongest local CIELAB profile change in a declared search window",
        "records": records,
    }
    if len(records) == 2:
        low, high = records
        volume_delta = high["volume_ml"] - low["volume_ml"]
        row_delta = low["selected_row"] - high["selected_row"]
        result["rows_per_ml"] = row_delta / volume_delta
        result["ml_per_row"] = volume_delta / row_delta
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

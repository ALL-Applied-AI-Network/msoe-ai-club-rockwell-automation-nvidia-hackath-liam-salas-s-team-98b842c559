#!/usr/bin/env python3
"""Build the reviewable Phase 1 mock fixture and local presentation assets."""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

PAIR_IDS = [
    "PAIR-BATCH_0ebf3f287cc1291a-dispense-1-t200-V1-V2",
    "PAIR-BATCH_0ebf3f287cc1291a-dispense-1-t200-V1-V3",
    "PAIR-BATCH_0ebf3f287cc1291a-dispense-2-t400-V1-V2",
    "PAIR-BATCH_0ebf3f287cc1291a-dispense-2-t400-V1-V3",
    "PAIR-BATCH_0ebf3f287cc1291a-dispense-3-t100-V1-V2",
    "PAIR-BATCH_0ebf3f287cc1291a-dispense-3-t100-V1-V3",
    "PAIR-BATCH_5a4a586e3ec3b085-dispense-1-t200-V1-V2",
    "PAIR-BATCH_5a4a586e3ec3b085-dispense-1-t200-V1-V3",
    "PAIR-BATCH_5a4a586e3ec3b085-dispense-2-t400-V1-V2",
]
OUTCOMES = [
    "keep",
    "keep",
    "insufficient_evidence",
    "keep",
    "keep",
    "keep",
    "discard",
    "discard",
    "discard",
]
WP5_CLAIM_BOUNDARY = (
    "Synthetic-twin masked colour measurements only; no keep/discard inference, "
    "physical-colour validation, or sim-to-real claim."
)
WP6_SCOPE = "limited_pre_access_all_abstention_no_accuracy_or_winner_claim"
MOCK_THRESHOLD_SOURCE = "phase1-mock-threshold-10"
COMPOSITE_SOURCE = "run-01-composite"
APPROVED_WP1_MANIFEST_SHA256 = "90418a6e9a98d456de29bcebe497f5e0db3ab2309c6942f428318534dda14a76"
APPROVED_WP1_PREEXEC_SHA256 = "d55f9490283569b9e50c29514cfa7182d673fb3d74f5f88db6a3d25ab6e1d288"
APPROVED_WP5_MEASUREMENTS_SHA256 = "862129e99fceaade6a9f44e74e29267b24366e8ed0fb3406002d2ef1a5e52b14"
APPROVED_WP5_LOCATORS_SHA256 = "3a10c45bc67718ab98270a97e4ecd043265695fa77f5e1fe0ebd9da23a0d19f4"
APPROVED_WP6_SHA256 = {
    "anomalythink-score.json": "a0dfd8aa795f1c6e3e6aad4b7b5b3c3a12f8ec956a1e976f8fdec31bbea52be3",
    "cosmos-score.json": "3b986e50d1ec0a7140c9ecff54fee5f94b256234a715876e225625e8b70fd2d9",
    "qwen-score.json": "3cd9ccd37af3453063f4aed803bc0c35ee3e2db5832ad10ef6c107a6df9040f9",
    "template-score.json": "ce036c2af554c201d45f2a600bdde6e301f9566c76e7232324f04553722c5843",
}
APPROVED_RENDER_RECEIPT_SHA256 = {
    "segment-00-entry": "f3b3695aa42cec9cd30130c6b1e0570fd1be39b92b3c2a4c7161f974fa98b3e5",
    "segment-01-between": "776c4cd0661917381e1b41dc863e98e041fe89be9d997e99a6e5bcab7dbfbc9e",
    "segment-02-between": "000ae5586969f13a653d66c2102bb9f29c2be96a88ed2232bceaa7cf970c486e",
    "segment-03-between": "5d077986b33a1e5a2821fcf5072f26565bdfd0ef2eb34a2bbcb63a9cccf32255",
    "segment-04-between": "f596de454afa56aca34edad9f388145532ccf3b057c573be70554654d5519a83",
    "segment-05-between": "fb0be848925c88fcfec7d040befec1e212502bf8a6d60f30c4d125fff8ebad86",
    "segment-06-between": "9e8e533d7572522ef8ff3b75dc3c59a7d1759b7b133f12df7994f9a8a1cdbf91",
    "segment-07-between": "9d53c28c384e225529c08dc07e54fd54ed19bca8310d3dfeb1f04dbc6c9c2a0e",
    "segment-08-between": "034402df8ee153f6f6ae07f311e00dac00d8b4ecc95cc98b2ffe224bdc9da2c7",
    "segment-09-exit": "1d86326dd2c50b20107983a57334ce3fb30d6776ac88fac2c204bf7646988bee",
}


def fail(message):
    raise SystemExit(message)


def require(condition, message):
    if not condition:
        fail(message)


def load_json(path, label):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"{label} could not be read as JSON: {path}: {exc}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_sha(path, expected, label):
    require(path.is_file(), f"{label} is missing: {path}")
    actual = sha256(path)
    require(actual == expected, f"{label} SHA-256 drift: expected {expected}, got {actual}")
    return actual


def checked_approved_sha(path, expected, label):
    require(path.is_file(), f"{label} is missing: {path}")
    actual = sha256(path)
    require(
        actual == expected,
        f"{label} does not match the approved SHA-256 {expected}: got {actual}",
    )
    return actual


def run_ffmpeg(command, label):
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, f"{label} failed: {result.stderr[-2000:]}")
    return result.stderr


def probe_duration(ffmpeg, path):
    result = subprocess.run(
        [str(ffmpeg), "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"ffmpeg probe/decode failed for {path}: {result.stderr[-2000:]}")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    require(match is not None, f"ffmpeg did not report a duration for {path}")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    require(duration > 0, f"ffmpeg reported a non-positive duration for {path}: {duration}")
    return duration


def validate_wp5(measurements_path, locator_path):
    checked_approved_sha(
        measurements_path, APPROVED_WP5_MEASUREMENTS_SHA256, "WP5 measurements input"
    )
    checked_approved_sha(locator_path, APPROVED_WP5_LOCATORS_SHA256, "WP5 locator input")
    wp5 = load_json(measurements_path, "WP5 measurements")
    locator_doc = load_json(locator_path, "WP5 prediction locators")
    require(wp5.get("status") == "measured", "WP5 status must be measured")
    require(wp5.get("data_origin") == "twin", "WP5 data_origin must be twin")
    require(wp5.get("real_data_used") is False, "WP5 real_data_used must be false")
    require(
        wp5.get("claim_boundary") == WP5_CLAIM_BOUNDARY,
        f"WP5 claim_boundary must exactly equal: {WP5_CLAIM_BOUNDARY}",
    )
    contract = wp5.get("measurement_contract", {})
    require(
        contract.get("scope") == "measurements_only_no_keep_discard_or_abstain_output",
        "WP5 measurement contract must remain measurements-only",
    )
    locator_hash = sha256(locator_path)
    require(
        contract.get("source_prediction_locator_sha256") == locator_hash,
        "WP5 measurement contract locator SHA-256 does not match the passed locator file",
    )
    require(locator_doc.get("data_origin") == "twin", "WP5 locator data_origin must be twin")
    require(isinstance(locator_doc.get("images"), dict), "WP5 locator images must be an object")
    return wp5, locator_doc["images"]


def select_pairs(wp5, locators):
    rows = {row["pair_id"]: row for row in wp5.get("same_run_stage_pairs", [])}
    measurements = {
        measurement["evidence_id"]: measurement
        for measurement in wp5.get("stage_measurements", [])
    }
    producer = wp5.get("mask_prediction_producer", {})
    require(
        producer.get("producer_id") == "accepted-empty-reference-rgb-predictor",
        "WP5 mask producer must be the accepted RGB-only predictor",
    )
    selected = []
    for pair_id in PAIR_IDS:
        require(pair_id in rows, f"selected WP5 pair is missing: {pair_id}")
        row = rows[pair_id]
        require(row.get("left_evidence_id") in measurements, f"{pair_id} left measurement is missing")
        require(row.get("right_evidence_id") in measurements, f"{pair_id} right measurement is missing")
        left = measurements[row["left_evidence_id"]]
        right = measurements[row["right_evidence_id"]]
        for side, measurement, vial_field in (
            ("left", left, "left_vial_id"),
            ("right", right, "right_vial_id"),
        ):
            for field in ("recipe_id", "stage_id", "run_id", "batch_id"):
                require(
                    measurement.get(field) == row.get(field),
                    f"{pair_id} {side} measurement {field} does not match the pair",
                )
            require(
                measurement.get("vial_id") == row.get(vial_field),
                f"{pair_id} {side} measurement vial_id does not match the pair",
            )
            side_locator_key = measurement.get("image_id")
            require(
                side_locator_key in locators,
                f"{pair_id} {side} locator key is missing: {side_locator_key}",
            )
            side_locator = locators[side_locator_key]
            require(
                measurement.get("rgb_sha256")
                == row.get(f"{side}_rgb_sha256")
                == side_locator.get("rgb_sha256"),
                f"{pair_id} {side} RGB hashes disagree across pair, measurement, and locator",
            )
            require(
                measurement.get("mask_sha256")
                == row.get(f"{side}_mask_sha256")
                == side_locator.get("mask_sha256"),
                f"{pair_id} {side} mask hashes disagree across pair, measurement, and locator",
            )
        locator_key = right.get("image_id")
        locator = locators[locator_key]
        rgb_path = Path(locator["rgb_path"])
        mask_path = Path(locator["mask_path"])
        checked_sha(rgb_path, locator["rgb_sha256"], f"{pair_id} RGB source")
        checked_sha(mask_path, locator["mask_sha256"], f"{pair_id} mask source")
        selected.append(
            {
                "pair_id": pair_id,
                "row": row,
                "left": left,
                "right": right,
                "locator_key": locator_key,
                "locator": locator,
                "rgb_path": rgb_path,
                "mask_path": mask_path,
                "producer": producer,
            }
        )
    return selected


def load_wp6_rows(wp6_dir):
    definitions = [
        ("template-score.json", "template", "Template"),
        ("qwen-score.json", "qwen3_vl", "Qwen3-VL"),
        ("anomalythink-score.json", "anomalythink_free_form", "AnomalyThink"),
        ("cosmos-score.json", "cosmos_strict_parse", "Cosmos"),
    ]
    for filename, _method, label in definitions:
        checked_approved_sha(
            wp6_dir / filename, APPROVED_WP6_SHA256[filename], f"WP6 {label} input"
        )
    rows = []
    for filename, method, label in definitions:
        path = wp6_dir / filename
        doc = load_json(path, f"WP6 {label} score")
        require(doc.get("scope") == WP6_SCOPE, f"{filename} has an unexpected scope")
        require(
            doc.get("denominators") == {"combined": 16, "ordered": 8, "shuffled": 8},
            f"{filename} denominators must be combined=16, ordered=8, shuffled=8",
        )
        attempts = doc.get("attempts")
        require(isinstance(attempts, list) and len(attempts) == 16, f"{filename} must contain 16 attempts")
        counts = Counter(attempt.get("variant") for attempt in attempts)
        require(
            counts == Counter({"ordered": 8, "shuffled": 8}),
            f"{filename} attempt sources must contain 8 ordered and 8 shuffled records",
        )
        correctness = doc.get("abstention_correctness", {})
        require(correctness.get("denominator") == 16, f"{filename} correctness denominator must be 16")
        passed = correctness.get("success_count")
        require(passed in (0, 16), f"{filename} success_count must be 0 or 16")
        expected = 16 if method in ("template", "qwen3_vl") else 0
        require(passed == expected, f"{filename} success_count must be {expected}")
        source_id = f"{filename}#sha256={sha256(path)}"
        if method == "template":
            method_note = "Template 16/16 limited all-abstention parser-contract successes."
            failure_note = None
        elif method == "qwen3_vl":
            method_note = "Qwen3-VL 16/16 limited all-abstention parser-contract successes."
            failure_note = None
        elif method == "anomalythink_free_form":
            method_note = "AnomalyThink free-form limited benchmark."
            failure_note = "AnomalyThink 0/16: free-form outputs failed the parser contract."
        else:
            method_note = "Cosmos strict-parse limited benchmark."
            failure_note = (
                "Cosmos 0/16 strict-parse is from the exact cosmos-score.json; "
                "Markdown-fence diagnosis is accepted task-dispatch/lead context and is "
                "not stored in that score file."
            )
        provenance_note = (
            f"Parsed exact {filename}: 16 combined attempts, with 8 ordered and "
            "8 shuffled source records."
        )
        if method == "cosmos_strict_parse":
            provenance_note = (
                "Numeric 0/16 parsed from exact cosmos-score.json (16 combined: 8 ordered, "
                "8 shuffled); Markdown-fence diagnosis is accepted task-dispatch/lead "
                "context and is not stored in that score file."
            )
        rows.append(
            {
                "method": method,
                "label": label,
                "passed": passed,
                "total": 16,
                "method_note": method_note,
                "failure_note": failure_note,
                "provenance": {
                    "kind": "documented",
                    "source_id": source_id,
                    "note": provenance_note,
                },
            }
        )
    return rows


def manifest_fill_jobs(wp1_batch):
    manifest_path = wp1_batch / "simple-production-manifest.json"
    checked_approved_sha(manifest_path, APPROVED_WP1_MANIFEST_SHA256, "WP1 manifest input")
    manifest = load_json(manifest_path, "WP1 production manifest")
    jobs = []
    seen = set()
    for job in manifest.get("jobs", [])[:10]:
        current = job.get("current_case")
        require(isinstance(current, dict), "WP1 manifest job is missing current_case")
        ordinal = current.get("vial_ordinal")
        if ordinal in seen:
            continue
        seen.add(ordinal)
        video = current.get("video", {})
        path = Path(video.get("path", ""))
        checked_sha(path, video.get("sha256"), f"WP1 fill video for vial {ordinal}")
        jobs.append(path)
    require(len(jobs) == 9, "WP1 manifest must provide nine unique, hash-bound fill videos")
    return jobs


def validate_transport_capture_files(render_dir, receipt, expected_mode, expected_indices):
    require(receipt.get("artifact_type") == "fluidmix_transport_stage_render_receipt_v1", "unexpected transport render receipt type")
    require(receipt.get("mode") == expected_mode, f"transport render receipt mode must be {expected_mode}")
    require(receipt.get("state_indices") == expected_indices, "transport render receipt state indices drifted")
    captures = receipt.get("captures")
    require(isinstance(captures, list) and len(captures) == len(expected_indices), "transport render receipt capture count drifted")
    frames = []
    for state_index, capture in zip(expected_indices, captures):
        expected_relative = f"states/state-{state_index:02d}.png"
        require(capture.get("state_index") == state_index, f"transport capture state index {state_index} drifted")
        require(capture.get("rgb_path") == expected_relative, f"transport capture path must be {expected_relative}")
        frame = render_dir / expected_relative
        checked_sha(frame, capture.get("rgb_sha256"), f"transport state {expected_relative}")
        frames.append(frame)
    return frames


def validated_transport_frames(wp1_batch, segment, mode):
    segment_name = f"segment-{segment:02d}-{mode}"
    require(segment_name in APPROVED_RENDER_RECEIPT_SHA256, f"unapproved transport segment: {segment_name}")
    render_dir = wp1_batch / segment_name / "render"
    receipt_path = render_dir / "transport-render-receipt.json"
    checked_approved_sha(
        receipt_path,
        APPROVED_RENDER_RECEIPT_SHA256[segment_name],
        f"{segment_name} render receipt",
    )
    receipt = load_json(receipt_path, f"{segment_name} render receipt")
    return validate_transport_capture_files(render_dir, receipt, mode, list(range(13)))


def validate_wp1_batch(wp1_batch):
    receipt_path = wp1_batch / "preexec-receipt.json"
    checked_approved_sha(receipt_path, APPROVED_WP1_PREEXEC_SHA256, "WP1 pre-execution receipt")
    checked_approved_sha(
        wp1_batch / "simple-production-manifest.json",
        APPROVED_WP1_MANIFEST_SHA256,
        "WP1 manifest input",
    )
    for segment_name, expected_sha in APPROVED_RENDER_RECEIPT_SHA256.items():
        checked_approved_sha(
            wp1_batch / segment_name / "render" / "transport-render-receipt.json",
            expected_sha,
            f"{segment_name} render receipt",
        )
    receipt = load_json(receipt_path, "WP1 pre-execution receipt")
    require(receipt.get("artifact_type") == "fluidmix_simple_transport_preexec_v1", "unexpected WP1 pre-execution receipt type")
    require(receipt.get("status") == "ready", "WP1 pre-execution receipt must be ready")
    require(receipt.get("manifest_sha256") == APPROVED_WP1_MANIFEST_SHA256, "WP1 pre-execution receipt manifest hash drifted")


def build_media(ffmpeg, wp1_batch, fills, destination):
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="fluidmix-composite-") as temp_name:
        temp = Path(temp_name)
        sections = []
        durations = []
        fill_ranges = []
        elapsed = 0.0
        for segment in range(10):
            mode = "entry" if segment == 0 else "exit" if segment == 9 else "between"
            frames = validated_transport_frames(wp1_batch, segment, mode)
            pattern = temp / f"state-{segment:02d}-%04d.jpg"
            for index, path in enumerate(frames):
                with Image.open(path) as image:
                    image.convert("RGB").resize((640, 480)).save(
                        str(pattern).replace("%04d", f"{index:04d}"), quality=82
                    )
            transport = temp / f"section-{len(sections):02d}.mp4"
            run_ffmpeg(
                [
                    str(ffmpeg), "-y", "-framerate", "10", "-i", str(pattern),
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-r", "10", str(transport),
                ],
                f"transport section {segment}",
            )
            transport_duration = probe_duration(ffmpeg, transport)
            sections.append(transport)
            durations.append(transport_duration)
            elapsed += transport_duration
            if segment < 9:
                fill = temp / f"section-{len(sections):02d}.mp4"
                run_ffmpeg(
                    [
                        str(ffmpeg), "-y", "-i", str(fills[segment]),
                        "-vf", "scale=640:480:force_original_aspect_ratio=depad,fps=10",
                        "-an", "-c:v", "libx264", "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p", "-r", "10", "-filter:v", "setpts=0.1*PTS",
                        str(fill),
                    ],
                    f"fill section {segment + 1}",
                )
                fill_duration = probe_duration(ffmpeg, fill)
                sections.append(fill)
                durations.append(fill_duration)
                fill_ranges.append((round(elapsed, 6), round(elapsed + fill_duration, 6)))
                elapsed += fill_duration
        require(len(sections) == 19 and len(durations) == 19, "composite must contain 19 probed sections")
        listing = temp / "concat.txt"
        listing.write_text("\n".join(f"file '{path}'" for path in sections) + "\n")
        run_ffmpeg(
            [
                str(ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", "-movflags", "+faststart", str(destination),
            ],
            "final composite",
        )
    final_duration = probe_duration(ffmpeg, destination)
    require(fill_ranges[0][0] > 0, "first fill cue must start after the entry section")
    require(fill_ranges[-1][1] <= final_duration, "last fill cue exceeds the final composite")
    for previous, current in zip(fill_ranges, fill_ranges[1:]):
        require(current[0] > previous[1], "fill cues must preserve non-overlapping transport gaps")
    return final_duration, fill_ranges


def evidence_records(vial_id, pair):
    pair_id = pair["pair_id"]
    left = pair["left"]
    right = pair["right"]
    locator = pair["locator"]
    producer = pair["producer"]
    provenances = {
        "rgb_frame": {
            "kind": "measured",
            "source_id": right["measurement_id"],
            "note": f"WP5 right measurement; locator RGB SHA-256 {locator['rgb_sha256']}.",
        },
        "predicted_mask": {
            "kind": "measured",
            "source_id": producer["producer_id"],
            "note": (
                f"WP5 right measurement {right['measurement_id']}; predicted locator mask "
                f"SHA-256 {locator['mask_sha256']}; model SHA-256 {producer['model_sha256']}."
            ),
        },
        "target_lab": {
            "kind": "measured",
            "source_id": right["measurement_id"],
            "note": "WP5 audited right-stage median Lab measurement.",
        },
        "reference_lab": {
            "kind": "measured",
            "source_id": left["measurement_id"],
            "note": "WP5 audited left-stage median Lab measurement.",
        },
        "delta_e": {
            "kind": "measured",
            "source_id": pair_id,
            "note": "WP5 audited pair CIE76 Delta E.",
        },
        "threshold": {
            "kind": "documented",
            "source_id": MOCK_THRESHOLD_SOURCE,
            "note": "Documented mock threshold; not measured and not a plant limit.",
        },
    }
    records = [
        {
            "evidence_id": f"evidence-{pair_id}-{target}",
            "vial_id": vial_id,
            "target": target,
            "label": f"{pair_id} {target}",
            "provenance": provenance,
        }
        for target, provenance in provenances.items()
    ]
    return provenances, records


def build_fixture(template, pairs, wp6_rows, duration, fill_ranges):
    data = copy.deepcopy(template)
    require(len(data.get("vials", [])) == 9, "UI template must contain nine vial slots")
    require(len(fill_ranges) == 9, "composite must provide nine fill cue ranges")
    data["dataset_id"] = "phase1-run-01-mock"
    data["label"] = "Synthetic twin data — RUN-01 mock"
    data["run"]["run_id"] = "RUN-01"
    data["run"]["title"] = "RUN-01 operator mock"
    data["run"]["video"].update(
        {
            "asset_id": COMPOSITE_SOURCE,
            "src": "/assets/run/run-01-composite.mp4",
            "duration_seconds": {
                "value": duration,
                "provenance": {
                    "kind": "inferred",
                    "source_id": COMPOSITE_SOURCE,
                    "note": "Actual ffmpeg-probed duration of the derived local composite; not plant timing.",
                },
            },
            "provenance": {
                "kind": "inferred",
                "source_id": COMPOSITE_SOURCE,
                "note": "Derived local RUN-01 presentation composite.",
            },
        }
    )
    all_evidence = []
    history = []
    for index, (vial, pair, outcome) in enumerate(
        zip(data["vials"], pairs, OUTCOMES), start=1
    ):
        pair_id = pair["pair_id"]
        row = pair["row"]
        left = pair["left"]
        right = pair["right"]
        vial_id = f"vial-{index:03d}"
        provenances, evidence = evidence_records(vial_id, pair)
        all_evidence.extend(evidence)
        vial.update(
            {
                "slot": index,
                "vial_id": vial_id,
                "recipe_id": row["recipe_id"],
                "rgb_frame": {
                    "asset_id": f"rgb-{vial_id}",
                    "src": f"/assets/vials/{vial_id}-rgb.jpg",
                    "provenance": provenances["rgb_frame"],
                },
                "predicted_mask": {
                    "asset_id": f"mask-{vial_id}",
                    "src": f"/assets/vials/{vial_id}-mask.png",
                    "provenance": provenances["predicted_mask"],
                },
                "target_lab": {
                    "l": right["median_lab"][0],
                    "a": right["median_lab"][1],
                    "b": right["median_lab"][2],
                    "provenance": provenances["target_lab"],
                },
                "reference_lab": {
                    "l": left["median_lab"][0],
                    "a": left["median_lab"][1],
                    "b": left["median_lab"][2],
                    "provenance": provenances["reference_lab"],
                },
                "history_refs": [f"history-mock-{index:03d}"],
            }
        )
        refs = [record["evidence_id"] for record in evidence]
        vial["decision"].update(
            {
                "outcome": outcome,
                "delta_e": {"value": row["delta_e76"], "provenance": provenances["delta_e"]},
                "threshold": {"value": 10.0, "provenance": provenances["threshold"]},
                "limit_codes": (
                    ["measurement_unavailable", "mock_quality_gate_unavailable"]
                    if outcome == "insufficient_evidence"
                    else ["within_delta_e_limit"] if outcome == "keep"
                    else ["exceeds_delta_e_limit"]
                ),
                "evidence_refs": refs,
                "explanation": {
                    "source": "deterministic_template",
                    "text": (
                        f"Mock-only {outcome} decision from measured Delta E "
                        f"{row['delta_e76']:.6f} against documented threshold 10.0; "
                        "WP5 provides measurements only."
                    ),
                    "evidence_refs": refs,
                },
            }
        )
        data["top_down"]["observations"][index - 1]["vial_id"] = vial_id
        data["top_down"]["observations"][index - 1]["decision_id"] = vial["decision"]["record_id"]
        geometry_provenance = {
            "kind": "inferred",
            "source_id": "phase1-mock-top-down-geometry",
            "note": "Inferred mock synthetic-twin top-down geometry; not a measured plant position.",
        }
        data["top_down"]["observations"][index - 1]["position"]["provenance"] = geometry_provenance
        data["top_down"]["observations"][index - 1]["bbox"]["provenance"] = geometry_provenance
        reference_evidence = f"evidence-{pair_id}-reference_lab"
        history.append(
            {
                "history_id": f"history-mock-{index:03d}",
                "recipe_id": row["recipe_id"],
                "run_id": row["run_id"],
                "vial_id": row["left_vial_id"],
                "time_seconds": {
                    "value": index,
                    "provenance": {
                        "kind": "inferred",
                        "source_id": COMPOSITE_SOURCE,
                        "note": "Unique mock relative ordering; not documented plant timing.",
                    },
                },
                "decision": outcome,
                "explanation": {
                    "source": "deterministic_template",
                    "text": (
                        f"Inferred mock prior {outcome} decision for measured reference "
                        f"{left['measurement_id']}."
                    ),
                    "evidence_refs": [reference_evidence],
                },
                "provenance": {
                    "kind": "inferred",
                    "source_id": pair_id,
                    "note": (
                        "Run, vial, and recipe identity come from the measured pair's left "
                        "record; relative timing and prior decision are inferred mock values."
                    ),
                },
            }
        )
    data["history"] = history
    data["evidence"] = all_evidence
    data["run"]["cue_times"] = [
        {
            "vial_id": f"vial-{index:03d}",
            "start_seconds": cue[0],
            "end_seconds": cue[1],
            "provenance": {
                "kind": "inferred",
                "source_id": COMPOSITE_SOURCE,
                "note": "Fill-section timing inferred from the derived local composite; not documented plant timing.",
            },
        }
        for index, cue in enumerate(fill_ranges, start=1)
    ]
    recipe_measurements = {}
    for pair in pairs:
        recipe_measurements.setdefault(pair["right"]["recipe_id"], pair["right"])
    data["recipes"] = [
        {
            "recipe_id": recipe_id,
            "label": f"WP5 recipe {recipe_id}",
            "rgb": {
                "r": int(measurement["median_srgb_0_255"][0]),
                "g": int(measurement["median_srgb_0_255"][1]),
                "b": int(measurement["median_srgb_0_255"][2]),
                "provenance": {
                    "kind": "measured",
                    "source_id": measurement["measurement_id"],
                    "note": "WP5 selected right-stage recipe RGB measurement.",
                },
            },
        }
        for recipe_id, measurement in sorted(recipe_measurements.items())
    ]
    data["wp6"]["rows"] = wp6_rows
    return data


def main():
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--wp1-batch", type=Path, required=True)
    parser.add_argument("--wp5-measurements", type=Path, required=True)
    parser.add_argument("--wp5-locators", type=Path, required=True)
    parser.add_argument("--wp6-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    args = parser.parse_args()

    output = args.root / "ui-data"
    vial_assets = output / "assets" / "vials"
    run_assets = output / "assets" / "run"
    vial_assets.mkdir(parents=True, exist_ok=True)
    run_assets.mkdir(parents=True, exist_ok=True)

    validate_wp1_batch(args.wp1_batch)
    wp5, locators = validate_wp5(args.wp5_measurements, args.wp5_locators)
    pairs = select_pairs(wp5, locators)
    wp6_rows = load_wp6_rows(args.wp6_dir)

    for index, pair in enumerate(pairs, start=1):
        with Image.open(pair["rgb_path"]) as image:
            image.convert("RGB").resize((320, 240)).save(
                vial_assets / f"vial-{index:03d}-rgb.jpg", quality=88
            )
        with Image.open(pair["mask_path"]) as image:
            image.convert("L").resize((320, 240)).save(
                vial_assets / f"vial-{index:03d}-mask.png"
            )

    fills = manifest_fill_jobs(args.wp1_batch)
    video_path = run_assets / "run-01-composite.mp4"
    duration, fill_ranges = build_media(args.ffmpeg, args.wp1_batch, fills, video_path)

    template = load_json(output / "example.json", "UI example template")
    fixture = build_fixture(template, pairs, wp6_rows, duration, fill_ranges)
    (output / "mock-run.json").write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()

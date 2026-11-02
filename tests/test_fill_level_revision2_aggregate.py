"""CPU-only tests for the Step 8 Revision 2 combined aggregate builder."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix import fill_level_training as training  # noqa: E402


SCRIPT = ROOT / "scripts" / "build_fill_level_revision2_aggregate.py"
SPEC = importlib.util.spec_from_file_location("fill_level_revision2_aggregate", SCRIPT)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)
REVIEW_SCRIPT = ROOT / "scripts" / "review_fill_level_exceptions.py"
REVIEW_SPEC = importlib.util.spec_from_file_location(
    "review_fill_level_exceptions_for_aggregate", REVIEW_SCRIPT
)
REVIEW = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(REVIEW)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _attempt_name(component, chunk_name):
    if component == "step6" and chunk_name.startswith("00-"):
        return "attempt-03"
    if component == "step6" and chunk_name.startswith("03-"):
        return "attempt-02"
    if component == "v4" and chunk_name.startswith("08-"):
        return "attempt-02"
    return "attempt-01"


def _manifest(role, seed_start, seed_end):
    records = []
    for index, seed in enumerate(range(seed_start, seed_end + 1)):
        records.append(
            json.dumps(
                {
                    "role": role,
                    "seed": seed,
                    "frame_index": index,
                    "outputs": {
                        "rgb": {"path": f"rgb/{seed}.png", "sha256": "a" * 64},
                        "mask": {"path": f"mask/{seed}.png", "sha256": "b" * 64},
                    },
                },
                sort_keys=True,
            )
        )
    return "\n".join(records) + "\n"


def _coverage(counts):
    by_role = {
        role: {
            "clean": count,
            "foam": 0,
            "sticker": 0,
            "edge_zone": 0,
            "hidden": 0,
            "wet_wall": 0,
            "zero_level": 0,
            "sampling_bands": {"fixture": count},
        }
        for role, count in counts.items()
    }
    total = {
        key: sum(by_role[role][key] for role in by_role)
        for key in ("clean", "foam", "sticker", "edge_zone", "hidden", "wet_wall", "zero_level")
    }
    total["sampling_bands"] = {"fixture": sum(counts.values())}
    return {"total": total, "by_role": by_role}


def _component(tmp_path, component, plan, counts, exceptions=()):
    collected = tmp_path / f"{component}-collected"
    collected.mkdir()
    chunks = []
    for chunk_name, role, seed_start, seed_end, frame_count in plan:
        attempt_name = _attempt_name(component, chunk_name)
        direct = component == "v4" and chunk_name.startswith("00-")
        attempt = collected / chunk_name if direct else collected / chunk_name / attempt_name
        manifest_hash = "f" * 64
        if role != "test":
            output = attempt / "output"
            output.mkdir(parents=True)
            (attempt / "state.txt").write_text("COMPLETE\n", encoding="utf-8")
            manifest = output / "manifest.jsonl"
            manifest.write_text(_manifest(role, seed_start, seed_end), encoding="utf-8")
            manifest_hash = _sha256(manifest)
        chunk = {
            "chunk": chunk_name,
            "role": role,
            "attempt": attempt_name,
            "frame_count": frame_count,
            "seed_start": seed_start,
            "seed_end": seed_end,
            "manifest_sha256": manifest_hash,
            "integrity_passed": True,
        }
        if direct:
            chunk["storage_layout"] = "direct_copy_of_attempt-01"
        chunks.append(chunk)

    summary = {
        "schema_version": 1,
        "purpose": f"{component} fixture",
        "integrity": {"passed": True, "errors": []},
        "batch_contract": {
            "expected_chunks": len(plan),
            "expected_role_counts": counts,
        },
        "progress": {
            "complete_chunks": len(plan),
            "expected_chunks": len(plan),
            "frames": sum(counts.values()),
            "expected_frames": sum(counts.values()),
            "role_counts": counts,
        },
        "case_coverage": _coverage(counts),
        "old_rule_review": {
            "exception_count": len(exceptions),
            "exceptions": list(exceptions),
        },
        "chunks": chunks,
    }
    path = tmp_path / f"{component}-summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    return path, collected


def _fixture(tmp_path):
    step6_exception = {
        "seed": 100327,
        "role": "train",
        "chunk": "03-train-0100200-0100399",
        "frame_index": 127,
        "level_ml": 0.05,
        "sampling_band": "full_0_85",
        "failed_old_rules": ["positive_level_without_visible_liquid"],
        "class_pixels": {"liquid": 0},
        "foam": {"visible_in_camera": True},
        "sticker": {"visible_in_camera": True},
    }
    v4_exception = {
        "seed": 1101007,
        "role": "val",
        "chunk": "08-val-1101000-1101059",
        "frame_index": 7,
        "level_ml": 50.0,
        "sampling_band": "fixture",
        "failed_old_rules": ["positive_level_without_visible_liquid"],
        "class_pixels": {"liquid": 0},
        "foam": {"visible_in_camera": True},
        "sticker": {"visible_in_camera": False},
    }
    step6_summary, step6_root = _component(
        tmp_path,
        "step6",
        BUILD.STEP6_PLAN,
        {"train": 3000, "val": 300, "test": 500},
        (step6_exception,),
    )
    v4_summary, v4_root = _component(
        tmp_path,
        "v4",
        BUILD.V4_PLAN,
        {"train": 480, "val": 60, "test": 60},
        (v4_exception,),
    )
    step6_value = json.loads(step6_summary.read_text(encoding="utf-8"))
    reviewed = []
    chunks = {chunk["chunk"]: chunk for chunk in step6_value["chunks"]}
    for item in step6_value["old_rule_review"]["exceptions"]:
        chunk = chunks[item["chunk"]]
        reviewed.append(
            {
                **item,
                "attempt": chunk["attempt"],
                "disposition": "accepted_legitimate_occlusion",
                "explanation": "fixture explanation",
                "overlay_sha256": "c" * 64,
                "source": {"manifest_sha256": chunk["manifest_sha256"]},
            }
        )
    step6_review = tmp_path / "step6-review.json"
    step6_review.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "batch_summary_sha256": _sha256(step6_summary),
                "exception_count": len(reviewed),
                "reviewed_count": len(reviewed),
                "unresolved_count": 0,
                "excluded_train_validation_seeds": [],
                "eligible_counts": {"train": 3000, "val": 300},
                "exceptions": reviewed,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return step6_summary, step6_review, step6_root, v4_summary, v4_root


def _build(tmp_path, monkeypatch):
    step6_summary, step6_review, step6_root, v4_summary, v4_root = _fixture(tmp_path)
    aggregate = tmp_path / "aggregate"
    report = tmp_path / "combined.json"

    def copy_attempt(path, target):
        shutil.copytree(target, path)

    monkeypatch.setattr(BUILD, "_create_directory_link", copy_attempt)
    result = BUILD.build_aggregate(
        step6_summary_path=step6_summary,
        step6_summary_sha256=_sha256(step6_summary),
        step6_review_path=step6_review,
        step6_review_sha256=_sha256(step6_review),
        step6_collected_root=step6_root,
        v4_summary_path=v4_summary,
        v4_summary_sha256=_sha256(v4_summary),
        v4_collected_root=v4_root,
        aggregate_root=aggregate,
        report_out=report,
    )
    return result, aggregate, report


def test_directory_link_works_without_windows_symlink_privilege(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    BUILD._create_directory_link(link, target)
    assert link.is_dir()
    assert link.resolve() == target.resolve()
    link.rmdir()
    assert target.is_dir()


def test_builds_hash_bound_train_val_only_aggregate(tmp_path, monkeypatch):
    result, aggregate, report_path = _build(tmp_path, monkeypatch)

    assert result["integrity"] == {
        "passed": True,
        "errors": [],
        "component_summaries_passed": 2,
        "complete_chunks": 30,
        "linked_train_validation_chunks": 26,
        "test_chunks_linked": 0,
        "linked_manifest_hashes_verified": 26,
        "aggregate_root_contains_test_chunk": False,
    }
    assert result["batch_contract"]["expected_role_counts"] == {
        "train": 3480,
        "val": 360,
        "test": 560,
    }
    assert result["case_coverage"]["by_role"]["val"]["clean"] == 360
    assert len(result["chunks"]) == 30
    assert not any("-test-" in path.name for path in aggregate.iterdir())
    assert {path.name for path in aggregate.iterdir()} == {
        chunk["chunk"] for chunk in result["chunks"] if chunk["role"] != "test"
    }
    assert any(chunk["chunk"] == "20-train-0104000-0104059" for chunk in result["chunks"])
    assert any(chunk["chunk"] == "29-test-2101000-2101059" for chunk in result["chunks"])
    assert next(
        chunk for chunk in result["chunks"] if chunk["chunk"] == "28-val-1101000-1101059"
    )["attempt"] == "attempt-02"
    assert (aggregate / "28-val-1101000-1101059" / "attempt-02").is_dir()
    assert result["old_rule_review"]["exception_count"] == 2
    assert {item["chunk"] for item in result["old_rule_review"]["exceptions"]} == {
        "03-train-0100200-0100399",
        "28-val-1101000-1101059",
    }
    assert result["components"]["step6"]["exception_review_sha256"] == _sha256(
        tmp_path / "step6-review.json"
    )

    def fake_verify(_exception, _record, output_dir):
        source = output_dir / "manifest.jsonl"
        measured = {"empty": 1, "liquid": 0, "foam": 0, "blocked": 0, "ignore": 0}
        return source, source, None, None, measured

    def fake_overlays(_overlay_dir, exception, _record, _rgb, _mask):
        return [
            {
                "filename": f"seed-{exception['seed']}-fixture.png",
                "relative_ignored_path": f"fixture/seed-{exception['seed']}.png",
                "sha256": "d" * 64,
                "width": 1,
                "height": 1,
            }
        ]

    monkeypatch.setattr(REVIEW, "_verify_record", fake_verify)
    monkeypatch.setattr(REVIEW, "_write_overlays", fake_overlays)
    monkeypatch.setattr(
        REVIEW,
        "_approved_disposition",
        lambda _exception, _measured: {
            "disposition": "accepted_legitimate_occlusion",
            "explanation": "fixture disposition",
        },
    )
    combined_review = REVIEW.review_exceptions(
        report_path,
        aggregate,
        tmp_path / "combined-overlays",
    )
    assert combined_review["reviewed_count"] == 2
    assert {item["chunk"] for item in combined_review["exceptions"]} == {
        "03-train-0100200-0100399",
        "28-val-1101000-1101059",
    }

    review = {
        "status": "complete",
        "batch_summary_sha256": _sha256(report_path),
        "eligible_counts": {"train": 3480, "val": 360},
        "excluded_train_validation_seeds": [],
        "exceptions": [
            {
                "seed": item["seed"],
                "role": item["role"],
                "disposition": "accepted_legitimate_occlusion",
                "explanation": "fixture disposition",
                "overlay_sha256": "c" * 64,
            }
            for item in result["old_rule_review"]["exceptions"]
        ],
    }
    review_path = tmp_path / "combined-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    audit = training.validate_exception_review(report_path, review_path)
    samples = training.discover_train_val_samples(
        aggregate,
        expected_train=3480,
        expected_val=360,
        excluded_seeds=audit["excluded_seeds"],
        expected_chunks=audit["approved_chunks"],
    )
    assert len(samples["train"]) == 3480
    assert len(samples["val"]) == 360
    assert len({sample.seed for role in samples.values() for sample in role}) == 3840


def test_refuses_reuse_and_component_hash_mismatch(tmp_path, monkeypatch):
    result, aggregate, report = _build(tmp_path, monkeypatch)
    assert result["integrity"]["passed"]
    with pytest.raises(FileExistsError, match="aggregate root"):
        BUILD.build_aggregate(
            step6_summary_path=tmp_path / "step6-summary.json",
            step6_summary_sha256=_sha256(tmp_path / "step6-summary.json"),
            step6_review_path=tmp_path / "step6-review.json",
            step6_review_sha256=_sha256(tmp_path / "step6-review.json"),
            step6_collected_root=tmp_path / "step6-collected",
            v4_summary_path=tmp_path / "v4-summary.json",
            v4_summary_sha256=_sha256(tmp_path / "v4-summary.json"),
            v4_collected_root=tmp_path / "v4-collected",
            aggregate_root=aggregate,
            report_out=tmp_path / "fresh.json",
        )
    aggregate.rename(tmp_path / "moved-aggregate")
    with pytest.raises(FileExistsError, match="combined summary"):
        BUILD.build_aggregate(
            step6_summary_path=tmp_path / "step6-summary.json",
            step6_summary_sha256=_sha256(tmp_path / "step6-summary.json"),
            step6_review_path=tmp_path / "step6-review.json",
            step6_review_sha256=_sha256(tmp_path / "step6-review.json"),
            step6_collected_root=tmp_path / "step6-collected",
            v4_summary_path=tmp_path / "v4-summary.json",
            v4_summary_sha256=_sha256(tmp_path / "v4-summary.json"),
            v4_collected_root=tmp_path / "v4-collected",
            aggregate_root=tmp_path / "fresh-aggregate",
            report_out=report,
        )


def test_final_summary_creation_is_exclusive_under_a_race(tmp_path, monkeypatch):
    step6_summary, step6_review, step6_root, v4_summary, v4_root = _fixture(tmp_path)
    aggregate = tmp_path / "aggregate"
    report = tmp_path / "combined.json"

    def copy_and_race(path, target):
        shutil.copytree(target, path)
        if not report.exists():
            report.write_text("raced\n", encoding="utf-8")

    monkeypatch.setattr(BUILD, "_create_directory_link", copy_and_race)
    with pytest.raises(FileExistsError):
        BUILD.build_aggregate(
            step6_summary_path=step6_summary,
            step6_summary_sha256=_sha256(step6_summary),
            step6_review_path=step6_review,
            step6_review_sha256=_sha256(step6_review),
            step6_collected_root=step6_root,
            v4_summary_path=v4_summary,
            v4_summary_sha256=_sha256(v4_summary),
            v4_collected_root=v4_root,
            aggregate_root=aggregate,
            report_out=report,
        )
    assert report.read_text(encoding="utf-8") == "raced\n"

    report.unlink()
    with pytest.raises(ValueError, match="summary SHA-256 mismatch"):
        BUILD.build_aggregate(
            step6_summary_path=tmp_path / "step6-summary.json",
            step6_summary_sha256="0" * 64,
            step6_review_path=tmp_path / "step6-review.json",
            step6_review_sha256=_sha256(tmp_path / "step6-review.json"),
            step6_collected_root=tmp_path / "step6-collected",
            v4_summary_path=tmp_path / "v4-summary.json",
            v4_summary_sha256=_sha256(tmp_path / "v4-summary.json"),
            v4_collected_root=tmp_path / "v4-collected",
            aggregate_root=tmp_path / "fresh-aggregate",
            report_out=report,
        )


def test_component_contract_tamper_is_rejected_before_output(tmp_path, monkeypatch):
    step6_summary, step6_review, step6_root, v4_summary, v4_root = _fixture(tmp_path)
    value = json.loads(v4_summary.read_text(encoding="utf-8"))
    value["batch_contract"]["expected_role_counts"]["val"] = 59
    v4_summary.write_text(json.dumps(value), encoding="utf-8")
    aggregate = tmp_path / "aggregate"
    report = tmp_path / "combined.json"
    monkeypatch.setattr(BUILD, "_create_directory_link", lambda path, target: None)

    with pytest.raises(ValueError, match="expected role counts"):
        BUILD.build_aggregate(
            step6_summary_path=step6_summary,
            step6_summary_sha256=_sha256(step6_summary),
            step6_review_path=step6_review,
            step6_review_sha256=_sha256(step6_review),
            step6_collected_root=step6_root,
            v4_summary_path=v4_summary,
            v4_summary_sha256=_sha256(v4_summary),
            v4_collected_root=v4_root,
            aggregate_root=aggregate,
            report_out=report,
        )
    assert not aggregate.exists()
    assert not report.exists()


def test_incomplete_component_and_unbound_step6_review_are_rejected(tmp_path, monkeypatch):
    step6_summary, step6_review, step6_root, v4_summary, v4_root = _fixture(tmp_path)
    aggregate = tmp_path / "aggregate"
    report = tmp_path / "combined.json"
    monkeypatch.setattr(BUILD, "_create_directory_link", lambda path, target: None)

    value = json.loads(v4_summary.read_text(encoding="utf-8"))
    value["progress"]["complete_chunks"] = 9
    v4_summary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="complete batch"):
        BUILD.build_aggregate(
            step6_summary_path=step6_summary,
            step6_summary_sha256=_sha256(step6_summary),
            step6_review_path=step6_review,
            step6_review_sha256=_sha256(step6_review),
            step6_collected_root=step6_root,
            v4_summary_path=v4_summary,
            v4_summary_sha256=_sha256(v4_summary),
            v4_collected_root=v4_root,
            aggregate_root=aggregate,
            report_out=report,
        )
    assert not aggregate.exists()

    value["progress"]["complete_chunks"] = 10
    v4_summary.write_text(json.dumps(value), encoding="utf-8")
    review = json.loads(step6_review.read_text(encoding="utf-8"))
    review["batch_summary_sha256"] = "0" * 64
    step6_review.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        BUILD.build_aggregate(
            step6_summary_path=step6_summary,
            step6_summary_sha256=_sha256(step6_summary),
            step6_review_path=step6_review,
            step6_review_sha256=_sha256(step6_review),
            step6_collected_root=step6_root,
            v4_summary_path=v4_summary,
            v4_summary_sha256=_sha256(v4_summary),
            v4_collected_root=v4_root,
            aggregate_root=aggregate,
            report_out=report,
        )
    assert not aggregate.exists()
    assert not report.exists()


def test_refuses_to_write_aggregate_or_report_inside_collected_evidence(tmp_path):
    step6_summary, step6_review, step6_root, v4_summary, v4_root = _fixture(tmp_path)
    common = {
        "step6_summary_path": step6_summary,
        "step6_summary_sha256": _sha256(step6_summary),
        "step6_review_path": step6_review,
        "step6_review_sha256": _sha256(step6_review),
        "step6_collected_root": step6_root,
        "v4_summary_path": v4_summary,
        "v4_summary_sha256": _sha256(v4_summary),
        "v4_collected_root": v4_root,
    }
    with pytest.raises(ValueError, match="aggregate root"):
        BUILD.build_aggregate(
            **common,
            aggregate_root=step6_root / "forbidden-aggregate",
            report_out=tmp_path / "combined.json",
        )
    with pytest.raises(ValueError, match="combined summary"):
        BUILD.build_aggregate(
            **common,
            aggregate_root=tmp_path / "aggregate",
            report_out=v4_root / "forbidden-summary.json",
        )
    with pytest.raises(ValueError, match="must not overlap"):
        BUILD.build_aggregate(
            **common,
            aggregate_root=tmp_path / "aggregate-with-summary",
            report_out=tmp_path / "aggregate-with-summary" / "combined.json",
        )
    with pytest.raises(ValueError, match="must not overlap"):
        BUILD.build_aggregate(
            **common,
            aggregate_root=tmp_path / "report-parent" / "aggregate",
            report_out=tmp_path / "report-parent",
        )

    escaped_chunk = v4_root / "01-train-0104060-0104119"
    outside = tmp_path / "outside-collected-root"
    escaped_chunk.rename(outside)
    BUILD._create_directory_link(escaped_chunk, outside)
    with pytest.raises(ValueError, match="chunk path escapes"):
        BUILD.build_aggregate(
            **common,
            aggregate_root=tmp_path / "fresh-aggregate",
            report_out=tmp_path / "fresh-summary.json",
        )
    if escaped_chunk.is_symlink():
        escaped_chunk.unlink()
    else:
        escaped_chunk.rmdir()
    assert not (step6_root / "forbidden-aggregate").exists()
    assert not (v4_root / "forbidden-summary.json").exists()

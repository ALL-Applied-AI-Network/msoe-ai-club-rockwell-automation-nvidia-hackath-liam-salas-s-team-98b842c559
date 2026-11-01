"""No-oracle extension of the accepted WP5 store by one V4 train run.

The inference boundary accepts only RGB images plus a same-vial pre-pour RGB
reference. Renderer receipts and offline validation records are hash-bound
provenance: their JSON content is never parsed by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set
from copy import deepcopy
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess


PREDICTOR = {
    "producer_id": "frozen-wp5-six-channel-cnn",
    "producer_kind": "rgb_only_liquid_segmenter",
    "model_id": "resnet18-unet-current-empty-rgb",
    "model_sha256": "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
    "config_id": "wp5-cnn-six-channel-frozen-v1",
    "config_sha256": "7be72579f0cb0dc94f6bbe2cf894f46e63f24b9254f131106a0ed7805e96c64f",
    "code_commit": "a5f29ea90272a3424d8c7de09dbf435509d28105",
    "threshold_source": "twin_validation_frozen",
    "input_fields": ["rgb"],
    "training_data_origin": "twin",
}
CNN_PREDICTOR_SOURCE_SHA256 = "ec9f4b589f088fe979f084ebff1ad51cb0fe0dd29cf12a64a7c993bab534be19"
CNN_CHECKPOINT_PATH = (
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp5-cnn-predictor-01a18128/candidate-a5f29ea/"
    "resnet18-unet-current-empty-rgb.pt"
)
CNN_REPLAY_PATH = (
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp5-cnn-harness-recovery-01a18128/replay-9edd739/"
    "deterministic-bounds-replay.json"
)
CNN_REVIEW_PATH = (
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp5-cnn-harness-recovery-01a18128/replay-9edd739/"
    "independent-review-root-wp5_cnn_replay_review.md"
)

SOURCE_PLAN_PATH = (
    "/home/hackathon5/liamreporoot/f2-varied-v4-package-660bc43/"
    "render-schedule-plan.json"
)
DEFAULT_LINEAGE: dict[str, object] = {
    "run_id": "RUN-V4-TRAIN-01",
    "batch_id": "BATCH_90290ed52b6d0e17",
    "source_plan_sha256": "867fa9516d13c79cbe9fd2960a25477926d17dda4834ec81adc5df55e046d37f",
    "vials": {
        "V1": {
            "case_id": "CASE_97a8c6cb8161881c",
            "previous_recipe_id": "14",
            "seed": 520101,
            "shard": "varied-v4-660bc43-rt10-n30-i00-r1",
            "source_receipt_sha256": "b29af7d809a62bbc6a30c3045e101ec4bf635ecaeb4c55f7d74e3de69c71b5e8",
            "validation_receipt_sha256": "3aaac81a861f847970d9c096c37a7785f4945b0703ccc407176164c78e155a9d",
            "prefill_rgb_sha256": "fd7a2b8b22c8de462d2f67badca9dc5526bf3717ad6dd3794df89aa87445426f",
            "fill_clip_sha256": "1470b69eb6d426bdd0a99549f904294b5b69f2232da5ac34242ed7dad2ff1962",
            "stage_files": [
                ("frame-0001-step-01412-t200-stage.png", "edcab91c3066612dec8b052ce849f5ea09db9ef23c1c20b33a7b9ed6a1b4ad69"),
                ("frame-0002-step-02622-t400-stage.png", "359ef307dadf99bda2455e3acf0bec18e13ecdd9f7b0b0374c3d2f5427d30138"),
                ("frame-0003-step-03730-t100-stage.png", "e6fcee398cffdbc29898a74f3f427478edb1f695811516e84cdfe5663d819a98"),
            ],
        },
        "V2": {
            "case_id": "CASE_59038b977a73fdfb",
            "previous_recipe_id": "5",
            "seed": 520102,
            "shard": "varied-v4-660bc43-rt10-n30-i01-r1",
            "source_receipt_sha256": "39c08867ecef45cd01c869208ec664f63dbb52a80740b08a83811ce5a85c447e",
            "validation_receipt_sha256": "6dfbceb863d9b27d4acf674e568d75be40bf88df3e6e0a0b9c296f2c08428e34",
            "prefill_rgb_sha256": "12ccb47c15bda47b6b29c1a67647dbedadb12b080b1e6b49bbe3a23a066b1c15",
            "fill_clip_sha256": "b1062068f2245b684522ac4de874c9e912c3666477e43fa2cbad183cc3b972c8",
            "stage_files": [
                ("frame-0001-step-01412-t200-stage.png", "bd8ca8ac46564fc2c8abf2a9d9c452dff3fcf78be41a50009b3188ea07c0122f"),
                ("frame-0002-step-02622-t400-stage.png", "327fc5a76ea3faeb21a5e9eca317f8ec9b4e50e5ba9b88badf2feb6aaf91162b"),
                ("frame-0003-step-03730-t100-stage.png", "a51b93ed8805a5d94f6f1e1bd26b929f5f0d3f0ed092390aefdfbe730954ff60"),
            ],
        },
        "V3": {
            "case_id": "CASE_0b872555c4f08091",
            "previous_recipe_id": "5",
            "seed": 520103,
            "shard": "varied-v4-660bc43-rt10-n30-i02-r1",
            "source_receipt_sha256": "b0df5192514ca6de3e0f35e2ec3f6073d3accbb97a237fca3473a2fb50002c22",
            "validation_receipt_sha256": "6328e3e5894bf2c88e31f11ece895ecc6b2d7f7e5634962b74ffc9dde3c3d684",
            "prefill_rgb_sha256": "b5f36a3e41e4eb42b55d4f77d9b84b0bf5bd60f095629654f509bc45142dea63",
            "fill_clip_sha256": "0e087aa4af21f951a9a024a155be959c82b76ea0358d3e6b6332538c652767b2",
            "stage_files": [
                ("frame-0001-step-01412-t200-stage.png", "e9bcdd5846de5d268dcd2cf4c480c7bc049411ce8ba9812547ff6873a2620d4f"),
                ("frame-0002-step-02622-t400-stage.png", "7c6be7982a616a8219a610a1c9f6cadce3f738b03ef0c6e5a843e48d8609bb1e"),
                ("frame-0003-step-03730-t100-stage.png", "e61e6d24a687a6880f9024e9eef69384844627ed97bc91a936455996aec53153"),
            ],
        },
    },
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOP_FIELDS = {
    "schema_version", "artifact_type", "status", "data_origin", "run_id",
    "batch_id", "split_role", "split_version", "source_plan_path",
    "source_plan_sha256", "predictor", "recipe_context", "vials",
}
_RECIPE_FIELDS = {
    "recipe_id", "recipe_ml", "dispense_order",
}
_VIAL_FIELDS = {
    "vial_id", "vial_ordinal", "case_id", "previous_recipe_id", "seed", "source_receipt_path",
    "source_receipt_sha256", "validation_receipt_path",
    "validation_receipt_sha256", "prefill_rgb_path", "prefill_rgb_sha256",
    "fill_clip_path", "fill_clip_sha256", "stages",
}
_STAGE_FIELDS = {
    "image_id", "stage_id", "stage_ordinal", "tank_id", "rgb_path",
    "rgb_sha256",
}
BASE_STORE_PATH = (
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp5-cnn-swap-store-fc694db-01a18128-attempt01/evidence.sqlite3"
)
BASE_STORE_SHA256 = "f41c29ef150f32ee88a3fa9376c0b6e26111654d2e14b3e663b6c2ee29f03dcd"
BASE_RECEIPT_PATH = (
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp5-cnn-swap-store-fc694db-01a18128-attempt01/population-receipt.json"
)
BASE_RECEIPT_SHA256 = "4752439c2d7a7a212188928b072fc625de2bd02ba33d588df78ce74bc426c636"
BASE_LOCATORS_PATH = (
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp5-cnn-swap-materialization-d89a3f9-01a18128-attempt02/"
    "prediction-locators.json"
)
BASE_LOCATORS_SHA256 = "578ab949803214b12cd4942cff7adeda3e89610eff92523505cf7c0436e5e269"
ARTIFACT_ROOT = "/home/hackathon5/liamreporoot"
EXISTING_RUN_IDS = frozenset(
    {"RUN-TRAIN-01", "RUN-TRAIN-02", "RUN-TRAIN-03", "RUN-TRAIN-04", "RUN-VALIDATION-01"}
)
_OWNER = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")


class SixthDevelopmentRunError(ValueError):
    """The bounded sixth-run extension failed closed."""


def default_source_manifest() -> dict[str, object]:
    """Return the exact accepted V4 train-01 RGB-only source declaration."""

    root = "/home/hackathon5/liamreporoot"
    raw_vials = DEFAULT_LINEAGE["vials"]
    assert isinstance(raw_vials, Mapping)
    vials: list[dict[str, object]] = []
    for ordinal, tank_vial in enumerate(("V1", "V2", "V3"), 1):
        lineage = raw_vials[tank_vial]
        assert isinstance(lineage, Mapping)
        case_id = str(lineage["case_id"])
        app_output = (
            f"{root}/f2-varied-v4-shards-660bc43/{lineage['shard']}/"
            f"cases/{case_id}/app-output"
        )
        stages = []
        for stage_ordinal, (tank_id, file_binding) in enumerate(
            zip(("t200", "t400", "t100"), lineage["stage_files"], strict=True), 1
        ):
            file_name, file_sha = file_binding
            stage_id = f"dispense-{stage_ordinal}-{tank_id}"
            stages.append(
                {
                    "image_id": f"{DEFAULT_LINEAGE['batch_id']}-{tank_vial}-{stage_id}",
                    "stage_id": stage_id,
                    "stage_ordinal": stage_ordinal,
                    "tank_id": tank_id,
                    "rgb_path": f"{app_output}/frames/{file_name}",
                    "rgb_sha256": file_sha,
                }
            )
        vials.append(
            {
                "vial_id": tank_vial,
                "vial_ordinal": ordinal,
                "case_id": case_id,
                "previous_recipe_id": lineage["previous_recipe_id"],
                "seed": lineage["seed"],
                "source_receipt_path": f"{app_output}/receipt.json",
                "source_receipt_sha256": lineage["source_receipt_sha256"],
                "validation_receipt_path": (
                    f"{root}/f2-varied-v4-stop-incident-20261028/validation/"
                    f"{case_id}.json"
                ),
                "validation_receipt_sha256": lineage["validation_receipt_sha256"],
                "prefill_rgb_path": f"{app_output}/pre-pour-parity/f2-prefill-source.png",
                "prefill_rgb_sha256": lineage["prefill_rgb_sha256"],
                "fill_clip_path": f"{app_output}/three-pour.avi",
                "fill_clip_sha256": lineage["fill_clip_sha256"],
                "stages": stages,
            }
        )
    return {
        "schema_version": 2,
        "artifact_type": "wp5_sixth_development_run_source",
        "status": "accepted",
        "data_origin": "twin",
        "run_id": DEFAULT_LINEAGE["run_id"],
        "batch_id": DEFAULT_LINEAGE["batch_id"],
        "split_role": "train",
        "split_version": "whole-run-varied-v4",
        "source_plan_path": SOURCE_PLAN_PATH,
        "source_plan_sha256": DEFAULT_LINEAGE["source_plan_sha256"],
        "predictor": deepcopy(PREDICTOR),
        "recipe_context": {
            "recipe_id": "5",
            "recipe_ml": {"t100": 10.0, "t200": 10.0, "t400": 10.0},
            "dispense_order": ["t200", "t400", "t100"],
        },
        "vials": vials,
    }


def _fail(code: str) -> SixthDevelopmentRunError:
    return SixthDevelopmentRunError(code)


def _exact(value: object, fields: set[str], role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _fail(f"{role}_fields_invalid")
    return value


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(f"{role}_sha256_invalid")
    return value


def _absolute(value: object, role: str) -> str:
    from pathlib import Path

    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise _fail(f"{role}_path_invalid")
    return value


def _hash_file(path_value: object, expected_value: object, role: str) -> dict[str, str]:
    from pathlib import Path

    path = Path(_absolute(path_value, role)).absolute()
    expected = _sha(expected_value, role)
    try:
        if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise OSError("source is not a canonical regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _fail(f"{role}_file_invalid") from error
    if digest.hexdigest() != expected:
        raise _fail(f"{role}_hash_mismatch")
    return {"role": role, "path": str(path), "sha256": expected}


def verify_source_files(manifest: Mapping[str, object]) -> list[dict[str, str]]:
    """Hash only allowed V4 RGB/clip/provenance files; never open the plan or masks."""

    reads: list[dict[str, str]] = []
    vials = manifest.get("vials")
    if not isinstance(vials, list) or len(vials) != 3:
        raise _fail("vial_denominator_invalid")
    for vial in vials:
        if not isinstance(vial, Mapping):
            raise _fail("vial_fields_invalid")
        reads.extend(
            (
                _hash_file(vial.get("source_receipt_path"), vial.get("source_receipt_sha256"), "source_receipt"),
                _hash_file(vial.get("validation_receipt_path"), vial.get("validation_receipt_sha256"), "validation_receipt"),
                _hash_file(vial.get("prefill_rgb_path"), vial.get("prefill_rgb_sha256"), "prefill_rgb"),
                _hash_file(vial.get("fill_clip_path"), vial.get("fill_clip_sha256"), "fill_clip"),
            )
        )
        stages = vial.get("stages")
        if not isinstance(stages, list) or len(stages) != 3:
            raise _fail("stage_denominator_invalid")
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise _fail("stage_fields_invalid")
            reads.append(
                _hash_file(stage.get("rgb_path"), stage.get("rgb_sha256"), "stage_rgb")
            )
    return reads


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _write_new(path: object, payload: bytes) -> None:
    from pathlib import Path

    target = Path(path)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _load_rgb(path_value: object, expected_value: object, role: str):
    from pathlib import Path
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    path = Path(_absolute(path_value, role)).absolute()
    expected = _sha(expected_value, role)
    if _hash_file(path_value, expected_value, role)["sha256"] != expected:
        raise AssertionError("hash verifier returned an inconsistent digest")
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 480):
                raise _fail(f"{role}_png_invalid")
            image.load()
            return np.asarray(image, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        if isinstance(error, SixthDevelopmentRunError):
            raise
        raise _fail(f"{role}_png_invalid") from error


def _load_frozen_cnn(**paths: object):
    from ...perception.materializer import _load_frozen_cnn as load

    return load(**paths)


def _predict_cnn_pair(model: object, rgb: object, reference: object):
    from ...perception.materializer import _predict_cnn_pair as predict

    return predict(model, rgb, reference)


def materialize_extension_predictions(
    manifest: Mapping[str, object], output_dir: object
) -> dict[str, object]:
    """Apply the frozen RGB-only predictor to exactly nine V4 stage images."""

    from pathlib import Path
    import numpy as np
    from PIL import Image

    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("prediction_output_not_fresh")
    predictor_path = (
        Path(__file__).resolve().parents[4]
        / "src" / "fluidmix" / "perception" / "cnn" / "cnn_predictor.py"
    )
    if _hash_file(
        str(predictor_path), CNN_PREDICTOR_SOURCE_SHA256, "predictor_source"
    )["sha256"] != CNN_PREDICTOR_SOURCE_SHA256:
        raise AssertionError("predictor verifier returned an inconsistent digest")
    model, producer, _, _ = _load_frozen_cnn(
        checkpoint_path=CNN_CHECKPOINT_PATH,
        replay_path=CNN_REPLAY_PATH,
        review_path=CNN_REVIEW_PATH,
    )
    if producer != PREDICTOR:
        raise _fail("predictor_lineage_invalid")
    vials = manifest.get("vials")
    if not isinstance(vials, list) or len(vials) != 3:
        raise _fail("vial_denominator_invalid")

    prepared: list[dict[str, object]] = []
    for vial in vials:
        if not isinstance(vial, Mapping):
            raise _fail("vial_fields_invalid")
        reference = _load_rgb(
            vial.get("prefill_rgb_path"), vial.get("prefill_rgb_sha256"), "prefill_rgb"
        )
        stages = vial.get("stages")
        if not isinstance(stages, list) or len(stages) != 3:
            raise _fail("stage_denominator_invalid")
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise _fail("stage_fields_invalid")
            rgb = _load_rgb(stage.get("rgb_path"), stage.get("rgb_sha256"), "stage_rgb")
            prediction = _predict_cnn_pair(model, rgb, reference)
            if (
                prediction.status != "visible_measured"
                or prediction.surface_row is None
                or prediction.visible_height_px is None
                or prediction.changed_pixel_count <= 0
                or not bool(np.any(prediction.mask))
            ):
                raise _fail(
                    f"frozen_predictor_gate_failed:{stage.get('image_id')}:{prediction.status}"
                )
            prepared.append(
                {"vial": vial, "stage": stage, "prediction": prediction}
            )
    if len(prepared) != 9:
        raise _fail("stage_denominator_invalid")

    output.mkdir(parents=True, exist_ok=False)
    masks = output / "masks"
    masks.mkdir()
    images: dict[str, dict[str, str]] = {}
    records: list[dict[str, object]] = []
    for item in prepared:
        vial = item["vial"]
        stage = item["stage"]
        prediction = item["prediction"]
        assert isinstance(vial, Mapping) and isinstance(stage, Mapping)
        image_id = str(stage["image_id"])
        mask_path = (masks / f"{image_id}.png").absolute()
        Image.fromarray(prediction.mask.astype(np.uint8) * 255, mode="L").save(
            mask_path, format="PNG"
        )
        mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        images[image_id] = {
            "rgb_path": str(stage["rgb_path"]),
            "rgb_sha256": str(stage["rgb_sha256"]),
            "mask_path": str(mask_path),
            "mask_sha256": mask_sha,
        }
        records.append(
            {
                "image_id": image_id,
                "run_id": manifest["run_id"],
                "batch_id": manifest["batch_id"],
                "vial_id": vial["vial_id"],
                "vial_ordinal": vial["vial_ordinal"],
                "case_id": vial["case_id"],
                "split_role": manifest["split_role"],
                "split_version": manifest["split_version"],
                "stage_id": stage["stage_id"],
                "stage_ordinal": stage["stage_ordinal"],
                "tank_id": stage["tank_id"],
                "rgb_path": stage["rgb_path"],
                "rgb_sha256": stage["rgb_sha256"],
                "reference_rgb_path": vial["prefill_rgb_path"],
                "reference_rgb_sha256": vial["prefill_rgb_sha256"],
                "mask_path": str(mask_path),
                "mask_sha256": mask_sha,
                "prediction_status": prediction.status,
                "surface_row": prediction.surface_row,
                "visible_height_px": prediction.visible_height_px,
                "changed_pixel_count": prediction.changed_pixel_count,
            }
        )
    locators = {
        "schema_version": 2,
        "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
        "producer": deepcopy(PREDICTOR),
        "images": images,
    }
    result = {
        "schema_version": 1,
        "artifact_type": "wp5_sixth_development_run_prediction_receipt",
        "status": "materialized",
        "predictor": deepcopy(PREDICTOR),
        "inference_inputs": {
            "input_fields": ["rgb"],
            "current_plus_same_run_prefill_rgb_channels": 6,
            "same_run_prefill_references": 3,
            "recipe_ml_inputs": 0,
            "oracle_masks_opened": 0,
            "semantic_masks_opened": 0,
            "label_inputs": 0,
            "video_inputs": 0,
        },
        "denominators": {"runs": 1, "vials": 3, "stages": 9},
        "records": records,
    }
    _write_new(output / "prediction-locators.json", _canonical_bytes(locators))
    _write_new(output / "prediction-receipt.json", _canonical_bytes(result))
    return result


def _verify_code_revision(code_commit: str) -> dict[str, object]:
    from pathlib import Path

    if not isinstance(code_commit, str) or re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise _fail("code_commit_invalid")
    root = Path(__file__).resolve().parents[4]
    tracked = (
        "src/fluidmix/memory/history/sixth_development_run.py",
        "src/fluidmix/pipeline/sixth_development_run.py",
        "src/fluidmix/storage/sqlite/evidence_store.py",
        "src/fluidmix/perception/materializer.py",
        "src/fluidmix/perception/cnn/cnn_predictor.py",
        "cli/fluidmix/memory/history/materialize_sixth_development_run.py",
    )
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--", *tracked],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise _fail("code_revision_unverifiable") from error
    if head != code_commit or dirty:
        raise _fail("code_revision_stale_or_dirty")
    files = {}
    for relative in tracked:
        path = root / relative
        binding = _hash_file(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), "code_source")
        try:
            head_bytes = subprocess.run(
                ["git", "-C", str(root), "show", f"HEAD:{relative}"], check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise _fail("code_revision_unverifiable") from error
        if hashlib.sha256(head_bytes).hexdigest() != binding["sha256"]:
            raise _fail("code_revision_blob_mismatch")
        files[relative] = binding
    return {"code_commit": head, "tracked_files": files, "relevant_worktree_clean": True}


def _load_strict_json(path_value: object, expected_value: object, role: str) -> dict[str, object]:
    from pathlib import Path

    try:
        path_string = os.fspath(path_value)
    except TypeError as error:
        raise _fail(f"{role}_path_invalid") from error
    binding = _hash_file(path_string, expected_value, role)
    try:
        value = json.loads(Path(binding["path"]).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"{role}_json_invalid") from error
    if not isinstance(value, dict):
        raise _fail(f"{role}_json_invalid")
    return value


def _delta(left: list[object], right: list[object]) -> float:
    value = math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))
    )
    if not math.isfinite(value):
        raise _fail("measurement_delta_nonfinite")
    return value


def _export_store_records(database: object) -> list[dict[str, object]]:
    from pathlib import Path

    query = """
        SELECT s.evidence_id, s.run_id, v.case_id, v.vial_ordinal,
               v.block_recipe_id, v.block_position, v.recipe_id,
               v.previous_recipe_id, v.recipe_json, v.dispense_order_json,
               s.split_role, r.split_version, s.stage_id, s.tank_id,
               s.stage_ordinal, clip.path, f.frame_range_json,
               f.transition_ranges_json, clip.sha256, i.image_id, rgb.sha256,
               mask.sha256, i.mask_source, m.measurement_id,
               m.median_lab_json, m.mad_lab_json, m.iqr_lab_json,
               m.pairwise_delta_e76_json
        FROM stages AS s
        JOIN runs AS r ON r.run_id = s.run_id
        JOIN vials AS v ON v.vial_id = s.vial_id
        JOIN fills AS f ON f.vial_id = v.vial_id
        JOIN clips AS c ON c.fill_id = f.fill_id
        JOIN artifacts AS clip ON clip.artifact_id = c.artifact_id
        JOIN images AS i ON i.evidence_id = s.evidence_id
        JOIN artifacts AS rgb ON rgb.artifact_id = i.rgb_artifact_id
        JOIN artifacts AS mask ON mask.artifact_id = i.mask_artifact_id
        JOIN measurements AS m ON m.evidence_id = s.evidence_id
        ORDER BY s.event_cursor, s.evidence_id
    """
    with sqlite3.connect(Path(database)) as connection:
        rows = connection.execute(query).fetchall()
    result = []
    for row in rows:
        result.append(
            {
                "run_id": row[1], "case_id": row[2], "vial_ordinal": row[3],
                "block_recipe_id": row[4], "block_position": row[5],
                "recipe_id": row[6], "previous_recipe_id": row[7],
                "recipe_ml": json.loads(row[8]), "dispense_order": json.loads(row[9]),
                "split_role": row[10], "split_version": row[11],
                "stage_id": row[12], "tank_id": row[13], "stage_ordinal": row[14],
                "fill_clip_path": row[15], "fill_frame_range": json.loads(row[16]),
                "transition_frame_ranges": json.loads(row[17]), "clip_sha256": row[18],
                "image_id": row[19], "rgb_sha256": row[20], "mask_sha256": row[21],
                "mask_source": row[22], "measurement_id": row[23],
                "median_lab": json.loads(row[24]), "mad_lab": json.loads(row[25]),
                "iqr_lab": json.loads(row[26]), "pairwise_delta_e76": json.loads(row[27]),
                "colour_regime_id": None, "evidence_id": row[0], "decision_id": None,
                "disposition": None, "limit_codes": [], "history_refs": [],
                "explanation_id": None, "model_revision": None, "prompt_sha256": None,
                "image_ids_sent": [], "guard_result": None,
            }
        )
    return result


def _build_six_run_development_store(
    *,
    output_dir: object,
    owner_thread_id: str,
    code_commit: str,
    evidence_store_factory: Callable[..., object],
    canonical_source_record_sha256_fn: Callable[[object, object], str],
) -> dict[str, object]:
    """Publish a fresh six-run store by appending one accepted V4 train run."""

    from pathlib import Path
    from ...perception.masks import build_trusted_predicted_mask_registry_v1
    from ...measurement.vision_measurement import measure_trusted_predicted_image

    if not callable(evidence_store_factory) or not callable(
        canonical_source_record_sha256_fn
    ):
        raise TypeError("sixth-run storage capabilities must be callable")
    if not isinstance(owner_thread_id, str) or _OWNER.fullmatch(owner_thread_id) is None:
        raise _fail("owner_thread_id_invalid")
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output_dir_not_fresh")
    code = _verify_code_revision(code_commit)
    _hash_file(BASE_STORE_PATH, BASE_STORE_SHA256, "base_store")
    base_receipt = _load_strict_json(BASE_RECEIPT_PATH, BASE_RECEIPT_SHA256, "base_receipt")
    base_locators = _load_strict_json(BASE_LOCATORS_PATH, BASE_LOCATORS_SHA256, "base_locators")
    manifest = validate_sixth_run_manifest(
        default_source_manifest(),
        expected_lineage=DEFAULT_LINEAGE,
        existing_run_ids=EXISTING_RUN_IDS,
    )
    source_reads = verify_source_files(manifest)

    prediction_receipt = materialize_extension_predictions(manifest, output / "predictions")
    _write_new(output / "source-manifest.json", _canonical_bytes(manifest))
    extension_locators = _load_strict_json(
        output / "predictions" / "prediction-locators.json",
        hashlib.sha256((output / "predictions" / "prediction-locators.json").read_bytes()).hexdigest(),
        "extension_locators",
    )
    base_images = base_locators.get("images")
    extension_images = extension_locators.get("images")
    if not isinstance(base_images, dict) or len(base_images) != 36:
        raise _fail("base_locator_denominator_invalid")
    if not isinstance(extension_images, dict) or len(extension_images) != 9:
        raise _fail("extension_locator_denominator_invalid")
    if set(base_images).intersection(extension_images):
        raise _fail("mixed_lineage_image_collision")
    combined_locators = deepcopy(base_locators)
    combined_locators["images"] = {**base_images, **extension_images}
    combined_path = output / "prediction-locators.json"
    _write_new(combined_path, _canonical_bytes(combined_locators))
    registry = build_trusted_predicted_mask_registry_v1(combined_locators)

    by_image = {str(row["image_id"]): row for row in prediction_receipt["records"]}
    measured: dict[str, dict[str, object]] = {}
    binding_receipts = []
    for image_id in sorted(extension_images):
        evidence_id = f"EVID-{image_id}"
        measurement_id = f"MEAS-{image_id}"
        result, binding = measure_trusted_predicted_image(
            registry.entry(image_id), measurement_id=measurement_id,
            evidence_id=evidence_id,
        )
        if result.get("validity") != "valid":
            raise _fail(f"trusted_measurement_invalid:{image_id}")
        measured[image_id] = {
            "evidence_id": evidence_id, "measurement_id": measurement_id,
            "median_lab": result["median_lab"], "mad_lab": result["mad_lab"],
            "iqr_lab": result["iqr_lab"],
        }
        binding_receipts.append(binding.receipt())
    pairwise: dict[str, list[dict[str, object]]] = {}
    for image_id, prediction in by_image.items():
        peers = [
            other_id for other_id, other in by_image.items()
            if other_id != image_id and other["stage_id"] == prediction["stage_id"]
        ]
        pairwise[measured[image_id]["evidence_id"]] = sorted(
            (
                {
                    "other_evidence_id": measured[peer]["evidence_id"],
                    "delta_e76": _delta(
                        measured[image_id]["median_lab"], measured[peer]["median_lab"]
                    ),
                }
                for peer in peers
            ),
            key=lambda row: str(row["other_evidence_id"]),
        )

    recipe = manifest["recipe_context"]
    records = []
    locators = {}
    source_hashes = {}
    cursors = {}
    ordered_predictions = sorted(
        prediction_receipt["records"],
        key=lambda row: (int(row["vial_ordinal"]), int(row["stage_ordinal"])),
    )
    vial_by_id = {str(vial["vial_id"]): vial for vial in manifest["vials"]}
    for offset, prediction in enumerate(ordered_predictions):
        image_id = str(prediction["image_id"])
        measurement = measured[image_id]
        vial = vial_by_id[str(prediction["vial_id"])]
        clip_relative = Path(str(vial["fill_clip_path"])).relative_to(Path(ARTIFACT_ROOT)).as_posix()
        record = {
            "run_id": manifest["run_id"], "case_id": prediction["case_id"],
            "vial_ordinal": prediction["vial_ordinal"],
            "block_recipe_id": manifest["batch_id"],
            "block_position": prediction["vial_ordinal"],
            "recipe_id": recipe["recipe_id"],
            "previous_recipe_id": vial["previous_recipe_id"],
            "recipe_ml": recipe["recipe_ml"], "dispense_order": recipe["dispense_order"],
            "split_role": manifest["split_role"], "split_version": manifest["split_version"],
            "stage_id": prediction["stage_id"], "tank_id": prediction["tank_id"],
            "stage_ordinal": prediction["stage_ordinal"], "fill_clip_path": clip_relative,
            "fill_frame_range": [0, 337],
            "transition_frame_ranges": {"entry": None, "exit": None},
            "clip_sha256": vial["fill_clip_sha256"], "image_id": image_id,
            "rgb_sha256": prediction["rgb_sha256"], "mask_sha256": prediction["mask_sha256"],
            "mask_source": "predicted", "measurement_id": measurement["measurement_id"],
            "median_lab": measurement["median_lab"], "mad_lab": measurement["mad_lab"],
            "iqr_lab": measurement["iqr_lab"],
            "pairwise_delta_e76": pairwise[measurement["evidence_id"]],
            "colour_regime_id": None, "evidence_id": measurement["evidence_id"],
            "decision_id": None, "disposition": None, "limit_codes": [], "history_refs": [],
            "explanation_id": None, "model_revision": None, "prompt_sha256": None,
            "image_ids_sent": [], "guard_result": None,
        }
        locator = extension_images[image_id]
        records.append(record)
        locators[image_id] = locator
        cursors[record["evidence_id"]] = 36 + offset
        normalized_locator = {
            "rgb_path": Path(str(locator["rgb_path"])).relative_to(ARTIFACT_ROOT).as_posix(),
            "rgb_sha256": locator["rgb_sha256"],
            "mask_path": Path(str(locator["mask_path"])).relative_to(ARTIFACT_ROOT).as_posix(),
            "mask_sha256": locator["mask_sha256"],
        }
        source_hashes[record["evidence_id"]] = canonical_source_record_sha256_fn(
            record, normalized_locator
        )
    measured_artifact = {
        "schema_version": 1,
        "artifact_type": "wp5_sixth_development_run_measured_evidence",
        "status": "measured",
        "mask_source": "rgb_only_prediction",
        "prediction_producer": deepcopy(PREDICTOR),
        "trusted_predicted_mask_registry_sha256": registry.registry_sha256,
        "denominators": {"runs": 1, "vials": 3, "stage_measurements": 9},
        "records": records,
        "trusted_measurement_bindings": binding_receipts,
    }
    _write_new(output / "measured-evidence.json", _canonical_bytes(measured_artifact))
    _write_new(output / "history-ready-records.json", _canonical_bytes({
        "schema_version": 1,
        "artifact_type": "wp5_history_ready_measured_records",
        "denominators": {"runs": 1, "vials": 3, "stages": 9},
        "records": records,
    }))

    database = output / "evidence.sqlite3"
    shutil.copyfile(BASE_STORE_PATH, database)
    store = evidence_store_factory(database, artifact_root=ARTIFACT_ROOT)
    ingested = store.ingest(records, locators, phase="measured", event_cursors=cursors)
    if (ingested.inserted, ingested.existing) != (9, 0):
        raise _fail("extension_ingest_denominator_invalid")
    history_receipts = []
    for record in records:
        query = store.vial_history(
            str(record["run_id"]), before=cursors[str(record["evidence_id"])],
            allowed_roles={"train"}, target_evidence_id=str(record["evidence_id"]),
        )
        history_receipts.append(
            {
                "target_evidence_id": record["evidence_id"],
                "query_sha256": query.query_sha256,
                "query_kind": query.query_kind,
                "filters": query.filters,
                "member_evidence_ids": [member.evidence_id for member in query.members],
            }
        )
    _write_new(output / "history-query-receipts.json", _canonical_bytes({
        "schema_version": 1,
        "artifact_type": "wp5_history_query_receipt_collection",
        "denominators": {"queries": 9, "target_runs": 1},
        "receipts": history_receipts,
    }))
    exported = _export_store_records(database)
    if len(exported) != 45:
        raise _fail("exported_store_denominator_invalid")
    _write_new(output / "production-evidence-records.json", _canonical_bytes({
        "schema_version": 1,
        "artifact_type": "wp5_production_evidence_record_export",
        "denominators": {"runs": 6, "vials": 18, "stages": 45},
        "records": exported,
    }))

    with sqlite3.connect(database) as connection:
        denominators = {
            "production_runs": connection.execute("SELECT count(*) FROM runs").fetchone()[0],
            "production_vials": connection.execute("SELECT count(*) FROM vials").fetchone()[0],
            "production_stage_records": connection.execute("SELECT count(*) FROM stages").fetchone()[0],
            "predicted_mask_images": connection.execute("SELECT count(*) FROM images").fetchone()[0],
            "rgb_measurements": connection.execute("SELECT count(*) FROM measurements").fetchone()[0],
            "fill_clips": connection.execute("SELECT count(*) FROM clips").fetchone()[0],
            "history_queries": connection.execute("SELECT count(*) FROM history_queries").fetchone()[0],
        }
        run_roles = dict(connection.execute("SELECT split_role, count(*) FROM runs GROUP BY split_role"))
    if denominators != {
        "production_runs": 6, "production_vials": 18, "production_stage_records": 45,
        "predicted_mask_images": 45, "rgb_measurements": 45, "fill_clips": 18,
        "history_queries": 9,
    } or run_roles != {"train": 5, "validation": 1}:
        raise _fail("six_run_store_gate_failed")
    if _hash_file(BASE_STORE_PATH, BASE_STORE_SHA256, "base_store_postrun")["sha256"] != BASE_STORE_SHA256:
        raise _fail("base_store_changed")
    if _hash_file(BASE_RECEIPT_PATH, BASE_RECEIPT_SHA256, "base_receipt_postrun")["sha256"] != BASE_RECEIPT_SHA256:
        raise _fail("base_receipt_changed")

    locator_sha = hashlib.sha256(combined_path.read_bytes()).hexdigest()
    database_sha = hashlib.sha256(database.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 2,
        "artifact_type": "f2_predicted_evidence_store_population_receipt",
        "status": "populated",
        "artifact_owner": {"owner_thread_id": owner_thread_id},
        "producer": {"producer_id": "fluidmix.memory.history.sixth_development_run", **code},
        "base_tuple": {
            "database_path": BASE_STORE_PATH, "database_sha256": BASE_STORE_SHA256,
            "receipt_path": BASE_RECEIPT_PATH, "receipt_sha256": BASE_RECEIPT_SHA256,
            "preserved_read_only": True,
        },
        "prediction_producer": deepcopy(PREDICTOR),
        "trusted_predicted_mask_registry_sha256": registry.registry_sha256,
        "combined_prediction_locators": {"path": str(combined_path), "sha256": locator_sha},
        "source_manifest_sha256": hashlib.sha256((output / "source-manifest.json").read_bytes()).hexdigest(),
        "source_read_ledger": source_reads,
        "split_roles": run_roles,
        "denominators": {
            **denominators,
            "train_stage_records": 39,
            "validation_stage_records": 6,
            "excluded_evaluation_stage_records": 6,
            "accounted_stage_records": 51,
        },
        "extension_source_record_sha256_by_evidence": source_hashes,
        "database": {
            "path": "evidence.sqlite3", "sha256": database_sha,
            "bytes": database.stat().st_size, "sqlite_user_version": 3,
        },
        "claim_boundary": (
            "Synthetic development evidence only. The extension applies the unchanged "
            "accepted RGB-only predictor to one accepted V4 train run; it opens no "
            "semantic/oracle masks, labels, evaluation/test inputs, or video-5 assets."
        ),
        "base_receipt_identity": {
            "artifact_type": base_receipt.get("artifact_type"),
            "status": base_receipt.get("status"),
        },
    }
    _write_new(output / "population-receipt.json", _canonical_bytes(receipt))
    return receipt


def validate_sixth_run_manifest(
    value: object,
    *,
    expected_lineage: Mapping[str, object],
    existing_run_ids: Set[str],
) -> dict[str, object]:
    """Validate metadata before any source file or image is opened."""

    artifact = _exact(value, _TOP_FIELDS, "source_manifest")
    if (
        artifact.get("schema_version") != 2
        or artifact.get("artifact_type") != "wp5_sixth_development_run_source"
        or artifact.get("status") != "accepted"
        or artifact.get("data_origin") != "twin"
        or artifact.get("split_role") != "train"
        or artifact.get("split_version") != "whole-run-varied-v4"
    ):
        raise _fail("source_manifest_identity_invalid")
    if artifact.get("predictor") != PREDICTOR:
        raise _fail("predictor_lineage_invalid")
    run_id = artifact.get("run_id")
    batch_id = artifact.get("batch_id")
    if run_id != expected_lineage.get("run_id") or batch_id != expected_lineage.get("batch_id"):
        raise _fail("run_lineage_invalid")
    if not isinstance(run_id, str) or run_id in existing_run_ids:
        raise _fail("run_id_not_disjoint")
    _absolute(artifact.get("source_plan_path"), "source_plan")
    plan_sha = _sha(artifact.get("source_plan_sha256"), "source_plan")
    if plan_sha != expected_lineage.get("source_plan_sha256"):
        raise _fail("source_plan_lineage_invalid")

    recipe = _exact(artifact.get("recipe_context"), _RECIPE_FIELDS, "recipe_context")
    if (
        recipe.get("recipe_id") != "5"
        or recipe.get("recipe_ml") != {"t100": 10.0, "t200": 10.0, "t400": 10.0}
        or recipe.get("dispense_order") != ["t200", "t400", "t100"]
    ):
        raise _fail("recipe_context_invalid")

    raw_vials = artifact.get("vials")
    expected_vials = expected_lineage.get("vials")
    if not isinstance(raw_vials, list) or len(raw_vials) != 3 or not isinstance(expected_vials, Mapping):
        raise _fail("vial_denominator_invalid")
    seen_images: set[str] = set()
    checked_vials: list[dict[str, object]] = []
    for ordinal, raw_vial in enumerate(raw_vials, 1):
        vial = _exact(raw_vial, _VIAL_FIELDS, "vial")
        vial_id = f"V{ordinal}"
        expected = expected_vials.get(vial_id)
        if not isinstance(expected, Mapping):
            raise _fail("vial_lineage_invalid")
        if (
            vial.get("vial_id") != vial_id
            or vial.get("vial_ordinal") != ordinal
            or vial.get("case_id") != expected.get("case_id")
            or vial.get("previous_recipe_id") != expected.get("previous_recipe_id")
            or vial.get("seed") != expected.get("seed")
            or vial.get("source_receipt_sha256") != expected.get("source_receipt_sha256")
            or vial.get("validation_receipt_sha256")
            != expected.get("validation_receipt_sha256")
        ):
            raise _fail("vial_lineage_invalid")
        for path_field in (
            "source_receipt_path", "validation_receipt_path", "prefill_rgb_path",
            "fill_clip_path",
        ):
            _absolute(vial.get(path_field), path_field.removesuffix("_path"))
        for sha_field in (
            "source_receipt_sha256", "validation_receipt_sha256",
            "prefill_rgb_sha256", "fill_clip_sha256",
        ):
            _sha(vial.get(sha_field), sha_field.removesuffix("_sha256"))
        stages = vial.get("stages")
        if not isinstance(stages, list) or len(stages) != 3:
            raise _fail("stage_denominator_invalid")
        checked_stages: list[dict[str, object]] = []
        for stage_ordinal, (raw_stage, tank_id) in enumerate(
            zip(stages, ("t200", "t400", "t100"), strict=True), 1
        ):
            stage = _exact(raw_stage, _STAGE_FIELDS, "stage")
            expected_stage_id = f"dispense-{stage_ordinal}-{tank_id}"
            expected_image_id = f"{batch_id}-{vial_id}-{expected_stage_id}"
            if (
                stage.get("stage_id") != expected_stage_id
                or stage.get("stage_ordinal") != stage_ordinal
                or stage.get("tank_id") != tank_id
                or stage.get("image_id") != expected_image_id
                or expected_image_id in seen_images
            ):
                raise _fail("stage_identity_invalid")
            seen_images.add(expected_image_id)
            _absolute(stage.get("rgb_path"), "stage_rgb")
            _sha(stage.get("rgb_sha256"), "stage_rgb")
            checked_stages.append(dict(stage))
        copied_vial = dict(vial)
        copied_vial["stages"] = checked_stages
        checked_vials.append(copied_vial)
    result = deepcopy(dict(artifact))
    result["vials"] = checked_vials
    return result


__all__ = [
    "DEFAULT_LINEAGE",
    "BASE_RECEIPT_SHA256",
    "BASE_STORE_SHA256",
    "PREDICTOR",
    "SixthDevelopmentRunError",
    "default_source_manifest",
    "materialize_extension_predictions",
    "validate_sixth_run_manifest",
    "verify_source_files",
]

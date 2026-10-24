from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def frozen_payload():
    return json.loads((ROOT / "data/twin_render_recovery_v1.json").read_text())


def write_payload(tmp_path, payload):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_recovery_manifest_is_exactly_one_fixed_fillcam_smoke():
    from fluidmix.render_manifest import load_render_manifest

    manifest = load_render_manifest(ROOT / "data/twin_render_recovery_v1.json")

    assert manifest.schema_version == 2
    assert manifest.attempt_id == "fillcam-recovery-s00-20261024-a1"
    assert manifest.camera == "FillCam"
    assert manifest.renderer == "PathTracing"
    assert manifest.resolution == (640, 480)
    assert manifest.subframes == 32
    assert manifest.timeout_seconds == 1200
    assert manifest.kit_args == (
        "--/telemetry/enableAnonymousData=false",
        "--/structuredLog/enable=false",
        "--/exts/isaacsim.core.throttling/enable_async=false",
    )
    assert manifest.smoke.variant_id == "S00"
    assert manifest.smoke.frame == 686
    assert not hasattr(manifest, "batch")


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p.update(schema_version=True), "schema_version"),
        (lambda p: p.update(attempt_id="attempt-a2"), "attempt_id"),
        (lambda p: p.update(camera="tank_level"), "camera"),
        (lambda p: p.update(renderer="RayTracedLighting"), "renderer"),
        (lambda p: p.update(resolution=[641, 480]), "resolution"),
        (lambda p: p.update(subframes=0), "subframes"),
        (lambda p: p.update(timeout_seconds=1201), "timeout_seconds"),
        (lambda p: p["kit_args"].append("--/unreviewed=true"), "kit_args"),
        (
            lambda p: p["kit_args"].append(
                "--/exts/omni.replicator.core/maxAssetLoadingTime=1000"
            ),
            "kit_args",
        ),
        (lambda p: p["smoke"].update(id="M00"), "smoke.id"),
        (lambda p: p["smoke"].update(frame=687), "smoke.frame"),
        (lambda p: p["smoke"].update(extra="no"), "smoke fields"),
        (lambda p: p.update(batch=[]), "top-level fields"),
    ],
)
def test_recovery_manifest_rejects_every_contract_drift(tmp_path, change, match):
    from fluidmix.render_manifest import load_render_manifest

    payload = frozen_payload()
    change(payload)

    with pytest.raises(ValueError, match=match):
        load_render_manifest(write_payload(tmp_path, payload))


@pytest.mark.parametrize("payload", [[], None, "manifest"])
def test_recovery_manifest_rejects_non_object_roots(tmp_path, payload):
    from fluidmix.render_manifest import load_render_manifest

    with pytest.raises(ValueError, match="root must be an object"):
        load_render_manifest(write_payload(tmp_path, payload))


def test_recovery_manifest_rejects_malformed_json_and_symlink(tmp_path):
    from fluidmix.render_manifest import load_render_manifest

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    with pytest.raises(ValueError, match="valid JSON"):
        load_render_manifest(malformed)

    target = write_payload(tmp_path, frozen_payload())
    link = tmp_path / "linked.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_render_manifest(link)


def test_recovery_manifest_contains_only_reviewed_setting_intervention():
    payload = deepcopy(frozen_payload())

    assert payload["kit_args"][-1] == (
        "--/exts/isaacsim.core.throttling/enable_async=false"
    )
    assert sum("enable_async" in item for item in payload["kit_args"]) == 1
    assert not any("maxAssetLoadingTime" in item for item in payload["kit_args"])

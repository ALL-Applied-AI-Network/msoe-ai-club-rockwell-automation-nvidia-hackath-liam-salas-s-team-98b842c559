from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def frozen_payload():
    return json.loads((ROOT / "data/twin_render_feasibility_v1.json").read_text())


def test_frozen_render_manifest_has_one_smoke_and_eight_neutral_batch_jobs():
    from fluidmix.render_manifest import load_render_manifest

    manifest = load_render_manifest(ROOT / "data/twin_render_feasibility_v1.json")

    assert manifest.schema_version == 1
    assert manifest.camera == "FillCam"
    assert manifest.subframes == 32
    assert manifest.timeout_seconds == 1200
    assert manifest.kit_args == (
        "--/telemetry/enableAnonymousData=false",
        "--/structuredLog/enable=false",
    )
    assert [job.variant_id for job in manifest.smoke] == ["S00"]
    assert [job.variant_id for job in manifest.batch] == [
        "M00", "M01", "M02", "M03", "M04", "M05", "F00", "F01"
    ]
    assert [job.frame for job in manifest.batch[-2:]] == [400, 500]
    assert manifest.batch[-2].material is None
    assert manifest.batch[-1].material is None


def test_frozen_interpolation_values_are_independently_recoverable():
    from fluidmix.render_manifest import load_render_manifest

    manifest = load_render_manifest(ROOT / "data/twin_render_feasibility_v1.json")
    jobs = {job.variant_id: job for job in manifest.batch}
    endpoint_a = (0.006, 0.2, 0.3)
    endpoint_b = (0.45, 0.12, 0.01)
    for variant_id, alpha in (("M02", 0.125), ("M03", 0.25), ("M04", 0.5)):
        expected = tuple((1 - alpha) * a + alpha * b for a, b in zip(endpoint_a, endpoint_b))
        assert jobs[variant_id].material.color == pytest.approx(expected)
        assert jobs[variant_id].material.opacity == 0.6


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda p: p.update(camera="tank_level"), "camera"),
        (lambda p: p.update(subframes=0), "subframes"),
        (lambda p: p.update(timeout_seconds=0), "timeout"),
        (lambda p: p["smoke"].append(deepcopy(p["smoke"][0])), "smoke"),
        (lambda p: p["batch"].append(deepcopy(p["batch"][0])), "eight"),
        (lambda p: p["batch"][0].update(id="S00"), "unique"),
        (lambda p: p["batch"][0].update(frame=687), "frame"),
        (lambda p: p["batch"][0].pop("opacity"), "material"),
    ],
)
def test_render_manifest_rejects_contract_drift(tmp_path, change, match):
    from fluidmix.render_manifest import load_render_manifest

    payload = frozen_payload()
    change(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        load_render_manifest(path)


def test_render_manifest_rejects_unreviewed_kit_args(tmp_path):
    from fluidmix.render_manifest import load_render_manifest

    payload = frozen_payload()
    payload["kit_args"].append("--/telemetry/mode=dev")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="kit args"):
        load_render_manifest(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(batch=[]),
        lambda p: p["smoke"][0].update(material_color=[0.1, 0.2, 0.3], opacity=0.6),
        lambda p: p["batch"][1].update(material_color=[0.44, 0.12, 0.01]),
        lambda p: p["batch"][7].update(frame=501),
    ],
)
def test_render_manifest_rejects_any_frozen_matrix_change(tmp_path, change):
    from fluidmix.render_manifest import load_render_manifest

    payload = frozen_payload()
    change(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="frozen render matrix"):
        load_render_manifest(path)

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.physx.isosurface_mesh_proxy import (
    PROXY_RGB_PARITY_THRESHOLDS,
    build_liquid_proxy_intent,
    canonical_mesh_snapshot,
    copy_exact_mesh_proxy,
    evaluate_proxy_rgb_parity_for_source_hide,
    refresh_exact_mesh_proxy,
)


POINTS = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
COUNTS = (3, 3, 3, 3)
INDICES = (0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3)
IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FakeMeshAdapter:
    def __init__(self):
        self.meshes = {
            "/FillStation/__fluidmix__/ParticleSystem/Isosurface": {
                "schema_type": "Mesh",
                "points": POINTS,
                "face_vertex_counts": COUNTS,
                "face_vertex_indices": INDICES,
                "local_transform": IDENTITY,
            }
        }
        self.materials = {}
        self.semantics = {}
        self.visibility_writes = []

    def read_mesh(self, path):
        return dict(self.meshes[path])

    def prim_exists(self, path):
        return path in self.meshes

    def define_standard_mesh(self, path):
        self.meshes[path] = {"schema_type": "Mesh"}

    def write_mesh(self, path, snapshot):
        self.meshes[path].update(
            {
                "points": snapshot.points,
                "face_vertex_counts": snapshot.face_vertex_counts,
                "face_vertex_indices": snapshot.face_vertex_indices,
                "local_transform": snapshot.local_transform,
            }
        )

    def bind_material(self, path, material_path):
        self.materials[path] = material_path

    def bind_semantics(self, path, semantic_type, labels):
        self.semantics[(path, semantic_type)] = tuple(labels)

    def read_material_binding(self, path):
        return self.materials.get(path)

    def read_semantics(self, path, semantic_type):
        return self.semantics.get((path, semantic_type))


def test_canonical_mesh_snapshot_accepts_one_closed_tetrahedron():
    snapshot = canonical_mesh_snapshot(
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        face_vertex_counts=(3, 3, 3, 3),
        face_vertex_indices=(0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3),
        local_transform=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )

    assert snapshot.point_count == 4
    assert snapshot.face_count == 4
    assert snapshot.corner_count == 12
    assert len(snapshot.points_sha256) == 64
    assert len(snapshot.topology_sha256) == 64
    assert len(snapshot.local_transform_sha256) == 64
    assert len(snapshot.mesh_sha256) == 64


def test_mesh_snapshot_hashes_are_frozen_and_domain_separated():
    snapshot = canonical_mesh_snapshot(POINTS, COUNTS, INDICES, IDENTITY)

    assert snapshot.points_sha256 == "3c38b835cc86ed93ab5e571de76876b5f86c7351651ec8e79f8d80d535e3652b"
    assert snapshot.face_vertex_counts_sha256 == "0ad40b10b3e4a4fe4d221fc8936d6e681314894bf9ba8d8066acc6b3f09c64fb"
    assert snapshot.face_vertex_indices_sha256 == "6fbfed4d2d05b728412153af0fb0c88a2bdd2cf94ea6d9cd50049b1abb5252ae"
    assert snapshot.topology_sha256 == "9a15342dae17dbb72d2ff84acabb7dc4472c745db82e8b8a9c2e522e8a4f1f75"
    assert snapshot.local_transform_sha256 == "228a35f6bfa1f40957a0f5be5b9965382bcc569cc5abe22d61f6705d32bb3cf7"
    assert snapshot.mesh_sha256 == "514cfc067dce3dc0d7b7a779f01aa1fec368df9951a40c62d0ed206627df7cb9"


@pytest.mark.parametrize(
    ("points", "counts", "indices", "transform", "error"),
    [
        ((), COUNTS, INDICES, IDENTITY, ValueError),
        (((0.0, 0.0),), (3,), (0, 0, 0), IDENTITY, ValueError),
        (((False, 0.0, 0.0),), (3,), (0, 0, 0), IDENTITY, TypeError),
        (((math.nan, 0.0, 0.0),), (3,), (0, 0, 0), IDENTITY, ValueError),
        (((1.0e40, 0.0, 0.0),), (3,), (0, 0, 0), IDENTITY, ValueError),
        (POINTS, (), (), IDENTITY, ValueError),
        (POINTS, (2,), (0, 1), IDENTITY, ValueError),
        (POINTS, (3,), (0, 1), IDENTITY, ValueError),
        (POINTS, (3,), (0, 1, 4), IDENTITY, ValueError),
        (POINTS, (True,), (0,), IDENTITY, TypeError),
        (POINTS, (3,), (0, 1, False), IDENTITY, TypeError),
        (POINTS, COUNTS, INDICES, (1.0,) * 15, ValueError),
        (POINTS, COUNTS, INDICES, (1.0,) * 15 + (math.inf,), ValueError),
    ],
)
def test_mesh_snapshot_rejects_malformed_nonfinite_or_invalid_topology(
    points, counts, indices, transform, error
):
    with pytest.raises(error):
        canonical_mesh_snapshot(points, counts, indices, transform)


def test_exact_proxy_copy_preserves_geometry_binds_evidence_and_does_not_hide_source():
    adapter = FakeMeshAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )

    evidence = copy_exact_mesh_proxy(adapter, intent)

    assert evidence["status"] == "proxy_authored_pending_rgb_parity"
    assert evidence["exact_geometry_match"] is True
    assert evidence["points_sha256"] == evidence["proxy_points_sha256"]
    assert evidence["topology_sha256"] == evidence["proxy_topology_sha256"]
    assert evidence["local_transform_sha256"] == evidence["proxy_local_transform_sha256"]
    assert evidence["material_binding"] == intent.material_path
    assert evidence["semantic_binding"] == {"type": "class", "labels": ["liquid"]}
    assert evidence["source_visibility"] == "unchanged"
    assert evidence["source_hidden"] is False
    assert evidence["shared_rgb_plan"]["hide_source_allowed"] is False
    assert evidence["shared_rgb_plan"]["parity_gate_status"] == "not_run"
    assert adapter.visibility_writes == []


def test_existing_proxy_refresh_copies_new_dynamic_geometry_exactly_without_rebinding():
    adapter = FakeMeshAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )
    copy_exact_mesh_proxy(adapter, intent)
    adapter.meshes[intent.source_path]["points"] = (
        (0.0, 0.0, 0.0),
        (1.25, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    evidence = refresh_exact_mesh_proxy(adapter, intent)

    assert evidence["status"] == "proxy_refreshed_exact"
    assert evidence["exact_geometry_match"] is True
    assert evidence["points_sha256"] == evidence["proxy_points_sha256"]
    assert evidence["topology_sha256"] == evidence["proxy_topology_sha256"]
    assert evidence["local_transform_sha256"] == evidence["proxy_local_transform_sha256"]
    assert evidence["material_binding"] == intent.material_path
    assert evidence["semantic_binding"] == {"type": "class", "labels": ["liquid"]}
    assert evidence["source_visibility"] == "unchanged"
    assert adapter.visibility_writes == []


def test_proxy_refresh_requires_existing_standard_proxy_and_preserved_bindings():
    adapter = FakeMeshAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )
    with pytest.raises(ValueError, match="must already exist"):
        refresh_exact_mesh_proxy(adapter, intent)

    copy_exact_mesh_proxy(adapter, intent)
    adapter.materials[intent.proxy_path] = "/Wrong/Material"
    with pytest.raises(ValueError, match="material binding"):
        refresh_exact_mesh_proxy(adapter, intent)


@pytest.mark.parametrize(
    "paths",
    [
        {"source_path": "relative", "proxy_path": "/World/Proxy", "material_path": "/World/Mat"},
        {"source_path": "/World/Source", "proxy_path": "/Elsewhere/Proxy", "material_path": "/World/Mat"},
        {"source_path": "/World/Source", "proxy_path": "/World/Source", "material_path": "/World/Mat"},
        {"source_path": "/World/Source", "proxy_path": "/World/Bad-Proxy", "material_path": "/World/Mat"},
    ],
)
def test_liquid_proxy_intent_rejects_unsafe_or_transform_mismatched_paths(paths):
    with pytest.raises((TypeError, ValueError)):
        build_liquid_proxy_intent(**paths)


def test_exact_proxy_copy_refuses_to_replace_an_existing_prim():
    adapter = FakeMeshAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )
    adapter.meshes[intent.proxy_path] = {"schema_type": "Cube"}

    with pytest.raises(ValueError, match="already exists"):
        copy_exact_mesh_proxy(adapter, intent)


@pytest.mark.parametrize(
    "field",
    ["points", "face_vertex_counts", "face_vertex_indices", "local_transform"],
)
def test_exact_proxy_copy_rejects_any_geometry_or_transform_readback_change(field):
    class CorruptingAdapter(FakeMeshAdapter):
        def write_mesh(self, path, snapshot):
            super().write_mesh(path, snapshot)
            if field == "points":
                self.meshes[path][field] = ((0.25, 0.0, 0.0),) + snapshot.points[1:]
            elif field == "face_vertex_counts":
                self.meshes[path][field] = (4, 3, 3, 2)
            elif field == "face_vertex_indices":
                self.meshes[path][field] = (0, 1, 2) + snapshot.face_vertex_indices[3:]
            else:
                self.meshes[path][field] = snapshot.local_transform[:-1] + (2.0,)

    adapter = CorruptingAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )

    with pytest.raises(ValueError):
        copy_exact_mesh_proxy(adapter, intent)


@pytest.mark.parametrize("binding", ["material", "semantic"])
def test_exact_proxy_copy_rejects_binding_readback_change(binding):
    class BindingAdapter(FakeMeshAdapter):
        def read_material_binding(self, path):
            if binding == "material":
                return "/Wrong/Material"
            return super().read_material_binding(path)

        def read_semantics(self, path, semantic_type):
            if binding == "semantic":
                return ("not_liquid",)
            return super().read_semantics(path, semantic_type)

    adapter = BindingAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )

    with pytest.raises(ValueError, match="binding readback"):
        copy_exact_mesh_proxy(adapter, intent)


def test_saved_same_state_rgb_parity_is_required_before_source_hide_is_allowed():
    adapter = FakeMeshAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )
    proxy = copy_exact_mesh_proxy(adapter, intent)
    parity = {
        "same_physics_step": True,
        "same_renderer_settings": True,
        "parity_capture_isolated": True,
        "source_restored_visible": True,
        "original_rgb_sha256": "1" * 64,
        "proxy_rgb_sha256": "2" * 64,
        "absdiff_sha256": "3" * 64,
        "mean_abs_rgb": 1.5,
        "p95_abs_rgb": 7.5,
        "fraction_abs_rgb_gt_16": 0.009,
        "thresholds": dict(PROXY_RGB_PARITY_THRESHOLDS),
    }

    decision = evaluate_proxy_rgb_parity_for_source_hide(proxy, parity)

    assert decision["status"] == "rgb_parity_pass_pending_shared_capture"
    assert decision["hide_source_allowed"] is True
    assert decision["proxy_visibility"] == "visible"
    assert decision["next_capture"] == "one_shared_rgb_semantic_capture"
    assert decision["semantic_mask_observed"] is False


def test_rgb_parity_accepts_the_exact_refreshed_mesh_that_was_rendered():
    adapter = FakeMeshAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )
    copy_exact_mesh_proxy(adapter, intent)
    refreshed = refresh_exact_mesh_proxy(adapter, intent)
    parity = {
        "same_physics_step": True,
        "same_renderer_settings": True,
        "parity_capture_isolated": True,
        "source_restored_visible": True,
        "original_rgb_sha256": "1" * 64,
        "proxy_rgb_sha256": "2" * 64,
        "absdiff_sha256": "3" * 64,
        "mean_abs_rgb": 0.0,
        "p95_abs_rgb": 0.0,
        "fraction_abs_rgb_gt_16": 0.0,
        "thresholds": dict(PROXY_RGB_PARITY_THRESHOLDS),
    }

    decision = evaluate_proxy_rgb_parity_for_source_hide(refreshed, parity)

    assert decision["hide_source_allowed"] is True


@pytest.mark.parametrize(
    ("metric", "value", "reason"),
    [
        ("mean_abs_rgb", 2.0001, "mean_abs_rgb_exceeds_threshold"),
        ("p95_abs_rgb", 8.0001, "p95_abs_rgb_exceeds_threshold"),
        (
            "fraction_abs_rgb_gt_16",
            0.0101,
            "fraction_abs_rgb_gt_16_exceeds_threshold",
        ),
    ],
)
def test_failed_rgb_parity_never_authorizes_hiding_the_source(metric, value, reason):
    adapter = FakeMeshAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )
    proxy = copy_exact_mesh_proxy(adapter, intent)
    parity = {
        "same_physics_step": True,
        "same_renderer_settings": True,
        "parity_capture_isolated": True,
        "source_restored_visible": True,
        "original_rgb_sha256": "1" * 64,
        "proxy_rgb_sha256": "2" * 64,
        "absdiff_sha256": "3" * 64,
        "mean_abs_rgb": 0.0,
        "p95_abs_rgb": 0.0,
        "fraction_abs_rgb_gt_16": 0.0,
        "thresholds": dict(PROXY_RGB_PARITY_THRESHOLDS),
    }
    parity[metric] = value

    decision = evaluate_proxy_rgb_parity_for_source_hide(proxy, parity)

    assert decision["status"] == "rgb_parity_failed_source_must_remain_visible"
    assert decision["hide_source_allowed"] is False
    assert decision["source_visibility_change_executed"] is False
    assert decision["next_capture"] is None
    assert reason in decision["failure_reasons"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("same_physics_step", False),
        ("same_renderer_settings", False),
        ("parity_capture_isolated", False),
        ("source_restored_visible", False),
        ("original_rgb_sha256", None),
        ("proxy_rgb_sha256", "not-a-hash"),
        ("absdiff_sha256", "A" * 64),
        ("mean_abs_rgb", math.nan),
        ("fraction_abs_rgb_gt_16", 1.1),
        ("thresholds", {}),
    ],
)
def test_malformed_or_unmatched_rgb_parity_evidence_fails_closed(key, value):
    adapter = FakeMeshAdapter()
    intent = build_liquid_proxy_intent(
        source_path="/FillStation/__fluidmix__/ParticleSystem/Isosurface",
        proxy_path="/FillStation/__fluidmix__/ParticleSystem/LiquidSemanticProxy",
        material_path="/FillStation/__fluidmix__/GlobalLiquidMaterial",
    )
    proxy = copy_exact_mesh_proxy(adapter, intent)
    parity = {
        "same_physics_step": True,
        "same_renderer_settings": True,
        "parity_capture_isolated": True,
        "source_restored_visible": True,
        "original_rgb_sha256": "1" * 64,
        "proxy_rgb_sha256": "2" * 64,
        "absdiff_sha256": "3" * 64,
        "mean_abs_rgb": 0.0,
        "p95_abs_rgb": 0.0,
        "fraction_abs_rgb_gt_16": 0.0,
        "thresholds": dict(PROXY_RGB_PARITY_THRESHOLDS),
    }
    parity[key] = value

    with pytest.raises((TypeError, ValueError)):
        evaluate_proxy_rgb_parity_for_source_hide(proxy, parity)

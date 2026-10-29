from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fluidmix.transport_stage import (  # noqa: E402
    ALL_STATE_INDICES,
    CAMERA_AND_LIGHT_PATHS,
    ENTRY_MODE,
    EXIT_MODE,
    INCOMING_LABEL_SOURCE_PATHS,
    INCOMING_PRIM_PATHS,
    LEGACY_ANIMATED_PATHS,
    OUTGOING_PRIM_PATHS,
    PILOT_VISUAL_REVIEW_CHECKS,
    QUEUED_PRIM_PATHS,
    QUEUE_TAIL_SOURCE_PATHS,
    SOURCE_CURVE_APPROVED_SHA256,
    SOURCE_CURVE_CANONICAL_13_SHA256,
    SOURCE_CURVE_CANONICAL_37_SHA256,
    SOURCE_CURVE_DELTA_37_SHA256,
    SOURCE_CURVE_SAMPLES,
    SOURCE_LAYER_HASHES,
    STATE_COUNT,
    TRANSPORT_FPS,
    BETWEEN_MODE,
    WP1_FILL_EXECUTION_PROFILE,
    build_run_transport_layout,
    build_state_plan,
    build_transport_layers,
    capture_evidence_closure,
    capture_transport_implementation,
    freeze_runtime_partition,
    package_state_indices,
    validate_evidence_closure,
    validate_runtime_partition,
    validate_wp1_fill_readiness,
)
from render_transport_stage import (  # noqa: E402
    IsaacTransportStageRuntime,
    PILOT_STATE_INDICES,
    _attest_scene_session_specs,
    _register_reviewed_mdl_search_path,
    _remove_reviewed_mdl_search_path,
    _transition_rtx_render_mode,
    run_transport_stage,
)
import render_transport_stage as transport_renderer  # noqa: E402
from build_transport_stage import _load_accepted_first_frame_endpoint  # noqa: E402
import build_transport_stage as transport_builder  # noqa: E402
from check_transport_stage import _validate_evidence_closure_shape  # noqa: E402
import check_transport_stage as transport_checker  # noqa: E402


OWNER_THREAD_ID = "01a18128-8dc4-7780-ace1-4d816b49d176"
IMPLEMENTATION_SHA256 = "f" * 64


class _ReachedFreshSessionStage(Exception):
    pass


class _DelayedRenderModeSettings:
    def __init__(self, *, initial: str, transition_after_updates: int | None) -> None:
        self.readback = initial
        self.requested: str | None = None
        self.transition_after_updates = transition_after_updates
        self.update_count = 0

    def get_as_string(self, _path: str) -> str:
        return self.readback

    def set_string(self, _path: str, value: str) -> None:
        self.requested = value
        if self.transition_after_updates == 0:
            self.readback = value

    def update(self) -> None:
        self.update_count += 1
        if (
            self.transition_after_updates is not None
            and self.update_count >= self.transition_after_updates
        ):
            assert self.requested is not None
            self.readback = self.requested


class _ReachedAfterRenderMode(Exception):
    pass


def test_isaac_open_waits_for_delayed_exact_render_mode_before_camera_use() -> None:
    settings = _DelayedRenderModeSettings(
        initial="PathTracing", transition_after_updates=2
    )

    class _Stage:
        def GetTimeCodesPerSecond(self) -> float:
            return 24.0

        def GetPrimAtPath(self, _path: str) -> None:
            assert settings.readback == "RaytracedLighting"
            raise _ReachedAfterRenderMode

    runtime = IsaacTransportStageRuntime.__new__(IsaacTransportStageRuntime)
    runtime._close_product = lambda: None
    runtime._open_fresh_session_stage = lambda _path: (_Stage(), object())
    runtime.timeline = SimpleNamespace(stop=lambda: None, set_current_time=lambda _time: None)
    runtime.receipt = {}
    runtime.apply_nuisance = lambda *_args, **_kwargs: None
    runtime.settings = settings
    runtime.app = SimpleNamespace(update=settings.update)
    runtime._render_mode_transitions = []
    runtime.f2 = SimpleNamespace(RTX_RENDER_MODE_SETTING="/rtx/rendermode")

    with pytest.raises(_ReachedAfterRenderMode):
        runtime._open(Path("state-00.usda"), 0)

    assert settings.update_count == 2
    assert runtime._render_mode_transitions[-1]["passed"] is True


def test_rtx_mode_transition_records_exact_immediate_target() -> None:
    settings = _DelayedRenderModeSettings(
        initial="PathTracing", transition_after_updates=0
    )
    records: list[dict[str, object]] = []

    result = _transition_rtx_render_mode(
        settings=settings,
        app=SimpleNamespace(update=settings.update),
        records=records,
        max_updates=3,
    )

    assert result == records[-1]
    assert result == {
        "setting_path": "/rtx/rendermode",
        "requested": "RaytracedLighting",
        "before": "PathTracing",
        "actual_readback": "RaytracedLighting",
        "app_updates": 0,
        "max_app_updates": 3,
        "passed": True,
    }


def test_rtx_mode_transition_polls_app_until_exact_delayed_target() -> None:
    settings = _DelayedRenderModeSettings(
        initial="PathTracing", transition_after_updates=2
    )
    records: list[dict[str, object]] = []

    result = _transition_rtx_render_mode(
        settings=settings,
        app=SimpleNamespace(update=settings.update),
        records=records,
        max_updates=3,
    )

    assert result["actual_readback"] == "RaytracedLighting"
    assert result["app_updates"] == 2
    assert result["passed"] is True
    assert settings.update_count == 2


def test_rtx_mode_transition_fails_after_bounded_non_target_readback() -> None:
    settings = _DelayedRenderModeSettings(
        initial="RayTracedLighting", transition_after_updates=None
    )
    records: list[dict[str, object]] = []

    with pytest.raises(RuntimeError, match="exact RaytracedLighting"):
        _transition_rtx_render_mode(
            settings=settings,
            app=SimpleNamespace(update=settings.update),
            records=records,
            max_updates=3,
        )

    assert settings.update_count == 3
    assert records == [
        {
            "setting_path": "/rtx/rendermode",
            "requested": "RaytracedLighting",
            "before": "RayTracedLighting",
            "actual_readback": "RayTracedLighting",
            "app_updates": 3,
            "max_app_updates": 3,
            "passed": False,
        }
    ]


def test_reviewed_mdl_directory_is_registered_and_removed_exactly(
    tmp_path: Path,
) -> None:
    mdl_file = tmp_path / "reviewed" / "FluidmixSpatialOmniSurface.mdl"
    mdl_file.parent.mkdir()
    mdl_file.write_text("mdl 1.7;", encoding="utf-8")
    calls: list[tuple[str, object]] = []

    class _Manager:
        enabled = False

        def is_extension_enabled(self, extension: str) -> bool:
            calls.append(("is_enabled", extension))
            return self.enabled

        def set_extension_enabled_immediate(
            self, extension: str, enabled: bool
        ) -> None:
            calls.append(("enable", (extension, enabled)))
            self.enabled = enabled

    class _Neuraylib:
        def addMdlSearchPath(self, path: str) -> str:
            calls.append(("add", path))
            return "reviewed-handle"

        def removeMdlSearchPath(self, handle: object) -> bool:
            calls.append(("remove", handle))
            return True

    manager = _Manager()
    neuraylib = _Neuraylib()
    registration = _register_reviewed_mdl_search_path(
        app=SimpleNamespace(get_extension_manager=lambda: manager),
        mdl_file=mdl_file,
        module_loader=lambda name: SimpleNamespace(get_neuraylib=lambda: neuraylib),
    )

    assert registration[0] is neuraylib
    assert registration[1] == "reviewed-handle"
    assert calls[-1] == ("add", str(mdl_file.parent.resolve()))
    _remove_reviewed_mdl_search_path(*registration)
    assert calls[-1] == ("remove", "reviewed-handle")


class _FakeSessionLayer:
    def __init__(
        self, identifier: str, *, anonymous: bool, root_prims: list[object]
    ) -> None:
        self.identifier = identifier
        self.anonymous = anonymous
        self.rootPrims = root_prims


class _FakeSessionStage:
    def __init__(self, session: _FakeSessionLayer) -> None:
        self.session = session

    def GetSessionLayer(self) -> _FakeSessionLayer:
        return self.session


class _FakeSessionContext:
    def __init__(self, observed_session: _FakeSessionLayer) -> None:
        self.observed_session = observed_session
        self.stage: _FakeSessionStage | None = None
        self.default_open_calls: list[str] = []
        self.explicit_open_calls: list[tuple[str, str]] = []

    def open_stage(self, path: str) -> bool:
        self.default_open_calls.append(path)
        self.stage = _FakeSessionStage(
            _FakeSessionLayer(
                "anon:stale:context-session.usda",
                anonymous=True,
                root_prims=[object()],
            )
        )
        return True

    def open_stage_with_session_layer(
        self,
        path: str,
        session_identifier: str,
        callback: object,
    ) -> bool:
        self.explicit_open_calls.append((path, session_identifier))
        self.stage = _FakeSessionStage(self.observed_session)
        callback(True, "")
        return True

    def get_stage_loading_status(self) -> tuple[int, int, int]:
        return (0, 0, 0)

    def get_stage(self) -> _FakeSessionStage | None:
        return self.stage


def _fake_session_runtime(
    requested_session: _FakeSessionLayer,
    observed_session: _FakeSessionLayer | None = None,
) -> tuple[IsaacTransportStageRuntime, _FakeSessionContext]:
    context = _FakeSessionContext(observed_session or requested_session)
    runtime = IsaacTransportStageRuntime.__new__(IsaacTransportStageRuntime)
    runtime._close_product = lambda: None
    runtime.context = context
    runtime.app = SimpleNamespace(update=lambda: None)
    runtime.Sdf = SimpleNamespace(
        Layer=SimpleNamespace(CreateAnonymous=lambda _name: requested_session)
    )
    return runtime, context


def test_isaac_runtime_opens_state_with_explicit_fresh_anonymous_session(
    tmp_path: Path,
) -> None:
    """Reproduce the Host-1 failure before any renderer or physics work."""

    fresh_session = _FakeSessionLayer(
        "anon:fresh:transport-session.usda", anonymous=True, root_prims=[]
    )
    state_layer = tmp_path / "state-00.usda"
    runtime, context = _fake_session_runtime(fresh_session)
    runtime.timeline = SimpleNamespace(
        stop=lambda: (_ for _ in ()).throw(_ReachedFreshSessionStage())
    )

    with pytest.raises(_ReachedFreshSessionStage):
        runtime._open(state_layer, 0)

    assert context.default_open_calls == []
    assert context.explicit_open_calls == [
        (str(state_layer), fresh_session.identifier)
    ]
    assert runtime.stage is context.stage
    assert runtime.session is fresh_session


def test_isaac_runtime_rejects_nonempty_session_from_fresh_layer_factory(
    tmp_path: Path,
) -> None:
    stale_session = _FakeSessionLayer(
        "anon:stale:transport-session.usda", anonymous=True, root_prims=[object()]
    )
    runtime, context = _fake_session_runtime(stale_session)

    with pytest.raises(RuntimeError, match="create a fresh anonymous session layer"):
        runtime._open_fresh_session_stage(tmp_path / "state-00.usda")

    assert context.explicit_open_calls == []


def test_isaac_runtime_rejects_context_session_other_than_requested_layer(
    tmp_path: Path,
) -> None:
    requested = _FakeSessionLayer(
        "anon:fresh:requested.usda", anonymous=True, root_prims=[]
    )
    observed = _FakeSessionLayer(
        "anon:stale:observed.usda", anonymous=True, root_prims=[]
    )
    runtime, context = _fake_session_runtime(requested, observed)

    with pytest.raises(RuntimeError, match="requested fresh session layer"):
        runtime._open_fresh_session_stage(tmp_path / "state-00.usda")

    assert context.explicit_open_calls == [
        (str(tmp_path / "state-00.usda"), requested.identifier)
    ]


def test_isaac_runtime_rejects_nonanonymous_context_session_layer(
    tmp_path: Path,
) -> None:
    requested = _FakeSessionLayer(
        "anon:fresh:requested.usda", anonymous=True, root_prims=[]
    )
    observed = _FakeSessionLayer(
        requested.identifier, anonymous=False, root_prims=[]
    )
    runtime, _context = _fake_session_runtime(requested, observed)

    with pytest.raises(RuntimeError, match="requested fresh session layer"):
        runtime._open_fresh_session_stage(tmp_path / "state-00.usda")


class _ReachedAcceptedCountBody(Exception):
    pass


def _host1_count_body_source_evidence() -> dict[str, object]:
    return {
        "resolved_points_time_code": 681,
        "resolved_normals_time_code": 681,
        "resolved_extent_time_code": 681,
        "source_point_count": 512,
        "source_face_count": 448,
        "source_corner_count": 1664,
        "source_fan_triangle_count": 768,
        "source_geometry_sha256": (
            "e1b9711dc50195e559f54f96db2ce5389a34ef719213c443498a5585b6bd712f"
        ),
        "source_topology_sha256": (
            "edba8a108a846fbb3fd661e95b9a642b727a06e7ba431ca70e013d1255132acd"
        ),
        "copied_primvar_names": [],
        "source_world_bottom_z_m": 0.9762065644917131,
        "source_world_height_m": 0.12289410311404914,
        "source_signed_volume_m3": 0.00006676288235206474,
        "source_effective_area_m2": 0.0005432553772747496,
        "xy_area_correction_scale": 0.9811347431975761,
        "initial_points_sha256": (
            "2d1bf6e7a243e71bbbee6f6add6e897f408bc9631a98ca23f2bec1ae3eee816a"
        ),
        "initial_visibility": "invisible",
        "body_color_mode": "height_banded_particle_composition",
    }


def _gb10_count_body_source_evidence() -> dict[str, object]:
    evidence = _host1_count_body_source_evidence()
    evidence.update(
        {
            "source_signed_volume_m3": 0.00006676288235206428,
            "source_effective_area_m2": 0.0005432553772747458,
            "xy_area_correction_scale": 0.9811347431975795,
            "initial_points_sha256": (
                "39ef6f9a05f7cb6defa2908582c1fa5c4bb7b4d3b30bf97b47da735f465c0226"
            ),
        }
    )
    return evidence


def _host1_final_capture_geometry() -> dict[str, object]:
    return {
        "body_material_color": [
            0.484047532081604,
            0.6231661438941956,
            0.3532128632068634,
        ],
        "body_material_color_sha256": (
            "b0348222a39d418b1ebfe737baf128493f25b18a903f791732c5424b8c085f8d"
        ),
        "body_visibility": "inherited",
        "contained_particle_count": 5500,
        "cross_section_area_m2": 0.0005229514163272396,
        "extent_sha256": (
            "27db99f9285a08ab8978edbd0b18df6849ec512d5afd014a8638a83b83cbfb5e"
        ),
        "particle_volume_m3": 1.1272866937444934e-8,
        "phase": "t100-stage",
        "physics_step": 4872,
        "points_sha256": (
            "875bdaa7e8ccc1e131afc22fa242c78082b433f8f519aad8c9787f08a3bfe44f"
        ),
        "post_transform_signed_volume_m3": 0.0000620007685285962,
        "post_transform_volume_residual_m3": 3.72649068720865e-13,
        "representation": "count_driven_twin_liquid_mesh",
        "target_height_m": 0.11855932734896701,
        "target_volume_m3": 0.00006200076815594714,
        "volume_proof_coordinate_path": (
            "local_points_vec3f_then_inherited_world_transform"
        ),
        "world_bottom_z_m": 0.9762065644917131,
        "world_extent_m": [
            [5.219446482856196, -1.9941538249898683, 0.9762065644917131],
            [5.245354173271724, -1.9682461345743407, 1.0947658905491484],
        ],
        "world_height_m": 0.11855932605743535,
        "world_points_sha256": (
            "bbf4ce652e29fbbb57b93f677a642c087033c1a006bb108812f2056063b4aca7"
        ),
    }


def _gb10_final_capture_geometry() -> dict[str, object]:
    geometry = _host1_final_capture_geometry()
    geometry.update(
        {
            "post_transform_signed_volume_m3": 0.0000620007685285968,
            "post_transform_volume_residual_m3": 3.7264966503205987e-13,
        }
    )
    return geometry


def test_liquid_builder_accepts_host1_proof_count_body_recomputed_on_gb10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce CASE_91365fbcc3369a1c at the production comparison boundary."""

    accepted = _host1_count_body_source_evidence()
    observed = _gb10_count_body_source_evidence()

    class FakeEditTarget:
        def __init__(self, layer: object) -> None:
            self.layer = layer

        def GetLayer(self) -> object:
            return self.layer

    class FakeStage:
        def __init__(self) -> None:
            self.edit_target: FakeEditTarget | None = None

        def SetEditTarget(self, target: FakeEditTarget) -> None:
            self.edit_target = target

        def GetEditTarget(self) -> FakeEditTarget | None:
            return self.edit_target

    session = SimpleNamespace(customLayerData={})
    stage = FakeStage()
    material = SimpleNamespace(GetPrim=lambda: object())
    sdf = SimpleNamespace(
        Layer=SimpleNamespace(
            FindOrOpen=lambda _path: object(),
            CreateAnonymous=lambda _name: session,
        )
    )
    usd = SimpleNamespace(
        Stage=SimpleNamespace(Open=lambda _root, _session: stage),
        EditTarget=FakeEditTarget,
        EditContext=lambda *_args: nullcontext(),
    )
    pxr = SimpleNamespace(
        Gf=object(),
        Sdf=sdf,
        Usd=usd,
        UsdGeom=SimpleNamespace(Scope=SimpleNamespace(Define=lambda *_args: None)),
        UsdSemantics=object(),
        UsdShade=SimpleNamespace(
            Material=SimpleNamespace(Define=lambda *_args: material)
        ),
        Vt=object(),
        Semantics=object(),
    )

    def reached_post_comparison(**_kwargs: object) -> object:
        raise _ReachedAcceptedCountBody

    f2 = SimpleNamespace(
        COMPOSITION_COMPONENTS=("t200", "t400", "t100"),
        COUNT_DRIVEN_SOURCE_TIME_CODE=686,
        _author_omnisurface=lambda *_args: object(),
        _author_direct_liquid_semantics=lambda *_args: None,
        _author_spatial_omnisurface=lambda *_args: object(),
        _author_count_driven_twin_liquid_body=lambda **_kwargs: ({}, observed),
        _count_driven_contained_mask=lambda _positions: np.asarray([True]),
        _update_count_driven_twin_liquid_body=lambda *_args, **_kwargs: {},
        _select_count_driven_body_capture_geometry=(
            lambda *_args, **_kwargs: _gb10_final_capture_geometry()
        ),
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "physx_three_pour_sequence", f2)
    monkeypatch.setattr(
        transport_builder,
        "_accepted_liquid_contract",
        lambda _loaded: {
            "count_body": accepted,
            "capture_geometry": _host1_final_capture_geometry(),
            "band_evidence": {},
            "material_state": {},
            "transmissions": {
                "t200": [0.92, 0.48, 0.16],
                "t400": [0.12, 0.72, 0.68],
                "t100": [1.0, 1.0, 1.0],
            },
            "particle_volume_m3": 1.1272866937444934e-8,
        },
    )
    monkeypatch.setattr(
        transport_builder, "_restore_accepted_band_display_colors", reached_post_comparison
    )

    with pytest.raises(_ReachedAcceptedCountBody):
        transport_builder.author_transport_liquid_layer(
            loaded={
                "arrays": {
                    "positions": np.zeros((1, 3)),
                    "current_compositions": np.zeros((1, 3)),
                },
                "final_physics_step": 4872,
            },
            stage_path=tmp_path / "fill_station.usda",
            output_path=tmp_path / "transport-liquid.usda",
            owner_thread_id=OWNER_THREAD_ID,
            transport_implementation_sha256=IMPLEMENTATION_SHA256,
        )


def test_count_body_attestations_record_cross_platform_derived_evidence() -> None:
    source = transport_builder._attest_count_body_source_evidence(
        _host1_count_body_source_evidence(), _gb10_count_body_source_evidence()
    )
    capture = transport_builder._attest_count_body_capture_geometry(
        _host1_final_capture_geometry(), _gb10_final_capture_geometry()
    )

    assert source["passed"] is True
    assert source["relative_tolerance"] == 1.0e-12
    assert source["absolute_tolerance"] == 1.0e-12
    assert source["accepted_initial_points_sha256"] == (
        "2d1bf6e7a243e71bbbee6f6add6e897f408bc9631a98ca23f2bec1ae3eee816a"
    )
    assert source["recomputed_initial_points_sha256"] == (
        "39ef6f9a05f7cb6defa2908582c1fa5c4bb7b4d3b30bf97b47da735f465c0226"
    )
    assert source["initial_points_sha256_exact_match"] is False
    assert capture["passed"] is True


@pytest.mark.parametrize(
    ("attestation", "field", "value", "reason"),
    [
        (
            "source",
            "source_geometry_sha256",
            "0" * 64,
            "source_geometry_sha256_exact_mismatch",
        ),
        (
            "source",
            "source_topology_sha256",
            "0" * 64,
            "source_topology_sha256_exact_mismatch",
        ),
        (
            "source",
            "resolved_points_time_code",
            680,
            "resolved_points_time_code_exact_mismatch",
        ),
        (
            "source",
            "source_effective_area_m2",
            0.0005442553772747458,
            "source_effective_area_m2_derived_mismatch",
        ),
        (
            "source",
            "source_signed_volume_m3",
            float("nan"),
            "source_signed_volume_m3_derived_mismatch",
        ),
        (
            "source",
            "xy_area_correction_scale",
            [0.9811347431975795],
            "xy_area_correction_scale_derived_mismatch",
        ),
        (
            "capture",
            "points_sha256",
            "0" * 64,
            "points_sha256_exact_mismatch",
        ),
        (
            "capture",
            "world_height_m",
            0.11865932605743535,
            "world_height_m_derived_mismatch",
        ),
        (
            "capture",
            "post_transform_signed_volume_m3",
            float("inf"),
            "post_transform_signed_volume_m3_derived_mismatch",
        ),
        (
            "capture",
            "world_extent_m",
            [5.219446482856196, -1.9941538249898683, 0.9762065644917131],
            "world_extent_m_derived_mismatch",
        ),
    ],
)
def test_count_body_attestations_fail_closed_on_real_change_nonfinite_or_shape(
    attestation: str, field: str, value: object, reason: str
) -> None:
    if attestation == "source":
        accepted = _host1_count_body_source_evidence()
        observed = _gb10_count_body_source_evidence()
        observed[field] = value
        proof = transport_builder._attest_count_body_source_evidence(
            accepted, observed
        )
    else:
        accepted = _host1_final_capture_geometry()
        observed = _gb10_final_capture_geometry()
        observed[field] = value
        proof = transport_builder._attest_count_body_capture_geometry(
            accepted, observed
        )

    assert proof["passed"] is False
    assert reason in proof["failure_reasons"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _wp1_readiness_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    case_result = {
        "case_id": "CASE_91365fbcc3369a1c",
        "ordinal": 4,
        "outcome": "passed",
        "return_code": 0,
        "timed_out": False,
        "launch_error": None,
        "evidence": {
            "exit_code": 0,
            "failures": [],
            "launch_failures": [],
            "receipt_failures": [],
            "observed_counts": {
                "decoded_video_frame_count": 433,
                "source_capture_count": 3,
            },
            "artifact_validation": {
                "status": "pass",
                "failures": [],
                "verified_artifact_count": 75,
                "warnings": [],
            },
        },
    }
    offline = {
        "status": "pass",
        "failures": [],
        "offline_revalidations": [
            {
                "revalidation_type": "current_carrier_occlusion_mask_revalidation",
                "passed": True,
            }
        ],
        "verified_artifact_count": 75,
        "warnings": [],
    }
    implementation = {
        "source_commit": "24889cad3cf9d7302622a4f844fb4b2a4977d4c9",
        "supervisor_sha256": "d34c8b5aaf94c90f69dcfe5a4e234d42be2d808609342355ed00c2f9f4cba01e",
        "renderer_sha256": "c8be7ec97131d2e9de3cb1656ffd1f23fbae80b03fc27864158db929c29f8c24",
        "offline_validator_sha256": "3780a83579cf807cf8f4041bdfc931f8b98c374ae8f7eaea386be04ed6ac2315",
        "occlusion_profile_sha256": "52a262b7dee1c3603049a879c6aeedc15b83bf94e8df98d03e63221070e95a0e",
        "fill_station_sha256": SOURCE_LAYER_HASHES["fill_station.usda"],
        "geometry_sha256": SOURCE_LAYER_HASHES["geometry/fill_station.usdc"],
    }
    host_profile = {
        "schema_version": 1,
        "artifact_type": "fluidmix_wp1_l40s_host_profile_v1",
        "status": "pass",
        "gpu_model": "NVIDIA L40S",
        "isaac_sim_version": "6.2.0.0",
        "kit_version": "110.4.0+feature.386251.187cf21b.gl",
        "implementation": implementation,
        "process_conflicts": [],
    }
    paths = {
        "supervisor_case_result": tmp_path / "case-result.json",
        "offline_validator_result": tmp_path / "offline-validator.json",
        "host_profile": tmp_path / "host-profile.json",
    }
    for path, value in zip(paths.values(), (case_result, offline, host_profile), strict=True):
        _write_json(path, value)
    receipt = {
        "schema_version": 1,
        "artifact_type": "fluidmix_wp1_fill_readiness_v1",
        "status": "pass",
        "owner_thread_id": OWNER_THREAD_ID,
        "execution_profile": {
            "gpu_model": "NVIDIA L40S",
            "isaac_sim_version": "6.2.0.0",
            "kit_version": "110.4.0+feature.386251.187cf21b.gl",
            "implementation": implementation,
        },
        "proof_case_id": "CASE_91365fbcc3369a1c",
        **{
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
    }
    receipt_path = tmp_path / "wp1-fill-readiness.json"
    _write_json(receipt_path, receipt)
    return receipt_path, receipt


def _wp1_readiness_args(tmp_path: Path) -> dict[str, object]:
    receipt_path, _receipt = _wp1_readiness_fixture(tmp_path)
    return {
        "wp1_fill_readiness_path": receipt_path,
        "wp1_fill_readiness_sha256": _sha256(receipt_path),
    }


def test_wp1_fill_readiness_profile_is_the_accepted_24889cad_proof() -> None:
    assert WP1_FILL_EXECUTION_PROFILE == {
        "gpu_model": "NVIDIA L40S",
        "isaac_sim_version": "6.2.0.0",
        "kit_version": "110.4.0+feature.386251.187cf21b.gl",
        "implementation": {
            "source_commit": "24889cad3cf9d7302622a4f844fb4b2a4977d4c9",
            "supervisor_sha256": (
                "d34c8b5aaf94c90f69dcfe5a4e234d42be2d808609342355ed00c2f9f4cba01e"
            ),
            "renderer_sha256": (
                "c8be7ec97131d2e9de3cb1656ffd1f23fbae80b03fc27864158db929c29f8c24"
            ),
            "offline_validator_sha256": (
                "3780a83579cf807cf8f4041bdfc931f8b98c374ae8f7eaea386be04ed6ac2315"
            ),
            "occlusion_profile_sha256": (
                "52a262b7dee1c3603049a879c6aeedc15b83bf94e8df98d03e63221070e95a0e"
            ),
            "fill_station_sha256": SOURCE_LAYER_HASHES["fill_station.usda"],
            "geometry_sha256": SOURCE_LAYER_HASHES["geometry/fill_station.usdc"],
        },
    }


@pytest.mark.parametrize(
    "stale_fields",
    [
        {
            "source_commit": "16fbb85582935ce7fa3ceeffcfa6ddffb5af5e83",
            "supervisor_sha256": (
                "d27a610fe307f1c4e23986fef56caac547747a57804e59a87e7f3635f92da528"
            ),
            "renderer_sha256": (
                "0a397e3de8046a3ad4754b0924e4068459939b17fe49073f916f278b30c6548c"
            ),
            "offline_validator_sha256": (
                "f7d5014b5fdccb744ebc4e666fee8ab680b3e9f9c65d5f550c38e283716ed1b0"
            ),
            "occlusion_profile_sha256": (
                "b056579aec18b2d2ace726d7640289e0401f6239fa4303057646040e880504f7"
            ),
        },
        {
            "renderer_sha256": (
                "0a397e3de8046a3ad4754b0924e4068459939b17fe49073f916f278b30c6548c"
            )
        },
    ],
    ids=("complete-stale-profile", "mixed-current-and-stale-profile"),
)
def test_wp1_fill_readiness_rejects_stale_or_mixed_implementation_profile(
    tmp_path: Path, stale_fields: dict[str, str]
) -> None:
    receipt_path, receipt = _wp1_readiness_fixture(tmp_path)
    receipt["execution_profile"]["implementation"].update(stale_fields)
    host_path = Path(receipt["host_profile"]["path"])
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["implementation"].update(stale_fields)
    _write_json(host_path, host)
    receipt["host_profile"]["sha256"] = _sha256(host_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="execution profile"):
        validate_wp1_fill_readiness(
            receipt_path,
            declared_sha256=_sha256(receipt_path),
            owner_thread_id=OWNER_THREAD_ID,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(status="failed"), "identity"),
        (lambda row: row.update(owner_thread_id="wrong-owner"), "identity"),
        (
            lambda row: row["execution_profile"].update(gpu_model="NVIDIA GB10"),
            "execution profile",
        ),
        (lambda row: row.update(proof_case_id="CASE_wrong"), "proof case"),
        (lambda row: row.update(unexpected=True), "fields"),
    ],
)
def test_wp1_fill_readiness_rejects_outer_mutations(
    tmp_path: Path, mutation: object, message: str
) -> None:
    receipt_path, receipt = _wp1_readiness_fixture(tmp_path)
    mutation(receipt)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match=message):
        validate_wp1_fill_readiness(
            receipt_path,
            declared_sha256=_sha256(receipt_path),
            owner_thread_id=OWNER_THREAD_ID,
        )


@pytest.mark.parametrize(
    ("binding", "mutation", "message"),
    [
        (
            "supervisor_case_result",
            lambda row: row.update(outcome="failed"),
            "supervisor case result",
        ),
        (
            "offline_validator_result",
            lambda row: row.update(status="insufficient_evidence"),
            "offline validator result",
        ),
        (
            "host_profile",
            lambda row: row.update(gpu_model="NVIDIA GB10"),
            "host profile",
        ),
    ],
)
def test_wp1_fill_readiness_inspects_bound_evidence_and_detects_drift(
    tmp_path: Path, binding: str, mutation: object, message: str
) -> None:
    receipt_path, receipt = _wp1_readiness_fixture(tmp_path)
    bound_path = Path(receipt[binding]["path"])
    value = json.loads(bound_path.read_text(encoding="utf-8"))
    mutation(value)
    _write_json(bound_path, value)

    with pytest.raises(ValueError, match=message):
        validate_wp1_fill_readiness(
            receipt_path,
            declared_sha256=_sha256(receipt_path),
            owner_thread_id=OWNER_THREAD_ID,
        )


@pytest.mark.parametrize(
    "revalidations",
    [
        [
            {
                "revalidation_type": "current_carrier_occlusion_mask_revalidation",
                "passed": False,
            }
        ],
        [{"revalidation_type": "current_carrier_occlusion_mask_revalidation"}],
        [
            {
                "revalidation_type": "current_carrier_occlusion_mask_revalidation",
                "passed": 1,
            }
        ],
        [
            {
                "revalidation_type": "current_carrier_occlusion_mask_revalidation",
                "passed": "true",
            }
        ],
        [
            {
                "revalidation_type": "current_carrier_occlusion_mask_revalidation",
                "status": "pass",
            }
        ],
    ],
    ids=(
        "passed-false",
        "passed-missing",
        "passed-integer-truth",
        "passed-string-truth",
        "status-only-invented-shape",
    ),
)
def test_wp1_fill_readiness_requires_exact_boolean_passed_revalidation_rows(
    tmp_path: Path, revalidations: list[dict[str, object]]
) -> None:
    receipt_path, receipt = _wp1_readiness_fixture(tmp_path)
    offline_path = Path(receipt["offline_validator_result"]["path"])
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    offline["offline_revalidations"] = revalidations
    _write_json(offline_path, offline)
    receipt["offline_validator_result"]["sha256"] = _sha256(offline_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="offline validator result"):
        validate_wp1_fill_readiness(
            receipt_path,
            declared_sha256=_sha256(receipt_path),
            owner_thread_id=OWNER_THREAD_ID,
        )


def test_wp1_fill_readiness_requires_declared_hash_and_returns_binding(
    tmp_path: Path,
) -> None:
    receipt_path, _receipt = _wp1_readiness_fixture(tmp_path)

    binding = validate_wp1_fill_readiness(
        receipt_path,
        declared_sha256=_sha256(receipt_path),
        owner_thread_id=OWNER_THREAD_ID,
    )

    assert binding["artifact_type"] == "fluidmix_wp1_fill_readiness_binding_v1"
    assert binding["receipt_path"] == str(receipt_path.resolve())
    assert binding["receipt_sha256"] == _sha256(receipt_path)
    with pytest.raises(ValueError, match="declared hash"):
        validate_wp1_fill_readiness(
            receipt_path,
            declared_sha256="0" * 64,
            owner_thread_id=OWNER_THREAD_ID,
        )


def test_launcher_rejects_all_readiness_mutations_before_simulation_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation_app_calls = 0

    def fake_simulation_app(*_args: object, **_kwargs: object) -> object:
        nonlocal simulation_app_calls
        simulation_app_calls += 1
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setitem(
        sys.modules, "isaacsim", SimpleNamespace(SimulationApp=fake_simulation_app)
    )
    monkeypatch.setattr(
        transport_renderer,
        "_load_build",
        lambda _segment: ({"owner_thread_id": OWNER_THREAD_ID}, {}, {}),
    )

    for index, kind in enumerate(
        (
            "missing",
            "declared-hash",
            "extra-field",
            "status",
            "owner",
            "profile",
            "proof-case",
            "case-result",
            "offline-validator",
            "host-profile",
            "bound-drift",
        )
    ):
        root = tmp_path / str(index)
        root.mkdir()
        receipt_path, receipt = _wp1_readiness_fixture(root)
        declared_hash = _sha256(receipt_path)
        if kind == "missing":
            receipt_path = root / "missing.json"
        elif kind == "declared-hash":
            declared_hash = "0" * 64
        elif kind == "extra-field":
            receipt["extra"] = True
        elif kind == "status":
            receipt["status"] = "failed"
        elif kind == "owner":
            receipt["owner_thread_id"] = "wrong-owner"
        elif kind == "profile":
            receipt["execution_profile"]["gpu_model"] = "NVIDIA GB10"
        elif kind == "proof-case":
            receipt["proof_case_id"] = "CASE_wrong"
        elif kind in {"case-result", "offline-validator", "host-profile", "bound-drift"}:
            binding = {
                "case-result": "supervisor_case_result",
                "offline-validator": "offline_validator_result",
                "host-profile": "host_profile",
                "bound-drift": "supervisor_case_result",
            }[kind]
            bound_path = Path(receipt[binding]["path"])
            value = json.loads(bound_path.read_text(encoding="utf-8"))
            if kind in {"case-result", "bound-drift"}:
                value["outcome"] = "failed"
            elif kind == "offline-validator":
                value["status"] = "insufficient_evidence"
            else:
                value["gpu_model"] = "NVIDIA GB10"
            _write_json(bound_path, value)
            if kind != "bound-drift":
                receipt[binding]["sha256"] = _sha256(bound_path)
        if kind not in {"missing", "declared-hash"}:
            _write_json(receipt_path, receipt)
            declared_hash = _sha256(receipt_path)

        assert (
            transport_renderer.main(
                [
                    "--segment-dir",
                    str(root / "segment"),
                    "--output-dir",
                    str(root / "render"),
                    "--pilot",
                    "--wp1-fill-readiness",
                    str(receipt_path),
                    "--wp1-fill-readiness-sha256",
                    declared_hash,
                ]
            )
            == 2
        )
        assert simulation_app_calls == 0


def test_next_endpoint_loader_consumes_only_receipt_and_video(tmp_path: Path) -> None:
    case = tmp_path / "app-output"
    video = _write(case / "three-pour.avi", b"accepted-video")
    receipt = {
        "schema_version": 20,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "status": "pass",
        "scene_contract": {
            "source_layers_unchanged": True,
            "source_layer_records": [
                {
                    "identifier": "fill_station.usda",
                    "resolved_path": "/source/fill_station.usda",
                    "bytes": 906,
                    "sha256": SOURCE_LAYER_HASHES["fill_station.usda"],
                },
                {
                    "identifier": "geometry/fill_station.usdc",
                    "resolved_path": "/source/geometry/fill_station.usdc",
                    "bytes": 26_464_568,
                    "sha256": SOURCE_LAYER_HASHES["geometry/fill_station.usdc"],
                },
            ],
        },
        "render": {
            "video_path": video.name,
            "video_sha256": _sha256(video),
            "video_decodable": True,
            "video_decoded_frame_count": 12,
        },
    }
    _write_json(case / "receipt.json", receipt)
    # A next fill's snapshot is deliberately absent: only its accepted first video frame
    # is consumed by the between-mode endpoint substitution.

    endpoint = _load_accepted_first_frame_endpoint(case)

    assert endpoint["first_pre_pour"]["video_path"] == str(video.resolve())
    assert endpoint["dependencies"] == [
        (case / "receipt.json").resolve(),
        video.resolve(),
    ]


def test_frozen_source_curve_and_exact_mover_sets() -> None:
    assert SOURCE_CURVE_APPROVED_SHA256 == (
        "c340e573d8b225326176a2cd3523ea2b0c19895c7d1187e0a67954a8a485a317"
    )
    assert SOURCE_CURVE_CANONICAL_37_SHA256 == (
        "58ac18e079d9c6f99280338046584b21772eea475eac976d9a83d75d15597625"
    )
    assert SOURCE_CURVE_DELTA_37_SHA256 == (
        "621410b9ece7875fc0a0c7e5ef6feb3e7a34d521a6f6490bbf454f578702e528"
    )
    assert SOURCE_CURVE_CANONICAL_13_SHA256 == (
        "aed5b56bcd9443684f376db70272a25940b4715051d047b0a98cf64e06b2a613"
    )
    assert [row.time_code for row in SOURCE_CURVE_SAMPLES] == list(range(180, 217, 3))
    assert [row.progress for row in SOURCE_CURVE_SAMPLES] == [
        0.0,
        0.01967620849609375,
        0.07407379150390625,
        0.15625,
        0.2592582702636719,
        0.3761558532714844,
        0.5,
        0.6238422393798828,
        0.7407398223876953,
        0.84375,
        0.9259243011474609,
        0.9803256988525391,
        1.0,
    ]
    assert STATE_COUNT == 13
    assert TRANSPORT_FPS == 10
    assert ALL_STATE_INDICES == tuple(range(13))
    assert len(OUTGOING_PRIM_PATHS) == 57
    assert len(INCOMING_PRIM_PATHS) == 27
    assert len(QUEUED_PRIM_PATHS) == 27
    assert len(INCOMING_LABEL_SOURCE_PATHS) == 29
    assert QUEUE_TAIL_SOURCE_PATHS == tuple(sorted(QUEUED_PRIM_PATHS))
    assert SOURCE_LAYER_HASHES == {
        "fill_station.usda": "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9",
        "geometry/fill_station.usdc": (
            "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db"
        ),
    }


@pytest.mark.parametrize("mode", [ENTRY_MODE, BETWEEN_MODE, EXIT_MODE])
def test_state_plan_is_source_time_one_and_hits_exact_mode_endpoints(mode: str) -> None:
    plan = build_state_plan(mode)

    assert len(plan) == 13
    assert [row["state_index"] for row in plan] == list(range(13))
    assert [row["source_time_code"] for row in plan] == [1] * 13
    assert [row["curve_time_code"] for row in plan] == list(range(180, 217, 3))
    assert all(row["physics_steps"] == 0 for row in plan)
    assert plan[0]["progress"] == 0.0
    assert plan[-1]["progress"] == 1.0
    assert all(
        plan[index]["progress"] < plan[index + 1]["progress"]
        for index in range(12)
    )

    if mode == ENTRY_MODE:
        assert plan[-1]["world_delta_y_m_by_group"] == {
            "incoming": 0.0848413319824885,
            "queued": 0.186441330754633,
            "queue_tail": 0.1016,
        }
        assert plan[-1]["outgoing_visibility"] == "invisible"
    elif mode == BETWEEN_MODE:
        assert plan[-1]["world_delta_y_m_by_group"] == {
            "outgoing": 0.19,
            "incoming": 0.0848413319824885,
            "queued": 0.186441330754633,
            "queue_tail": 0.1016,
        }
        assert plan[-1]["outgoing_visibility"] == "inherited"
    else:
        assert plan[-1]["world_delta_y_m_by_group"] == {"outgoing": 0.19}
        assert plan[-1]["incoming_label_root_local_y"] is None
        assert plan[-1]["queue_tail_root_local_y"] is None


def test_packaging_endpoint_substitution_and_wp1_wp1b_layouts() -> None:
    assert package_state_indices(ENTRY_MODE) == {
        "accepted_start_endpoint": None,
        "rendered_state_indices": list(range(12)),
        "accepted_end_endpoint": "next_fill_first_pre_pour",
    }
    assert package_state_indices(BETWEEN_MODE) == {
        "accepted_start_endpoint": "outgoing_fill_final",
        "rendered_state_indices": list(range(1, 12)),
        "accepted_end_endpoint": "next_fill_first_pre_pour",
    }
    assert package_state_indices(EXIT_MODE) == {
        "accepted_start_endpoint": "outgoing_fill_final",
        "rendered_state_indices": list(range(1, 13)),
        "accepted_end_endpoint": None,
    }
    layout = build_run_transport_layout([f"CASE_{index:016x}" for index in range(9)])
    assert [row["mode"] for row in layout] == [ENTRY_MODE] + [BETWEEN_MODE] * 8 + [EXIT_MODE]
    assert len(layout) == 10
    assert layout[0]["next_case_id"] == "CASE_0000000000000000"
    assert layout[-1]["outgoing_case_id"] == "CASE_0000000000000008"

    wp1b_layout = build_run_transport_layout(
        [f"CASE_{index:016x}" for index in range(3)]
    )
    assert [row["mode"] for row in wp1b_layout] == [
        ENTRY_MODE,
        BETWEEN_MODE,
        BETWEEN_MODE,
        EXIT_MODE,
    ]
    assert len(wp1b_layout) == 4


def test_start_seam_uses_actual_last_decoded_fill_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted_png = np.full((480, 640, 3), 11, dtype=np.uint8)
    decoded_frames = [
        np.full((480, 640, 3), 22, dtype=np.uint8),
        np.full((480, 640, 3), 33, dtype=np.uint8),
    ]
    monkeypatch.setattr(
        transport_checker, "_load_rgb", lambda *_args, **_kwargs: accepted_png
    )
    monkeypatch.setattr(
        transport_checker, "_decode_video", lambda *_args, **_kwargs: decoded_frames
    )

    start, video, end, end_video = transport_checker._accepted_endpoint_evidence(
        {
            "adjacent_endpoints": {
                "accepted_start": {
                    "path": "/accepted/final.png",
                    "sha256": "a" * 64,
                    "video_path": "/accepted/fill.mp4",
                    "video_sha256": "b" * 64,
                    "frame_index": -1,
                },
                "accepted_end": None,
            }
        },
        EXIT_MODE,
    )

    assert start is decoded_frames[-1]
    assert video is decoded_frames
    assert end is None
    assert end_video is None


def _occupancy_captures(mode: str) -> list[dict[str, object]]:
    captures: list[dict[str, object]] = []
    for state_index in (0, 6, 12):
        instance = np.zeros((480, 640), dtype=np.uint32)
        mapping: dict[str, str] = {}
        if mode in {BETWEEN_MODE, EXIT_MODE} and state_index == 0:
            mapping["1"] = sorted(OUTGOING_PRIM_PATHS)[0]
            instance[200:210, 100:110] = 1
        if mode in {ENTRY_MODE, BETWEEN_MODE} and state_index == 12:
            mapping["2"] = transport_checker.INCOMING_VIAL_PATH
            instance[200:210, 304:306] = 2
        masks = {}
        for name, prefixes in transport_checker.GROUP_PREFIXES.items():
            ids = [
                int(instance_id)
                for instance_id, path in mapping.items()
                if any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)
            ]
            masks[name] = np.isin(instance, ids).astype(np.uint8) * 255
        captures.append(
            {
                "state_index": state_index,
                "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                "instance": instance,
                "mapping": mapping,
                "masks": masks,
            }
        )
    return captures


@pytest.mark.parametrize("mode", [ENTRY_MODE, BETWEEN_MODE, EXIT_MODE])
def test_mode_occupancy_is_non_vacuous_for_pilot_states(mode: str) -> None:
    evidence = transport_checker._mode_occupancy_evidence(
        mode, _occupancy_captures(mode)
    )

    assert evidence["state_indices"] == [0, 6, 12]
    assert evidence["mode_occupancy_valid"] is True
    if mode in {ENTRY_MODE, BETWEEN_MODE}:
        assert evidence["pixel_counts_by_state"]["12"]["incoming"] > 0
        assert evidence["incoming_centered_late"] is True
    if mode in {BETWEEN_MODE, EXIT_MODE}:
        assert evidence["pixel_counts_by_state"]["0"]["outgoing"] > 0
        assert evidence["pixel_counts_by_state"]["12"]["outgoing"] == 0


@pytest.mark.parametrize(
    ("mode", "state_index", "group", "path"),
    [
        (ENTRY_MODE, 0, "outgoing", sorted(OUTGOING_PRIM_PATHS)[0]),
        (BETWEEN_MODE, 12, "outgoing", sorted(OUTGOING_PRIM_PATHS)[0]),
        (EXIT_MODE, 6, "queue_tail", transport_checker.GROUP_PREFIXES["queue_tail"][0]),
    ],
)
def test_mode_occupancy_rejects_unexpected_groups(
    mode: str, state_index: int, group: str, path: str
) -> None:
    captures = _occupancy_captures(mode)
    row = next(item for item in captures if item["state_index"] == state_index)
    row["mapping"]["9"] = path
    row["instance"][20:25, 20:25] = 9
    row["masks"][group] = (row["instance"] == 9).astype(np.uint8) * 255

    evidence = transport_checker._mode_occupancy_evidence(mode, captures)

    assert evidence["mode_occupancy_valid"] is False


@pytest.mark.parametrize(
    ("mode", "state_index", "group"),
    [
        (BETWEEN_MODE, 0, "outgoing"),
        (BETWEEN_MODE, 12, "incoming"),
        (EXIT_MODE, 0, "outgoing"),
    ],
)
def test_mode_occupancy_rejects_missing_required_groups(
    mode: str, state_index: int, group: str
) -> None:
    captures = _occupancy_captures(mode)
    row = next(item for item in captures if item["state_index"] == state_index)
    mask = row["masks"][group]
    ids = set(row["instance"][mask != 0].tolist())
    row["instance"][mask != 0] = 0
    row["masks"][group] = np.zeros_like(mask)
    for instance_id in ids:
        row["mapping"].pop(str(instance_id), None)

    evidence = transport_checker._mode_occupancy_evidence(mode, captures)

    assert evidence["mode_occupancy_valid"] is False


def test_mode_occupancy_rejects_forged_zero_group_mask() -> None:
    captures = _occupancy_captures(BETWEEN_MODE)
    captures[0]["masks"]["outgoing"][:] = 0

    with pytest.raises(ValueError, match="group mask differs"):
        transport_checker._mode_occupancy_evidence(BETWEEN_MODE, captures)


def test_builder_materializes_exact_default_valued_layers_without_camera_or_time_samples(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "source" / "fill_station.usda", b"source-usda")
    geometry = _write(tmp_path / "source" / "geometry" / "fill_station.usdc", b"source-usdc")
    liquid = _write(tmp_path / "liquid.usda", b"#usda 1.0\ndef Xform \"FillStation\" {}\n")
    output = tmp_path / "segment"

    manifest = build_transport_layers(
        mode=BETWEEN_MODE,
        owner_thread_id=OWNER_THREAD_ID,
        transport_implementation_sha256=IMPLEMENTATION_SHA256,
        source_stage=source,
        source_geometry=geometry,
        liquid_layer=liquid,
        output_dir=output,
        expected_source_hashes={
            "fill_station.usda": _sha256(source),
            "geometry/fill_station.usdc": _sha256(geometry),
        },
        liquid_evidence_sha256=_sha256(liquid),
    )

    assert manifest["source_time_code"] == 1
    assert manifest["owner_thread_id"] == OWNER_THREAD_ID
    assert manifest["transport_implementation_sha256"] == IMPLEMENTATION_SHA256
    assert len(manifest["state_layers"]) == 13
    assert manifest["mode"] == BETWEEN_MODE
    assert (output / "transport-stage.usda").is_file()
    assert (output / "transport-liquid.usda").read_bytes() == liquid.read_bytes()
    for row in manifest["state_layers"]:
        path = output / row["path"]
        assert _sha256(path) == row["sha256"]
        text = path.read_text(encoding="utf-8")
        assert "timeSamples" not in text
        assert "timeCodesPerSecond = 30" in text
        assert "framesPerSecond = 30" in text
        assert "fluidmixTransport" in text
        assert f'ownerThreadId = "{OWNER_THREAD_ID}"' in text
        assert f'transportImplementationSha256 = "{IMPLEMENTATION_SHA256}"' in text
        assert all(path_value not in text for path_value in CAMERA_AND_LIGHT_PATHS)
        assert all(
            path_value.rsplit("/", 1)[-1] not in text
            for path_value in CAMERA_AND_LIGHT_PATHS
        )
        assert "Physx" not in text and "Physics" not in text
    common = (output / "transport-stage.usda").read_text(encoding="utf-8")
    assert "sourceTimeCode = 1" in common
    assert f'ownerThreadId = "{OWNER_THREAD_ID}"' in common
    assert f'transportImplementationSha256 = "{IMPLEMENTATION_SHA256}"' in common
    assert "timeCodesPerSecond = 30" in common
    assert "framesPerSecond = 30" in common
    assert all(path_value not in common for path_value in CAMERA_AND_LIGHT_PATHS)
    assert all(
        path_value.rsplit("/", 1)[-1] not in common
        for path_value in CAMERA_AND_LIGHT_PATHS
    )
    assert "Physx" not in common and "Physics" not in common
    assert all(path_value.rsplit("/", 1)[-1] in common for path_value in LEGACY_ANIMATED_PATHS)


def test_evidence_closure_binds_all_thirteen_arrays_layers_and_revalidates_mutation(
    tmp_path: Path,
) -> None:
    sources = [
        _write(tmp_path / "source.usda", b"source"),
        _write(tmp_path / "source.usdc", b"geometry"),
    ]
    arrays = [
        _write(
            tmp_path / "arrays" / f"array-{index:02d}.npy",
            f"array-{index}".encode(),
        )
        for index in range(13)
    ]
    layers = [
        _write(
            tmp_path / "states" / f"state-{index:02d}.usda",
            f"state-{index}".encode(),
        )
        for index in range(13)
    ]
    receipt = _write(tmp_path / "receipt.json", b"receipt")
    counters = {
        "physics_steps": 0,
        "physics_callbacks": 0,
        "timeline_playing": False,
        "source_time_code": 1,
    }

    closure = capture_evidence_closure(
        source_files=sources,
        snapshot_arrays=arrays,
        state_layers=layers,
        dependency_files=[receipt],
        counters=counters,
    )

    assert len(closure["snapshot_arrays"]) == 13
    assert len(closure["state_layers"]) == 13
    assert _validate_evidence_closure_shape(closure) == closure
    validate_evidence_closure(
        closure,
        source_files=sources,
        snapshot_arrays=arrays,
        state_layers=layers,
        dependency_files=[receipt],
        counters=counters,
    )
    arrays[-1].write_bytes(b"mutated")
    with pytest.raises(ValueError, match="snapshot array evidence drifted"):
        validate_evidence_closure(
            closure,
            source_files=sources,
            snapshot_arrays=arrays,
            state_layers=layers,
            dependency_files=[receipt],
            counters=counters,
        )

    malformed = deepcopy(closure)
    malformed["snapshot_arrays"] = malformed["snapshot_arrays"][:-1]
    with pytest.raises(ValueError, match="snapshot_arrays closure membership"):
        _validate_evidence_closure_shape(malformed)


def test_post_annotator_runtime_partition_is_exact_and_hash_bound() -> None:
    prims = [
        "/Render",
        "/Render/Vars",
        "/Render/Vars/Rgb",
        "/Render/Product",
        "/Render/PostProcess",
        "/Render/PostProcess/SDGPipeline",
        "/Render/PostProcess/SDGPipeline/InstanceSegmentation",
    ]
    properties = [
        "/Render.purpose",
        "/Render/Product.camera",
        "/Render/Product.resolution",
        "/Render/Vars/Rgb.dataType",
        "/Render/PostProcess/SDGPipeline/InstanceSegmentation.inputs:renderProductPath",
    ]
    frozen = freeze_runtime_partition(prims, properties)

    validate_runtime_partition(frozen, prims, properties)
    changed = deepcopy(properties)
    changed.append("/Render/PostProcess/SDGPipeline/Rogue.inputs:value")
    with pytest.raises(ValueError, match="runtime render partition drifted"):
        validate_runtime_partition(frozen, prims, changed)
    with pytest.raises(ValueError, match="runtime render partition is malformed"):
        freeze_runtime_partition([*prims, "/RendererRogue"], properties)
    assert frozen["sha256"] == hashlib.sha256(
        json.dumps(
            {"prim_paths": sorted(prims), "property_paths": sorted(properties)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class _FakeSessionSpec:
    def __init__(
        self,
        path: str,
        *,
        fields: dict[str, object] | None = None,
        properties: list["_FakeSessionSpec"] | None = None,
        children: list["_FakeSessionSpec"] | None = None,
    ) -> None:
        self.path = path
        self.fields = fields or {}
        self.properties = properties or []
        self.nameChildren = children or []

    def ListInfoKeys(self) -> list[str]:
        return list(self.fields)

    def GetInfo(self, key: str) -> object:
        return self.fields[key]


@pytest.mark.parametrize("field", ["default", "timeSamples", "customMetadata"])
def test_scene_session_digest_attests_same_path_authored_values(field: str) -> None:
    property_spec = _FakeSessionSpec(
        "/Nuisance.exposure",
        fields={
            "default": 1.25,
            "timeSamples": {1.0: 1.5},
            "customMetadata": {"source": "accepted-fill"},
        },
    )
    nuisance = _FakeSessionSpec(
        "/Nuisance",
        fields={"specifier": "over"},
        properties=[property_spec],
    )
    render_property = _FakeSessionSpec(
        "/Render/Product.resolution", fields={"default": [640, 480]}
    )
    render = _FakeSessionSpec("/Render", properties=[render_property])
    before = _attest_scene_session_specs([nuisance, render])

    property_spec.fields[field] = {
        "default": 2.0,
        "timeSamples": {1.0: 2.5},
        "customMetadata": {"source": "mutated"},
    }[field]
    after = _attest_scene_session_specs([nuisance, render])

    assert before["prim_specs"] == after["prim_specs"]
    assert [row["path"] for row in before["property_specs"]] == [
        row["path"] for row in after["property_specs"]
    ]
    assert before["sha256"] != after["sha256"]


def _fake_segment(tmp_path: Path) -> tuple[Path, list[Path]]:
    source = _write(tmp_path / "source" / "fill_station.usda", b"source-usda")
    geometry = _write(
        tmp_path / "source" / "geometry" / "fill_station.usdc", b"source-usdc"
    )
    liquid = _write(
        tmp_path / "liquid.usda", b'#usda 1.0\ndef Xform "FillStation" {}\n'
    )
    segment = tmp_path / "segment"
    implementation = capture_transport_implementation(ROOT)
    manifest = build_transport_layers(
        mode=BETWEEN_MODE,
        owner_thread_id=OWNER_THREAD_ID,
        transport_implementation_sha256=str(implementation["sha256"]),
        source_stage=source,
        source_geometry=geometry,
        liquid_layer=liquid,
        output_dir=segment,
        expected_source_hashes={
            "fill_station.usda": _sha256(source),
            "geometry/fill_station.usdc": _sha256(geometry),
        },
        liquid_evidence_sha256=_sha256(liquid),
    )
    arrays = [
        _write(tmp_path / "arrays" / f"array-{index:02d}.npy", f"array-{index}".encode())
        for index in range(13)
    ]
    dependencies = [_write(tmp_path / "receipt.json", b"receipt")]
    state_layers = [segment / row["path"] for row in manifest["state_layers"]]
    closure = capture_evidence_closure(
        source_files=[source, geometry],
        snapshot_arrays=arrays,
        state_layers=state_layers,
        dependency_files=dependencies,
        counters={
            "physics_steps": 0,
            "physics_callbacks": 0,
            "timeline_playing": False,
            "source_time_code": 1,
        },
    )
    _write_json(
        segment / "transport-build-receipt.json",
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_stage_build_receipt_v1",
            "status": "pass",
            "mode": BETWEEN_MODE,
            "owner_thread_id": OWNER_THREAD_ID,
            "transport_implementation": implementation,
            "source_time_code": 1,
            "physics_steps": 0,
            "camera_and_light_specs_authored": [],
            "layer_manifest": manifest,
            "pre_output_evidence_closure": closure,
        },
    )
    return segment, arrays


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.partition = freeze_runtime_partition(
            [
                "/Render",
                "/Render/Vars",
                "/Render/Vars/Rgb",
                "/Render/Product",
                "/Render/PostProcess/SDGPipeline",
            ],
            [
                "/Render/Product.camera",
                "/Render/Vars/Rgb.dataType",
                "/Render/PostProcess/SDGPipeline.inputs:renderProductPath",
            ],
        )
        self.current = 0
        self.closed = False

    def _attestation(self, index: int) -> dict[str, object]:
        return {
            "state_index": index,
            "source_time_code": 1,
            "physics_steps": 0,
            "physics_callbacks": 0,
            "timeline_playing": False,
            "runtime_partition": self.partition,
            "camera_light_before_annotators_sha256": "1" * 64,
            "camera_light_sha256": "1" * 64,
            "scene_session_sha256": "2" * 64,
        }

    def preflight_state(self, _path: Path, state_index: int) -> dict[str, object]:
        self.current = state_index
        return self._attestation(state_index)

    def capture_state(self, _path: Path, state_index: int) -> dict[str, object]:
        self.current = state_index
        rgb = np.full((480, 640, 3), state_index + 10, dtype=np.uint8)
        instance = np.zeros((480, 640), dtype=np.uint32)
        empty = np.zeros((480, 640), dtype=np.uint8)
        return {
            "rgb": rgb,
            "instance_ids": instance,
            "instance_id_to_path": {},
            "group_masks": {
                "outgoing": empty.copy(),
                "incoming": empty.copy(),
                "queued": empty.copy(),
                "incoming_labels": empty.copy(),
                "queue_tail": empty.copy(),
            },
            "attestation": self._attestation(state_index),
        }

    def attest_current(self) -> dict[str, object]:
        return self._attestation(self.current)

    def close(self) -> None:
        self.closed = True


def _checker_result(*, pilot: bool = True) -> dict[str, object]:
    pending = None if pilot else True
    return {
        "schema_version": 1,
        "artifact_type": "fluidmix_transport_stage_checker_v1",
        "status": "pass",
        "pilot": pilot,
        "automatic_acceptance_gates": {
            "evidence_closure_exact": True,
            "runtime_partition_exact": True,
            "camera_and_lighting_unchanged": True,
            "source_time_one_and_zero_physics": True,
            "instance_mappings_and_masks_valid": True,
            "background_sentinels_within_2": True,
            "mode_occupancy_valid": True,
            "start_seam_passed": pending,
            "end_seam_passed": pending,
            "packaged_sequence_byte_unique": pending,
            "pilot_visual_review_accepted": pending,
        },
        "manual_visual_review_required": pilot,
        "pilot_visual_review_status": "required" if pilot else "accepted",
    }


def test_renderer_freezes_partition_before_output_and_publishes_after_recheck(
    tmp_path: Path,
) -> None:
    segment, _arrays = _fake_segment(tmp_path)
    output = tmp_path / "render"
    runtime = _FakeRuntime()

    receipt = run_transport_stage(
        segment_dir=segment,
        output_dir=output,
        state_indices=PILOT_STATE_INDICES,
        runtime=runtime,
        check_output=lambda **_kwargs: _checker_result(),
        **_wp1_readiness_args(tmp_path),
    )

    assert receipt["status"] == "pilot"
    assert receipt["owner_thread_id"] == OWNER_THREAD_ID
    assert receipt["pre_output_evidence_closure_sha256"] == receipt[
        "post_checker_evidence_closure"
    ]["sha256"]
    assert len(receipt["post_checker_evidence_closure"]["snapshot_arrays"]) == 13
    assert len(receipt["post_checker_evidence_closure"]["state_layers"]) == 13
    assert receipt["final_runtime_attestation"] == receipt[
        "preflight_attestations"
    ][-1]
    assert receipt["camera_light_before_annotators_sha256"] == receipt[
        "camera_light_sha256"
    ]
    assert output.is_dir()
    assert (output / "contact-sheet.png").is_file()
    assert runtime.closed is True


def test_full_renderer_requires_pilot_visual_review_before_any_preflight(
    tmp_path: Path,
) -> None:
    segment, _arrays = _fake_segment(tmp_path)
    output = tmp_path / "render"
    runtime = _FakeRuntime()

    with pytest.raises(ValueError, match="pilot visual review"):
        run_transport_stage(
            segment_dir=segment,
            output_dir=output,
            state_indices=ALL_STATE_INDICES,
            runtime=runtime,
            check_output=lambda **_kwargs: _checker_result(),
            **_wp1_readiness_args(tmp_path),
        )

    assert runtime.current == 0
    assert runtime.closed is True
    assert not output.exists()


def test_full_renderer_binds_exact_independently_accepted_pilot_review(
    tmp_path: Path,
) -> None:
    segment, _arrays = _fake_segment(tmp_path)
    pilot_output = tmp_path / "pilot"
    run_transport_stage(
        segment_dir=segment,
        output_dir=pilot_output,
        state_indices=PILOT_STATE_INDICES,
        runtime=_FakeRuntime(),
        check_output=lambda **_kwargs: _checker_result(),
        **_wp1_readiness_args(tmp_path),
    )
    pilot_receipt = pilot_output / "transport-render-receipt.json"
    contact_sheet = pilot_output / "contact-sheet.png"
    review_path = tmp_path / "reviews" / "transport-pilot-review.json"
    review_path.parent.mkdir()
    _write_json(
        review_path,
        {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_pilot_visual_review_v1",
            "status": "accepted",
            "owner_thread_id": OWNER_THREAD_ID,
            "transport_implementation_sha256": str(
                capture_transport_implementation(ROOT)["sha256"]
            ),
            "reviewer_id": "independent-reviewer",
            "independent_from_implementation": True,
            "reviewed_utc": "2026-10-28T18:00:00Z",
            "decision_reference": "test-review",
            "pilot_render_receipt": {
                "path": str(pilot_receipt),
                "sha256": _sha256(pilot_receipt),
            },
            "contact_sheet": {
                "path": str(contact_sheet),
                "sha256": _sha256(contact_sheet),
            },
            "checks": {key: True for key in PILOT_VISUAL_REVIEW_CHECKS},
        },
    )
    full_output = tmp_path / "full"

    receipt = run_transport_stage(
        segment_dir=segment,
        output_dir=full_output,
        state_indices=ALL_STATE_INDICES,
        runtime=_FakeRuntime(),
        check_output=lambda **_kwargs: _checker_result(pilot=False),
        **_wp1_readiness_args(tmp_path),
        pilot_visual_review_path=review_path,
    )

    assert receipt["status"] == "candidate"
    assert receipt["pilot_visual_review"]["status"] == "accepted"
    assert receipt["pilot_visual_review"]["review_sha256"] == _sha256(review_path)
    assert full_output.is_dir()


def test_renderer_revalidates_all_snapshot_arrays_after_final_checker(
    tmp_path: Path,
) -> None:
    segment, arrays = _fake_segment(tmp_path)
    output = tmp_path / "render"
    runtime = _FakeRuntime()
    calls = 0

    def mutate_after_checker(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            arrays[-1].write_bytes(b"mutated-after-checker")
        return _checker_result()

    with pytest.raises(ValueError, match="snapshot array evidence drifted"):
        run_transport_stage(
            segment_dir=segment,
            output_dir=output,
            state_indices=PILOT_STATE_INDICES,
            runtime=runtime,
            check_output=mutate_after_checker,
            **_wp1_readiness_args(tmp_path),
        )

    assert not output.exists()
    assert runtime.closed is True


def test_renderer_revalidates_build_receipt_after_final_checker(
    tmp_path: Path,
) -> None:
    segment, _arrays = _fake_segment(tmp_path)
    output = tmp_path / "render"
    runtime = _FakeRuntime()
    calls = 0

    def mutate_build_after_checker(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            path = segment / "transport-build-receipt.json"
            path.write_bytes(path.read_bytes() + b" ")
        return _checker_result()

    with pytest.raises(ValueError, match="build receipt evidence drifted"):
        run_transport_stage(
            segment_dir=segment,
            output_dir=output,
            state_indices=PILOT_STATE_INDICES,
            runtime=runtime,
            check_output=mutate_build_after_checker,
            **_wp1_readiness_args(tmp_path),
        )

    assert not output.exists()
    assert runtime.closed is True

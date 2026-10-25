"""Pure contracts for the bounded timeline-owned PhysX emission probe."""

from __future__ import annotations

import math


FROZEN_SEED = 20261025
FIXED_HZ = 60
TARGET_CALLBACK_COUNT = 12
EMISSION_CALLBACK_INDEX = 6
PARTICLES_PER_EMISSION = 4
CORRECTED_TARGET_CALLBACK_COUNT = 24
CORRECTED_PARTICLES_PER_EMISSION = 12


def build_timeline_probe_plan(*, seed: int) -> dict[str, object]:
    """Return the frozen timeline probe plan."""

    if seed != FROZEN_SEED:
        raise ValueError(f"seed must remain frozen at {FROZEN_SEED}")
    return {
        "schema_version": 1,
        "seed": seed,
        "attempt_limit": 1,
        "fixed_hz": FIXED_HZ,
        "fixed_dt_s": 1.0 / FIXED_HZ,
        "target_callback_count": TARGET_CALLBACK_COUNT,
        "emission_before_callback_index": EMISSION_CALLBACK_INDEX,
        "particles_per_emission": PARTICLES_PER_EMISSION,
        "simulation_owner": "timeline",
        "particle_set_model": "one_shared_points_set_created_on_main_thread",
        "hypothesis": (
            "a timeline-owned 60 Hz PhysX run ingests one main-thread-authored "
            "four-particle fluid set before callback 6, then reports finite downward "
            "particle motion and a substantive generated isosurface by callback 12"
        ),
        "falsifiers": [
            "callback count is not exactly 12",
            "a callback dt or consecutive timeline step differs from 1/60 s",
            "callbacks are not one-per-app-update under ticksPerFrame=1",
            "the one particle set is not authored after callback 5 and before callback 6",
            "four stable finite particle rows do not move downward",
            "the generated isosurface is missing or not substantive geometry",
            "the process log contains a PhysX GPU, CUDA, crash, or traceback marker",
        ],
        "stopping_condition": (
            "pause immediately after callback 12; no rerun or parameter change"
        ),
        "renderer_exercised": False,
        "brev_exercised": False,
        "claim_limits": (
            "engine ingestion and generated-mesh probe only; no twin scene, rendering, "
            "fluid-accuracy, real-data, or sim-to-real claim"
        ),
    }


def build_corrected_timeline_probe_plan(*, seed: int) -> dict[str, object]:
    """Return the separate A18 layer-corrected timeline probe plan."""

    if seed != FROZEN_SEED:
        raise ValueError(f"seed must remain frozen at {FROZEN_SEED}")
    return {
        "schema_version": 2,
        "seed": seed,
        "attempt_limit": 1,
        "fixed_hz": FIXED_HZ,
        "fixed_dt_s": 1.0 / FIXED_HZ,
        "target_callback_count": CORRECTED_TARGET_CALLBACK_COUNT,
        "emission_before_callback_index": EMISSION_CALLBACK_INDEX,
        "particles_per_emission": CORRECTED_PARTICLES_PER_EMISSION,
        "position_source": (
            "a15_emission_ordinals_1_through_3_concatenated_unchanged"
        ),
        "simulation_owner": "timeline",
        "particle_set_model": "one_shared_points_set_created_on_main_thread",
        "runtime_writeback_edit_target": "session_layer",
        "subscription_phase": "after_play_commit_at_time_zero",
        "writeback_evidence": (
            "changed composed points with session-only point spec, no root point spec, "
            "and no manual final points Set"
        ),
        "hypothesis": (
            "after excluding synchronous zero-time preroll callbacks and keeping every "
            "timeline update in the session edit target, one unchanged A15-derived "
            "twelve-particle batch authored before callback 6 reports exactly 24 "
            "lockstep callbacks, finite downward motion, session-only runtime property "
            "specs, and a substantive generated isosurface"
        ),
        "falsifiers": [
            "timeline time is not zero after play and commit but before subscription",
            "post-subscription callback count is not exactly 24",
            "a callback dt or consecutive timeline step differs from 1/60 s",
            "callbacks are not one-per-session-targeted app update",
            "the twelve-particle set is not authored after callback 5 and before callback 6",
            "points, velocities, or widths has a root-layer spec at final readback",
            "twelve stable finite particle rows do not move downward",
            "the generated isosurface is missing or not substantive geometry",
            "the process log contains a PhysX GPU, CUDA, crash, or traceback marker",
        ],
        "stopping_condition": (
            "pause immediately after callback 24; no rerun, geometry tuning, or workaround"
        ),
        "renderer_exercised": False,
        "brev_exercised": False,
        "claim_limits": (
            "engine ingestion and generated-mesh probe only; no twin scene, rendering, "
            "fluid-accuracy, real-data, or sim-to-real claim"
        ),
    }


def corrected_callback_completion_failures(callback_count: int) -> list[str]:
    """Fail the A18 process unless its post-subscription callback count is exact."""

    return (
        []
        if callback_count == CORRECTED_TARGET_CALLBACK_COUNT
        else ["callback_count_incomplete"]
    )


def movement_evidence(
    *, initial: list[list[float]], current: list[list[float]]
) -> dict[str, object]:
    """Measure finite row-stable displacement and mean vertical motion."""

    count_matches = len(initial) == len(current)
    shape_valid = all(len(row) == 3 for rows in (initial, current) for row in rows)
    finite = bool(shape_valid) and all(
        math.isfinite(float(value))
        for rows in (initial, current)
        for row in rows
        for value in row
    )
    if not count_matches or not initial or not finite:
        return {
            "row_count_matches": count_matches,
            "finite_positions": finite,
            "maximum_displacement_m": None,
            "mean_displacement_m": None,
            "mean_initial_z_m": None,
            "mean_current_z_m": None,
            "motion_observed": False,
            "gravity_moved_down": False,
        }
    distances = [
        math.sqrt(
            sum((float(after[axis]) - float(before[axis])) ** 2 for axis in range(3))
        )
        for before, after in zip(initial, current)
    ]
    mean_initial_z = sum(float(row[2]) for row in initial) / len(initial)
    mean_current_z = sum(float(row[2]) for row in current) / len(current)
    maximum = max(distances)
    return {
        "row_count_matches": True,
        "finite_positions": True,
        "maximum_displacement_m": maximum,
        "mean_displacement_m": sum(distances) / len(distances),
        "mean_initial_z_m": mean_initial_z,
        "mean_current_z_m": mean_current_z,
        "motion_observed": maximum > 0.0,
        "gravity_moved_down": mean_current_z < mean_initial_z - 1.0e-7,
    }


_BAD_LOG_MARKERS = (
    "PhysX error: GPU kernel",
    "Cuda context manager error, simulation will be stopped",
    "[Error] [omni.physx.plugin]",
    "CUDA error",
    "Segmentation fault",
    "Traceback (most recent call last)",
)


def _append_once(failures: list[str], reason: str) -> None:
    if reason not in failures:
        failures.append(reason)


def _finite_number(value: object) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_timeline_evidence(
    receipt: dict[str, object], *, log_text: str
) -> list[str]:
    """Return fail-closed reasons for one completed timeline probe."""

    return _validate_timeline_evidence(
        receipt,
        log_text=log_text,
        expected_plan=build_timeline_probe_plan(seed=FROZEN_SEED),
        target_callback_count=TARGET_CALLBACK_COUNT,
        particles_per_emission=PARTICLES_PER_EMISSION,
        expected_position_source=None,
    )


def validate_corrected_timeline_evidence(
    receipt: dict[str, object], *, log_text: str
) -> list[str]:
    """Return fail-closed reasons for the separate layer-corrected A18 probe."""

    failures = _validate_timeline_evidence(
        receipt,
        log_text=log_text,
        expected_plan=build_corrected_timeline_probe_plan(seed=FROZEN_SEED),
        target_callback_count=CORRECTED_TARGET_CALLBACK_COUNT,
        particles_per_emission=CORRECTED_PARTICLES_PER_EMISSION,
        expected_position_source=(
            "a15_emission_ordinals_1_through_3_concatenated_unchanged"
        ),
    )
    initialization = receipt.get("timeline_initialization")
    if not isinstance(initialization, dict) or not (
        initialization.get("play_committed_before_subscription") is True
        and initialization.get("timeline_time_before_subscription_s") == 0.0
        and initialization.get("callbacks_before_subscription") == 0
    ):
        _append_once(failures, "timeline_subscription_initialization_mismatch")
    specs = receipt.get("runtime_layer_specs")
    expected_specs = {
        "points_session_layer_spec": True,
        "points_root_layer_spec": False,
        "velocities_session_layer_spec": True,
        "velocities_root_layer_spec": False,
        "widths_session_layer_spec": True,
        "widths_root_layer_spec": False,
        "final_edit_target_is_session": True,
    }
    if not isinstance(specs, dict) or any(
        specs.get(key) is not expected for key, expected in expected_specs.items()
    ):
        _append_once(failures, "runtime_layer_spec_mismatch")
    callbacks = receipt.get("callbacks")
    if (
        not isinstance(callbacks, list)
        or len(callbacks) != CORRECTED_TARGET_CALLBACK_COUNT
        or any(
            not isinstance(row, dict)
            or not _finite_number(row.get("timeline_time_s"))
            or abs(float(row["timeline_time_s"]) - index / FIXED_HZ) > 1.0e-6
            for index, row in enumerate(callbacks, start=1)
        )
    ):
        _append_once(failures, "timeline_callback_absolute_time_mismatch")
    return failures


def _validate_timeline_evidence(
    receipt: dict[str, object],
    *,
    log_text: str,
    expected_plan: dict[str, object],
    target_callback_count: int,
    particles_per_emission: int,
    expected_position_source: str | None,
) -> list[str]:
    """Shared strict validation while preserving the frozen A17 contract."""

    failures: list[str] = []
    if any(marker in log_text for marker in _BAD_LOG_MARKERS):
        _append_once(failures, "physx_gpu_or_cuda_runtime_error")
    if not isinstance(receipt, dict) or receipt.get("status") != "completed":
        _append_once(failures, "run_status_not_completed")
        return failures
    if receipt.get("errors") not in ([], None):
        _append_once(failures, "receipt_reports_runtime_errors")
    if receipt.get("plan") != expected_plan:
        _append_once(failures, "frozen_plan_mismatch")

    expected_settings: dict[str, object] = {
        "stage_time_codes_per_second": 60.0,
        "timeline_time_codes_per_second": 60.0,
        "timeline_target_framerate": 60.0,
        "timeline_ticks_per_frame": 1,
        "timeline_play_every_frame": True,
        "use_fixed_time_stepping": True,
        "physics_time_steps_per_second": 60,
        "physics_min_frame_rate": 60,
    }
    settings = receipt.get("settings_observed")
    if not isinstance(settings, dict) or any(
        settings.get(key) != expected for key, expected in expected_settings.items()
    ):
        _append_once(failures, "fixed_timeline_settings_mismatch")

    callbacks = receipt.get("callbacks")
    if not isinstance(callbacks, list) or len(callbacks) != target_callback_count:
        _append_once(failures, "callback_count_mismatch")
        callback_rows: list[dict[str, object]] = []
    else:
        callback_rows = [row for row in callbacks if isinstance(row, dict)]
        if len(callback_rows) != target_callback_count:
            _append_once(failures, "callback_row_invalid")
    if callback_rows:
        if [row.get("callback_index") for row in callback_rows] != list(
            range(1, target_callback_count + 1)
        ):
            _append_once(failures, "callback_index_mismatch")
        expected_dt = 1.0 / FIXED_HZ
        if any(
            not _finite_number(row.get("dt_s"))
            or abs(float(row["dt_s"]) - expected_dt) > 1.0e-6
            for row in callback_rows
        ):
            _append_once(failures, "callback_dt_mismatch")
        update_indices = [row.get("app_update_index") for row in callback_rows]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in update_indices
        ) or any(
            right != left + 1
            for left, right in zip(update_indices, update_indices[1:])
        ):
            _append_once(failures, "callback_app_update_not_lockstep")
        timeline_times = [row.get("timeline_time_s") for row in callback_rows]
        if any(not _finite_number(value) for value in timeline_times) or any(
            abs((float(right) - float(left)) - expected_dt) > 1.0e-6
            for left, right in zip(timeline_times, timeline_times[1:])
        ):
            _append_once(failures, "timeline_step_skipped_or_duplicated")

    emission = receipt.get("emission")
    expected_emission = {
        "scheduled_before_callback_index": EMISSION_CALLBACK_INDEX,
        "authored_after_callback_count": EMISSION_CALLBACK_INDEX - 1,
        "particle_count": particles_per_emission,
    }
    if expected_position_source is not None:
        expected_emission["position_source"] = expected_position_source
    emission_valid = isinstance(emission, dict) and all(
        emission.get(key) == expected for key, expected in expected_emission.items()
    )
    if callback_rows and isinstance(emission, dict):
        emission_valid = bool(
            emission_valid
            and emission.get("authored_at_app_update_index")
            == callback_rows[EMISSION_CALLBACK_INDEX - 2].get("app_update_index")
        )
    if not emission_valid:
        _append_once(failures, "emission_timing_mismatch")

    initial = receipt.get("initial_positions_m")
    current = receipt.get("current_positions_m")
    if not isinstance(initial, list) or not isinstance(current, list):
        movement = movement_evidence(initial=[], current=[])
    else:
        movement = movement_evidence(initial=initial, current=current)
    if not movement["finite_positions"]:
        _append_once(failures, "particle_positions_nonfinite")
    if len(initial or []) != particles_per_emission or not movement["row_count_matches"]:
        _append_once(failures, "particle_row_count_mismatch")
    if not movement["motion_observed"] or not movement["gravity_moved_down"]:
        _append_once(failures, "particle_motion_not_observed")

    mesh = receipt.get("mesh_final")
    if not isinstance(mesh, dict) or not (
        mesh.get("mesh_geometry_valid") is True
        and isinstance(mesh.get("mesh_point_count"), int)
        and mesh.get("mesh_point_count", 0) >= 4
        and isinstance(mesh.get("mesh_face_count"), int)
        and mesh.get("mesh_face_count", 0) > 0
        and _valid_sha256(mesh.get("mesh_sha256"))
    ):
        _append_once(failures, "substantive_mesh_not_observed")
    return failures

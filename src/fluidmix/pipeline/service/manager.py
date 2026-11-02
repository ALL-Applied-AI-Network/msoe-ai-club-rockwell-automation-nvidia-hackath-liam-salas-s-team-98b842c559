from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
import math
import os
import re
import threading
import time
import uuid

from .device import select_device
from .models import validate_create_request, validate_runtime_payload
from .sources import IncrementalFrameSource
from .contract import EVENT_TYPES, PIPELINE_COMMIT

TERMINAL = frozenset({"completed", "failed", "stopped"})


class CursorExpired(ValueError):
    def __init__(self, earliest_after: int, latest_sequence: int):
        super().__init__("event cursor expired")
        self.earliest_after = earliest_after
        self.latest_sequence = latest_sequence


class CursorAhead(ValueError):
    def __init__(self, latest_sequence: int):
        super().__init__("event cursor is ahead of stream")
        self.latest_sequence = latest_sequence


class _Session:
    def __init__(self, session_id: str, request: dict[str, object], event_retention: int):
        self.id, self.request = session_id, request
        now = datetime.now(timezone.utc).isoformat()
        self.created_at, self.updated_at = now, now
        self.status, self.frames_processed = "starting", 0
        # Replay history remains complete while the retained session exists. The
        # legacy bound is accepted for deployment-config compatibility only.
        del event_retention
        self.events: deque[dict[str, object]] = deque()
        self.next_sequence = 1
        self.terminal_sequence: int | None = None
        self.vials: dict[str, dict[str, object]] = {}
        self.decisions: deque[dict[str, object]] = deque()
        self.latest_deterministic: dict[str, dict[str, object]] = {}
        self.stop, self.condition = threading.Event(), threading.Condition()
        self.error: str | None = None
        self.worker: threading.Thread | None = None
        self.alarms: dict[str, dict[str, object]] = {}
        self.cohort_number = 0
        self.cohort_finalized = False
        self.last_rate: dict[str, object] | None = None


class SessionManager:
    def __init__(self, *, adapter_factory: Callable[[Mapping[str, object], str], object],
                 frame_source: object | None = None,
                 loaded_models: Sequence[Mapping[str, object]] = (),
                 event_retention: int = 2048, session_retention: int = 64,
                 pipeline_commit: str | None = None,
                 accepted_pipeline_commit: str | None = None,
                 development_cohort: str | None = None):
        if event_retention < 2 or session_retention < 1:
            raise ValueError("retention limits are too small")
        self._adapter_factory = adapter_factory
        self._frame_source = frame_source or IncrementalFrameSource()
        self._sessions: dict[str, _Session] = {}
        self._lock, self._model_lock = threading.Lock(), threading.Lock()
        self._loaded_models = [dict(row) for row in loaded_models]
        self._event_retention, self._session_retention = event_retention, session_retention
        self._pipeline_commit_supplied = pipeline_commit is not None
        self._actual_pipeline_commit = (
            pipeline_commit if isinstance(pipeline_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", pipeline_commit) else None
        )
        self._service_commit_attested = self._actual_pipeline_commit is not None
        self._accepted_pipeline_commit = accepted_pipeline_commit
        self._pipeline_attested = accepted_pipeline_commit == PIPELINE_COMMIT
        self._development_cohort = development_cohort

    def create(self, request: object, *, validate_video_path: bool = True) -> dict[str, object]:
        if not self._service_commit_attested or not self._pipeline_attested:
            raise RuntimeError("service and accepted pipeline commits are not attested")
        if self._development_cohort not in {"varied-v3", "varied-v4"}:
            raise RuntimeError("Varied V3/V4 development cohort is not configured")
        checked = validate_create_request(request, validate_video_path=validate_video_path)
        selection = select_device(str(checked["device"]))
        session_id = uuid.uuid4().hex
        session = _Session(session_id, {**checked, "device_selection": selection.__dict__},
                           self._event_retention)
        with self._lock:
            self._evict_terminal_locked()
            if len(self._sessions) >= self._session_retention:
                raise RuntimeError("active session capacity reached")
            self._sessions[session_id] = session
        worker = threading.Thread(target=self._run, args=(session, selection.selected),
                                  name=f"fluidmix-session-{session_id[:8]}", daemon=False)
        session.worker = worker
        worker.start()
        return self.state(session_id)

    def _evict_terminal_locked(self) -> None:
        while len(self._sessions) >= self._session_retention:
            key = next((key for key, value in self._sessions.items()
                        if value.status in TERMINAL), None)
            if key is None:
                return
            del self._sessions[key]

    def _event_row(self, session: _Session, event_type: str,
                   payload: Mapping[str, object], vial_id: str | None) -> dict[str, object]:
        if event_type not in EVENT_TYPES:
            raise ValueError("event type is outside runner-event.v1")
        validate_runtime_payload(payload)
        checked = json.loads(json.dumps(dict(payload), sort_keys=True, allow_nan=False))
        row: dict[str, object] = {
            "schema_version": "runner-event.v1",
            "event_id": f"event-{session.next_sequence}",
            "session_id": session.id,
            "sequence": session.next_sequence,
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_commit": self._actual_pipeline_commit,
            "source": SessionManager._public_source(session.request["source"], session.id),
            "type": event_type, "payload": checked,
        }
        session.next_sequence += 1
        return row

    def _emit(self, session: _Session, event_type: str, payload: Mapping[str, object],
              *, vial_id: str | None = None) -> None:
        with session.condition:
            session.events.append(self._event_row(session, event_type, payload, vial_id))
            session.condition.notify_all()

    def _finish(self, session: _Session, status: str, *, error: str | None = None) -> None:
        with session.condition:
            if session.terminal_sequence is not None:
                return
            if error is not None:
                event_type = "error"
                payload: dict[str, object] = {
                    "code": error, "message": "The source or pipeline stage failed.",
                    "stage": "session", "retryable": False, "evidence_ids": [],
                }
            else:
                event_type = "rate_health"
                payload = session.last_rate or self._rate_payload(
                    session, status="healthy", processing_fps=0.0, source_fps=None,
                )
            row = self._event_row(session, event_type, payload, None)
            session.events.append(row)
            session.error, session.status = error, status
            session.updated_at = datetime.now(timezone.utc).isoformat()
            session.terminal_sequence = int(row["sequence"])
            session.condition.notify_all()

    def _run(self, session: _Session, device: str) -> None:
        adapter, failure, desired = None, False, "completed"
        failure_code = "source_runtime_failed"
        rolling = os.environ.get("FLUIDMIX_ROLLING_COHORTS") == "1"
        cohort_ordinal = 1
        acquired = False
        while not session.stop.is_set():
            if self._model_lock.acquire(timeout=0.05):
                acquired = True
                break
        if not acquired:
            self._finish(session, "stopped")
            self._remove_owned_upload(session)
            return
        try:
            try:
                session_factory = getattr(self._adapter_factory, "for_session", None)

                def create_adapter(ordinal: int):
                    if callable(session_factory):
                        request = session.request
                        if rolling:
                            request = {**session.request, "_rolling_cohort_ordinal": ordinal}
                        return session_factory(request, device)
                    return self._adapter_factory(session.request["run_context"], device)

                adapter = create_adapter(cohort_ordinal)
                boundary_plan = self._validated_boundary_plan(adapter)
                with session.condition:
                    session.status = "running"
                    session.updated_at = datetime.now(timezone.utc).isoformat()
                source = session.request["source"]
                iterator = iter(self._frame_source.frames(source, session.stop))
                index = 0
                wall_started = time.monotonic()
                first_source_elapsed: float | None = None
                source_profile_configured = False
                source_profile: Mapping[str, object] | None = None
                pending_reference: tuple[object, float, int] | None = None
                resync_previous: object | None = None
                resync_motion_seen = False
                resync_still_frames = 0
                while True:
                    try:
                        source_item = next(iterator)
                    except StopIteration:
                        break
                    except Exception:
                        if session.stop.is_set():
                            desired = "stopped"
                            break
                        raise
                    if not isinstance(source_item, tuple) or len(source_item) != 2:
                        raise TypeError("frame source item must contain frame and time")
                    frame, elapsed = source_item
                    source_elapsed = float(elapsed)
                    if not source_profile_configured:
                        profile_reader = getattr(self._frame_source, "profile", None)
                        configurator = getattr(adapter, "configure_source_profile", None)
                        if callable(profile_reader):
                            source_profile = profile_reader(session.stop)
                        if callable(profile_reader) and callable(configurator):
                            if source_profile is None:
                                raise RuntimeError("source stream profile is unavailable")
                            configurator(source_profile)
                        source_profile_configured = True
                    if first_source_elapsed is None:
                        first_source_elapsed = source_elapsed
                    if resync_previous is not None:
                        moving = self._vial_roi_motion(
                            resync_previous, frame, source_profile
                        )
                        resync_previous = frame
                        if not resync_motion_seen:
                            resync_motion_seen = moving
                        elif moving:
                            resync_still_frames = 0
                        else:
                            resync_still_frames += 1
                            if resync_still_frames >= self._resync_still_frame_count(
                                source_profile
                            ):
                                pending_reference = (frame, source_elapsed, index)
                                resync_previous = None
                    elif adapter is None and pending_reference is not None:
                        cohort_ordinal += 1
                        adapter = create_adapter(cohort_ordinal)
                        boundary_plan = self._validated_boundary_plan(adapter)
                        configurator = getattr(adapter, "configure_source_profile", None)
                        if callable(configurator):
                            if source_profile is None:
                                raise RuntimeError("source stream profile is unavailable")
                            configurator(source_profile)
                        reference_frame, reference_elapsed, reference_index = pending_reference
                        pending_reference = None
                        try:
                            for result in adapter.process_frame(
                                reference_frame, frame_index=reference_index,
                                elapsed_seconds=reference_elapsed,
                            ):
                                self._accept_result(session, adapter, result)
                            for result in adapter.process_frame(
                                frame, frame_index=index, elapsed_seconds=source_elapsed
                            ):
                                self._accept_result(session, adapter, result)
                        except Exception as error:
                            reason = self._rolling_failure_code(error)
                            if reason is None:
                                raise
                            self._emit_abstention(
                                session, reason, rolling_cohort_ordinal=cohort_ordinal
                            )
                            self._close_adapter(adapter)
                            adapter = None
                            resync_previous = frame
                            resync_motion_seen = False
                            resync_still_frames = 0
                    elif adapter is not None:
                        try:
                            for result in adapter.process_frame(
                                frame, frame_index=index, elapsed_seconds=source_elapsed
                            ):
                                self._accept_result(session, adapter, result)
                            if index in boundary_plan:
                                checked_boundary = boundary_plan[index]
                                observed_frame_sha256 = hashlib.sha256(
                                    memoryview(frame).cast("B")
                                ).hexdigest()
                                if observed_frame_sha256 != checked_boundary["frame_sha256"]:
                                    raise RuntimeError("vial boundary frame SHA-256 mismatch")
                                closer = getattr(adapter, "close_tracker_vial", None)
                                if not callable(closer):
                                    raise RuntimeError(
                                        "provenance-bound vial boundary requires tracker close"
                                    )
                                for result in closer(checked_boundary):
                                    self._accept_result(session, adapter, result)
                        except Exception as error:
                            reason = self._rolling_failure_code(error) if rolling else None
                            if reason is None:
                                raise
                            self._emit_abstention(
                                session, reason, rolling_cohort_ordinal=cohort_ordinal
                            )
                            self._close_adapter(adapter)
                            adapter = None
                            resync_previous = frame
                            resync_motion_seen = False
                            resync_still_frames = 0
                    session.frames_processed = index + 1
                    wall_delta = max(time.monotonic() - wall_started, 1e-9)
                    source_delta = source_elapsed - first_source_elapsed
                    rate = self._rate_payload(
                        session, status="healthy",
                        processing_fps=(index + 1) / wall_delta,
                        source_fps=(index / source_delta if index and source_delta > 0 else None),
                        active_model_id=getattr(adapter, "model_id", None),
                    )
                    session.last_rate = rate
                    self._emit(session, "rate_health", rate)
                    index += 1
                    if adapter is not None and bool(
                        getattr(adapter, "session_complete", False)
                    ):
                        if not rolling:
                            break
                        self._close_adapter(adapter)
                        adapter = None
                        boundary_plan = {}
                        pending_reference = (frame, source_elapsed, index - 1)
                if adapter is not None and not session.stop.is_set() and not bool(
                    getattr(adapter, "session_complete", False)
                ):
                    end_of_stream = getattr(adapter, "end_of_stream", None)
                    if callable(end_of_stream):
                        for result in end_of_stream():
                            self._accept_result(session, adapter, result)
                if (
                    adapter is not None
                    and not session.stop.is_set()
                    and bool(getattr(adapter, "requires_complete", False))
                    and not bool(getattr(adapter, "session_complete", False))
                ):
                    reason = getattr(adapter, "eof_abstention_reason", None)
                    self._emit_abstention(
                        session, reason or "tracker_incomplete_or_disagreed",
                        rolling_cohort_ordinal=(cohort_ordinal if rolling else None),
                    )
                desired = "stopped" if session.stop.is_set() else "completed"
            except Exception as error:
                failure_code = self._pipeline_failure_code(error)
                if failure_code is None:
                    failure = True
                    failure_code = "source_runtime_failed"
                else:
                    self._emit_abstention(
                        session, failure_code,
                        rolling_cohort_ordinal=(cohort_ordinal if rolling else None),
                    )
                    desired = "completed"
            finally:
                if adapter is not None and callable(getattr(adapter, "close", None)):
                    try:
                        adapter.close()
                    except Exception:
                        failure = True
        finally:
            self._model_lock.release()
        self._finish(session, "failed", error=failure_code) if failure else self._finish(session, desired)
        self._remove_owned_upload(session)

    @staticmethod
    def _close_adapter(adapter: object) -> None:
        closer = getattr(adapter, "close", None)
        if callable(closer):
            closer()

    @classmethod
    def _rolling_failure_code(cls, error: Exception) -> str | None:
        reason = cls._pipeline_failure_code(error)
        if reason in {"tracker_stream_state_count_disagrees",
                      "tracker_stream_order_disagrees"}:
            return "unsupported_recipe_profile"
        return reason

    @staticmethod
    def _resync_still_frame_count(profile: Mapping[str, object] | None) -> int:
        if profile is None:
            raise RuntimeError("source stream profile is unavailable")
        fps = profile.get("fps")
        if (
            isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps)) or float(fps) <= 0.0
        ):
            raise RuntimeError("source stream FPS is unavailable")
        return max(1, math.ceil(float(fps) * 0.5))

    @staticmethod
    def _vial_roi_motion(previous: object, current: object,
                         profile: Mapping[str, object] | None) -> bool:
        if profile is None:
            raise RuntimeError("source stream profile is unavailable")
        resolution = profile.get("resolution")
        if (
            not isinstance(resolution, list) or len(resolution) != 2
            or any(type(value) is not int or value <= 0 for value in resolution)
        ):
            raise RuntimeError("source stream profile is invalid")
        import numpy as np
        before, after = np.asarray(previous), np.asarray(current)
        width, height = resolution
        if before.shape != after.shape or before.shape[:2] != (height, width):
            raise RuntimeError("camera view is incompatible with frozen FillCam geometry")
        x0, y0 = math.floor(283 * width / 640), math.floor(194 * height / 480)
        x1, y1 = math.ceil(338 * width / 640), math.ceil(417 * height / 480)
        difference = np.abs(
            after[y0:y1, x0:x1].astype(np.int16)
            - before[y0:y1, x0:x1].astype(np.int16)
        )
        return float(np.mean(difference)) / 255.0 > 0.02

    @staticmethod
    def _pipeline_failure_code(error: Exception) -> str | None:
        mappings = {
            "camera stream never reaches a recipe tank state": "tracker_stream_state_missing",
            "no completed stream fill stage": "tracker_stable_tail_missing",
            "camera stream states exceed recipe order": "tracker_stream_state_count_disagrees",
            "debounced camera stream states do not match recipe order": "tracker_stream_order_disagrees",
            "stable volume window is missing": "tracker_stable_volume_window_missing",
            "CNN fill height is outside frozen calibration domain": "tracker_height_outside_calibration",
            "stream Lab tank classification is tied": "tracker_tank_classification_ambiguous",
            "vial closed without complete tracker agreement": "tracker_incomplete_or_disagreed",
            "disagreed fill stage cannot enter a VLM receipt": "tracker_incomplete_or_disagreed",
            "frozen profile supports only recipe 5": "unsupported_recipe_profile",
            "frozen profile requires 640x480 RGB frames": "tracker_frame_profile_mismatch",
            "camera view is incompatible with frozen FillCam geometry": "camera_view_unvalidated",
            "source stream profile is invalid": "camera_view_unvalidated",
            "source stream FPS is unavailable": "source_frame_rate_unavailable",
            "source stream profile is unavailable": "camera_view_unvalidated",
            "measured VLM stage evidence is unavailable": "measurement_stage_unavailable",
        }
        return mappings.get(str(error))

    def _emit_abstention(self, session: _Session, reason_code: str, *,
                         rolling_cohort_ordinal: int | None = None) -> None:
        placeholder_vial_id = "unavailable"
        if rolling_cohort_ordinal is not None:
            placeholder_vial_id = f"cohort-{rolling_cohort_ordinal}:unavailable"
        position = {
            "tray_index": 0, "vial_number": placeholder_vial_id,
            "slot_label": placeholder_vial_id,
        }
        decision_id = f"decision-abstain-{session.id}"
        if rolling_cohort_ordinal is not None:
            decision_id = f"{decision_id}-c{rolling_cohort_ordinal}"
        payload = {
            "position": position, "disposition": "insufficient_evidence",
            "decision_id": decision_id, "alarm": None,
            "measurement": {
                "measurement_id": "measurement-unavailable",
                "status": "unavailable", "rgb": None, "lab": None,
                "delta_e": None, "reference_ids": [], "evidence_ids": [],
                "unavailable_reason": reason_code,
            },
            "evidence_ids": [], "unavailable_reason": reason_code,
        }
        self._emit(session, "vial_decision", payload)
        decision = {
            "type": "abstention", "vial_id": placeholder_vial_id,
            "disposition": "insufficient_evidence",
            "limit_codes": [reason_code], "evidence_citations": [],
        }
        if rolling_cohort_ordinal is not None:
            decision["cohort_ordinal"] = rolling_cohort_ordinal
        session.decisions.append(decision)

    @staticmethod
    def _validate_vial_boundary(value: object) -> dict[str, object]:
        fields = {
            "schema_version", "case_id", "vial_id", "vial_position",
            "frame_index", "frame_sha256", "source_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("vial boundary provenance fields are not closed")
        if value.get("schema_version") != "provenance-bound-vial-boundary-v1":
            raise ValueError("vial boundary provenance schema is invalid")
        if any(
            not isinstance(value.get(field), str) or not value[field]
            for field in ("case_id", "vial_id")
        ):
            raise ValueError("vial boundary identity is invalid")
        position = value.get("vial_position")
        if type(position) is not int or position <= 0:
            raise ValueError("vial boundary position is invalid")
        frame_index = value.get("frame_index")
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("vial boundary frame index is invalid")
        for role in ("frame_sha256", "source_sha256"):
            digest = value.get(role)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"vial boundary {role} is invalid")
        return dict(value)

    @classmethod
    def _validated_boundary_plan(cls, adapter: object) -> dict[int, dict[str, object]]:
        value = getattr(adapter, "vial_boundaries", None)
        if value is None:
            return {}
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("adapter vial boundary plan must be a sequence")
        checked = [cls._validate_vial_boundary(row) for row in value]
        frames = [int(row["frame_index"]) for row in checked]
        positions = [int(row["vial_position"]) for row in checked]
        if (
            frames != sorted(frames)
            or len(frames) != len(set(frames))
            or positions != list(range(1, len(checked) + 1))
            or len({str(row["case_id"]) for row in checked}) != len(checked)
            or len({str(row["vial_id"]) for row in checked}) != len(checked)
        ):
            raise ValueError("adapter vial boundary plan is duplicate or out of order")
        return {int(row["frame_index"]): row for row in checked}

    @staticmethod
    def _remove_owned_upload(session: _Session) -> None:
        if session.request["source"]["kind"] == "file":
            from pathlib import Path
            Path(str(session.request["source"]["path"])).unlink(missing_ok=True)

    @staticmethod
    def _public_source(source: Mapping[str, object], session_id: str = "source") -> dict[str, object]:
        kind = "file" if source["kind"] in {"video", "file"} else str(source["kind"])
        if kind == "file":
            from pathlib import Path
            label = str(source.get("label") or Path(str(source["path"])).name)
            return {"source_id": f"source-{session_id}", "kind": "file", "label": label}
        if kind == "v4l2":
            return {"source_id": f"source-{session_id}", "kind": "v4l2", "label": str(source["device"])}
        from urllib.parse import urlsplit, urlunsplit
        parsed = urlsplit(str(source["url"]))
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        label = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        return {"source_id": f"source-{session_id}", "kind": "rtsp", "label": label}

    def _rate_payload(self, session: _Session, *, status: str, processing_fps: float,
                      source_fps: float | None,
                      active_model_id: object = None) -> dict[str, object]:
        selection = session.request["device_selection"]
        selected, reason = selection["selected"], selection["reason"]
        if selected == "cuda":
            gpu_status, limits = "available", ["gpu_free_memory_unmeasured"]
        elif selection["requested"] in {"gpu", "auto"}:
            gpu_status, limits = "unavailable", [str(reason)]
        else:
            gpu_status, limits = "not_required", []
        with self._lock:
            queue_depth = sum(
                candidate.status == "starting" and candidate is not session
                for candidate in self._sessions.values()
            )
        return {
            "status": status, "processing_fps": processing_fps, "source_fps": source_fps,
            "queue_depth": queue_depth, "active_model_id": active_model_id,
            "gpu_free_mib": None, "gpu_status": gpu_status, "limits": limits,
        }

    @staticmethod
    def _position(result: Mapping[str, object]) -> dict[str, object]:
        vial_id = str(result["vial_id"])
        position = int(result["vial_position"])
        return {"tray_index": position - 1, "vial_number": vial_id, "slot_label": vial_id}

    @staticmethod
    def _tracker_payload(result: Mapping[str, object]) -> dict[str, object]:
        tracker = result["tracker"]
        measured = tracker.get("measured_volume_ml")
        agreement = tracker.get("agreement") is True
        unavailable_reason = (
            None if agreement and measured is not None
            else "tracker_agreement_unavailable" if not agreement
            else "tracker_volume_unavailable"
        )
        return {
            "state": "tracking" if agreement else "unavailable",
            "geometry_source": "pipeline" if agreement else "unavailable",
            "fill_fraction": None, "confidence": None,
            "tank_id": str(result["tank_id"]),
            "fill_step_id": str(result["fill_step"]),
            "measured_volume_ml": measured,
            "volume_provenance": "measured" if measured is not None else "unavailable",
            "unavailable_reason": unavailable_reason,
            "locator": {"status": "unavailable", "bbox_normalized": None,
                        "centroid_normalized": None, "evidence_ids": [],
                        "unavailable_reason": "top_down_locator_unavailable"},
        }

    @staticmethod
    def _measurement_payload(result: Mapping[str, object]) -> dict[str, object]:
        measurement = result["measurement"]
        valid = measurement.get("validity") == "valid"
        evidence_id = str(measurement.get("evidence_id") or "measurement-unavailable")
        return {
            "measurement_id": f"measurement-{evidence_id}",
            "status": "measured" if valid else "unavailable", "rgb": None,
            "lab": measurement.get("median_lab") if valid else None, "delta_e": None,
            "reference_ids": [], "evidence_ids": [evidence_id] if valid else [],
            "unavailable_reason": None if valid else "measurement_invalid",
        }

    def _accept_result(self, session: _Session, adapter: object,
                       result: Mapping[str, object]) -> None:
        validate_runtime_payload(result)
        vial_id = str(result["vial_id"])
        vial_position = int(result["vial_position"])
        closed = result.get("closed") is True
        if closed and vial_position == 1:
            session.cohort_number += 1
            session.cohort_finalized = False
            session.vials.clear()
        position = self._position(result)
        tracker_payload = self._tracker_payload(result)
        measurement_payload = self._measurement_payload(result)
        self._emit(session, "measuring", {
            "position": position, "tracker": tracker_payload,
            "measurement": measurement_payload, "stage": "measuring",
        })
        if not closed:
            return
        if vial_position in {2, 3} and (
            not any(row.get("vial_position") == 1 for row in session.vials.values())
            or session.cohort_finalized
        ):
            self._emit(session, "waiting_for_references", {
                "position": position, "tracker": tracker_payload,
                "missing_reference_types": ["clean_recipe"],
                "available_reference_ids": sorted(session.vials),
                "reason": "closed_position_1_required",
            })
            return
        for stored_id, stored in list(session.vials.items()):
            if stored.get("vial_position") == vial_position and stored_id != vial_id:
                del session.vials[stored_id]
        session.vials[vial_id] = deepcopy(dict(result))
        positions = {row.get("vial_position"): key for key, row in session.vials.items()}
        if set(positions) != {1, 2, 3}:
            self._emit(session, "waiting_for_references", {
                "position": position, "tracker": tracker_payload,
                "missing_reference_types": ["clean_recipe"],
                "available_reference_ids": sorted(session.vials),
                "reason": "later_vials_not_closed",
            })
            return
        if session.cohort_finalized:
            return
        target_vial_id = positions[1]
        target_result = session.vials[target_vial_id]
        target_position = self._position(target_result)
        decide = getattr(adapter, "decide_cohort", None)
        if not callable(decide):
            raise RuntimeError("accepted adapter does not expose causal cohort decision")
        deterministic = dict(decide(deepcopy(session.vials)))
        disposition = str(deterministic["disposition"])
        alarm = None
        if disposition == "discard":
            alarm = {"alarm_id": f"alarm-c{session.cohort_number}-{target_vial_id}",
                     "state": "unacknowledged", "acknowledged_at": None,
                     "acknowledged_by": None}
            session.alarms[str(alarm["alarm_id"])] = alarm
        evidence_ids = list(deterministic.get("evidence_citations", []))
        self._emit(session, "vial_decision", {
            "position": target_position, "disposition": disposition,
            "decision_id": f"decision-c{session.cohort_number}-{target_vial_id}",
            "alarm": alarm, "measurement": self._measurement_payload(target_result),
            "evidence_ids": evidence_ids,
            "unavailable_reason": ("deterministic_decision_unavailable"
                                   if disposition == "insufficient_evidence" else None),
        })
        deterministic_row = {
            "type": "deterministic", "vial_id": target_vial_id, **deterministic,
        }
        session.decisions.append(deterministic_row)
        session.latest_deterministic[target_vial_id] = deterministic_row
        bundle = adapter.build_vlm_bundle(deepcopy(session.vials))
        attempts = tuple(adapter.run_vlm(bundle))
        admitted_ids: list[str] = []
        admitted_attempts: list[Mapping[str, object]] = []
        all_attempts: list[Mapping[str, object]] = []
        for index, attempt in enumerate(attempts, start=1):
            all_attempts.append(attempt)
            vote_id = str(attempt.get("source_id") or f"vote-{index}")
            score = attempt.get("discard_score")
            attempt_disposition = attempt.get("disposition")
            confidence = attempt.get("confidence")
            confidence_valid = (
                isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                and math.isfinite(float(confidence)) and 0.0 <= float(confidence) <= 1.0
            )
            repository_raw, revision_raw = attempt.get("repository"), attempt.get("revision")
            identity_valid = (
                isinstance(repository_raw, str) and bool(repository_raw)
                and isinstance(revision_raw, str) and bool(revision_raw)
            )
            score_valid = (
                isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(float(score)) and 0.0 <= float(score) <= 1.0
            )
            admitted = (
                attempt.get("guard_status") == "admitted"
                and attempt_disposition in {"keep", "discard"}
                and score_valid and confidence_valid and identity_valid
            )
            if admitted:
                admitted_ids.append(vote_id)
                admitted_attempts.append(attempt)
            if attempt.get("guard_status") != "admitted":
                reason = "guard_not_admitted"
            elif attempt_disposition not in {"keep", "discard"}:
                reason = "disposition_not_voteable"
            elif not score_valid:
                reason = "invalid_score"
            elif not confidence_valid:
                reason = "invalid_confidence"
            elif not identity_valid:
                reason = "model_identity_unavailable"
            else:
                reason = "admitted"
            repository = str(repository_raw or vote_id)
            revision = str(revision_raw or "unavailable")
            evidence = attempt.get("evidence_citations", [])
            images = attempt.get("image_citations", [])
            if not isinstance(evidence, list) or not isinstance(images, list):
                evidence, images = [], []
            citations = list(dict.fromkeys([
                *(item for item in evidence if isinstance(item, str) and item),
                *(item for item in images if isinstance(item, str) and item),
            ]))
            self._emit(session, "model_vote", {
                "position": target_position, "vote_id": vote_id,
                "model": {"model_id": repository, "revision": revision,
                          "role": "vlm", "source": "inferred"},
                "admission": {"admitted": admitted,
                              "reason": reason},
                "disposition": (str(attempt_disposition) if admitted
                                else "insufficient_evidence"),
                "confidence": float(confidence) if admitted and confidence_valid else None,
                "evidence_ids": citations,
                "unavailable_reason": None if admitted else "model_vote_unavailable",
            })
            explanation = attempt.get("explanation")
            if admitted and isinstance(explanation, str) and explanation:
                self._emit(session, "explanation", {
                    "position": target_position,
                    "explanation_id": f"explanation-c{session.cohort_number}-{vote_id}",
                    "plain_language": explanation,
                    "limits": [],
                    "evidence_ids": citations,
                    "producer": {"model_id": repository, "revision": revision,
                                 "source": "inferred"},
                    "unavailable_reason": None,
                })
        det = session.latest_deterministic[target_vial_id]
        det_score = det.get("discard_score")
        det_admitted = (
            det.get("disposition") in {"keep", "discard"}
            and isinstance(det_score, (int, float)) and not isinstance(det_score, bool)
            and math.isfinite(float(det_score)) and 0.0 <= float(det_score) <= 1.0
        )
        sources = ([det] if det_admitted else []) + all_attempts
        voting_sources = ([det] if det_admitted else []) + admitted_attempts
        votes = [float(row["discard_score"]) for row in voting_sources
                 if isinstance(row.get("discard_score"), (int, float))
                 and not isinstance(row.get("discard_score"), bool)
                 and math.isfinite(float(row["discard_score"]))
                 and 0.0 <= float(row["discard_score"]) <= 1.0]
        mean = sum(votes) / len(votes) if votes else None
        final = {
            "type": "final", "vial_id": target_vial_id, "weighting": "equal",
            "valid_source_count": len(votes), "mean_discard_score": mean,
            "final_disposition": ("insufficient_evidence" if len(votes) < 2
                                  else "discard" if mean is not None and mean >= 0.5
                                  else "keep"),
            "sources": [dict(row) for row in sources],
        }
        session.decisions.append(final)
        session.cohort_finalized = True
        final_disposition = str(final["final_disposition"])
        final_alarm = None
        if final_disposition == "discard":
            final_alarm = {"alarm_id": f"alarm-final-c{session.cohort_number}-{target_vial_id}",
                           "state": "unacknowledged", "acknowledged_at": None,
                           "acknowledged_by": None}
            session.alarms[str(final_alarm["alarm_id"])] = final_alarm
        all_vote_ids = (["deterministic"] if det_admitted else []) + admitted_ids
        discard = float(mean or 0.0)
        unavailable = final_disposition == "insufficient_evidence"
        scores = ({"keep": 0.0, "check": 1.0, "discard": 0.0} if unavailable
                  else {"keep": 1.0 - discard, "check": 0.0, "discard": discard})
        self._emit(session, "averaged_decision", {
            "position": target_position,
            "decision_id": f"final-c{session.cohort_number}-{target_vial_id}",
            "method": "equal_weight", "admitted_vote_ids": all_vote_ids,
            "scores": scores,
            "disposition": final_disposition, "alarm": final_alarm,
            "evidence_ids": list(det.get("evidence_citations", [])),
            "unavailable_reason": ("fewer_than_two_valid_votes"
                                   if final_disposition == "insufficient_evidence" else None),
        })
        self._emit(session, "explanation", {
            "position": target_position,
            "explanation_id": f"explanation-c{session.cohort_number}-{target_vial_id}",
            "plain_language": f"Final vial disposition: {final_disposition}.",
            "limits": list(det.get("limit_codes", [])),
            "evidence_ids": list(det.get("evidence_citations", [])),
            "producer": {"model_id": "deterministic-template", "revision": "v1",
                         "source": "inferred"}, "unavailable_reason": None,
        })

    def _get(self, session_id: str) -> _Session:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError("unknown session") from exc

    @staticmethod
    def _window(session: _Session, after: int) -> list[dict[str, object]]:
        if after < 0:
            raise ValueError("event cursor must be nonnegative")
        earliest = int(session.events[0]["sequence"]) if session.events else session.next_sequence
        latest = session.next_sequence - 1
        if after < earliest - 1:
            raise CursorExpired(earliest - 1, latest)
        if after > latest:
            raise CursorAhead(latest)
        return [row for row in session.events if int(row["sequence"]) > after]

    def state(self, session_id: str) -> dict[str, object]:
        session = self._get(session_id)
        with session.condition:
            earliest = int(session.events[0]["sequence"]) if session.events else session.next_sequence
            state = {"starting": "created", "completed": "stopped"}.get(
                session.status, session.status
            )
            return {
                "session_id": session.id, "state": state,
                "source": self._public_source(session.request["source"], session.id),
                "run_context": deepcopy(session.request["run_context"]),
                "created_at": session.created_at, "updated_at": session.updated_at,
                "pipeline_commit": self._actual_pipeline_commit,
                "last_sequence": session.next_sequence - 1, "test_only": False,
            }

    def terminal_sequence(self, session_id: str) -> int | None:
        session = self._get(session_id)
        with session.condition:
            return session.terminal_sequence

    def worker_alive(self, session_id: str) -> bool:
        session = self._get(session_id)
        return bool(session.worker and session.worker.is_alive())

    def events(self, session_id: str, *, after: int = 0) -> list[dict[str, object]]:
        session = self._get(session_id)
        with session.condition:
            return deepcopy(self._window(session, after))

    def wait_events(self, session_id: str, *, after: int,
                    timeout: float = 15) -> list[dict[str, object]]:
        session = self._get(session_id)
        with session.condition:
            rows = self._window(session, after)
            if not rows and session.terminal_sequence is None:
                session.condition.wait(timeout)
            return deepcopy(self._window(session, after))

    def vials(self, session_id: str) -> dict[str, object]:
        return deepcopy(self._get(session_id).vials)

    def decisions(self, session_id: str) -> list[dict[str, object]]:
        return deepcopy(list(self._get(session_id).decisions))

    def acknowledge_alarm(self, session_id: str, alarm_id: str,
                          acknowledged_by: object) -> dict[str, object]:
        session = self._get(session_id)
        if not isinstance(acknowledged_by, str) or not acknowledged_by:
            raise ValueError("acknowledged_by must be nonempty text")
        with session.condition:
            if alarm_id not in session.alarms:
                raise KeyError("unknown alarm")
            alarm = session.alarms[alarm_id]
            existing = alarm.get("acknowledged_by")
            if existing is not None and existing != acknowledged_by:
                raise RuntimeError("alarm acknowledgement identity conflict")
            if existing is None:
                alarm.update({
                    "state": "acknowledged",
                    "acknowledged_at": datetime.now(timezone.utc).isoformat(),
                    "acknowledged_by": acknowledged_by,
                })
            return deepcopy(alarm)

    def stop(self, session_id: str) -> dict[str, object]:
        session = self._get(session_id)
        with session.condition:
            if session.status not in TERMINAL:
                session.status = "stopping"
                session.updated_at = datetime.now(timezone.utc).isoformat()
                session.stop.set()
        cancel = getattr(self._frame_source, "cancel", None)
        if callable(cancel):
            cancel(session.stop)
        return self.state(session_id)

    def shutdown(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            if session.status not in TERMINAL:
                session.stop.set()
                cancel = getattr(self._frame_source, "cancel", None)
                if callable(cancel):
                    cancel(session.stop)
        deadline = time.monotonic() + timeout
        for session in sessions:
            if session.worker is not None and session.worker.is_alive():
                session.worker.join(max(0.0, deadline - time.monotonic()))
            if session.worker is not None and session.worker.is_alive():
                raise RuntimeError(f"session worker did not terminate: {session.id}")

    def health(self) -> dict[str, object]:
        with self._lock:
            ids = list(self._sessions)
        active = sum(self._get(sid).status not in TERMINAL for sid in ids)
        scope_ready = self._development_cohort in {"varied-v3", "varied-v4"}
        ready = self._service_commit_attested and self._pipeline_attested and scope_ready
        limits = []
        if not self._pipeline_commit_supplied:
            limits.append("pipeline_commit_unavailable")
        elif self._actual_pipeline_commit is None:
            limits.append("pipeline_commit_malformed")
        if not self._pipeline_attested:
            limits.append("accepted_pipeline_commit_mismatch")
        if not scope_ready:
            limits.append("development_cohort_unconfigured")
        return {
            "status": "healthy" if ready else "unhealthy",
            "service": "fluidmix-pipeline-runner", "api_version": "v1",
            "pipeline_commit": self._actual_pipeline_commit, "test_only": False,
            "inference": ready, "active_sessions": active, "limits": limits,
            "frozen_profile": {
                "profile_id": "varied-v3-v4-recipe5-video-agnostic-v1",
                "accepted_pipeline_commit": PIPELINE_COMMIT,
                "resolution": [640, 480], "fps": 10.0,
                "frames_per_vial": None, "vial_count": 3, "recipe_id": "5",
                "artifact_sha256": {
                    "cnn_checkpoint": "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
                    "tracker_freeze": "255319ca4df8e668b0bceaaf211642cabcd59983a4a023163b0bfbf5a8a323ce",
                    "topology_freeze": "62e56a9e2423b09b25cf735452ebdd36adfa056bf8f8821c3fc173dbc6415a3f",
                    "vlm_freeze": "b9cf76af4588812138090ced73221c823e9ff10406265a937168c23e1e362f33",
                },
            },
        }

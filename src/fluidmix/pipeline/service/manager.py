from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
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
        self.events: deque[dict[str, object]] = deque(maxlen=event_retention)
        self.next_sequence = 1
        self.terminal_sequence: int | None = None
        self.vials: dict[str, dict[str, object]] = {}
        self.decisions: deque[dict[str, object]] = deque(maxlen=event_retention)
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
        self._pipeline_attested = self._actual_pipeline_commit == PIPELINE_COMMIT
        self._development_cohort = development_cohort

    def create(self, request: object, *, validate_video_path: bool = True) -> dict[str, object]:
        if not self._pipeline_attested:
            raise RuntimeError("approved pipeline commit is not attested")
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
                adapter = self._adapter_factory(session.request["run_context"], device)
                with session.condition:
                    session.status = "running"
                    session.updated_at = datetime.now(timezone.utc).isoformat()
                source = session.request["source"]
                iterator = iter(self._frame_source.frames(source, session.stop))
                index = 0
                wall_started = time.monotonic()
                first_source_elapsed: float | None = None
                while True:
                    try:
                        frame, elapsed = next(iterator)
                    except StopIteration:
                        break
                    except Exception:
                        if session.stop.is_set():
                            desired = "stopped"
                            break
                        raise
                    source_elapsed = float(elapsed)
                    if first_source_elapsed is None:
                        first_source_elapsed = source_elapsed
                    for result in adapter.process_frame(
                        frame, frame_index=index, elapsed_seconds=float(elapsed)
                    ):
                        self._accept_result(session, adapter, result)
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
                desired = "stopped" if session.stop.is_set() else "completed"
            except Exception:
                failure = True
            finally:
                if adapter is not None and callable(getattr(adapter, "close", None)):
                    try:
                        adapter.close()
                    except Exception:
                        failure = True
        finally:
            self._model_lock.release()
        self._finish(session, "failed", error="source_runtime_failed") if failure else self._finish(session, desired)
        self._remove_owned_upload(session)

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
        return {
            "state": "tracking" if tracker.get("agreement") else "unavailable",
            "geometry_source": "pipeline" if tracker.get("agreement") else "unavailable",
            "fill_fraction": None, "confidence": None,
            "tank_id": str(result["tank_id"]),
            "fill_step_id": str(result["fill_step"]),
            "measured_volume_ml": measured,
            "volume_provenance": "measured" if measured is not None else "unavailable",
            "unavailable_reason": None if measured is not None else "tracker_volume_unavailable",
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
        for index, attempt in enumerate(attempts, start=1):
            vote_id = str(attempt.get("source_id") or f"vote-{index}")
            score = attempt.get("discard_score")
            attempt_disposition = attempt.get("disposition")
            confidence = attempt.get("disposition_confidence")
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
                    "limits": list(attempt.get("guard_reasons", [])),
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
        sources = ([det] if det_admitted else []) + admitted_attempts
        votes = [float(row["discard_score"]) for row in sources
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
        ready = self._pipeline_attested and scope_ready
        limits = []
        if not self._pipeline_commit_supplied:
            limits.append("pipeline_commit_unavailable")
        elif self._actual_pipeline_commit is None:
            limits.append("pipeline_commit_malformed")
        elif not self._pipeline_attested:
            limits.append("pipeline_commit_mismatch")
        if not scope_ready:
            limits.append("development_cohort_unconfigured")
        return {
            "status": "healthy" if ready else "unhealthy",
            "service": "fluidmix-pipeline-runner", "api_version": "v1",
            "pipeline_commit": self._actual_pipeline_commit, "test_only": False,
            "inference": ready, "active_sessions": active, "limits": limits,
        }

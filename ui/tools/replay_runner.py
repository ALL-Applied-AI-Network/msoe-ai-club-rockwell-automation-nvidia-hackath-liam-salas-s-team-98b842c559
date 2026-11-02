#!/usr/bin/env python3
"""DEVELOPMENT replay of one recorded runner session, for UI work and demos.

Serves a recorded `runner-event.v1` SSE log, the session's `/decisions` receipts and (optionally)
the WP7 evidence bundle and its crops through the same HTTP API as the runner. It runs no model and
makes no decision: every byte it returns was recorded from a real runner session. `/api/health`
carries a `replay` disclosure so the UI shows a "Recorded replay, not live" banner.

Differences from the recorded session, all disclosed in the health `replay.note`:
- Gaps between events are capped (`--max-gap`, default 2 s) so the 17-minute VLM wait replays quickly.
- Every POST /api/sessions starts a fresh replay of the same recorded session ID; the upload is ignored.
- `/evidence-bundle` and `/evidence/images/{id}` are not live-runner endpoints yet.

Example (from the repository root):
  python3 ui/tools/replay_runner.py \
    --events outputs/.../free-explanations-e2e-20261101T173805Z/events.sse \
    --decisions outputs/.../free-explanations-e2e-20261101T173805Z/decisions.json \
    --bundle outputs/.../wp7-vlm-dev-bundle-materializer-01a18128-attempt15/bundle.json \
    --port 8775
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

IMAGE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_sse(path: Path) -> list[dict]:
    events = []
    for block in path.read_text().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
    events.sort(key=lambda e: e["sequence"])
    return events


def parse_time(value: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(value).timestamp()


class Recording:
    def __init__(self, events: list[dict], decisions: list, bundle: dict | None, images_dir: Path | None, max_gap: float):
        if not events:
            raise SystemExit("recording has no events")
        self.events = events
        self.decisions = decisions
        self.bundle = bundle
        self.images_dir = images_dir
        self.session_id = events[0]["session_id"]
        self.commit = events[0]["pipeline_commit"]
        self.source_label = events[0]["source"]["label"]
        self.recorded_at = events[0]["emitted_at"]
        # Offsets (seconds after replay start) with capped gaps.
        self.offsets = [0.0]
        for prev, cur in zip(events, events[1:]):
            gap = max(0.0, parse_time(cur["emitted_at"]) - parse_time(prev["emitted_at"]))
            self.offsets.append(self.offsets[-1] + min(gap, max_gap))
        # Receipts appear in the same order as their events: the rule receipt with the rule decision,
        # the final record (with VLM sources) only once the averaged decision has been replayed.
        self.first_decision_seq = next((e["sequence"] for e in events if e["type"] == "vial_decision"), None)
        self.first_average_seq = next((e["sequence"] for e in events if e["type"] == "averaged_decision"), None)
        self.image_ids = {i["image_id"] for i in (bundle or {}).get("images", []) if isinstance(i, dict) and "image_id" in i}
        self.started_at: float | None = None
        self.acks: dict[str, str] = {}
        self.lock = threading.Lock()

    def restart(self) -> None:
        with self.lock:
            self.started_at = time.monotonic()
            self.acks.clear()

    def available(self) -> int:
        """Number of events released so far in the current replay."""
        if self.started_at is None:
            return 0
        elapsed = time.monotonic() - self.started_at
        return sum(1 for offset in self.offsets if offset <= elapsed)

    def state(self) -> str:
        if self.started_at is None:
            return "created"
        return "completed" if self.available() >= len(self.events) else "running"


class Handler(BaseHTTPRequestHandler):
    recording: Recording

    def log_message(self, format: str, *args: object) -> None:  # quiet
        return

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin and re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "content-type, last-event-id")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")

    def _json(self, status: int, value: object) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._json(HTTPStatus.NOT_FOUND, {"error": {"code": "not_found", "message": "Route not found."}})

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def _session_doc(self) -> dict:
        rec = self.recording
        return {"session_id": rec.session_id, "state": rec.state(), "pipeline_commit": rec.commit, "test_only": False,
                "last_sequence": rec.events[rec.available() - 1]["sequence"] if rec.available() else 0,
                "source": rec.events[0]["source"]}

    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        rec = self.recording
        if parts == ["api", "health"]:
            self._json(HTTPStatus.OK, {
                "status": "healthy", "service": "fluidmix-recorded-replay (development)", "api_version": "v1",
                "pipeline_commit": rec.commit, "test_only": False, "inference": False,
                "active_sessions": 1 if rec.state() == "running" else 0, "limits": ["recorded_replay_not_live"],
                "frozen_profile": {"fps": (rec.bundle or {}).get("camera_metadata", {}).get("frame_rate_hz"),
                                    "vial_count": 3, "frames_per_vial": None,
                                    "recipe_id": (rec.bundle or {}).get("recipe_id")},
                "replay": {"recorded_session_id": rec.session_id, "recorded_service_commit": rec.commit,
                           "recorded_at": rec.recorded_at, "source_label": rec.source_label,
                           "note": "No model runs here; event gaps are shortened and the uploaded file is ignored."},
            })
            return
        if len(parts) >= 3 and parts[:2] == ["api", "sessions"]:
            if parts[2] != rec.session_id:
                self._json(HTTPStatus.NOT_FOUND, {"error": {"code": "unknown_session", "message": "Session was not found."}})
                return
            tail = parts[3:]
            if not tail:
                self._json(HTTPStatus.OK, self._session_doc())
            elif tail == ["events"]:
                try:
                    after = int(parse_qs(url.query).get("after", [self.headers.get("Last-Event-ID") or "0"])[0])
                except ValueError:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": {"code": "invalid_request", "message": "after must be an integer"}})
                    return
                self._sse(after)
            elif tail == ["decisions"]:
                last = rec.events[rec.available() - 1]["sequence"] if rec.available() else 0
                if rec.first_average_seq is not None and last >= rec.first_average_seq:
                    rows = rec.decisions
                elif rec.first_decision_seq is not None and last >= rec.first_decision_seq:
                    rows = [r for r in rec.decisions if isinstance(r, dict) and r.get("type") == "deterministic"]
                else:
                    rows = []
                self._json(HTTPStatus.OK, rows)
            elif tail == ["evidence-bundle"] and rec.bundle is not None:
                self._json(HTTPStatus.OK, rec.bundle)
            elif len(tail) == 3 and tail[:2] == ["evidence", "images"] and rec.images_dir is not None:
                image_id = tail[2].removesuffix(".png")
                path = rec.images_dir / f"{image_id}.png"
                if not IMAGE_ID.match(image_id) or image_id not in rec.image_ids or not path.is_file():
                    self._not_found()
                    return
                data = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=3600")
                self._cors()
                self.end_headers()
                self.wfile.write(data)
            else:
                self._not_found()
            return
        self._not_found()

    def _sse(self, after: int) -> None:
        rec = self.recording
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        sent = after
        last_keepalive = time.monotonic()
        try:
            while True:
                released = rec.events[: rec.available()]
                for event in released:
                    if event["sequence"] > sent:
                        payload = json.dumps(event, sort_keys=True)
                        self.wfile.write(f"id: {event['sequence']}\nevent: {event['type']}\ndata: {payload}\n\n".encode())
                        sent = event["sequence"]
                self.wfile.flush()
                if rec.state() == "completed" and sent >= rec.events[-1]["sequence"]:
                    return
                if time.monotonic() - last_keepalive > 10:
                    self.wfile.write(b": keepalive\n\n")
                    last_keepalive = time.monotonic()
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)  # upload is ignored by design
        if parts != ["api", "sessions"]:
            self._not_found()
            return
        self.recording.restart()
        rec = self.recording
        self._json(HTTPStatus.CREATED, {"session_id": rec.session_id, "test_only": False, "pipeline_commit": rec.commit, "state": "running"})

    def do_PUT(self) -> None:
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except ValueError:
            body = None
        if len(parts) == 6 and parts[:2] == ["api", "sessions"] and parts[3] == "alarms" and parts[5] == "acknowledgement":
            who = body.get("acknowledged_by") if isinstance(body, dict) else None
            if not isinstance(who, str) or not who:
                self._json(HTTPStatus.BAD_REQUEST, {"error": {"code": "invalid_request", "message": "acknowledged_by required"}})
                return
            self.recording.acks[parts[4]] = who
            self._json(HTTPStatus.OK, {"alarm_id": parts[4], "state": "acknowledged", "acknowledged_by": who})
            return
        self._not_found()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--images-dir", type=Path, help="defaults to <bundle dir>/images")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--max-gap", type=float, default=2.0)
    args = parser.parse_args(argv)
    bundle = json.loads(args.bundle.read_text()) if args.bundle else None
    images_dir = args.images_dir or (args.bundle.parent / "images" if args.bundle else None)
    Handler.recording = Recording(load_sse(args.events), json.loads(args.decisions.read_text()), bundle, images_dir, args.max_gap)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DEVELOPMENT recorded replay of session {Handler.recording.session_id} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

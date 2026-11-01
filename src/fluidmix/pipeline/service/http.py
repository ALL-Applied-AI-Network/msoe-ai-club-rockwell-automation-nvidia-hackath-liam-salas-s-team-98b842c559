from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import cgi
import importlib
import json
from pathlib import Path
import tempfile
import threading
from urllib.parse import parse_qs, urlparse

from .contract import MAX_UPLOAD_BYTES, PIPELINE_COMMIT, UPLOAD_MIME_TYPES
from .manager import CursorAhead, CursorExpired, SessionManager


DEFAULT_ORIGINS = frozenset({"http://localhost:5173", "http://127.0.0.1:5173"})


class UploadTooLarge(ValueError):
    pass


class UnsupportedUpload(ValueError):
    pass


class _Handler(BaseHTTPRequestHandler):
    server_version = "FluidMixRunner/1"

    @property
    def manager(self) -> SessionManager:
        return self.server.manager  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _origin(self) -> str | None:
        return self.headers.get("Origin")

    def _admit_origin(self) -> bool:
        origin = self._origin()
        return origin is None or origin in self.server.allowed_origins  # type: ignore[attr-defined]

    def _cors(self) -> None:
        origin = self._origin()
        if origin is not None and self._admit_origin():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _json(self, status: int, value: object) -> None:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self._cors()
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"error": {"code": code, "message": message}})

    def _parts(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urlparse(self.path)
        return [part for part in parsed.path.split("/") if part], parse_qs(parsed.query)

    def _guard_origin(self) -> bool:
        if self._admit_origin():
            return True
        self._error(HTTPStatus.FORBIDDEN, "origin_forbidden", "Origin is not admitted.")
        return False

    def do_OPTIONS(self) -> None:
        if not self._guard_origin():
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Last-Event-ID")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._guard_origin():
            return
        parts, query = self._parts()
        try:
            if parts == ["api", "health"]:
                self._json(HTTPStatus.OK, self.manager.health())
                return
            if parts == ["api", "capabilities"]:
                self._json(HTTPStatus.OK, {
                    "sources": {
                        "file_upload": {"supported": True, "max_bytes": MAX_UPLOAD_BYTES,
                                        "mime_types": list(UPLOAD_MIME_TYPES)},
                        "v4l2": {"supported": True}, "rtsp": {"supported": True},
                        "server_local_file_path": {"supported": False},
                    },
                    "events": {"transport": "sse", "replay": True},
                    "alarm_acknowledgement": True,
                })
                return
            if len(parts) < 3 or parts[:2] != ["api", "sessions"]:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
                return
            session_id = parts[2]
            if len(parts) == 3:
                self._json(HTTPStatus.OK, self.manager.state(session_id))
            elif parts[3:] == ["vials"]:
                self._json(HTTPStatus.OK, self.manager.vials(session_id))
            elif parts[3:] == ["decisions"]:
                self._json(HTTPStatus.OK, self.manager.decisions(session_id))
            elif parts[3:] == ["events"]:
                query_cursor = query.get("after", [None])
                if len(query_cursor) != 1:
                    raise ValueError("after cursor must be singular")
                query_value = query_cursor[0]
                header_value = self.headers.get("Last-Event-ID")
                if query_value is not None and header_value is not None and query_value != header_value:
                    raise ValueError("after and Last-Event-ID must agree")
                cursor_text = query_value if query_value is not None else header_value or "0"
                after = int(cursor_text)
                self._sse(session_id, after)
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "unknown_session", "Session was not found.")
        except CursorExpired as exc:
            self._json(HTTPStatus.CONFLICT, {"error": {
                "code": "replay_unavailable", "message": "Requested events expired.",
                "earliest_after": exc.earliest_after,
                "latest_sequence": exc.latest_sequence,
            }})
        except CursorAhead as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": {
                "code": "cursor_ahead", "message": "Cursor is ahead of stream.",
                "latest_sequence": exc.latest_sequence,
            }})
        except (ValueError, TypeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "Request is invalid.")

    def _sse(self, session_id: str, after: int) -> None:
        self.manager.events(session_id, after=after)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        cursor = after
        while True:
            rows = self.manager.wait_events(session_id, after=cursor, timeout=10)
            if not rows:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
            for row in rows:
                cursor = int(row["sequence"])
                encoded = json.dumps(row, sort_keys=True, allow_nan=False)
                self.wfile.write(
                    f"id: {cursor}\nevent: {row['type']}\ndata: {encoded}\n\n".encode()
                )
                self.wfile.flush()
            terminal = self.manager.terminal_sequence(session_id)
            if terminal is not None and cursor >= terminal:
                return

    def do_POST(self) -> None:
        if not self._guard_origin():
            return
        parts, _query = self._parts()
        try:
            if parts == ["api", "sessions"]:
                content_type = self.headers.get_content_type()
                if content_type == "multipart/form-data":
                    created = self._create_upload()
                elif content_type == "application/json":
                    created = self._create_live_json()
                else:
                    self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type",
                                "Use multipart/form-data or application/json.")
                    return
                self._json(HTTPStatus.CREATED, created)
                return
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "stop":
                self._json(HTTPStatus.OK, self.manager.stop(parts[2]))
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "unknown_session", "Session was not found.")
        except RuntimeError:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "runner_unavailable",
                        "Runner deployment is unavailable.")
        except UploadTooLarge:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "upload_too_large",
                        "Upload exceeds 512 MiB.")
        except UnsupportedUpload:
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type",
                        "Upload MIME type is unsupported.")
        except (ValueError, TypeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "Request is invalid.")

    def _create_live_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_048_576:
            raise ValueError("JSON body size invalid")
        raw = json.loads(self.rfile.read(length))
        if not isinstance(raw, dict) or raw.get("source", {}).get("kind") not in {"v4l2", "rtsp"}:
            raise ValueError("JSON creation supports only V4L2 or RTSP")
        return self.manager.create(raw)

    def _create_upload(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES + 1_048_576:
            raise UploadTooLarge("multipart body size invalid")
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={"REQUEST_METHOD": "POST",
                                         "CONTENT_TYPE": self.headers["Content-Type"],
                                         "CONTENT_LENGTH": str(length)},
                                keep_blank_values=True)
        if set(form.keys()) - {"file", "source_kind", "run_context"}:
            raise ValueError("multipart fields are not closed")
        upload = form["file"]
        if isinstance(upload, list) or not upload.filename:
            raise ValueError("upload is invalid")
        if upload.type not in UPLOAD_MIME_TYPES:
            raise UnsupportedUpload("upload MIME type unsupported")
        if form.getfirst("source_kind") != "file":
            raise ValueError("source_kind must be file")
        context_text = form.getfirst("run_context") or json.dumps({
            "operator_run_id": "operator-upload", "recipe_id": None, "tray_id": "tray-1"
        })
        context = json.loads(context_text)
        suffix = ".mp4" if upload.type == "video/mp4" else ".avi"
        handle = tempfile.NamedTemporaryFile(prefix="fluidmix-upload-", suffix=suffix, delete=False)
        total = 0
        try:
            with handle:
                while True:
                    chunk = upload.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise UploadTooLarge("upload exceeds limit")
                    handle.write(chunk)
            return self.manager.create({
                "source": {"kind": "file", "path": handle.name,
                           "label": Path(upload.filename).name},
                "run_context": context, "device": "cpu",
            })
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def do_PUT(self) -> None:
        if not self._guard_origin():
            return
        parts, _query = self._parts()
        try:
            if (len(parts) == 6 and parts[:2] == ["api", "sessions"]
                    and parts[3] == "alarms" and parts[5] == "acknowledgement"):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) != {"acknowledged_by"}:
                    raise ValueError
                result = self.manager.acknowledge_alarm(parts[2], parts[4],
                                                        body["acknowledged_by"])
                self._json(HTTPStatus.OK, result)
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "unknown_alarm", "Alarm was not found.")
        except RuntimeError:
            self._error(HTTPStatus.CONFLICT, "acknowledgement_conflict",
                        "Alarm was acknowledged by another operator.")
        except (ValueError, TypeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "Request is invalid.")


class RunnerHTTPServer:
    def __init__(self, address: tuple[str, int], manager: SessionManager,
                 *, allowed_origins=DEFAULT_ORIGINS):
        self._server = ThreadingHTTPServer(address, _Handler)
        self._server.manager = manager  # type: ignore[attr-defined]
        self._server.allowed_origins = frozenset(allowed_origins)  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def close(self) -> None:
        self._server.shutdown()
        self._server.manager.shutdown(timeout=5)  # type: ignore[attr-defined]
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _load_factory(value: str):
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("adapter factory must be MODULE:CALLABLE")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise ValueError("adapter factory is not callable")
    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Fluid Mixing live sessions")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--adapter-factory", required=True, metavar="MODULE:CALLABLE")
    args = parser.parse_args(argv)
    factory = _load_factory(args.adapter_factory)
    manager = SessionManager(
        adapter_factory=factory,
        pipeline_commit=getattr(factory, "pipeline_commit", None),
        development_cohort=getattr(factory, "development_cohort", None),
    )
    RunnerHTTPServer((args.host, args.port), manager).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

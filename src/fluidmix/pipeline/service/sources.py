from __future__ import annotations

from collections.abc import Iterator, Mapping
import threading
import time


class IncrementalFrameSource:
    """Decode one frame at a time; never materialize a complete video."""

    def __init__(self, *, max_frames: int | None = None, pace_file: bool = True):
        self.max_frames = max_frames
        self.pace_file = pace_file
        self._active: dict[int, object] = {}
        self._lock = threading.Lock()

    def cancel(self, stop) -> None:
        with self._lock:
            container = self._active.get(id(stop))
        if container is not None:
            container.close()

    def frames(self, source: Mapping[str, object], stop) -> Iterator[tuple[object, float]]:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required for live frame decoding") from exc
        kind = source["kind"]
        target = source.get("path") or source.get("device") or source.get("url")
        source_format = "video4linux2" if kind == "v4l2" else None
        started = time.monotonic()
        open_kwargs = {"format": source_format}
        if kind == "rtsp":
            open_kwargs["timeout"] = (5.0, 2.0)
        with av.open(str(target), **open_kwargs) as container:
            with self._lock:
                self._active[id(stop)] = container
            try:
                stream = container.streams.video[0]
                for index, decoded in enumerate(container.decode(stream)):
                    if stop.is_set() or (self.max_frames is not None and index >= self.max_frames):
                        return
                    elapsed = (
                        float(decoded.time) if decoded.time is not None
                        else time.monotonic() - started
                    )
                    if kind == "video" and self.pace_file and decoded.time is not None:
                        delay = started + elapsed - time.monotonic()
                        if delay > 0 and stop.wait(delay):
                            return
                    yield decoded.to_ndarray(format="rgb24"), elapsed
            finally:
                with self._lock:
                    self._active.pop(id(stop), None)

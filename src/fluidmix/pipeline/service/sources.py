from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import chain
import math
import threading
import time


class IncrementalFrameSource:
    """Decode one frame at a time; never materialize a complete video."""

    def __init__(self, *, max_frames: int | None = None, pace_file: bool = True):
        self.max_frames = max_frames
        self.pace_file = pace_file
        self._active: dict[int, object] = {}
        self._profiles: dict[int, dict[str, object]] = {}
        self._lock = threading.Lock()

    def profile(self, stop) -> dict[str, object] | None:
        """Return metadata read from the currently opened video stream."""
        with self._lock:
            value = self._profiles.get(id(stop))
            return None if value is None else dict(value)

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
                raw_rate = (
                    getattr(stream, "average_rate", None)
                    or getattr(stream, "guessed_rate", None)
                    or getattr(stream, "base_rate", None)
                )
                width, height = getattr(stream, "width", None), getattr(stream, "height", None)
                if (
                    type(width) is not int or type(height) is not int
                    or width <= 0 or height <= 0
                ):
                    raise RuntimeError("source stream profile is invalid")
                try:
                    fps = float(raw_rate)
                except (TypeError, ValueError, ZeroDivisionError):
                    fps = float("nan")
                decoded_iter = iter(container.decode(stream))
                pending = []
                fps_source = "decoded_stream_metadata"
                if not math.isfinite(fps) or fps <= 0.0:
                    try:
                        first = next(decoded_iter)
                    except StopIteration:
                        return
                    first_wall = time.monotonic()
                    try:
                        second = next(decoded_iter)
                    except StopIteration as error:
                        raise RuntimeError("source stream FPS is unavailable") from error
                    second_wall = time.monotonic()
                    pending = [first, second]
                    if first.time is not None and second.time is not None:
                        delta = float(second.time) - float(first.time)
                        fps_source = "decoded_frame_pts_delta"
                    else:
                        delta = second_wall - first_wall
                        fps_source = "measured_decode_arrival_delta"
                    if not math.isfinite(delta) or delta <= 0.0:
                        raise RuntimeError("source stream FPS is unavailable")
                    fps = 1.0 / delta
                with self._lock:
                    self._profiles[id(stop)] = {
                        "fps": fps, "resolution": [width, height],
                        "fps_source": fps_source,
                    }
                for index, decoded in enumerate(chain(pending, decoded_iter)):
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
                    self._profiles.pop(id(stop), None)

"""Conservative camera-local IDs for a fixed, upward-moving conveyor demo.

This module does not infer recipe identity, vial quality, or robot handoffs.
"""

from dataclasses import dataclass
from math import hypot, isfinite


@dataclass
class Track:
    """One camera-local observation history; IDs are never recycled."""

    track_id: str
    bbox: tuple[int, int, int, int]
    first_frame: int
    last_frame: int
    hits: int = 1
    state: str = "visible"
    identity_valid: bool = True
    exit_frame: int | None = None

    @property
    def center(self):
        """Return the center in source-image pixels."""
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2


class ConveyorTracker:
    """Match only unique nearby observations; uncertainty never becomes an exit.

    Args:
        entry_y: Minimum center y for a new confidently initialized track.
        exit_y: Line crossed in the upward direction to emit a zone exit.
        max_gap: Maximum frames since the last detection before identity expires.
        gate: Maximum displacement in pixels between observations.
        min_hits: Minimum observations required for a confirmed exit.
        retire_on_exit: Stop tracking at the inspection line; false follows visible
            detections beyond that line without issuing another exit.
    """

    def __init__(self, entry_y=400, exit_y=100, max_gap=15, gate=24, min_hits=3, retire_on_exit=True):
        if not all(isfinite(v) for v in (entry_y, exit_y, max_gap, gate, min_hits)):
            raise ValueError("tracker settings must be finite")
        if entry_y <= exit_y or max_gap < 1 or gate <= 0 or min_hits < 1:
            raise ValueError("invalid tracker settings")
        self.entry_y, self.exit_y = entry_y, exit_y
        self.max_gap, self.gate, self.min_hits = max_gap, gate, min_hits
        self.retire_on_exit = retire_on_exit
        self.tracks = []
        self.last_frame = -1

    def update(self, frame, boxes):
        """Return visible tracks and newly confirmed exit events.

        Frames must increase. Missing/ambiguous observations produce no guessed box.
        Detections in the middle of the lane cannot inherit a lost vial's ID.
        """
        if not isinstance(frame, int) or frame <= self.last_frame:
            raise ValueError("frames must be strictly increasing integers")
        checked = [tuple(b) for b in boxes]
        if any(len(b) != 4 or not all(isfinite(v) for v in b) or b[2] <= 0 or b[3] <= 0
               for b in checked):
            raise ValueError("boxes must have finite x, y, positive width and height")
        self.last_frame = frame
        active = []
        for track in self.tracks:
            if track.state in ("exited", "lost", "ambiguous"):
                continue
            if frame - track.last_frame > self.max_gap:
                track.state = "lost"
                track.identity_valid = False
            else:
                active.append(track)
        centers = [(x + w / 2, y + h / 2) for x, y, w, h in checked]
        edges = {i: [j for j, c in enumerate(centers)
                     if hypot(c[0] - t.center[0], c[1] - t.center[1]) <= self.gate]
                 for i, t in enumerate(active)}
        reverse = {j: [i for i in edges if j in edges[i]] for j in range(len(checked))}
        used, visible, events = set(), [], []
        for i, track in enumerate(active):
            candidates = edges[i]
            if len(candidates) > 1 or any(len(reverse[j]) > 1 for j in candidates):
                track.state, track.identity_valid = "ambiguous", False
                used.update(candidates)
                continue
            if not candidates:
                track.state = "temporarily_missing"
                continue
            j = candidates[0]
            used.add(j)
            previous_y = track.center[1]
            track.bbox, track.last_frame = checked[j], frame
            track.hits += 1
            track.state = "visible"
            visible.append(track)
            if (track.exit_frame is None and previous_y > self.exit_y >= track.center[1]
                    and track.hits >= self.min_hits and track.identity_valid):
                track.exit_frame = frame
                if self.retire_on_exit:
                    track.state = "exited"
                events.append({"track_id": track.track_id, "frame": frame,
                               "event": "left_inspection_zone", "removed": False})
        for j, box in enumerate(checked):
            if j in used or reverse[j] or centers[j][1] < self.entry_y:
                continue
            track = Track(f"CV{len(self.tracks) + 1:03d}", box, frame, frame)
            self.tracks.append(track)
            visible.append(track)
        return visible, events

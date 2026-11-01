"""Guarded conveyor tracker with mid-lane re-birth: a protocol variant, not the frozen harness tracker.

Same association, gap, ambiguity and exit rules as ConveyorGuardedTracker (the harness's v4 revision04), with
one change declared on 2026-11-01 before any scoring: a lane box that no active track can claim may start a
track anywhere in the lane, not only in the entry band. A track born above the entry band is provisional and is
neither emitted nor allowed to emit a reference-line exit until it has min_hits accepted observations; it never
inherits a lost ID, so a re-born cap is scored as a fragmentation. The frozen harness protocol stays the headline;
replaying every model's saved detections through this variant is a protocol comparison
(scripts/topdown/rebirth_tracker_sim.py, docs/topdown-cap-segmenter.md).
"""

from math import hypot, isfinite

from .conveyor_tracking import ConveyorTracker, Track


class ConveyorRebirthTracker(ConveyorTracker):
    """ConveyorGuardedTracker plus provisional mid-lane births (midlane_births=False reproduces it exactly).

    Args:
        recovery_gap: Extra frame intervals after max_gap for established motion.
            Expiry always uses the last accepted observation, never a prediction.
        ambiguity_margin: Required separation between squared normalized distances.
        initial_velocity: Optional synthetic-derived (x, y) pixels/frame prior.
            Without a prior, an unconfirmed track uses a bounded displacement envelope.
        **kwargs: Original ConveyorTracker entry, exit, gap, gate and hit settings.
            gate bounds residual distance to predicted position and velocity.
    """

    def __init__(self, recovery_gap=10, ambiguity_margin=.15,
                 initial_velocity=None, midlane_births=True, **kwargs):
        super().__init__(**kwargs)
        if (type(recovery_gap) is not int or recovery_gap < 0
                or type(self.max_gap) is not int or type(self.min_hits) is not int):
            raise ValueError("gap and hit settings must be integers; recovery_gap >= 0")
        if not isfinite(ambiguity_margin) or ambiguity_margin <= 0:
            raise ValueError("ambiguity_margin must be positive and finite")
        if initial_velocity is not None:
            initial_velocity = tuple(initial_velocity)
            if (len(initial_velocity) != 2
                    or not all(isfinite(v) for v in initial_velocity)
                    or initial_velocity[1] > 0 or hypot(*initial_velocity) > self.gate):
                raise ValueError("initial_velocity must be finite, upward and within gate")
        self.recovery_gap = recovery_gap
        self.ambiguity_margin = ambiguity_margin
        self.initial_velocity = initial_velocity
        self._velocity = {}
        self._max_moving_step = {}
        self._pending = {}
        self._startup_clipped = set()
        self.midlane_births = midlane_births
        self._provisional = set()   # mid-lane births awaiting min_hits accepted observations
        self.confirmed = []         # tracks that were ever emitted

    def _uncertain_startup_motion(self, track):
        """Require extra observations when entrance boxes share a clipped edge."""
        return (self.initial_velocity is None and
                (track.hits == 1 or (track.hits < self.min_hits and
                                     track.track_id in self._startup_clipped)))

    def _abrupt_observation(self, track, center):
        """Flag an innovation larger than the trusted object size or motion step."""
        velocity = self._velocity.get(track.track_id)
        if velocity is None:
            return False
        predicted = (track.center[0] + velocity[0], track.center[1] + velocity[1])
        # A stop/restart changes the step; localization noise is additional.
        # Using their maximum would reject a normal stop plus tiny box jitter.
        tolerance = (max(hypot(*velocity), self._max_moving_step.get(track.track_id, 0.))
                     + max(1., .5 * hypot(track.bbox[2], track.bbox[3])))
        return hypot(center[0] - predicted[0], center[1] - predicted[1]) > tolerance + 1e-9

    def update(self, frame, boxes, confidences=None):
        """Return observed tracks and crossing events, abstaining on uncertainty.

        Boxes use x, y, width, height in source pixels. Optional detector scores
        are validated for caller compatibility but do not resolve identity ties.
        Single-track ambiguity is recoverable within the original time horizon;
        a tied observation shared by two identities permanently invalidates them.
        After a gap or an abrupt consecutive jump, require two observations with
        consistent motion before restoring an ID. The first is withheld, never backfilled as a known ID.
        Gap-spanning crossings include a frame_interval, not an exact event time.
        """
        if type(frame) is not int or frame <= self.last_frame:
            raise ValueError("frames must be strictly increasing integers")
        checked = [tuple(box) for box in boxes]
        if any(len(b) != 4 or not all(isfinite(v) for v in b)
               or b[2] <= 0 or b[3] <= 0 for b in checked):
            raise ValueError("boxes must have finite x, y, positive width and height")
        if confidences is not None:
            scores = list(confidences)
            if len(scores) != len(checked) or any(
                    not isfinite(c) or not 0 <= c <= 1 for c in scores):
                raise ValueError("one finite confidence in [0,1] required per box")
        self.last_frame = frame
        active = []
        for track in self.tracks:
            if track.state in ("exited", "lost", "ambiguous"):
                continue
            horizon = self.max_gap + self.recovery_gap
            if frame - track.last_frame > horizon:
                track.state, track.identity_valid = "lost", False
                self._pending.pop(track.track_id, None)
            else:
                active.append(track)

        centers = [(x + w / 2, y + h / 2) for x, y, w, h in checked]
        edges = {}
        reverse = {j: [] for j in range(len(checked))}
        for i, track in enumerate(active):
            gap = frame - track.last_frame
            vx, vy = self._velocity.get(track.track_id, self.initial_velocity or (0., 0.))
            cold = self._uncertain_startup_motion(track) and gap > 1
            px, py = track.center[0] + vx * gap, track.center[1] + vy * gap
            edges[i] = []
            for j, (cx, cy) in enumerate(centers):
                distance = hypot(cx - px, cy - py)
                speed = hypot(cx - track.center[0], cy - track.center[1]) / gap
                # Permit sub-box localization jitter, but not sustained reverse motion.
                in_gate = (abs(cx - track.center[0]) <= self.gate if cold
                           else distance <= self.gate)
                if (in_gate and speed <= self.gate
                        and cy <= track.center[1] + track.bbox[3] / 2):
                    # Without observed velocity, all plausible candidates tie.
                    # Distance to an immature motion estimate is not identity evidence.
                    cost = 0. if cold else (distance / self.gate) ** 2
                    edges[i].append((cost, j))
                    reverse[j].append((cost, i))
            edges[i].sort()
        for candidates in reverse.values():
            candidates.sort()

        def tied(candidates):
            return (len(candidates) > 1
                    and candidates[1][0] - candidates[0][0] < self.ambiguity_margin)

        uncertain = {i for i, candidates in edges.items() if tied(candidates)}
        collisions = set()
        for candidates in reverse.values():
            if tied(candidates):
                collisions.update(i for _, i in candidates)
        blocked = {j for i in uncertain | collisions for _, j in edges[i]}
        visible, events = [], []
        for i, track in enumerate(active):
            if i in collisions:
                track.state, track.identity_valid = "ambiguous", False
                self._pending.pop(track.track_id, None)
                continue
            if i in uncertain:
                track.state, track.identity_valid = "uncertain", False
                self._pending.pop(track.track_id, None)
                continue
            candidates = edges[i]
            if not candidates:
                track.state = "temporarily_missing"
                self._pending.pop(track.track_id, None)
                continue
            _, j = candidates[0]
            if j in blocked or reverse[j][0][1] != i:
                track.state = "temporarily_missing"
                self._pending.pop(track.track_id, None)
                continue
            previous, previous_frame = track.center, track.last_frame
            gap = frame - previous_frame
            if gap > 1 or self._abrupt_observation(track, centers[j]):
                pending = self._pending.get(track.track_id)
                velocity = self._velocity.get(track.track_id, self.initial_velocity or (0., 0.))
                if self._uncertain_startup_motion(track):
                    velocity = ((centers[j][0] - previous[0]) / gap,
                                (centers[j][1] - previous[1]) / gap)
                consistent = False
                if pending is not None:
                    pending_frame, pending_center = pending
                    elapsed = frame - pending_frame
                    observed_velocity = ((centers[j][0] - pending_center[0]) / elapsed,
                                         (centers[j][1] - pending_center[1]) / elapsed)
                    # Fixed synthetic protocol: tolerate localization jitter or
                    # half the pre-gap speed, but reject a stationary impostor
                    # after a clearly moving track. A stop during occlusion may
                    # remain unresolved until motion resumes; do not guess.
                    tolerance = max(1., .5 * hypot(*velocity))
                    # Include the floating-point boundary of the declared tolerance.
                    consistent = hypot(observed_velocity[0] - velocity[0],
                                       observed_velocity[1] - velocity[1]) <= tolerance + 1e-9
                if not consistent:
                    self._pending[track.track_id] = (frame, centers[j])
                    track.state, track.identity_valid = "recovering", False
                    continue
            self._pending.pop(track.track_id, None)
            if (track.hits == 1 and gap == 1
                    and abs(track.bbox[1] + track.bbox[3] - checked[j][1] - checked[j][3]) <= 1e-9
                    and abs(track.bbox[3] - checked[j][3]) > 1e-9):
                # A fixed lower edge with changing height is clipping evidence;
                # its moving center is not yet a reliable physical velocity.
                self._startup_clipped.add(track.track_id)
            self._velocity[track.track_id] = (
                (centers[j][0] - previous[0]) / gap,
                min(0., (centers[j][1] - previous[1]) / gap),
            )
            step = hypot(*self._velocity[track.track_id])
            if step > 1e-9:
                # Retain the largest accepted step for this track's lifetime.
                # Stop-time localization jitter must not erase its moving scale.
                # The association speed gate bounds every accepted step.
                self._max_moving_step[track.track_id] = max(
                    step, self._max_moving_step.get(track.track_id, 0.))
            track.bbox, track.last_frame = checked[j], frame
            track.hits += 1
            track.state, track.identity_valid = "visible", True
            if track.track_id in self._provisional:
                if track.hits < self.min_hits:
                    continue  # confirmed only after min_hits observations; never emitted before
                self._provisional.discard(track.track_id)
            if track not in self.confirmed: self.confirmed.append(track)
            visible.append(track)
            if (track.exit_frame is None and previous[1] > self.exit_y >= track.center[1]
                    and track.hits >= self.min_hits):
                track.exit_frame = frame
                if self.retire_on_exit:
                    track.state = "exited"
                events.append({
                    "track_id": track.track_id, "frame": frame,
                    "frame_interval": [previous_frame + 1, frame],
                    "event": "left_inspection_zone", "removed": False,
                })

        # Any plausible existing association blocks a new birth, even if deferred.
        # A reappearance outside entry can only recover an unexpired existing ID.
        for j, box in enumerate(checked):
            if reverse[j]:
                continue
            if centers[j][1] < self.entry_y:
                if not self.midlane_births:
                    continue
                track = Track(f"CV{len(self.tracks) + 1:03d}", box, frame, frame)
                self.tracks.append(track)
                self._provisional.add(track.track_id)   # provisional: not emitted until confirmed
                continue
            track = Track(f"CV{len(self.tracks) + 1:03d}", box, frame, frame)
            self.tracks.append(track)
            self.confirmed.append(track)
            visible.append(track)
        return visible, events

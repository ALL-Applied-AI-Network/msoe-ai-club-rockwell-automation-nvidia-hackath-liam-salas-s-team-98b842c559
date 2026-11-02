"""Synthetic failure-path checks; these do not validate real camera measurements."""
import sys
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.carryover.decide import decide
from fluidmix.carryover.measure import measure_vials


@pytest.fixture
def captured_video(monkeypatch):
    import cv2

    def install(frames, opened=True):
        class Capture:
            def __init__(self):
                self.index = 0
                self.seeks = []
                self.released = False

            def isOpened(self):
                return opened

            def set(self, prop, value):
                self.index = value
                self.seeks.append(value)
                return True

            def read(self):
                frame = frames.get(self.index)
                self.index += 1
                return frame is not None, frame

            def release(self):
                self.released = True

        capture = Capture()
        monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
        return capture

    return install


def coloured_frame():
    frame = np.full((480, 640, 3), 120, np.uint8)
    frame[200:210, 250:260] = [40, 40, 200]  # BGR, small saturated patch
    return frame


def windows():
    return [dict(vial=v, eligible=True, start_frame=v, end_frame=v) for v in (1, 2, 3)]


def test_unmeasurable_vial_abstains_without_losing_other_vials(captured_video):
    good = coloured_frame()
    cap = captured_video({1: good, 2: np.full_like(good, 120), 3: good})
    measured = measure_vials("synthetic.mp4", windows())
    records = {v: dict(vial=v, recipe_id=6, position_in_run=v + 1, **m)
               for v, m in measured.items()}
    decide(records)
    assert cap.seeks == [1, 2, 3] and cap.released
    assert records[1]["decision"] == records[3]["decision"] == "keep"
    failed = records[2]
    assert failed["decision"] == "insufficient_evidence"
    assert "no liquid pixels selected" in failed["reason"]
    assert failed["measured_lab"] is failed["measurement_noise_dE"] is None
    assert failed["diagnostics"] is None and failed["window_frames"] == []


@pytest.mark.parametrize("fail_all", [False, True])
def test_runner_writes_abstentions_and_report(captured_video, monkeypatch, tmp_path, fail_all):
    path = Path(__file__).resolve().parents[1] / "scripts/carryover_video5.py"
    spec = importlib.util.spec_from_file_location("carryover_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    records = runner.load_records()
    good = coloured_frame()
    blank = np.full_like(good, 120)
    frames = {f: blank if fail_all or v == 5 else good
              for v, r in records.items() if r["window"] is not None
              for f in range(r["window"][0], r["window"][1] + 1)}
    cap = captured_video(frames)
    video = tmp_path / "fixture.mp4"
    video.touch()  # Only the decoder is faked; measurement, decision and reporting run normally.
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [str(path), "--video", str(video), "--out", str(out)])
    runner.main()
    saved = json.loads((out / "records.json").read_text())
    by_vial = {r["vial"]: r for r in saved["vials"]}
    assert len(by_vial) == 30 and cap.released
    assert by_vial[5]["decision"] == "insufficient_evidence"
    assert "no liquid pixels selected" in by_vial[5]["reason"]
    assert by_vial[5]["measurement_error"] == by_vial[5]["reason"]
    assert by_vial[5]["evidence"]["window_frames"] == []
    if fail_all:
        assert all(r["decision"] == "insufficient_evidence" for r in by_vial.values())
    else:
        assert by_vial[8]["decision"] == "keep"
        assert by_vial[8]["evidence"]["window_frames"]
    report = (out / "results.md").read_text()
    assert "no liquid pixels selected" in report
    assert not re.search(r"\bnan\b", report, re.I)


@pytest.mark.parametrize("bad", [
    [float("nan"), 0, 0], [float("inf"), 0, 0], [-float("inf"), 0, 0],
    [], [50], 50, [[50, 0, 0]], ["50", 0, 0], [True, 0, 0],
    np.ma.array([50, 0, 0], mask=[True, False, False]),
])
def test_invalid_colour_abstains_and_cannot_contaminate_references(bad):
    records = {
        1: dict(vial=1, recipe_id=6, position_in_run=1, measured_lab=[50, 0, 0]),
        2: dict(vial=2, recipe_id=6, position_in_run=2, measured_lab=bad),
        3: dict(vial=3, recipe_id=6, position_in_run=3, measured_lab=[50, 0, 0]),
    }
    decide(records)
    assert records[2]["decision"] == "insufficient_evidence"
    assert "invalid" in records[2]["reason"]
    assert records[2]["measured_lab"] is records[2]["dE_vs_reference"] is None
    assert records[2]["known_colour_check"] is None
    assert records[1]["decision"] == "keep" and records[1]["known_colour_check"] == "pass"
    assert records[1]["reference_lab"] == [50, 0, 0]
    assert records[3]["decision"] == "insufficient_evidence"  # No valid leave-one-out peer.
    json.dumps(records, allow_nan=False)


@pytest.mark.parametrize("bad_frame", [None, np.empty((0, 0, 3), np.uint8),
                                      np.zeros((480, 640), np.uint8),
                                      np.full((480, 640, 3), np.inf)])
def test_unusable_frame_does_not_stop_later_windows(captured_video, bad_frame):
    cap = captured_video({1: coloured_frame(), 2: bad_frame, 3: coloured_frame()})
    measured = measure_vials("synthetic.mp4", windows())
    assert measured[1]["measured_lab"] is not None
    assert measured[2]["measured_lab"] is None and measured[2]["measurement_error"]
    assert measured[3]["measured_lab"] is not None
    assert cap.seeks == [1, 2, 3] and cap.released


def test_failed_seek_abstains_instead_of_measuring_wrong_frame(captured_video, monkeypatch):
    cap = captured_video({v: coloured_frame() for v in (1, 2, 3)})
    seek = cap.set
    monkeypatch.setattr(cap, "set", lambda prop, value: False if value == 2 else seek(prop, value))
    measured = measure_vials("synthetic.mp4", windows())
    assert measured[2]["measured_lab"] is None
    assert "seek" in measured[2]["measurement_error"]
    assert measured[3]["measured_lab"] is not None and cap.released


def test_incomplete_window_is_not_reported_as_a_partial_measurement(captured_video):
    cap = captured_video({1: coloured_frame(), 3: coloured_frame()})
    measured = measure_vials("synthetic.mp4", [
        dict(vial=1, eligible=True, start_frame=1, end_frame=2),
        dict(vial=2, eligible=True, start_frame=3, end_frame=3),
    ])
    assert measured[1]["measured_lab"] is None and measured[1]["window_frames"] == []
    assert "could not read frame 2" in measured[1]["measurement_error"]
    assert measured[2]["measured_lab"] is not None and cap.released


def test_unopenable_source_is_still_fatal(captured_video):
    cap = captured_video({}, opened=False)
    with pytest.raises(FileNotFoundError):
        measure_vials("missing.mp4", windows())
    assert cap.released


def test_programming_error_is_not_hidden_as_an_abstention(captured_video, monkeypatch):
    import fluidmix.carryover.measure as measurement

    cap = captured_video({1: coloured_frame()})
    def bug(*args):
        raise RuntimeError("unexpected bug")
    monkeypatch.setattr(measurement, "lab", bug)
    with pytest.raises(RuntimeError, match="unexpected bug"):
        measure_vials("synthetic.mp4", windows())
    assert cap.released


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1), (1.5, 2), (True, 2)])
def test_invalid_window_configuration_is_still_fatal(captured_video, start, end):
    cap = captured_video({})
    with pytest.raises(ValueError, match="window frames"):
        measure_vials("synthetic.mp4", [dict(vial=1, eligible=True, start_frame=start, end_frame=end)])
    assert cap.released and cap.seeks == []


@pytest.mark.parametrize("tau", [float("nan"), float("inf"), -1, True])
def test_invalid_threshold_is_a_configuration_error(tau):
    records = {v: dict(vial=v, recipe_id=6, position_in_run=v + 1, measured_lab=[50, 0, 0])
               for v in (1, 2)}
    with pytest.raises(ValueError, match="tau"):
        decide(records, tau=tau)


def test_decoder_exception_abstains_and_attempts_later_windows(captured_video, monkeypatch):
    import cv2

    cap = captured_video({v: coloured_frame() for v in (1, 2, 3)})
    read = cap.read
    def decode():
        if cap.index == 2:
            raise cv2.error("decoder failed")
        return read()
    monkeypatch.setattr(cap, "read", decode)
    measured = measure_vials("synthetic.mp4", windows())
    assert measured[2]["measured_lab"] is None
    assert "decoder failed" in measured[2]["measurement_error"]
    assert measured[3]["measured_lab"] is not None and cap.released

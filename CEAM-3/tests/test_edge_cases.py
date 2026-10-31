"""Edge case tests for input handling, single-frame processing, zero points, and empty matrices."""

import numpy as np
import pytest

from vmt.config import TrackerConfig
from vmt.detector import MOG2Detector
from vmt.hungarian import linear_sum_assignment
from vmt.io_utils import open_video_capture, write_tracks_csv
from vmt.tracker import FrameTrackRecord, Track, TrackState, VehicleTracker


def test_empty_input_file_raises_clean_error() -> None:
    with pytest.raises(FileNotFoundError):
        open_video_capture("non_existent_video_path.mp4")


def test_single_frame_execution() -> None:
    cfg = TrackerConfig()
    tracker = VehicleTracker(config=cfg)
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)

    # Synthetic single frame with one artificial detection box
    records = tracker.step(frame, detections=[(100, 100, 50, 30)])
    assert len(records) == 1
    rec = records[0]
    assert rec.frame == 1
    assert rec.dx == 0.0
    assert rec.dy == 0.0
    assert rec.speed_px == 0.0
    assert not np.isnan(rec.cx)
    assert not np.isnan(rec.inlier_ratio)


def test_zero_tracked_points_graceful_handling() -> None:
    cfg = TrackerConfig()
    tracker = VehicleTracker(config=cfg)
    prev = np.full((200, 200), 120, dtype=np.uint8)
    curr = np.full((200, 200), 120, dtype=np.uint8)

    # Initialize a track with zero points
    track = Track(
        track_id=99,
        box=(50.0, 50.0, 40.0, 30.0),
        state=TrackState.CONFIRMED,
        points=np.empty((0, 1, 2), dtype=np.float32),
    )

    # Optical flow step must not crash, should mark lost
    tracker._track_lk_optical_flow(prev, curr, track)
    assert track.state == TrackState.LOST
    assert track.dx == 0.0
    assert track.dy == 0.0
    assert track.inlier_ratio == 0.0


def test_linear_sum_assignment_edge_cases() -> None:
    # 0x0
    r, c = linear_sum_assignment(np.empty((0, 0)))
    assert len(r) == 0 and len(c) == 0

    # 0x5
    r, c = linear_sum_assignment(np.empty((0, 5)))
    assert len(r) == 0 and len(c) == 0

    # 5x0
    r, c = linear_sum_assignment(np.empty((5, 0)))
    assert len(r) == 0 and len(c) == 0


def test_csv_nan_validation(tmp_path) -> None:
    csv_file = str(tmp_path / "test.csv")
    valid_record = FrameTrackRecord(
        frame=1,
        track_id=1,
        cx=10.0,
        cy=10.0,
        w=20.0,
        h=20.0,
        dx=1.0,
        dy=1.0,
        speed_px=1.41,
        n_points=10,
        inlier_ratio=1.0,
        state="CONFIRMED",
    )
    write_tracks_csv([valid_record], csv_file)
    assert tmp_path.joinpath("test.csv").exists()

    nan_record = FrameTrackRecord(
        frame=2,
        track_id=1,
        cx=float("nan"),
        cy=10.0,
        w=20.0,
        h=20.0,
        dx=1.0,
        dy=1.0,
        speed_px=1.41,
        n_points=10,
        inlier_ratio=1.0,
        state="CONFIRMED",
    )
    with pytest.raises(ValueError):
        write_tracks_csv([nan_record], csv_file)

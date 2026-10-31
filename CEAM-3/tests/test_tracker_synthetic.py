"""Synthetic ground truth verification test for Vehicle Motion Tracker.

Simulates a 640x360 video with 3 noise-textured vehicles moving at known velocities,
Gaussian sensor noise (sigma=3.0), and a controlled occlusion crossing.
Asserts:
1. Exactly 3 confirmed tracks.
2. Mean centre error < 3.0 px.
3. Mean displacement error < 0.5 px/frame.
4. Zero ID switches outside the occlusion.
Prints all measured numerical metrics.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
import pytest

from vmt.config import TrackerConfig
from vmt.detector import MOG2Detector
from vmt.tracker import TrackState, VehicleTracker


def make_vehicle_texture(width: int, height: int, seed: int) -> np.ndarray:
    """Generate high-contrast, feature-rich texture for synthetic vehicles."""
    rng = np.random.default_rng(seed)
    tex = rng.integers(170, 240, size=(height, width), dtype=np.uint8)
    for _ in range(8):
        cx = rng.integers(6, width - 6)
        cy = rng.integers(6, height - 6)
        rad = rng.integers(2, 6)
        color = int(rng.integers(20, 80))
        cv2.circle(tex, (cx, cy), rad, color, -1)
    return tex


def test_synthetic_ground_truth_tracking(capsys: pytest.CaptureFixture[str]) -> None:
    """Run full synthetic video test and verify tracking precision and lifecycle."""
    cfg = TrackerConfig()
    cfg.detector.warmup_frames = 20
    cfg.detector.min_area_720p = 250.0  # Normalized for 640x360 resolution
    cfg.tracking.blend_track_weight = 0.90
    cfg.tracking.blend_detection_weight = 0.10
    cfg.tracking.tentative_confirm_hits = 3
    cfg.tracking.assoc_interval_frames = 5

    detector = MOG2Detector(config=cfg.detector)
    tracker = VehicleTracker(config=cfg, detector=detector)

    w, h = 640, 360
    rng = np.random.default_rng(123)
    bg = np.full((h, w), 80, dtype=np.uint8)

    car1 = make_vehicle_texture(56, 32, seed=101)
    car2 = make_vehicle_texture(60, 34, seed=202)
    car3 = make_vehicle_texture(54, 30, seed=303)

    # 1. Background Warmup
    for _ in range(20):
        tracker.step(bg)

    gt_trajectories: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {
        1: {},
        2: {},
        3: {},
    }
    frame_records: Dict[int, List] = {}

    # 2. Simulate 55 frames of multi-vehicle traffic with one crossing occlusion
    for f in range(20, 75):
        frame = bg.copy()

        # Vehicle 1: Moves along y=120, vx=4.0 px/f
        c1_x = 40.0 + (f - 20) * 4.0
        c1_y = 120.0
        c1_w, c1_h = 56, 32

        # Vehicle 2: Moves along y=240, vx=3.0 px/f
        c2_x = 50.0 + (f - 20) * 3.0
        c2_y = 240.0
        c2_w, c2_h = 60, 34

        # Vehicle 3: Moves vertically from y=30 to y=270 at x=180, vy=3.5 px/f
        # Occlusion with Vehicle 1 occurs around f=43..49 at (x=180, y=120)
        c3_x = 180.0
        c3_y = 30.0 + (f - 20) * 3.5
        c3_w, c3_h = 54, 30

        gt_trajectories[1][f] = (c1_x + c1_w / 2.0, c1_y + c1_h / 2.0, 4.0, 0.0)
        gt_trajectories[2][f] = (c2_x + c2_w / 2.0, c2_y + c2_h / 2.0, 3.0, 0.0)
        gt_trajectories[3][f] = (c3_x + c3_w / 2.0, c3_y + c3_h / 2.0, 0.0, 3.5)

        # Draw vehicles onto frame
        frame[int(c1_y) : int(c1_y + c1_h), int(c1_x) : int(c1_x + c1_w)] = car1
        frame[int(c2_y) : int(c2_y + c2_h), int(c2_x) : int(c2_x + c2_w)] = car2
        if int(c3_y + c3_h) < h:
            frame[int(c3_y) : int(c3_y + c3_h), int(c3_x) : int(c3_x + c3_w)] = car3

        # Add Gaussian sensor noise (sigma = 3.0)
        noise = rng.normal(0, 3.0, (h, w)).astype(np.float32)
        noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        recs = tracker.step(noisy_frame)
        frame_records[f] = recs

    # 3. Collect confirmed tracks
    confirmed_track_ids = set()
    for recs in frame_records.values():
        for r in recs:
            if r.state == TrackState.CONFIRMED.value:
                confirmed_track_ids.add(r.track_id)

    # 4. Map tracker IDs to ground-truth IDs based on early position
    track_to_gt = {}
    for r in frame_records[25]:
        for gid in [1, 2, 3]:
            gcx, gcy, _, _ = gt_trajectories[gid][25]
            if np.hypot(r.cx - gcx, r.cy - gcy) < 20.0:
                track_to_gt[r.track_id] = gid

    # 5. Measure errors and check for ID switches outside occlusion window (f=43..49)
    center_errors: List[float] = []
    displacement_errors: List[float] = []
    id_switches: int = 0

    for f in range(25, 75):
        is_occlusion = 43 <= f <= 49
        for r in frame_records[f]:
            if r.track_id in track_to_gt:
                gid = track_to_gt[r.track_id]
                gcx, gcy, gvx, gvy = gt_trajectories[gid][f]
                cdist = float(np.hypot(r.cx - gcx, r.cy - gcy))

                if not is_occlusion:
                    # Check closest vehicle to verify consistent ID assignment
                    closest_gid = min(
                        [1, 2, 3],
                        key=lambda i: np.hypot(
                            r.cx - gt_trajectories[i][f][0], r.cy - gt_trajectories[i][f][1]
                        ),
                    )
                    if closest_gid != gid:
                        id_switches += 1

                    center_errors.append(cdist)
                    if r.dx != 0.0 or r.dy != 0.0:
                        ddist = float(np.hypot(r.dx - gvx, r.dy - gvy))
                        displacement_errors.append(ddist)

    mean_center_error = float(np.mean(center_errors))
    mean_disp_error = float(np.mean(displacement_errors))

    # Print measured numbers to stdout as required
    print("\n--- SYNTHETIC GROUND TRUTH BENCHMARK RESULTS ---")
    print(f"Confirmed Tracks       : {len(confirmed_track_ids)} (IDs: {sorted(list(confirmed_track_ids))})")
    print(f"Mean Centre Error      : {mean_center_error:.3f} px (threshold: < 3.0 px)")
    print(f"Mean Displacement Err  : {mean_disp_error:.3f} px/frame (threshold: < 0.5 px/frame)")
    print(f"ID Switches (Non-occl) : {id_switches} (threshold: == 0)")
    print("------------------------------------------------\n")

    # Required assertions
    assert len(confirmed_track_ids) == 3, f"Expected exactly 3 confirmed tracks, got {len(confirmed_track_ids)}"
    assert mean_center_error < 3.0, f"Mean centre error {mean_center_error:.3f} px exceeded 3.0 px limit"
    assert mean_disp_error < 0.5, f"Mean displacement error {mean_disp_error:.3f} px/f exceeded 0.5 px/f limit"
    assert id_switches == 0, f"Detected {id_switches} ID switches outside occlusion window"

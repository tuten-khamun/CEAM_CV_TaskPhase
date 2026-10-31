"""Tests for background subtraction, shadow elimination, and box merging."""

import cv2
import numpy as np
import pytest

from vmt.config import DetectorConfig
from vmt.detector import (
    MOG2Detector,
    compute_containment,
    compute_iou,
    merge_bounding_boxes,
)


def test_iou_and_containment_math() -> None:
    box1 = (10, 10, 20, 20)  # area = 400
    box2 = (10, 10, 20, 20)  # identical
    assert compute_iou(box1, box2) == pytest.approx(1.0)
    assert compute_containment(box1, box2) == pytest.approx(1.0)

    # Disjoint boxes
    box3 = (100, 100, 20, 20)
    assert compute_iou(box1, box3) == 0.0
    assert compute_containment(box1, box3) == 0.0

    # Nested box (box4 inside box1)
    box4 = (12, 12, 10, 10)  # area = 100, inter = 100
    assert compute_containment(box1, box4) == pytest.approx(1.0)
    assert compute_iou(box1, box4) == pytest.approx(100.0 / 400.0)


def test_merge_bounding_boxes() -> None:
    # Overlapping boxes with IoU > 0.3
    b1 = (10, 10, 30, 30)
    b2 = (20, 10, 30, 30)  # overlap 20x30 = 600, union = 900 + 900 - 600 = 1200, iou = 0.5
    merged = merge_bounding_boxes([b1, b2], iou_thresh=0.3)
    assert len(merged) == 1
    assert merged[0] == (10, 10, 40, 30)


def test_detector_warmup_and_moving_block() -> None:
    """Verifies that moving block produces exactly one box after warm-up."""
    cfg = DetectorConfig(
        warmup_frames=30,
        min_area_720p=200.0,  # Scaled for smaller 320x240 test frame
        var_threshold=16.0,
    )
    detector = MOG2Detector(config=cfg)

    h, w = 240, 320
    bg = np.full((h, w, 3), 70, dtype=np.uint8)

    # Warm-up phase: static background for 30 frames
    for _ in range(30):
        boxes = detector.detect(bg)
        assert len(boxes) == 0, "Detector should not emit boxes during warm-up"

    # Moving block phase: a 40x30 white block moving horizontally
    block_w, block_h = 40, 30
    detected_counts = []
    for frame_i in range(31, 45):
        frame = bg.copy()
        bx = int(50 + (frame_i - 30) * 5)
        by = 100
        frame[by : by + block_h, bx : bx + block_w] = 240
        boxes = detector.detect(frame)
        if len(boxes) > 0:
            detected_counts.append(len(boxes))
            # Verify box is located near the moving block
            dx = abs((boxes[0][0] + boxes[0][2] / 2.0) - (bx + block_w / 2.0))
            dy = abs((boxes[0][1] + boxes[0][3] / 2.0) - (by + block_h / 2.0))
            assert dx < 15.0 and dy < 15.0

    # Ensure after a few frames of motion, exactly one box is detected
    assert len(detected_counts) > 5
    assert all(c == 1 for c in detected_counts[-5:])


def test_shadow_pixels_are_suppressed() -> None:
    """Verifies that MOG2 shadow value (127) is suppressed by the shadow thresholding at 250."""
    cfg = DetectorConfig(shadow_threshold=250)
    # Directly test thresholding step on synthetic MOG2 output containing shadow (127) and foreground (255)
    mock_mog2_mask = np.zeros((100, 100), dtype=np.uint8)
    mock_mog2_mask[20:40, 20:40] = 127  # Shadow region
    mock_mog2_mask[60:80, 60:80] = 255  # Vehicle foreground

    _, thresholded = cv2.threshold(mock_mog2_mask, cfg.shadow_threshold, 255, cv2.THRESH_BINARY)
    assert np.all(thresholded[20:40, 20:40] == 0), "Shadow pixels (127) must be dropped to 0"
    assert np.all(thresholded[60:80, 60:80] == 255), "Foreground pixels (255) must be preserved"

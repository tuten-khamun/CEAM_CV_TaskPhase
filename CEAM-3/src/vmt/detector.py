"""Motion detection module using classical background subtraction and morphological filtering.

Implements MOG2 Gaussian Mixture background subtraction with shadow elimination,
morphological noise filtering, ROI polygon masking, and bounding box clustering.

Failure Modes of MOG2 Background Subtraction:
1. Stopped Vehicles: Vehicles that remain stationary (e.g. at red lights or in traffic jams)
   gradually fade into the background model as their history window elapses, causing missed detections.
2. Shadows and Illumination Shifts: Strong vehicle shadows cast onto neighboring lanes,
   headlight glare at dusk/night, and sudden sunlight shifts cause false-positive blobs.
3. Camera Ego-Motion: MOG2 strictly assumes a stationary platform; any camera shake, wind vibration,
   or panning causes global frame difference and widespread spurious detections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vmt.config import DetectorConfig

BoundingBox = Tuple[int, int, int, int]  # (x, y, w, h)


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_containment(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute containment ratio: intersection_area / min(area1, area2)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    min_area = min(w1 * h1, w2 * h2)
    if min_area <= 0:
        return 0.0
    return float(inter_area / min_area)


def merge_bounding_boxes(
    boxes: Sequence[BoundingBox],
    iou_thresh: float = 0.30,
    containment_thresh: float = 0.80,
) -> List[BoundingBox]:
    """Iteratively merge overlapping or nested bounding boxes."""
    if not boxes:
        return []

    merged = [list(b) for b in boxes]
    changed = True

    while changed:
        changed = False
        new_merged: List[List[int]] = []
        visited = [False] * len(merged)

        for i in range(len(merged)):
            if visited[i]:
                continue
            b1 = merged[i]
            visited[i] = True

            for j in range(i + 1, len(merged)):
                if visited[j]:
                    continue
                b2 = merged[j]
                t1 = (b1[0], b1[1], b1[2], b1[3])
                t2 = (b2[0], b2[1], b2[2], b2[3])

                iou = compute_iou(t1, t2)
                containment = compute_containment(t1, t2)

                if iou > iou_thresh or containment > containment_thresh:
                    # Merge b2 into b1
                    min_x = min(b1[0], b2[0])
                    min_y = min(b1[1], b2[1])
                    max_x = max(b1[0] + b1[2], b2[0] + b2[2])
                    max_y = max(b1[1] + b1[3], b2[1] + b2[3])
                    b1 = [min_x, min_y, max_x - min_x, max_y - min_y]
                    visited[j] = True
                    changed = True

            new_merged.append(b1)
        merged = new_merged

    return [(b[0], b[1], b[2], b[3]) for b in merged]


class BaseDetector(ABC):
    """Abstract interface for vehicle detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect vehicle bounding boxes in the given frame.

        Args:
            frame: BGR or grayscale image frame.

        Returns:
            List of bounding boxes (x, y, w, h).
        """
        pass

    @property
    @abstractmethod
    def debug_mask(self) -> Optional[np.ndarray]:
        """Return binary foreground mask for debug visualization."""
        pass


class MOG2Detector(BaseDetector):
    """MOG2 Gaussian Mixture Model background subtractor with morphological filtering."""

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        roi_polygon: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.cfg = config or DetectorConfig()
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32) if roi_polygon is not None else None
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=self.cfg.detect_shadows,
        )
        self.frame_idx: int = 0
        self._last_mask: Optional[np.ndarray] = None
        self._roi_mask: Optional[np.ndarray] = None

    @property
    def debug_mask(self) -> Optional[np.ndarray]:
        return self._last_mask

    def _get_roi_mask(self, h: int, w: int) -> Optional[np.ndarray]:
        if self.roi_polygon is None:
            return None
        if self._roi_mask is None or self._roi_mask.shape != (h, w):
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [self.roi_polygon], 255)
            self._roi_mask = mask
        return self._roi_mask

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        self.frame_idx += 1
        h, w = frame.shape[:2]

        # 1. Update background model
        raw_mask = self.subtractor.apply(frame)

        # 2. Suppress shadows: MOG2 marks shadows as 127; threshold at 250 drops them to 0
        _, binary_mask = cv2.threshold(
            raw_mask, self.cfg.shadow_threshold, 255, cv2.THRESH_BINARY
        )

        # 3. Optional ROI polygon restriction
        roi_mask = self._get_roi_mask(h, w)
        if roi_mask is not None:
            binary_mask = cv2.bitwise_and(binary_mask, roi_mask)

        # 4. Morphological sequence:
        # a. Median blur (ksize=5) to suppress impulse salt-and-pepper noise
        denoised = cv2.medianBlur(binary_mask, self.cfg.median_blur_ksize)

        # b. Morphological Opening (3x3 ellipse) to eliminate small disjoint spurious blobs
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.cfg.morph_open_ksize)
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_open)

        # c. Morphological Closing (7x7 ellipse) to bridge holes inside vehicle interiors
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.cfg.morph_close_ksize)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # d. Dilate (3x3 ellipse) once to reinforce vehicle boundaries
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.cfg.morph_dilate_ksize)
        cleaned_mask = cv2.dilate(closed, kernel_dilate, iterations=self.cfg.morph_dilate_iterations)

        self._last_mask = cleaned_mask

        # 5. During warm-up frames, accumulate background statistics without emitting detections
        if self.frame_idx <= self.cfg.warmup_frames:
            return []

        # 6. Extract candidate contours and convert to bounding boxes
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Area threshold scaled to frame resolution relative to 720p (1280x720 = 921600 px)
        scale_area = (w * h) / (1280.0 * 720.0)
        min_area = self.cfg.min_area_720p * scale_area
        max_area = self.cfg.max_area_ratio * (w * h)

        candidate_boxes: List[BoundingBox] = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < min_area or area > max_area:
                continue

            aspect_ratio = bw / float(bh)
            if aspect_ratio < self.cfg.min_aspect_ratio or aspect_ratio > self.cfg.max_aspect_ratio:
                continue

            candidate_boxes.append((bx, by, bw, bh))

        # 7. Merge overlapping and nested detections
        merged_boxes = merge_bounding_boxes(
            candidate_boxes,
            iou_thresh=self.cfg.merge_iou_threshold,
            containment_thresh=self.cfg.merge_containment_threshold,
        )

        return merged_boxes

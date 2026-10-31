"""Core vehicle motion tracker implementing Shi-Tomasi seeding, Lucas-Kanade optical flow,
forward-backward validation, RANSAC motion fitting, Hungarian association, and ORB re-acquisition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vmt.config import TrackerConfig
from vmt.detector import BaseDetector, BoundingBox, compute_iou
from vmt.features import (
    BaseFeatureExtractor,
    create_feature_extractor,
    match_descriptors,
    verify_matches,
)
from vmt.hungarian import linear_sum_assignment

logger = logging.getLogger(__name__)


class TrackState(str, Enum):
    """Lifecycle states of a tracked vehicle."""

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DELETED = "DELETED"


@dataclass
class Track:
    """Represents the trajectory, appearance, and optical flow state of a single vehicle."""

    track_id: int
    box: Tuple[float, float, float, float]  # (x, y, w, h)
    state: TrackState = TrackState.TENTATIVE
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 2), dtype=np.float32))
    n_initial_points: int = 0
    inlier_ratio: float = 1.0
    low_inlier_consecutive: int = 0
    dx: float = 0.0
    dy: float = 0.0
    speed_px: float = 0.0
    hits: int = 1
    consecutive_hits: int = 1
    misses: int = 0
    age: int = 1
    frames_since_reseed: int = 0
    lost_age: int = 0
    trail: List[Tuple[float, float]] = field(default_factory=list)
    template_patch: Optional[np.ndarray] = None
    template_kps: List[cv2.KeyPoint] = field(default_factory=list)
    template_desc: Optional[np.ndarray] = None
    inlier_points_prev: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    inlier_points_curr: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])


@dataclass
class FrameTrackRecord:
    """Single row record for CSV output."""

    frame: int
    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    dx: float
    dy: float
    speed_px: float
    n_points: int
    inlier_ratio: float
    state: str


class VehicleTracker:
    """Classical vehicle motion tracker coordinating optical flow, RANSAC, and lifecycle."""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        detector: Optional[BaseDetector] = None,
        feature_name: str = "orb",
    ) -> None:
        self.cfg = config or TrackerConfig()
        self.detector = detector
        self.feature_extractor: BaseFeatureExtractor = create_feature_extractor(
            feature_name, self.cfg.feature
        )

        self.tracks: Dict[int, Track] = {}
        self.next_track_id: int = 1
        self.frame_idx: int = 0
        self.prev_gray: Optional[np.ndarray] = None
        self.reacquisition_count: int = 0

    def _seed_features_in_box(
        self, gray: np.ndarray, box: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Seed Shi-Tomasi corners strictly within 15% shrunken box and extract ORB template."""
        x, y, w, h = [int(v) for v in box]
        gh, gw = gray.shape[:2]

        # Clamp box to image bounds
        x1 = max(0, min(gw - 1, x))
        y1 = max(0, min(gh - 1, y))
        x2 = max(0, min(gw, x + w))
        y2 = max(0, min(gh, y + h))

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return np.empty((0, 1, 2), dtype=np.float32), None, [], None

        patch = gray[y1:y2, x1:x2].copy()
        kps, desc = self.feature_extractor.detect_and_compute(patch)

        # Shrink by 15% per side to reject road asphalt and cast shadow boundaries
        margin_x = int(w * self.cfg.tracking.seed_box_margin)
        margin_y = int(h * self.cfg.tracking.seed_box_margin)
        sx1 = max(0, min(gw - 1, x + margin_x))
        sy1 = max(0, min(gh - 1, y + margin_y))
        sx2 = max(0, min(gw, x + w - margin_x))
        sy2 = max(0, min(gh, y + h - margin_y))

        if (sx2 - sx1) < 4 or (sy2 - sy1) < 4:
            return np.empty((0, 1, 2), dtype=np.float32), patch, kps, desc

        inner_mask = np.zeros((gh, gw), dtype=np.uint8)
        inner_mask[sy1:sy2, sx1:sx2] = 255

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.tracking.st_max_corners,
            qualityLevel=self.cfg.tracking.st_quality_level,
            minDistance=self.cfg.tracking.st_min_distance,
            blockSize=self.cfg.tracking.st_block_size,
            mask=inner_mask,
        )

        if corners is None:
            corners = np.empty((0, 1, 2), dtype=np.float32)
        else:
            corners = corners.astype(np.float32)

        return corners, patch, kps, desc

    def _reseed_track_points(self, gray: np.ndarray, track: Track) -> None:
        """Refresh feature points within current bounding box to limit drift."""
        new_corners, _, _, _ = self._seed_features_in_box(gray, track.box)
        if len(new_corners) == 0:
            return

        if len(track.points) == 0:
            track.points = new_corners
            track.n_initial_points = max(track.n_initial_points, len(new_corners))
            return

        # Keep existing points and add new points that are sufficiently separated
        existing = track.points.reshape(-1, 2)
        candidates = new_corners.reshape(-1, 2)
        merged = list(existing)
        min_dist_sq = self.cfg.tracking.st_min_distance ** 2

        for c in candidates:
            dists_sq = np.sum((np.array(merged) - c) ** 2, axis=1)
            if np.all(dists_sq >= min_dist_sq):
                merged.append(c)
                if len(merged) >= self.cfg.tracking.st_max_corners:
                    break

        track.points = np.array(merged, dtype=np.float32).reshape(-1, 1, 2)
        track.n_initial_points = max(track.n_initial_points, len(track.points))

    def _track_lk_optical_flow(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray, track: Track
    ) -> None:
        """Run pyramidal Lucas-Kanade with bidirectional forward-backward validation and RANSAC."""
        old_x, old_y, old_w, old_h = track.box
        old_center = track.center
        old_area = track.area
        gh, gw = curr_gray.shape[:2]

        if len(track.points) < 3:
            track.inlier_points_prev = np.empty((0, 2), dtype=np.float32)
            track.inlier_points_curr = np.empty((0, 2), dtype=np.float32)
            track.inlier_ratio = 0.0
            track.dx = 0.0
            track.dy = 0.0
            track.speed_px = 0.0
            track.state = TrackState.LOST
            return

        p0 = track.points
        # 1. Forward Optical Flow: prev -> curr
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.cfg.optical_flow.criteria_max_count,
            self.cfg.optical_flow.criteria_epsilon,
        )
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0,
            None,
            winSize=self.cfg.optical_flow.win_size,
            maxLevel=self.cfg.optical_flow.max_level,
            criteria=criteria,
        )

        # 2. Backward Optical Flow: curr -> prev
        p0_back, st0, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            prev_gray,
            p1,
            None,
            winSize=self.cfg.optical_flow.win_size,
            maxLevel=self.cfg.optical_flow.max_level,
            criteria=criteria,
        )

        if p1 is None or p0_back is None or st1 is None or st0 is None:
            track.state = TrackState.LOST
            return

        # 3. Forward-backward consistency check: deviation |p0 - p0_back| <= 1.0 px
        fb_dist = np.linalg.norm((p0 - p0_back).reshape(-1, 2), axis=1)
        valid = (st1.ravel() == 1) & (st0.ravel() == 1) & (fb_dist <= self.cfg.optical_flow.fb_max_distance)

        # Point boundary check
        pts_curr_flat = p1.reshape(-1, 2)
        in_bounds = (
            (pts_curr_flat[:, 0] >= 0)
            & (pts_curr_flat[:, 0] < gw)
            & (pts_curr_flat[:, 1] >= 0)
            & (pts_curr_flat[:, 1] < gh)
        )
        valid = valid & in_bounds

        survivors_p0 = p0[valid]
        survivors_p1 = p1[valid]
        n_survivors = len(survivors_p1)

        # 4. Motion Model Fitting via RANSAC
        if n_survivors >= 3:
            M, inlier_mask = cv2.estimateAffinePartial2D(
                survivors_p0,
                survivors_p1,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.feature.ransac_reproj_threshold,
                confidence=self.cfg.feature.ransac_confidence,
            )

            if M is not None and inlier_mask is not None:
                inliers_bool = inlier_mask.ravel().astype(bool)
                n_inliers = int(np.sum(inliers_bool))
                track.inlier_ratio = float(n_inliers / max(1, n_survivors))

                # Transform 4 box corners with affine matrix M
                corners = np.array(
                    [
                        [old_x, old_y],
                        [old_x + old_w, old_y],
                        [old_x + old_w, old_y + old_h],
                        [old_x, old_y + old_h],
                    ],
                    dtype=np.float32,
                ).reshape(-1, 1, 2)
                transformed_corners = cv2.transform(corners, M).reshape(-1, 2)

                new_x = float(np.min(transformed_corners[:, 0]))
                new_y = float(np.min(transformed_corners[:, 1]))
                new_w = float(np.max(transformed_corners[:, 0]) - new_x)
                new_h = float(np.max(transformed_corners[:, 1]) - new_y)

                track.points = survivors_p1[inliers_bool]
                track.inlier_points_prev = survivors_p0[inliers_bool].reshape(-1, 2)
                track.inlier_points_curr = survivors_p1[inliers_bool].reshape(-1, 2)
            else:
                # Fallback to median displacement if RANSAC fit fails but >= 3 points survive
                shifts = (survivors_p1 - survivors_p0).reshape(-1, 2)
                med_dx = float(np.median(shifts[:, 0]))
                med_dy = float(np.median(shifts[:, 1]))
                new_x = old_x + med_dx
                new_y = old_y + med_dy
                new_w = old_w
                new_h = old_h
                n_inliers = n_survivors
                track.inlier_ratio = 0.50
                track.points = survivors_p1
                track.inlier_points_prev = survivors_p0.reshape(-1, 2)
                track.inlier_points_curr = survivors_p1.reshape(-1, 2)
        else:
            n_inliers = n_survivors
            track.inlier_ratio = 0.0
            new_x, new_y, new_w, new_h = old_x, old_y, old_w, old_h
            track.points = np.empty((0, 1, 2), dtype=np.float32)
            track.inlier_points_prev = np.empty((0, 2), dtype=np.float32)
            track.inlier_points_curr = np.empty((0, 2), dtype=np.float32)

        # Update box, displacement, and speed
        track.box = (new_x, new_y, new_w, new_h)
        new_center = track.center
        track.dx = float(new_center[0] - old_center[0])
        track.dy = float(new_center[1] - old_center[1])
        track.speed_px = float(np.sqrt(track.dx ** 2 + track.dy ** 2))

        # Check inlier ratio consecutive drops
        if track.inlier_ratio < self.cfg.tracking.inlier_ratio_lost_threshold:
            track.low_inlier_consecutive += 1
        else:
            track.low_inlier_consecutive = 0

        # Check Lost Conditions
        new_area = track.area
        area_ratio = (new_area / old_area) if old_area > 0 else 999.0
        center_out_of_bounds = (
            new_center[0] < 0 or new_center[0] >= gw or new_center[1] < 0 or new_center[1] >= gh
        )
        area_exploded = area_ratio > self.cfg.tracking.max_area_change_factor or (
            area_ratio < (1.0 / self.cfg.tracking.max_area_change_factor)
        )

        if (
            n_inliers < self.cfg.tracking.min_inliers_lost
            or track.low_inlier_consecutive >= self.cfg.tracking.inlier_ratio_lost_consecutive_frames
            or center_out_of_bounds
            or area_exploded
        ):
            track.state = TrackState.LOST

    def _attempt_reacquisition(self, curr_gray: np.ndarray, track: Track) -> bool:
        """Attempt to re-acquire lost track within local search window using descriptor matching."""
        if track.template_desc is None or len(track.template_desc) < 2:
            return False

        gh, gw = curr_gray.shape[:2]
        x, y, w, h = track.box
        cx, cy = track.center

        # Search window expanded by 1.5x around last known bounding box
        win_w = int(w * self.cfg.tracking.reacq_window_scale)
        win_h = int(h * self.cfg.tracking.reacq_window_scale)
        wx1 = max(0, int(cx - win_w / 2.0))
        wy1 = max(0, int(cy - win_h / 2.0))
        wx2 = min(gw, int(cx + win_w / 2.0))
        wy2 = min(gh, int(cy + win_h / 2.0))

        if (wx2 - wx1) < 10 or (wy2 - wy1) < 10:
            return False

        search_roi = curr_gray[wy1:wy2, wx1:wx2]
        roi_kps, roi_desc = self.feature_extractor.detect_and_compute(search_roi)
        if roi_desc is None or len(roi_desc) < 2:
            return False

        matches, _ = match_descriptors(
            track.template_desc,
            roi_desc,
            norm_type=self.feature_extractor.norm_type,
            ratio_threshold=self.cfg.feature.ratio_test_threshold,
        )

        matrix, inlier_mask = verify_matches(
            track.template_kps,
            roi_kps,
            matches,
            method="affine",
            ransac_thresh=self.cfg.feature.ransac_reproj_threshold,
            confidence=self.cfg.feature.ransac_confidence,
        )

        n_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None and len(inlier_mask) > 0 else 0
        if n_inliers >= self.cfg.tracking.reacq_min_inliers and matrix is not None:
            # Re-acquisition successful!
            # Map template patch corners through matrix + search ROI offset
            tw = track.template_patch.shape[1] if track.template_patch is not None else w
            th = track.template_patch.shape[0] if track.template_patch is not None else h
            tpl_corners = np.array(
                [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32
            ).reshape(-1, 1, 2)
            transformed = cv2.transform(tpl_corners, matrix).reshape(-1, 2)
            transformed[:, 0] += wx1
            transformed[:, 1] += wy1

            nx = float(np.min(transformed[:, 0]))
            ny = float(np.min(transformed[:, 1]))
            nw = float(np.max(transformed[:, 0]) - nx)
            nh = float(np.max(transformed[:, 1]) - ny)

            track.box = (nx, ny, nw, nh)
            track.state = TrackState.CONFIRMED
            track.lost_age = 0
            track.misses = 0
            track.low_inlier_consecutive = 0
            self._reseed_track_points(curr_gray, track)
            self.reacquisition_count += 1
            logger.info("Track %d re-acquired with %d inliers at frame %d", track.track_id, n_inliers, self.frame_idx)
            return True

        return False

    def step(
        self, frame: np.ndarray, detections: Optional[Sequence[BoundingBox]] = None
    ) -> List[FrameTrackRecord]:
        """Process a single video frame, update all tracks, and emit record statistics.

        Args:
            frame: Current BGR or grayscale image frame.
            detections: Pre-computed detections, or None to run internal detector.

        Returns:
            List of FrameTrackRecord for all active/lost tracks in the current frame.
        """
        self.frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        # Phase 1: Optical flow tracking for existing active tracks
        if self.prev_gray is not None:
            for track in list(self.tracks.values()):
                if track.state in (TrackState.TENTATIVE, TrackState.CONFIRMED):
                    self._track_lk_optical_flow(self.prev_gray, curr_gray, track)
                    track.age += 1
                    track.frames_since_reseed += 1

                    # Periodic point re-seeding to counter drift
                    if track.state != TrackState.LOST:
                        pts_dropped = len(track.points) < (
                            self.cfg.tracking.reseed_points_ratio * track.n_initial_points
                        )
                        interval_due = (
                            track.frames_since_reseed >= self.cfg.tracking.reseed_interval_frames
                        )
                        if pts_dropped or interval_due:
                            self._reseed_track_points(curr_gray, track)
                            track.frames_since_reseed = 0

                elif track.state == TrackState.LOST:
                    track.lost_age += 1
                    track.dx = 0.0
                    track.dy = 0.0
                    track.speed_px = 0.0
                    track.inlier_points_prev = np.empty((0, 2), dtype=np.float32)
                    track.inlier_points_curr = np.empty((0, 2), dtype=np.float32)

                    if track.lost_age <= self.cfg.tracking.reacq_max_lost_age:
                        self._attempt_reacquisition(curr_gray, track)
                    else:
                        track.state = TrackState.DELETED

        # Always run detection on every frame so MOG2 background subtractor
        # updates its continuous Gaussian mixture statistics and debug mask
        current_detections: Sequence[BoundingBox] = []
        if detections is not None:
            current_detections = detections
        elif self.detector is not None:
            current_detections = self.detector.detect(frame)

        # Phase 2: Hungarian Association (every 5 frames or frame 1)
        run_assoc = (
            (self.frame_idx == 1)
            or (self.frame_idx % self.cfg.tracking.assoc_interval_frames == 0)
        )
        if run_assoc:
            # Active candidate tracks for association
            active_ids = [
                tid
                for tid, t in self.tracks.items()
                if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)
            ]

            if len(active_ids) > 0 and len(current_detections) > 0:
                cost_matrix = np.zeros((len(active_ids), len(current_detections)), dtype=float)
                for i, tid in enumerate(active_ids):
                    tbox = self.tracks[tid].box
                    for j, dbox in enumerate(current_detections):
                        iou = compute_iou(tbox, dbox)
                        cost_matrix[i, j] = 1.0 - iou

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                matched_tracks = set()
                matched_dets = set()

                for r, c in zip(row_ind, col_ind):
                    iou = 1.0 - cost_matrix[r, c]
                    if iou >= self.cfg.tracking.assoc_min_iou:
                        tid = active_ids[r]
                        t = self.tracks[tid]
                        dbox = current_detections[c]

                        # Blend matched boxes (70% track / 30% detection) to correct drift
                        bw_t = self.cfg.tracking.blend_track_weight
                        bw_d = self.cfg.tracking.blend_detection_weight
                        bx = bw_t * t.box[0] + bw_d * dbox[0]
                        by = bw_t * t.box[1] + bw_d * dbox[1]
                        bw = bw_t * t.box[2] + bw_d * dbox[2]
                        bh = bw_t * t.box[3] + bw_d * dbox[3]
                        t.box = (bx, by, bw, bh)

                        t.hits += 1
                        t.consecutive_hits += 1
                        t.misses = 0
                        matched_tracks.add(tid)
                        matched_dets.add(c)

                        # Promote tentative track after 3 hits
                        if t.state == TrackState.TENTATIVE and t.consecutive_hits >= self.cfg.tracking.tentative_confirm_hits:
                            t.state = TrackState.CONFIRMED

                # Unmatched tracks receive a miss
                for tid in active_ids:
                    if tid not in matched_tracks:
                        t = self.tracks[tid]
                        t.misses += 1
                        t.consecutive_hits = 0
                        if t.misses >= self.cfg.tracking.max_misses_delete:
                            t.state = TrackState.DELETED

                # Unmatched detections spawn tentative tracks
                for j, dbox in enumerate(current_detections):
                    if j not in matched_dets:
                        self._create_track(curr_gray, dbox)

            elif len(current_detections) > 0:
                # No active tracks: all detections spawn tentative tracks
                for dbox in current_detections:
                    self._create_track(curr_gray, dbox)

            elif len(active_ids) > 0:
                # No detections: all active tracks receive a miss
                for tid in active_ids:
                    t = self.tracks[tid]
                    t.misses += 1
                    t.consecutive_hits = 0
                    if t.misses >= self.cfg.tracking.max_misses_delete:
                        t.state = TrackState.DELETED

            # Deduplication: if two tracks overlap with IoU > 0.6, keep the older one
            self._deduplicate_tracks()

        # Update trails and build output records
        records: List[FrameTrackRecord] = []
        for tid, track in list(self.tracks.items()):
            if track.state == TrackState.DELETED:
                continue

            cx, cy = track.center
            track.trail.append((cx, cy))
            if len(track.trail) > self.cfg.viz.trail_length:
                track.trail.pop(0)

            records.append(
                FrameTrackRecord(
                    frame=self.frame_idx,
                    track_id=track.track_id,
                    cx=float(round(cx, 2)),
                    cy=float(round(cy, 2)),
                    w=float(round(track.box[2], 2)),
                    h=float(round(track.box[3], 2)),
                    dx=float(round(track.dx, 2)),
                    dy=float(round(track.dy, 2)),
                    speed_px=float(round(track.speed_px, 2)),
                    n_points=len(track.points),
                    inlier_ratio=float(round(track.inlier_ratio, 3)),
                    state=track.state.value,
                )
            )

        # Purge deleted tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.state != TrackState.DELETED}
        self.prev_gray = curr_gray
        return records

    def _create_track(self, gray: np.ndarray, box: BoundingBox) -> Track:
        """Initialize a new tentative track with Shi-Tomasi corners and template descriptors."""
        float_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        corners, patch, kps, desc = self._seed_features_in_box(gray, float_box)

        new_track = Track(
            track_id=self.next_track_id,
            box=float_box,
            state=TrackState.TENTATIVE,
            points=corners,
            n_initial_points=len(corners),
            template_patch=patch,
            template_kps=kps,
            template_desc=desc,
        )
        self.tracks[self.next_track_id] = new_track
        self.next_track_id += 1
        return new_track

    def _deduplicate_tracks(self) -> None:
        """Cull younger duplicate tracks when two active tracks have IoU > 0.6."""
        active = [
            t for t in self.tracks.values() if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)
        ]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                t1, t2 = active[i], active[j]
                if t1.state == TrackState.DELETED or t2.state == TrackState.DELETED:
                    continue
                iou = compute_iou(t1.box, t2.box)
                if iou > self.cfg.tracking.dedup_iou_threshold:
                    # Keep older track (lower track_id or greater age)
                    if t1.age >= t2.age:
                        t2.state = TrackState.DELETED
                    else:
                        t1.state = TrackState.DELETED

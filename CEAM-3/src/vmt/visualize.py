"""Visualization module for rendering vehicle trajectories, flow arrows, HUD, and masks."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from vmt.config import VisualizationConfig
from vmt.tracker import Track, TrackState


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a deterministic, vibrant BGR color for a given track ID using golden ratio HSV."""
    golden_ratio = 0.618033988749895
    hue = int(((track_id * golden_ratio) % 1.0) * 180)  # OpenCV hue range [0, 180)
    hsv_pixel = np.uint8([[[hue, 220, 240]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def draw_outlined_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    outline_thickness: int = 3,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Render high-contrast text with a dark outline for clean readability over complex backgrounds."""
    # Outer dark stroke
    cv2.putText(
        img,
        text,
        org,
        font_face,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA,
    )
    # Inner colored stroke
    cv2.putText(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


class Visualizer:
    """Renders tracks, motion trails, optical flow vectors, and HUD onto video frames."""

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self.cfg = config or VisualizationConfig()

    def render(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Track],
        frame_idx: int,
        fps: float,
        detector_name: str = "MOG2",
        feature_name: str = "ORB",
        debug_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render complete overlay onto frame, returning annotated image (or side-by-side if debug).

        Args:
            frame: Raw BGR input frame.
            tracks: Dictionary of active/lost Track instances.
            frame_idx: Current frame index.
            fps: Measured execution frames per second.
            detector_name: Name of detection algorithm.
            feature_name: Name of feature matching algorithm.
            debug_mask: Optional binary foreground mask.

        Returns:
            Annotated BGR image frame (or side-by-side comparison if debug_mask is provided).
        """
        vis = frame.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # 1. Render motion trails (fading trajectory of center points)
        for track in tracks.values():
            if track.state == TrackState.DELETED or len(track.trail) < 2:
                continue
            color = get_track_color(track.track_id)
            n_pts = len(track.trail)
            for i in range(1, n_pts):
                pt1 = (int(track.trail[i - 1][0]), int(track.trail[i - 1][1]))
                pt2 = (int(track.trail[i][0]), int(track.trail[i][1]))
                # Alpha weighting based on trail age
                weight = float(i) / float(n_pts)
                faded_color = (
                    int(color[0] * weight),
                    int(color[1] * weight),
                    int(color[2] * weight),
                )
                thick = max(1, int(2 * weight))
                cv2.line(vis, pt1, pt2, faded_color, thick, cv2.LINE_AA)

        # 2. Render optical flow vectors for inlier points (magnified 3x)
        for track in tracks.values():
            if track.state != TrackState.CONFIRMED:
                continue
            if len(track.inlier_points_prev) > 0 and len(track.inlier_points_curr) > 0:
                p_prev = track.inlier_points_prev
                p_curr = track.inlier_points_curr
                scale = self.cfg.flow_arrow_scale

                for p0, p1 in zip(p_prev, p_curr):
                    x0, y0 = int(p1[0]), int(p1[1])
                    flow_x = (p1[0] - p0[0]) * scale
                    flow_y = (p1[1] - p0[1]) * scale
                    x_end = int(x0 + flow_x)
                    y_end = int(y0 + flow_y)

                    # Draw keypoint dot
                    cv2.circle(vis, (x0, y0), 2, (0, 255, 255), -1, cv2.LINE_AA)
                    # Draw magnified flow arrow if motion > 0.5 px
                    if (flow_x ** 2 + flow_y ** 2) > 0.5:
                        cv2.arrowedLine(
                            vis,
                            (x0, y0),
                            (x_end, y_end),
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                            0,
                            0.3,
                        )

        # 3. Render bounding boxes and ID/speed badges
        for track in tracks.values():
            if track.state == TrackState.DELETED:
                continue

            x, y, w, h = [int(v) for v in track.box]
            color = get_track_color(track.track_id)
            # Dashed or faded box if lost/tentative, solid if confirmed
            box_color = color if track.state == TrackState.CONFIRMED else (120, 120, 120)
            thickness = self.cfg.box_thickness if track.state == TrackState.CONFIRMED else 1

            cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, thickness=thickness)

            # Badge text
            badge = f"ID:{track.track_id} {track.state.value[:4]}"
            speed_str = f"{track.speed_px:.1f} px/f"
            draw_outlined_text(
                vis,
                badge,
                (x, max(15, y - 6)),
                font_scale=0.45,
                color=(255, 255, 255),
                thickness=1,
                outline_thickness=3,
            )
            draw_outlined_text(
                vis,
                speed_str,
                (x, y + h + 14),
                font_scale=0.40,
                color=(200, 255, 200),
                thickness=1,
                outline_thickness=3,
            )

        # 4. Render Head-Up Display (HUD)
        n_confirmed = sum(1 for t in tracks.values() if t.state == TrackState.CONFIRMED)
        n_active = sum(
            1 for t in tracks.values() if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)
        )

        hud_lines = [
            f"Frame: {frame_idx:04d}",
            f"Tracks (Conf/Active): {n_confirmed} / {n_active}",
            f"Throughput: {fps:.1f} FPS",
            f"Detector: {detector_name} | Matcher: {feature_name}",
        ]

        y_offset = 25
        for line in hud_lines:
            draw_outlined_text(
                vis,
                line,
                (15, y_offset),
                font_scale=self.cfg.hud_font_scale,
                color=(255, 255, 255),
                thickness=self.cfg.hud_thickness,
                outline_thickness=self.cfg.hud_outline_thickness,
            )
            y_offset += 24

        # 5. Debug mode: Stack mask side-by-side if requested
        if debug_mask is not None:
            if len(debug_mask.shape) == 2:
                mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_bgr = debug_mask.copy()
            # Overlay a small HUD badge on mask
            draw_outlined_text(
                mask_bgr,
                "MOG2 Foreground Mask",
                (15, 25),
                font_scale=0.6,
                color=(0, 255, 255),
                thickness=1,
                outline_thickness=3,
            )
            vis = np.hstack([vis, mask_bgr])

        return vis

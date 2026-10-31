"""Synthetic daylight traffic video generator for testing classical tracking pipelines.

Renders a 1280x720 30 FPS multi-lane highway scene with realistic asphalt texture,
painted lane markings, vehicles (> 50 px wide) with structural details (windshields,
grilles, lights, wheels), cast shadows (luminance ~125), and varying velocities.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np


class SyntheticVehicle:
    """Represents a simulated vehicle with structured visual components and motion dynamics."""

    def __init__(
        self,
        color_bgr: Tuple[int, int, int],
        width: int,
        height: int,
        y_lane: int,
        speed_px: float,
        start_frame: int,
        vehicle_type: str = "sedan",
    ) -> None:
        self.color = color_bgr
        self.w = width
        self.h = height
        self.y = y_lane
        self.speed = speed_px
        self.start_frame = start_frame
        self.type = vehicle_type

    def get_position(self, frame_idx: int) -> Tuple[float, float]:
        """Compute top-left coordinates at the given frame index."""
        dx = (frame_idx - self.start_frame) * self.speed
        x = -self.w + dx
        return x, float(self.y)

    def draw(self, canvas: np.ndarray, x: int, y: int) -> None:
        """Draw vehicle body, details, and cast shadow onto canvas."""
        ch, cw = canvas.shape[:2]
        w, h = self.w, self.h

        # 1. Cast shadow (darkening beneath and slightly to bottom-right, luminance ~120)
        shadow_poly = np.array(
            [
                [x + 10, y + h - 2],
                [x + w + 12, y + h - 2],
                [x + w + 20, y + h + 14],
                [x + 18, y + h + 14],
            ],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [shadow_poly], (50, 50, 50))
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        # 2. Main vehicle chassis
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.color, -1)
        # Metallic outline
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 40, 40), 2)

        # 3. Structural details for abundant corner features:
        # Front/rear windshields and roof
        roof_inset_x = int(w * 0.22)
        roof_inset_y = int(h * 0.15)
        roof_w = int(w * 0.56)
        roof_h = int(h * 0.70)
        # Dark cabin glass
        cv2.rectangle(
            canvas,
            (x + roof_inset_x, y + roof_inset_y),
            (x + roof_inset_x + roof_w, y + roof_inset_y + roof_h),
            (30, 35, 40),
            -1,
        )
        # Colored roof center
        cv2.rectangle(
            canvas,
            (x + roof_inset_x + 8, y + roof_inset_y + 4),
            (x + roof_inset_x + roof_w - 8, y + roof_inset_y + roof_h - 4),
            [max(0, c - 30) for c in self.color],
            -1,
        )

        # Headlights (right side: vehicle traveling left-to-right)
        cv2.rectangle(canvas, (x + w - 4, y + 3), (x + w, y + 9), (200, 255, 255), -1)
        cv2.rectangle(canvas, (x + w - 4, y + h - 9), (x + w, y + h - 3), (200, 255, 255), -1)

        # Taillights (left side)
        cv2.rectangle(canvas, (x, y + 3), (x + 4, y + 9), (30, 30, 220), -1)
        cv2.rectangle(canvas, (x, y + h - 9), (x + 4, y + h - 3), (30, 30, 220), -1)

        # Wheels / tires
        tire_w, tire_h = 14, 5
        cv2.rectangle(canvas, (x + int(w * 0.18), y - 2), (x + int(w * 0.18) + tire_w, y + 3), (20, 20, 20), -1)
        cv2.rectangle(canvas, (x + int(w * 0.70), y - 2), (x + int(w * 0.70) + tire_w, y + 3), (20, 20, 20), -1)
        cv2.rectangle(canvas, (x + int(w * 0.18), y + h - 3), (x + int(w * 0.18) + tire_w, y + h + 2), (20, 20, 20), -1)
        cv2.rectangle(canvas, (x + int(w * 0.70), y + h - 3), (x + int(w * 0.70) + tire_w, y + h + 2), (20, 20, 20), -1)


def generate_synthetic_traffic_video(
    output_path: str = "data/traffic.mp4",
    num_frames: int = 480,  # 16 seconds @ 30 FPS
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    seed: int = 42,
) -> None:
    """Generate high-fidelity synthetic highway video."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rng = np.random.default_rng(seed)

    # Highway geometry
    road_top = 180
    road_bottom = 560
    road_height = road_bottom - road_top
    lane_height = road_height // 3

    # Define vehicle schedules
    vehicles: List[SyntheticVehicle] = [
        # Lane 1 (top lane, y=210)
        SyntheticVehicle(color_bgr=(180, 50, 40), width=90, height=44, y_lane=205, speed_px=6.5, start_frame=35),
        SyntheticVehicle(color_bgr=(50, 160, 210), width=85, height=42, y_lane=205, speed_px=7.0, start_frame=190),
        SyntheticVehicle(color_bgr=(200, 200, 205), width=92, height=45, y_lane=205, speed_px=6.2, start_frame=330),
        # Lane 2 (middle lane, y=330)
        SyntheticVehicle(color_bgr=(40, 140, 60), width=88, height=44, y_lane=335, speed_px=5.2, start_frame=40),
        SyntheticVehicle(color_bgr=(210, 180, 50), width=95, height=46, y_lane=335, speed_px=5.5, start_frame=160),
        SyntheticVehicle(color_bgr=(60, 60, 190), width=88, height=43, y_lane=335, speed_px=5.4, start_frame=290),
        # Lane 3 (bottom lane, y=460)
        SyntheticVehicle(color_bgr=(190, 190, 190), width=115, height=52, y_lane=460, speed_px=4.2, start_frame=32, vehicle_type="van"),
        SyntheticVehicle(color_bgr=(45, 45, 50), width=90, height=44, y_lane=465, speed_px=4.5, start_frame=170),
        SyntheticVehicle(color_bgr=(160, 90, 50), width=96, height=46, y_lane=465, speed_px=4.3, start_frame=310),
    ]

    # Pre-render static road background
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Grassy embankments
    bg[0:road_top, :] = (45, 95, 40)
    bg[road_bottom:height, :] = (45, 95, 40)
    # Asphalt roadbed
    bg[road_top:road_bottom, :] = (65, 68, 72)

    # Road shoulders (solid white lines)
    cv2.line(bg, (0, road_top + 4), (width, road_top + 4), (220, 220, 220), 3)
    cv2.line(bg, (0, road_bottom - 4), (width, road_bottom - 4), (220, 220, 220), 3)

    # Dashed white lane dividers
    dash_len = 35
    dash_gap = 35
    for lane_divider_y in [road_top + lane_height, road_top + 2 * lane_height]:
        x_curr = 0
        while x_curr < width:
            cv2.line(bg, (x_curr, lane_divider_y), (min(width, x_curr + dash_len), lane_divider_y), (220, 220, 220), 2)
            x_curr += dash_len + dash_gap

    # Add realistic subtle asphalt texture
    road_noise = rng.normal(0, 3.5, (road_height, width, 3)).astype(np.float32)
    bg[road_top:road_bottom, :] = np.clip(
        bg[road_top:road_bottom, :].astype(np.float32) + road_noise, 0, 255
    ).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create VideoWriter at '{output_path}'")

    print(f"Generating synthetic traffic video: {num_frames} frames -> '{output_path}'...")
    try:
        for f in range(num_frames):
            frame = bg.copy()

            # Render active vehicles
            for v in vehicles:
                vx, vy = v.get_position(f)
                if -v.w <= vx <= width:
                    v.draw(frame, int(round(vx)), int(round(vy)))

            # Camera sensor noise
            sensor_noise = rng.normal(0, 1.8, (height, width, 3)).astype(np.float32)
            noisy_frame = np.clip(frame.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)
            writer.write(noisy_frame)
    finally:
        writer.release()

    print(f"Successfully generated '{output_path}' ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    generate_synthetic_traffic_video()

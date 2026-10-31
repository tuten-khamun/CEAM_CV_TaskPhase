"""
Script to generate synthetic test road images.

Generates realistic perspective highway scenes across:
- Multiple resolutions: 640x360, 1280x720, 1920x1080, 800x800 (square)
- Multiple file formats: JPG, JPEG, PNG, BMP
- Multiple lane marking colors: Yellow solid + White dashed, Dual White
- Multiple lighting conditions: Bright daylight, dusk/dim lighting
- Edge/negative cases: Blank road with no markings, single lane marking only
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def draw_perspective_line(
    image: np.ndarray,
    start_pt: Tuple[float, float],
    end_pt: Tuple[float, float],
    color_bgr: Tuple[int, int, int],
    thickness_bottom: int,
    thickness_top: int,
    dashed: bool = False,
    dash_count: int = 6,
):
    """Draws a perspective-tapered lane line (solid or dashed) onto the image."""
    x1, y1 = start_pt
    x2, y2 = end_pt

    if not dashed:
        pts = [
            [int(x1 - thickness_bottom / 2), int(y1)],
            [int(x1 + thickness_bottom / 2), int(y1)],
            [int(x2 + thickness_top / 2), int(y2)],
            [int(x2 - thickness_top / 2), int(y2)],
        ]
        cv2.fillPoly(image, [np.array(pts, dtype=np.int32)], color_bgr)
        return

    for i in range(dash_count):
        t_start = (i + 0.25) / dash_count
        t_end = (i + 0.75) / dash_count

        dy_s = y1 * (1 - t_start) + y2 * t_start
        dx_s = x1 * (1 - t_start) + x2 * t_start

        dy_e = y1 * (1 - t_end) + y2 * t_end
        dx_e = x1 * (1 - t_end) + x2 * t_end

        w_s = thickness_bottom * (1 - t_start) + thickness_top * t_start
        w_e = thickness_bottom * (1 - t_end) + thickness_top * t_end

        pts = [
            [int(dx_s - w_s / 2), int(dy_s)],
            [int(dx_s + w_s / 2), int(dy_s)],
            [int(dx_e + w_e / 2), int(dy_e)],
            [int(dx_e - w_e / 2), int(dy_e)],
        ]
        cv2.fillPoly(image, [np.array(pts, dtype=np.int32)], color_bgr)


def create_synthetic_road_image(
    width: int,
    height: int,
    lane_style: str = "standard_yellow_white",
    lighting: str = "daylight",
) -> np.ndarray:
    """Creates a synthetic perspective road image for testing.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        lane_style: 'standard_yellow_white', 'dual_white', 'single_right_only', or 'blank_no_markings'.
        lighting: 'daylight' or 'dim_dusk'.

    Returns:
        np.ndarray: BGR image array.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    horizon_y = int(height * 0.45)
    vanishing_x = int(width * 0.5)

    if lighting == "daylight":
        sky_top = np.array([240, 180, 120], dtype=np.float32)
        sky_bottom = np.array([210, 210, 180], dtype=np.float32)
        asphalt_base = (70, 70, 70)
        grass_color = (45, 100, 40)
    else:
        sky_top = np.array([90, 50, 40], dtype=np.float32)
        sky_bottom = np.array([120, 80, 70], dtype=np.float32)
        asphalt_base = (45, 45, 45)
        grass_color = (25, 55, 25)

    # Render sky gradient
    for y in range(horizon_y):
        t = y / max(1, horizon_y)
        color = (1 - t) * sky_top + t * sky_bottom
        img[y, :] = color.astype(np.uint8)

    # Render grass background below horizon
    img[horizon_y:, :] = grass_color

    # Road polygon
    road_pts = np.array(
        [
            [int(width * 0.05), height],
            [int(vanishing_x - width * 0.08), horizon_y],
            [int(vanishing_x + width * 0.08), horizon_y],
            [int(width * 0.95), height],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [road_pts], asphalt_base)

    # Add asphalt noise
    noise = np.random.normal(0, 3, (height - horizon_y, width)).astype(np.int16)
    road_slice = img[horizon_y:, :].astype(np.int16) + noise[:, :, None]
    img[horizon_y:, :] = np.clip(road_slice, 0, 255).astype(np.uint8)

    # Calculate line thicknesses
    thickness_bottom = max(6, int(width * 0.015))
    thickness_top = max(2, int(width * 0.003))

    left_bottom = (width * 0.22, float(height))
    left_top = (vanishing_x - width * 0.04, horizon_y + height * 0.12)

    right_bottom = (width * 0.78, float(height))
    right_top = (vanishing_x + width * 0.04, horizon_y + height * 0.12)

    if lighting == "daylight":
        yellow_bgr = (20, 215, 240)
        white_bgr = (245, 245, 245)
    else:
        yellow_bgr = (15, 175, 195)
        white_bgr = (195, 195, 195)

    if lane_style == "standard_yellow_white":
        draw_perspective_line(
            img, left_bottom, left_top, yellow_bgr, thickness_bottom, thickness_top, dashed=False
        )
        draw_perspective_line(
            img, right_bottom, right_top, white_bgr, thickness_bottom, thickness_top, dashed=True
        )
    elif lane_style == "dual_white":
        draw_perspective_line(
            img, left_bottom, left_top, white_bgr, thickness_bottom, thickness_top, dashed=False
        )
        draw_perspective_line(
            img, right_bottom, right_top, white_bgr, thickness_bottom, thickness_top, dashed=True
        )
    elif lane_style == "single_right_only":
        draw_perspective_line(
            img, right_bottom, right_top, white_bgr, thickness_bottom, thickness_top, dashed=True
        )
    elif lane_style == "blank_no_markings":
        pass

    return img


def generate_all_sample_images(output_folder: Path = Path("sample_images")) -> None:
    """Generates all synthetic benchmark images into output_folder."""
    output_folder.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    test_cases = [
        ("highway_720p_standard.jpg", 1280, 720, "standard_yellow_white", "daylight"),
        ("highway_1080p_standard.png", 1920, 1080, "standard_yellow_white", "daylight"),
        ("highway_360p_lowres.jpeg", 640, 360, "standard_yellow_white", "daylight"),
        ("highway_square_800p.bmp", 800, 800, "standard_yellow_white", "daylight"),
        ("highway_dual_white.png", 1280, 720, "dual_white", "daylight"),
        ("highway_dusk_dim.jpg", 1280, 720, "standard_yellow_white", "dim_dusk"),
        ("blank_road_no_lanes.png", 1280, 720, "blank_no_markings", "daylight"),
        ("single_lane_right_only.jpg", 1280, 720, "single_right_only", "daylight"),
    ]

    print(f"Generating {len(test_cases)} sample images in '{output_folder}'...")
    for filename, w, h, style, light in test_cases:
        filepath = output_folder / filename
        img = create_synthetic_road_image(w, h, lane_style=style, lighting=light)
        success = cv2.imwrite(str(filepath), img)
        if success:
            print(f" [+] Generated: {filename} ({w}x{h}, {style}, {light})")
        else:
            print(f" [-] Failed to write {filename}")


if __name__ == "__main__":
    generate_all_sample_images()

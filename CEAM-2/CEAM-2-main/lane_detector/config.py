"""
Configuration parameters for the Classical Lane Detector pipeline.

All parameters have robust defaults suited for general highway driving scenes,
while allowing dynamic customization through CLI flags or programmatically
without modifying any core algorithm code.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class LaneDetectorConfig:
    """Configuration settings for classical lane detection pipeline.

    Attributes:
        yellow_hsv_lower: Lower bound for yellow lane markings in HSV [H, S, V].
                          OpenCV HSV ranges: H: 0-179, S: 0-255, V: 0-255.
        yellow_hsv_upper: Upper bound for yellow lane markings in HSV [H, S, V].
        white_hsv_lower: Lower bound for white lane markings in HSV [H, S, V].
        white_hsv_upper: Upper bound for white lane markings in HSV [H, S, V].
        gaussian_kernel_size: Size of Gaussian blur filter kernel (must be odd).
        canny_low_thresh: Lower threshold for hysteresis in Canny edge detector.
        canny_high_thresh: Higher threshold for hysteresis in Canny edge detector.
        roi_top_ratio: Proportional apex y-position of ROI trapezoid [0.0 - 1.0].
        roi_bottom_ratio: Proportional base y-position of ROI trapezoid [0.0 - 1.0].
        roi_top_left_x: Proportional apex left x-position [0.0 - 1.0].
        roi_top_right_x: Proportional apex right x-position [0.0 - 1.0].
        roi_bottom_left_x: Proportional base left x-position [0.0 - 1.0].
        roi_bottom_right_x: Proportional base right x-position [0.0 - 1.0].
        roi_polygon_ratios: Proportional vertices for the Region of Interest (ROI)
                            trapezoid formatted as [(x_ratio, y_ratio), ...].
                            Coordinates are scaled by image (width, height),
                            ensuring zero hardcoded pixel positions.
        hough_rho: Distance resolution of the accumulator in pixels.
        hough_theta: Angle resolution of the accumulator in radians.
        hough_threshold: Accumulator threshold parameter; only lines that get
                         enough votes are returned.
        hough_min_line_len_ratio: Minimum line length as a ratio of image height.
        hough_max_line_gap_ratio: Maximum allowed gap between points on the same
                                  line to link them, as a ratio of image height.
        min_slope_threshold: Minimum absolute slope to reject horizontal noise.
        max_slope_threshold: Maximum absolute slope to reject vertical artifacts.
        lane_overlay_color: BGR color tuple for drawing detected lane lines.
        fill_lane_area: Whether to shade the detected driving corridor between lanes.
        corridor_overlay_color: BGR color tuple for lane corridor fill.
        corridor_alpha: Transparency alpha factor for the corridor overlay [0.0 - 1.0].
        show_roi_overlay: Whether to draw the ROI boundary polygon onto final visualizations.
    """

    # HSV Color Filtering Thresholds:
    # Expanded ranges accommodate diverse lighting, shadows, and camera color profiles
    yellow_hsv_lower: np.ndarray = field(
        default_factory=lambda: np.array([10, 35, 70], dtype=np.uint8)
    )
    yellow_hsv_upper: np.ndarray = field(
        default_factory=lambda: np.array([42, 255, 255], dtype=np.uint8)
    )

    white_hsv_lower: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 160], dtype=np.uint8)
    )
    white_hsv_upper: np.ndarray = field(
        default_factory=lambda: np.array([179, 75, 255], dtype=np.uint8)
    )

    # Gaussian Blur & Edge Detection
    gaussian_kernel_size: int = 5
    canny_low_thresh: int = 40
    canny_high_thresh: int = 120

    # Dynamic Region of Interest (ROI) Ratios:
    # Base spans 0.05 W to 0.95 W; Apex spans 0.20 W to 0.80 W.
    # Reaches from base (0.93 H, avoiding hood/dash/watermarks) up to 0.42 H (horizon).
    roi_top_ratio: Optional[float] = 0.42
    roi_bottom_ratio: Optional[float] = 0.93
    roi_top_left_x: float = 0.20
    roi_top_right_x: float = 0.80
    roi_bottom_left_x: float = 0.05
    roi_bottom_right_x: float = 0.95
    roi_polygon_ratios: Optional[List[Tuple[float, float]]] = None

    # Hough Transform Parameters
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180.0
    hough_threshold: int = 18
    hough_min_line_len_ratio: float = 0.025  # 2.5% of image height
    hough_max_line_gap_ratio: float = 0.050  # 5.0% of image height

    # Slope Filtering:
    # min_slope_threshold rejects horizontal lines (crosswalks, shadows, seams < 19 deg)
    # max_slope_threshold permits steep perspective lines approaching vanishing point (up to 88.8 deg)
    min_slope_threshold: float = 0.35
    max_slope_threshold: float = 50.00

    # Visualization
    lane_overlay_color: Tuple[int, int, int] = (0, 255, 0)
    corridor_overlay_color: Tuple[int, int, int] = (0, 200, 100)
    corridor_alpha: float = 0.30
    fill_lane_area: bool = True
    show_roi_overlay: bool = False

    # Curved Road & Polynomial Extrapolation Settings
    enable_curve_fit: bool = True
    poly_degree: int = 2
    num_roi_slices: int = 10
    segmented_window_margin_ratio: float = 0.08
    min_slice_pixels: int = 15
    use_segmented_roi: bool = True

    def __post_init__(self):
        """Initializes and synchronizes ROI polygon ratios with top/bottom ratios."""
        if self.roi_polygon_ratios is None:
            top_y = self.roi_top_ratio if self.roi_top_ratio is not None else 0.42
            bot_y = self.roi_bottom_ratio if self.roi_bottom_ratio is not None else 0.93
            self.roi_polygon_ratios = [
                (self.roi_bottom_left_x, bot_y),
                (self.roi_top_left_x, top_y),
                (self.roi_top_right_x, top_y),
                (self.roi_bottom_right_x, bot_y),
            ]
            super().__setattr__("roi_top_ratio", top_y)
            super().__setattr__("roi_bottom_ratio", bot_y)
        else:
            super().__setattr__("roi_top_ratio", min(p[1] for p in self.roi_polygon_ratios))
            super().__setattr__("roi_bottom_ratio", max(p[1] for p in self.roi_polygon_ratios))

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(self, "roi_polygon_ratios") and self.roi_polygon_ratios is not None:
            if name == "roi_top_ratio" and value is not None:
                val = float(value)
                if len(self.roi_polygon_ratios) == 4:
                    super().__setattr__(
                        "roi_polygon_ratios",
                        [
                            (self.roi_polygon_ratios[0][0], self.roi_polygon_ratios[0][1]),
                            (self.roi_polygon_ratios[1][0], val),
                            (self.roi_polygon_ratios[2][0], val),
                            (self.roi_polygon_ratios[3][0], self.roi_polygon_ratios[3][1]),
                        ],
                    )
            elif name == "roi_bottom_ratio" and value is not None:
                val = float(value)
                if len(self.roi_polygon_ratios) == 4:
                    super().__setattr__(
                        "roi_polygon_ratios",
                        [
                            (self.roi_polygon_ratios[0][0], val),
                            (self.roi_polygon_ratios[1][0], self.roi_polygon_ratios[1][1]),
                            (self.roi_polygon_ratios[2][0], self.roi_polygon_ratios[2][1]),
                            (self.roi_polygon_ratios[3][0], val),
                        ],
                    )
            elif name == "roi_polygon_ratios" and value is not None:
                super().__setattr__("roi_top_ratio", min(p[1] for p in value))
                super().__setattr__("roi_bottom_ratio", max(p[1] for p in value))

    def update_roi_ratios(
        self,
        top_ratio: Optional[float] = None,
        bottom_ratio: Optional[float] = None,
        polygon_ratios: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Dynamically updates ROI ratios and synchronizes vertices."""
        if polygon_ratios is not None:
            self.roi_polygon_ratios = list(polygon_ratios)
            super().__setattr__("roi_top_ratio", min(p[1] for p in self.roi_polygon_ratios))
            super().__setattr__("roi_bottom_ratio", max(p[1] for p in self.roi_polygon_ratios))
            return

        if top_ratio is not None:
            self.roi_top_ratio = top_ratio
        if bottom_ratio is not None:
            self.roi_bottom_ratio = bottom_ratio

    def get_roi_top_ratio(self) -> float:
        """Returns the y-ratio of the top ROI edge for line extrapolation."""
        if self.roi_top_ratio is not None:
            return float(self.roi_top_ratio)
        if self.roi_polygon_ratios:
            return min(p[1] for p in self.roi_polygon_ratios)
        return 0.42

    def get_roi_bottom_ratio(self) -> float:
        """Returns the y-ratio of the bottom ROI edge for line extrapolation."""
        if self.roi_bottom_ratio is not None:
            return float(self.roi_bottom_ratio)
        if self.roi_polygon_ratios:
            return max(p[1] for p in self.roi_polygon_ratios)
        return 0.93

    def get_effective_roi_polygon_ratios(self) -> List[Tuple[float, float]]:
        """Returns the active polygon ratio vertices."""
        if self.roi_polygon_ratios is not None:
            return list(self.roi_polygon_ratios)
        top_y = self.get_roi_top_ratio()
        bot_y = self.get_roi_bottom_ratio()
        return [
            (self.roi_bottom_left_x, bot_y),
            (self.roi_top_left_x, top_y),
            (self.roi_top_right_x, top_y),
            (self.roi_bottom_right_x, bot_y),
        ]

"""
Core Lane Detector module implementing Classical Computer Vision techniques.

Pipeline stages:
1. Image Loading & Format Validation (JPG, JPEG, PNG, BMP, WEBP)
2. HSV Conversion & Color Filtering (Yellow + White lane markings)
3. Grayscale CLAHE Contrast Enhancement & Gaussian Blurring (Noise reduction)
4. Dual-Branch Canny Edge Detection (Gradient + Color boundary finding)
5. Dynamic ROI Masking (Proportional trapezoid based on image dimensions)
6. Hough Transform (Probabilistic line segment candidate detection)
7. Slope, Position & Length Filtering with Dominant Ego-Lane Clustering
8. Line Fitting, Extrapolation & Cross-Over Prevention (No Fabrication)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .config import LaneDetectorConfig


@dataclass
class LineSegment:
    """Represents a single detected line segment (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def dx(self) -> int:
        return self.x2 - self.x1

    @property
    def dy(self) -> int:
        return self.y2 - self.y1

    @property
    def slope(self) -> float:
        """Slope in pixel coordinates: dy / dx.

        Returns 1e6 or -1e6 for vertical lines to avoid division by zero.
        """
        if self.dx == 0:
            return 1e6 if self.dy >= 0 else -1e6
        return float(self.dy) / float(self.dx)

    @property
    def length(self) -> float:
        """Euclidean length of the line segment in pixels."""
        return float(np.hypot(self.dx, self.dy))

    @property
    def midpoint(self) -> Tuple[float, float]:
        """(x_mid, y_mid) center of segment."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def bottom_intercept(self, height: int) -> float:
        """Projects the line segment down to the image bottom y = height."""
        m = self.slope
        if abs(m) < 1e-4:
            return float(self.x1)
        return float(self.x1 + (height - self.y1) / m)


@dataclass
class LaneDetectionResult:
    """Structured container for lane extraction output and intermediate stages.

    Attributes:
        left_lane: (x_bottom, y_bottom, x_top, y_top) for left lane or None.
        right_lane: (x_bottom, y_bottom, x_top, y_top) for right lane or None.
        status: Detection outcome category.
        messages: Informational and warning messages generated during processing.
        stages: Dictionary containing numpy image arrays for every pipeline stage.
        metadata: Details such as image dimensions, segment counts, and execution metrics.
        left_curve: Array of (x, y) coordinates along the joined polynomial curve for left lane.
        right_curve: Array of (x, y) coordinates along the joined polynomial curve for right lane.
        left_poly: Polynomial equation coefficients for left lane or None.
        right_poly: Polynomial equation coefficients for right lane or None.
        curvature_radius_m: Estimated road radius of curvature in meters.
        curvature_direction: "STRAIGHT", "CURVE_LEFT", or "CURVE_RIGHT".
    """

    left_lane: Optional[Tuple[int, int, int, int]] = None
    right_lane: Optional[Tuple[int, int, int, int]] = None
    status: str = "NO_LANES_DETECTED"
    messages: List[str] = field(default_factory=list)
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    left_curve: Optional[np.ndarray] = None
    right_curve: Optional[np.ndarray] = None
    left_poly: Optional[np.ndarray] = None
    right_poly: Optional[np.ndarray] = None
    curvature_radius_m: Optional[float] = None
    curvature_direction: Optional[str] = None

    @property
    def both_detected(self) -> bool:
        return self.left_lane is not None and self.right_lane is not None

    @property
    def any_detected(self) -> bool:
        return self.left_lane is not None or self.right_lane is not None

    @property
    def is_curved(self) -> bool:
        return self.curvature_direction in ("CURVE_LEFT", "CURVE_RIGHT")


class LaneDetector:
    """Extracts highway lane lines from road images using classical OpenCV methods.

    Dynamically adapts to any image dimension, resolution, and aspect ratio.
    Never fabricates missing lane lines when markings are absent or undetectable.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, config: Optional[LaneDetectorConfig] = None):
        """Initialize the lane detector with configuration.

        Args:
            config: LaneDetectorConfig instance or None for sensible defaults.
        """
        self.config = config if config is not None else LaneDetectorConfig()

    # -------------------------------------------------------------------------
    # STAGE 1: Image Loading & Validation
    # -------------------------------------------------------------------------
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Loads and validates a user-provided image from disk.

        Supports JPG, JPEG, PNG, BMP, and WEBP formats. Handles any resolution.

        Args:
            image_path: Path to the image file.

        Returns:
            np.ndarray: Loaded image in standard OpenCV BGR format.

        Raises:
            FileNotFoundError: If the specified image path does not exist.
            ValueError: If the file extension is unsupported or image cannot be decoded.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: '{image_path}'")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported image format '{path.suffix}'. Supported formats: {allowed}"
            )

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(
                f"Failed to decode image from '{image_path}'. The file may be corrupt or empty."
            )

        return image

    # -------------------------------------------------------------------------
    # STAGE 2: HSV Color Transformation & Filtering
    # -------------------------------------------------------------------------
    def convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """Converts a BGR image to the HSV (Hue, Saturation, Value) color space.

        HSV decouples color information (Hue) from illumination intensity (Value),
        making it significantly more robust than BGR under varying lighting,
        shadows, and bright sunlight.

        Args:
            image: BGR image (H, W, 3).

        Returns:
            np.ndarray: HSV image (H, W, 3).
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def filter_lane_colors(
        self, image: np.ndarray, hsv_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filters yellow and white lane markings using cv2.inRange().

        Computes separate binary masks for yellow and white markings using HSV
        ranges, then combines them using a bitwise OR operation.

        Args:
            image: Original BGR image (H, W, 3).
            hsv_image: Image converted to HSV color space (H, W, 3).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - combined_mask: Binary mask where yellow OR white pixels are 255.
                - yellow_mask: Binary mask for detected yellow lane pixels.
                - white_mask: Binary mask for detected white lane pixels.
        """
        yellow_mask = cv2.inRange(
            hsv_image, self.config.yellow_hsv_lower, self.config.yellow_hsv_upper
        )
        white_mask = cv2.inRange(
            hsv_image, self.config.white_hsv_lower, self.config.white_hsv_upper
        )
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

        return combined_mask, yellow_mask, white_mask

    # -------------------------------------------------------------------------
    # STAGE 3: Contrast Enhancement & Gaussian Blur
    # -------------------------------------------------------------------------
    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Improves gradient contrast along lane markings under shadows and overcast skies.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def apply_gaussian_blur(
        self, image_or_gray: np.ndarray, kernel_size: Optional[int] = None
    ) -> np.ndarray:
        """Applies Gaussian Blur to suppress high-frequency noise and asphalt grain.

        Args:
            image_or_gray: Grayscale (H, W) or BGR (H, W, 3) image.
            kernel_size: Odd integer kernel size. If None, uses config value.

        Returns:
            np.ndarray: Blurred grayscale image.
        """
        if kernel_size is None:
            kernel_size = self.config.gaussian_kernel_size

        k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)

        if len(image_or_gray.shape) == 3:
            gray = cv2.cvtColor(image_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_or_gray

        return cv2.GaussianBlur(gray, (k, k), 0)

    # -------------------------------------------------------------------------
    # STAGE 4: Canny Edge Detection (Hybrid Gradient + Color)
    # -------------------------------------------------------------------------
    def apply_canny_edge_detection(
        self,
        blurred_gray: np.ndarray,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
    ) -> np.ndarray:
        """Computes intensity gradients using the Canny edge detector.

        Args:
            blurred_gray: Single-channel blurred image.
            low_threshold: Lower threshold for hysteresis.
            high_threshold: Upper threshold for hysteresis.

        Returns:
            np.ndarray: Binary edge map where edge pixels are 255.
        """
        low = low_threshold if low_threshold is not None else self.config.canny_low_thresh
        high = high_threshold if high_threshold is not None else self.config.canny_high_thresh
        return cv2.Canny(blurred_gray, low, high)

    # -------------------------------------------------------------------------
    # STAGE 5: Dynamic Region of Interest (ROI) Masking
    # -------------------------------------------------------------------------
    def generate_proportional_roi_mask(
        self,
        height: int,
        width: int,
        polygon_ratios: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dynamically generates a polygon road mask proportional to image dimensions.

        Ensures zero hardcoded pixel coordinates, automatically scaling to any
        resolution (e.g., 360p, 720p, 1080p, 4K) or aspect ratio.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.
            polygon_ratios: List of (x_ratio, y_ratio) defining the ROI boundary.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mask: Binary mask with 255 inside ROI and 0 outside.
                - polygon_points: Int32 vertices array used for cv2.fillPoly.
        """
        ratios = (
            polygon_ratios
            if polygon_ratios is not None
            else self.config.get_effective_roi_polygon_ratios()
        )

        points = []
        for x_ratio, y_ratio in ratios:
            px = int(np.clip(x_ratio * width, 0, width - 1))
            py = int(np.clip(y_ratio * height, 0, height - 1))
            points.append([px, py])

        polygon_points = np.array([points], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, polygon_points, 255)

        return mask, polygon_points

    def apply_roi_mask(
        self, edges: np.ndarray, polygon_ratios: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Applies dynamic ROI mask to an edge map using cv2.bitwise_and()."""
        height, width = edges.shape[:2]
        roi_mask, polygon_points = self.generate_proportional_roi_mask(
            height, width, polygon_ratios
        )
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        return masked_edges, roi_mask, polygon_points

    # -------------------------------------------------------------------------
    # STAGE 6: Probabilistic Hough Line Detection
    # -------------------------------------------------------------------------
    def detect_hough_lines(
        self,
        masked_edges: np.ndarray,
        threshold: Optional[int] = None,
        min_line_len: Optional[int] = None,
        max_line_gap: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Detects line segment candidates using cv2.HoughLinesP().

        Line length and gap parameters dynamically scale with image height if not
        explicitly specified, ensuring resolution independence.
        """
        height, width = masked_edges.shape[:2]

        thresh = threshold if threshold is not None else self.config.hough_threshold

        if min_line_len is None:
            min_line_len = max(8, int(height * self.config.hough_min_line_len_ratio))

        if max_line_gap is None:
            max_line_gap = max(5, int(height * self.config.hough_max_line_gap_ratio))

        lines = cv2.HoughLinesP(
            masked_edges,
            rho=self.config.hough_rho,
            theta=self.config.hough_theta,
            threshold=thresh,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )
        return lines

    # -------------------------------------------------------------------------
    # STAGE 7: Slope, Position & Length Filtering with Clustering
    # -------------------------------------------------------------------------
    def filter_and_classify_lines(
        self,
        lines: Optional[np.ndarray],
        image_shape: Tuple[int, int, ...],
    ) -> Tuple[List[LineSegment], List[LineSegment], List[str]]:
        """Filters line candidates by slope, position, and line length.

        Separates lines into left and right lane candidates:
        - In image coordinates, y increases downwards.
        - Left lane lines point upwards and towards the center: slope dy/dx < 0.
        - Right lane lines point downwards away from the center: slope dy/dx > 0.
        - Rejects flat/horizontal artifacts (|slope| < min_slope_threshold).
        - Allows steep lines approaching vanishing point (|slope| <= max_slope_threshold).
        - Enforces spatial position constraints relative to the camera frame.
        """
        height, width = image_shape[:2]
        messages: List[str] = []

        if lines is None or len(lines) == 0:
            messages.append("Hough transform detected 0 raw line segments.")
            return [], [], messages

        left_candidates: List[LineSegment] = []
        right_candidates: List[LineSegment] = []

        # Road center division with generous tolerance
        left_bound_x = width * 0.58
        right_bound_x = width * 0.42

        for raw_line in lines:
            line_arr = np.asarray(raw_line).flatten()
            if len(line_arr) < 4:
                continue
            x1, y1, x2, y2 = int(line_arr[0]), int(line_arr[1]), int(line_arr[2]), int(line_arr[3])
            seg = LineSegment(x1, y1, x2, y2)

            if seg.length < 5.0:
                continue

            slope = seg.slope
            abs_slope = abs(slope)

            # Slope filter: reject horizontal lines, allow steep lines
            if not (
                self.config.min_slope_threshold
                <= abs_slope
                <= self.config.max_slope_threshold
            ):
                continue

            mid_x, _ = seg.midpoint
            x_bot = seg.bottom_intercept(height)

            # Left lane: negative slope in image coordinates, on left half
            if slope < -self.config.min_slope_threshold and mid_x <= left_bound_x and x_bot <= left_bound_x:
                left_candidates.append(seg)

            # Right lane: positive slope in image coordinates, on right half
            elif slope > self.config.min_slope_threshold and mid_x >= right_bound_x and x_bot >= right_bound_x:
                right_candidates.append(seg)

        if not left_candidates:
            messages.append("Warning: No valid left lane marking candidates found.")
        if not right_candidates:
            messages.append("Warning: No valid right lane marking candidates found.")

        return left_candidates, right_candidates, messages

    # -------------------------------------------------------------------------
    # STAGE 8: Dominant Ego-Lane Clustering, Fitting & Extrapolation
    # -------------------------------------------------------------------------
    def fit_dominant_lane(
        self,
        candidates: List[LineSegment],
        height: int,
        width: int,
        y_min: int,
        y_max: int,
        is_left: bool,
    ) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """Fits a smooth lane line from candidate segments using 1D ego-clustering.

        When multiple lanes exist (e.g. multi-lane highways), clusters segments by their
        projected bottom intercept x_bottom and selects the dominant cluster closest to the
        ego-vehicle path, eliminating cross-lane distortion.

        Returns None if candidates is empty. Never fabricates fake lane lines!

        Returns:
            Tuple: ((x_bottom, y_bottom, x_top, y_top), poly_coefficients) or None.
        """
        if not candidates:
            return None

        x_bots = np.array([c.bottom_intercept(height) for c in candidates])
        weights = np.array([c.length for c in candidates])

        # Cluster bin width proportional to image width (e.g. ~5-6% of width)
        bin_width = max(18.0, width * 0.055)
        min_x = np.min(x_bots)
        max_x = np.max(x_bots)
        bins = np.arange(min_x, max_x + bin_width, bin_width)

        if len(bins) <= 1:
            inliers = candidates
        else:
            bin_idx = np.clip(np.digitize(x_bots, bins) - 1, 0, len(bins) - 1)
            bin_weights = np.zeros(len(bins))
            for idx, wgt in zip(bin_idx, weights):
                bin_weights[idx] += wgt

            # Pick the strongest cluster
            best_bin = int(np.argmax(bin_weights))
            target_x = float(bins[best_bin] + bin_width / 2.0)

            # Keep inliers within the dominant cluster window
            inliers = [
                c for c in candidates
                if abs(c.bottom_intercept(height) - target_x) <= bin_width * 1.6
            ]
            if not inliers:
                inliers = candidates

        # Weighted polynomial fit of degree 1: x = poly[0] * y + poly[1]
        y_vals: List[float] = []
        x_vals: List[float] = []
        w_vals: List[float] = []

        for seg in inliers:
            w = max(1.0, seg.length)
            y_vals.extend([float(seg.y1), float(seg.y2)])
            x_vals.extend([float(seg.x1), float(seg.x2)])
            w_vals.extend([w, w])

        y_arr = np.array(y_vals, dtype=np.float64)
        x_arr = np.array(x_vals, dtype=np.float64)
        w_arr = np.array(w_vals, dtype=np.float64)

        try:
            poly = np.polyfit(y_arr, x_arr, deg=1, w=w_arr)
        except Exception:
            try:
                poly = np.polyfit(y_arr, x_arr, deg=1)
            except Exception:
                return None

        m_y, b_y = poly[0], poly[1]

        y_bottom = float(y_max)
        y_top = float(y_min)

        x_bottom = m_y * y_bottom + b_y
        x_top = m_y * y_top + b_y

        if not (np.isfinite(x_bottom) and np.isfinite(x_top)):
            return None

        x_b_clamped = int(np.clip(round(x_bottom), -width * 0.25, width * 1.25))
        x_t_clamped = int(np.clip(round(x_top), 0, width - 1))

        coords = (x_b_clamped, int(y_bottom), x_t_clamped, int(y_top))
        return coords, poly

    # -------------------------------------------------------------------------
    # STAGE 8a: Dynamic Segmented ROI Masking (Idea 1)
    # -------------------------------------------------------------------------
    def generate_segmented_roi_mask(
        self,
        feature_map: np.ndarray,
        height: int,
        width: int,
        y_min: int,
        y_max: int,
        left_candidates: Optional[List[LineSegment]] = None,
        right_candidates: Optional[List[LineSegment]] = None,
        num_slices: Optional[int] = None,
        margin_ratio: Optional[float] = None,
        min_slice_pixels: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[int, int, int, int, str]]]:
        """Generates dynamic horizontal ROI segment windows tracking curved lane paths.

        Implements Idea 1: Segmented ROI Mask.
        Instead of a single rigid trapezoid, divides the road perspective range [y_min, y_max]
        into N horizontal slices. Each slice tracks the lane path dynamically using sliding
        windows centered around previous slice centroids, isolating curved lane markings
        from roadside clutter.

        Args:
            feature_map: Binary image combining Canny edges and color mask within general ROI.
            height: Image height in pixels.
            width: Image width in pixels.
            y_min: Apex vertical coordinate of road ROI.
            y_max: Base vertical coordinate of road ROI.
            left_candidates: Detected Hough line segments for left lane candidates.
            right_candidates: Detected Hough line segments for right lane candidates.
            num_slices: Number of vertical slices (default from config).
            margin_ratio: Horizontal search margin as fraction of width (default from config).
            min_slice_pixels: Minimum active pixels required to update window centroid.

        Returns:
            Tuple containing:
                - segmented_roi_mask: Binary mask of all tracked segment windows.
                - left_points: List of (x, y) coordinates belonging to left lane.
                - right_points: List of (x, y) coordinates belonging to right lane.
                - slice_windows: List of (x1, y1, x2, y2, side) bounding boxes.
        """
        n_slices = num_slices if num_slices is not None else self.config.num_roi_slices
        margin = int(width * (margin_ratio if margin_ratio is not None else self.config.segmented_window_margin_ratio))
        min_pixels = min_slice_pixels if min_slice_pixels is not None else self.config.min_slice_pixels
        slice_h = max(4, int((y_max - y_min) / max(1, n_slices)))

        left_pts: List[Tuple[float, float]] = []
        right_pts: List[Tuple[float, float]] = []
        slice_windows: List[Tuple[int, int, int, int, str]] = []
        segmented_roi_mask = np.zeros((height, width), dtype=np.uint8)

        left_cands = left_candidates if left_candidates is not None else []
        right_cands = right_candidates if right_candidates is not None else []

        # Sample points along candidate Hough segments
        for c in left_cands:
            steps = max(5, int(c.length / 4.0))
            for t in np.linspace(0, 1, steps):
                left_pts.append((float((1 - t) * c.x1 + t * c.x2), float((1 - t) * c.y1 + t * c.y2)))

        for c in right_cands:
            steps = max(5, int(c.length / 4.0))
            for t in np.linspace(0, 1, steps):
                right_pts.append((float((1 - t) * c.x1 + t * c.x2), float((1 - t) * c.y1 + t * c.y2)))

        # Determine spatial priors for base tracking
        has_left = len(left_cands) > 0
        has_right = len(right_cands) > 0

        bot_region = feature_map[max(0, y_max - 2 * slice_h):y_max, :]
        hist_l = np.sum(bot_region[:, :int(width * 0.48)], axis=0) if bot_region.size > 0 else np.array([])
        hist_r = np.sum(bot_region[:, int(width * 0.52):], axis=0) if bot_region.size > 0 else np.array([])

        if not has_left and hist_l.size > 0 and np.max(hist_l) >= min_pixels * 2:
            has_left = True
        if not has_right and hist_r.size > 0 and np.max(hist_r) >= min_pixels * 2:
            has_right = True

        if left_cands:
            cur_xl = int(np.median([c.midpoint[0] for c in left_cands]))
        elif hist_l.size > 0 and np.max(hist_l) > 0:
            cur_xl = int(np.argmax(hist_l))
        else:
            cur_xl = int(width * 0.25)

        if right_cands:
            cur_xr = int(np.median([c.midpoint[0] for c in right_cands]))
        elif hist_r.size > 0 and np.max(hist_r) > 0:
            cur_xr = int(width * 0.52 + np.argmax(hist_r))
        else:
            cur_xr = int(width * 0.75)

        drift_l = 0.0
        drift_r = 0.0

        for s in range(n_slices):
            y_bot_s = y_max - s * slice_h
            y_top_s = max(y_min, y_bot_s - slice_h)
            if y_top_s >= y_bot_s:
                continue

            if has_left:
                xl1 = max(0, int(cur_xl - margin))
                xl2 = min(width - 1, int(cur_xl + margin))
                if xl2 > xl1:
                    sl = feature_map[y_top_s:y_bot_s, xl1:xl2]
                    ys, xs = np.nonzero(sl)
                    if len(xs) >= min_pixels:
                        new_xl = float(np.mean(xs + xl1))
                        drift_l = new_xl - cur_xl
                        cur_xl = int(new_xl)
                        for py, px in zip(ys + y_top_s, xs + xl1):
                            left_pts.append((float(px), float(py)))
                    else:
                        cur_xl = int(cur_xl + drift_l * 0.4)
                    cv2.rectangle(segmented_roi_mask, (xl1, y_top_s), (xl2, y_bot_s), 255, -1)
                    slice_windows.append((xl1, y_top_s, xl2, y_bot_s, "left"))

            if has_right:
                xr1 = max(0, int(cur_xr - margin))
                xr2 = min(width - 1, int(cur_xr + margin))
                if xr2 > xr1:
                    sr = feature_map[y_top_s:y_bot_s, xr1:xr2]
                    ys, xs = np.nonzero(sr)
                    if len(xs) >= min_pixels:
                        new_xr = float(np.mean(xs + xr1))
                        drift_r = new_xr - cur_xr
                        cur_xr = int(new_xr)
                        for py, px in zip(ys + y_top_s, xs + xr1):
                            right_pts.append((float(px), float(py)))
                    else:
                        cur_xr = int(cur_xr + drift_r * 0.4)
                    cv2.rectangle(segmented_roi_mask, (xr1, y_top_s), (xr2, y_bot_s), 255, -1)
                    slice_windows.append((xr1, y_top_s, xr2, y_bot_s, "right"))

        return segmented_roi_mask, left_pts, right_pts, slice_windows

    # -------------------------------------------------------------------------
    # STAGE 8b: Polynomial Curve Joining & Curvature Analysis (Idea 2)
    # -------------------------------------------------------------------------
    def fit_curved_polynomial_lane(
        self,
        points: List[Tuple[float, float]],
        height: int,
        width: int,
        y_min: int,
        y_max: int,
        degree: Optional[int] = None,
        candidate_segments: Optional[List[LineSegment]] = None,
    ) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray, float, str]]:
        """Fits a polynomial equation to join line segments into a smooth curve.

        Implements Idea 2: Polynomial Equations to Join the Lines.
        Fits x(y) = a_d * y^d + ... + a_1 * y + a_0 to collected points, joining
        dashes and segmented lane markings into a continuous trajectory.
        Computes the road's radius of curvature and curvature direction.

        Args:
            points: List of (x, y) coordinates from segmented tracking and candidate lines.
            height: Image height in pixels.
            width: Image width in pixels.
            y_min: Top apex y-coordinate for evaluation.
            y_max: Bottom base y-coordinate for evaluation.
            degree: Degree of polynomial (default from config, e.g. 2).
            candidate_segments: Optional candidate segments for extra directional reinforcement.

        Returns:
            Tuple containing:
                - endpoints: (x_bottom, y_bottom, x_top, y_top) 4-tuple.
                - poly_coeffs: np.ndarray polynomial coefficients.
                - curve_points: np.ndarray of shape (K, 2) dense (x, y) coordinates.
                - curvature_radius_m: Estimated road curvature radius in meters.
                - curvature_direction: "STRAIGHT", "CURVE_LEFT", or "CURVE_RIGHT".
            or None if insufficient inliers or invalid geometry.
        """
        all_pts = list(points)
        if candidate_segments:
            for c in candidate_segments:
                steps = max(4, int(c.length / 5.0))
                for t in np.linspace(0, 1, steps):
                    all_pts.append((float((1 - t) * c.x1 + t * c.x2), float((1 - t) * c.y1 + t * c.y2)))

        if len(all_pts) < 20:
            return None

        y_arr = np.array([p[1] for p in all_pts], dtype=np.float64)
        x_arr = np.array([p[0] for p in all_pts], dtype=np.float64)

        y_span = float(np.max(y_arr) - np.min(y_arr))
        if y_span < (y_max - y_min) * 0.15:
            return None

        deg = degree if degree is not None else self.config.poly_degree
        if len(all_pts) < 35:
            deg = 1

        try:
            poly = np.polyfit(y_arr, x_arr, deg=deg)
        except Exception:
            try:
                poly = np.polyfit(y_arr, x_arr, deg=1)
                deg = 1
            except Exception:
                return None

        dense_y = np.linspace(y_max, y_min, int(y_max - y_min + 1))
        dense_x = np.polyval(poly, dense_y)

        x_bottom = int(np.clip(round(dense_x[0]), -int(width * 0.25), int(width * 1.25)))
        x_top = int(np.clip(round(dense_x[-1]), 0, width - 1))
        endpoints = (x_bottom, y_max, x_top, y_min)

        curve_points = np.column_stack([np.clip(dense_x, 0, width - 1), dense_y]).astype(np.int32)

        if deg >= 2:
            a = float(poly[0])
            b = float(poly[1])
            dy_eval = float(y_max)
            dx_dy = 2.0 * a * dy_eval + b
            d2x_dy2 = 2.0 * a
            denom = (1.0 + dx_dy ** 2) ** 1.5
            kappa = abs(d2x_dy2) / max(1e-9, denom)
            radius_px = 1.0 / max(1e-9, kappa)
            meters_per_px = 3.7 / max(40.0, width * 0.35)
            radius_m = radius_px * meters_per_px

            if abs(a) < 1.0e-4 or radius_m > 1500.0:
                direction = "STRAIGHT"
            elif a < 0:
                direction = "CURVE_LEFT"
            else:
                direction = "CURVE_RIGHT"
        else:
            radius_m = float("inf")
            direction = "STRAIGHT"

        return endpoints, poly, curve_points, radius_m, direction

    # -------------------------------------------------------------------------
    # FULL END-TO-END PIPELINE
    # -------------------------------------------------------------------------
    def process_image(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        polygon_ratios: Optional[List[Tuple[float, float]]] = None,
        roi_top_ratio: Optional[float] = None,
        roi_bottom_ratio: Optional[float] = None,
    ) -> LaneDetectionResult:
        """Runs the complete classical lane line extraction pipeline on an image."""
        # Step 1: Load Image
        if isinstance(image_or_path, (str, Path)):
            original_image = self.load_image(image_or_path)
            source_name = str(image_or_path)
        elif isinstance(image_or_path, np.ndarray):
            if image_or_path.size == 0 or len(image_or_path.shape) < 2:
                raise ValueError("Provided numpy array is empty or invalid.")
            original_image = image_or_path.copy()
            source_name = "<numpy_array>"
        else:
            raise TypeError(
                f"Expected str, Path or np.ndarray, got {type(image_or_path).__name__}"
            )

        height, width = original_image.shape[:2]
        messages: List[str] = []

        # Determine effective ROI ratios for this run
        active_polygon_ratios = polygon_ratios
        if active_polygon_ratios is None:
            if roi_top_ratio is not None or roi_bottom_ratio is not None:
                base_ratios = self.config.get_effective_roi_polygon_ratios()
                eff_top = roi_top_ratio if roi_top_ratio is not None else self.config.get_roi_top_ratio()
                eff_bot = roi_bottom_ratio if roi_bottom_ratio is not None else self.config.get_roi_bottom_ratio()
                if len(base_ratios) == 4:
                    active_polygon_ratios = [
                        (base_ratios[0][0], eff_bot),
                        (base_ratios[1][0], eff_top),
                        (base_ratios[2][0], eff_top),
                        (base_ratios[3][0], eff_bot),
                    ]
                else:
                    active_polygon_ratios = [
                        (p[0], eff_top if i in (1, 2) else eff_bot)
                        for i, p in enumerate(base_ratios)
                    ]
            else:
                active_polygon_ratios = self.config.get_effective_roi_polygon_ratios()

        effective_roi_top_ratio = min(p[1] for p in active_polygon_ratios)
        effective_roi_bottom_ratio = max(p[1] for p in active_polygon_ratios)

        # Step 2: HSV Color Transformation & Dual-Band Filtering
        hsv_image = self.convert_to_hsv(original_image)
        color_mask, yellow_mask, white_mask = self.filter_lane_colors(
            original_image, hsv_image
        )

        # Step 3: Grayscale CLAHE Contrast Enhancement & Gaussian Blurring
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.enhance_contrast(gray_original)
        blurred_gray = self.apply_gaussian_blur(enhanced_gray)

        # Step 4: Hybrid Canny Edge Detection (Gradient + Color)
        canny_edges_gray = self.apply_canny_edge_detection(blurred_gray)
        blurred_color = cv2.GaussianBlur(color_mask, (5, 5), 0)
        canny_edges_color = self.apply_canny_edge_detection(blurred_color)
        canny_edges = cv2.bitwise_or(canny_edges_gray, canny_edges_color)

        # Step 5: Dynamic ROI Masking
        masked_edges, roi_mask, roi_polygon = self.apply_roi_mask(
            canny_edges, polygon_ratios=active_polygon_ratios
        )

        # Step 6: Probabilistic Hough Line Detection
        raw_lines = self.detect_hough_lines(masked_edges)

        # Step 7: Slope, Position & Length Filtering
        left_candidates, right_candidates, filter_msgs = (
            self.filter_and_classify_lines(raw_lines, original_image.shape)
        )
        messages.extend(filter_msgs)

        # Step 8: Dynamic Segmented ROI Masking (Idea 1)
        feature_map = cv2.bitwise_and(cv2.bitwise_or(canny_edges, color_mask), roi_mask)
        y_max = int(height * effective_roi_bottom_ratio)
        y_min = int(height * effective_roi_top_ratio)

        if self.config.enable_curve_fit and self.config.use_segmented_roi:
            segmented_roi_mask, left_pts, right_pts, slice_windows = (
                self.generate_segmented_roi_mask(
                    feature_map=feature_map,
                    height=height,
                    width=width,
                    y_min=y_min,
                    y_max=y_max,
                    left_candidates=left_candidates,
                    right_candidates=right_candidates,
                    num_slices=self.config.num_roi_slices,
                    margin_ratio=self.config.segmented_window_margin_ratio,
                    min_slice_pixels=self.config.min_slice_pixels,
                )
            )
        else:
            segmented_roi_mask = roi_mask.copy()
            left_pts, right_pts, slice_windows = [], [], []

        # Step 9: Lane Fitting & Polynomial Curve Joining (Idea 2)
        left_lane, right_lane = None, None
        left_poly, right_poly = None, None
        left_curve, right_curve = None, None
        curv_rad_l, curv_dir_l = None, None
        curv_rad_r, curv_dir_r = None, None

        if self.config.enable_curve_fit:
            if left_candidates or len(left_pts) >= 20:
                curv_l = self.fit_curved_polynomial_lane(
                    points=left_pts,
                    height=height,
                    width=width,
                    y_min=y_min,
                    y_max=y_max,
                    degree=self.config.poly_degree,
                    candidate_segments=left_candidates,
                )
                if curv_l:
                    left_lane, left_poly, left_curve, curv_rad_l, curv_dir_l = curv_l

            if right_candidates or len(right_pts) >= 20:
                curv_r = self.fit_curved_polynomial_lane(
                    points=right_pts,
                    height=height,
                    width=width,
                    y_min=y_min,
                    y_max=y_max,
                    degree=self.config.poly_degree,
                    candidate_segments=right_candidates,
                )
                if curv_r:
                    right_lane, right_poly, right_curve, curv_rad_r, curv_dir_r = curv_r

        # Fallback to dominant linear fit if curve fit not used or produced None for candidates
        if left_lane is None and left_candidates:
            left_fit = self.fit_dominant_lane(
                left_candidates, height, width, y_min, y_max, is_left=True
            )
            if left_fit:
                left_lane, left_poly = left_fit

        if right_lane is None and right_candidates:
            right_fit = self.fit_dominant_lane(
                right_candidates, height, width, y_min, y_max, is_left=False
            )
            if right_fit:
                right_lane, right_poly = right_fit

        # Cross-over & Vanishing Point Sanity Check:
        if left_curve is not None and right_curve is not None:
            # Check intersection along curve points
            cross_indices = np.where(left_curve[:, 0] >= right_curve[:, 0])[0]
            if len(cross_indices) > 0:
                k = cross_indices[0]
                if k == 0:
                    if len(left_candidates) >= len(right_candidates):
                        right_lane = None
                        right_curve = None
                        right_poly = None
                        messages.append("Cross-over anomaly at bottom: discarded right line.")
                    else:
                        left_lane = None
                        left_curve = None
                        left_poly = None
                        messages.append("Cross-over anomaly at bottom: discarded left line.")
                elif k > 5:
                    left_curve = left_curve[:k]
                    right_curve = right_curve[:k]
                    left_lane = (left_lane[0], left_lane[1], int(left_curve[-1, 0]), int(left_curve[-1, 1]))
                    right_lane = (right_lane[0], right_lane[1], int(right_curve[-1, 0]), int(right_curve[-1, 1]))
        elif left_lane and right_lane and left_poly is not None and right_poly is not None and len(left_poly) == 2 and len(right_poly) == 2:
            poly_l, poly_r = left_poly, right_poly
            denom = poly_l[0] - poly_r[0]
            if abs(denom) > 1e-5:
                y_int = (poly_r[1] - poly_l[1]) / denom
                lx_at_top = poly_l[0] * y_min + poly_l[1]
                rx_at_top = poly_r[0] * y_min + poly_r[1]
                if lx_at_top >= rx_at_top:
                    if y_int > height * 0.75:
                        if len(left_candidates) >= len(right_candidates):
                            right_lane = None
                            messages.append("Cross-over anomaly at bottom: discarded right line.")
                        else:
                            left_lane = None
                            messages.append("Cross-over anomaly at bottom: discarded left line.")
                    else:
                        new_y_top = min(y_max - 10, int(y_int + max(2, int(height * 0.01))))
                        lx_t = int(poly_l[0] * new_y_top + poly_l[1])
                        rx_t = int(poly_r[0] * new_y_top + poly_r[1])
                        if lx_t < rx_t:
                            left_lane = (left_lane[0], left_lane[1], lx_t, new_y_top)
                            right_lane = (right_lane[0], right_lane[1], rx_t, new_y_top)

        # Curvature aggregation
        overall_curvature_dir = None
        overall_curvature_rad = None
        if curv_dir_l and curv_dir_r:
            if curv_dir_l == curv_dir_r:
                overall_curvature_dir = curv_dir_l
            elif curv_dir_l == "STRAIGHT":
                overall_curvature_dir = curv_dir_r
            elif curv_dir_r == "STRAIGHT":
                overall_curvature_dir = curv_dir_l
            else:
                overall_curvature_dir = (
                    curv_dir_l if (curv_rad_l or float("inf")) < (curv_rad_r or float("inf")) else curv_dir_r
                )

            radii = [r for r in [curv_rad_l, curv_rad_r] if r is not None and np.isfinite(r)]
            overall_curvature_rad = float(np.mean(radii)) if radii else None
        elif curv_dir_l:
            overall_curvature_dir = curv_dir_l
            overall_curvature_rad = curv_rad_l if (curv_rad_l and np.isfinite(curv_rad_l)) else None
        elif curv_dir_r:
            overall_curvature_dir = curv_dir_r
            overall_curvature_rad = curv_rad_r if (curv_rad_r and np.isfinite(curv_rad_r)) else None

        # Determine overall detection status
        if left_lane is not None and right_lane is not None:
            status = "BOTH_LANES_DETECTED"
            messages.append("Success: Both left and right lane lines successfully extracted.")
        elif left_lane is not None:
            status = "LEFT_ONLY_DETECTED"
            messages.append("Partial: Left lane line detected; right lane line not visible or absent.")
        elif right_lane is not None:
            status = "RIGHT_ONLY_DETECTED"
            messages.append("Partial: Right lane line detected; left lane line not visible or absent.")
        else:
            status = "NO_LANES_DETECTED"
            messages.append("Notice: No lane lines detected in the given image. Zero lines fabricated.")

        # Create intermediate stage visualizations
        hough_canvas = np.zeros_like(original_image)
        if raw_lines is not None:
            for l in raw_lines:
                arr = np.asarray(l).flatten()
                if len(arr) >= 4:
                    x1, y1, x2, y2 = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
                    cv2.line(hough_canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)

        roi_vis = original_image.copy()
        cv2.polylines(roi_vis, roi_polygon, isClosed=True, color=(0, 255, 255), thickness=2)

        stages = {
            "01_original": original_image,
            "02_hsv": hsv_image,
            "03_color_mask": color_mask,
            "04_yellow_mask": yellow_mask,
            "05_white_mask": white_mask,
            "06_gaussian_blur": blurred_gray,
            "07_canny_edges": canny_edges,
            "08_roi_mask": roi_mask,
            "09_roi_visualization": roi_vis,
            "10_masked_edges": masked_edges,
            "11_hough_segments": hough_canvas,
            "12_segmented_roi_mask": segmented_roi_mask,
        }

        metadata = {
            "source": source_name,
            "dimensions": {"width": width, "height": height},
            "raw_hough_segments_count": 0 if raw_lines is None else len(raw_lines),
            "left_candidate_segments": len(left_candidates),
            "right_candidate_segments": len(right_candidates),
            "left_lane_coordinates": left_lane,
            "right_lane_coordinates": right_lane,
            "roi_polygon_points": roi_polygon[0].tolist(),
            "roi_top_ratio": effective_roi_top_ratio,
            "roi_bottom_ratio": effective_roi_bottom_ratio,
            "roi_polygon_ratios": active_polygon_ratios,
            "is_curved": overall_curvature_dir in ("CURVE_LEFT", "CURVE_RIGHT"),
            "curvature_direction": overall_curvature_dir,
            "curvature_radius_m": overall_curvature_rad,
            "poly_degree": self.config.poly_degree if self.config.enable_curve_fit else 1,
            "num_roi_slices": self.config.num_roi_slices if self.config.use_segmented_roi else 0,
        }

        return LaneDetectionResult(
            left_lane=left_lane,
            right_lane=right_lane,
            status=status,
            messages=messages,
            stages=stages,
            metadata=metadata,
            left_curve=left_curve,
            right_curve=right_curve,
            left_poly=left_poly,
            right_poly=right_poly,
            curvature_radius_m=overall_curvature_rad,
            curvature_direction=overall_curvature_dir,
        )

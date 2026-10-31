"""
Comprehensive test suite for the Classical Lane Line Extractor.

Verifies:
1. Image loading and format handling (JPG, JPEG, PNG, BMP)
2. HSV transformation and dual-color filtering (Yellow + White)
3. Gaussian blur and noise reduction
4. Canny edge detection
5. Dynamic proportional ROI generation without hardcoded coordinates
6. Hough line detection and slope/position/length filtering
7. Fitting, extrapolation, and the strict NO-FABRICATION rule
8. Negative test cases (empty roads, single lanes)
9. Diverse resolutions and aspect ratios (360p, 720p, 1080p, square 800x800)
10. Visualizer and multi-panel montage generation
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from generate_sample_images import create_synthetic_road_image
from lane_detector import LaneDetector, LaneDetectorConfig, Visualizer
from lane_detector.detector import LineSegment


class TestLaneDetectorCore(unittest.TestCase):
    """Unit tests for individual pipeline operations."""

    def setUp(self):
        self.config = LaneDetectorConfig()
        self.detector = LaneDetector(config=self.config)
        self.visualizer = Visualizer(config=self.config)

    # -------------------------------------------------------------------------
    # 1. Image Loading Tests
    # -------------------------------------------------------------------------
    def test_image_loading_valid_formats(self):
        """Verifies loading across all supported formats: JPG, JPEG, PNG, BMP, WEBP."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img[20:80, 20:80] = (255, 255, 255)

            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                file_path = tmp_path / f"test_img{ext}"
                cv2.imwrite(str(file_path), test_img)

                loaded = self.detector.load_image(file_path)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.shape, (100, 100, 3))
                self.assertEqual(loaded.dtype, np.uint8)

    def test_image_loading_nonexistent_file(self):
        """Ensures FileNotFoundError is raised when image file does not exist."""
        with self.assertRaises(FileNotFoundError):
            self.detector.load_image("non_existent_highway_path_12345.png")

    def test_image_loading_unsupported_format(self):
        """Ensures ValueError is raised for unsupported extensions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_file = Path(tmp_dir) / "test.txt"
            dummy_file.write_text("not an image")
            with self.assertRaises(ValueError):
                self.detector.load_image(dummy_file)

    # -------------------------------------------------------------------------
    # 2. HSV Transformation & Color Filtering Tests
    # -------------------------------------------------------------------------
    def test_hsv_conversion(self):
        """Checks BGR to HSV conversion validity."""
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        bgr[:, :] = (0, 255, 255)  # Pure yellow in BGR
        hsv = self.detector.convert_to_hsv(bgr)

        self.assertEqual(hsv.shape, (50, 50, 3))
        # Hue for yellow in OpenCV is around 30
        self.assertTrue(25 <= hsv[25, 25, 0] <= 35)

    def test_color_filtering_yellow_and_white(self):
        """Verifies that yellow and white lane colors are accurately segmented."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a yellow patch on the left
        img[:, 10:30] = (20, 215, 240)  # BGR yellow
        # Create a white patch on the right
        img[:, 70:90] = (245, 245, 245)  # BGR white

        hsv = self.detector.convert_to_hsv(img)
        combined_mask, yellow_mask, white_mask = self.detector.filter_lane_colors(
            img, hsv
        )

        # Check yellow mask catches yellow patch and ignores white patch
        self.assertTrue(np.all(yellow_mask[:, 10:30] == 255))
        self.assertTrue(np.all(yellow_mask[:, 70:90] == 0))

        # Check white mask catches white patch and ignores yellow patch
        self.assertTrue(np.all(white_mask[:, 70:90] == 255))
        self.assertTrue(np.all(white_mask[:, 10:30] == 0))

        # Check combined mask catches both
        self.assertTrue(np.all(combined_mask[:, 10:30] == 255))
        self.assertTrue(np.all(combined_mask[:, 70:90] == 255))
        # Background remains 0
        self.assertTrue(np.all(combined_mask[:, 40:60] == 0))

    # -------------------------------------------------------------------------
    # 3. Gaussian Blur Tests
    # -------------------------------------------------------------------------
    def test_gaussian_blur(self):
        """Verifies noise smoothing and odd kernel size handling."""
        noisy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = self.detector.apply_gaussian_blur(noisy, kernel_size=5)

        self.assertEqual(blurred.shape, (100, 100))
        self.assertEqual(blurred.dtype, np.uint8)

        # Passing an even kernel size should automatically adjust to odd >= 3
        blurred_even = self.detector.apply_gaussian_blur(noisy, kernel_size=4)
        self.assertEqual(blurred_even.shape, (100, 100))

    # -------------------------------------------------------------------------
    # 4. Canny Edge Detection Tests
    # -------------------------------------------------------------------------
    def test_canny_edge_detection(self):
        """Verifies edge map generation on high contrast step."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[:, 50:] = 255  # Vertical sharp step edge at x=50

        edges = self.detector.apply_canny_edge_detection(gray, low_threshold=50, high_threshold=150)
        self.assertEqual(edges.shape, (100, 100))
        # Edges should be located around column 49/50
        self.assertTrue(np.any(edges[:, 48:52] == 255))
        # Uniform flat regions must have zero edges
        self.assertTrue(np.all(edges[:, 0:40] == 0))
        self.assertTrue(np.all(edges[:, 60:100] == 0))

    # -------------------------------------------------------------------------
    # 5. Dynamic Proportional ROI Tests (No Hardcoded Pixels)
    # -------------------------------------------------------------------------
    def test_dynamic_roi_proportional_coordinates(self):
        """Ensures ROI dynamically calculates proportional polygon coordinates.

        Verifies that for arbitrary resolutions, mask dimensions match exactly,
        and vertices scale proportionally with height and width.
        """
        resolutions = [
            (360, 640),
            (720, 1280),
            (1080, 1920),
            (800, 800),
            (480, 854),
        ]

        for h, w in resolutions:
            mask, points = self.detector.generate_proportional_roi_mask(h, w)
            self.assertEqual(mask.shape, (h, w), f"Mask shape mismatch for {w}x{h}")

            pts = points[0]
            # Verify all points fall inside image bounds
            for px, py in pts:
                self.assertGreaterEqual(px, 0)
                self.assertLess(px, w)
                self.assertGreaterEqual(py, 0)
                self.assertLess(py, h)

            # Bottom points must be lower (greater y) than top points
            bottom_left, top_left, top_right, bottom_right = pts
            self.assertGreater(bottom_left[1], top_left[1])
            self.assertGreater(bottom_right[1], top_right[1])
            self.assertLess(top_left[0], top_right[0])

    # -------------------------------------------------------------------------
    # 6. Hough Line Filtering Tests
    # -------------------------------------------------------------------------
    def test_filter_and_classify_lines(self):
        """Verifies slope, position, and line length filtering logic."""
        h, w = 720, 1280

        # Construct synthetic line segments:
        # 1. Valid left lane segment: slanting up towards center, on left side
        left_seg = np.array([[[200, 700, 450, 450]]], dtype=np.int32)
        # 2. Valid right lane segment: slanting down away from center, on right side
        right_seg = np.array([[[850, 450, 1100, 700]]], dtype=np.int32)
        # 3. Flat/horizontal segment (should be rejected by slope filter)
        flat_seg = np.array([[[100, 600, 500, 605]]], dtype=np.int32)
        # 4. Vertical noise spike (should be rejected by max slope filter)
        vert_seg = np.array([[[640, 200, 640, 600]]], dtype=np.int32)

        test_lines = np.vstack([left_seg, right_seg, flat_seg, vert_seg])

        left_cands, right_cands, msgs = self.detector.filter_and_classify_lines(
            test_lines, (h, w)
        )

        self.assertEqual(len(left_cands), 1)
        self.assertEqual(len(right_cands), 1)
        self.assertEqual((left_cands[0].x1, left_cands[0].y1), (200, 700))
        self.assertEqual((right_cands[0].x1, right_cands[0].y1), (850, 450))

    # -------------------------------------------------------------------------
    # 7. Strict NO FABRICATION Rule Verification
    # -------------------------------------------------------------------------
    def test_no_fabrication_on_blank_road(self):
        """CRITICAL: On an empty road with no markings, NEVER fabricate lines."""
        blank_img = create_synthetic_road_image(
            1280, 720, lane_style="blank_no_markings", lighting="daylight"
        )
        result = self.detector.process_image(blank_img)

        # Must not fabricate any line
        self.assertIsNone(
            result.left_lane, "Left lane was fabricated on a blank road!"
        )
        self.assertIsNone(
            result.right_lane, "Right lane was fabricated on a blank road!"
        )
        self.assertEqual(result.status, "NO_LANES_DETECTED")
        self.assertFalse(result.both_detected)
        self.assertFalse(result.any_detected)
        self.assertTrue(
            any("No lane lines detected" in msg for msg in result.messages)
        )

    def test_single_lane_line_no_fabrication(self):
        """CRITICAL: If only right lane marking exists, do NOT fabricate left lane."""
        single_img = create_synthetic_road_image(
            1280, 720, lane_style="single_right_only", lighting="daylight"
        )
        result = self.detector.process_image(single_img)

        # Left lane must be None (never fabricated)
        self.assertIsNone(
            result.left_lane, "Left lane was fabricated when only right lane existed!"
        )
        # Right lane should be detected
        self.assertIsNotNone(result.right_lane)
        self.assertEqual(result.status, "RIGHT_ONLY_DETECTED")
        self.assertTrue(result.any_detected)
        self.assertFalse(result.both_detected)

    # -------------------------------------------------------------------------
    # 8. Multi-Resolution & Aspect Ratio End-to-End Tests
    # -------------------------------------------------------------------------
    def test_various_resolutions_and_aspect_ratios(self):
        """Tests end-to-end detection on 360p, 720p, 1080p, and square 800x800."""
        configs = [
            (640, 360, "360p_lowres"),
            (1280, 720, "720p_standard"),
            (1920, 1080, "1080p_hd"),
            (800, 800, "square_800p"),
        ]

        for w, h, name in configs:
            img = create_synthetic_road_image(w, h, lane_style="standard_yellow_white")
            result = self.detector.process_image(img)

            self.assertEqual(
                result.status,
                "BOTH_LANES_DETECTED",
                f"Failed to detect both lanes on resolution {w}x{h} ({name})",
            )
            self.assertIsNotNone(result.left_lane)
            self.assertIsNotNone(result.right_lane)

            # Check extrapolated bounds
            lx_b, ly_b, lx_t, ly_t = result.left_lane
            rx_b, ry_b, rx_t, ry_t = result.right_lane

            expected_y_bottom = int(h * self.config.get_roi_bottom_ratio())
            self.assertEqual(ly_b, expected_y_bottom)
            self.assertEqual(ry_b, expected_y_bottom)
            self.assertLess(lx_b, rx_b, "Left lane must be left of right lane at bottom")
            self.assertLess(lx_t, rx_t, "Left lane must be left of right lane at top")

    # -------------------------------------------------------------------------
    # 9. Visualizer & Montage Tests
    # -------------------------------------------------------------------------
    def test_visualizer_and_montage(self):
        """Verifies visualization generation, stage preservation, and montage export."""
        img = create_synthetic_road_image(1280, 720, lane_style="standard_yellow_white")
        result = self.detector.process_image(img)

        # Test overlay drawing
        overlay = self.visualizer.draw_lanes(img, result)
        self.assertEqual(overlay.shape, img.shape)
        self.assertEqual(overlay.dtype, np.uint8)

        # Test montage creation
        montage = self.visualizer.create_stages_montage(result, panel_width=320)
        self.assertEqual(len(montage.shape), 3)
        self.assertEqual(montage.shape[2], 3)
        # Montage consists of 2 rows x 3 columns
        self.assertEqual(montage.shape[1], 320 * 3)

        # Test saving results to disk
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = self.visualizer.save_results(
                result=result,
                output_dir=tmp_dir,
                file_stem="test_export",
                save_individual_stages=True,
            )
            self.assertTrue(Path(saved["final_detected"]).exists())
            self.assertTrue(Path(saved["stages_montage"]).exists())
            self.assertTrue(Path(saved["stages_directory"]).is_dir())

    # -------------------------------------------------------------------------
    # 10. Real-World Sample Images Verification
    # -------------------------------------------------------------------------
    def test_user_sample_images(self):
        """Tests processing on user-provided sample images in sample_images folder."""
        sample_dir = Path("sample_images")
        if not sample_dir.is_dir():
            return

        expected_both = ["images.jpg", "images1.jpg", "images2.jpg"]
        for fname in expected_both:
            img_path = sample_dir / fname
            if img_path.is_file():
                res = self.detector.process_image(img_path)
                self.assertEqual(
                    res.status,
                    "BOTH_LANES_DETECTED",
                    f"Expected BOTH_LANES_DETECTED on {fname}, got {res.status}",
                )
                self.assertIsNotNone(res.left_lane)
                self.assertIsNotNone(res.right_lane)
                # Verify non-crossing corridor
                lx_b, ly_b, lx_t, ly_t = res.left_lane
                rx_b, ry_b, rx_t, ry_t = res.right_lane
                self.assertLess(lx_b, rx_b, f"Lanes crossed at bottom on {fname}")
                self.assertLess(lx_t, rx_t, f"Lanes crossed at top on {fname}")

    # -------------------------------------------------------------------------
    # 11. ROI Ratio Dynamism and Reflection Tests
    # -------------------------------------------------------------------------
    def test_roi_ratio_changes_reflect_in_config_and_mask(self):
        """Verifies that changing roi_top_ratio or roi_bottom_ratio dynamically reflects."""
        cfg = LaneDetectorConfig(roi_top_ratio=0.35, roi_bottom_ratio=0.85)
        self.assertAlmostEqual(cfg.get_roi_top_ratio(), 0.35)
        self.assertAlmostEqual(cfg.get_roi_bottom_ratio(), 0.85)

        # Check polygon ratios match
        ratios = cfg.get_effective_roi_polygon_ratios()
        self.assertAlmostEqual(ratios[0][1], 0.85)
        self.assertAlmostEqual(ratios[1][1], 0.35)
        self.assertAlmostEqual(ratios[2][1], 0.35)
        self.assertAlmostEqual(ratios[3][1], 0.85)

        # Dynamic attribute modification
        cfg.roi_top_ratio = 0.50
        self.assertAlmostEqual(cfg.get_roi_top_ratio(), 0.50)
        self.assertAlmostEqual(cfg.roi_polygon_ratios[1][1], 0.50)

        cfg.roi_bottom_ratio = 0.80
        self.assertAlmostEqual(cfg.get_roi_bottom_ratio(), 0.80)
        self.assertAlmostEqual(cfg.roi_polygon_ratios[0][1], 0.80)

        # Mask generation reflects new ratios
        det = LaneDetector(config=cfg)
        mask, pts = det.generate_proportional_roi_mask(1000, 1000)
        self.assertEqual(pts[0][1][1], 500)
        self.assertEqual(pts[0][0][1], 800)

    def test_roi_ratio_changes_reflect_in_detection_extrapolation(self):
        """Ensures changing ROI ratios in process_image or config reflects on detected lines."""
        img = create_synthetic_road_image(1280, 720, lane_style="standard_yellow_white")

        # Test with custom ROI top ratio 0.60
        cfg1 = LaneDetectorConfig(roi_top_ratio=0.60, roi_bottom_ratio=0.90)
        det1 = LaneDetector(config=cfg1)
        res1 = det1.process_image(img)
        self.assertEqual(res1.status, "BOTH_LANES_DETECTED")
        self.assertEqual(res1.left_lane[1], int(720 * 0.90))
        self.assertEqual(res1.left_lane[3], int(720 * 0.60))

        # Test overriding per-call in process_image
        res2 = det1.process_image(img, roi_top_ratio=0.50, roi_bottom_ratio=0.80)
        self.assertEqual(res2.status, "BOTH_LANES_DETECTED")
        self.assertEqual(res2.left_lane[1], int(720 * 0.80))
        self.assertEqual(res2.left_lane[3], int(720 * 0.50))
        self.assertAlmostEqual(res2.metadata["roi_top_ratio"], 0.50)
        self.assertAlmostEqual(res2.metadata["roi_bottom_ratio"], 0.80)

    # -------------------------------------------------------------------------
    # 11. Curved Road & Polynomial Extrapolation Tests
    # -------------------------------------------------------------------------
    def test_generate_segmented_roi_mask(self):
        """Verifies horizontal segmented ROI mask generation and slice tracking (Idea 1)."""
        height, width = 360, 480
        y_min, y_max = 150, 334
        feature_map = np.zeros((height, width), dtype=np.uint8)

        # Draw a synthetic curved left lane and right lane
        for y in range(y_min, y_max):
            # Left arc curving towards center: x = 0.001 * (y - 200)^2 + 100
            xl = int(0.0015 * (y - 150) ** 2 + 100)
            xr = int(-0.0015 * (y - 150) ** 2 + 380)
            if 0 <= xl < width:
                feature_map[y, max(0, xl - 2):min(width, xl + 3)] = 255
            if 0 <= xr < width:
                feature_map[y, max(0, xr - 2):min(width, xr + 3)] = 255

        cands_l = [LineSegment(100, 155, 120, 200)]
        cands_r = [LineSegment(380, 155, 360, 200)]

        mask, l_pts, r_pts, windows = self.detector.generate_segmented_roi_mask(
            feature_map=feature_map,
            height=height,
            width=width,
            y_min=y_min,
            y_max=y_max,
            left_candidates=cands_l,
            right_candidates=cands_r,
            num_slices=10,
            margin_ratio=0.08,
            min_slice_pixels=10,
        )

        self.assertEqual(mask.shape, (height, width))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(len(l_pts), 10, "Failed to collect points for left segmented track")
        self.assertGreater(len(r_pts), 10, "Failed to collect points for right segmented track")
        self.assertEqual(len(windows), 20, "Expected 10 left and 10 right slice windows")
        self.assertGreater(np.count_nonzero(mask), 0, "Segmented ROI mask should have active window pixels")

    def test_fit_curved_polynomial_lane(self):
        """Verifies polynomial equation curve fitting, dense sampling, and curvature classification (Idea 2)."""
        height, width = 360, 480
        y_min, y_max = 150, 334

        # 1. Left curve: x(y) = -0.002 * y^2 + 0.8 * y + 50 (a < 0 -> CURVE_LEFT)
        ys = np.linspace(y_max, y_min, 60)
        xs_left = -0.002 * (ys ** 2) + 0.8 * ys + 50
        pts_left = list(zip(xs_left, ys))

        res_left = self.detector.fit_curved_polynomial_lane(
            points=pts_left, height=height, width=width, y_min=y_min, y_max=y_max, degree=2
        )
        self.assertIsNotNone(res_left)
        endpoints_l, poly_l, curve_l, rad_l, dir_l = res_left
        self.assertEqual(len(endpoints_l), 4)
        self.assertEqual(endpoints_l[1], y_max)
        self.assertEqual(endpoints_l[3], y_min)
        self.assertEqual(len(poly_l), 3)  # degree 2: 3 coefficients
        self.assertEqual(dir_l, "CURVE_LEFT")
        self.assertLess(poly_l[0], 0)
        self.assertGreater(rad_l, 0)
        self.assertEqual(curve_l.shape[1], 2)

        # 2. Right curve: x(y) = 0.003 * y^2 - 1.2 * y + 350 (a > 0 -> CURVE_RIGHT)
        xs_right = 0.003 * (ys ** 2) - 1.2 * ys + 350
        pts_right = list(zip(xs_right, ys))

        res_right = self.detector.fit_curved_polynomial_lane(
            points=pts_right, height=height, width=width, y_min=y_min, y_max=y_max, degree=2
        )
        self.assertIsNotNone(res_right)
        endpoints_r, poly_r, curve_r, rad_r, dir_r = res_right
        self.assertEqual(dir_r, "CURVE_RIGHT")
        self.assertGreater(poly_r[0], 0)

        # 3. Straight line: x(y) = 0.0 * y^2 - 0.5 * y + 400
        xs_straight = -0.5 * ys + 400
        pts_straight = list(zip(xs_straight, ys))

        res_straight = self.detector.fit_curved_polynomial_lane(
            points=pts_straight, height=height, width=width, y_min=y_min, y_max=y_max, degree=2
        )
        self.assertIsNotNone(res_straight)
        _, _, _, _, dir_s = res_straight
        self.assertEqual(dir_s, "STRAIGHT")

        # 4. Insufficient points (< 20) must return None
        sparse_pts = [(100.0, 200.0), (105.0, 210.0), (110.0, 220.0)]
        self.assertIsNone(
            self.detector.fit_curved_polynomial_lane(
                points=sparse_pts, height=height, width=width, y_min=y_min, y_max=y_max, degree=2
            )
        )

    def test_curved_road_sample_images(self):
        """Tests curved road detection and polynomial curve joining on user curved images."""
        curved_images = [
            ("sample_images/curved1.jpg", "CURVE_LEFT"),
            ("sample_images/curved2.jpg", "CURVE_RIGHT"),
            ("sample_images/curved3.jpg", None),  # either curve direction
        ]

        for img_path_str, expected_dir in curved_images:
            path = Path(img_path_str)
            if not path.exists():
                continue

            result = self.detector.process_image(path)
            self.assertEqual(
                result.status,
                "BOTH_LANES_DETECTED",
                f"Failed to detect both lanes on curved road {path.name}",
            )
            self.assertIsNotNone(result.left_lane)
            self.assertIsNotNone(result.right_lane)
            self.assertIsNotNone(result.left_curve, f"Left curve missing on {path.name}")
            self.assertIsNotNone(result.right_curve, f"Right curve missing on {path.name}")
            self.assertIn("12_segmented_roi_mask", result.stages)
            self.assertTrue(result.is_curved, f"Expected {path.name} to be detected as curved")

            if expected_dir is not None:
                self.assertEqual(
                    result.curvature_direction,
                    expected_dir,
                    f"Wrong curvature direction for {path.name}",
                )

    def test_negative_cases_with_curve_fit_enabled(self):
        """CRITICAL: Enforces zero fabrication on blank/single lane roads with curve fitting enabled."""
        cfg = LaneDetectorConfig(enable_curve_fit=True, use_segmented_roi=True)
        det = LaneDetector(config=cfg)

        # 1. Blank road: NEVER fabricate lines or curves
        blank = create_synthetic_road_image(640, 360, lane_style="blank_no_markings")
        res_blank = det.process_image(blank)
        self.assertEqual(res_blank.status, "NO_LANES_DETECTED")
        self.assertIsNone(res_blank.left_lane)
        self.assertIsNone(res_blank.right_lane)
        self.assertIsNone(res_blank.left_curve)
        self.assertIsNone(res_blank.right_curve)
        self.assertFalse(res_blank.is_curved)

        # 2. Single right lane: NEVER fabricate left lane or left curve
        single = create_synthetic_road_image(640, 360, lane_style="single_right_only")
        res_single = det.process_image(single)
        self.assertEqual(res_single.status, "RIGHT_ONLY_DETECTED")
        self.assertIsNone(res_single.left_lane)
        self.assertIsNone(res_single.left_curve)
        self.assertIsNotNone(res_single.right_lane)
        self.assertIsNotNone(res_single.right_curve)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Visualization utilities for classical lane line extraction.

Handles:
- Drawing detected lane lines and optional driving corridor overlay.
- Dynamic font and line sizing proportional to image dimensions.
- Composing a multi-panel visual montage of all intermediate processing stages.
- Exporting visualization artifacts to disk.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .config import LaneDetectorConfig
from .detector import LaneDetectionResult


class Visualizer:
    """Renders overlays, annotations, and stage montages for lane detection."""

    def __init__(self, config: Optional[LaneDetectorConfig] = None):
        """Initialize the visualizer.

        Args:
            config: LaneDetectorConfig or None for defaults.
        """
        self.config = config if config is not None else LaneDetectorConfig()

    def draw_lanes(
        self,
        image: np.ndarray,
        result: LaneDetectionResult,
    ) -> np.ndarray:
        """Draws detected lane lines and corridor polygon onto the image.

        Scales line thickness and font sizes proportionally based on image dimensions.
        Never fabricates lines: if a lane is None, nothing is drawn for that side.

        Args:
            image: Original BGR image (H, W, 3).
            result: LaneDetectionResult containing lane coordinates.

        Returns:
            np.ndarray: BGR image with lane overlay and HUD status banner.
        """
        height, width = image.shape[:2]
        output = image.copy()
        overlay = image.copy()

        # Dynamic thickness proportional to image size (e.g. 3-8 pixels)
        line_thickness = max(2, int(min(height, width) * 0.007))
        font_scale = max(0.45, min(height, width) * 0.0006)
        text_thickness = max(1, int(font_scale * 1.8))

        # 0. Draw ROI polygon boundary overlay (if enabled in config)
        if getattr(self.config, "show_roi_overlay", False) and "roi_polygon_points" in result.metadata:
            pts = np.array(result.metadata["roi_polygon_points"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                output,
                [pts],
                isClosed=True,
                color=(0, 255, 255),  # Yellow trapezoid boundary
                thickness=max(1, int(min(height, width) * 0.003)),
                lineType=cv2.LINE_AA,
            )

        # 1. Fill corridor area if both left and right lane lines are detected
        if (
            self.config.fill_lane_area
            and result.both_detected
        ):
            if result.left_curve is not None and result.right_curve is not None:
                corridor_poly = np.vstack([result.left_curve, result.right_curve[::-1]])
                cv2.fillPoly(overlay, [corridor_poly], self.config.corridor_overlay_color)
                cv2.addWeighted(
                    overlay,
                    self.config.corridor_alpha,
                    output,
                    1.0 - self.config.corridor_alpha,
                    0,
                    output,
                )
            elif result.left_lane is not None and result.right_lane is not None:
                lx_b, ly_b, lx_t, ly_t = result.left_lane
                rx_b, ry_b, rx_t, ry_t = result.right_lane

                if lx_b < rx_b and lx_t < rx_t:
                    corridor_poly = np.array(
                        [[[lx_b, ly_b], [lx_t, ly_t], [rx_t, ry_t], [rx_b, ry_b]]],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(overlay, corridor_poly, self.config.corridor_overlay_color)
                    cv2.addWeighted(
                        overlay,
                        self.config.corridor_alpha,
                        output,
                        1.0 - self.config.corridor_alpha,
                        0,
                        output,
                    )

        # 2. Draw Left Lane (if detected)
        if result.left_curve is not None:
            cv2.polylines(
                output,
                [result.left_curve],
                isClosed=False,
                color=(0, 0, 255),  # Red for left boundary
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )
        elif result.left_lane is not None:
            lx_b, ly_b, lx_t, ly_t = result.left_lane
            cv2.line(
                output,
                (lx_b, ly_b),
                (lx_t, ly_t),
                (0, 0, 255),  # Red for left boundary
                line_thickness,
                lineType=cv2.LINE_AA,
            )

        # 3. Draw Right Lane (if detected)
        if result.right_curve is not None:
            cv2.polylines(
                output,
                [result.right_curve],
                isClosed=False,
                color=(255, 0, 0),  # Blue for right boundary
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )
        elif result.right_lane is not None:
            rx_b, ry_b, rx_t, ry_t = result.right_lane
            cv2.line(
                output,
                (rx_b, ry_b),
                (rx_t, ry_t),
                (255, 0, 0),  # Blue for right boundary
                line_thickness,
                lineType=cv2.LINE_AA,
            )

        # 4. Draw HUD Status Banner at top
        banner_height = max(36, int(height * 0.065))
        banner_overlay = output.copy()
        cv2.rectangle(
            banner_overlay,
            (0, 0),
            (width, banner_height),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(banner_overlay, 0.70, output, 0.30, 0, output)

        # Status text and color
        if result.both_detected:
            if result.is_curved and result.curvature_radius_m is not None and np.isfinite(result.curvature_radius_m):
                status_text = f"STATUS: BOTH LANES | {result.curvature_direction.replace('_', ' ')} (R~{int(result.curvature_radius_m)}m)"
            elif result.curvature_direction == "STRAIGHT":
                status_text = "STATUS: BOTH LANES DETECTED (STRAIGHT)"
            else:
                status_text = "STATUS: BOTH LANES DETECTED"
            status_color = (0, 255, 0)  # Green
        elif result.any_detected:
            side = "LEFT" if result.left_lane is not None else "RIGHT"
            if result.is_curved and result.curvature_direction:
                status_text = f"STATUS: PARTIAL ({side} ONLY) | {result.curvature_direction.replace('_', ' ')}"
            else:
                status_text = f"STATUS: PARTIAL ({side} LANE ONLY)"
            status_color = (0, 255, 255)  # Yellow
        else:
            status_text = "STATUS: NO LANE DETECTED (NO FABRICATION)"
            status_color = (0, 140, 255)  # Orange-red

        # Put text
        margin_y = int(banner_height * 0.65)
        cv2.putText(
            output,
            status_text,
            (int(width * 0.02), margin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            status_color,
            text_thickness,
            cv2.LINE_AA,
        )

        roi_top = result.metadata.get("roi_top_ratio", self.config.get_roi_top_ratio())
        roi_bot = result.metadata.get("roi_bottom_ratio", self.config.get_roi_bottom_ratio())
        dim_text = f"Res: {width}x{height} | ROI: [{roi_top:.2f}-{roi_bot:.2f}]"
        (t_w, _), _ = cv2.getTextSize(
            dim_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, text_thickness
        )
        cv2.putText(
            output,
            dim_text,
            (width - t_w - int(width * 0.02), margin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85,
            (200, 200, 200),
            text_thickness,
            cv2.LINE_AA,
        )

        return output

    def create_stages_montage(
        self,
        result: LaneDetectionResult,
        panel_width: int = 480,
    ) -> np.ndarray:
        """Builds a neat 2x3 or 2x4 visual montage showing intermediate stages.

        Panels displayed:
            1. Original Road Image
            2. HSV Color Mask (Yellow + White combined)
            3. Grayscale & Gaussian Blurred
            4. Canny Edge Detection
            5. Dynamic ROI Masked Edges
            6. Final Detected Lane Lines Overlay

        Args:
            result: LaneDetectionResult with stages dictionary.
            panel_width: Width in pixels to resize each subpanel for consistent display.

        Returns:
            np.ndarray: Combined BGR montage image.
        """
        stages = result.stages
        original = stages.get("01_original")
        if original is None:
            raise ValueError("Result does not contain original image stage.")

        orig_h, orig_w = original.shape[:2]
        panel_height = int(panel_width * (orig_h / orig_w))

        def prep_panel(img: np.ndarray, title: str) -> np.ndarray:
            """Resizes image to panel dimensions and adds a clean top header title."""
            # Convert single channel grayscale/mask to 3-channel BGR
            if len(img.shape) == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img.copy()

            resized = cv2.resize(
                bgr, (panel_width, panel_height), interpolation=cv2.INTER_AREA
            )

            # Header band
            header_h = 28
            panel = np.zeros((panel_height + header_h, panel_width, 3), dtype=np.uint8)
            panel[header_h:, :] = resized

            # Draw dark header bar
            cv2.rectangle(panel, (0, 0), (panel_width, header_h), (35, 35, 35), -1)
            cv2.putText(
                panel,
                title,
                (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            # Thin border line
            cv2.rectangle(panel, (0, 0), (panel_width - 1, panel_height + header_h - 1), (70, 70, 70), 1)
            return panel

        final_overlay = self.draw_lanes(original, result)

        panel1 = prep_panel(stages["01_original"], "1. Original Input Image")
        panel2 = prep_panel(stages["03_color_mask"], "2. HSV Color Mask (Yellow + White)")
        panel3 = prep_panel(stages["06_gaussian_blur"], "3. Grayscale + Gaussian Blur")
        panel4 = prep_panel(stages["07_canny_edges"], "4. Canny Edge Detection")
        panel5 = prep_panel(stages["10_masked_edges"], "5. Dynamic ROI Masked Edges")
        panel6 = prep_panel(final_overlay, "6. Final Lane Line Extraction")

        # Layout: 2 rows x 3 columns
        row1 = np.hstack([panel1, panel2, panel3])
        row2 = np.hstack([panel4, panel5, panel6])
        montage = np.vstack([row1, row2])

        return montage

    def save_results(
        self,
        result: LaneDetectionResult,
        output_dir: Union[str, Path],
        file_stem: str,
        save_individual_stages: bool = True,
    ) -> Dict[str, Path]:
        """Saves final visualization, montage, and intermediate images to disk.

        Args:
            result: LaneDetectionResult object.
            output_dir: Target directory path.
            file_stem: Base name for output files (e.g., 'road_sample_1').
            save_individual_stages: If True, exports each stage as a separate image.

        Returns:
            Dict[str, Path]: Map of output type to saved file path.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_files: Dict[str, Path] = {}

        original = result.stages.get("01_original")
        if original is None:
            return saved_files

        # 1. Save final lane overlay
        final_overlay = self.draw_lanes(original, result)
        final_file = out_path / f"{file_stem}_detected.png"
        cv2.imwrite(str(final_file), final_overlay)
        saved_files["final_detected"] = final_file

        # 2. Save 6-panel stages montage
        montage = self.create_stages_montage(result)
        montage_file = out_path / f"{file_stem}_stages_montage.png"
        cv2.imwrite(str(montage_file), montage)
        saved_files["stages_montage"] = montage_file

        # 3. Optional individual stages
        if save_individual_stages:
            stages_dir = out_path / f"{file_stem}_stages"
            stages_dir.mkdir(parents=True, exist_ok=True)
            for stage_name, stage_img in result.stages.items():
                stage_file = stages_dir / f"{stage_name}.png"
                cv2.imwrite(str(stage_file), stage_img)
            saved_files["stages_directory"] = stages_dir

        return saved_files

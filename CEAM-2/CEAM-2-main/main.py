#!/usr/bin/env python3
"""
Classical Computer Vision Lane Line Extractor CLI.

Processes user-provided road images using classical OpenCV algorithms.
Dynamically adapts to image dimensions, resolutions, aspect ratios, and lighting.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import cv2

from lane_detector import LaneDetector, LaneDetectorConfig, Visualizer


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract highway lane lines from static road images using OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core Input/Output Arguments
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to an input image (JPG, JPEG, PNG, BMP) or a directory containing images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="output",
        help="Directory where output images and visualizations will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display resulting images in an interactive OpenCV window (press any key to advance).",
    )
    parser.add_argument(
        "--no-individual-stages",
        action="store_true",
        help="Do not save individual stage images (only save final overlay and montage).",
    )

    # Algorithm Tuning Arguments (Dynamic Customization without modifying code)
    group_cv = parser.add_argument_group("OpenCV Tuning Parameters")
    group_cv.add_argument(
        "--canny-low",
        type=int,
        default=40,
        help="Canny edge detection lower hysteresis threshold.",
    )
    group_cv.add_argument(
        "--canny-high",
        type=int,
        default=120,
        help="Canny edge detection upper hysteresis threshold.",
    )
    group_cv.add_argument(
        "--gaussian-kernel",
        type=int,
        default=5,
        help="Gaussian blur kernel size (odd integer >= 3).",
    )
    group_cv.add_argument(
        "--hough-threshold",
        type=int,
        default=18,
        help="Hough transform accumulator voting threshold.",
    )
    group_cv.add_argument(
        "--roi-top-ratio",
        type=float,
        default=None,
        help="Apex y-ratio of ROI trapezoid (fraction of image height, 0.0 - 1.0).",
    )
    group_cv.add_argument(
        "--roi-bottom-ratio",
        type=float,
        default=None,
        help="Base y-ratio of ROI trapezoid (fraction of image height, 0.0 - 1.0).",
    )
    group_cv.add_argument(
        "--roi-ratio",
        type=float,
        default=None,
        help="Apex y-ratio of ROI trapezoid (convenience alias for --roi-top-ratio).",
    )
    group_cv.add_argument(
        "--roi-top-left-x",
        type=float,
        default=None,
        help="Apex left vertex x-ratio (fraction of image width, 0.0 - 1.0).",
    )
    group_cv.add_argument(
        "--roi-top-right-x",
        type=float,
        default=None,
        help="Apex right vertex x-ratio (fraction of image width, 0.0 - 1.0).",
    )
    group_cv.add_argument(
        "--roi-bottom-left-x",
        type=float,
        default=None,
        help="Base left vertex x-ratio (fraction of image width, 0.0 - 1.0).",
    )
    group_cv.add_argument(
        "--roi-bottom-right-x",
        type=float,
        default=None,
        help="Base right vertex x-ratio (fraction of image width, 0.0 - 1.0).",
    )
    group_cv.add_argument(
        "--show-roi",
        action="store_true",
        help="Render the Region of Interest (ROI) polygon overlay onto the detected lane image.",
    )
    group_cv.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        help="Polynomial degree for curved lane line joining (default: 2, e.g. 2 or 3).",
    )
    group_cv.add_argument(
        "--num-slices",
        type=int,
        default=10,
        help="Number of horizontal segmented ROI slices for curve tracking (default: 10).",
    )
    group_cv.add_argument(
        "--no-curve",
        action="store_true",
        help="Disable curved polynomial extrapolation and force linear straight lines.",
    )

    return parser


def find_target_images(input_path: Path) -> List[Path]:
    """Resolves input path into a list of image files."""
    if not input_path.exists():
        print(f"[-] Error: Input path does not exist: '{input_path}'", file=sys.stderr)
        sys.exit(1)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if input_path.is_file():
        if input_path.suffix.lower() not in supported_exts:
            print(
                f"[-] Error: File '{input_path.name}' is not a supported image format ({', '.join(sorted(supported_exts))})",
                file=sys.stderr,
            )
            sys.exit(1)
        return [input_path]

    elif input_path.is_dir():
        files = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in supported_exts
        ]
        if not files:
            print(
                f"[-] Warning: No supported image files found in directory '{input_path}'",
                file=sys.stderr,
            )
            sys.exit(1)
        return files

    else:
        print(f"[-] Error: Invalid input path '{input_path}'", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI execution flow."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Configure custom parameters from CLI (preserves any custom defaults set in config.py)
    config = LaneDetectorConfig(
        canny_low_thresh=args.canny_low,
        canny_high_thresh=args.canny_high,
        gaussian_kernel_size=args.gaussian_kernel,
        hough_threshold=args.hough_threshold,
        show_roi_overlay=args.show_roi,
        enable_curve_fit=not args.no_curve,
        poly_degree=args.poly_degree,
        num_roi_slices=args.num_slices,
    )

    # Dynamic CLI overrides for ROI ratios
    top_ratio = args.roi_top_ratio if args.roi_top_ratio is not None else args.roi_ratio
    bottom_ratio = args.roi_bottom_ratio

    if (
        top_ratio is not None
        or bottom_ratio is not None
        or args.roi_top_left_x is not None
        or args.roi_top_right_x is not None
        or args.roi_bottom_left_x is not None
        or args.roi_bottom_right_x is not None
    ):
        eff_top = top_ratio if top_ratio is not None else config.get_roi_top_ratio()
        eff_bot = bottom_ratio if bottom_ratio is not None else config.get_roi_bottom_ratio()
        tl_x = args.roi_top_left_x if args.roi_top_left_x is not None else config.roi_top_left_x
        tr_x = args.roi_top_right_x if args.roi_top_right_x is not None else config.roi_top_right_x
        bl_x = args.roi_bottom_left_x if args.roi_bottom_left_x is not None else config.roi_bottom_left_x
        br_x = args.roi_bottom_right_x if args.roi_bottom_right_x is not None else config.roi_bottom_right_x

        config.update_roi_ratios(
            top_ratio=eff_top,
            bottom_ratio=eff_bot,
            polygon_ratios=[
                (bl_x, eff_bot),
                (tl_x, eff_top),
                (tr_x, eff_top),
                (br_x, eff_bot),
            ],
        )

    detector = LaneDetector(config=config)
    visualizer = Visualizer(config=config)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_target_images(input_path)
    print("=" * 70)
    print(f" Classical Computer Vision Lane Line Extractor")
    print(f" Found {len(images)} image(s) to process.")
    print(f" Output Directory: {output_dir.resolve()}")
    print("=" * 70)

    success_count = 0

    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] Processing: {img_path.name}")
        try:
            result = detector.process_image(img_path)

            dims = result.metadata["dimensions"]
            print(f"  Dimensions: {dims['width']}x{dims['height']} pixels")
            print(f"  ROI Ratios:        top={result.metadata.get('roi_top_ratio', config.get_roi_top_ratio()):.2f}, bottom={result.metadata.get('roi_bottom_ratio', config.get_roi_bottom_ratio()):.2f}")
            print(f"  Hough Raw Segments: {result.metadata['raw_hough_segments_count']}")
            print(f"  Left Candidates:   {result.metadata['left_candidate_segments']}")
            print(f"  Right Candidates:  {result.metadata['right_candidate_segments']}")
            print(f"  Detection Status:  {result.status}")

            if result.left_lane:
                lx_b, ly_b, lx_t, ly_t = result.left_lane
                print(f"  Left Lane Line:    ({lx_b}, {ly_b}) -> ({lx_t}, {ly_t})")
            else:
                print("  Left Lane Line:    None detected")

            if result.right_lane:
                rx_b, ry_b, rx_t, ry_t = result.right_lane
                print(f"  Right Lane Line:   ({rx_b}, {ry_b}) -> ({rx_t}, {ry_t})")
            else:
                print("  Right Lane Line:   None detected")

            if result.is_curved:
                rad_str = f"{result.curvature_radius_m:.1f} m" if result.curvature_radius_m else "N/A"
                print(f"  Road Curvature:    {result.curvature_direction} (Radius ~{rad_str})")
            elif result.both_detected:
                print("  Road Curvature:    STRAIGHT")

            # Print messages/warnings
            for msg in result.messages:
                print(f"  -> {msg}")

            # Save artifacts with distinct file stems to avoid clashes across formats
            stem = f"{img_path.stem}_{img_path.suffix.lstrip('.')}" if any(
                p.stem == img_path.stem and p != img_path for p in images
            ) else img_path.stem
            saved = visualizer.save_results(
                result=result,
                output_dir=output_dir,
                file_stem=stem,
                save_individual_stages=not args.no_individual_stages,
            )
            print(f"  Saved Final Overlay:  {saved.get('final_detected')}")
            print(f"  Saved Stages Montage: {saved.get('stages_montage')}")

            # Interactive display if requested
            if args.show:
                try:
                    montage = visualizer.create_stages_montage(result)
                    window_name = f"Lane Detection: {img_path.name} (Press any key)"
                    cv2.imshow(window_name, montage)
                    cv2.waitKey(0)
                    cv2.destroyWindow(window_name)
                except Exception as e:
                    print(f"  [-] Display unavailable: {e}")

            success_count += 1

        except Exception as err:
            print(f"  [-] Error processing {img_path.name}: {err}", file=sys.stderr)

    print("\n" + "=" * 70)
    print(f" Execution Completed! Successfully processed {success_count}/{len(images)} images.")
    print("=" * 70)


if __name__ == "__main__":
    main()

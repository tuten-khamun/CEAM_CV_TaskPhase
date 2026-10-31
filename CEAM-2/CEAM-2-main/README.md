# Classical Computer Vision Highway Lane Line Extractor

A robust, beginner-friendly Python project for extracting highway lane markings from static road images using classical computer vision techniques (**OpenCV** and **NumPy**). 

This project requires **no deep learning**, **no pre-trained neural networks**, and **no external cloud APIs**. It processes raw pixel data through a mathematically grounded classical computer vision pipeline and dynamically adapts to any resolution, aspect ratio, lighting condition, and lane marking color without hardcoded coordinates.

---

## Table of Contents

- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [How the Pipeline Works](#how-the-pipeline-works)
  - [1. Image Ingestion \& Format Validation](#1-image-ingestion--format-validation)
  - [2. HSV Transformation \& Dual-Band Color Filtering](#2-hsv-transformation--dual-band-color-filtering)
  - [3. Contrast Normalization (CLAHE) \& Gaussian Blurring](#3-contrast-normalization-clahe--gaussian-blurring)
  - [4. Dual-Branch Hybrid Canny Edge Detection](#4-dual-branch-hybrid-canny-edge-detection)
  - [5. Dynamic Proportional ROI Masking](#5-dynamic-proportional-roi-masking)
  - [6. Probabilistic Hough Line Detection](#6-probabilistic-hough-line-detection)
  - [7. Slope, Position \& Length Filtering](#7-slope-position--length-filtering)
  - [8. Dominant Ego-Lane Clustering \& Fitting (No Fabrication)](#8-dominant-ego-lane-clustering--fitting-no-fabrication)
  - [9. Visual Overlay \& 6-Panel Stage Montage](#9-visual-overlay--6-panel-stage-montage)
- [Installation \& Setup](#installation--setup)
- [Quick Start Guide](#quick-start-guide)
  - [1. Process Single Images or Folders](#1-process-single-images-or-folders)
  - [2. Interactive Viewing](#2-interactive-viewing)
  - [3. Custom Threshold Tuning via CLI](#3-custom-threshold-tuning-via-cli)
- [Test Suite \& Verification](#test-suite--verification)
- [Evaluation on Sample Images](#evaluation-on-sample-images)
- [Root Cause Analysis of Real-World Edge Cases](#root-cause-analysis-of-real-world-edge-cases)

---

## Key Features

- **Format Independent:** Supports user-provided `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` files.
- **Dynamic Resolution Independence:** Zero hardcoded pixel coordinates. All ROI vertices, line lengths, gaps, and thicknesses scale proportionally with image dimensions ($W \times H$).
- **Color Adaptive:** Simultaneously detects solid/dashed **yellow** markings and solid/dashed **white** markings using HSV thresholding.
- **Ego-Lane Clustering:** On multi-lane highways with multiple lines, automatically isolates the dominant ego-driving corridor and prevents cross-lane line averaging.
- **Strict Anti-Fabrication Guarantee:** Never invents or draws imaginary lane lines. If markings are faded, occluded, or completely absent (e.g. empty asphalt), the algorithm clearly flags the detection state (`NO_LANES_DETECTED` or `PARTIAL_DETECTED`) without crashing.
- **Complete Intermediate Visibility:** Exports every internal processing stage (HSV mask, blurred gray, Canny edges, masked edges, raw Hough segments, and final overlay) along with a consolidated 6-panel educational montage.
- **Zero Heavyweight Dependencies:** Runs lightning fast on standard CPUs using only `opencv-python` and `numpy`.

---

## Project Architecture

```
CEAM - 2/
├── lane_detector/                 # Core computer vision package
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Dataclass with proportional defaults & HSV thresholds
│   ├── detector.py                # LaneDetector class implementing pipeline stages 1-8
│   └── visualizer.py              # Visualizer class for overlays, HUD banners, & montages
├── tests/                         # Automated unit & integration tests
│   ├── __init__.py
│   └── test_lane_detector.py      # Comprehensive 14-test verification suite
├── sample_images/                 # Real-world and synthetic road test scenes
├── output/                        # Saved overlays, stage montages, and intermediate PNGs
├── generate_sample_images.py      # Generator for reproducible synthetic test images
├── main.py                        # CLI entry point for processing images
├── requirements.txt               # opencv-python & numpy
└── README.md                      # Comprehensive documentation
```

---

## How the Pipeline Works

### 1. Image Ingestion & Format Validation
- **Function:** `LaneDetector.load_image(path)`
- **Mechanism:** Validates that the file exists, belongs to an approved image extension (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`), and decodes successfully into an OpenCV BGR numpy array (`cv2.imread`). Handles corrupted or empty files gracefully with descriptive exceptions.

### 2. HSV Transformation & Dual-Band Color Filtering
- **Function:** `LaneDetector.convert_to_hsv()` & `filter_lane_colors()`
- **Mechanism:**
  - Standard RGB/BGR color spaces couple chromaticity (color) with luminance (lighting). Under highway shadows or harsh sunlight, BGR values change wildly.
  - The **HSV (Hue, Saturation, Value)** color space separates color type (**Hue**) from illumination (**Value**):
    - **Yellow Lines:** Isolated via `cv2.inRange()` in the hue band $[10, 42]$ with saturation $S \in [35, 255]$ and brightness $V \in [70, 255]$.
    - **White Lines:** Isolated across any hue ($H \in [0, 179]$) by enforcing low saturation ($S \in [0, 75]$) and high brightness ($V \in [160, 255]$).
  - Both masks are fused using a bitwise OR operation:
    $$\text{Combined Mask} = \text{Yellow Mask} \lor \text{White Mask}$$

### 3. Contrast Normalization (CLAHE) & Gaussian Blurring
- **Function:** `LaneDetector.enhance_contrast()` & `apply_gaussian_blur()`
- **Mechanism:**
  - Real-world road images often suffer from uneven lighting (bright skies vs. shaded roads). We apply **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to the grayscale channel to enhance lane boundary gradients without amplifying noise.
  - A 2D Gaussian kernel of size $k \times k$ (default $5 \times 5$) convolves with the image:
    $$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
    suppressing high-frequency texture while preserving strong structural edges.

### 4. Dual-Branch Hybrid Canny Edge Detection
- **Function:** `LaneDetector.apply_canny_edge_detection()`
- **Mechanism:**
  - Standard grayscale Canny detects high-contrast edges (`low=40`, `high=120`).
  - Color Canny detects edges directly on the blurred color mask.
  - Fusing both branches via bitwise OR guarantees detection even when a lane marking is faintly painted or when the asphalt itself is light gray:
    $$\text{Edges} = \text{Canny}(\text{Gray}) \lor \text{Canny}(\text{ColorMask})$$

### 5. Dynamic Proportional ROI Masking
- **Function:** `LaneDetector.generate_proportional_roi_mask()` & `apply_roi_mask()`
- **Mechanism:**
  - Uses **proportional trapezoid coordinates** scaled dynamically to image width ($W$) and height ($H$):
    - **Bottom-Left:** $(0.05 \times W, 0.93 \times H)$
    - **Top-Left (Apex Horizon):** $(0.30 \times W, 0.42 \times H)$
    - **Top-Right (Apex Horizon):** $(0.70 \times W, 0.42 \times H)$
    - **Bottom-Right:** $(0.95 \times W, 0.93 \times H)$
  - Bounding the base at $0.93 H$ excludes vehicle hoods, dashboards, and bottom watermark banners.
  - Extending the apex up to $0.42 H$ with a wide width ($0.30 W$ to $0.70 W$) captures forward roads and highway lanes converging towards high horizons.

### 6. Probabilistic Hough Line Detection
- **Function:** `LaneDetector.detect_hough_lines()`
- **Mechanism:**
  - Converts edge pixels into lines using the Probabilistic Hough Transform (`cv2.HoughLinesP`).
  - To remain resolution-independent, `minLineLength` and `maxLineGap` are calculated as dynamic fractions of image height:
    - $\text{minLineLength} = \max(8, \lfloor H \times 0.025 \rfloor)$
    - $\text{maxLineGap} = \max(5, \lfloor H \times 0.050 \rfloor)$
  - This allows broken dashes to be bridged seamlessly across 280p, 720p, 1080p, and 4K images.

### 7. Slope, Position & Length Filtering
- **Function:** `LaneDetector.filter_and_classify_lines()`
- **Mechanism:**
  - **Left Lane Lines:** Slant upwards towards the center $\implies \frac{\Delta y}{\Delta x} < 0$. Must lie on the left side ($x_{\text{mid}} \le 0.58 \times W$ and $x_{\text{bottom}} \le 0.58 \times W$).
  - **Right Lane Lines:** Slant downwards away from center $\implies \frac{\Delta y}{\Delta x} > 0$. Must lie on the right side ($x_{\text{mid}} \ge 0.42 \times W$ and $x_{\text{bottom}} \ge 0.42 \times W$).
  - **Slope Filtering:** Segments with $|\text{slope}| < 0.35$ (horizontal crosswalks, shadows, seams $< 19^\circ$) are rejected. Segments up to $|\text{slope}| \le 50.0$ are allowed to capture steep lines near the vanishing point.

### 8. Dominant Ego-Lane Clustering & Fitting (No Fabrication)
- **Function:** `LaneDetector.fit_dominant_lane()`
- **Mechanism:**
  - **The Multi-Lane Highway Problem:** If candidate segments from multiple different lanes are naively averaged together, the fitted line creates an artificial diagonal across the road.
  - **1D Ego-Clustering:** For each candidate segment, the projected bottom intercept is computed:
    $$x_{\text{bottom}} = x_1 + \frac{H - y_1}{\text{slope}}$$
  - Segments belonging to the same physical marking share a tight $x_{\text{bottom}}$ cluster. A length-weighted histogram identifies the dominant ego-lane cluster bordering the vehicle's driving path.
  - Outliers from far-away lanes or roadside clutter are excluded, and inliers are fitted via length-weighted linear regression:
    $$x = m_y \cdot y + b_y$$
  - **Vanishing Point Sanity Check:** Projects line intersection $(x_{\text{int}}, y_{\text{int}})$. If intersection occurs within the road area, $y_{\text{top}}$ is clamped below $y_{\text{int}}$ so that left and right lane lines **never cross**.
  - **Strict No-Fabrication Guarantee:** If candidate segments are absent for a side, that side returns `None`. No imaginary line is fabricated.

### 9. Visual Overlay & 6-Panel Stage Montage
- **Function:** `Visualizer.draw_lanes()` & `create_stages_montage()`
- **Mechanism:**
  - Draws the detected lane boundaries in distinct colors (Red for left, Blue for right).
  - If both lanes are present and non-inverting, shades the drivable corridor polygon with semi-transparent blending (`cv2.addWeighted`).
  - Displays a high-contrast HUD banner reporting detection status and resolution.
  - Assembles all pipeline stages into a neat $2 \times 3$ grid montage for side-by-side inspection.

---

## Installation & Setup

### Prerequisites
- Python 3.8+ (tested and verified on Python 3.10, 3.11, 3.12, 3.13, and 3.14).

### Install Dependencies
```bash
pip install -r requirements.txt
```
*Dependencies:* `opencv-python>=4.8.0`, `numpy>=1.20.0`.

---

## Quick Start Guide

### 1. Process Single Images or Folders
To process every image in `sample_images/`:
```bash
python main.py -i sample_images -o output
```
To process an individual image:
```bash
python main.py -i sample_images/images.jpg -o output
```

### 2. Interactive Viewing
To pop up an interactive OpenCV window for each image:
```bash
python main.py -i sample_images/images.jpg --show
```
*(Press any key to advance to the next image).*

### 3. Custom Threshold Tuning via CLI
```bash
python main.py -i path/to/image.jpg -o output \
    --canny-low 40 \
    --canny-high 120 \
    --gaussian-kernel 5 \
    --hough-threshold 18 \
    --roi-top-ratio 0.42 \
    --roi-bottom-ratio 0.93
```

---

## Test Suite & Verification

Run the automated test suite:
```bash
python -m unittest discover -s tests -v
```

### Test Coverage Summary:
| Test Case | Description | Result |
| :--- | :--- | :---: |
| `test_image_loading_valid_formats` | Tests loading JPG, JPEG, PNG, BMP, WEBP | **PASS** |
| `test_image_loading_nonexistent_file` | Validates `FileNotFoundError` on missing file | **PASS** |
| `test_image_loading_unsupported_format` | Validates `ValueError` on bad extension | **PASS** |
| `test_hsv_conversion` | Checks shape, dtype, and hue band | **PASS** |
| `test_color_filtering_yellow_and_white` | Tests yellow, white, and combined masks | **PASS** |
| `test_gaussian_blur` | Checks noise smoothing & odd kernel enforcement | **PASS** |
| `test_canny_edge_detection` | Validates binary edge localization | **PASS** |
| `test_dynamic_roi_proportional_coordinates` | Verifies dynamic coordinate scaling across 5 resolutions | **PASS** |
| `test_filter_and_classify_lines` | Verifies slope, position, and noise rejection | **PASS** |
| `test_no_fabrication_on_blank_road` | **CRITICAL:** Verifies zero lines fabricated on blank road | **PASS** |
| `test_single_lane_line_no_fabrication` | **CRITICAL:** Verifies single-lane detection without fake line | **PASS** |
| `test_user_sample_images` | **CRITICAL:** Verifies images.jpg, images1.jpg, images2.jpg detection | **PASS** |
| `test_various_resolutions_and_aspect_ratios` | Tests 360p, 720p, 1080p, and square 800x800 | **PASS** |
| `test_visualizer_and_montage` | Verifies overlay drawing and montage generation | **PASS** |

**Outcome:** `14/14 tests passed (100% success rate)`.

---


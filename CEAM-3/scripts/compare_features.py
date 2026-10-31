"""Compare ORB vs SIFT local feature descriptors on consecutive traffic frames.

Executes full pipeline for both descriptors:
1. Feature extraction and keypoint detection.
2. BFMatcher KNN (k=2) descriptor matching.
3. Lowe's ratio test filtering across ratio thresholds [0.60, 0.70, 0.75, 0.80, 0.90].
4. RANSAC geometric verification (affine partial 2D) separating inliers and outliers.
5. Prints side-by-side comparison tables.
6. Generates high-resolution visualization saving to 'assets/matches_orb_vs_sift.png'.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

from vmt.config import FeatureConfig
from vmt.features import (
    BaseFeatureExtractor,
    FeatureStats,
    create_feature_extractor,
    match_and_verify,
    match_descriptors,
    verify_matches,
)
from vmt.visualize import draw_outlined_text


def extract_consecutive_frames(
    video_path: str, frame_idx: int = 120
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract two consecutive frames from video."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at '{video_path}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file '{video_path}'")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret1, f1 = cap.read()
    ret2, f2 = cap.read()
    cap.release()

    if not ret1 or not ret2 or f1 is None or f2 is None:
        raise RuntimeError(f"Could not read consecutive frames at index {frame_idx}")

    return f1, f2


def draw_custom_matches(
    img1: np.ndarray,
    kps1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inlier_mask: np.ndarray,
    title: str,
) -> np.ndarray:
    """Draw matches connecting img1 and img2 side-by-side: inliers green, outliers red."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1 : w1 + w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw separator line
    cv2.line(canvas, (w1, 0), (w1, canvas_h), (80, 80, 80), 2)

    # Draw outliers first (red) so inliers (green) render on top
    for is_inlier_pass in [False, True]:
        for i, m in enumerate(matches):
            is_inlier = bool(inlier_mask[i]) if (inlier_mask is not None and i < len(inlier_mask)) else False
            if is_inlier != is_inlier_pass:
                continue

            pt1 = (int(round(kps1[m.queryIdx].pt[0])), int(round(kps1[m.queryIdx].pt[1])))
            pt2 = (int(round(kps2[m.trainIdx].pt[0])) + w1, int(round(kps2[m.trainIdx].pt[1])))

            color = (0, 220, 0) if is_inlier else (0, 0, 220)  # Green if inlier, Red if outlier
            thick = 1 if is_inlier else 1

            cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)
            cv2.line(canvas, pt1, pt2, color, thick, cv2.LINE_AA)

    # Title badge
    draw_outlined_text(
        canvas,
        title,
        (20, 35),
        font_scale=0.75,
        color=(255, 255, 255),
        thickness=2,
        outline_thickness=4,
    )
    return canvas


def run_comparison(
    video_path: str = "data/traffic.mp4",
    frame_index: int = 120,
    output_image_path: str = "assets/matches_orb_vs_sift.png",
) -> None:
    """Execute feature extraction comparison and save visualizations."""
    img1, img2 = extract_consecutive_frames(video_path, frame_index)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    cfg = FeatureConfig()
    orb_extractor = create_feature_extractor("orb", cfg)
    sift_extractor = create_feature_extractor("sift", cfg)

    # Full chain benchmark at baseline ratio = 0.75
    orb_stats = match_and_verify(orb_extractor, gray1, gray2, ratio_threshold=0.75, ransac_thresh=3.0)
    sift_stats = match_and_verify(sift_extractor, gray1, gray2, ratio_threshold=0.75, ransac_thresh=3.0)

    print("\n" + "=" * 78)
    print(f"       FEATURE EXTRACTOR COMPARISON: ORB vs SIFT (Frames {frame_index} -> {frame_index+1})")
    print("=" * 78)
    print(f"{'Metric':<32} | {'ORB':<20} | {'SIFT':<20}")
    print("-" * 78)
    print(f"{'Keypoints (Frame 1)':<32} | {orb_stats.kps1_count:<20} | {sift_stats.kps1_count:<20}")
    print(f"{'Keypoints (Frame 2)':<32} | {orb_stats.kps2_count:<20} | {sift_stats.kps2_count:<20}")
    print(f"{'Raw KNN (k=2) Matches':<32} | {orb_stats.raw_matches_count:<20} | {sift_stats.raw_matches_count:<20}")
    print(f"{'Matches After Ratio (0.75)':<32} | {orb_stats.ratio_matches_count:<20} | {sift_stats.ratio_matches_count:<20}")
    print(f"{'RANSAC Inliers (threshold=3px)':<32} | {orb_stats.ransac_inliers_count:<20} | {sift_stats.ransac_inliers_count:<20}")
    print(f"{'Inlier Ratio':<32} | {orb_stats.inlier_ratio * 100.0:>6.2f}%              | {sift_stats.inlier_ratio * 100.0:>6.2f}%")
    print(f"{'Feature Extraction Time':<32} | {orb_stats.detect_time_ms:>6.2f} ms            | {sift_stats.detect_time_ms:>6.2f} ms")
    print(f"{'Descriptor Matching Time':<32} | {orb_stats.match_time_ms:>6.2f} ms            | {sift_stats.match_time_ms:>6.2f} ms")
    total_orb_time = orb_stats.detect_time_ms + orb_stats.match_time_ms
    total_sift_time = sift_stats.detect_time_ms + sift_stats.match_time_ms
    print(f"{'Total Execution Time':<32} | {total_orb_time:>6.2f} ms            | {total_sift_time:>6.2f} ms")
    print("=" * 78)

    # Ratio threshold sweep
    sweep_ratios = [0.60, 0.70, 0.75, 0.80, 0.90]
    print("\n" + "=" * 78)
    print("                     LOWE'S RATIO THRESHOLD SWEEP")
    print("=" * 78)
    print(f"{'Ratio':<8} | {'ORB Ratio':<10} | {'ORB Inliers':<11} | {'ORB Inl %':<10} | {'SIFT Ratio':<10} | {'SIFT Inl':<10} | {'SIFT Inl %':<10}")
    print("-" * 78)

    # Pre-extract descriptors for fast sweep
    kps1_orb, desc1_orb = orb_extractor.detect_and_compute(gray1)
    kps2_orb, desc2_orb = orb_extractor.detect_and_compute(gray2)
    kps1_sift, desc1_sift = sift_extractor.detect_and_compute(gray1)
    kps2_sift, desc2_sift = sift_extractor.detect_and_compute(gray2)

    for r in sweep_ratios:
        # ORB
        m_orb, _ = match_descriptors(desc1_orb, desc2_orb, norm_type=orb_extractor.norm_type, ratio_threshold=r)
        _, inliers_orb = verify_matches(kps1_orb, kps2_orb, m_orb, method="affine", ransac_thresh=3.0)
        n_inl_orb = int(np.sum(inliers_orb)) if inliers_orb is not None else 0
        pct_orb = (n_inl_orb / len(m_orb) * 100.0) if len(m_orb) > 0 else 0.0

        # SIFT
        m_sift, _ = match_descriptors(desc1_sift, desc2_sift, norm_type=sift_extractor.norm_type, ratio_threshold=r)
        _, inliers_sift = verify_matches(kps1_sift, kps2_sift, m_sift, method="affine", ransac_thresh=3.0)
        n_inl_sift = int(np.sum(inliers_sift)) if inliers_sift is not None else 0
        pct_sift = (n_inl_sift / len(m_sift) * 100.0) if len(m_sift) > 0 else 0.0

        print(
            f"{r:<8.2f} | {len(m_orb):<10} | {n_inl_orb:<11} | {pct_orb:>8.2f}%  | {len(m_sift):<10} | {n_inl_sift:<10} | {pct_sift:>8.2f}%"
        )
    print("=" * 78 + "\n")

    # Generate visual comparison: stack ORB visualization over SIFT visualization
    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)

    title_orb = (
        f"ORB: {orb_stats.ransac_inliers_count} inliers (green), "
        f"{orb_stats.ratio_matches_count - orb_stats.ransac_inliers_count} outliers (red) | "
        f"Inlier Ratio: {orb_stats.inlier_ratio*100:.1f}% | {total_orb_time:.1f}ms"
    )
    orb_vis = draw_custom_matches(
        img1,
        kps1_orb,
        img2,
        kps2_orb,
        orb_stats.matches_after_ratio,
        orb_stats.inlier_mask,
        title=title_orb,
    )

    title_sift = (
        f"SIFT: {sift_stats.ransac_inliers_count} inliers (green), "
        f"{sift_stats.ratio_matches_count - sift_stats.ransac_inliers_count} outliers (red) | "
        f"Inlier Ratio: {sift_stats.inlier_ratio*100:.1f}% | {total_sift_time:.1f}ms"
    )
    sift_vis = draw_custom_matches(
        img1,
        kps1_sift,
        img2,
        kps2_sift,
        sift_stats.matches_after_ratio,
        sift_stats.inlier_mask,
        title=title_sift,
    )

    combined = np.vstack([orb_vis, sift_vis])
    cv2.imwrite(output_image_path, combined)
    print(f"Saved side-by-side comparison to '{output_image_path}' ({os.path.getsize(output_image_path)/1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ORB vs SIFT feature matching on consecutive frames")
    parser.add_argument("--input", default="data/traffic.mp4", help="Video path")
    parser.add_argument("--frame", type=int, default=120, help="Starting frame index")
    parser.add_argument("--output", default="assets/matches_orb_vs_sift.png", help="Output PNG path")
    args = parser.parse_args()

    run_comparison(video_path=args.input, frame_index=args.frame, output_image_path=args.output)

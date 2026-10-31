"""I/O utilities for video stream handling, CSV serialization, and summary analytics."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vmt.tracker import FrameTrackRecord


CSV_COLUMNS = [
    "frame",
    "track_id",
    "cx",
    "cy",
    "w",
    "h",
    "dx",
    "dy",
    "speed_px",
    "n_points",
    "inlier_ratio",
    "state",
]


def write_tracks_csv(records: List[FrameTrackRecord], output_path: str) -> None:
    """Serialize tracking records to a clean CSV file with no NaNs.

    Args:
        records: List of FrameTrackRecord instances.
        output_path: Target destination CSV file path.

    Raises:
        ValueError: If any numeric field contains NaN or Inf.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)

        for rec in records:
            # Validate absence of NaNs
            for field_name in ["cx", "cy", "w", "h", "dx", "dy", "speed_px", "inlier_ratio"]:
                val = getattr(rec, field_name)
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Encountered invalid numeric value '{val}' in column '{field_name}' at frame {rec.frame}")

            writer.writerow(
                [
                    rec.frame,
                    rec.track_id,
                    f"{rec.cx:.2f}",
                    f"{rec.cy:.2f}",
                    f"{rec.w:.2f}",
                    f"{rec.h:.2f}",
                    f"{rec.dx:.2f}",
                    f"{rec.dy:.2f}",
                    f"{rec.speed_px:.2f}",
                    rec.n_points,
                    f"{rec.inlier_ratio:.3f}",
                    rec.state,
                ]
            )


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    """Safely open a video capture stream, validating file existence and readability.

    Args:
        video_path: Path to target video file.

    Returns:
        Opened cv2.VideoCapture object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        RuntimeError: If OpenCV fails to decode video stream.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: '{video_path}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video file '{video_path}'. Check codec support.")

    return cap


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    fourcc_code: str = "mp4v",
) -> cv2.VideoWriter:
    """Initialize a video writer for MP4 output.

    Args:
        output_path: Destination path for video.
        fps: Playback frame rate.
        frame_size: (width, height) tuple.
        fourcc_code: 4-character codec code (default 'mp4v').

    Returns:
        Configured cv2.VideoWriter object.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        # Fallback to alternate fourcc
        fallback_fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fallback_fourcc, fps, frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to initialize VideoWriter for '{output_path}'.")
    return writer


def print_cli_summary(
    total_frames: int,
    records: List[FrameTrackRecord],
    reacquisitions: int,
    elapsed_sec: float,
    detector_name: str,
    feature_name: str,
) -> None:
    """Print structured summary metrics to standard output."""
    fps = total_frames / elapsed_sec if elapsed_sec > 0 else 0.0

    # Group records by track_id
    track_lifetimes: dict[int, int] = {}
    track_states: dict[int, set[str]] = {}
    inlier_ratios: list[float] = []

    for r in records:
        track_lifetimes[r.track_id] = track_lifetimes.get(r.track_id, 0) + 1
        if r.track_id not in track_states:
            track_states[r.track_id] = set()
        track_states[r.track_id].add(r.state)
        inlier_ratios.append(r.inlier_ratio)

    unique_tracks = len(track_lifetimes)
    confirmed_tracks = sum(1 for states in track_states.values() if "CONFIRMED" in states)
    lifetimes = list(track_lifetimes.values())
    mean_lifetime = float(np.mean(lifetimes)) if lifetimes else 0.0
    median_lifetime = float(np.median(lifetimes)) if lifetimes else 0.0
    mean_inlier_ratio = float(np.mean(inlier_ratios)) if inlier_ratios else 0.0

    separator = "=" * 60
    print(f"\n{separator}")
    print("           VEHICLE MOTION TRACKER: RUN SUMMARY")
    print(separator)
    print(f" Detector                  : {detector_name}")
    print(f" Feature Matcher           : {feature_name}")
    print(f" Processed Frames          : {total_frames}")
    print(f" Processing Time           : {elapsed_sec:.2f} s")
    print(f" Mean Throughput           : {fps:.2f} FPS")
    print(f" Unique Tracks Spawned     : {unique_tracks}")
    print(f" Confirmed Tracks          : {confirmed_tracks}")
    print(f" Mean Track Lifetime       : {mean_lifetime:.1f} frames")
    print(f" Median Track Lifetime     : {median_lifetime:.1f} frames")
    print(f" Re-acquisitions           : {reacquisitions}")
    print(f" Mean Inlier Ratio         : {mean_inlier_ratio:.3f}")
    print(f"{separator}\n")

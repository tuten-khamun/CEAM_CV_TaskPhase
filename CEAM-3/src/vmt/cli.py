"""Command-Line Interface (CLI) entrypoint for the Vehicle Motion Tracker."""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Optional

import cv2
import numpy as np

from vmt.config import TrackerConfig
from vmt.detector import MOG2Detector
from vmt.io_utils import (
    create_video_writer,
    open_video_capture,
    print_cli_summary,
    write_tracks_csv,
)
from vmt.tracker import VehicleTracker
from vmt.visualize import Visualizer


def set_seed(seed: int = 42) -> None:
    """Ensure full reproducibility by seeding standard random generators."""
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Classical Vehicle Motion Tracker (OpenCV, Shi-Tomasi, LK Optical Flow, ORB/SIFT RANSAC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/traffic.mp4",
        help="Path to input traffic video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/annotated.mp4",
        help="Path to save annotated output video",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/tracks.csv",
        help="Path to save trajectory CSV records",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="mog2",
        choices=["mog2"],
        help="Motion detection algorithm",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="orb",
        choices=["orb", "sift"],
        help="Local feature descriptor for re-acquisition",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based frame index to begin processing",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=1280,
        help="Target frame width for processing and rendering (preserves aspect ratio; -1 to disable)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live GUI playback window (interactive mode)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include MOG2 foreground mask alongside video in output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main execution function."""
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)

    # 1. Safely open input video stream
    try:
        cap = open_video_capture(args.input)
    except Exception as exc:
        print(f"Error opening video input: {exc}", file=sys.stderr)
        return 1

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.resize_width is not None and args.resize_width > 0 and args.resize_width != orig_width:
        target_w = args.resize_width
        target_h = int(round(orig_height * (target_w / orig_width)))
    else:
        target_w = orig_width
        target_h = orig_height

    out_w = target_w * 2 if args.debug else target_w
    out_h = target_h

    # Fast-forward to start frame
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # 2. Initialize tracking components
    cfg = TrackerConfig()
    detector = MOG2Detector(config=cfg.detector)
    tracker = VehicleTracker(config=cfg, detector=detector, feature_name=args.feature)
    visualizer = Visualizer(config=cfg.viz)

    # 3. Initialize video writer if requested
    writer = None
    if args.output:
        try:
            writer = create_video_writer(args.output, fps=fps, frame_size=(out_w, out_h))
        except Exception as exc:
            print(f"Error initializing video writer: {exc}", file=sys.stderr)
            cap.release()
            return 1

    all_records = []
    frame_count = 0
    t_start = time.perf_counter()
    prev_time = t_start
    measured_fps = fps

    print(f"Tracking '{args.input}' -> '{args.output}' [{target_w}x{target_h} @ {fps:.1f} fps]...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if args.resize_width is not None and args.resize_width > 0 and (target_w != orig_width):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Execution timing for HUD
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                measured_fps = 0.9 * measured_fps + 0.1 * (1.0 / dt)

            # Step tracker
            records = tracker.step(frame)
            all_records.extend(records)

            # Render visualization
            debug_mask = detector.debug_mask if args.debug else None
            rendered = visualizer.render(
                frame=frame,
                tracks=tracker.tracks,
                frame_idx=tracker.frame_idx,
                fps=measured_fps,
                detector_name=args.detector.upper(),
                feature_name=args.feature.upper(),
                debug_mask=debug_mask,
            )

            if writer is not None:
                writer.write(rendered)

            if args.show:
                cv2.imshow("Vehicle Motion Tracker", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start

    # 4. Serialize CSV
    if args.csv:
        write_tracks_csv(all_records, args.csv)

    # 5. Print summary
    print_cli_summary(
        total_frames=frame_count,
        records=all_records,
        reacquisitions=tracker.reacquisition_count,
        elapsed_sec=elapsed,
        detector_name=args.detector.upper(),
        feature_name=args.feature.upper(),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

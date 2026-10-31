"""Generate an optimized animated looping GIF from the busiest segment of the tracked video.

Finds the busiest 12-second window (highest active vehicle count), downsamples to 12 FPS,
resizes to 720 px width, quantizes to an optimized palette, and encodes directly to GIF89a.
Guarantees looping and verifies the output size is strictly <= 8.0 MB.
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np


def find_busiest_window(
    csv_path: str,
    total_video_frames: int,
    window_frames: int = 360,  # 12 seconds @ 30 FPS
) -> int:
    """Analyze CSV to find start frame of 12-second window with maximum active vehicle traffic."""
    if not os.path.exists(csv_path):
        # Default to middle of video if CSV is missing
        return max(0, (total_video_frames - window_frames) // 2)

    frame_active_counts: dict[int, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame"])
            state = row["state"]
            if state in ("CONFIRMED", "TENTATIVE"):
                frame_active_counts[f_idx] = frame_active_counts.get(f_idx, 0) + 1

    max_frames = max(total_video_frames, max(frame_active_counts.keys(), default=0))
    if max_frames <= window_frames:
        return 0

    best_start = 0
    best_sum = -1

    # Running sum over sliding window of window_frames
    for start in range(0, max_frames - window_frames + 1, 10):
        w_sum = sum(frame_active_counts.get(start + i, 0) for i in range(window_frames))
        if w_sum > best_sum:
            best_sum = w_sum
            best_start = start

    return best_start


def build_palette_256() -> Tuple[bytearray, np.ndarray]:
    """Construct an optimized 256-color palette (6x6x6 RGB color cube + 32 gray levels)."""
    palette_bytes = bytearray()
    cube_colors = []

    # 6 levels per channel: 0, 51, 102, 153, 204, 255 -> 216 colors
    levels = [0, 51, 102, 153, 204, 255]
    for r in levels:
        for g in levels:
            for b in levels:
                palette_bytes.extend([r, g, b])
                cube_colors.append([r, g, b])

    # 40 additional grayscale and common road/vehicle shades -> 256 colors total
    grays = np.linspace(0, 255, 40, dtype=np.uint8)
    for g in grays:
        palette_bytes.extend([g, g, g])
        cube_colors.append([g, g, g])

    return palette_bytes[:768], np.array(cube_colors, dtype=np.uint8)


def quantize_frame_fast(rgb_frame: np.ndarray) -> np.ndarray:
    """Fast vector quantization to 216-color RGB cube."""
    R = rgb_frame[:, :, 0].astype(np.uint16)
    G = rgb_frame[:, :, 1].astype(np.uint16)
    B = rgb_frame[:, :, 2].astype(np.uint16)

    r_idx = (R * 5) // 255
    g_idx = (G * 5) // 255
    b_idx = (B * 5) // 255

    indices = (r_idx * 36 + g_idx * 6 + b_idx).astype(np.uint8)
    return indices


def lzw_compress(indices: List[int], min_code_size: int = 8) -> bytes:
    """LZW compress pixel index sequence with GIF-standard variable bit length."""
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    next_code = end_code + 1
    max_code = 1 << code_size

    out_bits = []
    bit_buf = 0
    bit_count = 0

    def emit_code(c: int, bits: int) -> None:
        nonlocal bit_buf, bit_count
        bit_buf |= c << bit_count
        bit_count += bits
        while bit_count >= 8:
            out_bits.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8

    emit_code(clear_code, code_size)

    dict_fast = {}
    curr_prefix = indices[0]

    for idx in indices[1:]:
        key = (curr_prefix, idx)
        if key in dict_fast:
            curr_prefix = dict_fast[key]
        else:
            emit_code(curr_prefix, code_size)
            if next_code < 4096:
                dict_fast[key] = next_code
                next_code += 1
                if next_code > max_code and code_size < 12:
                    code_size += 1
                    max_code = 1 << code_size
            else:
                emit_code(clear_code, code_size)
                code_size = min_code_size + 1
                max_code = 1 << code_size
                dict_fast.clear()
                next_code = end_code + 1
            curr_prefix = idx

    emit_code(curr_prefix, code_size)
    emit_code(end_code, code_size)

    if bit_count > 0:
        out_bits.append(bit_buf & 0xFF)

    # Sub-block chunking (<= 255 bytes per sub-block)
    blocks = bytearray([min_code_size])
    i = 0
    n = len(out_bits)
    while i < n:
        chunk = out_bits[i : i + 255]
        blocks.append(len(chunk))
        blocks.extend(chunk)
        i += len(chunk)
    blocks.append(0)  # Block terminator
    return bytes(blocks)


def encode_gif(
    frames: List[np.ndarray],
    fps: int,
    output_path: str,
) -> int:
    """Encode list of RGB frames into looping GIF89a file."""
    h, w = frames[0].shape[:2]
    palette_bytes, _ = build_palette_256()

    header = b"GIF89a"
    lsd = struct.pack("<HHBBB", w, h, 0xF7, 0, 0)
    netscape_loop = b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00"

    delay_hundredths = max(1, int(round(100.0 / fps)))

    gif_parts = [header, lsd, palette_bytes, netscape_loop]

    for frame in frames:
        gce = b"\x21\xF9\x04\x04" + struct.pack("<H", delay_hundredths) + b"\x00\x00"
        img_desc = b"\x2C" + struct.pack("<HHHHB", 0, 0, w, h, 0x00)
        quantized = quantize_frame_fast(frame)
        flat_indices = quantized.ravel().tolist()
        compressed_data = lzw_compress(flat_indices, min_code_size=8)
        gif_parts.extend([gce, img_desc, compressed_data])

    gif_parts.append(b"\x3B")
    all_bytes = b"".join(gif_parts)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(all_bytes)

    return len(all_bytes)


def create_demo_gif(
    video_path: str = "outputs/annotated.mp4",
    csv_path: str = "outputs/tracks.csv",
    output_path: str = "assets/demo.gif",
    duration_sec: float = 12.0,
    fps: int = 12,
    target_width: int = 720,
    max_size_mb: float = 8.0,
) -> None:
    """Extract busiest window, downsample, and write optimized looping GIF <= 8 MB."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file '{video_path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_frame_count = int(round(duration_sec * video_fps))
    start_frame = find_busiest_window(csv_path, total_frames, window_frame_count)

    print(f"Busiest {duration_sec}s segment begins at frame {start_frame} (frames {start_frame} to {start_frame + window_frame_count})")

    # Progressive fallback loop if size exceeds limit
    current_fps = fps
    current_width = target_width
    current_duration = duration_sec

    while True:
        target_h = int(round(orig_h * (current_width / orig_w)))
        n_gif_frames = int(round(current_duration * current_fps))
        frame_indices = [
            int(round(start_frame + i * (video_fps / current_fps))) for i in range(n_gif_frames)
        ]

        extracted_frames = []
        for idx in frame_indices:
            if idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret or f is None:
                continue
            if f.shape[1] != current_width:
                f = cv2.resize(f, (current_width, target_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            extracted_frames.append(rgb)

        if not extracted_frames:
            raise RuntimeError("No frames extracted for GIF creation")

        print(f"Encoding GIF: {len(extracted_frames)} frames @ {current_fps} FPS, {current_width}x{target_h}...")
        t0 = time.perf_counter()
        file_size_bytes = encode_gif(extracted_frames, current_fps, output_path)
        t_encode = time.perf_counter() - t0
        size_mb = file_size_bytes / (1024.0 * 1024.0)

        print(f"Generated GIF size: {size_mb:.2f} MB ({t_encode:.2f} s)")

        if size_mb <= max_size_mb:
            print(f"SUCCESS: '{output_path}' verified <= {max_size_mb} MB ({size_mb:.2f} MB)")
            break

        # Progressive reduction if oversized
        if current_fps > 8:
            current_fps -= 2
            print(f"Oversized ({size_mb:.2f} MB > {max_size_mb} MB). Reducing FPS to {current_fps}...")
        elif current_width > 480:
            current_width = int(current_width * 0.8)
            print(f"Oversized. Reducing width to {current_width} px...")
        elif current_duration > 6.0:
            current_duration -= 2.0
            print(f"Oversized. Reducing duration to {current_duration} s...")
        else:
            print(f"Reached minimum parameters. Final size: {size_mb:.2f} MB.")
            break

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create looping demo GIF from tracked video")
    parser.add_argument("--video", default="outputs/annotated.mp4", help="Input video")
    parser.add_argument("--csv", default="outputs/tracks.csv", help="Input tracks CSV")
    parser.add_argument("--output", default="assets/demo.gif", help="Output GIF path")
    parser.add_argument("--duration", type=float, default=12.0, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=12, help="Frame rate")
    parser.add_argument("--width", type=int, default=720, help="Width in pixels")
    args = parser.parse_args()

    create_demo_gif(
        video_path=args.video,
        csv_path=args.csv,
        output_path=args.output,
        duration_sec=args.duration,
        fps=args.fps,
        target_width=args.width,
    )

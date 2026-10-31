# Traffic Video Dataset Specification

This directory holds input video sequences for the Vehicle Motion Tracker.
Per project requirements, video binary files (`*.mp4`, `*.avi`, etc.) are gitignored.

## Current Video: Synthetic Daylight Highway Benchmark (`data/traffic.mp4`)

- **Resolution:** 1280 x 720 (720p HD)
- **Frame Rate:** 30.0 FPS
- **Duration:** 16.0 seconds (480 frames)
- **Generator:** `scripts/generate_traffic_video.py`
- **Licence:** CC0 / Public Domain (Synthetic Procedural Generation)
- **Description:** Features a 3-lane daylight highway with structured asphalt texture, painted lane lines, and 9 passing vehicles across three lanes (sedans, SUVs, and vans). Vehicles are 85 to 115 px wide (exceeding the >= 40 px specification), with visual structures (hoods, windshields, headlights, taillights, roof panels), cast shadows (luminance ~125), and realistic highway velocities (4.2 to 7.0 px/frame).

## Using a Real Traffic Clip

To run the pipeline on an external real-world traffic clip:
1. Place your fixed-camera daylight video file at `data/traffic.mp4`.
2. Recommended specifications:
   - Fixed camera (no panning, tilting, or ego-motion)
   - Daylight illumination
   - Vehicles at least ~40 px wide
   - Resolution: 720p (1280x720) or 1080p (1920x1080)
3. Execute the tracker:
   ```bash
   python -m vmt.cli --input data/traffic.mp4 --output outputs/annotated.mp4 --csv outputs/tracks.csv
   ```

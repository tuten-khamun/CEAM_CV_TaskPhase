"""Configuration module for Vehicle Motion Tracker (VMT).

Defines strongly-typed dataclasses for every tunable parameter across
detection, feature extraction, optical flow, tracking lifecycle, and visualization.
Every parameter includes units and an explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DetectorConfig:
    """Tunable parameters for background subtraction and contour filtering."""

    # History length for Gaussian Mixture Model (GMM) background modeling [frames]
    history: int = 500

    # Mahalanobis distance squared threshold for background classification [variance units]
    var_threshold: float = 25.0

    # Whether MOG2 flags shadow pixels with value 127 [-]
    detect_shadows: bool = True

    # Initial frames dedicated solely to building background model before detections emit [frames]
    warmup_frames: int = 30

    # Binary threshold applied to suppress gray shadow pixels (value 127) to 0 [0-255]
    shadow_threshold: int = 250

    # Median filter kernel aperture size for salt-and-pepper noise suppression [pixels, odd]
    median_blur_ksize: int = 5

    # Elliptical structuring element size for morphological opening (drops false noise) [pixels]
    morph_open_ksize: Tuple[int, int] = (3, 3)

    # Elliptical structuring element size for morphological closing (bridges vehicle holes) [pixels]
    morph_close_ksize: Tuple[int, int] = (7, 7)

    # Structuring element size for dilation to connect separated vehicle parts [pixels]
    morph_dilate_ksize: Tuple[int, int] = (3, 3)

    # Number of iterations for post-closing dilation [-]
    morph_dilate_iterations: int = 1

    # Base minimum bounding box area at 1280x720 (720p) resolution [px^2]
    # Scaled proportionally for different frame resolutions: area * (W*H) / (1280*720)
    min_area_720p: float = 800.0

    # Maximum bounding box area expressed as fraction of total frame area [ratio]
    max_area_ratio: float = 0.40

    # Minimum bounding box aspect ratio (width / height) to discard tall vertical artifacts [-]
    min_aspect_ratio: float = 0.30

    # Maximum bounding box aspect ratio (width / height) to discard wide horizontal artifacts [-]
    max_aspect_ratio: float = 4.00

    # Intersection-over-Union (IoU) threshold for merging overlapping bounding boxes [-]
    merge_iou_threshold: float = 0.30

    # Containment threshold (intersection / min(area1, area2)) to merge nested boxes [-]
    merge_containment_threshold: float = 0.80


@dataclass
class FeatureConfig:
    """Tunable parameters for ORB/SIFT feature extraction and geometric verification."""

    # Maximum number of keypoints retained by ORB detector [-]
    orb_nfeatures: int = 1500

    # Pyramid decimation ratio for multi-scale ORB extraction [-]
    orb_scale_factor: float = 1.2

    # Number of pyramid levels in ORB [-]
    orb_nlevels: int = 8

    # Margin border size where ORB keypoints are not sampled [pixels]
    orb_edge_threshold: int = 15

    # FAST corner detector threshold for ORB keypoint acceptance [-]
    orb_fast_threshold: int = 10

    # Maximum number of keypoints retained by SIFT detector [-]
    sift_nfeatures: int = 1500

    # Lowe's ratio test threshold (first_dist / second_dist) for ambiguous match rejection [-]
    ratio_test_threshold: float = 0.75

    # RANSAC reprojection error threshold for geometric verification [pixels]
    ransac_reproj_threshold: float = 3.0

    # RANSAC confidence level (probability of drawing at least one outlier-free sample) [ratio]
    ransac_confidence: float = 0.99


@dataclass
class OpticalFlowConfig:
    """Tunable parameters for Lucas-Kanade pyramidal sparse optical flow."""

    # Size of the search window at each pyramid level [pixels]
    win_size: Tuple[int, int] = (21, 21)

    # 0-based maximal pyramid level number (3 corresponds to 4 resolution levels) [-]
    max_level: int = 3

    # Termination criteria: maximum solver iterations [-]
    criteria_max_count: int = 30

    # Termination criteria: convergence displacement threshold [pixels]
    criteria_epsilon: float = 0.01

    # Forward-backward consistency threshold: max allowed deviation |p0 - p0_back| [pixels]
    fb_max_distance: float = 1.0


@dataclass
class TrackingConfig:
    """Tunable parameters for track seeding, motion filtering, lifecycle, and re-acquisition."""

    # Margin fraction shrunk on each side of detection box before seeding corners [ratio]
    # Prevents seeding points on asphalt or shadow boundaries
    seed_box_margin: float = 0.15

    # Maximum number of Shi-Tomasi corners to seed per vehicle track [-]
    st_max_corners: int = 60

    # Shi-Tomasi minimal accepted eigenvalue quality level relative to maximum [ratio]
    st_quality_level: float = 0.01

    # Minimum Euclidean distance enforced between seeded Shi-Tomasi corners [pixels]
    st_min_distance: float = 5.0

    # Size of averaging block for Shi-Tomasi structure tensor derivative integration [pixels]
    st_block_size: int = 7

    # Absolute minimum surviving inlier points required to maintain track [points]
    min_inliers_lost: int = 6

    # Minimum inlier ratio threshold (inliers / valid_points) [-]
    inlier_ratio_lost_threshold: float = 0.40

    # Number of consecutive frames below inlier ratio threshold before marking track lost [frames]
    inlier_ratio_lost_consecutive_frames: int = 3

    # Maximum single-frame bounding box area change factor before marking track lost [-]
    max_area_change_factor: float = 3.0

    # Fraction of initial points below which point re-seeding is triggered [ratio]
    reseed_points_ratio: float = 0.40

    # Periodic interval in frames to refresh tracked feature points [frames]
    reseed_interval_frames: int = 10

    # Periodic interval in frames to execute Hungarian detection-to-track association [frames]
    assoc_interval_frames: int = 5

    # Minimum IoU required to accept a Hungarian detection-to-track assignment [-]
    assoc_min_iou: float = 0.30

    # Weight assigned to current optical flow predicted box during box blending [-]
    blend_track_weight: float = 0.70

    # Weight assigned to matched MOG2 detection box during box blending [-]
    blend_detection_weight: float = 0.30

    # Number of detection hits required to promote a tentative track to confirmed [hits]
    tentative_confirm_hits: int = 3

    # Maximum consecutive missed association frames before permanently deleting a track [frames]
    max_misses_delete: int = 15

    # Overlap threshold (IoU) between tracks where the younger duplicate track is culled [-]
    dedup_iou_threshold: float = 0.60

    # Expansion factor defining search window around lost track for feature re-acquisition [-]
    reacq_window_scale: float = 1.50

    # Maximum grace frames a lost track remains eligible for descriptor re-acquisition [frames]
    reacq_max_lost_age: int = 20

    # Minimum RANSAC inlier matches required to successfully re-acquire a lost track [matches]
    reacq_min_inliers: int = 12


@dataclass
class VisualizationConfig:
    """Tunable parameters for rendering annotated video, HUD, and debug streams."""

    # Number of historical track center points retained for motion trail rendering [frames]
    trail_length: int = 30

    # Multiplier applied to optical flow vectors for clear arrow visualization [-]
    flow_arrow_scale: float = 3.0

    # Font scale for HUD on-screen text [-]
    hud_font_scale: float = 0.60

    # Text stroke thickness for HUD text [pixels]
    hud_thickness: int = 1

    # Contrast outline thickness for HUD text [pixels]
    hud_outline_thickness: int = 3

    # Line thickness for vehicle bounding boxes [pixels]
    box_thickness: int = 2


@dataclass
class TrackerConfig:
    """Root configuration aggregating all sub-system configurations."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)

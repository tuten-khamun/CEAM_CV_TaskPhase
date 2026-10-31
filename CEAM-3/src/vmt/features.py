"""Feature extraction, descriptor matching, and geometric verification module.

Provides a unified factory for ORB (binary, Hamming) and SIFT (gradient-histogram, L2),
Lowe's ratio test filtering, and RANSAC-based geometric verification (Affine / Homography).
All routines are guaranteed not to crash on empty or degenerate inputs.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vmt.config import FeatureConfig


@dataclass
class FeatureStats:
    """Statistics collected from a feature extraction, matching, and verification run."""

    extractor_name: str
    kps1_count: int
    kps2_count: int
    raw_matches_count: int
    ratio_matches_count: int
    ransac_inliers_count: int
    inlier_ratio: float
    detect_time_ms: float
    match_time_ms: float
    transform_matrix: Optional[np.ndarray] = None
    matches_after_ratio: List[cv2.DMatch] = None  # type: ignore[assignment]
    inlier_mask: Optional[np.ndarray] = None


class BaseFeatureExtractor(ABC):
    """Abstract base class for classical local feature extractors."""

    def __init__(self, name: str, norm_type: int) -> None:
        self.name = name
        self.norm_type = norm_type

    @abstractmethod
    def detect_and_compute(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect keypoints and compute descriptors on a grayscale image.

        Args:
            image: Grayscale uint8 image.
            mask: Optional uint8 binary mask of same dimensions.

        Returns:
            Tuple of (list of cv2.KeyPoint, numpy array of descriptors or None).
        """
        pass


class ORBExtractor(BaseFeatureExtractor):
    """ORB (Oriented FAST and Rotated BRIEF) feature extractor."""

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        cfg = config or FeatureConfig()
        super().__init__(name="ORB", norm_type=cv2.NORM_HAMMING)
        self.orb = cv2.ORB_create(
            nfeatures=cfg.orb_nfeatures,
            scaleFactor=cfg.orb_scale_factor,
            nlevels=cfg.orb_nlevels,
            edgeThreshold=cfg.orb_edge_threshold,
            fastThreshold=cfg.orb_fast_threshold,
        )

    def detect_and_compute(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if image is None or image.size == 0:
            return [], None
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, desc = self.orb.detectAndCompute(image, mask)
        return list(kps), desc


class SIFTExtractor(BaseFeatureExtractor):
    """SIFT (Scale-Invariant Feature Transform) feature extractor."""

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        cfg = config or FeatureConfig()
        super().__init__(name="SIFT", norm_type=cv2.NORM_L2)
        self.sift = cv2.SIFT_create(nfeatures=cfg.sift_nfeatures)

    def detect_and_compute(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if image is None or image.size == 0:
            return [], None
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, desc = self.sift.detectAndCompute(image, mask)
        return list(kps), desc


def create_feature_extractor(
    name: str = "orb", config: Optional[FeatureConfig] = None
) -> BaseFeatureExtractor:
    """Factory function to instantiate feature extractors by name.

    Args:
        name: 'orb' or 'sift' (case-insensitive).
        config: FeatureConfig instance or None for defaults.

    Returns:
        BaseFeatureExtractor implementation.

    Raises:
        ValueError: If an unsupported extractor name is provided.
    """
    normalized = name.strip().lower()
    if normalized == "orb":
        return ORBExtractor(config)
    elif normalized == "sift":
        return SIFTExtractor(config)
    else:
        raise ValueError(f"Unknown feature extractor '{name}'. Supported: 'orb', 'sift'.")


def match_descriptors(
    desc1: Optional[np.ndarray],
    desc2: Optional[np.ndarray],
    norm_type: int = cv2.NORM_HAMMING,
    ratio_threshold: float = 0.75,
) -> Tuple[List[cv2.DMatch], int]:
    """Perform k-nearest-neighbors (k=2) matching and filter with Lowe's ratio test.

    Lowe's ratio test states that a match is unambiguous if the distance to the
    nearest neighbor is significantly smaller than the distance to the second-nearest:
        distance(best) < ratio_threshold * distance(second_best)

    Args:
        desc1: Descriptors from query image (N, D).
        desc2: Descriptors from reference image (M, D).
        norm_type: cv2.NORM_HAMMING or cv2.NORM_L2.
        ratio_threshold: Threshold ratio in (0, 1]. Default 0.75.

    Returns:
        Tuple of (list of accepted cv2.DMatch, total raw candidate pairs count).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return [], 0

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    try:
        raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return [], 0

    total_raw = len(raw_matches)
    ratio_matches: List[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                ratio_matches.append(m)

    return ratio_matches, total_raw


def verify_matches(
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    method: str = "affine",
    ransac_thresh: float = 3.0,
    confidence: float = 0.99,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Geometrically verify matches using RANSAC.

    Fits an affine partial 2D (scale, rotation, translation - 4 DOF) or a full
    planar homography (8 DOF) and separates consensus inliers from outliers.

    Args:
        kps1: Keypoints from query image.
        kps2: Keypoints from reference image.
        matches: List of cv2.DMatch passing Lowe's ratio test.
        method: 'affine' (requires >= 3 points) or 'homography' (requires >= 4 points).
        ransac_thresh: Max reprojection error in pixels to count as inlier.
        confidence: RANSAC confidence level.

    Returns:
        Tuple of (transform_matrix or None, boolean inlier mask of shape (len(matches),)).
    """
    n_matches = len(matches)
    if n_matches == 0:
        return None, np.zeros(0, dtype=bool)

    min_points = 4 if method == "homography" else 3
    if n_matches < min_points:
        return None, np.zeros(n_matches, dtype=bool)

    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    try:
        if method == "homography":
            matrix, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, ransac_thresh, confidence=confidence
            )
        else:
            matrix, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh, confidence=confidence
            )
    except cv2.error:
        return None, np.zeros(n_matches, dtype=bool)

    if matrix is None or mask is None:
        return None, np.zeros(n_matches, dtype=bool)

    inlier_mask = mask.ravel().astype(bool)
    return matrix, inlier_mask


def match_and_verify(
    extractor: BaseFeatureExtractor,
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: Optional[np.ndarray] = None,
    mask2: Optional[np.ndarray] = None,
    ratio_threshold: float = 0.75,
    ransac_thresh: float = 3.0,
    method: str = "affine",
) -> FeatureStats:
    """Run full pipeline: extraction -> KNN match -> ratio test -> RANSAC verification.

    Args:
        extractor: Feature extractor instance.
        img1: Query image (grayscale or BGR).
        img2: Reference image (grayscale or BGR).
        mask1: Optional query mask.
        mask2: Optional reference mask.
        ratio_threshold: Lowe's ratio test threshold.
        ransac_thresh: RANSAC inlier threshold in pixels.
        method: 'affine' or 'homography'.

    Returns:
        FeatureStats object with complete metrics.
    """
    t0 = time.perf_counter()
    kps1, desc1 = extractor.detect_and_compute(img1, mask1)
    kps2, desc2 = extractor.detect_and_compute(img2, mask2)
    t_detect = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    ratio_matches, raw_count = match_descriptors(
        desc1, desc2, norm_type=extractor.norm_type, ratio_threshold=ratio_threshold
    )
    matrix, inliers = verify_matches(
        kps1, kps2, ratio_matches, method=method, ransac_thresh=ransac_thresh
    )
    t_match = (time.perf_counter() - t1) * 1000.0

    n_inliers = int(np.sum(inliers)) if inliers is not None and len(inliers) > 0 else 0
    inlier_ratio = (n_inliers / len(ratio_matches)) if len(ratio_matches) > 0 else 0.0

    return FeatureStats(
        extractor_name=extractor.name,
        kps1_count=len(kps1),
        kps2_count=len(kps2),
        raw_matches_count=raw_count,
        ratio_matches_count=len(ratio_matches),
        ransac_inliers_count=n_inliers,
        inlier_ratio=inlier_ratio,
        detect_time_ms=t_detect,
        match_time_ms=t_match,
        transform_matrix=matrix,
        matches_after_ratio=ratio_matches,
        inlier_mask=inliers,
    )

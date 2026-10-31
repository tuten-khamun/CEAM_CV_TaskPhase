"""Tests for feature extraction, matching, and geometric verification."""

import cv2
import numpy as np
import pytest

from vmt.features import (
    ORBExtractor,
    SIFTExtractor,
    create_feature_extractor,
    match_and_verify,
    match_descriptors,
    verify_matches,
)


@pytest.fixture
def textured_image() -> np.ndarray:
    """Generate a reproducible, richly textured synthetic pattern."""
    rng = np.random.default_rng(42)
    img = np.zeros((300, 300), dtype=np.uint8)

    # Draw various overlapping geometric shapes with high contrast
    for i in range(15):
        cx, cy = rng.integers(50, 250, size=2)
        r = rng.integers(10, 40)
        color = int(rng.integers(50, 255))
        cv2.circle(img, (int(cx), int(cy)), int(r), color, -1)

    for i in range(15):
        x1, y1 = rng.integers(20, 260, size=2)
        w, h = rng.integers(20, 60, size=2)
        color = int(rng.integers(30, 240))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # Add subtle high-frequency Gaussian noise to ensure abundant corners
    noise = rng.normal(0, 10, img.shape).astype(np.float32)
    img_noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img_noisy


def test_factory_creation() -> None:
    orb = create_feature_extractor("orb")
    assert isinstance(orb, ORBExtractor)
    assert orb.norm_type == cv2.NORM_HAMMING

    sift = create_feature_extractor("sift")
    assert isinstance(sift, SIFTExtractor)
    assert sift.norm_type == cv2.NORM_L2

    with pytest.raises(ValueError):
        create_feature_extractor("invalid_name")


def test_blank_and_empty_images_do_not_crash() -> None:
    blank = np.zeros((100, 100), dtype=np.uint8)
    empty = np.array([], dtype=np.uint8)

    for name in ["orb", "sift"]:
        extractor = create_feature_extractor(name)
        kps, desc = extractor.detect_and_compute(blank)
        assert isinstance(kps, list)

        kps_empty, desc_empty = extractor.detect_and_compute(empty)
        assert kps_empty == []
        assert desc_empty is None

        stats = match_and_verify(extractor, blank, blank)
        assert stats.transform_matrix is None
        assert stats.ransac_inliers_count == 0


def test_warp_recovery_similarity_transform(textured_image: np.ndarray) -> None:
    """Warp textured image with known similarity transform and verify RANSAC recovers it."""
    # Ground truth affine partial (rotation + uniform scale + translation)
    angle_deg = 5.0
    scale = 1.0
    tx, ty = 12.0, -8.0

    center = (textured_image.shape[1] / 2.0, textured_image.shape[0] / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, scale)
    rot_mat[0, 2] += tx
    rot_mat[1, 2] += ty

    warped = cv2.warpAffine(
        textured_image,
        rot_mat,
        (textured_image.shape[1], textured_image.shape[0]),
        flags=cv2.INTER_LINEAR,
    )

    for name in ["orb", "sift"]:
        extractor = create_feature_extractor(name)
        stats = match_and_verify(
            extractor,
            textured_image,
            warped,
            ratio_threshold=0.75,
            ransac_thresh=3.0,
            method="affine",
        )

        assert stats.transform_matrix is not None, f"{name} failed to find transform matrix"
        assert stats.ransac_inliers_count >= 15, f"{name} had too few inliers: {stats.ransac_inliers_count}"
        assert stats.inlier_ratio >= 0.5, f"{name} inlier ratio too low: {stats.inlier_ratio:.2f}"

        # Test recovery accuracy of transformation matrix
        recovered = stats.transform_matrix
        diff = np.abs(recovered - rot_mat)
        assert np.max(diff[:2, :2]) < 0.08, f"{name} rotation/scale matrix error too high: {np.max(diff[:2, :2])}"
        assert np.max(diff[:2, 2]) < 3.5, f"{name} translation error too high: {np.max(diff[:2, 2])}"


def test_ratio_test_monotonicity(textured_image: np.ndarray) -> None:
    """Verifies that stricter (lower) ratio thresholds accept fewer or equal matches."""
    extractor = create_feature_extractor("orb")
    kps1, desc1 = extractor.detect_and_compute(textured_image)

    # Shift image by 5 px
    M = np.float32([[1, 0, 5], [0, 1, 0]])
    shifted = cv2.warpAffine(textured_image, M, (textured_image.shape[1], textured_image.shape[0]))
    kps2, desc2 = extractor.detect_and_compute(shifted)

    counts = []
    for r in [0.6, 0.7, 0.75, 0.8, 0.9]:
        matches, _ = match_descriptors(desc1, desc2, norm_type=extractor.norm_type, ratio_threshold=r)
        counts.append(len(matches))

    # Assert non-decreasing match count as ratio threshold relaxes
    for i in range(len(counts) - 1):
        assert counts[i] <= counts[i + 1]

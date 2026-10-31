# Codebase Walkthrough & Examiner Defense Guide

This document maps every module and public function in the Vehicle Motion Tracker to its mathematical concept, and provides 15 rigorous oral defense questions with detailed model answers.

---

## 1. Architectural Map: Code to Theory

| Module / Class | Public Function / Method | Classical CV Concept Implemented |
| :--- | :--- | :--- |
| `vmt.detector.MOG2Detector` | `detect(frame)` | Adaptive Gaussian Mixture Model (GMM) background subtraction, binary shadow rejection ($I > 250$), morphological opening/closing filtering, contour bounding box filtering by scaled area and aspect ratio. |
| `vmt.detector` | `merge_bounding_boxes(boxes)` | Non-Maximum Suppression / box clustering using Intersection-over-Union (IoU $> 0.3$) and spatial containment ($> 0.8$). |
| `vmt.tracker.VehicleTracker` | `_seed_features_in_box(gray, box)` | Shi-Tomasi corner detection ($\min(\lambda_1, \lambda_2) > \kappa$) restricted to a 15% inner-margin box to eliminate road asphalt and shadow edges; ORB template patch descriptor caching. |
| `vmt.tracker.VehicleTracker` | `_track_lk_optical_flow(prev, curr, track)` | Pyramidal Lucas-Kanade optical flow ($21 \times 21$ window, 4 pyramid levels), bidirectional forward-backward consistency validation ($|\mathbf{p}_0 - \mathbf{p}_0'| \le 1.0\text{ px}$), RANSAC Affine Partial 2D motion estimation, median displacement fallback. |
| `vmt.tracker.VehicleTracker` | `_reseed_track_points(gray, track)` | Feature renewal mechanism to eliminate cumulative optical flow drift by re-populating corners inside the current bounding box. |
| `vmt.tracker.VehicleTracker` | `_attempt_reacquisition(curr_gray, track)` | Appearance-based track recovery: local search window expansion ($1.5\times$), ORB descriptor KNN matching ($k=2$), Lowe's ratio test ($0.75$), and RANSAC affine consensus ($\ge 12$ inliers). |
| `vmt.tracker.VehicleTracker` | `step(frame, detections)` | Multi-stage tracking coordinator: optical flow propagation $\to$ Hungarian bipartite association $\to$ 70/30 drift-correcting box blend $\to$ lifecycle state transitions (Tentative, Confirmed, Lost, Deleted). |
| `vmt.hungarian` | `linear_sum_assignment(cost)` | Optimal minimum-cost bipartite assignment via Kuhn-Munkres dual potential reduction ($O(N^2 M)$). |
| `vmt.features` | `create_feature_extractor(name)` | Factory pattern instantiating ORB (binary, Hamming metric) or SIFT (floating-point gradient histogram, $L_2$ metric). |
| `vmt.features` | `match_descriptors(...)` | Brute-force nearest-neighbor matching ($k=2$) filtered by Lowe's second-neighbor ratio test ($d_1 / d_2 < 0.75$). |
| `vmt.features` | `verify_matches(...)` | RANSAC robust geometric estimation for Affine Partial 2D or Homography, separating consensus inliers from outliers. |
| `vmt.visualize.Visualizer` | `render(...)` | Trajectory rendering with 30-frame temporal fading, $3\times$ magnified optical flow vector arrows, high-contrast outlined HUD typography, and debug mask side-by-side composition. |
| `vmt.io_utils` | `write_tracks_csv(records, path)` | Deterministic CSV serialization with strict NaN / Inf numerical validation. |

---

## 2. 15 Examiner Defense Questions

Before reviewing the solutions below, attempt to answer each question as if defending your project in an oral exam.

1. **Why do we shrink the bounding box by 15% on each side before seeding Shi-Tomasi corners?**
2. **Why does OpenCV's MOG2 assign shadow pixels a value of 127, and what happens to the tracker if you omit the threshold at 250?**
3. **Explain the aperture problem in optical flow. How does the Shi-Tomasi corner detector mathematically prevent it?**
4. **What are the three fundamental assumptions behind the optical flow constraint equation $I_x u + I_y v + I_t = 0$? Under what real-world traffic conditions do they fail?**
5. **Why is a Gaussian pyramid essential for tracking vehicles moving at 60 km/h (e.g. 8–15 pixels/frame)?**
6. **How does bidirectional forward-backward validation detect failed optical flow points without knowing ground truth?**
7. **Why fit an Affine Partial 2D model (4 DOF) to surviving flow points instead of a full 8-DOF Homography or 6-DOF Affine?**
8. **What is the inlier ratio in RANSAC, and why is it a superior confidence metric compared to the raw number of tracked points?**
9. **Explain Lowe's ratio test. Why does a threshold of 0.75 reject false matches on repetitive textures like road surfaces?**
10. **Why must ORB descriptors be compared using the Hamming distance, while SIFT descriptors require the Euclidean $L_2$ distance?**
11. **What is the Hungarian algorithm solving in our pipeline, and why is it better than a simple greedy nearest-neighbor match?**
12. **Why do we blend matched bounding boxes as 70% optical flow / 30% MOG2 detection rather than replacing the box with the detector output entirely?**
13. **What happens to MOG2 background subtraction when a vehicle stops at a traffic signal for 30 seconds? How does our tracker respond?**
14. **Why does the project strictly report displacement and velocity in `pixels/frame` rather than claiming `km/h`?**
15. **How does the ORB re-acquisition mechanism differ from optical flow, and why is it capable of recovering tracks after complete occlusions?**

---

## 3. Answers — Attempt First!

### Q1: Seeding Margin Box Shrinkage
**Answer:** Bounding boxes produced by background subtraction contain the vehicle plus surrounding background context (road asphalt, lane markings, and cast shadows). If corners were seeded near the perimeter, feature points would latch onto stationary road markings or moving shadow edges. By shrinking the box by 15% on each side ($x_{\text{inner}} = x + 0.15w$, $w_{\text{inner}} = 0.70w$), corners are placed exclusively on the vehicle's rigid bodywork (hood creases, windshield trim, roof lines, grilles), ensuring optical flow measures the vehicle's true motion.

### Q2: MOG2 Shadow Suppression
**Answer:** In MOG2, a pixel is classified as a shadow if its chromaticity remains similar to the background model while its brightness decreases. OpenCV assigns foreground pixels $255$ and shadow pixels $127$. If thresholding at $250$ is omitted, shadow regions remain in the binary mask. As a result, vehicle bounding boxes expand to engulf dark road surfaces and merge with shadows of adjacent vehicles, corrupting track centers and causing false bounding box mergers.

### Q3: Aperture Problem & Shi-Tomasi Condition
**Answer:** The optical flow constraint $I_x u + I_y v + I_t = 0$ provides only 1 equation for 2 unknowns $(u, v)$. In regions with 1D edges (where gradients exist in only one direction, e.g. $\lambda_1 \gg 0, \lambda_2 \approx 0$), motion parallel to the edge produces zero intensity change, making parallel velocity unobservable. The Shi-Tomasi detector evaluates the eigenvalues of the structure tensor $M = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$. By enforcing $R = \min(\lambda_1, \lambda_2) > \kappa$, it guarantees that the patch has strong orthogonal gradients in two directions, ensuring $M$ is well-conditioned and invertable without the aperture problem.

### Q4: Optical Flow Assumptions and Failure Modes
**Answer:**
1. **Brightness Constancy:** The intensity of a physical scene point does not change over time ($I(x, y, t) = I(x+u, y+v, t+1)$). Fails during sudden sunlight shifts, vehicle headlight flashes, or specular reflections on curved windshields.
2. **Small Motion:** Displacements between frames are small enough for first-order Taylor series expansion ($\Delta x \le 1\text{ px}$). Fails on fast highway vehicles unless multi-scale pyramids are used.
3. **Spatial Coherence:** Neighboring pixels share identical motion. Fails at vehicle silhouette boundaries where foreground vehicle pixels move at $6\text{ px/f}$ while adjacent background road pixels move at $0\text{ px/f}$.

### Q5: Gaussian Pyramids in Lucas-Kanade
**Answer:** At 30 FPS, highway vehicles easily move 8 to 20 pixels per frame. Standard Lucas-Kanade fails because the spatial Taylor series linearisation breaks down when $\Delta x$ exceeds the gradient support ($> 1\text{ px}$). By building a 4-level Gaussian pyramid ($L=0, 1, 2, 3$), each octave subsamples resolution by $2\times$. A 16-pixel displacement at $L=0$ becomes:
$$\Delta x^{L=3} = \frac{16}{2^3} = 2\text{ pixels}$$
Solving optical flow iteratively from coarse ($L=3$) to fine ($L=0$) allows large displacements to be tracked while preserving sub-pixel precision.

### Q6: Forward-Backward Validation
**Answer:** Optical flow solvers always return a displacement vector $\mathbf{p}_1$ even when tracking fails (e.g. if the feature was occluded or moved out of frame). In forward-backward validation, $\mathbf{p}_1$ is used as the initial coordinate to track backward from frame $t+1 \to t$, yielding $\mathbf{p}_0'$. If the motion is physically valid and unoccluded, the round-trip error $\|\mathbf{p}_0 - \mathbf{p}_0'\|_2 \le 1.0\text{ px}$. If the point was occluded or drifted, the backward gradient points elsewhere, causing a large closure discrepancy and triggering rejection.

### Q7: Affine Partial 2D vs Homography
**Answer:** A planar homography requires 8 degrees of freedom (4 point correspondences) and models general projective warp. Because our camera is static and distant, vehicles undergo rigid planar translation, minor rotation (yaw during lane changes), and uniform scale change (perspective as vehicles approach or recede). Affine Partial 2D models exactly 4 DOF:
$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s \cos\theta & -s \sin\theta \\ s \sin\theta & s \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$
It requires only $s=2$ point pairs (minimal sample), making RANSAC vastly faster and mathematically immune to non-physical shearing or projective distortions that plague homography fits on near-collinear points.

### Q8: Inlier Ratio as a Confidence Metric
**Answer:** Raw point count can be misleading: a large vehicle might track 40 points, but if 25 of them are drifting on road clutter, the motion estimate is corrupted. The inlier ratio:
$$\text{Inlier Ratio} = \frac{N_{\text{RANSAC Inliers}}}{N_{\text{Surviving Points}}}$$
measures the geometric consensus of the motion model. A ratio $> 0.8$ proves that points are moving congruently on a single rigid body, whereas a ratio $< 0.4$ indicates that the track has intersected an occlusion, entered a shadow, or lost rigidity.

### Q9: Lowe's Ratio Test Mechanics
**Answer:** In feature space, nearest-neighbor matching finds the closest descriptor $\mathbf{d}_1$. However, repetitive textures (such as road asphalt, crash barriers, or brick walls) produce many descriptors with near-identical distances. Lowe's ratio test evaluates the ratio of the distance of the best match to the second-best match:
$$\frac{\|\mathbf{d}_q - \mathbf{d}_1\|}{\|\mathbf{d}_q - \mathbf{d}_2\|} < 0.75$$
For a distinct physical feature, $\mathbf{d}_1$ is an isolated match and $\mathbf{d}_2$ is far away ($\text{ratio} \ll 0.75$). For repetitive patterns, $\mathbf{d}_1$ and $\mathbf{d}_2$ have nearly identical distances ($\text{ratio} \approx 1.0$), correctly rejecting ambiguous matches.

### Q10: Hamming vs Euclidean Metric
**Answer:** SIFT computes 128-dimensional floating-point histograms of spatial gradients; the mathematical distance between two histograms is the $L_2$ Euclidean norm $\|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum (a_i - b_i)^2}$. ORB computes binary bitstrings (256 bits) where each bit represents a pairwise intensity test ($\tau = 1$ if $I(x) < I(y)$). The distance between two binary strings is the Hamming distance—the number of differing bits—computed via bitwise XOR and hardware population count (`POPCNT`), which runs in a single CPU cycle.

### Q11: Hungarian Algorithm vs Greedy Association
**Answer:** A greedy algorithm pairs the highest-IoU detection first. In dense traffic, if Detection A is greedily assigned to Track 1, Track 2 may be left with no valid match or forced into a poor assignment, causing ID switches. The Hungarian (Kuhn-Munkres) algorithm solves the global bipartite matching problem:
$$\min_{\pi} \sum_{i} C_{i, \pi(i)}$$
where $C_{ij} = 1 - \operatorname{IoU}(T_i, D_j)$, guaranteeing the globally optimal assignment that minimizes total association cost across all tracks simultaneously in $O(N^2 M)$ time.

### Q12: 70/30 Bounding Box Blending
**Answer:** Optical flow tracks physical vehicle features with high sub-pixel precision ($< 0.2\text{ px}$), but over hundreds of frames, minor errors accumulate into drift. Conversely, MOG2 background subtraction contours fluctuate from frame to frame due to shadow variations and vehicle wakes (elongating bounding boxes backward by 2–4 pixels). By blending:
$$\mathbf{B}_{\text{new}} = 0.70 \mathbf{B}_{\text{track}} + 0.30 \mathbf{B}_{\text{detector}}$$
we retain the temporal smoothness and sub-pixel accuracy of optical flow while using MOG2 to pull the box back to the global vehicle footprint, preventing unbounded drift.

### Q13: Stopped Vehicles & Background Absorption
**Answer:** MOG2 models each pixel with a running history of Gaussian mixtures (e.g. `history=500` frames). When a vehicle stops, its pixel intensities remain constant. After ~30–50 frames, the Gaussian mixture incorporates the vehicle pixels into the background model, and the foreground mask vanishes.
In our tracker:
1. MOG2 stops emitting detections for the vehicle.
2. The track misses association and increments its `misses` counter.
3. Optical flow continues tracking stationary Shi-Tomasi corners ($u \approx 0, v \approx 0$).
4. After 15 missed detection cycles (75 frames), the track transitions to LOST and is eventually deleted. Mitigating this requires a stationary vehicle memory module or dual-state tracking.

### Q14: Pixels/Frame vs km/h
**Answer:** Perspective projection maps a 3D physical world coordinate $(X, Y, Z)$ to a 2D camera pixel $(u, v)$ via the camera projection matrix $P = K [R \mid \mathbf{t}]$. Without extrinsic camera calibration (tilt angle, camera height above road) and intrinsic focal length, metric scale is unobservable: a displacement of 5 pixels near the horizon may represent 10 metres, whereas 5 pixels in the foreground represents 0.5 metres. Claiming km/h without camera calibration or ground control points is mathematically invalid.

### Q15: ORB Re-acquisition vs Optical Flow
**Answer:** Lucas-Kanade optical flow is purely differential and temporal: it requires the feature to exist in frame $t$ to compute its displacement in frame $t+1$. When a vehicle is occluded (e.g. passing behind a bridge or another vehicle), all optical flow points fail forward-backward validation and the track is marked LOST.
ORB re-acquisition stores the vehicle's appearance as invariant binary descriptors. Even after 15 frames of total occlusion, when the vehicle re-emerges, ORB detects scale- and rotation-invariant keypoints in an expanded search window, matches them to the stored template via Lowe's ratio test, and verifies the match using RANSAC. Surviving $\ge 12$ inliers resurrects the track with its original identity, preventing track fragmentation.

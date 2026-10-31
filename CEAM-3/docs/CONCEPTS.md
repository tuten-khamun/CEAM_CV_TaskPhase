# Classical Computer Vision Concepts: Theory, Derivations, & Worked Examples

This document serves as your theoretical defense guide for Week 3: *Classical Tracking & Features*.
Every concept below is structured with its **governing equations**, **plain-words physical intuition**, **step-by-step worked numeric example**, and **OpenCV implementation signature**.

---

## Table of Contents
1. [Scale-Invariant Feature Transform (SIFT)](#1-scale-invariant-feature-transform-sift)
   - [Difference of Gaussians (DoG) Scale Space](#difference-of-gaussians-dog-scale-space)
   - [Sub-pixel Keypoint Localisation](#sub-pixel-keypoint-localisation)
   - [Canonical Orientation Assignment](#canonical-orientation-assignment)
   - [128-D Gradient Histogram Descriptor](#128-d-gradient-histogram-descriptor)
2. [Oriented FAST and Rotated BRIEF (ORB)](#2-oriented-fast-and-rotated-brief-orb)
   - [FAST Corner Detection](#fast-corner-detection)
   - [Intensity Centroid Orientation](#intensity-centroid-orientation)
   - [Steered / Rotated BRIEF](#steered--rotated-brief)
   - [Hamming Distance Metric](#hamming-distance-metric)
3. [Lowe's Ratio Test](#3-lowes-ratio-test)
4. [RANdom SAmple Consensus (RANSAC)](#4-random-sample-consensus-ransac)
5. [Lucas-Kanade Sparse Pyramidal Optical Flow](#5-lucas-kanade-sparse-pyramidal-optical-flow)
   - [Optical Flow Constraint Equation](#optical-flow-constraint-equation)
   - [The $2 \times 2$ Least-Squares Structure Tensor](#the-2-times-2-least-squares-structure-tensor)
   - [Eigenvalue Trackability & The Aperture Problem](#eigenvalue-trackability--the-aperture-problem)
   - [Gaussian Pyramids for Large Displacements](#gaussian-pyramids-for-large-displacements)
   - [Bidirectional Forward-Backward Consistency](#bidirectional-forward-backward-consistency)

---

## 1. Scale-Invariant Feature Transform (SIFT)

### Difference of Gaussians (DoG) Scale Space
#### The Equation
To identify features invariant to scale changes, the image $I(x, y)$ is convolved with Gaussian kernels $G(x, y, \sigma) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$ across octaves and scale intervals $k$:
$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$
The Difference of Gaussians (DoG) closely approximates the scale-normalized Laplacian of Gaussian (LoG), $\sigma^2 \nabla^2 G$, but requires only simple subtractions:
$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)$$

#### Plain-Words Intuition
Blurring an image wipes out fine details. If you subtract a slightly more blurred version from a slightly less blurred version, you isolate the structures whose characteristic physical size matches that specific blur radius $\sigma$. A feature is detected at the scale where its contrast response reaches a local maximum.

#### Worked Numeric Example
Consider a $1\text{D}$ pixel profile: $[10, 50, 10]$.
- Blur at $\sigma_1 = 1$: smoothed profile $\approx [20, 40, 20]$.
- Blur at $\sigma_2 = 1.6$: smoothed profile $\approx [25, 30, 25]$.
- Difference of Gaussians value at center:
  $$D = L(x, \sigma_1) - L(x, \sigma_2) = 40 - 30 = 10$$
Comparing $D=10$ with its 26 discrete spatio-scale neighbors ($3 \times 3$ across 3 adjacent scales) determines if it is an extremum.

#### OpenCV Implementation
```python
sift = cv2.SIFT_create(nfeatures=1500)
keypoints, descriptors = sift.detectAndCompute(gray_image, mask=None)
```

---

### Sub-pixel Keypoint Localisation
#### The Equation
Taylor expansion of the scale-space function $D(\mathbf{x})$ around sample point $\mathbf{x}_0 = (x_0, y_0, \sigma_0)^T$:
$$D(\mathbf{x}) = D + \frac{\partial D^T}{\partial \mathbf{x}}\mathbf{x} + \frac{1}{2} \mathbf{x}^T \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$
Setting $\nabla_{\mathbf{x}} D = \mathbf{0}$ yields the sub-pixel offset:
$$\hat{\mathbf{x}} = - \left( \frac{\partial^2 D}{\partial \mathbf{x}^2} \right)^{-1} \frac{\partial D}{\partial \mathbf{x}}$$
Points with $|D(\hat{\mathbf{x}})| < 0.03$ or principal curvature ratio $r > 10$ along edges (via the Hessian trace/determinant) are rejected.

#### Plain-Words Intuition
Discrete pixels only approximate the true continuous optical center of a physical feature. Fitting a 3D quadratic bowl locates the true sub-pixel peak. Checking the curvature ratio rejects edge responses (which suffer from the aperture problem).

---

### Canonical Orientation Assignment
#### The Equation
Gradient magnitude $m(x, y)$ and direction $\theta(x, y)$ in the scale image $L$:
$$m(x, y) = \sqrt{(L(x+1, y) - L(x-1, y))^2 + (L(x, y+1) - L(x, y-1))^2}$$
$$\theta(x, y) = \operatorname{atan2}(L(x, y+1) - L(x, y-1), L(x+1, y) - L(x-1, y))$$
Gradients within a Gaussian-weighted circular window are accumulated into a 36-bin orientation histogram (each bin spanning $10^\circ$).

#### Plain-Words Intuition
By determining the dominant local lighting gradient direction and rotating all future measurements relative to this angle, the descriptor becomes invariant to 2D image rotation.

---

### 128-D Gradient Histogram Descriptor
#### The Equation & Architecture
Around the oriented keypoint, a $16 \times 16$ pixel neighborhood is divided into a $4 \times 4$ grid of spatial sub-regions ($4 \times 4$ pixels each). Within each sub-region, gradients are binned into 8 orientation directions ($45^\circ$ bins).
$$\text{Descriptor length} = 4 \times 4 \times 8 = 128 \text{ dimensions}$$
The resulting vector $\mathbf{f} \in \mathbb{R}^{128}$ is normalized to unit length, thresholded at $0.2$ to reduce non-linear illumination sensitivity, and re-normalized:
$$\mathbf{f}_{\text{final}} = \frac{\min(\mathbf{f} / \|\mathbf{f}\|_2, 0.2)}{\|\min(\mathbf{f} / \|\mathbf{f}\|_2, 0.2)\|_2}$$

#### Worked Numeric Example
Suppose a single $4 \times 4$ sub-block contains 16 gradients all pointing right ($\theta = 0^\circ$) with magnitude $m=2.0$.
The 8 orientation bins are $[0^\circ, 45^\circ, 90^\circ, \dots, 315^\circ]$.
- Bin 0 ($0^\circ$) accumulates: $\sum m = 16 \times 2.0 = 32.0$.
- Bins 1..7 remain $0.0$.
Sub-block vector: $[32, 0, 0, 0, 0, 0, 0, 0]$. Concatenating all 16 sub-blocks yields the 128-element vector.

---

## 2. Oriented FAST and Rotated BRIEF (ORB)

### FAST Corner Detection
#### The Equation
A candidate pixel $p$ at intensity $I_p$ is tested against 16 pixels on a Bresenham circle of radius 3. $p$ is a corner if there exists a contiguous arc of $N=12$ (or 9) pixels $x$ such that:
$$I_x > I_p + t \quad \text{or} \quad I_x < I_p - t$$
where $t$ is the intensity contrast threshold (e.g. $t=10$).

#### Plain-Words Intuition
A true physical corner causes a sharp boundary between dark and bright regions. By testing pixels 1, 9, 5, 13 first, non-corners are rejected in only 2-4 operations, making FAST roughly 20x faster than SIFT's DoG scale space.

---

### Intensity Centroid Orientation
#### The Equation
The orientation $\theta$ is computed from the image moments $m_{pq}$ over a circular patch of radius $r$:
$$m_{pq} = \sum_{x=-r}^{r} \sum_{y=-r}^{r} x^p y^q I(x, y)$$
The patch centroid is $\mathbf{C} = \left(\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}\right)$, giving the canonical orientation:
$$\theta = \operatorname{atan2}(m_{01}, m_{10})$$

#### Worked Numeric Example
Consider a $3 \times 3$ patch centered at $(0,0)$:
$$\begin{bmatrix} 10 & 10 & 200 \\ 10 & 50 & 200 \\ 10 & 10 & 200 \end{bmatrix}$$
- $m_{00} = 10+10+200+10+50+200+10+10+200 = 700$
- $m_{10} = (-1)(10+10+10) + (0)(10+50+10) + (+1)(200+200+200) = -30 + 600 = 570$
- $m_{01} = (-1)(10+10+200) + (0)(10+50+200) + (+1)(10+10+200) = -220 + 220 = 0$
- $\theta = \operatorname{atan2}(0, 570) = 0^\circ$ (pointing directly towards the bright right edge).

---

### Steered / Rotated BRIEF
#### The Equation
BRIEF builds a binary bitstring by testing 256 pixel pairs $(x_i, y_i)$:
$$\tau(p; x, y) = \begin{cases} 1 & \text{if } I(p+x) < I(p+y) \\ 0 & \text{otherwise} \end{cases}$$
To make it rotation-invariant, the test coordinate matrix $S = \begin{pmatrix} x_1 & \dots & x_n \\ y_1 & \dots & y_n \end{pmatrix}$ is steered by the 2D rotation matrix $R_\theta$:
$$S_\theta = R_\theta S$$
where $R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.

---

### Hamming Distance Metric
#### The Equation
The distance between two 256-bit binary descriptors $\mathbf{a}, \mathbf{b} \in \{0, 1\}^{256}$:
$$d_H(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{256} a_i \oplus b_i = \operatorname{popcount}(\mathbf{a} \oplus \mathbf{b})$$

#### Plain-Words Intuition
Comparing binary descriptors requires no floating-point arithmetic. Modern CPUs execute bitwise XOR and hardware population count (`POPCNT`) in a single clock cycle.

#### OpenCV Implementation
```python
orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=10)
keypoints, descriptors = orb.detectAndCompute(gray_image, mask=None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
```

---

## 3. Lowe's Ratio Test

### The Equation
Given query descriptor $\mathbf{d}_q$ and its two nearest neighbors $\mathbf{d}_1$ (best) and $\mathbf{d}_2$ (second-best) in the target image:
$$\text{Accept match if } \frac{\|\mathbf{d}_q - \mathbf{d}_1\|}{\|\mathbf{d}_q - \mathbf{d}_2\|} < \text{threshold } \rho$$
Typically, $\rho \in [0.70, 0.80]$ (default $\rho = 0.75$).

### Plain-Words Intuition
If a patch is unique (e.g. a specific license plate digit), its best match $\mathbf{d}_1$ will be very close in feature space, while all other background matches $\mathbf{d}_2$ will be far away ($\text{ratio} \ll 1$). If the patch is repetitive (e.g. uniform road asphalt or window grids), $\mathbf{d}_1$ and $\mathbf{d}_2$ will have nearly identical distances ($\text{ratio} \approx 1.0$), signaling ambiguity.

### Worked Numeric Example
- **Case A (Unique feature):** $d_1 = 30$, $d_2 = 120$. $\text{Ratio} = 30 / 120 = 0.25 < 0.75 \implies \mathbf{Accepted}$.
- **Case B (Repetitive asphalt):** $d_1 = 78$, $d_2 = 82$. $\text{Ratio} = 78 / 82 = 0.95 \ge 0.75 \implies \mathbf{Rejected}$.

---

## 4. RANdom SAmple Consensus (RANSAC)

### Sample Sizing Formula
To find an outlier-free model with probability $p$ (e.g. $p=0.99$), where the inlier fraction is $w \in [0, 1]$ and each minimal sample requires $s$ points ($s=3$ for affine, $s=4$ for homography):
$$(1 - w^s)^N = 1 - p \implies N = \frac{\ln(1 - p)}{\ln(1 - w^s)}$$

### Plain-Words Intuition
If half of all matches are false outliers ($w=0.5$), the odds that all $s=3$ randomly drawn points are true inliers is $(0.5)^3 = 0.125$. The probability of picking at least one good subset in $N$ independent draws increases exponentially toward 1.

### Worked Numeric Example
Target confidence $p = 0.99$, inlier fraction $w = 0.50$, minimal sample $s = 3$ (Affine Partial 2D):
$$w^s = (0.50)^3 = 0.125$$
$$1 - w^s = 0.875$$
$$N = \frac{\ln(1 - 0.99)}{\ln(0.875)} = \frac{\ln(0.01)}{\ln(0.875)} = \frac{-4.605}{-0.1335} \approx 34.48 \implies N = 35 \text{ iterations}$$
Thus, only 35 random samples guarantee a 99% probability of recovering the true vehicle motion!

#### OpenCV Implementation
```python
matrix, inlier_mask = cv2.estimateAffinePartial2D(
    src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99
)
```

---

## 5. Lucas-Kanade Sparse Pyramidal Optical Flow

### Optical Flow Constraint Equation
#### Derivation from Brightness Constancy
**Assumption:** The intensity of a physical scene point $(x, y)$ remains constant between frame $t$ and frame $t + \Delta t$:
$$I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)$$
Applying first-order Taylor series expansion to the right-hand side:
$$I(x + \Delta x, y + \Delta y, t + \Delta t) \approx I(x, y, t) + \frac{\partial I}{\partial x}\Delta x + \frac{\partial I}{\partial y}\Delta y + \frac{\partial I}{\partial t}\Delta t$$
Subtracting $I(x, y, t)$ and dividing by $\Delta t \to 0$:
$$\frac{\partial I}{\partial x} \frac{\Delta x}{\Delta t} + \frac{\partial I}{\partial y} \frac{\Delta y}{\Delta t} + \frac{\partial I}{\partial t} = 0$$
$$\boxed{I_x u + I_y v + I_t = 0}$$
where $u = \frac{dx}{dt}$ and $v = \frac{dy}{dt}$ are the unknown horizontal and vertical pixel velocities.

---

### The $2 \times 2$ Least-Squares Structure Tensor
#### The Equation
A single pixel provides one scalar equation for two unknowns $(u, v)$—an underdetermined problem. Lucas & Kanade assumed that motion is locally uniform across a small spatial window $\Omega$ of size $m \times m$ (e.g. $21 \times 21$):
$$\begin{bmatrix} I_{x1} & I_{y1} \\ I_{x2} & I_{y2} \\ \vdots & \vdots \\ I_{xn} & I_{yn} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = - \begin{bmatrix} I_{t1} \\ I_{t2} \\ \vdots \\ I_{tn} \end{bmatrix} \iff A \mathbf{u} = \mathbf{b}$$
Multiplying by $A^T$ yields the $2 \times 2$ normal equations:
$$(A^T A) \mathbf{u} = A^T \mathbf{b}$$
$$\boxed{\begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = - \begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}}$$
The matrix $M = A^T A$ is the **Harris/Shi-Tomasi Structure Tensor**.

---

### Eigenvalue Trackability & The Aperture Problem
To invert $M$, its eigenvalues $\lambda_1, \lambda_2$ must both be strictly positive and well-conditioned:
1. **Flat regions:** $\lambda_1 \approx 0, \lambda_2 \approx 0$. No intensity gradients; motion cannot be measured.
2. **Edge regions (The Aperture Problem):** $\lambda_1 \gg 0, \lambda_2 \approx 0$. Gradients exist only perpendicular to the edge. Motion parallel to the edge is invisible (e.g. a barber-pole illusion or a smooth horizontal car hood).
3. **Corner regions (Good Features to Track):** $\lambda_1 \ge \lambda_2 > \kappa > 0$. High orthogonal gradients in both directions; $M$ is well-conditioned and invertable without noise amplification.

**Shi-Tomasi Criterion:**
$$R = \min(\lambda_1, \lambda_2) > \lambda_{\text{threshold}}$$

#### Worked Numeric Example
Suppose across a $3 \times 3$ window:
$\sum I_x^2 = 100$, $\sum I_y^2 = 100$, $\sum I_x I_y = 0$, $\sum I_x I_t = -200$, $\sum I_y I_t = -100$.
The system becomes:
$$\begin{bmatrix} 100 & 0 \\ 0 & 100 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = - \begin{bmatrix} -200 \\ -100 \end{bmatrix} = \begin{bmatrix} 200 \\ 100 \end{bmatrix}$$
$$\mathbf{u} = \begin{bmatrix} 100 & 0 \\ 0 & 100 \end{bmatrix}^{-1} \begin{bmatrix} 200 \\ 100 \end{bmatrix} = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} \implies u = 2.0\text{ px/f}, \, v = 1.0\text{ px/f}$$

---

### Gaussian Pyramids for Large Displacements
The Taylor expansion assumes small displacements ($\Delta x, \Delta y \le 1\text{ px}$). Highway vehicles travel at 5 to 15 px/frame, violating this assumption.
A Gaussian pyramid downsamples the image by factors of 2:
$$I^L(x, y) = \frac{1}{4} I^{L-1}(2x, 2y)$$
At level $L=3$, a 16-pixel displacement is reduced to:
$$\Delta x^{L=3} = \frac{16}{2^3} = 2\text{ px}$$
Optical flow is solved at the coarsest level $L=3$, and its displacement estimate is multiplied by 2 to seed the initial estimate at level $L=2$, cascading down to the full-resolution level $L=0$.

---

### Bidirectional Forward-Backward Consistency
Points are tracked forward from $I_{t} \to I_{t+1}$ yielding $\mathbf{p}_1$, then immediately tracked backward from $I_{t+1} \to I_t$ yielding $\mathbf{p}_0'$.
$$\text{Consistency Error: } \|\mathbf{p}_0 - \mathbf{p}_0'\|_2 \le 1.0\text{ px}$$
Any point that drifts due to occlusion, specular reflection, or feature loss fails round-trip closure and is discarded before motion fitting.

#### OpenCV Implementation
```python
p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
    prev_gray, curr_gray, p0, None,
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)
```

from pathlib import Path

import numpy as np
import tifffile

from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp

# ORB finds corner-like features in the speckle/texture pattern of the
# fluorescent substrate — plenty of candidates even without distinct objects.
N_KEYPOINTS = 1000

# Px tolerance for a match to count as consistent with a single affine
# transform between the two frames — separates genuine tracked features from
# descriptor look-alikes (the speckle pattern has lots of similar texture).
RANSAC_RESIDUAL_PX = 2.0


def load_gray(path: Path) -> np.ndarray:
    # Only the green channel carries data for CH4 frames (R and B are all
    # zero) — use it directly rather than averaging, which would just dilute
    # the signal by a factor of 3.
    return tifffile.imread(path)[:, :, 1]


def estimate_alignment(gray_a: np.ndarray, gray_b: np.ndarray) -> AffineTransform:
    """Detect ORB keypoints in each frame, cross-check-match their
    descriptors, then RANSAC-fit a single affine transform mapping
    frame-A coordinates to frame-B coordinates — discarding descriptor
    look-alikes (the speckle pattern has lots of similar texture) that
    don't agree with one coherent transform, i.e. aren't real
    correspondences."""
    orb = ORB(n_keypoints=N_KEYPOINTS)
    orb.detect_and_extract(gray_a)
    keypoints_a, descriptors_a = orb.keypoints, orb.descriptors
    orb.detect_and_extract(gray_b)
    keypoints_b, descriptors_b = orb.keypoints, orb.descriptors

    matches = match_descriptors(descriptors_a, descriptors_b, cross_check=True)

    # skimage keypoints are (row, col); transforms expect (x, y) = (col, row).
    src = keypoints_a[matches[:, 0]][:, ::-1]
    dst = keypoints_b[matches[:, 1]][:, ::-1]
    model, inliers = ransac(
        (src, dst), AffineTransform,
        min_samples=3, residual_threshold=RANSAC_RESIDUAL_PX, max_trials=2000,
    )
    print(f"  {len(keypoints_a)} / {len(keypoints_b)} keypoints, {len(matches)} matches, "
          f"{inliers.sum()} kept as coherent correspondences")
    return model


def common_crop(valid_mask: np.ndarray) -> tuple[slice, slice]:
    """Largest axis-aligned rectangle fully inside the valid (non-fill)
    region of the warped frame, found by repeatedly trimming whichever
    border currently holds the most invalid (fill) pixels. The valid region
    is convex — an affine-warped rectangle intersected with another
    rectangle — so the transform's small rotation/scale components leave
    only a thin wedge of fill pixels near the border, which this converges
    past in a handful of steps (a pure-translation crop would miss it and
    leave that wedge inside the "common" region)."""
    top, bottom = 0, valid_mask.shape[0]
    left, right = 0, valid_mask.shape[1]
    while True:
        region = valid_mask[top:bottom, left:right]
        if region.all():
            return slice(top, bottom), slice(left, right)
        counts = {
            "top": int((~region[0, :]).sum()),
            "bottom": int((~region[-1, :]).sum()),
            "left": int((~region[:, 0]).sum()),
            "right": int((~region[:, -1]).sum()),
        }
        edge = max(counts, key=counts.__getitem__)
        if edge == "top":
            top += 1
        elif edge == "bottom":
            bottom -= 1
        elif edge == "left":
            left += 1
        else:
            right -= 1


def main() -> None:
    data_dir = Path("data")
    out_dir = Path("results") / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_a = data_dir / "Image_T0001_XY01_Z001_CH4.tif"
    file_b = data_dir / "Image_T0047_XY01_Z001_CH4.tif"

    print(f"Loading {file_a.name} and {file_b.name} ...")
    gray_a, gray_b = load_gray(file_a), load_gray(file_b)
    h, w = gray_a.shape

    print("Estimating alignment via ORB keypoint matching + RANSAC affine fit ...")
    model = estimate_alignment(gray_a, gray_b)
    tx, ty = model.translation
    print(f"  estimated shift: dx={tx:.2f} px, dy={ty:.2f} px")

    print("Warping frame B onto frame A's coordinate grid ...")
    aligned_b = warp(gray_b, model, output_shape=(h, w), preserve_range=True, cval=0)
    # Where the warp had to sample outside frame B's original bounds, it
    # filled with cval=0 — track that with the same warp on an all-ones
    # image so the crop below excludes those fabricated-zero pixels.
    valid_mask = warp(np.ones_like(gray_b, dtype=np.float64), model, output_shape=(h, w), cval=0) > 0.999
    aligned_b = aligned_b.astype(gray_b.dtype)

    row_crop, col_crop = common_crop(valid_mask)
    cw, ch = col_crop.stop - col_crop.start, row_crop.stop - row_crop.start
    print(f"Common valid region: {cw}x{ch} (rows {row_crop}, cols {col_crop})")

    out_a, out_b = out_dir / file_a.name, out_dir / file_b.name
    tifffile.imwrite(out_a, gray_a[row_crop, col_crop])
    tifffile.imwrite(out_b, aligned_b[row_crop, col_crop])
    print(f"Done — wrote two aligned, pixel-registered {cw}x{ch} frames:\n  {out_a}\n  {out_b}")


if __name__ == "__main__":
    main()

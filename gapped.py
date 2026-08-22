from pathlib import Path
import argparse
import itertools
import re

import imageio.v3 as iio
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

from piv_fttc import render_combined_frame
from piv import compute_piv, make_gapped_settings
from piv import to_gray as to_gray_uint8
from stabilize import UPSAMPLE_FACTOR, apply_shift, common_crop
from stabilize import to_gray as to_gray_float
from traction import (
    DISPLACEMENT_SMOOTH_SIGMA,
    POISSON_RATIO,
    TRACTION_VMIN_PERCENTILE,
    YOUNGS_MODULUS,
    fttc_traction,
)

FRAME_INDEX_RE = re.compile(r"(\d+)")


def _block_mean(arr: np.ndarray, n: int) -> np.ndarray:
    H, W = arr.shape
    H_trim = (H // n) * n
    W_trim = (W // n) * n
    return (
        arr[:H_trim, :W_trim]
        .reshape(H_trim // n, n, W_trim // n, n)
        .mean(axis=(1, 3))
    )


def export_traction_csv(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    grid_mean: int,
    px_to_um: float | None = None,
) -> None:
    if grid_mean > 1:
        x, y, tx, ty = (_block_mean(a, grid_mean) for a in (x, y, tx, ty))
    # Traction stress (Pa) is unit-independent of pixel size (see
    # traction.py's fttc_traction) — only the x, y positions need
    # rescaling to micrometers.
    if px_to_um is not None:
        x, y = x * px_to_um, y * px_to_um
    header = "x_um,y_um,stress" if px_to_um is not None else "x,y,stress"
    rows = np.column_stack([x.ravel(), y.ravel(), np.sqrt(tx**2 + ty**2).ravel()])
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


def export_displacement_csv(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    grid_mean: int,
    px_to_um: float | None = None,
) -> None:
    if grid_mean > 1:
        x, y, u, v = (_block_mean(a, grid_mean) for a in (x, y, u, v))
    if px_to_um is not None:
        x, y, u, v = (a * px_to_um for a in (x, y, u, v))
    magnitude = np.sqrt(u**2 + v**2)
    direction = np.degrees(np.arctan2(v, u))
    header = ("x_um,y_um,magnitude_um,direction" if px_to_um is not None
               else "x,y,magnitude,direction")
    rows = np.column_stack([x.ravel(), y.ravel(), magnitude.ravel(), direction.ravel()])
    np.savetxt(path, rows, delimiter=",", header=header, comments="")

# Acquisition interval, used only to label each pair's elapsed time —
# matches the dense time series (10 minutes/frame).
FRAME_INTERVAL_MINUTES = 10


def frame_index(path: Path) -> int:
    m = FRAME_INDEX_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not find a _T<index>_ timestamp in {path.name}")
    return int(m.group(1))


def find_frames(gapped_dir: Path, channel: str) -> dict[int, Path]:
    files = sorted(gapped_dir.glob(f"*.tif"))
    if not files:
        raise FileNotFoundError(f"No *.tif files found in {gapped_dir}")
    return {frame_index(f): f for f in files}


def stabilize_pair(
    ref_micro: np.ndarray, ref_bf: np.ndarray,
    tgt_micro: np.ndarray, tgt_bf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Co-register the target frame pair to the reference: estimate drift
    between the two microscope frames (the data PIV is computed from) and
    crop both channels of both frames to the region with real data in both,
    exactly as stabilize.py does for the dense time series — but here as a
    single pairwise registration per (reference, target) pair, since each
    target may have drifted by a different amount."""
    h, w = ref_micro.shape[:2]
    shift, _, _ = phase_cross_correlation(
        to_gray_float(ref_micro), to_gray_float(tgt_micro), upsample_factor=UPSAMPLE_FACTOR
    )
    shifts = np.array([[0.0, 0.0], shift])
    row_crop, col_crop = common_crop(shifts, h, w)

    ref_micro_c = ref_micro[row_crop, col_crop]
    ref_bf_c = ref_bf[row_crop, col_crop]
    tgt_micro_c = apply_shift(tgt_micro, shift)[row_crop, col_crop]
    tgt_bf_c = apply_shift(tgt_bf, shift)[row_crop, col_crop]
    return ref_micro_c, ref_bf_c, tgt_micro_c, tgt_bf_c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PIV displacement and FTTC traction stress from "
                    "the first (reference) frame in --gapped-dir to each "
                    "later frame, and render a combined visualization "
                    "overlaying any mix of: the reference frame, PIV "
                    "displacement arrows, and the FTTC traction-stress "
                    "heatmap. All three are on by default. Unlike "
                    "piv_fttc.py's consecutive-frame video, the "
                    "displacement here is the net motion since the "
                    "reference frame, so the traction field is the absolute "
                    "stress (not a frame-to-frame change), assuming the "
                    "reference frame is unstressed."
    )
    parser.add_argument(
        "--gapped-dir", type=Path, default=Path("gapped"),
        help="Directory of sparsely time-spaced *.tif "
             "(microscope) / Image_T*_CH1.tif (brightfield) snapshots "
             "(default: gapped).",
    )
    parser.add_argument(
        "--no-background", action="store_true",
        help="Omit the snapshot (target) frame from the rendering.",
    )
    parser.add_argument(
        "--no-piv", action="store_true",
        help="Omit PIV displacement arrows from the rendering.",
    )
    parser.add_argument(
        "--no-traction", action="store_true",
        help="Omit the FTTC traction-stress heatmap from the rendering.",
    )
    parser.add_argument(
        "--transparent", action="store_true",
        help="For renders without the background frame (--no-background, or "
             "the backgroundless combinations under --all), use a "
             "transparent canvas instead of the default opaque black/white "
             "one, and save PNGs with an alpha channel. Has no effect on "
             "renders that include the background frame.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Render all 7 non-empty combinations of background/PIV/traction "
             "layers (overrides --no-background/--no-piv/--no-traction), each "
             "into its own <output-dir>/piv_fttc_<layers>/ directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/gapped"),
        metavar="DIR",
        help="Root directory for all outputs (default: results/gapped).",
    )
    parser.add_argument(
        "--grid-mean", type=int, default=1, metavar="N",
        help="Before writing the traction CSV, average every N×N block of PIV "
             "grid cells into a single point (default: 1, no averaging).",
    )
    parser.add_argument(
        "--px-to-um", type=float, default=None, metavar="SCALE",
        help="Pixel-to-micrometer conversion factor (micrometers per pixel). "
             "If given, displacement and spatial axes in the rendered "
             "output and exported CSVs are shown in micrometers instead of "
             "pixels (traction stress in Pa is unaffected — it's already "
             "unit-independent of pixel size). Default: pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        # All 7 non-empty (background, PIV, traction) combinations.
        combos = [c for c in itertools.product([False, True], repeat=3) if any(c)]
    else:
        show_background = not args.no_background
        show_piv        = not args.no_piv
        show_traction   = not args.no_traction
        if not (show_background or show_piv or show_traction):
            raise SystemExit("Nothing to render — at least one of background/PIV/"
                             "traction must stay enabled")
        combos = [(show_background, show_piv, show_traction)]

    # PIV/traction are computed once per target frame and reused across
    # combos — only skip them if no combo needs them at all.
    need_piv      = any(piv or traction for _, piv, traction in combos)
    need_traction = any(traction for _, _, traction in combos)

    gapped_dir = args.gapped_dir
    results_dir = args.output_dir
    stabilized_dir = results_dir / "stabilized"
    stabilized_dir.mkdir(parents=True, exist_ok=True)

    # Output dir per combo reflects exactly which layers are in that
    # rendering, so different combinations don't overwrite each other.
    out_dirs = {}
    for combo in combos:
        layers = [name for name, on in zip(("bg", "piv", "traction"), combo) if on]
        out_dir = results_dir / f"piv_fttc_{'_'.join(layers)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[combo] = out_dir

    micro_files = find_frames(gapped_dir, "CH4")
    bf_files = find_frames(gapped_dir, "CH1")
    matched = sorted(set(micro_files) & set(bf_files))
    if len(matched) < 2:
        raise ValueError(
            f"Need at least two timepoints with both CH4 and CH1 frames in "
            f"{gapped_dir}, found {len(matched)}"
        )

    ref_idx, *target_idxs = matched
    ref_micro = tifffile.imread(micro_files[ref_idx])
    ref_bf = tifffile.imread(bf_files[ref_idx])
    print(f"Reference: T{ref_idx:04d} ({len(target_idxs)} target frame(s))")

    # Pass 1: co-register and compute PIV/traction for every target frame,
    # so the color/arrow scales below can be fixed across all of them —
    # otherwise each frame would pick its own 99th-percentile scale and the
    # outputs wouldn't be visually comparable.
    pairs = []
    for tgt_idx in target_idxs:
        elapsed_h = (tgt_idx - ref_idx) * FRAME_INTERVAL_MINUTES / 60
        print(f"\nT{ref_idx:04d} -> T{tgt_idx:04d} (~{elapsed_h:.1f} h) ...")

        tgt_micro = tifffile.imread(micro_files[tgt_idx])
        tgt_bf = tifffile.imread(bf_files[tgt_idx])
        if tgt_micro.shape[:2] != ref_micro.shape[:2] or tgt_bf.shape[:2] != ref_bf.shape[:2]:
            raise ValueError(
                f"T{tgt_idx:04d} frames have a different (H, W) than the "
                f"reference T{ref_idx:04d} — cannot co-register"
            )

        print("  Co-registering to the reference (phase cross-correlation) ...")
        ref_micro_c, ref_bf_c, tgt_micro_c, tgt_bf_c = stabilize_pair(
            ref_micro, ref_bf, tgt_micro, tgt_bf
        )

        pair_dir = stabilized_dir / f"T{ref_idx:04d}_T{tgt_idx:04d}"
        pair_dir.mkdir(exist_ok=True)
        ref_micro_path = pair_dir / f"Image_T{ref_idx:04d}_CH4.tif"
        ref_bf_path = pair_dir / f"Image_T{ref_idx:04d}_CH1.tif"
        tgt_micro_path = pair_dir / f"Image_T{tgt_idx:04d}_CH4.tif"
        tgt_bf_path = pair_dir / f"Image_T{tgt_idx:04d}_CH1.tif"
        tifffile.imwrite(ref_micro_path, ref_micro_c)
        tifffile.imwrite(ref_bf_path, ref_bf_c)
        tifffile.imwrite(tgt_micro_path, tgt_micro_c)
        tifffile.imwrite(tgt_bf_path, tgt_bf_c)

        x = y = u = v = tx = ty = None

        exports_dir = results_dir / "exports"
        pair_tag = f"T{ref_idx:04d}_T{tgt_idx:04d}_{elapsed_h:.1f}h"

        if need_piv:
            print("  Computing PIV displacement (windef multi-pass) ...")
            gray_ref = to_gray_uint8(ref_micro_c)
            gray_tgt = to_gray_uint8(tgt_micro_c)
            x, y, u, v, flags = compute_piv(
                gray_ref, gray_tgt, make_gapped_settings(gray_ref.shape[:2])
            )

            exports_dir.mkdir(parents=True, exist_ok=True)
            disp_path = exports_dir / f"{pair_tag}_displacement.csv"
            export_displacement_csv(disp_path, x, y, u, v, args.grid_mean, args.px_to_um)
            print(f"  Displacement CSV -> {disp_path}")

        if need_traction:
            print(f"  Low-pass filtering displacement (sigma={DISPLACEMENT_SMOOTH_SIGMA} grid "
                  f"cells) and computing traction (FTTC, E={YOUNGS_MODULUS:.0f} Pa, "
                  f"nu={POISSON_RATIO}) ...")
            u_lp = gaussian_filter(u, sigma=DISPLACEMENT_SMOOTH_SIGMA)
            v_lp = gaussian_filter(v, sigma=DISPLACEMENT_SMOOTH_SIGMA)
            mesh_x_px = x[0, 1] - x[0, 0]
            mesh_y_px = y[1, 0] - y[0, 0]
            tx, ty = fttc_traction(u_lp, v_lp, mesh_x_px, mesh_y_px, YOUNGS_MODULUS, POISSON_RATIO)

            exports_dir.mkdir(parents=True, exist_ok=True)
            traction_path = exports_dir / f"{pair_tag}_traction.csv"
            export_traction_csv(traction_path, x, y, tx, ty, args.grid_mean, args.px_to_um)
            print(f"  Traction CSV -> {traction_path}")

        pairs.append(dict(
            tgt_idx=tgt_idx, elapsed_h=elapsed_h,
            ref_micro_path=ref_micro_path, tgt_bf_path=tgt_bf_path,
            x=x, y=y, u=u, v=v, tx=tx, ty=ty,
        ))

    # Fixed scales across all target frames (99th percentile, floored at
    # TRACTION_VMIN_PERCENTILE for stress) so colors/arrow lengths are
    # directly comparable between e.g. the 2h and 24h renderings.
    disp_unit = "μm" if args.px_to_um is not None else "px"
    disp_vmax = stress_vmin = stress_vmax = 0.0
    if need_piv:
        all_disp = np.concatenate([np.sqrt(p["u"] ** 2 + p["v"] ** 2).ravel() for p in pairs])
        disp_vmax = float(np.percentile(all_disp, 99))
        disp_vmax_display = disp_vmax * args.px_to_um if args.px_to_um is not None else disp_vmax
        print(f"\nDisplacement scale: 0-{disp_vmax_display:.2f} {disp_unit} (arrow key)")
    if need_traction:
        all_stress = np.concatenate([np.sqrt(p["tx"] ** 2 + p["ty"] ** 2).ravel() for p in pairs])
        stress_vmin = float(np.percentile(all_stress, TRACTION_VMIN_PERCENTILE))
        stress_vmax = float(np.percentile(all_stress, 99))
        print(f"Traction scale: {stress_vmin:.2f}-{stress_vmax:.2f} Pa (black→red, "
              f"floored at p{TRACTION_VMIN_PERCENTILE})")

    # Pass 2: render every target frame against the shared scales above.
    for p in pairs:
        print(f"\nRendering T{ref_idx:04d} -> T{p['tgt_idx']:04d} (~{p['elapsed_h']:.1f} h) ...")
        for show_background, show_piv, show_traction in combos:
            rendered = render_combined_frame(
                p["ref_micro_path"], p["tgt_bf_path"] if show_background else None,
                p["x"], p["y"],
                p["u"] if show_piv else None,
                p["v"] if show_piv else None,
                p["tx"] if show_traction else None,
                p["ty"] if show_traction else None,
                disp_vmax, stress_vmin, stress_vmax,
                show_background, show_piv, show_traction,
                color_piv_by_magnitude=not show_traction,
                px_to_um=args.px_to_um,
                transparent=args.transparent,
            )
            out_path = (out_dirs[(show_background, show_piv, show_traction)]
                        / f"T{ref_idx:04d}__T{p['tgt_idx']:04d}_{p['elapsed_h']:.1f}h.png")
            iio.imwrite(out_path, rendered)
            print(f"  Done — {out_path}")


if __name__ == "__main__":
    main()

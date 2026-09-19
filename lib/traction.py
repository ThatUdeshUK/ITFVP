from pathlib import Path
import argparse
import io

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

import openpiv.tools as piv_tools
from lib.piv import (
    SMOOTH_WINDOW,
    ensure_cache,
    load_files,
    save_video,
)

# Black at the floor (below-threshold noise), then rainbow spectrum up to
# red. Shared with piv_fttc.py's combined render (imported from here) so the
# traction heatmap looks identical whether viewed standalone or overlaid.
TRACTION_CMAP = LinearSegmentedColormap.from_list(
    "traction",
    ["black", "blue", "cyan", "green", "yellow", "orange", "red"],
)

# Substrate mechanical properties (user-supplied; needed to convert a
# displacement field into physically-meaningful traction stresses).
YOUNGS_MODULUS = 3_000.0   # Pa  (E = 3 kPa)
POISSON_RATIO  = 0.5       # nu

# FTTC's inversion is proportional to spatial wavenumber k, so it amplifies
# whatever high-frequency noise sits in the (PIV-measured) displacement field —
# the classic FTTC "salt-and-pepper" artifact. The substrate is an elastic
# continuum, so genuine displacement varies smoothly; fine-scale wiggles are
# measurement noise. Low-pass the displacement field (in grid-cell units)
# before inverting, rather than denoising the already-amplified output.
DISPLACEMENT_SMOOTH_SIGMA = 1.0

# Even after the low-pass filtering above, residual measurement noise still
# pervades the field at low magnitude — visually it shows up as background
# "speckle" that competes with genuine hotspots for the colormap's dynamic
# range. Anchoring the color scale's lower bound at this percentile (rather
# than 0) clips that speckle to a single flat background color and devotes
# the full rainbow range to the upper half of the field — exactly where the
# spatially-coherent, physically-meaningful structure lives.
TRACTION_VMIN_PERCENTILE = 50


def fttc_traction(
    u_px: np.ndarray, v_px: np.ndarray,
    mesh_x_px: float, mesh_y_px: float,
    E: float, nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fourier Transform Traction Cytometry (Butler et al. 2002): invert the
    Boussinesq Green's function for a linear-elastic half-space in Fourier
    space to convert a substrate displacement field on a regular grid into
    the traction-stress field that would produce it. Returns (tx, ty) in
    pascals on the same grid as the input.

    The elastic problem is linear, so by superposition the *interpretation*
    of the result depends entirely on what displacement you feed in: the net
    displacement since a null-force reference yields absolute traction stress,
    while frame-to-frame displacement yields the incremental change in
    traction stress between those two frames (and the per-frame increments
    sum to the absolute field).
    """
    ny, nx = u_px.shape
    pad_y, pad_x = ny // 4, nx // 4
    u_pad = np.pad(u_px, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    v_pad = np.pad(v_px, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")

    # Wavenumbers and displacements are both expressed in pixel units —
    # converting both to a physical unit (e.g. meters) via a px-to-m factor
    # would cancel out exactly (stress = E * strain, and strain is the
    # dimensionless ratio of displacement to lengthscale), so there's no
    # need to do it.
    ny_p, nx_p = u_pad.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx_p, d=mesh_x_px)
    ky = 2 * np.pi * np.fft.fftfreq(ny_p, d=mesh_y_px)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0  # dummy nonzero value — the (0, 0) mode is zeroed below anyway

    # Closed-form inverse of the Green's function tensor for markers at the
    # substrate surface (depth = 0). The k = 0 mode corresponds to a uniform
    # rigid-body translation / net force, which FTTC cannot resolve — by
    # convention it's dropped (assumes the field of view is force-balanced).
    prefactor = E / (2 * (1 - nu ** 2) * k)
    Ginv_xx = prefactor * (k ** 2 - nu * ky ** 2)
    Ginv_xy = prefactor * (nu * kx * ky)
    Ginv_yy = prefactor * (k ** 2 - nu * kx ** 2)
    Ginv_xx[0, 0] = Ginv_xy[0, 0] = Ginv_yy[0, 0] = 0.0

    Ftu_x, Ftu_y = np.fft.fft2(u_pad), np.fft.fft2(v_pad)
    Ftt_x = Ginv_xx * Ftu_x + Ginv_xy * Ftu_y
    Ftt_y = Ginv_xy * Ftu_x + Ginv_yy * Ftu_y
    tx_pad = np.fft.ifft2(Ftt_x).real
    ty_pad = np.fft.ifft2(Ftt_y).real

    tx = tx_pad[pad_y:pad_y + ny, pad_x:pad_x + nx]
    ty = ty_pad[pad_y:pad_y + ny, pad_x:pad_x + nx]
    return tx, ty


def render_traction_frame(
    image_path: Path, x, y, tx, ty, vmin: float, vmax: float,
    background_mode: str = "white",
    background_path: Path | None = None,
    px_to_um: float | None = None,
) -> np.ndarray:
    """background_mode "white" (default, matches the original behavior)
    leaves the plain matplotlib canvas; "black" fills it opaque black;
    "channel" draws background_path beneath a translucent heatmap. Traction
    stress (Pa) is unit-independent of pixel size (see fttc_traction), so
    px_to_um only rescales the spatial axes, not vmin/vmax."""
    im = piv_tools.imread(str(image_path))
    h, w = im.shape[:2]
    magnitude = np.sqrt(tx ** 2 + ty ** 2)

    xd, yd = (x, y) if px_to_um is None else (x * px_to_um, y * px_to_um)
    hd, wd = (h, w) if px_to_um is None else (h * px_to_um, w * px_to_um)

    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=300)

    show_channel_bg = background_mode == "channel" and background_path is not None
    if background_mode == "black":
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
    elif show_channel_bg:
        bg = piv_tools.imread(str(background_path))
        bg_vmin, bg_vmax = np.percentile(bg, (1, 99))
        bg_norm = np.clip((bg.astype(float) - bg_vmin) / (bg_vmax - bg_vmin), 0, 1)
        ax.imshow(bg_norm, extent=[0, wd, 0, hd], cmap="gray", vmin=0, vmax=1)

    # imshow renders the field as a single smoothly-interpolated raster (the
    # PIV grid is regular) — unlike pcolormesh, there are no per-cell quad
    # edges to show through as a faint grid overlay, even with gouraud shading.
    extent = [xd.min(), xd.max(), yd.min(), yd.max()]
    origin = "upper" if y[0, 0] > y[-1, 0] else "lower"
    mesh = ax.imshow(
        magnitude,
        extent=extent,
        origin=origin,
        cmap=TRACTION_CMAP,
        interpolation="bicubic",
        vmin=vmin,
        vmax=vmax,
        alpha=0.35 if show_channel_bg else 1.0,
    )
    cbar = fig.colorbar(mesh, ax=ax, label="Δ traction stress (Pa)", shrink=0.8)
    if background_mode == "black":
        cbar.ax.set_facecolor("black")
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")
    ax.set_aspect(1.)
    ax.set_xlim(0, wd)
    ax.set_ylim(0, hd)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return iio.imread(buf)[:, :, :3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute frame-to-frame traction-stress-change fields "
                    "(FTTC) from PIV displacement and render a rainbow "
                    "traction timelapse."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("stabilized"),
        help="Directory of stabilized microscope-channel frames to compute "
             "traction for, as written by stabilize.py --microscope-dir "
             "(default: stabilized).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Propagate --force to piv.py so it recomputes its "
             "cached PIV fields (slow) even if results/piv_fields.npz "
             "already exists.",
    )
    parser.add_argument(
        "--use-cache", dest="force", action="store_false",
        help="Propagate --use-cache instead — trust the existing "
             "results/piv_fields.npz cache as-is and never invoke "
             "piv.py (error if it's missing).",
    )
    parser.set_defaults(force=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir        = args.data_dir
    results_dir     = Path("results")
    traction_dir    = results_dir / "traction"
    piv_fields_path = results_dir / "piv_fields.npz"
    fps             = 8

    traction_dir.mkdir(parents=True, exist_ok=True)

    ensure_cache("lib/piv.py", piv_fields_path, args.force,
                 ["--data-dir", str(data_dir)])

    files = load_files(data_dir)
    stems = [f.stem for f in files]
    n_pairs = len(files) - 1

    print(f"Loading cached PIV displacement fields from {piv_fields_path} ...")
    cached = np.load(piv_fields_path)
    x, y, u_raw, v_raw = cached["x"], cached["y"], cached["u"], cached["v"]

    mesh_x_px = x[0, 1] - x[0, 0]
    mesh_y_px = y[1, 0] - y[0, 0]

    print(f"Low-pass filtering displacement (sigma={DISPLACEMENT_SMOOTH_SIGMA} grid "
          f"cells) before inversion to curb FTTC noise amplification ...")
    u_lp = gaussian_filter(u_raw, sigma=(0, DISPLACEMENT_SMOOTH_SIGMA, DISPLACEMENT_SMOOTH_SIGMA))
    v_lp = gaussian_filter(v_raw, sigma=(0, DISPLACEMENT_SMOOTH_SIGMA, DISPLACEMENT_SMOOTH_SIGMA))

    print(f"Computing frame-to-frame traction-stress change (FTTC, "
          f"E={YOUNGS_MODULUS:.0f} Pa, nu={POISSON_RATIO}) ...")
    tx_raw = np.empty_like(u_lp)
    ty_raw = np.empty_like(v_lp)
    for i in range(len(u_lp)):
        tx_raw[i], ty_raw[i] = fttc_traction(
            u_lp[i], v_lp[i], mesh_x_px, mesh_y_px, YOUNGS_MODULUS, POISSON_RATIO
        )

    print(f"Smoothing traction fields over time (window={SMOOTH_WINDOW}) ...")
    tx_smooth = uniform_filter1d(tx_raw, size=SMOOTH_WINDOW, axis=0, mode="nearest")
    ty_smooth = uniform_filter1d(ty_raw, size=SMOOTH_WINDOW, axis=0, mode="nearest")

    # Fix the rainbow color scale across the whole video: floor at the
    # TRACTION_VMIN_PERCENTILE to push background noise into a single flat
    # color and devote the colormap's range to the genuine signal above it,
    # ceiling at the 99th percentile so a few stray spikes don't wash out
    # the rest of the range.
    magnitude = np.sqrt(tx_smooth ** 2 + ty_smooth ** 2)
    vmin = float(np.percentile(magnitude, TRACTION_VMIN_PERCENTILE))
    vmax = float(np.percentile(magnitude, 99))
    print(f"Color scale: {vmin:.2f}-{vmax:.2f} Pa (rainbow, "
          f"floored at p{TRACTION_VMIN_PERCENTILE})")

    print("Rendering timelapse_traction.mp4 ...")
    frames_out = []
    for i in range(n_pairs):
        rendered = render_traction_frame(
            files[i], x, y, tx_smooth[i], ty_smooth[i], vmin, vmax
        )
        frames_out.append(rendered)
        iio.imwrite(traction_dir / f"{stems[i]}__{stems[i + 1]}.png", rendered)
        print(f"  {i + 1}/{n_pairs}", end="\r")
    print()

    out_video = results_dir / "timelapse_traction.mp4"
    save_video(frames_out, out_video, fps=fps)
    print(f"Done — {out_video}")


if __name__ == "__main__":
    main()

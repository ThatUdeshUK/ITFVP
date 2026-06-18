from pathlib import Path
import argparse
import io
import re

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

DPI = 300

import openpiv.tools as piv_tools
from piv import (
    QUIVER_LENGTH_FRACTION,
    QUIVER_WIDTH,
    SMOOTH_WINDOW,
    ensure_cache,
    load_files,
    save_video,
)
from traction import (
    DISPLACEMENT_SMOOTH_SIGMA,
    POISSON_RATIO,
    TRACTION_VMIN_PERCENTILE,
    YOUNGS_MODULUS,
    fttc_traction,
)

FRAME_INDEX_RE = re.compile(r"_T(\d+)_")


def frame_index(path: Path) -> str:
    m = FRAME_INDEX_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not find a _T<index>_ timestamp in {path.name}")
    return m.group(1)


def match_background_files(files: list[Path], background_files: list[Path]) -> list[Path]:
    """Pair each microscope frame with its co-registered brightfield
    counterpart by shared _T<index>_ timestamp (stabilize.py applies the same
    drift correction to both, so a matching timestamp means a matching crop),
    so the rendered background always shows the same instant as the
    overlaid PIV/traction fields."""
    bg_by_idx = {frame_index(f): f for f in background_files}
    try:
        return [bg_by_idx[frame_index(f)] for f in files]
    except KeyError as e:
        raise ValueError(
            f"No co-registered brightfield frame for timepoint T{e.args[0]} — "
            f"--background-dir must be a stabilize.py output co-registered "
            f"with --data-dir"
        ) from e


def render_combined_frame(
    image_path: Path, background_path: Path | None, x, y, u, v, tx, ty,
    disp_vmax: float, stress_vmin: float, stress_vmax: float,
    show_background: bool, show_piv: bool, show_traction: bool,
    title: str | None = None,
    color_piv_by_magnitude: bool = False,
) -> np.ndarray:
    # The microscope frame only sets the canvas size — stabilize.py
    # co-registers and crops it identically to the brightfield frame, so
    # both share the same (h, w) and the same PIV-grid coordinate system.
    im = piv_tools.imread(str(image_path))
    h, w = im.shape[:2]

    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=DPI)

    # With neither the frame nor the traction heatmap drawn, a white canvas
    # washes out the (often pale) arrow colors — switch to black and flip
    # the colorbar/quiverkey text to match.
    piv_only = show_piv and not show_background and not show_traction
    if piv_only:
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

    if show_background:
        # Co-registered brightfield frame as background, in the same
        # coordinate system as the PIV grid (row 0 = top of image = largest
        # y, matching transform_coordinates). Brightfield frames only span
        # ~56-255, not the full 0-255 range — stretch that into the display
        # range so the cell texture isn't washed out.
        bg = piv_tools.imread(str(background_path))
        ax.imshow(bg, extent=[0, w, 0, h], cmap="gray", vmin=55, vmax=255)

    if show_traction:
        # Traction-change heatmap as a translucent overlay so the frame (if
        # drawn) stays visible underneath it. imshow renders the field as a
        # single smoothly-interpolated raster (the PIV grid is regular) —
        # unlike pcolormesh, there are no per-cell quad edges to show through
        # as a faint grid overlay, even with gouraud shading.
        stress_magnitude = np.sqrt(tx ** 2 + ty ** 2)
        extent = [x.min(), x.max(), y.min(), y.max()]
        origin = "upper" if y[0, 0] > y[-1, 0] else "lower"
        heat = ax.imshow(
            stress_magnitude,
            extent=extent,
            origin=origin,
            cmap="rainbow",
            interpolation="bicubic",
            alpha=0.35 if show_background else 1.0,
            vmin=stress_vmin,
            vmax=stress_vmax,
        )
        fig.colorbar(heat, ax=ax, label="Δ traction stress (Pa)", shrink=0.7, pad=0.02)

    if show_piv:
        # Same magnitude/scale = arrow length (in pixel units) calibration as
        # render_piv_frame — see QUIVER_LENGTH_FRACTION in piv.py.
        mesh_spacing = abs(x[0, 1] - x[0, 0])
        quiver_scale = disp_vmax / (QUIVER_LENGTH_FRACTION * mesh_spacing)
        if color_piv_by_magnitude:
            # No traction heatmap competing for the colormap, so color arrows
            # by displacement magnitude (rainbow), as render_piv_frame does.
            magnitude = np.sqrt(u ** 2 + v ** 2)
            quiver = ax.quiver(
                x, y, u, v, magnitude,
                cmap="rainbow",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                width=QUIVER_WIDTH,
            )
            quiver.set_clim(0, disp_vmax)
            cbar = fig.colorbar(quiver, ax=ax, label="displacement (px)", shrink=0.7, pad=0.02)
            if piv_only:
                cbar.ax.yaxis.label.set_color("white")
                cbar.ax.tick_params(colors="white")
        else:
            # Solid-color arrows — a single bold color keeps the vector field
            # legible against the rainbow traction heatmap and the image
            # beneath it. Both fields are derived from the same displacement
            # measurement, so they represent the same instant and should
            # visually correlate.
            quiver = ax.quiver(
                x, y, u, v,
                color="white" if piv_only else "black",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                width=QUIVER_WIDTH,
            )
        qk = ax.quiverkey(
            quiver, X=0.83, Y=1.04, U=disp_vmax,
            label=f"{disp_vmax:.1f} px displacement", labelpos="E",
            coordinates="axes", fontproperties={"size": 8},
        )
        if piv_only:
            qk.text.set_color("white")

    if title:
        ax.set_title(title)

    ax.set_aspect(1.)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return iio.imread(buf)[:, :, :3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a combined timelapse overlaying any mix of: the "
                    "original frame, PIV displacement arrows, and the FTTC "
                    "traction-stress-change heatmap. All three are on by "
                    "default. Arrows and heatmap are both derived from the "
                    "same frame-to-frame PIV displacement field, so they "
                    "represent the same instant and stay visually consistent."
    )
    parser.add_argument(
        "--no-background", action="store_true",
        help="Omit the original frame from the rendering.",
    )
    parser.add_argument(
        "--no-piv", action="store_true",
        help="Omit PIV displacement arrows from the rendering.",
    )
    parser.add_argument(
        "--no-traction", action="store_true",
        help="Omit the traction-stress-change heatmap from the rendering.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("stabilized"),
        help="Directory of stabilized microscope-channel frames that PIV "
             "and traction are computed from, as written by stabilize.py "
             "--microscope-dir (default: stabilized).",
    )
    parser.add_argument(
        "--background-dir", type=Path, default=Path("stabilized_bf"),
        help="Directory of stabilized brightfield frames used only as the "
             "render background — must be co-registered with --data-dir, "
             "as written by stabilize.py --brightfield-dir (default: "
             "stabilized_bf).",
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
    show_background = not args.no_background
    show_piv        = not args.no_piv
    show_traction   = not args.no_traction

    if not (show_background or show_piv or show_traction):
        raise SystemExit("Nothing to render — at least one of background/PIV/"
                         "traction must stay enabled")

    data_dir        = args.data_dir
    results_dir     = Path("results")
    piv_fields_path = results_dir / "piv_fields.npz"
    fps             = 8

    # Output name reflects exactly which layers are in this rendering, so
    # different combinations don't overwrite each other.
    layers = [name for name, on in (("bg", show_background),
                                    ("piv", show_piv),
                                    ("traction", show_traction)) if on]
    suffix = "_".join(layers)
    out_dir = results_dir / f"piv_fttc_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Both PIV arrows and the traction-change heatmap are derived from the
    # same frame-to-frame displacement field, so a single cache covers either
    # (or both) of them.
    if show_piv or show_traction:
        ensure_cache("piv.py", piv_fields_path, args.force,
                     ["--data-dir", str(data_dir)])

    files = load_files(data_dir)
    stems = [f.stem for f in files]
    n_frames = len(files) - 1

    background_files = None
    if show_background:
        background_files = match_background_files(files, load_files(args.background_dir))

    print(f"Loading cached PIV displacement fields from {piv_fields_path} ...")
    cached = np.load(piv_fields_path)
    x, y, u_raw, v_raw = cached["x"], cached["y"], cached["u"], cached["v"]

    u_smooth = v_smooth = None
    disp_vmax = 0.0

    if show_piv:
        print(f"Smoothing PIV displacement over time (window={SMOOTH_WINDOW}) ...")
        u_smooth = uniform_filter1d(u_raw, size=SMOOTH_WINDOW, axis=0, mode="nearest")
        v_smooth = uniform_filter1d(v_raw, size=SMOOTH_WINDOW, axis=0, mode="nearest")
        disp_vmax = float(np.percentile(np.sqrt(u_smooth ** 2 + v_smooth ** 2), 99))

    tx_smooth = ty_smooth = None
    stress_vmin = stress_vmax = 0.0

    if show_traction:
        mesh_x_px = x[0, 1] - x[0, 0]
        mesh_y_px = y[1, 0] - y[0, 0]
        print(f"Low-pass filtering displacement (sigma={DISPLACEMENT_SMOOTH_SIGMA} grid cells) "
              f"and computing frame-to-frame traction change (FTTC, "
              f"E={YOUNGS_MODULUS:.0f} Pa, nu={POISSON_RATIO}) ...")
        u_lp = gaussian_filter(u_raw, sigma=(0, DISPLACEMENT_SMOOTH_SIGMA, DISPLACEMENT_SMOOTH_SIGMA))
        v_lp = gaussian_filter(v_raw, sigma=(0, DISPLACEMENT_SMOOTH_SIGMA, DISPLACEMENT_SMOOTH_SIGMA))
        tx_raw = np.empty_like(u_lp)
        ty_raw = np.empty_like(v_lp)
        for i in range(len(u_lp)):
            tx_raw[i], ty_raw[i] = fttc_traction(
                u_lp[i], v_lp[i], mesh_x_px, mesh_y_px, YOUNGS_MODULUS, POISSON_RATIO
            )

        print(f"Smoothing traction fields over time (window={SMOOTH_WINDOW}) ...")
        tx_smooth = uniform_filter1d(tx_raw, size=SMOOTH_WINDOW, axis=0, mode="nearest")
        ty_smooth = uniform_filter1d(ty_raw, size=SMOOTH_WINDOW, axis=0, mode="nearest")

        # Fixed scale across the whole video: floor at TRACTION_VMIN_PERCENTILE
        # to push background noise into a single flat color (devoting the
        # colormap's range to the genuine signal above it), ceiling at the
        # 99th percentile so a few stray spikes don't wash out the rest.
        stress_magnitude_all = np.sqrt(tx_smooth ** 2 + ty_smooth ** 2)
        stress_vmin = float(np.percentile(stress_magnitude_all, TRACTION_VMIN_PERCENTILE))
        stress_vmax = float(np.percentile(stress_magnitude_all, 99))

    if show_piv:
        print(f"Displacement scale: 0-{disp_vmax:.2f} px (arrow key)")
    if show_traction:
        print(f"Traction-change color scale: {stress_vmin:.2f}-{stress_vmax:.2f} Pa "
              f"(rainbow, floored at p{TRACTION_VMIN_PERCENTILE})")

    out_video = results_dir / f"timelapse_piv_fttc_{suffix}.mp4"
    print(f"Rendering {out_video.name} (layers: {', '.join(layers)}) ...")
    frames_out = []
    for i in range(n_frames):
        rendered = render_combined_frame(
            files[i], background_files[i] if show_background else None,
            x, y,
            u_smooth[i] if show_piv else None,
            v_smooth[i] if show_piv else None,
            tx_smooth[i] if show_traction else None,
            ty_smooth[i] if show_traction else None,
            disp_vmax, stress_vmin, stress_vmax,
            show_background, show_piv, show_traction,
        )
        frames_out.append(rendered)
        iio.imwrite(out_dir / f"{stems[i]}__{stems[i + 1]}.png", rendered)
        print(f"  {i + 1}/{n_frames}", end="\r")
    print()

    save_video(frames_out, out_video, fps=fps)
    print(f"Done — {out_video}")


if __name__ == "__main__":
    main()

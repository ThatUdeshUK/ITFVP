from pathlib import Path
import argparse
import io
import subprocess
import sys

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import openpiv.tools as piv_tools
import openpiv.windef as windef
import tifffile
from scipy.ndimage import uniform_filter1d

# Quiver display calibration. Raw frame-to-frame displacements here are
# sub-pixel (this dataset's 99th-percentile motion is ~0.5-0.7 px against a
# 16px PIV grid spacing) — drawn at literal data scale they're indistinguishable
# from dots. ax.quiver(..., angles="xy", scale_units="xy", scale=s) draws an
# arrow of length magnitude/s in the same (pixel) units as x, y, so picking s
# relative to the grid spacing keeps arrow length calibrated to the grid
# regardless of how big the underlying motion happens to be: the fixed,
# whole-video vmax (99th-percentile displacement) is stretched to span
# QUIVER_LENGTH_FRACTION of one grid cell — long enough to read direction and
# relative magnitude at a glance, short enough that neighboring arrows don't
# overlap into illegibility.
QUIVER_LENGTH_FRACTION = 1
QUIVER_WIDTH = 0.0025

# Per-frame displacement estimates are noisy (sub-pixel real motion vs.
# estimation error of similar magnitude), which makes the arrows flicker
# frame to frame. A short moving average over time smooths that jitter out
# while preserving genuine, temporally-coherent motion trends.
SMOOTH_WINDOW = 3

# Multi-pass window sizes and overlaps (coarse → fine).
# Brightfield frames lack the small bright trackable features that fine
# (16px) windows need — at that scale correlation is dominated by noise,
# producing large spurious displacement vectors scattered across the whole
# frame. Coarser windows average over more texture per interrogation area,
# giving stable estimates that stay near zero in the static background and
# only stand out where coherent motion actually occurs.
WINDOW_SIZES = (64, 32, 16)
OVERLAPS     = (32, 16, 8)


def make_settings() -> windef.PIVSettings:
    s = windef.PIVSettings()
    s.windowsizes    = WINDOW_SIZES
    s.overlap        = OVERLAPS
    s.num_iterations = len(WINDOW_SIZES)

    # Validation thresholds
    # s2n values for this brightfield data cluster tightly around 1.0
    # regardless of method, so the default sig2noise_validate (threshold=1.0)
    # flags ~half the field on noise alone — replacing/smearing vectors
    # everywhere instead of only the genuine outliers. Disable it and rely on
    # statistical/spatial consistency checks, which flag only the actual
    # spurious spikes (real displacements are sub-pixel; outliers reach 10s
    # of pixels).
    s.sig2noise_method   = "peak2peak"
    s.sig2noise_validate = False
    s.std_threshold      = 3
    s.median_normalized  = False
    s.median_threshold   = 2.0
    s.median_size        = 2

    s.replace_vectors      = True
    s.filter_method        = "localmean"
    s.max_filter_iteration = 4
    s.filter_kernel_size   = 2

    s.show_plot      = False
    s.show_all_plots = False
    s.save_plot      = False
    return s


def load_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("Image_T*.tif"))
    if not files:
        raise FileNotFoundError(f"No Image_T*.tif files found in {data_dir}")
    return files


def to_gray(frame: np.ndarray) -> np.ndarray:
    # Cast to uint8 (not int32) so exported PNGs are written at 8-bit depth —
    # int32 gets saved as a 16-bit PNG, making 0-255 values appear almost
    # black in viewers that normalize against the full 16-bit range.
    # Multi-channel (e.g. fluorescence) frames are collapsed via a per-pixel
    # max across channels rather than an average — in this data only one
    # channel actually carries the fluorescence signal (the others are all
    # zero), so averaging would divide the real ~75-255 signal range down to
    # ~25-85 for nothing; max preserves whichever channel(s) carry signal at
    # full strength. Brightfield frames are already single-channel uint8.
    if frame.ndim == 3:
        frame = np.max(frame, axis=2)
    return frame.astype(np.uint8)


def compute_piv(
    gray_a: np.ndarray, gray_b: np.ndarray, settings: windef.PIVSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # simple_multipass runs multi-pass deformation, validates, replaces
    # outliers, and calls transform_coordinates internally.
    # x, y are in pixels; u, v are in pixels/frame.
    x, y, u, v, flags = windef.simple_multipass(gray_a, gray_b, settings)
    return x, y, u, v, flags


def render_piv_frame(
    image_path: Path, x, y, u, v, vmin: float, vmax: float
) -> np.ndarray:
    # Color each arrow by its displacement magnitude on a rainbow scale
    # (cool = small/no motion, warm = large motion) instead of the default
    # valid/invalid blue-red scheme — vmin/vmax are fixed across the whole
    # video so colors stay comparable from frame to frame.
    im = piv_tools.imread(str(image_path))
    h, w = im.shape[:2]
    magnitude = np.sqrt(u ** 2 + v ** 2)

    # angles="xy"/scale_units="xy" draw arrows whose length is magnitude/scale
    # in the same pixel units as x, y — see QUIVER_LENGTH_FRACTION above for
    # why scale is derived from the grid spacing rather than fixed outright.
    mesh_spacing = abs(x[0, 1] - x[0, 0])
    quiver_scale = vmax / (QUIVER_LENGTH_FRACTION * mesh_spacing)

    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=300)
    # ax.imshow(im,  extent=[0, w, 0, h])
    quiver = ax.quiver(
        x, y, u, v, magnitude,
        cmap="rainbow",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale * 10,
        width=QUIVER_WIDTH,
    )
    quiver.set_clim(vmin, vmax)
    fig.colorbar(quiver, ax=ax, label="displacement (px)", shrink=0.8)
    ax.set_aspect(1.)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return iio.imread(buf)[:, :, :3]


def save_video(frames: list[np.ndarray], output: Path, fps: int = 8) -> None:
    iio.imwrite(output, frames, fps=fps, codec="libx264")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PIV on a stabilized microscope-channel's frames "
                    "and render a rainbow vector-field timelapse."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("stabilized"),
        help="Directory of stabilized microscope-channel frames to run PIV "
             "on, as written by stabilize.py --microscope-dir (default: "
             "stabilized).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute PIV fields even if a cached results/piv_fields.npz "
             "already exists (needed after changing WINDOW_SIZES/make_settings).",
    )
    parser.add_argument(
        "--use-cache", dest="force", action="store_false",
        help="Reuse results/piv_fields.npz and only re-render (default "
             "when the cache file exists).",
    )
    parser.set_defaults(force=None)
    return parser.parse_args()


def ensure_cache(
    script: str, cache_path: Path, force: bool | None,
    extra_args: list[str] = (),
) -> None:
    """Make sure cache_path exists before being loaded, by invoking the
    script that owns it — propagating a --force/--use-cache choice (and any
    extra_args, e.g. --data-dir, needed to point it at the same cache) down
    as that script's own flags, exactly as if the user had run it directly.
    Shared by traction.py / piv_fttc.py, both of which consume
    piv_fields.npz rather than computing PIV themselves."""
    if force is False:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"--use-cache given but {cache_path} does not exist — "
                f"run `uv run python {script}` first"
            )
        return

    if force is True or not cache_path.exists():
        flag = ["--force"] if force else []
        args = [*extra_args, *flag]
        print(f"Invoking {script} {' '.join(args)} to build {cache_path.name} ...")
        subprocess.run([sys.executable, script, *args], check=True)


def compute_and_export_fields(
    files: list[Path], stems: list[str], gray_dir: Path, results_dir: Path,
    fields_path: Path, fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the (slow) PIV pass once and export the raw per-pair fields to
    fields_path so later runs can re-render without recomputing PIV."""
    frames = [tifffile.imread(f) for f in files]
    h, w = frames[0].shape[:2]
    print(f"Loaded {len(frames)} frames ({w}x{h})")
    print(f"Window passes: {WINDOW_SIZES}, overlaps: {OVERLAPS}")

    print("Saving timelapse.mp4 and grayscale images ...")
    gray_frames = []
    for frame, stem in zip(frames, stems):
        gray = to_gray(frame)
        gray_frames.append(gray)
        iio.imwrite(gray_dir / f"{stem}.png", gray)
    save_video(frames, results_dir / "timelapse.mp4", fps=fps)

    settings = make_settings()

    print("Computing PIV fields (windef multi-pass) ...")
    n_pairs = len(gray_frames) - 1
    x = y = None
    us, vs = [], []
    for i in range(n_pairs):
        x, y, u, v, flags = compute_piv(gray_frames[i], gray_frames[i + 1], settings)
        us.append(u)
        vs.append(v)
        print(f"  {i + 1}/{n_pairs}", end="\r")
    print()

    u_raw, v_raw = np.stack(us), np.stack(vs)
    print(f"Exporting raw PIV fields to {fields_path} ...")
    np.savez(fields_path, x=x, y=y, u=u_raw, v=v_raw)
    return x, y, u_raw, v_raw


def main() -> None:
    args = parse_args()

    data_dir    = args.data_dir
    results_dir = Path("results")
    gray_dir    = results_dir / "gray"
    piv_dir     = results_dir / "piv"
    fields_path = results_dir / "piv_fields.npz"
    fps         = 8

    gray_dir.mkdir(parents=True, exist_ok=True)
    piv_dir.mkdir(parents=True, exist_ok=True)

    files = load_files(data_dir)
    stems = [f.stem for f in files]
    n_pairs = len(files) - 1

    # --force always recomputes; --use-cache always reuses piv_fields.npz;
    # with neither flag, reuse it automatically if it already exists.
    use_cache = fields_path.exists() if args.force is None else not args.force

    if use_cache:
        if not fields_path.exists():
            raise FileNotFoundError(
                f"--use-cache given but {fields_path} does not exist — "
                "run once without flags (or with --force) to generate it"
            )
        print(f"Loading cached PIV fields from {fields_path} ...")
        cached = np.load(fields_path)
        x, y, u_raw, v_raw = cached["x"], cached["y"], cached["u"], cached["v"]
    else:
        x, y, u_raw, v_raw = compute_and_export_fields(
            files, stems, gray_dir, results_dir, fields_path, fps
        )

    print(f"Smoothing displacement fields over time (window={SMOOTH_WINDOW}) ...")
    u_smooth = uniform_filter1d(u_raw, size=SMOOTH_WINDOW, axis=0, mode="nearest")
    v_smooth = uniform_filter1d(v_raw, size=SMOOTH_WINDOW, axis=0, mode="nearest")

    # Fix the rainbow color scale across the whole video (clipped at the 99th
    # percentile so a few stray spikes don't wash out the rest of the range)
    # so arrow colors are directly comparable from frame to frame.
    magnitude = np.sqrt(u_smooth ** 2 + v_smooth ** 2)
    vmin, vmax = 0.0, float(np.percentile(magnitude, 99))
    print(f"Color scale: {vmin:.2f}-{vmax:.2f} px (rainbow)")

    print("Rendering timelapse_piv.mp4 ...")
    piv_frames = []
    for i in range(n_pairs):
        rendered = render_piv_frame(files[i], x, y, u_smooth[i], v_smooth[i], vmin, vmax)
        piv_frames.append(rendered)
        stem_a, stem_b = stems[i], stems[i + 1]
        iio.imwrite(piv_dir / f"{stem_a}__{stem_b}.png", rendered)
        print(f"  {i + 1}/{n_pairs}", end="\r")
    print()

    piv_video = results_dir / "timelapse_piv.mp4"
    save_video(piv_frames, piv_video, fps=fps)
    print(f"Done — {piv_video}")


if __name__ == "__main__":
    main()

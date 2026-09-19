"""Driver functions the GUI calls to run the pipeline.

These wrap the exact same numerics as the CLI scripts in lib/
(stabilize.py, piv.py, traction.py, piv_fttc.py) — imported directly rather
than reimplemented — but are driven from explicit, already-ordered file
lists (instead of directory globs + filename-timestamp regexes) and report
progress via a callback instead of printing to stdout, so a GUI worker
thread can surface per-frame progress in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, uniform_filter1d

from lib import piv as _piv
from lib import piv_fttc as _combined
from lib import stabilize as _stabilize
from lib import traction as _traction

# (done, total, message) — total may be 0 for a single indivisible step.
ProgressFn = Callable[[int, int, str], None]


def _noop_progress(done: int, total: int, message: str) -> None:
    pass


@dataclass
class StabilizeResult:
    files: list[Path]
    background_files: list[Path] | None


@dataclass
class PIVResult:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    u_smooth: np.ndarray
    v_smooth: np.ndarray


@dataclass
class RenderStyle:
    """Cosmetic render options shared (and kept in sync) across the PIV,
    traction, and combined timelines — see gui/state.py's RenderSettings for
    the live, GUI-synced version this is snapshotted from."""
    background_mode: str = "white"        # "white" | "black" | "channel"
    px_to_um: float | None = None
    arrow_width: float = _piv.QUIVER_WIDTH
    arrow_length: float = _piv.QUIVER_LENGTH_FRACTION
    arrow_color_mode: str = "rainbow"     # "rainbow" | "solid"
    arrow_color: str = "black"


@dataclass
class TractionResult:
    tx: np.ndarray
    ty: np.ndarray
    tx_smooth: np.ndarray
    ty_smooth: np.ndarray


def run_stabilize(
    micro_files: list[Path],
    out_dir: Path,
    background_files: list[Path] | None,
    bg_out_dir: Path | None,
    progress: ProgressFn = _noop_progress,
) -> StabilizeResult:
    """Estimate cumulative stage drift from micro_files (sequential phase
    cross-correlation) and apply the same correction to both channels, as
    stabilize.py does — but paired by list position rather than a filename
    timestamp, since the GUI already knows the correct pairing."""
    has_bg = background_files is not None
    if has_bg and len(background_files) != len(micro_files):
        raise ValueError(
            f"Main channel has {len(micro_files)} frames but background "
            f"channel has {len(background_files)} — they must be paired 1:1"
        )

    n = len(micro_files)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.tif"):
        f.unlink()
    if has_bg:
        bg_out_dir.mkdir(parents=True, exist_ok=True)
        for f in bg_out_dir.glob("*.tif"):
            f.unlink()

    progress(0, n, "Loading frames...")
    micro_frames = [tifffile.imread(f) for f in micro_files]
    bg_frames = [tifffile.imread(f) for f in background_files] if has_bg else None
    if has_bg and bg_frames[0].shape[:2] != micro_frames[0].shape[:2]:
        raise ValueError(
            f"Main-channel and background frames have different (H, W) — "
            f"{micro_frames[0].shape[:2]} vs {bg_frames[0].shape[:2]} — they "
            f"must come from the same field of view to be co-registered"
        )

    grays = [_stabilize.to_gray(f) for f in micro_frames]
    h, w = grays[0].shape

    progress(0, n, "Estimating stage drift (phase cross-correlation)...")
    shifts = _stabilize.estimate_cumulative_shifts(grays)
    row_crop, col_crop = _stabilize.common_crop(shifts, h, w)

    out_files: list[Path] = []
    bg_out_files: list[Path] | None = [] if has_bg else None
    for i, (micro_frame, micro_file, shift) in enumerate(zip(micro_frames, micro_files, shifts)):
        cropped = _stabilize.apply_shift(micro_frame, shift)[row_crop, col_crop]
        out_path = out_dir / micro_file.name
        tifffile.imwrite(out_path, cropped)
        out_files.append(out_path)
        if has_bg:
            bg_cropped = _stabilize.apply_shift(bg_frames[i], shift)[row_crop, col_crop]
            bg_out_path = bg_out_dir / background_files[i].name
            tifffile.imwrite(bg_out_path, bg_cropped)
            bg_out_files.append(bg_out_path)
        progress(i + 1, n, f"Stabilized frame {i + 1}/{n}")

    return StabilizeResult(files=out_files, background_files=bg_out_files)


def run_piv(
    files: list[Path],
    use_gapped_settings: bool = False,
    window_sizes: tuple[int, ...] | None = None,
    progress: ProgressFn = _noop_progress,
) -> PIVResult:
    """Multi-pass windowed PIV (OpenPIV windef) between each consecutive
    pair of frames, exactly as piv.py does.

    window_sizes overrides the coarse-to-fine pass sizes (default: piv.py's
    WINDOW_SIZES / GAPPED_WINDOW_SIZES, chosen by use_gapped_settings, which
    also still governs the validation thresholds). Overlap is always half
    the window size for each pass."""
    n_pairs = len(files) - 1
    if n_pairs < 1:
        raise ValueError("Need at least 2 frames to compute PIV")

    settings = _piv.make_gapped_settings() if use_gapped_settings else _piv.make_settings()
    if window_sizes:
        settings.windowsizes = tuple(window_sizes)
        settings.overlap = tuple(max(w // 2, 1) for w in window_sizes)
        settings.num_iterations = len(window_sizes)

    progress(0, n_pairs, "Loading frames...")
    gray_frames = [_piv.to_gray(tifffile.imread(f)) for f in files]

    x = y = None
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    for i in range(n_pairs):
        x, y, u, v, _flags = _piv.compute_piv(gray_frames[i], gray_frames[i + 1], settings)
        us.append(u)
        vs.append(v)
        progress(i + 1, n_pairs, f"PIV pair {i + 1}/{n_pairs}")

    u_raw, v_raw = np.stack(us), np.stack(vs)
    u_smooth = uniform_filter1d(u_raw, size=_piv.SMOOTH_WINDOW, axis=0, mode="nearest")
    v_smooth = uniform_filter1d(v_raw, size=_piv.SMOOTH_WINDOW, axis=0, mode="nearest")
    return PIVResult(x=x, y=y, u=u_raw, v=v_raw, u_smooth=u_smooth, v_smooth=v_smooth)


def run_traction(
    piv_result: PIVResult,
    youngs_modulus: float = _traction.YOUNGS_MODULUS,
    poisson_ratio: float = _traction.POISSON_RATIO,
    progress: ProgressFn = _noop_progress,
) -> TractionResult:
    """FTTC inversion of the PIV displacement field into traction stress,
    exactly as traction.py does."""
    x, y, u_raw, v_raw = piv_result.x, piv_result.y, piv_result.u, piv_result.v
    n = len(u_raw)
    mesh_x_px = x[0, 1] - x[0, 0]
    mesh_y_px = y[1, 0] - y[0, 0]

    progress(0, n, "Low-pass filtering displacement...")
    u_lp = gaussian_filter(u_raw, sigma=(0, _traction.DISPLACEMENT_SMOOTH_SIGMA, _traction.DISPLACEMENT_SMOOTH_SIGMA))
    v_lp = gaussian_filter(v_raw, sigma=(0, _traction.DISPLACEMENT_SMOOTH_SIGMA, _traction.DISPLACEMENT_SMOOTH_SIGMA))

    tx_raw = np.empty_like(u_lp)
    ty_raw = np.empty_like(v_lp)
    for i in range(n):
        tx_raw[i], ty_raw[i] = _traction.fttc_traction(
            u_lp[i], v_lp[i], mesh_x_px, mesh_y_px, youngs_modulus, poisson_ratio
        )
        progress(i + 1, n, f"FTTC inversion {i + 1}/{n}")

    tx_smooth = uniform_filter1d(tx_raw, size=_piv.SMOOTH_WINDOW, axis=0, mode="nearest")
    ty_smooth = uniform_filter1d(ty_raw, size=_piv.SMOOTH_WINDOW, axis=0, mode="nearest")
    return TractionResult(tx=tx_raw, ty=ty_raw, tx_smooth=tx_smooth, ty_smooth=ty_smooth)


def _reset_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()


def render_piv_timeline(
    piv_result: PIVResult,
    frame_files: list[Path],
    out_dir: Path,
    background_files: list[Path] | None = None,
    style: RenderStyle | None = None,
    fps: int = 8,
    progress: ProgressFn = _noop_progress,
) -> tuple[Path, Path]:
    style = style or RenderStyle()
    _reset_dir(out_dir)
    magnitude = np.sqrt(piv_result.u_smooth ** 2 + piv_result.v_smooth ** 2)
    vmin, vmax = 0.0, float(np.percentile(magnitude, 99))
    n = len(piv_result.u_smooth)
    use_channel_bg = style.background_mode == "channel" and background_files is not None
    frames = []
    for i in range(n):
        rendered = _piv.render_piv_frame(
            frame_files[i], piv_result.x, piv_result.y,
            piv_result.u_smooth[i], piv_result.v_smooth[i], vmin, vmax,
            background_mode=style.background_mode,
            background_path=background_files[i] if use_channel_bg else None,
            arrow_width=style.arrow_width,
            arrow_length_fraction=style.arrow_length,
            color_mode=style.arrow_color_mode,
            arrow_color=style.arrow_color,
            px_to_um=style.px_to_um,
        )
        frames.append(rendered)
        iio.imwrite(out_dir / f"frame_{i:04d}.png", rendered)
        progress(i + 1, n, f"Rendering PIV frame {i + 1}/{n}")
    video_path = out_dir / "timeline_piv.mp4"
    _piv.save_video(frames, video_path, fps=fps)
    return out_dir, video_path


def render_traction_timeline(
    traction_result: TractionResult,
    piv_result: PIVResult,
    frame_files: list[Path],
    out_dir: Path,
    background_files: list[Path] | None = None,
    style: RenderStyle | None = None,
    fps: int = 8,
    progress: ProgressFn = _noop_progress,
) -> tuple[Path, Path]:
    style = style or RenderStyle()
    _reset_dir(out_dir)
    magnitude = np.sqrt(traction_result.tx_smooth ** 2 + traction_result.ty_smooth ** 2)
    vmin = float(np.percentile(magnitude, _traction.TRACTION_VMIN_PERCENTILE))
    vmax = float(np.percentile(magnitude, 99))
    n = len(traction_result.tx_smooth)
    use_channel_bg = style.background_mode == "channel" and background_files is not None
    frames = []
    for i in range(n):
        rendered = _traction.render_traction_frame(
            frame_files[i], piv_result.x, piv_result.y,
            traction_result.tx_smooth[i], traction_result.ty_smooth[i], vmin, vmax,
            background_mode=style.background_mode,
            background_path=background_files[i] if use_channel_bg else None,
            px_to_um=style.px_to_um,
        )
        frames.append(rendered)
        iio.imwrite(out_dir / f"frame_{i:04d}.png", rendered)
        progress(i + 1, n, f"Rendering traction frame {i + 1}/{n}")
    video_path = out_dir / "timeline_traction.mp4"
    _piv.save_video(frames, video_path, fps=fps)
    return out_dir, video_path


def render_combined_timeline(
    piv_result: PIVResult,
    traction_result: TractionResult,
    frame_files: list[Path],
    background_files: list[Path] | None,
    out_dir: Path,
    style: RenderStyle | None = None,
    fps: int = 8,
    progress: ProgressFn = _noop_progress,
) -> tuple[Path, Path]:
    style = style or RenderStyle()
    _reset_dir(out_dir)
    show_background = style.background_mode == "channel" and background_files is not None
    # "auto" (used when a channel background was requested but isn't
    # available) falls back to render_combined_frame's own white/black
    # heuristic rather than forcing either.
    canvas_color = style.background_mode if style.background_mode in ("white", "black") else "auto"
    disp_vmax = float(np.percentile(np.sqrt(piv_result.u_smooth ** 2 + piv_result.v_smooth ** 2), 99))
    stress_mag = np.sqrt(traction_result.tx_smooth ** 2 + traction_result.ty_smooth ** 2)
    stress_vmin = float(np.percentile(stress_mag, _traction.TRACTION_VMIN_PERCENTILE))
    stress_vmax = float(np.percentile(stress_mag, 99))
    n = len(piv_result.u_smooth)
    frames = []
    for i in range(n):
        rendered = _combined.render_combined_frame(
            frame_files[i], background_files[i] if show_background else None,
            piv_result.x, piv_result.y,
            piv_result.u_smooth[i], piv_result.v_smooth[i],
            traction_result.tx_smooth[i], traction_result.ty_smooth[i],
            disp_vmax, stress_vmin, stress_vmax,
            show_background, True, True,
            px_to_um=style.px_to_um,
            canvas_color=canvas_color,
            arrow_width=style.arrow_width,
            arrow_length_fraction=style.arrow_length,
            arrow_color=style.arrow_color if style.arrow_color_mode == "solid" else None,
            color_piv_by_magnitude=(style.arrow_color_mode == "rainbow"),
        )
        frames.append(rendered)
        iio.imwrite(out_dir / f"frame_{i:04d}.png", rendered)
        progress(i + 1, n, f"Rendering combined frame {i + 1}/{n}")
    video_path = out_dir / "timeline_combined.mp4"
    _piv.save_video(frames, video_path, fps=fps)
    return out_dir, video_path

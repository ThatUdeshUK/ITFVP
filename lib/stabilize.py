from pathlib import Path
import argparse
import re

import numpy as np
import tifffile
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

UPSAMPLE_FACTOR = 10  # sub-pixel precision for drift estimation

FRAME_INDEX_RE = re.compile(r"_T(\d+)_")


def load_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("Image_T*.tif"))
    if not files:
        raise FileNotFoundError(f"No Image_T*.tif files found in {data_dir}")
    return files


def frame_index(path: Path) -> str:
    m = FRAME_INDEX_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not find a _T<index>_ timestamp in {path.name}")
    return m.group(1)


def match_frames(
    microscope_files: list[Path], brightfield_files: list[Path]
) -> tuple[list[Path], list[Path]]:
    """The microscope and brightfield acquisitions don't necessarily cover
    exactly the same timepoints (e.g. one may be missing its first frame) —
    pair files up by their shared _T<index>_ timestamp and keep only
    timepoints present in both, so each (microscope, brightfield) pair below
    captures the same instant and the two stabilized sequences stay
    frame-for-frame aligned."""
    micro_by_idx = {frame_index(f): f for f in microscope_files}
    bf_by_idx = {frame_index(f): f for f in brightfield_files}
    shared = sorted(set(micro_by_idx) & set(bf_by_idx))
    if not shared:
        raise ValueError("No shared _T<index>_ timestamps between the microscope "
                         "and brightfield directories — cannot pair frames")

    micro_only = sorted(set(micro_by_idx) - set(bf_by_idx))
    bf_only = sorted(set(bf_by_idx) - set(micro_by_idx))
    if micro_only or bf_only:
        print(f"Dropping unmatched timepoints — microscope-only: {micro_only or 'none'}, "
              f"brightfield-only: {bf_only or 'none'}")

    return [micro_by_idx[i] for i in shared], [bf_by_idx[i] for i in shared]


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Collapse a frame to single-channel float for registration. Multi-channel
    (e.g. fluorescence) frames are collapsed via a per-pixel max across
    channels rather than an average — in this data only one channel actually
    carries the fluorescence signal (the others are all zero), so averaging
    would divide the real ~75-255 signal range down to ~25-85 for nothing;
    max preserves whichever channel(s) carry signal at full strength.
    Brightfield frames are already single-channel and just need a float cast."""
    if frame.ndim == 3:
        return np.max(frame, axis=2).astype(np.float64)
    return frame.astype(np.float64)


def estimate_cumulative_shifts(grays: list[np.ndarray]) -> np.ndarray:
    """Sequentially register each frame to the previous one and accumulate
    the (dy, dx) steps to get drift relative to the first frame."""
    shifts = np.zeros((len(grays), 2))
    cumulative = np.zeros(2)
    for i in range(1, len(grays)):
        step, _, _ = phase_cross_correlation(
            grays[i - 1], grays[i], upsample_factor=UPSAMPLE_FACTOR
        )
        cumulative = cumulative + step
        shifts[i] = cumulative
    return shifts


def apply_shift(frame: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Shift a frame by (dy, dx); new regions are filled with zeros.
    Multi-channel frames are shifted one channel at a time."""
    if frame.ndim == 3:
        channels = [
            ndi_shift(frame[..., c], shift=shift, order=1, mode="constant", cval=0)
            for c in range(frame.shape[-1])
        ]
        return np.stack(channels, axis=-1).astype(frame.dtype)
    return ndi_shift(frame, shift=shift, order=1, mode="constant", cval=0).astype(frame.dtype)


def reset_output_dir(out_dir: Path) -> None:
    """Clear any frames left over from a previous run before writing this
    run's matched set. Each run produces a self-consistent set of co-cropped
    frames — leaving stale frames in place (e.g. an unmatched timepoint that
    a prior run did include, cropped to a different size) would silently mix
    incompatible frames into the directory and break downstream processing
    that assumes every frame in it shares one shape (e.g. save_video)."""
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*.tif"):
        f.unlink()


def common_crop(shifts: np.ndarray, h: int, w: int) -> tuple[slice, slice]:
    """Region with real (non-fill) data in every stabilized frame."""
    dy, dx = shifts[:, 0], shifts[:, 1]
    top    = int(np.ceil(max(dy.max(), 0)))
    bottom = int(np.ceil(max(-dy.min(), 0)))
    left   = int(np.ceil(max(dx.max(), 0)))
    right  = int(np.ceil(max(-dx.min(), 0)))
    return slice(top, h - bottom), slice(left, w - right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Co-register a microscope channel with a brightfield "
                    "channel from the same acquisition: estimate drift from "
                    "the microscope frames (the data PIV/traction are "
                    "computed from downstream) and apply that exact "
                    "correction to both, so the brightfield frames stay "
                    "pixel-aligned with the microscope frames for use as a "
                    "render background later in the pipeline."
    )
    parser.add_argument(
        "--microscope-dir", type=Path,
        help="Raw microscope-channel TIFF directory — drift is estimated "
             "from these frames, and this is the data PIV/traction are "
             "computed from downstream (default: data/CH4).",
    )
    parser.add_argument(
        "--brightfield-dir", type=Path,
        help="Raw brightfield TIFF directory — co-registered to the "
             "microscope channel here, but otherwise only used as the "
             "render background downstream (default: data/CH1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    micro_out_dir = Path("stabilized")
    bf_out_dir    = Path("stabilized_bf")
    reset_output_dir(micro_out_dir)
    reset_output_dir(bf_out_dir)

    micro_files, bf_files = [], []
    if args.microscope_dir:
        micro_files = load_files(args.microscope_dir)

    if args.brightfield_dir:
        bf_files = load_files(args.microscope_dir)

    has_both = args.microscope_dir and args.brightfield_dir
    if has_both:
        micro_files, bf_files = match_frames(
            load_files(args.microscope_dir), load_files(args.brightfield_dir)
        )

    micro_frames = [tifffile.imread(f) for f in micro_files]
    bf_frames = [tifffile.imread(f) for f in bf_files]
    if has_both and bf_frames[0].shape[:2] != micro_frames[0].shape[:2]:
        raise ValueError(
            f"Microscope and brightfield frames have different (H, W) — "
            f"{micro_frames[0].shape[:2]} vs {bf_frames[0].shape[:2]} — "
            f"they must come from the same field of view to be co-registered"
        )

    grays = [to_gray(f) for f in micro_frames]
    h, w = grays[0].shape
    print(f"Loaded {len(micro_frames)} matched microscope/brightfield frame pairs ({w}x{h})")

    print("Estimating drift from the microscope channel via sequential phase cross-correlation ...")
    shifts = estimate_cumulative_shifts(grays)
    print(
        f"Cumulative drift range: "
        f"dy [{shifts[:, 0].min():.2f}, {shifts[:, 0].max():.2f}] px, "
        f"dx [{shifts[:, 1].min():.2f}, {shifts[:, 1].max():.2f}] px"
    )

    row_crop, col_crop = common_crop(shifts, h, w)
    cw, ch = col_crop.stop - col_crop.start, row_crop.stop - row_crop.start
    print(f"Common valid region after correction: {cw}x{ch} (rows {row_crop}, cols {col_crop})")

    if has_both:
        print(f"Applying the microscope channel's drift correction to both channels "
              f"and writing to {micro_out_dir}/ and {bf_out_dir}/ ...")
        for micro_frame, micro_file, bf_frame, bf_file, shift in zip(
            micro_frames, micro_files, bf_frames, bf_files, shifts
        ):
            micro_cropped = apply_shift(micro_frame, shift)[row_crop, col_crop]
            bf_cropped = apply_shift(bf_frame, shift)[row_crop, col_crop]
            tifffile.imwrite(micro_out_dir / micro_file.name, micro_cropped)
            tifffile.imwrite(bf_out_dir / bf_file.name, bf_cropped)
    else:
        print(f"Applying the microscope channel's drift correction "
              f"and writing to {micro_out_dir}/ ...")
        for micro_frame, micro_file, shift in zip(
            micro_frames, micro_files, shifts
        ):
            micro_cropped = apply_shift(micro_frame, shift)[row_crop, col_crop]
            tifffile.imwrite(micro_out_dir / micro_file.name, micro_cropped)

    print(f"Done — {len(micro_files)} co-registered frame pairs ({cw}x{ch}) "
          f"in {micro_out_dir}/ and {bf_out_dir}/")


if __name__ == "__main__":
    main()

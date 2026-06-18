# ITFVP — Image to Traction Force Visualization Pipeline

A Python pipeline for computing and visualising cell traction forces from fluorescence microscopy time-series images. It corrects for stage drift, computes substrate displacement via PIV (Particle Image Velocimetry), and inverts those displacements to traction stress via FTTC (Fourier Transform Traction Cytometry).

---

## Project structure

```
.
├── stabilize.py        # Stage-drift correction via phase cross-correlation
├── piv.py       # PIV displacement fields (OpenPIV multi-pass windef)
├── traction.py         # FTTC traction stress from PIV displacement
├── piv_fttc.py     # Combined overlay renderer (background + PIV + traction)
├── gapped.py           # Same pipeline for gapped timepoint snapshots
├── align_keypoints.py  # Alternative: affine alignment via ORB keypoints + RANSAC
├── data/               # Your input image directory (this contains `CH4` and `CH1`)
└── results/            # All outputs written here (created automatically)
```

---

## Setup

**Install uv** (if not already installed):

```bash
# macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies (Python 3.14 is fetched automatically):

```bash
uv sync
```

---

## Quick run

End-to-end pipeline in two commands:

```bash
# 1. Correct stage drift and co-register the brightfield channel
uv run python stabilize.py --microscope-dir data/CH4 --brightfield-dir data/CH1

# 2. Compute PIV displacement, FTTC traction stress, and render the combined overlay
uv run python piv_fttc.py
```

Outputs land in `results/piv_fttc_bg_piv_traction/` (one PNG per consecutive frame pair) and `results/timelapse_piv_fttc_bg_piv_traction.mp4`.

---

## Pipeline — dense time series

The four scripts form a sequential pipeline. Each step writes outputs consumed by the next.

### 1. Stabilize — correct stage drift

```bash
uv run python stabilize.py [--microscope-dir data/CH4] [--brightfield-dir data/CH1]
```

Estimates cumulative stage drift from the microscope channel (CH4) via sequential phase cross-correlation, then applies the same correction to both channels so they remain pixel-aligned throughout.

**Outputs:** `stabilized/` (drift-corrected CH4 TIFFs), `stabilized_bf/` (drift-corrected CH1 TIFFs)

---

### 2. PIV — compute displacement fields

```bash
uv run python piv.py [--data-dir stabilized] [--force | --use-cache]
```

Runs OpenPIV's multi-pass window deformation on consecutive stabilized frame pairs to measure substrate displacement (in pixels) between each pair.

**Outputs:**
- `results/piv_fields.npz` — cached `x, y, u, v` arrays (reused by downstream scripts)
- `results/timelapse.mp4` — raw frame timelapse
- `results/timelapse_piv.mp4` — PIV arrow overlay timelapse
- `results/piv/*.png` — per-pair arrow images

---

### 3. Traction — FTTC stress fields

```bash
uv run python traction.py [--use-cache | --force]
```

Inverts the Boussinesq Green's function in Fourier space (FTTC, Butler et al. 2002) to convert the PIV displacement field into traction stress (Pa). Substrate properties are set at the top of the file: `YOUNGS_MODULUS = 3000 Pa`, `POISSON_RATIO = 0.5`.

**Outputs:**
- `results/timelapse_traction.mp4` — traction heatmap timelapse

---

### 4. Combined render — overlay any mix of layers

```bash
uv run python piv_fttc.py [--no-background] [--no-piv] [--no-traction] [--use-cache | --force]
```

Renders any combination of three layers on a single canvas:
- **background** — co-registered brightfield frame (gray)
- **piv** — displacement arrows (black, scaled to grid spacing)
- **traction** — FTTC stress heatmap (rainbow, translucent over background)

The output directory name encodes the active layers, so all combinations can coexist:

| Flags | Output directory |
|---|---|
| *(default — all on)* | `results/piv_fttc_bg_piv_traction/` |
| `--no-traction` | `results/piv_fttc_bg_piv/` |
| `--no-background` | `results/piv_fttc_piv_traction/` |
| `--no-background --no-piv` | `results/piv_fttc_traction/` |

**Outputs:** `results/piv_fttc_<layers>/`, `results/timelapse_piv_fttc_<layers>.mp4`

---

## Pipeline — sparse timepoint snapshots (`gapped/`)

For snapshots taken at widely spaced intervals (e.g. 2h, 6h, 12h, 24h), displacement and traction are computed from the **first (reference) frame to each later snapshot**, giving absolute traction stress (not frame-to-frame increments) — assuming the reference frame is unstressed.

```bash
uv run python gapped.py [--gapped-dir gapped] [--all] [--no-background] [--no-piv] [--no-traction]
```

| Flag | Effect |
|---|---|
| *(default — all on)* | Render all three layers |
| `--all` | Render all 7 non-empty layer combinations at once |
| `--no-background` | Omit the snapshot brightfield frame |
| `--no-piv` | Omit PIV displacement arrows |
| `--no-traction` | Omit FTTC traction heatmap |

Color scales (displacement and stress) are fixed across all timepoints so outputs are directly comparable. PIV-only renders use a black background to improve arrow visibility.

**Outputs:** `results/gapped/piv_fttc_<layers>/T{ref}__T{tgt}_{elapsed}h.png`

---

## Key parameters

| File | Parameter | Default | Effect |
|---|---|---|---|
| `piv.py` | `WINDOW_SIZES` | `(64, 32, 16)` | Interrogation window sizes (coarse → fine) |
| `piv.py` | `OVERLAPS` | `(32, 16, 8)` | Window overlaps per pass |
| `piv.py` | `QUIVER_LENGTH_FRACTION` | `1` | Arrow length relative to grid spacing |
| `traction.py` | `YOUNGS_MODULUS` | `3000 Pa` | Substrate stiffness |
| `traction.py` | `POISSON_RATIO` | `0.5` | Substrate compressibility |
| `traction.py` | `DISPLACEMENT_SMOOTH_SIGMA` | `1.0` | Low-pass filter applied before FTTC inversion |
| `traction.py` | `TRACTION_VMIN_PERCENTILE` | `50` | Colormap floor (clips background noise) |
| `stabilize.py` | `UPSAMPLE_FACTOR` | `10` | Sub-pixel precision for drift estimation |

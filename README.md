# ITFVP — Image to Traction Force Visualization Pipeline

A Python pipeline for computing and visualising cell traction forces from fluorescence microscopy time-series images. It corrects for stage drift, computes substrate displacement via PIV (Particle Image Velocimetry), and inverts those displacements to traction stress via FTTC (Fourier Transform Traction Cytometry).

---

## Project structure

```
.
├── main.py                  # GUI entry point (uv run python main.py)
├── gui/                     # PySide6 desktop GUI — see the GUI section below
├── lib/                     # Pipeline scripts — run via `uv run python -m lib.<name>`
│   ├── stabilize.py         # Stage-drift correction via phase cross-correlation
│   ├── piv.py                # PIV displacement fields (OpenPIV multi-pass windef)
│   ├── traction.py           # FTTC traction stress from PIV displacement
│   ├── piv_fttc.py          # Combined overlay renderer (background + PIV + traction)
│   ├── gapped.py             # Same pipeline for gapped timepoint snapshots
│   └── align_keypoints.py    # Alternative: affine alignment via ORB keypoints + RANSAC
├── data/                    # Your input image directory (this contains `CH4` and `CH1`)
└── results/                 # All outputs written here (created automatically)
```

`lib/` is a real Python package: its scripts import each other (e.g. `piv_fttc.py`
needs `piv.py` and `traction.py`), so run them as modules from the repo root
(`uv run python -m lib.piv_fttc`) rather than by file path — module execution is
what makes those imports resolve.

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
uv run python -m lib.stabilize --microscope-dir data/CH4 --brightfield-dir data/CH1

# 2. Compute PIV displacement, FTTC traction stress, and render the combined overlay
uv run python -m lib.piv_fttc
```

Outputs land in `results/piv_fttc_bg_piv_traction/` (one PNG per consecutive frame pair) and `results/timelapse_piv_fttc_bg_piv_traction.mp4`.

---

## GUI

A cross-platform desktop GUI (PySide6/Qt) wraps the same pipeline for interactive use:

```bash
uv run python main.py
```

On launch you choose or create a **project folder** — a JSON file
(`itfvp_project.json`) alongside cached PIV/traction fields is kept there so
reopening the same folder later restores exactly where you left off. Opening a
folder (new or existing) lands you on one flat row of stage tabs — **Images**,
**Stabilize**, **PIV displacement**, **Traction stress**, **Combined overlay**.
**Import...** (select/copy in your time-series TIFFs — originals are never
modified) opens as its own screen from the toolbar at any point.

**Images** is the image-management view, one sub-tab per channel (main,
background): a thumbnail grid (rename/reorder/remove) paired with a playback
timeline — selecting a thumbnail jumps the timeline to it, and scrubbing or
playing the timeline selects the matching thumbnail back. The other four tabs
each pair a timeline viewer with a side panel of actions (run/re-run the
computation, a **Render properties** section — background (white/black/the
imported background channel), axis unit (px or µm, given a scale), and for PIV
and Combined the arrow width/length/color — then render the timeline, export as
`.mp4` or a `.zip` of frames. Properties shared between tabs (background, axis
unit, and arrow style between PIV and Combined) stay synchronized, so changing
one updates the others. Run a step, watch the render update next to it, and
iterate without switching tabs. See `gui/` — `gui/pipeline.py` calls the exact
same functions as the CLI
scripts below rather than reimplementing the numerics.

---

## Pipeline — dense time series

The four scripts form a sequential pipeline. Each step writes outputs consumed by the next.

### 1. Stabilize — correct stage drift

```bash
uv run python -m lib.stabilize [--microscope-dir data/CH4] [--brightfield-dir data/CH1]
```

Estimates cumulative stage drift from the microscope channel (CH4) via sequential phase cross-correlation, then applies the same correction to both channels so they remain pixel-aligned throughout.

**Outputs:** `stabilized/` (drift-corrected CH4 TIFFs), `stabilized_bf/` (drift-corrected CH1 TIFFs)

---

### 2. PIV — compute displacement fields

```bash
uv run python -m lib.piv [--data-dir stabilized] [--force | --use-cache]
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
uv run python -m lib.traction [--use-cache | --force]
```

Inverts the Boussinesq Green's function in Fourier space (FTTC, Butler et al. 2002) to convert the PIV displacement field into traction stress (Pa). Substrate properties are set at the top of the file: `YOUNGS_MODULUS = 3000 Pa`, `POISSON_RATIO = 0.5`.

**Outputs:**
- `results/timelapse_traction.mp4` — traction heatmap timelapse

---

### 4. Combined render — overlay any mix of layers

```bash
uv run python -m lib.piv_fttc [--no-background] [--no-piv] [--no-traction] [--use-cache | --force]
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
uv run python -m lib.gapped [--gapped-dir gapped] [--all] [--no-background] [--no-piv] [--no-traction]
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
| `lib/piv.py` | `WINDOW_SIZES` | `(64, 32, 16)` | Interrogation window sizes (coarse → fine) |
| `lib/piv.py` | `OVERLAPS` | `(32, 16, 8)` | Window overlaps per pass |
| `lib/piv.py` | `QUIVER_LENGTH_FRACTION` | `1` | Arrow length relative to grid spacing |
| `lib/traction.py` | `YOUNGS_MODULUS` | `3000 Pa` | Substrate stiffness |
| `lib/traction.py` | `POISSON_RATIO` | `0.5` | Substrate compressibility |
| `lib/traction.py` | `DISPLACEMENT_SMOOTH_SIGMA` | `1.0` | Low-pass filter applied before FTTC inversion |
| `lib/traction.py` | `TRACTION_VMIN_PERCENTILE` | `50` | Colormap floor (clips background noise) |
| `lib/stabilize.py` | `UPSAMPLE_FACTOR` | `10` | Sub-pixel precision for drift estimation |

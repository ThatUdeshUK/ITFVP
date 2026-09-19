"""PIV displacement tab: compute displacement fields and render the
arrow-overlay timeline.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lib import piv as _piv
from lib.gapped import export_displacement_csv
from .. import pipeline
from ..state import AppState
from ..widgets import ActionRunner, FrameViewer, RenderOptionsPanel
from .tab_common import (
    ExportMixin,
    action_splitter,
    export_csv_archive,
    is_gapped_piv,
    properties_group,
    side_panel,
)


class PIVTab(QWidget, ExportMixin):
    key = "piv"
    title = "PIV"

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self.viewer = FrameViewer(empty_text="No PIV render yet — run PIV computation, then Render.")

        self.piv_preset = QComboBox()
        self.piv_preset.addItem("Dense time series (fine passes)", False)
        self.piv_preset.addItem("Gapped / large displacement (coarse-to-fine)", True)
        self.piv_preset.currentIndexChanged.connect(self._update_window_size_placeholder)

        self.window_sizes_edit = QLineEdit()
        self.window_sizes_edit.textChanged.connect(self._update_overlap_preview)
        self.overlap_preview = QLabel()
        self.overlap_preview.setWordWrap(True)

        self.compute_btn = QPushButton("Run PIV Computation")
        self.compute_btn.clicked.connect(self._run_compute)

        self.computation_group = properties_group(
            state, "piv.computation", "Computation properties",
            QLabel("Validation preset:"),
            self.piv_preset,
            QLabel("Window sizes (px, coarse→fine, comma-separated; blank = preset default):"),
            self.window_sizes_edit,
            self.overlap_preview,
        )

        self.style_panel = RenderOptionsPanel(
            state, "piv.render", show_arrows=True,
            background_available=lambda: bool(self.state.stabilized_bg_files),
        )

        self.render_btn = QPushButton("Render")
        self.render_btn.clicked.connect(self._run_render)

        self.export_computations_btn = QPushButton("Export Computations (.zip)...")
        self.export_computations_btn.setToolTip(
            "Exports the raw per-pair displacement field as CSV (one file per "
            "frame pair) — only available for a Gapped-preset PIV computation."
        )
        self.export_computations_btn.clicked.connect(self._export_computations)

        self.runner = ActionRunner()
        self.runner.manage([self.compute_btn, self.render_btn])

        side = side_panel(
            QLabel("Multi-pass window deformation (OpenPIV) between consecutive "
                   "stabilized frames. Overlap is always half the window size."),
            self.computation_group,
            self.compute_btn,
            self.style_panel,
            self.render_btn,
            self._build_export_row(),
            self.export_computations_btn,
            footer=self.runner,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(action_splitter(self.viewer, side))

        self._update_window_size_placeholder()
        self.state.changed.connect(self.refresh)
        self.state.project_loaded.connect(self._restore_from_meta)
        self.refresh()
        self._restore_from_meta()

    def _restore_from_meta(self) -> None:
        """Shows what actually produced the currently cached PIV result (if
        any) rather than leaving the form at whatever it last had — see
        AppState.project_loaded."""
        meta = self.state.piv_meta
        if meta and meta.get("computed"):
            use_gapped = bool(meta.get("use_gapped_settings", False))
            self.piv_preset.setCurrentIndex(self.piv_preset.findData(use_gapped))
            window_sizes = meta.get("window_sizes")
            self.window_sizes_edit.setText(", ".join(str(w) for w in window_sizes) if window_sizes else "")
        else:
            self.piv_preset.setCurrentIndex(0)
            self.window_sizes_edit.clear()

    def _preset_default_windows(self) -> tuple[int, ...]:
        use_gapped = bool(self.piv_preset.currentData())
        return _piv.GAPPED_WINDOW_SIZES if use_gapped else _piv.WINDOW_SIZES

    def _update_window_size_placeholder(self) -> None:
        self.window_sizes_edit.setPlaceholderText(", ".join(str(w) for w in self._preset_default_windows()))
        self._update_overlap_preview()

    def _parse_window_sizes(self) -> tuple[int, ...] | None:
        """None means "use the preset's default". Raises ValueError on bad input."""
        text = self.window_sizes_edit.text().strip()
        if not text:
            return None
        sizes: list[int] = []
        prev: int | None = None
        for part in text.replace(",", " ").split():
            if not part.isdigit() or int(part) <= 0:
                raise ValueError(f"'{part}' is not a positive integer")
            value = int(part)
            if prev is not None and value > prev:
                raise ValueError("sizes must be listed coarse to fine (non-increasing)")
            sizes.append(value)
            prev = value
        return tuple(sizes) if sizes else None

    def _update_overlap_preview(self) -> None:
        try:
            sizes = self._parse_window_sizes() or self._preset_default_windows()
        except ValueError as exc:
            self.overlap_preview.setText(f"Invalid window sizes: {exc}")
            return
        overlaps = [max(w // 2, 1) for w in sizes]
        self.overlap_preview.setText(
            "Windows: " + ", ".join(str(w) for w in sizes)
            + "\nOverlaps (window/2): " + ", ".join(str(o) for o in overlaps)
        )

    def refresh(self) -> None:
        busy = self.runner.is_busy()
        has_stabilized = bool(self.state.stabilized_files)
        has_piv = self.state.piv_result is not None

        self.compute_btn.setEnabled(has_stabilized and not busy)
        self.compute_btn.setText("Re-run PIV Computation" if has_piv else "Run PIV Computation")
        self.render_btn.setEnabled(has_piv and not busy)
        self.export_computations_btn.setEnabled(has_piv and not busy and is_gapped_piv(self.state))
        # Re-applies the persisted collapsed/expanded state — needed on
        # project (re)load, since this tab is built once at startup before
        # any project is open (see properties_group in tab_common.py).
        self.computation_group.set_expanded(self.state.is_group_expanded("piv.computation"))
        self.style_panel.refresh()

        info = self._render_info()
        if info:
            frames = sorted(Path(info["frames_dir"]).glob("frame_*.png"))
            self.viewer.set_frames(frames)
        else:
            self.viewer.set_frames([])

    def _run_compute(self) -> None:
        if not self.state.stabilized_files:
            QMessageBox.warning(self, "No stabilized frames", "Run Stabilize first (Stabilize tab).")
            return
        try:
            window_sizes = self._parse_window_sizes()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid window sizes", str(exc))
            return
        files = list(self.state.stabilized_files)
        use_gapped = bool(self.piv_preset.currentData())

        def job(cb):
            return pipeline.run_piv(files, use_gapped_settings=use_gapped, window_sizes=window_sizes, progress=cb)

        def on_done(result: pipeline.PIVResult) -> None:
            self.state.set_piv_result(result, use_gapped, window_sizes)
            self.runner.log_message("PIV computation complete.")

        self.runner.run(job, on_done, "Computing PIV...", parent_for_errors=self)

    def _run_render(self) -> None:
        if self.state.piv_result is None:
            QMessageBox.warning(self, "No PIV result", "Run PIV computation first.")
            return
        piv_result = self.state.piv_result
        frame_files = list(self.state.stabilized_files)
        bg_files = list(self.state.stabilized_bg_files) if self.state.stabilized_bg_files else None
        out_dir = self.state.project_dir / "results" / "piv"
        style = self.state.render_settings.snapshot()

        def job(cb):
            return pipeline.render_piv_timeline(piv_result, frame_files, out_dir, bg_files, style, progress=cb)

        def on_done(result) -> None:
            frames_dir, video = result
            self.state.set_render_output("piv", frames_dir, video)
            self.runner.log_message("Rendering complete.")

        self.runner.run(job, on_done, "Rendering PIV timeline...", parent_for_errors=self)

    def _export_computations(self) -> None:
        if self.state.piv_result is None:
            QMessageBox.warning(self, "No PIV result", "Run PIV computation first.")
            return
        result = self.state.piv_result
        files = list(self.state.stabilized_files)
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export PIV computations as zip",
            str(Path.home() / "piv_displacement_csv.zip"), "Zip Archive (*.zip)",
        )
        if not dest:
            return
        px_to_um = self.state.render_settings.px_to_um()
        n_pairs = len(result.u_smooth)

        def write_csv(tmp_dir: Path, i: int) -> None:
            name = f"{files[i].stem}_to_{files[i + 1].stem}_displacement.csv"
            export_displacement_csv(
                tmp_dir / name, result.x, result.y, result.u_smooth[i], result.v_smooth[i],
                grid_mean=1, px_to_um=px_to_um,
            )

        export_csv_archive(self, dest, n_pairs, write_csv)

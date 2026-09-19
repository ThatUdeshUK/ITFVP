"""Traction stress tab: invert PIV displacement into FTTC traction stress
and render the heatmap timeline.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lib import traction as _traction
from .. import pipeline
from ..state import AppState
from ..widgets import ActionRunner, FrameViewer, RenderOptionsPanel
from .tab_common import ExportMixin, action_splitter, properties_group, side_panel


class TractionTab(QWidget, ExportMixin):
    key = "traction"
    title = "Traction"

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self.viewer = FrameViewer(empty_text="No traction render yet — run traction computation, then Render.")

        self.youngs_modulus = QDoubleSpinBox()
        self.youngs_modulus.setRange(1.0, 1_000_000.0)
        self.youngs_modulus.setValue(_traction.YOUNGS_MODULUS)
        self.youngs_modulus.setSuffix(" Pa")
        self.youngs_modulus.setDecimals(0)
        self.poisson_ratio = QDoubleSpinBox()
        self.poisson_ratio.setRange(0.0, 0.5)
        self.poisson_ratio.setSingleStep(0.05)
        self.poisson_ratio.setValue(_traction.POISSON_RATIO)

        self.compute_btn = QPushButton("Run Traction")
        self.compute_btn.clicked.connect(self._run_compute)

        self.computation_group = properties_group(
            state, "traction.computation", "Computation properties",
            QLabel("Young's modulus:"),
            self.youngs_modulus,
            QLabel("Poisson's ratio:"),
            self.poisson_ratio,
        )

        self.style_panel = RenderOptionsPanel(
            state, "traction.render", show_arrows=False, show_unit=False,
            background_available=lambda: bool(self.state.stabilized_bg_files),
        )

        self.render_btn = QPushButton("Render")
        self.render_btn.clicked.connect(self._run_render)

        self.runner = ActionRunner()
        self.runner.manage([self.compute_btn, self.render_btn])

        side = side_panel(
            QLabel("Inverts the PIV displacement field into traction stress "
                   "(FTTC, Butler et al. 2002)."),
            self.computation_group,
            self.compute_btn,
            self.style_panel,
            self.render_btn,
            self._build_export_row(),
            footer=self.runner,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(action_splitter(self.viewer, side))

        self.state.changed.connect(self.refresh)
        self.state.project_loaded.connect(self._restore_from_meta)
        self.refresh()
        self._restore_from_meta()

    def _restore_from_meta(self) -> None:
        """Shows what actually produced the currently cached traction
        result (if any) — see AppState.project_loaded."""
        meta = self.state.traction_meta
        if meta and meta.get("computed"):
            self.youngs_modulus.setValue(meta.get("youngs_modulus", _traction.YOUNGS_MODULUS))
            self.poisson_ratio.setValue(meta.get("poisson_ratio", _traction.POISSON_RATIO))
        else:
            self.youngs_modulus.setValue(_traction.YOUNGS_MODULUS)
            self.poisson_ratio.setValue(_traction.POISSON_RATIO)

    def refresh(self) -> None:
        busy = self.runner.is_busy()
        has_piv = self.state.piv_result is not None
        has_traction = self.state.traction_result is not None

        self.compute_btn.setEnabled(has_piv and not busy)
        self.compute_btn.setText("Re-run Traction" if has_traction else "Run Traction")
        self.render_btn.setEnabled(has_traction and not busy)
        self.computation_group.set_expanded(self.state.is_group_expanded("traction.computation"))
        self.style_panel.refresh()

        info = self._render_info()
        if info:
            frames = sorted(Path(info["frames_dir"]).glob("frame_*.png"))
            self.viewer.set_frames(frames)
        else:
            self.viewer.set_frames([])

    def _run_compute(self) -> None:
        if self.state.piv_result is None:
            QMessageBox.warning(self, "No PIV result", "Run PIV computation first (PIV displacement tab).")
            return
        piv_result = self.state.piv_result
        e = self.youngs_modulus.value()
        nu = self.poisson_ratio.value()

        def job(cb):
            return pipeline.run_traction(piv_result, youngs_modulus=e, poisson_ratio=nu, progress=cb)

        def on_done(result: pipeline.TractionResult) -> None:
            self.state.set_traction_result(result, e, nu)
            self.runner.log_message("Traction computation complete.")

        self.runner.run(job, on_done, "Computing traction (FTTC)...", parent_for_errors=self)

    def _run_render(self) -> None:
        if self.state.traction_result is None:
            QMessageBox.warning(self, "No traction result", "Run traction computation first.")
            return
        traction_result = self.state.traction_result
        piv_result = self.state.piv_result
        frame_files = list(self.state.stabilized_files)
        bg_files = list(self.state.stabilized_bg_files) if self.state.stabilized_bg_files else None
        out_dir = self.state.project_dir / "results" / "traction"
        style = self.state.render_settings.snapshot(include_unit=False)

        def job(cb):
            return pipeline.render_traction_timeline(
                traction_result, piv_result, frame_files, out_dir, bg_files, style, progress=cb
            )

        def on_done(result) -> None:
            frames_dir, video = result
            self.state.set_render_output("traction", frames_dir, video)
            self.runner.log_message("Rendering complete.")

        self.runner.run(job, on_done, "Rendering traction timeline...", parent_for_errors=self)

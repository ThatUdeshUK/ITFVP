"""Combined overlay tab: render background + PIV arrows + traction heatmap
together — no computation of its own, just an overlay of the already
computed PIV and traction results.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from .. import pipeline
from ..state import AppState
from ..widgets import ActionRunner, FrameViewer, RenderOptionsPanel
from .tab_common import ExportMixin, action_splitter, side_panel


class CombinedTab(QWidget, ExportMixin):
    key = "combined"
    title = "Combined"

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self.viewer = FrameViewer(empty_text="No combined render yet — needs PIV and traction computed first.")

        self.style_panel = RenderOptionsPanel(
            state, "combined.render", show_arrows=True,
            background_available=lambda: bool(self.state.stabilized_bg_files),
        )

        self.render_btn = QPushButton("Render")
        self.render_btn.clicked.connect(self._run_render)

        self.runner = ActionRunner()
        self.runner.manage([self.render_btn])

        side = side_panel(
            QLabel("Overlays the background (per the render properties below), PIV "
                   "arrows, and the traction heatmap — requires PIV and traction to "
                   "already be computed (their own tabs)."),
            self.style_panel,
            self.render_btn,
            self._build_export_row(),
            footer=self.runner,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(action_splitter(self.viewer, side))

        self.state.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        busy = self.runner.is_busy()
        ready = self.state.piv_result is not None and self.state.traction_result is not None
        self.render_btn.setEnabled(ready and not busy)
        self.style_panel.refresh()

        info = self._render_info()
        if info:
            frames = sorted(Path(info["frames_dir"]).glob("frame_*.png"))
            self.viewer.set_frames(frames)
        else:
            self.viewer.set_frames([])

    def _run_render(self) -> None:
        if self.state.piv_result is None or self.state.traction_result is None:
            QMessageBox.warning(self, "Nothing to render", "Run PIV and traction computation first.")
            return
        piv_result = self.state.piv_result
        traction_result = self.state.traction_result
        frame_files = list(self.state.stabilized_files)
        bg_files = list(self.state.stabilized_bg_files) if self.state.stabilized_bg_files else None
        out_dir = self.state.project_dir / "results" / "combined"
        style = self.state.render_settings.snapshot()

        def job(cb):
            return pipeline.render_combined_timeline(
                piv_result, traction_result, frame_files, bg_files, out_dir, style, progress=cb
            )

        def on_done(result) -> None:
            frames_dir, video = result
            self.state.set_render_output("combined", frames_dir, video)
            self.runner.log_message("Rendering complete.")

        self.runner.run(job, on_done, "Rendering combined timeline...", parent_for_errors=self)

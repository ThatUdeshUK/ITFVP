"""Stabilize tab: run stage-drift correction and preview the result.

Empty state shows a big centered 'Run Stabilize' action; once stabilized
frames exist, the viewer takes over and the side panel's 'Re-run
Stabilize' button covers iterating afterward.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import pipeline
from ..state import AppState
from ..utils import imaging
from ..widgets import ActionRunner, FrameViewer
from .tab_common import SOURCE_FRAME_PREVIEW_SIZE, action_splitter, side_panel


class StabilizedTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self.viewer = FrameViewer(
            loader=lambda p: imaging.load_preview_pixmap(p, SOURCE_FRAME_PREVIEW_SIZE),
        )
        self.empty_widget = self._build_empty_state()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty_widget)
        self.stack.addWidget(self.viewer)

        self.skip_checkbox = QCheckBox("Skip (frames are already registered / no drift)")
        self.rerun_btn = QPushButton("Run Stabilize")
        self.rerun_btn.clicked.connect(self._run)

        self.runner = ActionRunner()
        self.runner.manage([self.run_btn, self.rerun_btn])

        side = side_panel(
            QLabel("Estimates cumulative stage drift from the main channel and "
                   "co-registers both channels."),
            self.skip_checkbox,
            self.rerun_btn,
            footer=self.runner,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(action_splitter(self.stack, side))

        self.state.changed.connect(self.refresh)
        self.state.project_loaded.connect(self._restore_from_meta)
        self.refresh()
        self._restore_from_meta()

    def _restore_from_meta(self) -> None:
        skipped = bool(self.state.stabilize_meta and self.state.stabilize_meta.get("skipped"))
        self.skip_checkbox.setChecked(skipped)

    def _build_empty_state(self) -> QWidget:
        msg = QLabel("No stabilized frames yet.")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.run_btn = QPushButton("Run Stabilize")
        self.run_btn.setMinimumSize(220, 44)
        self.run_btn.clicked.connect(self._run)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.run_btn)
        button_row.addStretch(1)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(msg)
        layout.addLayout(button_row)
        layout.addStretch(1)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def refresh(self) -> None:
        busy = self.runner.is_busy()
        has_micro = bool(self.state.micro_files)
        has_stabilized = bool(self.state.stabilized_files)

        self.run_btn.setEnabled(has_micro and not busy)
        self.rerun_btn.setEnabled(has_micro and not busy)
        self.rerun_btn.setText("Re-run Stabilize" if has_stabilized else "Run Stabilize")

        if has_stabilized:
            self.viewer.set_frames(list(self.state.stabilized_files), fps=4)
            self.stack.setCurrentWidget(self.viewer)
        else:
            self.stack.setCurrentWidget(self.empty_widget)

    def _run(self) -> None:
        if not self.state.micro_files:
            QMessageBox.warning(self, "No images", "Import main-channel images first.")
            return

        if self.skip_checkbox.isChecked():
            self.state.set_stabilized(
                list(self.state.micro_files),
                list(self.state.bg_files) if self.state.bg_files else None,
                skipped=True,
            )
            self.runner.log_message("Skipped stabilization — using imported frames as-is.")
            return

        micro_files = list(self.state.micro_files)
        bg_files = list(self.state.bg_files) if self.state.bg_files else None
        out_dir = self.state.project_dir / "stabilized"
        bg_out_dir = self.state.project_dir / "stabilized_bg" if bg_files else None

        def job(cb):
            return pipeline.run_stabilize(micro_files, out_dir, bg_files, bg_out_dir, progress=cb)

        def on_done(result: pipeline.StabilizeResult) -> None:
            self.state.set_stabilized(result.files, result.background_files)
            self.runner.log_message("Stabilization complete.")

        self.runner.run(job, on_done, "Stabilizing...", parent_for_errors=self)

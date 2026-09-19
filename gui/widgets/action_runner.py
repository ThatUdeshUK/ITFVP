"""Inline progress bar + status + log, with a run() method that drives a
background PipelineWorker. Used by the Results page's per-tab action panels
(Stabilize/PIV/Traction/Combined) — long computations get live progress
without a modal dialog, so the user can keep switching tabs while it runs.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..utils import PipelineWorker


class ActionRunner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: PipelineWorker | None = None
        self._managed_buttons: list[QPushButton] = []

        self.status_label = QLabel("Idle")
        self.progress = QProgressBar()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setFixedHeight(110)

        # Progress/log on top, status last — so "Idle"/"Failed"/etc. sits at
        # the bottom of this block, which is itself pinned to the bottom of
        # the side panel (see gui/pages/tab_common.py's side_panel).
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        layout.addWidget(self.status_label)

    def manage(self, buttons: list[QPushButton]) -> None:
        """Buttons to disable while a job is running — set once at setup."""
        self._managed_buttons = buttons

    def is_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def log_message(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def run(
        self, job: Callable, on_done: Callable[[object], None], label: str,
        parent_for_errors: QWidget | None = None,
    ) -> None:
        if self.is_busy():
            return
        self.status_label.setText(label)
        self.progress.setRange(0, 0)
        for b in self._managed_buttons:
            b.setEnabled(False)

        worker = PipelineWorker(job)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(lambda result: self._finished(on_done, result))
        worker.failed.connect(lambda msg: self._failed(msg, parent_for_errors))
        self._worker = worker
        worker.start()

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        if msg:
            self.log_message(msg)

    def _finished(self, on_done: Callable[[object], None], result: object) -> None:
        self._worker = None
        self.status_label.setText("Idle")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        for b in self._managed_buttons:
            b.setEnabled(True)
        on_done(result)

    def _failed(self, message: str, parent_for_errors: QWidget | None) -> None:
        self._worker = None
        self.status_label.setText("Failed")
        self.log_message(f"ERROR: {message}")
        for b in self._managed_buttons:
            b.setEnabled(True)
        QMessageBox.critical(parent_for_errors or self, "Pipeline error", message)

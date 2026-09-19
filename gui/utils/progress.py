"""Modal progress dialog for a background job, so file-copy/rename/export
operations don't block (or appear to freeze) the UI thread."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QWidget

from .workers import PipelineWorker


def run_with_progress(
    parent: QWidget,
    job: Callable[[Callable[[int, int, str], None]], object],
    title: str,
    label_text: str = "Working...",
):
    """Run job(progress_cb) on a background thread behind a modal progress
    dialog, blocking the caller (but not the UI, whose event loop keeps
    pumping) until it finishes. Returns job's result, or raises
    RuntimeError with job's error message if it failed."""
    dialog = QProgressDialog(label_text, None, 0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(300)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)

    outcome: dict = {}
    worker = PipelineWorker(job)

    def on_progress(done: int, total: int, msg: str) -> None:
        if total > 0:
            dialog.setRange(0, total)
            dialog.setValue(done)
        if msg:
            dialog.setLabelText(msg)

    def on_finished(result) -> None:
        outcome["result"] = result
        dialog.close()

    def on_failed(message: str) -> None:
        outcome["error"] = message
        dialog.close()

    worker.progress.connect(on_progress)
    worker.finished_ok.connect(on_finished)
    worker.failed.connect(on_failed)
    worker.start()
    dialog.exec()
    worker.wait()

    if "error" in outcome:
        raise RuntimeError(outcome["error"])
    return outcome.get("result")

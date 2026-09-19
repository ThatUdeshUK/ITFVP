"""Background-thread runner so pipeline steps don't block the Qt event loop."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    # (done, total, message)
    progress = Signal(int, int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[Callable[[int, int, str], None]], object], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn(lambda done, total, msg: self.progress.emit(done, total, msg))
        except Exception as exc:  # surfaced to the UI rather than crashing the app
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished_ok.emit(result)

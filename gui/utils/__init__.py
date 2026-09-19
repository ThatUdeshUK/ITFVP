"""Small standalone helpers with no Qt-widget identity of their own:
background-thread execution (workers.py/PipelineWorker) and a modal
progress dialog built on it (progress.py/run_with_progress), re-exported
below. imaging.py (TIFF -> QPixmap/QImage preview rendering) is used as a
module — `from ..utils import imaging` — since call sites reference several
of its functions.
"""

from .progress import run_with_progress
from .workers import PipelineWorker

__all__ = ["run_with_progress", "PipelineWorker"]

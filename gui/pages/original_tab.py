"""Original tab: image management, one sub-tab per channel (see
ImagesPage), each pairing a thumbnail grid with a synced playback
timeline — see ImageGridPanel.
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..state import AppState
from .images_page import ImagesPage


class OriginalTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.images_page = ImagesPage(state)
        layout = QVBoxLayout(self)
        layout.addWidget(self.images_page)

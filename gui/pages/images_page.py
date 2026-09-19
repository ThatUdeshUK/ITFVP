"""Page 2: view imported images as thumbnails and rename them."""

from __future__ import annotations

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from ..state import AppState
from ..widgets import ImageGridPanel


class ImagesPage(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        tabs = QTabWidget()
        tabs.addTab(ImageGridPanel(state, "micro"), "Main channel")
        tabs.addTab(ImageGridPanel(state, "bg"), "Background channel")

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)

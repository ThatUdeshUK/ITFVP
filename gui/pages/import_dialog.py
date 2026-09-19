"""Import, opened as its own screen from the workspace toolbar."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from ..state import AppState
from .import_page import ImportPage


class ImportDialog(QDialog):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Images")
        self.resize(900, 640)

        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(done_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(ImportPage(state))
        layout.addLayout(btn_row)

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .pages.combined_tab import CombinedTab
from .pages.import_dialog import ImportDialog
from .pages.original_tab import OriginalTab
from .pages.piv_tab import PIVTab
from .pages.project_select_page import ProjectSelectPage
from .pages.stabilized_tab import StabilizedTab
from .pages.traction_tab import TractionTab
from .state import AppState


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ITFVP — Image to Traction Force Visualization Pipeline")
        self.resize(1280, 860)

        self.state = AppState()

        self.select_page = ProjectSelectPage()
        self.select_page.project_opened.connect(self._open_project)

        self.workspace = self._build_workspace()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.select_page)
        self.stack.addWidget(self.workspace)
        self.setCentralWidget(self.stack)
        self.stack.setCurrentWidget(self.select_page)

    def _build_workspace(self) -> QWidget:
        self.project_label = QLabel()
        self.project_label.setStyleSheet("font-weight: bold;")

        import_btn = QPushButton("Import Images...")
        import_btn.clicked.connect(self._open_import_dialog)
        switch_btn = QPushButton("Switch Project...")
        switch_btn.clicked.connect(self._switch_project)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Project:"))
        top_row.addWidget(self.project_label, 1)
        top_row.addWidget(import_btn)
        top_row.addWidget(switch_btn)

        self.tabs = QTabWidget()
        self.tabs.addTab(OriginalTab(self.state), "Images")
        self.tabs.addTab(StabilizedTab(self.state), "Stabilize")
        self.tabs.addTab(PIVTab(self.state), "PIV displacement")
        self.tabs.addTab(TractionTab(self.state), "Traction stress")
        self.tabs.addTab(CombinedTab(self.state), "Combined overlay")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top_row)
        layout.addWidget(self.tabs)
        return container

    def _open_project(self, path: Path) -> None:
        self.state.open_project(path)
        self.project_label.setText(str(path))
        self.setWindowTitle(f"ITFVP — {path.name}")
        self.tabs.setCurrentIndex(0)  # Images
        self.stack.setCurrentWidget(self.workspace)

    def _switch_project(self) -> None:
        self.select_page.refresh_recent()
        self.stack.setCurrentWidget(self.select_page)

    def _open_import_dialog(self) -> None:
        dialog = ImportDialog(self.state, self)
        dialog.exec()

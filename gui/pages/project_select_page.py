"""Landing screen: choose or create a project folder before anything else.

An existing project (one with itfvp_project.json already in it) reopens
with its imported images and prior results; a brand-new folder starts
empty. Either way the caller lands on the Images tab — importing more
images is a separate action from there, not a required first step here.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..state import is_existing_project, recent_projects

# Fixed width of the centered content block — kept constant regardless of
# window size, rather than padding that scales with it.
CENTER_WIDTH = 480


class _CreateProjectDialog(QDialog):
    """Gathers a location + name and combines them into a new project path
    — the actual directory/JSON creation happens the same way an opened
    "Browse..." path does, in AppState.open_project (see MainWindow)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.project_path: Path | None = None

        self.location_edit = QLineEdit(str(Path.home()))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_location)
        location_row = QHBoxLayout()
        location_row.addWidget(self.location_edit, 1)
        location_row.addWidget(browse_btn)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("my-project")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Location:"))
        layout.addLayout(location_row)
        layout.addWidget(QLabel("Project name:"))
        layout.addWidget(self.name_edit)
        layout.addWidget(buttons)
        self.setMinimumWidth(420)

    def _browse_location(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a location", self.location_edit.text())
        if chosen:
            self.location_edit.setText(chosen)

    def _on_accept(self) -> None:
        location = self.location_edit.text().strip()
        name = self.name_edit.text().strip()
        if not location:
            QMessageBox.warning(self, "Missing location", "Choose a location for the project.")
            return
        if not name:
            QMessageBox.warning(self, "Missing name", "Enter a name for the project.")
            return
        if any(sep in name for sep in ("/", "\\")):
            QMessageBox.warning(self, "Invalid name", "Project name can't contain a path separator.")
            return
        candidate = Path(location) / name
        if candidate.exists():
            QMessageBox.warning(
                self, "Already exists",
                f"{candidate} already exists — choose a different name or location.",
            )
            return
        self.project_path = candidate
        self.accept()


class ProjectSelectPage(QWidget):
    project_opened = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)

        title = QLabel("ITFVP")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        subtitle = QLabel(
            "Select or create a project folder to begin. A project holds your "
            "imported time-series images and everything the pipeline produces "
            "from them (stabilized frames, PIV/traction fields, rendered "
            "timelines)."
        )
        subtitle.setWordWrap(True)

        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._open_recent)

        create_btn = QPushButton("Create New Project...")
        create_btn.clicked.connect(self._create_project)
        browse_btn = QPushButton("Open Existing Folder...")
        browse_btn.clicked.connect(self._browse)
        new_project_row = QHBoxLayout()
        new_project_row.addWidget(create_btn)
        new_project_row.addWidget(browse_btn)

        open_btn = QPushButton("Open Selected")
        open_btn.clicked.connect(self._open_recent_selected)

        recent_buttons = QHBoxLayout()
        recent_buttons.addWidget(open_btn)
        recent_buttons.addStretch(1)

        center = QVBoxLayout()
        center.addWidget(title)
        center.addWidget(subtitle)
        center.addSpacing(12)
        center.addWidget(QLabel("Recent projects (double-click to open):"))
        center.addWidget(self.recent_list)
        center.addLayout(recent_buttons)
        center.addSpacing(12)
        center.addLayout(new_project_row)

        center_widget = QWidget()
        center_widget.setLayout(center)
        center_widget.setFixedWidth(CENTER_WIDTH)

        center_row = QHBoxLayout()
        center_row.addStretch(1)
        center_row.addWidget(center_widget)
        center_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addLayout(center_row)
        layout.addStretch(2)

        self.refresh_recent()

    def refresh_recent(self) -> None:
        self.recent_list.clear()
        for path in recent_projects.get_recent_projects():
            label = f"{path.name}  —  {path}"
            if is_existing_project(path):
                label = f"{label}  [project]"
            self.recent_list.addItem(label)
        self._recent_paths = recent_projects.get_recent_projects()

    def _create_project(self) -> None:
        dialog = _CreateProjectDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.project_path is not None:
            self._open(dialog.project_path)

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose an existing project folder", str(Path.home()))
        if chosen:
            self._open(Path(chosen))

    def _open_recent(self, item) -> None:
        row = self.recent_list.row(item)
        self._open(self._recent_paths[row])

    def _open_recent_selected(self) -> None:
        row = self.recent_list.currentRow()
        if row >= 0:
            self._open(self._recent_paths[row])

    def _open(self, path: Path) -> None:
        recent_projects.add_recent_project(path)
        self.refresh_recent()
        self.project_opened.emit(path)

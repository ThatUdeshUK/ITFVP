"""Import time-series TIFFs into the already-chosen project directory."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..state import AppState, natural_key
from ..utils import run_with_progress

TIFF_FILTER = "TIFF images (*.tif *.tiff)"


class _ChannelImportBox(QGroupBox):
    def __init__(self, state: AppState, channel: str, title: str, parent=None):
        super().__init__(title, parent)
        self.state = state
        self.channel = channel

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        add_files_btn = QPushButton("Add Files...")
        add_files_btn.clicked.connect(self._add_files)
        add_folder_btn = QPushButton("Add Folder...")
        add_folder_btn.clicked.connect(self._add_folder)
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.list_widget.selectAll)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        remove_all_btn = QPushButton("Remove All")
        remove_all_btn.clicked.connect(self._remove_all)

        buttons = QHBoxLayout()
        for b in (add_files_btn, add_folder_btn, select_all_btn, remove_btn, remove_all_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.list_widget)

        self.state.changed.connect(self.refresh)
        self.refresh()

    def _require_project_dir(self) -> bool:
        if self.state.project_dir is None:
            QMessageBox.warning(self, "No project directory", "Choose a project directory first.")
            return False
        return True

    def _add_files(self) -> None:
        if not self._require_project_dir():
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Select time-series TIFF images", str(Path.home()), TIFF_FILTER)
        if files:
            paths = sorted((Path(f) for f in files), key=natural_key)
            self._import(paths)

    def _add_folder(self) -> None:
        if not self._require_project_dir():
            return
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of TIFF images", str(Path.home()))
        if not folder:
            return
        paths = sorted(
            [*Path(folder).glob("*.tif"), *Path(folder).glob("*.tiff")],
            key=natural_key,
        )
        if not paths:
            QMessageBox.information(self, "No images found", f"No .tif/.tiff files found in {folder}")
            return
        self._import(paths)

    def _import(self, paths: list[Path]) -> None:
        job = self.state.import_files_job(self.channel, paths)
        try:
            dest_paths = run_with_progress(self, job, "Importing Images", f"Copying {len(paths)} file(s)...")
        except RuntimeError as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self.state.apply_imported_files(self.channel, dest_paths)

    def _remove_selected(self) -> None:
        rows = sorted((self.list_widget.row(i) for i in self.list_widget.selectedItems()), reverse=True)
        for row in rows:
            self.state.remove_file(self.channel, row)

    def _remove_all(self) -> None:
        if not self.state.files_for(self.channel):
            return
        self.state.clear_channel(self.channel)

    def refresh(self) -> None:
        self.list_widget.clear()
        for f in self.state.files_for(self.channel):
            self.list_widget.addItem(f.name)


class ImportPage(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        note = QLabel(
            "Imported files are copied into the project directory — your "
            "original data is never modified or moved."
        )
        note.setWordWrap(True)

        micro_box = _ChannelImportBox(state, "micro", "Main channel (required) — the time series PIV/traction are computed from")
        bg_box = _ChannelImportBox(state, "bg", "Background channel (optional) — a co-registered channel used only as a render backdrop")

        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(micro_box)
        layout.addWidget(bg_box)

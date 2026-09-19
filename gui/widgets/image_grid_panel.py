"""Thumbnail grid + rename/reorder/remove controls, paired with a
FrameViewer for the same channel — used by gui/pages/images_page.py.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..state import AppState
from ..utils import imaging, run_with_progress
from .frame_viewer import FrameViewer

THUMB_SIZE = 96
PREVIEW_SIZE = 420


class ImageGridPanel(QWidget):
    """Thumbnail grid + rename/reorder/remove controls, paired with a
    playback timeline for the same channel ("micro" or "bg") — selecting a
    thumbnail jumps the timeline to it, and scrubbing/playing the timeline
    selects the matching thumbnail."""

    def __init__(self, state: AppState, channel: str, parent=None):
        super().__init__(parent)
        self.state = state
        self.channel = channel
        self._syncing = False

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSpacing(8)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.list_widget.itemSelectionChanged.connect(self._on_selection)
        self.list_widget.itemDoubleClicked.connect(self._rename_one)

        self.viewer = FrameViewer(
            loader=lambda p: imaging.load_preview_pixmap(p, PREVIEW_SIZE),
            empty_text="No images imported for this channel yet.",
            min_size=(PREVIEW_SIZE, PREVIEW_SIZE),
        )
        self.viewer.slider.valueChanged.connect(self._on_viewer_frame_changed)

        move_up_btn = QPushButton("Move Up")
        move_up_btn.clicked.connect(lambda: self._move_selected(-1))
        move_down_btn = QPushButton("Move Down")
        move_down_btn.clicked.connect(lambda: self._move_selected(1))
        rename_btn = QPushButton("Batch Rename...")
        rename_btn.clicked.connect(self._batch_rename)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)

        buttons = QHBoxLayout()
        for b in (move_up_btn, move_down_btn, rename_btn, remove_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)

        hint_label = QLabel("Double-click an image to rename it. Selecting an "
                             "image jumps the timeline to it, and vice versa.")
        hint_label.setWordWrap(True)

        left = QVBoxLayout()
        left.addWidget(hint_label)
        left.addWidget(self.list_widget, 1)
        left.addLayout(buttons)
        left_widget = QWidget()
        left_widget.setLayout(left)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self.state.changed.connect(self.refresh)
        self.refresh()

    def _files(self) -> list[Path]:
        return self.state.files_for(self.channel)

    def refresh(self) -> None:
        files = self._files()
        if not files:
            self.list_widget.clear()
            self.viewer.set_frames([])
            return

        def job(progress):
            n = len(files)
            images = []
            for i, f in enumerate(files):
                try:
                    images.append(imaging.load_preview_qimage(f, THUMB_SIZE))
                except Exception:
                    images.append(None)
                progress(i + 1, n, f"Loading thumbnail {i + 1}/{n}")
            return images

        try:
            qimages = run_with_progress(self, job, "Loading Images", f"Loading {len(files)} thumbnail(s)...")
        except RuntimeError:
            qimages = [None] * len(files)

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, (f, qimg) in enumerate(zip(files, qimages)):
            item = QListWidgetItem(f.name)
            if qimg is not None:
                item.setIcon(QIcon(QPixmap.fromImage(qimg)))
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

        self.viewer.set_frames(list(files), fps=4)

    def _on_selection(self) -> None:
        if self._syncing:
            return
        items = self.list_widget.selectedItems()
        if not items:
            return
        idx = items[-1].data(Qt.ItemDataRole.UserRole)
        if idx >= len(self._files()):
            return
        self._syncing = True
        try:
            self.viewer.set_current_index(idx)
        finally:
            self._syncing = False

    def _on_viewer_frame_changed(self, idx: int) -> None:
        if self._syncing or not (0 <= idx < self.list_widget.count()):
            return
        self._syncing = True
        try:
            self.list_widget.setCurrentRow(idx)
        finally:
            self._syncing = False

    def _selected_indices(self) -> list[int]:
        return sorted(item.data(Qt.ItemDataRole.UserRole) for item in self.list_widget.selectedItems())

    def _rename_one(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.ItemDataRole.UserRole)
        f = self._files()[idx]
        new_stem, ok = QInputDialog.getText(self, "Rename image", "New name (without extension):", text=f.stem)
        if ok and new_stem.strip():
            try:
                self.state.rename_file(self.channel, idx, new_stem.strip())
            except Exception as exc:
                QMessageBox.critical(self, "Rename failed", str(exc))

    def _batch_rename(self) -> None:
        files = self._files()
        if not files:
            return
        pattern, ok = QInputDialog.getText(
            self, "Batch rename all images",
            "Name pattern — use {index} for a 1-based frame number, "
            "e.g. Frame_{index:04d}:",
            text="Frame_{index:04d}",
        )
        if not (ok and pattern.strip()):
            return
        job = self.state.batch_rename_job(self.channel, pattern.strip())
        try:
            new_files = run_with_progress(self, job, "Renaming Images", f"Renaming {len(files)} file(s)...")
        except RuntimeError as exc:
            QMessageBox.critical(self, "Rename failed", str(exc))
            return
        self.state.apply_batch_rename(self.channel, new_files)

    def _remove_selected(self) -> None:
        for idx in reversed(self._selected_indices()):
            self.state.remove_file(self.channel, idx)

    def _move_selected(self, delta: int) -> None:
        indices = self._selected_indices()
        if len(indices) != 1:
            return
        self.state.move_file(self.channel, indices[0], delta)

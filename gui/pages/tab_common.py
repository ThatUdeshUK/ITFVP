"""Shared building blocks for the pipeline-stage tabs (original_tab.py,
stabilized_tab.py, piv_tab.py, traction_tab.py, combined_tab.py): the
fixed-width scrollable side panel, the collapsible "properties" group, the
viewer/side-panel splitter, and the Export Video/Frames mixin.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..state import AppState
from ..utils import run_with_progress
from ..widgets import CollapsibleGroupBox

# Raw/stabilized TIFFs are shown at a higher resolution than the small
# thumbnail grid inside ImageGridPanel, since these viewers fill most of
# their tab.
SOURCE_FRAME_PREVIEW_SIZE = 1600

# Content width the panel is built for — wide enough that the longest
# control (e.g. the PIV preset combo's "Gapped / large displacement
# (coarse-to-fine)" item) doesn't get squeezed and elided. The scroll area
# adds a little extra for its vertical scrollbar so content never has to
# share that space.
SIDE_PANEL_WIDTH = 400
SIDE_PANEL_SCROLL_WIDTH = SIDE_PANEL_WIDTH + 16


def _add_item(layout: QVBoxLayout, item: QWidget | QLayout) -> None:
    if isinstance(item, QWidget):
        if isinstance(item, QLabel):
            item.setWordWrap(True)
        layout.addWidget(item)
    else:
        layout.addLayout(item)


def side_panel(*items: QWidget | QLayout, footer: QWidget | None = None) -> QWidget:
    """A fixed-width panel: items scroll in the region above, while footer
    (e.g. ActionRunner's status/progress/log) stays static and pinned to
    the bottom, outside the scrollable area, always visible."""
    layout = QVBoxLayout()
    for item in items:
        _add_item(layout, item)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    content = QWidget()
    content.setLayout(layout)
    content.setFixedWidth(SIDE_PANEL_WIDTH)

    scroll = QScrollArea()
    scroll.setWidget(content)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(4)
    outer.addWidget(scroll, 1)
    if footer is not None:
        outer.addWidget(footer)
    container.setFixedWidth(SIDE_PANEL_SCROLL_WIDTH)
    return container


def properties_group(state: AppState, key: str, title: str, *items: QWidget | QLayout) -> CollapsibleGroupBox:
    """key identifies this group's collapsed/expanded state in the
    project's JSON (see AppState.is_group_expanded/set_group_expanded) —
    groups start collapsed unless a prior session left them expanded."""
    box = CollapsibleGroupBox(title, start_expanded=state.is_group_expanded(key))
    box.toggled.connect(lambda expanded: state.set_group_expanded(key, expanded))
    for item in items:
        if isinstance(item, QLabel):
            item.setWordWrap(True)
    box.add(*items)
    return box


def action_splitter(main: QWidget, side: QWidget) -> QSplitter:
    """side is expected to be a fixed-width panel (see side_panel) — all
    stretch on window resize goes to main, and the divider can't collapse
    or resize the fixed side away."""
    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.addWidget(main)
    splitter.addWidget(side)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 0)
    splitter.setCollapsible(1, True)
    return splitter


class ExportMixin:
    """Mixed into a rendered-timeline tab (self.state, self.key, self.title
    must be set) to add Export Video/Frames buttons wired to
    state.render_outputs[self.key]."""

    def _render_info(self) -> dict | None:
        return self.state.render_outputs.get(self.key)

    def _build_export_row(self) -> QVBoxLayout:
        # Stacked rather than side-by-side — the fixed-width side panel
        # (see side_panel) isn't wide enough for both buttons on one line
        # without clipping their text.
        export_video_btn = QPushButton("Export Video (.mp4)...")
        export_video_btn.clicked.connect(self._export_video)
        export_frames_btn = QPushButton("Export Frames (.zip)...")
        export_frames_btn.clicked.connect(self._export_frames)
        row = QVBoxLayout()
        row.addWidget(export_video_btn)
        row.addWidget(export_frames_btn)
        return row

    def _export_video(self) -> None:
        info = self._render_info()
        if not info or not Path(info["video"]).exists():
            QMessageBox.information(self, "Nothing to export", f"Render the {self.title} timeline first.")
            return
        video_path = Path(info["video"])
        dest, _ = QFileDialog.getSaveFileName(self, "Export video", str(Path.home() / video_path.name), "MP4 Video (*.mp4)")
        if not dest:
            return

        def job(progress):
            progress(0, 0, f"Copying {video_path.name}...")
            shutil.copy2(video_path, dest)

        try:
            run_with_progress(self, job, "Exporting Video", f"Copying {video_path.name}...")
        except RuntimeError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _export_frames(self) -> None:
        info = self._render_info()
        if not info or not Path(info["frames_dir"]).exists():
            QMessageBox.information(self, "Nothing to export", f"Render the {self.title} timeline first.")
            return
        default_name = f"{self.key}_frames.zip"
        dest, _ = QFileDialog.getSaveFileName(self, "Export frames as zip", str(Path.home() / default_name), "Zip Archive (*.zip)")
        if not dest:
            return
        if dest.lower().endswith(".zip"):
            dest = dest[:-4]
        frames_dir = info["frames_dir"]

        def job(progress):
            progress(0, 0, "Compressing frames...")
            shutil.make_archive(dest, "zip", frames_dir)

        try:
            run_with_progress(self, job, "Exporting Frames", "Compressing frames...")
        except RuntimeError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

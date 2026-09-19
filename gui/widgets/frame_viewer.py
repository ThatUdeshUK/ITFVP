"""A scrubbable frame viewer for timelines — used both standalone (raw/
stabilized/rendered timelines) and paired with a thumbnail grid (see
image_grid_panel.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class FrameViewer(QWidget):
    """Scrub through an image sequence with a slider and play/pause —
    avoids relying on a video-codec backend just to view a timeline.

    By default frames are loaded as-is via QPixmap (rendered PNG timelines);
    pass loader= to read some other format instead (e.g. raw TIFFs via
    gui.utils.imaging, which QPixmap can't decode correctly on its own)."""

    def __init__(
        self, parent=None,
        loader: Callable[[Path], QPixmap] | None = None,
        empty_text: str = "No rendered frames yet — run the pipeline first.",
        min_size: tuple[int, int] = (480, 360),
    ):
        super().__init__(parent)
        self._frame_paths: list[Path] = []
        self._playing = False
        self._loader = loader
        self._empty_text = empty_text

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self.image_label = QLabel(self._empty_text)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(*min_size)
        self.image_label.setStyleSheet("QLabel { border: 1px solid palette(mid); }")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._show_frame)

        self.frame_label = QLabel("0 / 0")
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        prev_btn = QPushButton("<")
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn = QPushButton(">")
        next_btn.clicked.connect(lambda: self._step(1))

        controls = QHBoxLayout()
        controls.addWidget(prev_btn)
        controls.addWidget(self.play_button)
        controls.addWidget(next_btn)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.frame_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, 1)
        layout.addLayout(controls)

    def set_frames(self, frame_paths: list[Path], fps: int = 8) -> None:
        self._timer.stop()
        self._playing = False
        self.play_button.setText("Play")
        self._frame_paths = frame_paths
        self._timer.setInterval(max(int(1000 / fps), 20))
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(len(frame_paths) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._show_frame(0)

    def set_current_index(self, idx: int) -> None:
        """Jump to a frame without re-entering set_frames — used to sync
        against an external selection (see ImageGridPanel)."""
        self.slider.setValue(idx)

    def _show_frame(self, idx: int) -> None:
        if not self._frame_paths:
            self.image_label.setText(self._empty_text)
            self.frame_label.setText("0 / 0")
            return
        idx = max(0, min(idx, len(self._frame_paths) - 1))
        path = self._frame_paths[idx]
        pixmap = self._loader(path) if self._loader is not None else QPixmap(str(path))
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.frame_label.setText(f"{idx + 1} / {len(self._frame_paths)}")

    def _step(self, delta: int) -> None:
        self.slider.setValue(self.slider.value() + delta)

    def _advance(self) -> None:
        nxt = self.slider.value() + 1
        if nxt > self.slider.maximum():
            nxt = 0
        self.slider.setValue(nxt)

    def _toggle_play(self) -> None:
        if self._playing:
            self._timer.stop()
            self.play_button.setText("Play")
        elif self._frame_paths:
            self._timer.start()
            self.play_button.setText("Pause")
        self._playing = not self._playing

"""A collapsible section with a clickable header bar (title + arrow) — used
for the side panel's "properties" sections (Computation properties, Render
properties) so a tab with several of them doesn't force a tall window.

The header-click-to-toggle interaction (rather than a small checkbox) is
adapted from EsoCoding/PySide6-Collapsible-Widget
(https://github.com/EsoCoding/PySide6-Collapsible-Widget, itself a PySide6
port of aronamao/PySide2-Collapsible-Widget). Rewritten against the app's Qt
palette (rather than the original's hardcoded dark-gray background with
black icon/label text, which has no contrast in a light theme) and against
this file's own body/body_layout/add() API so nothing else here has to
change.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)


class _CollapsibleHeader(QWidget):
    """Clickable title bar with an expand/collapse arrow indicator."""

    toggled = Signal(bool)  # emits the new expanded state

    _EXPAND_ICON = "▸"    # ▸
    _COLLAPSE_ICON = "▾"  # ▾

    def __init__(self, title: str, content: QWidget, parent=None):
        super().__init__(parent)
        self._content = content
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "_CollapsibleHeader { background-color: palette(mid); border-radius: 3px; }"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)

        self._icon = QLabel(self._COLLAPSE_ICON)
        self._icon.setStyleSheet("QLabel { font-weight: bold; background: transparent; }")
        row.addWidget(self._icon)

        font = QFont()
        font.setBold(True)
        label = QLabel(title)
        label.setFont(font)
        label.setStyleSheet("QLabel { background: transparent; }")
        row.addWidget(label)
        row.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.collapse() if self._content.isVisible() else self.expand()
        super().mousePressEvent(event)

    def expand(self) -> None:
        self._content.setVisible(True)
        self._icon.setText(self._COLLAPSE_ICON)
        self.toggled.emit(True)

    def collapse(self) -> None:
        self._content.setVisible(False)
        self._icon.setText(self._EXPAND_ICON)
        self.toggled.emit(False)


class CollapsibleGroupBox(QWidget):
    """A titled section: click the header to expand/collapse. Add content
    via add() (widgets/layouts) or by building directly into body_layout."""

    toggled = Signal(bool)  # emits the new expanded state

    def __init__(self, title: str, parent=None, start_expanded: bool = True):
        super().__init__(parent)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 8, 4, 4)

        self._header = _CollapsibleHeader(title, self.body)
        self._header.toggled.connect(self.toggled)
        (self._header.expand if start_expanded else self._header.collapse)()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(self._header)
        outer.addWidget(self.body)

    def set_expanded(self, expanded: bool) -> None:
        (self._header.expand if expanded else self._header.collapse)()

    def add(self, *items: QWidget | QLayout) -> None:
        for item in items:
            if isinstance(item, QWidget):
                self.body_layout.addWidget(item)
            else:
                self.body_layout.addLayout(item)

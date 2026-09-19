"""The 'Render properties' panel shown before each stage's Render action.

One reusable widget bound to the project's single shared RenderSettings
instance (see gui/state/app_state.py) — every panel instance edits and
listens to the same object, so background/unit/arrow choices stay
synchronized across the PIV, traction, and combined tabs without those
tabs knowing about each other directly.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from ..state import AppState
from .collapsible_group import CollapsibleGroupBox


class RenderOptionsPanel(CollapsibleGroupBox):
    def __init__(
        self,
        state: AppState,
        key: str,
        show_arrows: bool,
        show_unit: bool = True,
        background_available: Callable[[], bool] = lambda: False,
        parent=None,
    ):
        super().__init__("Render properties", parent, start_expanded=state.is_group_expanded(key))
        self.toggled.connect(lambda expanded: state.set_group_expanded(key, expanded))
        self._state = state
        self._key = key
        self.settings = state.render_settings
        self.show_arrows = show_arrows
        self.show_unit = show_unit
        self.background_available = background_available
        self._syncing = False

        layout = self.body_layout

        self.bg_combo = QComboBox()
        self.bg_combo.addItem("White", "white")
        self.bg_combo.addItem("Black", "black")
        self.bg_combo.addItem("Background channel", "channel")
        self.bg_combo.currentIndexChanged.connect(self._on_bg_changed)
        layout.addWidget(QLabel("Background:"))
        layout.addWidget(self.bg_combo)

        if show_unit:
            self.unit_combo = QComboBox()
            self.unit_combo.addItem("Pixels", "px")
            self.unit_combo.addItem("Micrometers", "um")
            self.unit_combo.currentIndexChanged.connect(self._on_unit_changed)
            self.scale_spin = QDoubleSpinBox()
            self.scale_spin.setRange(0.0001, 1000.0)
            self.scale_spin.setDecimals(4)
            self.scale_spin.setSingleStep(0.01)
            self.scale_spin.valueChanged.connect(self._on_scale_changed)
            unit_row = QHBoxLayout()
            unit_row.addWidget(self.unit_combo)
            unit_row.addWidget(QLabel("µm/px:"))
            unit_row.addWidget(self.scale_spin)
            layout.addWidget(QLabel("Axis unit:"))
            layout.addLayout(unit_row)

        if show_arrows:
            self.width_spin = QDoubleSpinBox()
            self.width_spin.setRange(0.0005, 0.05)
            self.width_spin.setDecimals(4)
            self.width_spin.setSingleStep(0.0005)
            self.width_spin.valueChanged.connect(self._on_width_changed)

            self.length_spin = QDoubleSpinBox()
            self.length_spin.setRange(0.1, 5.0)
            self.length_spin.setDecimals(2)
            self.length_spin.setSingleStep(0.1)
            self.length_spin.valueChanged.connect(self._on_length_changed)

            self.color_mode_combo = QComboBox()
            self.color_mode_combo.addItem("Rainbow (by displacement)", "rainbow")
            self.color_mode_combo.addItem("Solid color", "solid")
            self.color_mode_combo.currentIndexChanged.connect(self._on_color_mode_changed)

            self.color_btn = QPushButton("Pick Color...")
            self.color_btn.clicked.connect(self._pick_color)

            layout.addWidget(QLabel("Arrow width:"))
            layout.addWidget(self.width_spin)
            layout.addWidget(QLabel("Arrow length:"))
            layout.addWidget(self.length_spin)
            layout.addWidget(QLabel("Arrow color:"))
            layout.addWidget(self.color_mode_combo)
            layout.addWidget(self.color_btn)

        self.settings.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        # Re-applies whenever the project (re)loads, e.g. after opening a
        # project whose group_expanded state wasn't yet known when this
        # panel was constructed (tabs are built once at startup, before any
        # project is open).
        self.set_expanded(self._state.is_group_expanded(self._key))

        if self._syncing:
            return
        self._syncing = True
        try:
            available = self.background_available()
            channel_item = self.bg_combo.model().item(self.bg_combo.findData("channel"))
            channel_item.setEnabled(available)
            self.bg_combo.setCurrentIndex(self.bg_combo.findData(self.settings.background_mode))

            if self.show_unit:
                self.unit_combo.setCurrentIndex(self.unit_combo.findData(self.settings.unit_mode))
                self.scale_spin.setEnabled(self.settings.unit_mode == "um")
                if self.scale_spin.value() != self.settings.scale_um_per_px:
                    self.scale_spin.setValue(self.settings.scale_um_per_px)

            if self.show_arrows:
                if self.width_spin.value() != self.settings.arrow_width:
                    self.width_spin.setValue(self.settings.arrow_width)
                if self.length_spin.value() != self.settings.arrow_length:
                    self.length_spin.setValue(self.settings.arrow_length)
                self.color_mode_combo.setCurrentIndex(
                    self.color_mode_combo.findData(self.settings.arrow_color_mode)
                )
                self.color_btn.setEnabled(self.settings.arrow_color_mode == "solid")
                self.color_btn.setStyleSheet(f"background-color: {self.settings.arrow_color};")
        finally:
            self._syncing = False

    def _on_bg_changed(self) -> None:
        if self._syncing:
            return
        self.settings.background_mode = self.bg_combo.currentData()
        self.settings.changed.emit()

    def _on_unit_changed(self) -> None:
        if self._syncing:
            return
        self.settings.unit_mode = self.unit_combo.currentData()
        self.settings.changed.emit()

    def _on_scale_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.settings.scale_um_per_px = value
        self.settings.changed.emit()

    def _on_width_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.settings.arrow_width = value
        self.settings.changed.emit()

    def _on_length_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.settings.arrow_length = value
        self.settings.changed.emit()

    def _on_color_mode_changed(self) -> None:
        if self._syncing:
            return
        self.settings.arrow_color_mode = self.color_mode_combo.currentData()
        self.settings.changed.emit()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self.settings.arrow_color), self, "Pick arrow color")
        if color.isValid():
            self.settings.arrow_color = color.name()
            self.settings.changed.emit()

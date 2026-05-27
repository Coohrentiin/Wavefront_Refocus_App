"""Interpolation controls: smoothing factor + run button."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import config


class InterpolationPanel(QWidget):
    """Emits ``run_requested(smoothing)`` to interpolate unchosen frames."""

    run_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        box = QGroupBox("Interpolation tools")

        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 1e6)
        self.smooth_spin.setDecimals(4)
        self.smooth_spin.setValue(config.DEFAULT_SMOOTHING)
        self.smooth_spin.setToolTip(
            "Spline smoothing factor s (0 = interpolating spline)."
        )

        self.run_btn = QPushButton("Interpolate planes")
        self.run_btn.clicked.connect(
            lambda: self.run_requested.emit(self.smooth_spin.value())
        )

        row = QHBoxLayout()
        row.addWidget(self.smooth_spin)
        row.addWidget(self.run_btn)
        bl = QVBoxLayout(box)
        bl.addLayout(row)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(box)

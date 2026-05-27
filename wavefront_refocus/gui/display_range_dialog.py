"""Floating (non-modal) window to set the image display range."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class DisplayRangeDialog(QDialog):
    """Live controls for the colour-scale percentiles and border crop.

    Emits ``range_changed(pct_low, pct_high, margin)`` whenever a value
    changes so the view can recolour immediately. It is a detached, non-modal
    top-level window (own taskbar entry, not tied to the main window).
    """

    range_changed = Signal(float, float, int)

    def __init__(self, pct_low: float, pct_high: float, margin: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display range")
        # A standalone window rather than a Tool dialog tied to a parent.
        self.setWindowFlags(Qt.Window)
        self.setModal(False)

        self.low_spin = QDoubleSpinBox()
        self.low_spin.setRange(0.0, 100.0)
        self.low_spin.setDecimals(2)
        self.low_spin.setSuffix(" %")
        self.low_spin.setValue(pct_low)

        self.high_spin = QDoubleSpinBox()
        self.high_spin.setRange(0.0, 100.0)
        self.high_spin.setDecimals(2)
        self.high_spin.setSuffix(" %")
        self.high_spin.setValue(pct_high)

        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 512)
        self.margin_spin.setSuffix(" px")
        self.margin_spin.setValue(margin)
        self.margin_spin.setToolTip("Border pixels cropped before scaling.")

        reset = QPushButton("Reset (1–99 %, 16 px)")
        reset.clicked.connect(self._reset)

        form = QFormLayout()
        form.addRow("Low percentile", self.low_spin)
        form.addRow("High percentile", self.high_spin)
        form.addRow("Border crop", self.margin_spin)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(reset)

        for w in (self.low_spin, self.high_spin, self.margin_spin):
            w.valueChanged.connect(self._emit)

    def _emit(self, *_):
        low = self.low_spin.value()
        high = max(low, self.high_spin.value())
        if high != self.high_spin.value():
            self.high_spin.blockSignals(True)
            self.high_spin.setValue(high)
            self.high_spin.blockSignals(False)
        self.range_changed.emit(low, high, self.margin_spin.value())

    def _reset(self):
        self.low_spin.setValue(1.0)
        self.high_spin.setValue(99.0)
        self.margin_spin.setValue(16)
        self._emit()

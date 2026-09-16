"""Interpolation controls: method, smoothing factor, target frames, run."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..core.interpolation import METHOD_LINEAR, METHOD_SPLINE


class InterpolationPanel(QWidget):
    """Emits ``run_requested(method, smoothing, frames_text)``.

    ``frames_text`` is the raw contents of the "Apply to frames" field; the
    main window parses it (empty = every unchosen frame).
    """

    run_requested = Signal(str, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        box = QGroupBox("Interpolation tools")

        self.method_combo = QComboBox()
        self.method_combo.addItem("Smoothing spline", METHOD_SPLINE)
        self.method_combo.addItem("Linear", METHOD_LINEAR)
        self.method_combo.setToolTip(
            "Spline: smooth cubic fit through the chosen planes.\n"
            "Linear: straight segments between them — passes exactly through\n"
            "every chosen plane and never overshoots."
        )
        self.method_combo.currentIndexChanged.connect(self._sync_smoothing)

        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 1e6)
        self.smooth_spin.setDecimals(4)
        self.smooth_spin.setValue(config.DEFAULT_SMOOTHING)
        self.smooth_spin.setToolTip(
            "Spline smoothing factor s (0 = interpolating spline)."
        )

        self.frames_edit = QLineEdit()
        self.frames_edit.setPlaceholderText("all unchosen frames")
        self.frames_edit.setToolTip(
            "Restrict interpolation to these frames, e.g. \"1,38,39,100\" or\n"
            "\"0-9, 25, 40-45\". Leave empty to (re)interpolate every frame\n"
            "you have not chosen a plane for. Listed frames are recomputed\n"
            "even if they already have an interpolated position."
        )
        self.frames_edit.returnPressed.connect(self._emit_run)

        self.run_btn = QPushButton("Interpolate planes")
        self.run_btn.clicked.connect(self._emit_run)

        form = QFormLayout()
        form.addRow("Method", self.method_combo)
        form.addRow("Smoothing", self.smooth_spin)
        form.addRow("Apply to frames", self.frames_edit)

        bl = QVBoxLayout(box)
        bl.addLayout(form)
        bl.addWidget(self.run_btn)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(box)

        self._sync_smoothing()

    # ── state ──
    def method(self) -> str:
        return self.method_combo.currentData()

    def frames_text(self) -> str:
        return self.frames_edit.text()

    def _sync_smoothing(self) -> None:
        """Smoothing only affects the spline fit; grey it out for linear."""
        self.smooth_spin.setEnabled(self.method() == METHOD_SPLINE)

    def _emit_run(self) -> None:
        self.run_requested.emit(
            self.method(), self.smooth_spin.value(), self.frames_edit.text()
        )

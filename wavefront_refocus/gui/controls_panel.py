"""Left-panel controls: optical params, sweep params, compute & export."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..core.model import OpticalParams, SweepParams


class ControlsPanel(QWidget):
    """Emits parameter changes and action requests; holds no app state."""

    params_changed = Signal()
    compute_requested = Signal()
    cancel_requested = Signal()
    export_requested = Signal()
    export_json_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── optical params ──
        self.pixel_spin = self._dspin(0.001, 1000.0, config.DEFAULT_PIXEL_PITCH * 1e6, " µm", 4)
        self.mag_spin = self._dspin(0.1, 1000.0, config.DEFAULT_MAGNIFICATION, "×", 2)
        self.wl_spin = self._dspin(100.0, 5000.0, config.DEFAULT_WAVELENGTH * 1e9, " nm", 3)
        self.na_spin = self._dspin(0.01, 2.0, config.DEFAULT_NA, "", 3)

        optics_box = QGroupBox("Optics")
        of = QFormLayout(optics_box)
        of.addRow("Pixel size", self.pixel_spin)
        of.addRow("Magnification", self.mag_spin)
        of.addRow("Reference wavelength", self.wl_spin)
        of.addRow("NA", self.na_spin)

        # ── sweep params ──
        self.range_spin = self._dspin(0.001, 10000.0, config.DEFAULT_HALF_RANGE_SAMPLE * 1e6, " µm", 4)
        self.nplanes_spin = QSpinBox()
        self.nplanes_spin.setRange(3, 1001)
        self.nplanes_spin.setValue(config.DEFAULT_N_PLANES)

        sweep_box = QGroupBox("Propagation sweep (sample space)")
        sf = QFormLayout(sweep_box)
        sf.addRow("Min. propagation dist. (half-range)", self.range_spin)
        sf.addRow("Number of planes", self.nplanes_spin)

        self.compute_btn = QPushButton("Compute current image")
        self.compute_btn.clicked.connect(self.compute_requested)
        self.compute_btn.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2a90d8; }"
            "QPushButton:disabled { background-color: #b0b0b0; color: #efefef; }"
        )
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_requested)
        self.cancel_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        self.export_btn = QPushButton("Compute folder with new positions…")
        self.export_btn.clicked.connect(self.export_requested)
        self.export_json_btn = QPushButton("Export distances JSON…")
        self.export_json_btn.clicked.connect(self.export_json_requested)

        lay = QVBoxLayout(self)
        lay.addWidget(optics_box)
        lay.addWidget(sweep_box)
        lay.addWidget(self.compute_btn)
        lay.addWidget(self.cancel_btn)
        lay.addWidget(self.progress)
        lay.addStretch(1)
        lay.addWidget(self.export_btn)
        lay.addWidget(self.export_json_btn)

        for w in (self.pixel_spin, self.mag_spin, self.wl_spin, self.na_spin,
                  self.range_spin, self.nplanes_spin):
            w.valueChanged.connect(self.params_changed)

    @staticmethod
    def _dspin(lo, hi, val, suffix, decimals):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setValue(val)
        return s

    # ── value access (UI units -> SI) ──
    def optical_params(self) -> OpticalParams:
        return OpticalParams(
            pixel_pitch=self.pixel_spin.value() * 1e-6,
            magnification=self.mag_spin.value(),
            wavelength=self.wl_spin.value() * 1e-9,
            NA=self.na_spin.value(),
        )

    def sweep_params(self) -> SweepParams:
        return SweepParams(
            half_range_sample=self.range_spin.value() * 1e-6,
            n_planes=self.nplanes_spin.value(),
        )

    def set_optical_params(self, optics: OpticalParams) -> None:
        self.pixel_spin.setValue(optics.pixel_pitch * 1e6)
        self.mag_spin.setValue(optics.magnification)
        self.wl_spin.setValue(optics.wavelength * 1e9)
        self.na_spin.setValue(optics.NA)

    # ── compute UI state ──
    def set_computing(self, busy: bool) -> None:
        self.compute_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setValue(0)

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)

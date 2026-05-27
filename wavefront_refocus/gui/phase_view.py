"""Top-right phase/OPD view: zoomable image, frame slider, plane scrollbar."""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class PhaseView(QWidget):
    """Displays the unwrapped phase / OPD of the current frame.

    A horizontal slider selects the frame; a vertical scrollbar (enabled only
    after a propagation result is available) scrubs through the planes.
    """

    frame_changed = Signal(int)
    z_changed = Signal(int)
    choose_requested = Signal()
    display_range_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        self._im = None
        self._cbar = None

        # display-range state (configurable via the floating window)
        self._pct_low = 1.0
        self._pct_high = 99.0
        self._margin = 16
        self._raw_data = None
        self._last_data = None
        self._last_title = ""

        # vertical plane scrollbar
        self.z_bar = QScrollBar(Qt.Vertical)
        self.z_bar.setEnabled(False)
        self.z_bar.valueChanged.connect(self._on_z)

        canvas_row = QWidget()
        crl = QHBoxLayout(canvas_row)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.addWidget(self.canvas, 1)
        crl.addWidget(self.z_bar)

        # horizontal frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame)
        self.frame_label = QLabel("frame —")

        self.choose_btn = QPushButton("Choose this plane")
        self.choose_btn.setEnabled(False)
        self.choose_btn.clicked.connect(self.choose_requested)
        self._choose_active_style = (
            "QPushButton { background-color: #2ca02c; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #36c036; }"
        )
        self._choose_idle_style = ""

        self.status = QLabel("")

        self.range_btn = QPushButton("Display range…")
        self.range_btn.clicked.connect(self.display_range_requested)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Frame:"))
        bottom.addWidget(self.frame_slider, 1)
        bottom.addWidget(self.frame_label)
        bottom.addWidget(self.range_btn)
        bottom.addWidget(self.choose_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self.toolbar)
        lay.addWidget(canvas_row, 1)
        lay.addLayout(bottom)
        lay.addWidget(self.status)

        self._block = False

    # ── frame slider ──
    def set_frame_count(self, n: int) -> None:
        self._block = True
        self.frame_slider.setEnabled(n > 0)
        self.frame_slider.setRange(0, max(0, n - 1))
        self._block = False

    def set_current_frame(self, index: int) -> None:
        self._block = True
        self.frame_slider.setValue(index)
        self.frame_label.setText(f"frame {index}")
        self._block = False

    def _on_frame(self, value: int):
        if not self._block:
            self.frame_label.setText(f"frame {value}")
            self.frame_changed.emit(value)

    # ── plane scrollbar ──
    def enable_planes(self, n_planes: int, z0: int) -> None:
        self._block = True
        self.z_bar.setEnabled(True)
        self.z_bar.setRange(0, max(0, n_planes - 1))
        self.z_bar.setValue(z0)
        self.choose_btn.setEnabled(True)
        self.choose_btn.setStyleSheet(self._choose_active_style)
        self._block = False

    def disable_planes(self) -> None:
        self._block = True
        self.z_bar.setEnabled(False)
        self.choose_btn.setEnabled(False)
        self.choose_btn.setStyleSheet(self._choose_idle_style)
        self._block = False

    def current_z(self) -> int:
        return self.z_bar.value()

    def _on_z(self, value: int):
        if not self._block:
            self.z_changed.emit(value)

    # ── image ──
    def show_image(self, data: np.ndarray, title: str = "") -> None:
        self._raw_data = data
        self._last_title = title
        m = self._margin
        if data.shape[0] > 2 * m and data.shape[1] > 2 * m:
            data = data[m:-m, m:-m]  # crop borders
        data = data - np.mean(data)  # zero-center for better color scaling
        self._last_data = data
        if self._im is None:
            self._im = self.ax.imshow(data, cmap="viridis")
            self._cbar = self.fig.colorbar(self._im, ax=self.ax, fraction=0.046)
        else:
            self._im.set_data(data)
        self._apply_clim()
        self.ax.set_title(title)
        self.canvas.draw_idle()

    def _apply_clim(self) -> None:
        """Set the colour limits from the current percentile range."""
        if self._im is None or self._last_data is None:
            return
        lo = float(np.percentile(self._last_data, self._pct_low))
        hi = float(np.percentile(self._last_data, self._pct_high))
        if hi <= lo:
            hi = lo + 1e-9
        self._im.set_clim(lo, hi)

    def set_display_range(self, pct_low: float, pct_high: float,
                          margin: int | None = None) -> None:
        """Update the percentile clim (and optional border margin), redraw."""
        self._pct_low = max(0.0, min(pct_low, 100.0))
        self._pct_high = max(self._pct_low, min(pct_high, 100.0))
        if margin is not None:
            self._margin = max(0, int(margin))
        # A margin change alters the crop, so re-show from the raw data.
        if margin is not None and self._raw_data is not None:
            self.show_image(self._raw_data, self._last_title)
        else:
            self._apply_clim()
            self.canvas.draw_idle()

    def display_range(self) -> tuple[float, float, int]:
        return self._pct_low, self._pct_high, self._margin

    def set_status(self, text: str) -> None:
        self.status.setText(text)

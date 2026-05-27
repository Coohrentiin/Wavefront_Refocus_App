"""Distance-vs-frame graph (sample space) with clickable points."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .. import config
from ..core.model import Session
from ..core.optics import image_to_sample


class DistanceGraph(QWidget):
    """Plots per-frame distances (sample space, µm) and emits picks.

    Series:
      - base: the initial JSON distance (always shown for reference);
      - chosen: user-committed planes;
      - interpolated: spline estimates, drawn together with the fitted curve.
    The current frame is highlighted with a hollow marker. Clicking any point
    emits ``point_clicked(frame_index)``.
    """

    point_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(4, 2.5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

        self._session: Optional[Session] = None
        self._current_index: int = 0
        # The fitted interpolation curve, in display (sample µm) space:
        # (xs array, ys array) or None.
        self._interp_curve: Optional[tuple[np.ndarray, np.ndarray]] = None
        # map matplotlib artist -> list of frame indices in plotted order
        self._artist_frames: dict = {}
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def set_session(self, session: Session) -> None:
        self._session = session
        self._interp_curve = None
        self.redraw()

    def set_current_index(self, index: int) -> None:
        """Highlight the frame the user is currently viewing."""
        self._current_index = index
        self.redraw()

    def set_interp_curve(self, xs, ys) -> None:
        """Provide the fitted interpolation curve to overlay (display µm)."""
        self._interp_curve = (np.asarray(xs, float), np.asarray(ys, float))
        self.redraw()

    def clear_interp_curve(self) -> None:
        self._interp_curve = None
        self.redraw()

    # ── display-space helpers (sample µm) ──
    def _base_um(self, frame) -> float:
        M = self._session.optics.magnification
        return image_to_sample(frame.base_dz_image, M) * 1e6

    def _display_um(self, frame) -> float:
        """Current absolute display distance for a frame (sample µm)."""
        dz = frame.final_dz_sample or 0.0
        return self._base_um(frame) + dz * 1e6

    def redraw(self) -> None:
        self.ax.clear()
        self._artist_frames.clear()
        if self._session is None or not self._session.frames:
            self.canvas.draw_idle()
            return

        frames = self._session.frames

        # Base points are ALWAYS shown (greyed for chosen frames so the
        # original reference stays visible alongside the new position).
        base_idx = [f.index for f in frames]
        base_y = [self._base_um(f) for f in frames]
        base_colors = [
            config.COLOR_BASE_USED if f.is_chosen else config.COLOR_BASE
            for f in frames
        ]
        self._scatter(base_idx, base_y, base_colors, "base", s=26, zorder=2)

        chosen = [f for f in frames if f.is_chosen]
        if chosen:
            self._scatter(
                [f.index for f in chosen], [self._display_um(f) for f in chosen],
                config.COLOR_CHOSEN, "chosen", s=42, zorder=4,
            )

        interp = [f for f in frames if not f.is_chosen and f.interpolated_dz_sample is not None]
        if interp:
            self._scatter(
                [f.index for f in interp], [self._display_um(f) for f in interp],
                config.COLOR_INTERP, "interpolated", s=42, zorder=4,
            )

        # Fitted interpolation curve.
        if self._interp_curve is not None:
            xs, ys = self._interp_curve
            self.ax.plot(xs, ys, "-", color=config.COLOR_INTERP,
                         lw=1.5, alpha=0.8, zorder=3, label="interp. curve")

        # Highlight the current frame with a hollow ring at its display value.
        cur = self._session.current_frame
        if cur is not None:
            self.ax.scatter(
                [cur.index], [self._display_um(cur)],
                s=160, facecolors="none", edgecolors=config.COLOR_CURRENT,
                linewidths=2.0, zorder=5, label="current",
            )

        self.ax.set_xlabel("Frame index")
        self.ax.set_ylabel("Distance (µm, sample)")
        self.ax.legend(loc="best", fontsize="x-small")
        self.canvas.draw_idle()

    def _scatter(self, idx: List[int], y: List[float], color, label: str,
                 s: int = 30, zorder: int = 2):
        if not idx:
            return
        art = self.ax.scatter(idx, y, c=color, label=label, picker=5, s=s,
                              zorder=zorder)
        self._artist_frames[art] = list(idx)

    def _on_pick(self, event):
        frames = self._artist_frames.get(event.artist)
        if not frames:
            return
        pos = event.ind[0]
        if 0 <= pos < len(frames):
            self.point_clicked.emit(int(frames[pos]))

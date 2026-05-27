"""Main window: owns the Session and wires the panels together."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from skimage.restoration import unwrap_phase

from ..core.engine import build_compute_kwargs
from ..core.interpolation import build_evaluator, fit_and_eval
from ..core.model import Frame, Session
from ..core.optics import image_to_sample
from ..core.worker import PropagationController
from ..io import distances as dist_io
from ..io.export import export_folder
from .controls_panel import ControlsPanel
from .display_range_dialog import DisplayRangeDialog
from .distance_graph import DistanceGraph
from .export_dialog import ExportDialog
from .input_dialog import InputDialog
from .interpolation_panel import InterpolationPanel
from .phase_view import PhaseView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wavefront Refocus App")
        self.resize(1200, 800)

        self.session = Session()
        self._raw_distances: dict = {}
        self.controller = PropagationController(self)
        self._display_dialog = None

        # ── widgets ──
        self.controls = ControlsPanel()
        self.interp_panel = InterpolationPanel()
        self.graph = DistanceGraph()
        self.phase_view = PhaseView()

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.addWidget(self.controls, 1)
        left_lay.addWidget(self.interp_panel)
        left_lay.addWidget(self.graph, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.phase_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 780])

        central = QWidget()
        cl = QHBoxLayout(central)
        cl.addWidget(splitter)
        self.setCentralWidget(central)

        self._build_menu()
        self._connect()

    # ── setup ──
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        open_act = m.addAction("Open dataset…")
        open_act.triggered.connect(self.open_dataset)
        m.addSeparator()
        quit_act = m.addAction("Quit")
        quit_act.triggered.connect(self.close)

    def _connect(self):
        self.controls.params_changed.connect(self._on_params_changed)
        self.controls.compute_requested.connect(self.compute_current)
        self.controls.cancel_requested.connect(self.controller.cancel)
        self.controls.export_requested.connect(self.export_folder)
        self.controls.export_json_requested.connect(self.export_json)

        self.phase_view.frame_changed.connect(self.select_frame)
        self.phase_view.z_changed.connect(self._on_z_changed)
        self.phase_view.choose_requested.connect(self.choose_plane)
        self.phase_view.display_range_requested.connect(self.open_display_range)

        self.graph.point_clicked.connect(self.select_frame)
        self.interp_panel.run_requested.connect(self.run_interpolation)

        self.controller.progress.connect(self.controls.set_progress)
        self.controller.finished.connect(self._on_compute_finished)
        self.controller.failed.connect(self._on_compute_failed)

    # ── dataset ──
    def open_dataset(self):
        dlg = InputDialog(self)
        # seed the dialog wavelength from the controls
        dlg.wl_spin.setValue(self.controls.optical_params().wavelength * 1e9)
        if dlg.exec() != InputDialog.Accepted or dlg.loaded is None:
            return

        loaded = dlg.loaded
        optics = self.controls.optical_params()
        optics.wavelength = dlg.wavelength_nm * 1e-9
        self.controls.set_optical_params(optics)

        frames = [Frame(index=i, source=src) for i, src in enumerate(loaded.sources)]

        self._raw_distances = {}
        if dlg.distances_path:
            try:
                self._raw_distances = dist_io.load_distances(dlg.distances_path)
                dist_io.apply_base_distances(frames, self._raw_distances, optics.wavelength_nm)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Distances", f"Could not load distances:\n{exc}")

        self.session = Session(
            frames=frames,
            optics=optics,
            sweep=self.controls.sweep_params(),
            input_format=loaded.input_format,
            distances_path=dlg.distances_path or None,
            current_index=0,
            loader=loaded.loader,
        )

        self.phase_view.set_frame_count(len(frames))
        self.phase_view.set_current_frame(0)
        self.graph.set_session(self.session)
        self.select_frame(0)

    # ── parameters ──
    def _on_params_changed(self):
        if not self.session.frames:
            return
        optics = self.controls.optical_params()
        self.session.optics = optics
        self.session.sweep = self.controls.sweep_params()
        # Re-bind base distances to the (possibly new) reference wavelength.
        if self._raw_distances:
            dist_io.apply_base_distances(
                self.session.frames, self._raw_distances, optics.wavelength_nm
            )
        self.graph.redraw()

    # ── frame selection ──
    def select_frame(self, index: int):
        if not (0 <= index < len(self.session.frames)):
            return
        self.session.current_index = index
        frame = self.session.frames[index]
        self.phase_view.set_current_frame(index)
        self.phase_view.disable_planes()
        self.graph.set_current_index(index)
        self._show_base_or_position(frame)

    def _show_base_or_position(self, frame: Frame):
        """Show the frame at its base plane, or at its chosen/interp plane."""
        try:
            phase, amp = self.session.loader(frame.source)
        except Exception as exc:  # noqa: BLE001
            self.phase_view.set_status(f"Load error: {exc}")
            return

        dz = frame.final_dz_sample
        if dz is None or dz == 0.0:
            disp = self._zero_mean_opd(np.exp(1j * phase), amp)
            label = f"frame {frame.index} — base plane"
        else:
            from ..core.engine import refocus_one
            field = refocus_one(phase, amp, dz, self.session.optics)
            disp = self._zero_mean_opd(field, None)
            label = f"frame {frame.index} — {dz * 1e6:.3f} µm"
        self.phase_view.show_image(disp, label)
        self.phase_view.set_status(label)

    def _zero_mean_opd(self, field_or_phase, amp) -> np.ndarray:
        """Unwrapped, zero-mean OPD (nm) for display consistency across planes."""
        if amp is not None:
            field = amp * field_or_phase
        else:
            field = field_or_phase
        unwrapped = unwrap_phase(np.angle(field))
        opd = unwrapped * self.session.optics.wavelength_nm / (2.0 * np.pi)
        return (opd - opd.mean()).astype(np.float32)

    # ── compute current ──
    def compute_current(self):
        frame = self.session.current_frame
        if frame is None:
            return
        if self.controller.is_running():
            return
        self.session.optics = self.controls.optical_params()
        self.session.sweep = self.controls.sweep_params()
        try:
            phase, amp = self.session.loader(frame.source)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        kwargs = build_compute_kwargs(phase, amp, self.session.optics, self.session.sweep)
        self.controls.set_computing(True)
        self.phase_view.set_status("Computing…")
        if not self.controller.start(kwargs):
            self.controls.set_computing(False)

    def _on_compute_finished(self, result):
        self.controls.set_computing(False)
        frame = self.session.current_frame
        if frame is None:
            return
        frame.result = result
        self.phase_view.enable_planes(len(result.dz_sample_array), result.z0_index)
        self._show_plane(result.z0_index)
        self.phase_view.set_status(
            f"Computed {len(result.dz_sample_array)} planes ({result.backend})"
        )

    def _on_compute_failed(self, message: str):
        self.controls.set_computing(False)
        if message == "cancelled":
            self.phase_view.set_status("Cancelled")
        else:
            QMessageBox.critical(self, "Propagation failed", message)
            self.phase_view.set_status("Failed")

    # ── display range floating window ──
    def open_display_range(self):
        if self._display_dialog is None:
            low, high, margin = self.phase_view.display_range()
            # parent=None -> a detached, independent top-level window. We keep
            # a reference so it isn't garbage-collected.
            self._display_dialog = DisplayRangeDialog(low, high, margin, parent=None)
            self._display_dialog.range_changed.connect(
                self.phase_view.set_display_range
            )
        self._display_dialog.show()
        self._display_dialog.raise_()
        self._display_dialog.activateWindow()

    def _on_z_changed(self, z: int):
        self._show_plane(z)

    def _show_plane(self, z: int):
        frame = self.session.current_frame
        if frame is None or frame.result is None:
            return
        res = frame.result
        z = max(0, min(z, len(res.dz_sample_array) - 1))
        dz_um = res.dz_sample_array[z] * 1e6
        self.phase_view.show_image(
            res.opd_stack[z], f"frame {frame.index} — plane dz = {dz_um:.3f} µm"
        )
        self.phase_view.set_status(f"plane {z}/{len(res.dz_sample_array) - 1}, dz = {dz_um:.3f} µm")

    # ── choose plane ──
    def choose_plane(self):
        frame = self.session.current_frame
        if frame is None or frame.result is None:
            return
        z = self.phase_view.current_z()
        frame.chosen_dz_sample = float(frame.result.dz_sample_array[z])
        frame.interpolated_dz_sample = None
        frame.is_chosen = True
        self.graph.redraw()
        self.phase_view.set_status(
            f"Chose plane for frame {frame.index}: dz = {frame.chosen_dz_sample * 1e6:.3f} µm"
        )

    # ── interpolation ──
    def run_interpolation(self, smoothing: float):
        chosen = self.session.chosen_frames()
        if not chosen:
            QMessageBox.information(self, "Interpolation",
                                    "Choose a plane for at least one frame first.")
            return
        self.session.smoothing_factor = smoothing
        M = self.session.optics.magnification

        # Fit the spline in ABSOLUTE sample space (the distance shown on the
        # graph), so interpolated points lie on the drawn curve. Each chosen
        # frame's absolute distance is base + chosen dz.
        def absolute_sample(frame) -> float:
            return image_to_sample(frame.base_dz_image, M) + frame.chosen_dz_sample

        chosen_idx = [f.index for f in chosen]
        chosen_abs = [absolute_sample(f) for f in chosen]
        all_idx = [f.index for f in self.session.frames]

        abs_by_idx = fit_and_eval(chosen_idx, chosen_abs, all_idx, smoothing)

        # Store back as a RELATIVE dz per frame (subtract that frame's base),
        # keeping export/JSON math unchanged.
        for f in self.session.frames:
            if not f.is_chosen:
                base_sample = image_to_sample(f.base_dz_image, M)
                f.interpolated_dz_sample = abs_by_idx[f.index] - base_sample

        # Dense fitted curve for overlay (display µm), spanning all frames.
        evaluator = build_evaluator(chosen_idx, chosen_abs, smoothing)
        lo, hi = min(all_idx), max(all_idx)
        xs = np.linspace(lo, hi, max(2, (hi - lo) * 8 + 1))
        ys_um = np.array([evaluator(x) for x in xs]) * 1e6
        self.graph.set_interp_curve(xs, ys_um)
        self.phase_view.set_status("Interpolated planes for unchosen frames.")

    # ── export ──
    def export_folder(self):
        if not self.session.frames:
            return
        dlg = ExportDialog(default_format=self.session.input_format, parent=self)
        if dlg.exec() != ExportDialog.Accepted:
            return
        self.session.optics = self.controls.optical_params()

        total = len(self.session.frames)
        progress = QProgressDialog("Exporting refocused frames…", None, 0, total, self)
        progress.setWindowTitle("Export")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def on_progress(done: int, n: int):
            progress.setMaximum(n)
            progress.setValue(done)
            QApplication.processEvents()

        try:
            export_folder(self.session, dlg.out_format, dlg.out_dir,
                          progress_cb=on_progress)
        except Exception as exc:  # noqa: BLE001
            progress.close()
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        progress.close()
        QMessageBox.information(self, "Export", f"Exported to:\n{dlg.out_dir}")

    def export_json(self):
        if not self.session.frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export distances JSON", "Reconstruction_Distances.json",
            filter="JSON (*.json)",
        )
        if not path:
            return
        self.session.optics = self.controls.optical_params()
        dist_io.write_distances(path, self.session.frames, self.session.optics)
        QMessageBox.information(self, "Export", f"Distances written to:\n{path}")

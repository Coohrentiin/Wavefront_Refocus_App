"""Dialog to choose an input layout and its files, plus reference wavelength."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    DEFAULT_WAVELENGTH,
    FMT_INTENSITY_OPD,
    FMT_INTENSITY_PHASE,
    FMT_STACK,
    FMT_STACK_FOLDER,
    FORMAT_LABELS,
)
from ..io import inputs


class _PathRow(QWidget):
    """A read-only line edit with a Browse button (folder or file)."""

    def __init__(self, is_dir: bool, parent=None):
        super().__init__(parent)
        self._is_dir = is_dir
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit)
        lay.addWidget(btn)

    def _browse(self):
        if self._is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select file", filter="TIFF (*.tif *.tiff)"
            )
        if path:
            self.edit.setText(path)

    def path(self) -> str:
        return self.edit.text()


class InputDialog(QDialog):
    """Collects an input layout, paths, a distances JSON and a wavelength."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open dataset")
        self.loaded: Optional[inputs.LoadedInput] = None
        self.distances_path: str = ""
        self.wavelength_nm: float = DEFAULT_WAVELENGTH * 1e9

        self.combo = QComboBox()
        for fmt in (FMT_INTENSITY_OPD, FMT_INTENSITY_PHASE, FMT_STACK, FMT_STACK_FOLDER):
            self.combo.addItem(FORMAT_LABELS[fmt], fmt)
        self.combo.currentIndexChanged.connect(self._update_rows)

        # Two path rows; meaning depends on the selected format.
        self.row1 = _PathRow(is_dir=True)
        self.row2 = _PathRow(is_dir=True)
        self.label1 = QLabel()
        self.label2 = QLabel()

        self.wl_spin = QDoubleSpinBox()
        self.wl_spin.setRange(100.0, 5000.0)
        self.wl_spin.setDecimals(3)
        self.wl_spin.setSuffix(" nm")
        self.wl_spin.setValue(self.wavelength_nm)

        self.dist_row = _PathRow(is_dir=False)
        # distances JSON browse should accept .json
        self.dist_row.edit.setPlaceholderText("Reconstruction_Distances.json")

        form = QFormLayout()
        form.addRow("Input layout", self.combo)
        form.addRow(self.label1, self.row1)
        form.addRow(self.label2, self.row2)
        form.addRow("Reference wavelength", self.wl_spin)
        form.addRow("Distances JSON", self._json_row())

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

        self._update_rows()

    def _json_row(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.json_edit = QLineEdit()
        self.json_edit.setReadOnly(True)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_json)
        lay.addWidget(self.json_edit)
        lay.addWidget(btn)
        return w

    def _browse_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select distances JSON", filter="JSON (*.json)"
        )
        if path:
            self.json_edit.setText(path)

    def _fmt(self) -> str:
        return self.combo.currentData()

    def _update_rows(self):
        fmt = self._fmt()
        if fmt == FMT_INTENSITY_OPD:
            self.label1.setText("Intensity folder")
            self.label2.setText("OPD folder (nm)")
            self.row2.setVisible(True)
            self.label2.setVisible(True)
            self._set_dir(self.row1, True)
            self._set_dir(self.row2, True)
        elif fmt == FMT_INTENSITY_PHASE:
            self.label1.setText("Intensity folder")
            self.label2.setText("Phase folder (rad)")
            self.row2.setVisible(True)
            self.label2.setVisible(True)
            self._set_dir(self.row1, True)
            self._set_dir(self.row2, True)
        elif fmt == FMT_STACK:
            self.label1.setText("Wavefront stack (.tif)")
            self.label2.setVisible(False)
            self.row2.setVisible(False)
            self._set_dir(self.row1, False)
        elif fmt == FMT_STACK_FOLDER:
            self.label1.setText("Folder of stacks")
            self.label2.setVisible(False)
            self.row2.setVisible(False)
            self._set_dir(self.row1, True)

    @staticmethod
    def _set_dir(row: _PathRow, is_dir: bool):
        row._is_dir = is_dir

    def _on_accept(self):
        from PySide6.QtWidgets import QMessageBox

        fmt = self._fmt()
        wl = self.wl_spin.value()
        try:
            if fmt == FMT_INTENSITY_OPD:
                loaded = inputs.open_intensity_opd(
                    self.row1.path(), self.row2.path(), wl
                )
            elif fmt == FMT_INTENSITY_PHASE:
                loaded = inputs.open_intensity_phase(
                    self.row1.path(), self.row2.path()
                )
            elif fmt == FMT_STACK:
                loaded = inputs.open_stack(self.row1.path())
            else:
                loaded = inputs.open_stack_folder(self.row1.path())
        except Exception as exc:  # noqa: BLE001 — surface load errors to user
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.loaded = loaded
        self.distances_path = self.json_edit.text()
        self.wavelength_nm = wl
        self.accept()

"""Dark theme: Fusion QPalette plus matching matplotlib figure/axes colours."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette

# Matplotlib palette, kept consistent with the Qt QPalette below.
MPL_WINDOW = "#353535"   # QPalette.Window  (53, 53, 53)
MPL_BASE = "#232323"     # QPalette.Base    (35, 35, 35)
FG = "#ffffff"           # white text
BORDER = "#5a5a5a"


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(53, 53, 53))
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Base, QColor(35, 35, 35))
    p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.ToolTipBase, Qt.white)
    p.setColor(QPalette.ToolTipText, Qt.white)
    p.setColor(QPalette.Text, Qt.white)
    p.setColor(QPalette.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.BrightText, Qt.red)
    p.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    p.setColor(QPalette.HighlightedText, Qt.black)
    # Disabled-state legibility.
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    return p


def apply_dark(app) -> None:
    """Apply the dark Fusion palette and matching matplotlib rcParams."""
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    # Fusion leaves tooltips with a default border; give it a thin one.
    app.setStyleSheet(
        "QToolTip { color: #ffffff; background-color: #353535;"
        " border: 1px solid #5a5a5a; }"
    )

    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": MPL_WINDOW,
        "axes.facecolor": MPL_BASE,
        "savefig.facecolor": MPL_WINDOW,
        "text.color": FG,
        "axes.labelcolor": FG,
        "axes.edgecolor": BORDER,
        "xtick.color": FG,
        "ytick.color": FG,
        "grid.color": BORDER,
        "legend.facecolor": MPL_BASE,
        "legend.edgecolor": BORDER,
    })


def style_figure(fig) -> None:
    """Recolour an already-created figure/axes to the dark palette."""
    fig.set_facecolor(MPL_WINDOW)
    for ax in fig.axes:
        ax.set_facecolor(MPL_BASE)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=FG)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)

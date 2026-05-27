"""Default optical parameters and UI constants."""

# ── Default optical parameters (SI units; overridable from the UI) ───────
DEFAULT_WAVELENGTH = 660e-9     # m
DEFAULT_NA = 0.8
DEFAULT_MAGNIFICATION = 20.0
DEFAULT_PIXEL_PITCH = 5.86e-6   # m

# ── Default sweep parameters (sample space) ──────────────────────────────
DEFAULT_HALF_RANGE_SAMPLE = 10e-6   # m, "minimal propagation distance"
DEFAULT_N_PLANES = 41

# ── Interpolation ────────────────────────────────────────────────────────
DEFAULT_SMOOTHING = 0.0

# ── Graph colours ─────────────────────────────────────────────────────────
COLOR_BASE = "#1f77b4"          # initial JSON distance
COLOR_BASE_USED = "#9e9e9e"     # former base of a chosen frame (greyed)
COLOR_CHOSEN = "#d62728"        # user-committed plane
COLOR_INTERP = "#2ca02c"        # spline-interpolated plane
COLOR_CURRENT = "#ff7f0e"       # highlight ring on the current frame

# ── Input / output format identifiers ─────────────────────────────────────
FMT_INTENSITY_OPD = "intensity_opd"
FMT_INTENSITY_PHASE = "intensity_phase"
FMT_STACK = "stack"
FMT_STACK_FOLDER = "stack_folder"

FORMAT_LABELS = {
    FMT_INTENSITY_OPD: "Intensity folder + OPD folder (nm)",
    FMT_INTENSITY_PHASE: "Intensity folder + Phase folder (rad)",
    FMT_STACK: "Single multi-frame wavefront stack",
    FMT_STACK_FOLDER: "Folder of wavefront stacks",
}

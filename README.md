# Wavefront Refocus App

Interactive refocusing of wavefronts reconstructed from a Digital Holographic
Microscope (DHM). Reconstructed images are often not on the ideal focal plane;
this tool lets you numerically propagate each frame, pick the best plane,
interpolate planes for the remaining frames, and re-export the whole sequence
refocused.

## Install

```bash
pip install -e .
# optional GPU acceleration (CUDA build matching your system):
pip install torch   # see https://pytorch.org/get-started/locally/
```

The numerical engine uses PyTorch + CUDA when available and falls back to NumPy
otherwise.

## Run

```bash
python -m wavefront_refocus
# or, after install:
wavefront-refocus
```

## Input layouts

The app loads four data organisations (chosen in the open dialog):

1. **Intensity folder + OPD folder** — per-frame `.tif`; OPD in nanometres.
   You are asked for the reference wavelength to convert OPD → phase.
2. **Intensity folder + Phase folder** — per-frame `.tif`; phase in radians.
3. **Single multi-frame wavefront stack** — one TIFF, 2 channels
   (phase, intensity) with a time axis.
4. **Folder of wavefront stacks** — each file is one frame's phase+intensity
   stack.

Frame order is the natural-sorted file order (or the time axis for a single
stack). Alongside the data you load a `Reconstruction_Distances.json` mapping
each frame index to a `{wavelength_nm: distance_m}` table.

## Conventions

- **Reconstruction distances are absolute and in image space.** For each frame
  the distance is read from the wavelength key nearest the UI reference
  wavelength.
- **The UI works and displays in sample space.** The axial conversion is
  `dz_image = dz_sample * M²` (M = magnification). On export, the chosen
  sample-space displacement is converted back to image space and added to the
  absolute distance.
- **Sweep parameters**: "minimal propagation distance" is the *half-range*
  (maximum excursion each side, sample space) and "number of planes" is the
  total plane count (rounded up to an odd number so the current plane is
  included), distributed symmetrically around the current plane.
- **Interpolation** fits a smoothing spline to (frame index, chosen sample
  distance) over the frames you chose, with an exposed smoothing factor and
  explicit linear extrapolation beyond the chosen range.

## Workflow

1. Open a dataset and its distances JSON.
2. Select a frame (slider or by clicking a point in the distance graph) to view
   its unwrapped phase at the file/base plane.
3. Adjust optical and sweep parameters, press **Compute current image** to
   propagate around the current plane (runs in the background).
4. Scrub the vertical plane scrollbar and press **Choose this plane**. The old
   point greys out and a coloured point appears at the new position.
5. Repeat for several frames, then run **Interpolation tools** to estimate the
   planes of the remaining frames (shown in another colour, clickable).
6. **Compute folder with new positions** to export the refocused sequence in
   one of the four formats, plus an updated distances JSON. A progress bar
   tracks the export.

## Display

The UI uses a dark theme. The image colour scale is set from percentiles of
the (border-cropped, zero-centred) data; open **Display range…** under the
phase view for a floating window to adjust the low/high percentiles and the
border crop live.

## Development

```bash
pip install -e ".[dev]"
pytest
```

# UCM Implementation Status

Current state of the Urban Cooling Model implementation in the Ecosystem Explorer
vs canonical `natcap.invest.urban_cooling_model`.

## What's canonical

- **Per-pixel CC formula** (`CC = 0.6·shade + 0.2·albedo + 0.2·ETI`, with
  `ETI = Kc × ET / max(ET)`): exact match to InVEST's `calc_cc_op_factors`.
- **HMI algorithm** (`HMI = max(CC_local, CC_park)` — exponential park-proximity
  decay sourced from green areas, gated on a 2-hectare green-area threshold
  within `d_cool = 450 m`): exact match to InVEST's `mask_cc_green_areas_op` →
  exponential-decay convolution → `hm_op` chain.
- **Validation:** `compare_ucm_invest.py` confirms **MAE = 0.0000, r = 1.0000**
  against `natcap.invest.urban_cooling_model.execute()` on the MN baseline
  (raw CC and HMI both). Closed in commit 512fff8.

## What remains divergent

- **Per-pixel vs per-building energy aggregation.** Canonical InVEST UCM samples
  T_air per building, averaged over a 600 m `t_air_average_radius`, before
  applying the consumption rate. The prototype applies the energy formula per
  pixel. This affects *Cooling Energy Savings* dollar magnitudes (read them as
  order-of-magnitude) but **not** *Temperature Change*, which is a direct
  function of the canonical HMI raster.

## Implementation notes

- Helpers in `app.py`: `_compute_cc_raw_pure` (per-pixel CC),
  `_compute_cc_park_raster` (exponential park cooling), `_compute_green_area_sum`
  (2-ha eligibility count), `_compute_hmi_raster` / `_compute_hmi_raster_pure`
  (the `hm_op` max logic).
- Convolutions use `scipy.signal.fftconvolve` with an edge correction
  (`_convolve_edge_corrected`) that reproduces
  `pygeoprocessing.convolve_2d(ignore_nodata_and_edges=True, normalize_kernel=False)`:
  the raw convolution is divided by the kernel weight overlapping valid data,
  then rescaled by the kernel sum.
- Kernels (`_HMI_EXP_KERNEL`, `_HMI_DICH_KERNEL`) match
  `pygeoprocessing.kernels` geometry. Parameters are hardcoded at InVEST
  canonical values: `d_cool = 450 m`, decay distance = 15 px at 30 m
  resolution, 2-hectare green-area threshold = `2e4 / 30² ≈ 22.2` pixels.

## Historical context

This document originated as an audit of the HMI park-proximity gap (closed in
commit 512fff8) and the per-building energy aggregation gap (still open). See
the "Official InVEST alignment" section in `REFERENCE.md` for the per-metric
parity table and per-model gap notes.

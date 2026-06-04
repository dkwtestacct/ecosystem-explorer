"""Ownership mode data tables — pure constants, no logic.

Moved out of `app.py` (Constants Refactor / Task #52). Values are
byte-identical to the prior app.py definitions; the mask-building
helpers, the resolver, and the normalizer all stay in `app.py` because
they depend on `app.py` runtime objects (the band-1 / band-2 rasters
in `_CURRENT_CITY_STATE`).

Two tables:
  - OWNERSHIP_MODES: the 16-key mode dict (8 single-class + 3 coarse
    rollups + 6 per-class vacant composites — Finer Ownership Classes
    Pass + School / University Split + Batch 4 v2 composites). Each
    entry is `{label: str, band1_eq?: int, band1_in?: tuple[int], band2_eq?: int}`.
    Consumed by `_build_ownership_mask(band1, band2, mode_cfg)` in
    `app.py`, the export-bundle rich-block composition, the comparison-
    table Area/Ownership columns, the audit expander, and the CSV
    export.
  - ELIGIBLE_FILTER_PRIMARY_MODES: the ordered tuple of "primary"
    (non-composite) mode keys the sidebar checkbox / selectbox UI
    surfaces directly. Per-class vacant composites are resolved by
    the vacant-overlay checkbox at filter time.

Both consumers (app.py + verify_baselines.py) import directly from
this module — see also the docstring in `region_local_metrics.py`.
"""

# mapping for `data/sa/sa_public_vacant_30m.tif` (codes -1/0/1/2/3). The UI
# selectbox surfaces `label`s; the caller composes a boolean mask via
# `np.isin(ownership_raster, codes)`. SA-only.
# Finer Ownership Classes Pass (`OWNERSHIP_FINER_CLASSES_SPEC.md`) — the
# two-band raster encodes band 1 = class enum (0-6) and band 2 = vacant
# flag (0/1). Each mode below resolves to a boolean mask via
# `_build_ownership_mask` (selector keys: band1_eq / band1_in / band2_eq;
# absent key = unconstrained on that axis).
#
# School / University Split (Batch 2 pre-push addendum): the combined
# `school_university` class is split into two — `school` (K-12 public
# districts; folded into the public rollup) and `university` (mixed
# public + private higher-ed; kept out of public, flagged mixed in the
# DATA_INVENTORY caveat).
#
# `public` rollup = city + county + state-federal + school. School
# districts are government; folding them in restores the obvious case
# the prior over-broad "public excludes all education" rollup missed.
# University stays OUT of public — that bucket includes private campuses
# (Trinity, St. Mary's, OLLU) and a planning-screen "Publicly-owned land"
# filter shouldn't pretend a private campus is public land.
OWNERSHIP_MODES = {
    # ── Coarse rollups (unchanged keys; band1_in expanded to include
    # school after the split) ──
    # `short` is the terse provenance-bar variant (used in the visible
    # Source line via `_ownership_source_suffix`); `label` is the full
    # form that stays in the audit expander, comparison table, export
    # bundle, and Source-detail surfaces.
    'public': {
        'label':    'Publicly-owned land',
        'short':    'public land',
        'band1_in': (1, 2, 3, 4),  # city + county + state-federal + school
    },
    'vacant': {
        'label':    'Vacant land',
        'short':    'vacant land',
        'band2_eq': 1,
    },
    'vacant_public': {
        'label':    'Vacant publicly-owned land',
        'short':    'vacant public land',
        'band1_in': (1, 2, 3, 4),
        'band2_eq': 1,
    },
    # ── Finer modes ──
    'city': {
        'label':    'City-owned land',
        'short':    'city land',
        'band1_eq': 1,
    },
    'county': {
        'label':    'County-owned land',
        'short':    'county land',
        'band1_eq': 2,
    },
    'state_federal': {
        'label':    'State or federal land',
        'short':    'state/federal land',
        'band1_eq': 3,
    },
    'school': {
        'label':    'School district land (K-12 public)',
        'short':    'school land',
        'band1_eq': 4,
    },
    'university': {
        'label':    'College or university land',
        'short':    'university land',
        'band1_eq': 6,
    },
    'private': {
        'label':    'Privately-owned land',
        'short':    'private land',
        'band1_eq': 0,
    },
    'unknown': {
        'label':    'Unknown ownership',
        'short':    'unknown',
        'band1_eq': 5,
    },
    # ── Vacant-overlay composites (Batch 4 of OWNERSHIP_FINER_CLASSES_SPEC.md) ──
    # The sidebar's "Limit to vacant parcels only" checkbox composes the
    # vacant flag (band2_eq=1) with the selected class. These per-class
    # vacant variants are the resolved mode keys; the selectbox shows
    # only the primary class options, the checkbox composes the variant.
    # `vacant` (class-unconstrained) and `vacant_public` (rollup) live
    # above with the coarse rollups and serve the All-ownership + vacant
    # and Publicly-owned + vacant cases. Saved scenarios from before this
    # batch resolve cleanly under their existing mode keys.
    'city_vacant': {
        'label':    'City-owned land (vacant only)',
        'short':    'city land (vacant)',
        'band1_eq': 1,
        'band2_eq': 1,
    },
    'county_vacant': {
        'label':    'County-owned land (vacant only)',
        'short':    'county land (vacant)',
        'band1_eq': 2,
        'band2_eq': 1,
    },
    'state_federal_vacant': {
        'label':    'State or federal land (vacant only)',
        'short':    'state/federal land (vacant)',
        'band1_eq': 3,
        'band2_eq': 1,
    },
    'school_vacant': {
        'label':    'School district land (K-12 public, vacant only)',
        'short':    'school land (vacant)',
        'band1_eq': 4,
        'band2_eq': 1,
    },
    'university_vacant': {
        'label':    'College or university land (vacant only)',
        'short':    'university land (vacant)',
        'band1_eq': 6,
        'band2_eq': 1,
    },
    'private_vacant': {
        'label':    'Privately-owned land (vacant only)',
        'short':    'private land (vacant)',
        'band1_eq': 0,
        'band2_eq': 1,
    },
}

# Eligible land filter — selectbox primary modes (Batch 4). The "vacant"
# overlay is a separate checkbox in the UI; per-class vacant composites
# (city_vacant, school_vacant, …) are resolved at filter time from
# (selected primary class, vacant overlay). Order = display order in the
# selectbox; "(no filter)" prepended at render time.
ELIGIBLE_FILTER_PRIMARY_MODES = (
    "public",         # rollup
    "city",
    "county",
    "state_federal",
    "school",
    "university",
    "private",
    "unknown",
)

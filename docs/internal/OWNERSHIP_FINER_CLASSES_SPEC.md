# Finer Ownership Classes — Build Spec (four batches)

**Audience:** Internal
**Status:** Building — four batches built in strict sequence (1→2→3→4); each held for eyeball before the next.
**Depends on:** Ownership Filter (SA live), Subset Invariants (live), Eligibility Funnel (live), Optimizer Reversal (live), **Ownership Feasibility Profiling (live — locks the taxonomy and the rules)**.
**Builds:** A six-class ownership taxonomy (City / County / State-federal / School-university / Private / Unknown) on a two-band raster, the matching `OWNERSHIP_MODES` expansion, new subset-invariant cells exercising region × finer-class masks, and a UI capstone "Eligible land filter" panel. SA-only — MN has no ownership data.
**Source of truth for:** the four-batch sequencing, the raster encoding contract, the rollup-from-band-1 rule, and the locked UI text + KNOWN_DIVERGENCES seed entry.

---

## Why

`OWNERSHIP_FEASIBILITY_PROFILING.md` proved the six-way split classifies 99.9% of public acreage cleanly. The current `OWNERSHIP_MODES` only exposes three modes (`public` / `vacant` / `vacant_public`) — a planner can ask "limit to publicly-owned land" but can't ask "limit to City-owned only" or "limit to vacant land that's school district". The finer classes unlock those filters; the workstream below ships them end-to-end.

The four-batch ordering exists because Batch 1 is load-bearing — the raster IS the data; a class mislabel propagates everywhere downstream. The reconciliation assertion against the feasibility doc breakdown is the safety net.

## Raster encoding (the foundation)

**Two bands. Not more single-band codes.**

A single-band encoding for six classes × {vacant, not-vacant} = 12 codes is brittle: adding a class later means picking a new code, hoping nothing collides, and updating every consumer's filter expression. Two bands keep the dimensions orthogonal:

- **Band 1 — ownership class enum:** `0=private`, `1=city`, `2=county`, `3=state-federal`, `4=school-university`, `5=unknown`. NODATA `-1` (outside the SA AOI). Adding a class later is a new band-1 value with no code reshuffle.
- **Band 2 — vacant flag:** `0=not-vacant`, `1=vacant`. NODATA `-1`. The vacant rule keys on tax-exemption + improvement value (per the existing logic at `download_bexar_parcels.py:393-418`), independent of ownership class.

Consumers compose masks via AND:
```
city_mask           = band1 == 1
county_mask         = band1 == 2
state_federal_mask  = band1 == 3
school_uni_mask     = band1 == 4
private_mask        = band1 == 0
unknown_mask        = band1 == 5
vacant_mask         = band2 == 1

# "Public" rollup — government-owned land: city + county + state-federal.
# School-university is INTENTIONALLY EXCLUDED — that bucket spans both
# public institutions (ISDs, Alamo CCD, UT, TX A&M) AND private
# institutions (Trinity, St. Mary's, OLLU). The split would require more
# regex work; until then, treating the whole bucket as not-public avoids
# wrongly counting private campuses as publicly-available planning land.
# School-university stays selectable on its own.
public_mask         = np.isin(band1, [1, 2, 3])
vacant_public_mask  = public_mask & vacant_mask

# Composable per-class × vacant:
city_vacant_mask    = (band1 == 1) & (band2 == 1)
```

**Path:** new file `data/sa/sa_ownership_2band_30m.tif`. The legacy `sa_public_vacant_30m.tif` (single-band codes 0/1/2/3) stays in place until Batch 2 retires it — keeps Batch 1 strictly additive.

## Backward-compat: coarse modes as rollups

`OWNERSHIP_MODES` retains its existing keys (`public`, `vacant`, `vacant_public`). The mask-build path in `app.py` evaluates them as band-derived rollups so existing saved scenarios, comparison rows, and exported metadata continue to read identically.

```python
OWNERSHIP_MODES = {
  # Coarse rollups — `public` = city + county + state-federal ONLY
  # (school-university intentionally excluded; see "Public rollup
  # composition" below).
  'public':         {'label': 'Publicly-owned land',          'rollup': True,  'band1_in': (1, 2, 3)},
  'vacant':         {'label': 'Vacant land',                  'rollup': True,  'band2_eq': 1},
  'vacant_public':  {'label': 'Vacant publicly-owned land',   'rollup': True,  'band1_in': (1, 2, 3), 'band2_eq': 1},
  # Finer modes (new in Batch 2)
  'city':            {'label': 'City-owned land',                  'band1_eq': 1},
  'county':          {'label': 'County-owned land',                'band1_eq': 2},
  'state_federal':   {'label': 'State or federal land',            'band1_eq': 3},
  'school_university': {'label': 'School-district or university land', 'band1_eq': 4},
  # ... and matching vacant-overlay variants if/when the UI requests them
}
```

The exact shape of `OWNERSHIP_MODES` is settled in Batch 2; the contract is that **the `public` rollup excludes school-university** (`band1_in: (1, 2, 3)`, not `(1, 2, 3, 4)`).

### Public rollup composition — why school-university is excluded

The spot-check during Batch 1 surfaced that the `school_university` class includes **both public AND private institutions**: ISDs (San Antonio ISD, Northside ISD, NEISD, …) and state universities (Alamo CCD, UT System, Texas A&M) are clearly public — but Trinity University, St. Mary's University, and Our Lady of the Lake University are private and were caught by the same `\b(UNIVERSITY|COLLEGE)\b` rule.

Splitting the bucket further would require additional regex work (recognizing private religious / private nonprofit university names). Rather than do that work now, **school-university is excluded from the `public` rollup entirely** — keeping a private Trinity campus out of "Publicly-owned land" matters more for the planning-screen use case than keeping a state-university campus in.

School-university stays a first-class selectable filter (a planner who explicitly wants "all school district + university campuses, public or private" can pick it directly). It just isn't a default member of the `public` rollup.

## Batches

### Batch 1 — Classifier rewrite + raster re-encoding (foundation)

**Goal:** Apply the OWNERSHIP_FEASIBILITY_PROFILING.md rules to the archived BCAD polygons; rasterize as two bands; commit the new raster + script update + DATA_INVENTORY entry.

**Steps:**
1. Refactor `classify_and_rasterize()` in `scripts/data/download_bexar_parcels.py`:
   - New `_classify_six_way(owner)` function (regex-driven, HOA filter applied to the County branch).
   - New `_rasterize_two_band(g_5070, out_path, ref_raster)` writer.
   - Existing `classify_and_rasterize()` continues to write the legacy single-band `sa_public_vacant_30m.tif` (for backward compat); new function adds the two-band file alongside.
2. Add `--reclassify-from <gpkg>` flag so we can re-rasterize from the archived GPKG without re-fetching BCAD pages. Reads the archived GeoDataFrame, runs the new classifier, writes the new two-band raster + updated polygon GPKG.
3. Run the re-classification on `/Users/dkw-testing/Desktop/ecosystem_explorer_archive/sa_ownership_bexar_2026-05-31.gpkg`.
4. Commit:
   - `data/sa/sa_ownership_2band_30m.tif` (new two-band raster on the SA grid).
   - `scripts/data/download_bexar_parcels.py` (new classifier + writer + `--reclassify-from`).
   - `docs/internal/DATA_INVENTORY.md` (new taxonomy + breakdown + caveat).

**Scope delta vs the feasibility doc.** The feasibility doc's Step 5 numbers
(City 117,646 / State-federal 42,392 / County 2,849 / School-university
2,637) were scoped **within `is_public=1`** (the old 11-way classifier's
public-government set = 165,653 ac). Batch 1 applies the same rule to the
**full parcel set** (all 710,772 parcels), which catches gov-owned land
that the old conservative `is_public` flag missed — mostly
`tax_exempt_other` parcels that are genuinely city (CPS Energy, SAWS,
suburban city governments) or state (TX Parks & Wildlife) but weren't
in the original `is_public` whitelist, plus `university` (which was its
own old class, not in `is_public`) folding into `school_university`.

Spot-check of the ~26k acres of newly classified land:

| Class flip | Acres | Top examples | All gov? |
|---|---|---|---|
| tax_exempt_other → city | 8,923 ac | CPS Energy (3,467 ac), SAWS (2,880 ac), small TX cities (Universal City, Converse, Helotes, …) | yes |
| tax_exempt_other → state_federal | 12,096 ac | TX Parks & Wildlife Dept (12,106 ac), Port Authority of SA | yes |
| university → school_university | 3,677 ac | Alamo CCD, TX A&M System, UT System (public); Trinity, St. Mary's, OLLU (private) | **no — mixed** |
| church → school_university | 94 ac | Religious-college tail | mixed |
| private → state_federal | 395 ac | Genuine federal (US Government) parcels misflagged in the old run | yes |
| private → county | 130 ac | Bexar County properties (non-HOA) caught by tightened rule | yes |

The `school_university` flips include both public and private institutions — Trinity, St. Mary's, and OLLU are private universities caught by the same `\b(UNIVERSITY|COLLEGE)\b` rule. This is why the **`public` rollup excludes school-university entirely** (see "Public rollup composition" above). School-university stays a selectable class on its own — but isn't a default member of "Publicly-owned land".

**Rule refinement after spot-check.** The feasibility doc's regex included
standalone `\bUSA\b` and `\bFEDERAL\b` in the state-federal rule. Applied
to all parcels, these caught business names: "BORALIS USA INC", "FORESTAR
(USA) REAL ESTATE GROUP", "HOME DEPOT USA INC", "SECURITY SERVICE FEDERAL
CREDIT UNION" — ~1,272 ac of private companies. Batch 1 drops these from
the state-federal regex; "UNITED STATES" / "U S GOVERNMENT" / "U.S." still
catch all the federal-government patterns observed in the public-set
analysis.

**Final per-class polygon-Acres (full Bexar County, after rule refinement
— the reconciliation target):**

```
private             606,379 ac
city                126,634 ac
state_federal        54,883 ac
school_university     6,430 ac
county                3,018 ac
unknown               1,735 ac
TOTAL               799,079 ac
```

**Verification (Batch 1):**
- `verify_baselines.py` 40/40 byte-identical (no app-code consumer of the new raster yet).
- **NEW — rule-output reconciliation:** re-apply the six-way classifier to the archived GPKG, aggregate `Acres` per class, assert match to the full-parcel breakdown above within **±0.5%**. This is the load-bearing correctness check — it surfaces every classification difference. Recomputes from the raw GPKG so it's not a tautology.
- **NEW — raster-integrity reconciliation:** read the new raster's band 1, count pixels per class, multiply by `PIXEL_AREA_ACRES`, assert each class matches geometry-acres-in-AOI within **±5%** (rasterization rounding + AOI-boundary clip effects). This catches rasterization regressions independently.
- Eyeball: open the new raster in QGIS or read a slice in a Python REPL; spot-check that the City class fills the recognizable City of San Antonio footprint, the State-federal fills the Lackland / Fort Sam / TX-parks footprints, the School-university fills the Alamo CCD / UT / Trinity / St. Mary's campuses.

**Caveats that live in DATA_INVENTORY (not the assertion):**

- **BCAD `Acres` field ≠ polygon geometry for two classes.** Reported `Acres` and geometry-derived acres reconcile within ~2% for private / county / school_university / state_federal, but diverge for City (reported 126,634; in-AOI geom ~40,504) and Unknown (reported 1,735; in-AOI geom ~15,905). City: the single "CITY OF SAN ANTONIO" master record reports ~115k aggregated acres but its polygon footprint is much smaller — BCAD overcounts. Unknown: parcels with Owner blank are mostly road rights-of-way / water bodies with `Acres=0` but real polygon footprints. Documented; not a bug.
- **Rule-derived, not ground-truthed.** Classes come from regex-parsing BCAD `Owner` + `Exempts`. No cross-check against an authoritative title registry. The filter is a planning screen.

### Batch 2 — `OWNERSHIP_MODES` expansion (app.py + config)

**Goal:** Add the 6 finer modes; preserve coarse rollups; switch the reader to consume the new two-band raster.

**Steps:**
1. `app.py:OWNERSHIP_MODES` — expand with finer keys; coarse keys gain rollup metadata (band1_in / band2_eq).
2. `_load_city_runtime_state` — load the two-band raster instead of single-band. Store as `ownership_raster` (band 1) + `ownership_vacant_raster` (band 2), or as a 2D tuple — pick whichever minimizes downstream churn.
3. The mask-build site in `app.py:4374+` reads both bands and composes the active mode's mask.
4. CRS assertion fires on the new file just like the old one.

**Verification (Batch 2):**
- `verify_baselines.py` 40/40 byte-identical.
- Existing region/ownership assertions still pass. The new `public` rollup = city + county + state_federal (NOT school-university — see "Public rollup composition") differs in semantics from the old `is_public=1` set, which included `isd` and `river_auth`. The eligible_pixels_in_region assertion for `vacant_public` is re-baselined in Batch 2 against the new rollup definition; the spot-check delta gets documented.
- Smoke: select each new finer mode in the sidebar; confirm the mask shape resolves correctly (City-only mask has ~117k acres of polygon-Acres; County-only has ~3k; etc.).

### Batch 3 — Subset-invariants extension (verify_baselines.py)

**Goal:** Lock the finer-class contract — `converted ⊆ city_mask`, `converted ⊆ county_mask`, etc.

**Steps:**
1. Add new matrix cells, all SA:
   - `SA / region + City-only` — district 5 + ownership=`city`. converted ⊆ city_mask.
   - `SA / region + State-federal-only` — district 5 + ownership=`state_federal`.
   - `SA / region + School-university-only` — district 5 + ownership=`school_university`.
   - (County-only is a tiny set — only ~3k acres total; skip unless useful.)
2. Each cell exercises all three subset checks (eligible / region / ownership) where the ownership mask is the finer-class mask.
3. The matrix's existing coarse-rollup cells (`region + ownership-only (vacant_public)` etc.) **must continue to pass** — they're unions over band-1 values, so they're still subset of each contributing finer class. Same invariant; richer coverage.

**Verification (Batch 3):**
- `verify_baselines.py` 40/40 byte-identical.
- Matrix is now SA 10 + MN 4 = 14 cells; ~12 region-active cells reconcile to record.

### Batch 4 — UI capstone: "Eligible land filter" panel + KNOWN_DIVERGENCES seed entry

**Goal:** One sidebar panel that says "where can conversions land" — exclusions first, ownership options second, with the finer classes.

**Layout:**
```
Eligible land filter

  Conversions can never be placed on:
    • Building footprints (always excluded)
    • Roads (always excluded)
    • Existing natural land (always excluded)

  Ownership filter:                      [ All ownership ▼ ]
    Options: All ownership / City-owned / County-owned /
             State or federal / School or university /
             Publicly-owned (any of the above) /
             Privately-owned / Unknown ownership
    [ ] Limit to vacant parcels only        ← composable overlay
```

The "Limit to vacant parcels only" checkbox composes with the ownership selectbox: an explicit AND. Vacant-only without an ownership filter is a valid mode (`vacant`).

**Exact UI text (no paraphrasing):**

> Ownership filters are feasibility constraints. They limit where conversions may be placed but do not change the biophysical model equations.

**KNOWN_DIVERGENCES seed entry (new):**
```
{
  "id": "ownership_rule_derived",
  "title": "Ownership classes are rule-derived, not authoritative",
  "summary": "Ownership classes (City / County / State-federal / "
             "School-university / Private / Unknown) are derived by "
             "parsing the BCAD parcel-attribute Owner and Exempts fields "
             "with regex rules. They are not validated against an "
             "authoritative title registry. The filter is a planning "
             "screen — useful for narrowing where a hypothetical "
             "conversion could land — not a substitute for verified "
             "ownership data.",
  "scope": "ownership_filter",
}
```

The `verify_baselines.py` completeness assertion already enforces that every KNOWN_DIVERGENCES entry appears in metadata.json; the new entry is picked up by Batch 4's run automatically.

**Tooltip on the ownership selectbox** echoes the locked feasibility text + a one-line pointer to the divergence entry.

**Pairs with the funnel:** the panel is the *input* surface (where can conversions land); the funnel is the *resulting chain* (how the inputs flow through to converted acres). The two read consistently — same record, same masks, same numbers.

**Verification (Batch 4):**
- `verify_baselines.py` 40/40 byte-identical.
- Honesty-Surface completeness assertion picks up the new divergence id (auto-checked by the existing block).
- Eyeball: finer classes filter correctly under the new panel; funnel reflects the finer-class drop in its "After ownership filter" row; tooltip text matches the spec.

## Sequencing

Batches 1→2→3→4. **Eyeball after each, push after eyeball.** Batch 1 is load-bearing — class errors propagate everywhere. The reconciliation assertion is the safety net but doesn't replace visual confirmation that the raster matches expectations (esp. the City class, which dominates by acreage).

## Not touched

- MN ownership — no data; MN's `ownership_layer` config stays empty.
- `evaluate_scenario` math — ownership is a placement constraint, not a metric input.
- `SCENARIO_SCHEMA_VERSION` — the bundle's `ownership_filter` schema is unchanged (still `{mode, label, allowed_classes, source, data_date}`); only the mode-key universe grows.
- Existing saved scenarios — their `ownership_filter: 'public'` still resolves correctly under the new rollup semantics.

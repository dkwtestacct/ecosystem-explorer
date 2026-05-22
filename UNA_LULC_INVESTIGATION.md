# UNA LULC Investigation

Comparison of the prototype's existing cooling LULC and the InVEST UNA
sample LULC (for MN), as input to the canonical InVEST UNA implementation.
Determines whether the choice of LULC raster materially affects the UNA
result.

**Date:** 2026-05-22

**Script:** `compare_una_lulc.py` · **Output:** `comparisons/una_lulc_comparison_mn.csv`

---

## Headline finding

**The two rasters are byte-for-byte identical.** The prototype's cooling
LULC (`data/cooling/land_use_2021.tif`) is a renamed copy of the InVEST UNA
sample LULC (`LULC_NLCD_2021.tif`) — same MD5 (`56d1080fa70576cad15896642a107a3d`),
same 297,417 bytes, confirmed identical by a deep byte compare (`filecmp`).
There is no difference to measure: per-pixel agreement is 100.0000%, and any
UNA run on one is bit-identical to the same run on the other.

This is **outcome #1** of the three the investigation set out to choose
between: *the LULCs are effectively the same — use the cooling LULC for UNA.*
In fact the finding is stronger than "effectively the same": they are
literally the same file.

---

## The two rasters

### Cooling LULC (existing prototype)

- **Path:** `data/cooling/land_use_2021.tif`
- **Config key:** `CITIES['Minneapolis, MN']['cooling_lulc_file']` =
  `land_use_2021.tif`, resolved against `data_dir_cooling` = `data/cooling`.
- This is the raster the prototype already uses for the Urban Cooling Model
  (HMI / Cooling Capacity), spatial scenario mapping, and — via the shared
  loader — every other LULC-driven metric for Minneapolis.

| Property | Value |
|---|---|
| Shape (rows × cols) | 356 × 360 |
| CRS | EPSG:26915 (NAD83 / UTM 15N) |
| dtype | int16 |
| nodata | -128 |
| Pixel size | 30 m × 30 m |
| Transform origin | (478738.77, 4979994.26) |
| Total bounds | (478738.77, 4969314.26, 489538.77, 4979994.26) |
| File size | 297,417 bytes |
| MD5 | `56d1080fa70576cad15896642a107a3d` |

### UNA sample LULC (InVEST published)

- **Path:** `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_NLCD_2021.tif`
- Ships inside the InVEST `UrbanNatureAccess_sample_data_MN` bundle. The
  bundle was already extracted in the data directory (no ZIP to open).
- Source per the bundle's `_README`: USGS NLCD 2021.

| Property | Value |
|---|---|
| Shape (rows × cols) | 356 × 360 |
| CRS | EPSG:26915 (NAD83 / UTM 15N) |
| dtype | int16 |
| nodata | -128 |
| Pixel size | 30 m × 30 m |
| Transform origin | (478738.77, 4979994.26) |
| Total bounds | (478738.77, 4969314.26, 489538.77, 4979994.26) |
| File size | 297,417 bytes |
| MD5 | `56d1080fa70576cad15896642a107a3d` |

Every property is identical — same MD5, so the rasters are the same bytes.

### LULC class distribution (identical for both)

The 356 × 360 grid is 128,160 pixels: 57,292 nodata (-128, outside the AOI)
and 70,868 valid land-cover pixels. "Nature?" follows the UNA biophysical
table (`LULC_attribute_table_UNA.csv`): a class is nature when
`urban_nature > 0`.

| Code | Description | Pixels | Nature? (`urban_nature`) |
|---:|---|---:|:---|
| -128 | (nodata, outside AOI) | 57,292 | — |
| 11 | Open Water | 3,991 | yes (1.0) |
| 21 | Developed, Open Space | 6,751 | yes (0.5) |
| 22 | Developed, Low Intensity | 19,899 | no |
| 23 | Developed, Medium Intensity | 24,780 | no |
| 24 | Developed, High Intensity | 10,638 | no |
| 31 | Barren Land | 284 | no |
| 41 | Deciduous Forest | 1,479 | yes (1.0) |
| 42 | Evergreen Forest | 25 | yes (1.0) |
| 43 | Mixed Forest | 5 | yes (1.0) |
| 52 | Shrub/Scrub | 2 | yes (1.0) |
| 71 | Herbaceous | 164 | yes (1.0) |
| 81 | Hay/Pasture | 147 | yes (0.5) |
| 90 | Woody Wetlands | 1,994 | yes (1.0) |
| 95 | Emergent Herbaceous Wetlands | 709 | yes (1.0) |

The cooling and UNA rasters produce identical counts in every row.

---

## Side-by-side comparison

### Step 1 — raster identity

| Check | Result |
|---|---|
| MD5 match | **True** |
| Byte-for-byte equal (`filecmp` deep compare) | **True** |
| Shape / CRS / transform / nodata / dtype match | True (all) |
| Class histogram match | True |

### Step 2 — per-pixel agreement

Because the two rasters share an identical grid (same shape and transform),
the overlap is the entire raster — there is no partial-extent overlap to
isolate.

| Metric | Value |
|---|---:|
| Total pixels in overlap | 128,160 |
| Pixels where the two rasters agree | 128,160 (100.0000%) |
| Pixels where they disagree | 0 (0.0000%) |

There are zero disagreements, so there is no minor-reclassification vs
major-change distribution to report — the disagreement set is empty.

### Nature-class pixels and population

Population is from the prototype's Census 2020 raster
(`data/population/minneapolis_pop_2020.tif`, 154,242 residents in the AOI),
which is grid-aligned to the LULC. Counts below are identical for both LULC
rasters because the rasters themselves are identical.

| Code | Description | Pixels | Population on class |
|---:|---|---:|---:|
| 11 | Open Water | 3,991 | 30 |
| 21 | Developed, Open Space | 6,751 | 1,530 |
| 41 | Deciduous Forest | 1,479 | 149 |
| 42 | Evergreen Forest | 25 | 1 |
| 43 | Mixed Forest | 5 | 0 |
| 52 | Shrub/Scrub | 2 | 0 |
| 71 | Herbaceous | 164 | 5 |
| 81 | Hay/Pasture | 147 | 16 |
| 90 | Woody Wetlands | 1,994 | 1 |
| 95 | Emergent Herbaceous Wetlands | 709 | 1 |
| | **Total nature classes** | **15,267** | **1,734** |

Nature classes are 11.9% of all pixels; only 1.1% of the population sits
directly on a nature pixel — expected for a dense downtown AOI where most
residents live on developed classes (22/23/24) and reach nature through the
search radius rather than by living on it.

---

## UNA result comparison

The originally-scoped step 3 was to run InVEST UNA **twice**, once per LULC,
and compare. Because step 1 proved the two inputs are byte-identical, InVEST
UNA was run **once** instead. The reasoning is not a shortcut, it is the
correct method: InVEST UNA is deterministic, so identical input bytes with
identical parameters produce bit-identical output. The per-LULC comparison
is therefore **Pearson r = 1.000 and MAE = 0 by construction** — and
file-level byte-identity is stronger evidence of "no difference" than two
model runs could ever be.

The single run used the parameters from `UNA_IMPLEMENTATION_NOTES.md`
(natcap.invest 3.16.2):

| Parameter | Value |
|---|---|
| `urban_nature_demand` | 16.7 m²/capita |
| `search_radius_mode` | `uniform radius` |
| `search_radius` | 800 m |
| `decay_function` | `dichotomy` |
| `aggregate_by_pop_group` | False |
| LULC attribute table | `LULC_attribute_table_UNA.csv` |
| Population raster | Census 2020 (`minneapolis_pop_2020.tif`) |
| Admin boundary | single bounding polygon from LULC extent |

### Results (identical for either LULC input)

| Metric | Value |
|---|---:|
| % population with supply ≥ demand | **46.86%** |
| Mean per-pixel `accessible_urban_nature` | 298,015.5 m² |
| Mean per-pixel `urban_nature_supply_percapita` | 89,638.69 m²/person |
| Modelable-extent population | 66,945 |
| Runtime | ~0.6 s |
| Pearson r between the two LULCs' per-pixel outputs | 1.000 (by construction) |
| Mean absolute difference per pixel | 0 (by construction) |

> **Reading the supply-per-capita figure.** The unweighted mean
> `urban_nature_supply_percapita` (89,639 m²/person) is inflated by
> low-population pixels — a pixel with a near-zero population denominator
> yields an enormous per-capita ratio. It is not a population-grounded
> number. The meaningful adequacy figure is the **46.86% of residents whose
> per-capita supply meets the 16.7 m²/capita demand**. This is the same
> caveat documented in the Phase 1 `compare_una_invest.py` notes.

---

## Conclusion and recommendation

**Outcome #1: the LULCs are effectively the same — use the cooling LULC for
UNA.**

This is the strongest possible version of outcome #1. The two candidate
rasters are not merely "the same source with the same values" — they are
the same file, byte-for-byte. The prototype's `data/cooling/land_use_2021.tif`
is a renamed copy of the InVEST UNA sample raster `LULC_NLCD_2021.tif`.

Consequently:

- Per-pixel agreement is 100.0000%; the disagreement set is empty.
- The UNA result does not depend on which of the two paths is used as input
  — the outputs are bit-identical.
- There is **no tradeoff to weigh**. Outcome #3 (the LULCs differ
  meaningfully, so the choice matters) is ruled out. The "cross-metric
  consistency vs canonical alignment" tension that outcome #3 would have
  forced does not exist here, because using the cooling LULC *is* using the
  canonical InVEST UNA sample raster.

**Recommendation: use the cooling LULC** (`city_cfg['cooling_lulc_file']` /
`state.cooling_lulc`) as the LULC input to the UNA implementation.

---

## Implications for the UNA implementation

For the forthcoming **UNA Session 1** (canonical InVEST UNA):

1. **Use the cooling LULC.** Reference it the way every other metric does —
   through `city_cfg['cooling_lulc_file']` and the loaded `state.cooling_lulc`
   array. Do **not** add the InVEST UNA sample raster as a second, parallel
   LULC input. It would be a redundant copy of a raster already loaded, and
   parallel-raster management (keeping two rasters in sync, deciding which
   one each code path reads) would be pure overhead with zero benefit.

2. **Cross-metric consistency and canonical alignment are not in tension
   here.** UNA will run on exactly the same LULC as UCM, UFR, carbon, and
   the placement strategies. For Minneapolis that LULC also happens to *be*
   the InVEST UNA sample raster, so the implementation is fully
   canonically-aligned on the LULC input at the same time. Both goals are
   satisfied by the single choice.

3. **The biophysical table and parameters are separate decisions, already
   settled.** This investigation only concerns the LULC raster. The
   `una_table_file` config key already points at the InVEST UNA biophysical
   table (`LULC_attribute_table_UNA.csv`), and the demand / radius / decay /
   aggregation parameters are chosen in `UNA_IMPLEMENTATION_NOTES.md`
   (16.7 / 800 m / dichotomy / no pop-group aggregation). Note that the
   InVEST args JSON shipped with the sample data uses *different* values
   (demand 250, radius 1000, exponential decay) — but that is a parameter
   choice, independent of and unaffected by this LULC finding.

4. **This byte-identity is Minneapolis-specific — do not generalize it.**
   The cooling LULC equals the InVEST UNA sample raster *for the MN downtown
   city only*, because the prototype's MN data was seeded from InVEST sample
   bundles. For **San Antonio** and **Minneapolis Full** there is no InVEST
   UNA sample LULC at all — those cities use NLCD clipped to their own
   bounding boxes (`land_use_2021_sa.tif`, `lulc_nlcd_2021_mpls_full.tif`).
   So when UNA is extended beyond MN downtown, the cooling LULC is not just
   the recommended input — it is the **only** input. Building the UNA
   implementation on `state.cooling_lulc` from the start makes it portable
   across all three cities with no per-city LULC-sourcing branch.

---

## Method notes

- **Script:** `compare_una_lulc.py` (new; pure investigation, no shipped-code
  change, no `SCENARIO_SCHEMA_VERSION` bump).
- Raster inspection and the byte/MD5 compare use `rasterio` and `filecmp`.
- The InVEST UNA run mirrors the pattern in `compare_una_invest.py`
  (Streamlit-free; serialize inputs to a temp workspace; single bounding
  polygon for the admin boundary since per-pixel outputs do not depend on
  admin geometry).
- Run with the anaconda-base `python3` (natcap.invest 3.16.2) — the same
  environment `compare_una_invest.py` and `compare_ucm_invest.py` use.
- `verify_baselines.py` was confirmed to pass unchanged after this work
  (no shipped code was touched).

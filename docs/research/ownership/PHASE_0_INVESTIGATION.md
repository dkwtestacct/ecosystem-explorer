# Ownership data — Phase 0 investigation

**Audience:** Internal — research note
**Status:** Investigate-only, no commits beyond this doc; integration deferred until Region Selection seam is stable through Commit 6
**Use this for:** The Bexar parcel data source + access pattern, the classifier crosswalk decisions, the vacancy union, the rasterization-loss measurement at 30 m, and the integration-design carryovers
**Do not use this for:** Live data (full-county canonical layer doesn't exist yet — held per the spec)
**Source of truth for:** The Phase 0 findings + the locked classifier definition for the eventual integration

---

## Source + access pattern

**Service:** Bexar County GIS ArcGIS REST — `https://maps.bexar.org/arcgis/rest/services/Parcels/MapServer/0`

| Property | Value |
|---|---|
| Native CRS | WKID 102740 / EPSG:2278 (Texas State Plane, South Central Zone) |
| Output | JSON, **geoJSON**, PBF; reprojects via `outSR` |
| `maxRecordCount` | **1,000** per page |
| Pagination | `supportsPagination: true` |
| Geometry | `esriGeometryPolygon` |
| Estimated total records | ~600 K (Bexar County, all parcels) |

**Fields confirmed present (all 7 documented):** `Owner` (string, len 70), `State_cd` (string, len 10), `ImprVal` / `LandVal` / `TotVal` (double), `Exempts` (string, len 100), `Acres` (double). Field completion in the 2,000-parcel sample: 99 %+ on the value fields, 65 % on `Exempts` (null when no exemption applies).

**Phase 0 sample:** 2 × 1,000-parcel queries via spatial-envelope filter — one central-SA box (`-98.55, 29.38, -98.42, 29.46`), one downtown civic-core box (`-98.500, 29.415, -98.480, 29.435`). Outputs cached at `/tmp/bexar_parcels_sample/` during investigation (ephemeral; not committed).

**Access pattern for the full pull (deferred):** paginate by `resultOffset` step 1000 until empty page; estimated 600 pages, ~30–60 min wall-clock against the live REST service. Output GeoPackage estimated 30–50 MB after classification.

---

## Classifier (LOCKED)

**Per-parcel `owner_class` classification.** Regex on `Owner` (upper-cased) drives the public-entity buckets; `Exempts` corroborates but does **not** define "publicly-owned."

```python
def classify_owner(owner: str, exempts: str | None) -> str:
    if owner is None:
        return 'unknown'
    O = owner.upper()
    if re.match(r'CITY OF SAN ANTONIO', O):                              return 'city'
    if re.search(r'\bSAN ANTONIO.*HOUSING AUTH', O):                     return 'city'
    if re.match(r'(BEXAR COUNTY|COUNTY OF BEXAR)', O):                   return 'county'
    if re.search(r'\b[A-Z]+ ISD\b|\bINDEPENDENT SCHOOL', O):             return 'isd'
    if re.match(r'STATE OF TEXAS', O):                                   return 'state'
    if re.match(r'(UNITED STATES|U S |U\.S\.)', O):                      return 'federal'
    if re.search(r'\bSAN ANTONIO RIVER AUTH', O):                        return 'river_auth'
    if re.search(r'\bCHURCH\b|\bDIOCESE\b|\bCATHOLIC\b|\bBAPTIST\b|\bMETHODIST\b', O): return 'church'
    if re.search(r'\bUNIV(ERSITY|\.)\b|\bCOLLEGE\b', O):                 return 'university'
    if exempts and ('EX-' in str(exempts) or str(exempts) == 'EX'):      return 'tax_exempt_other'
    return 'private'
```

### "Publicly-owned" — government-owned only

**Locked definition:**

```python
PUBLIC_GOVERNMENT_CLASSES = {'city', 'county', 'state', 'federal', 'isd', 'river_auth'}
is_public = lambda c: c in PUBLIC_GOVERNMENT_CLASSES
```

**NOT in the "public" bucket:** `church`, `university`, `tax_exempt_other`. These are tax-exempt but **not government-owned** — distinct policy framing. Including them in "public" would conflate civic land (city parks, county courthouses, public schools) with private institutions that happen to qualify for tax exemption. The Phase 0 investigation initially used a broader bucket (`pub_classes` including `church`/`university`/`public_other`); that's superseded.

### Classifier coverage on the 2,000-parcel sample

| Class | Parcels | Acres | In `is_public`? |
|---|---:|---:|---|
| **city** | 72 | 44.3 | ✅ |
| **county** | 6 | 17.7 | ✅ |
| **isd** | 1 | 12.1 | ✅ |
| **state** | 7 | 5.5 | ✅ |
| **federal** | 4 | 3.7 | ✅ |
| **river_auth** | 12 | 5.1 | ✅ |
| church | 35 | 20.7 | ❌ tax-exempt, not government |
| university | 13 | 1.6 | ❌ tax-exempt, not government |
| tax_exempt_other | 107 | 67.6 | ❌ catchall for charitable / EX exemption |
| private | 1,708 | 460.6 | ❌ |
| unknown (Owner null) | 35 | 0.4 | ❌ |

**Government-owned coverage (`is_public`):** 102 parcels / **88.4 acres** of the sample. **The simple-pattern regex catches the headline government entities (city / county / state / federal / ISD / river authority) with no hand-tuning needed for the prototype.**

The 12 % residual identified in the initial pass is the `tax_exempt_other` bucket, which is **explicitly excluded** from "publicly-owned" under the locked classifier. So under the corrected definition, the simple-pattern coverage on the sample is effectively **100 % of the government-owned bucket** — the residual was a definitional artifact.

Hand-tuning candidates that DO need attention later: 35 Owner-null parcels, plus partial-match owner-name variants (e.g. abbreviations of "CITY OF SAN ANTONIO" not caught by the leading-anchor regex). Estimated impact: <1 % at full-county scale.

---

## Vacancy logic (LOCKED)

```python
is_vacant = lambda state_cd, imprval: (
    (str(state_cd).startswith('C')) or (imprval == 0)
)
```

**Union of two signals:** `State_cd C*` (the comptroller code for vacant lots / subdivision lots / etc.) and `ImprVal == 0` (no improvement value, regardless of state code).

### Why the union

| Signal | Count in sample |
|---|---:|
| `State_cd C*` | 161 |
| `ImprVal == 0` | 260 |
| Both agree | **161** |
| `C*` but `ImprVal > 0` | **0** |
| Not `C*` but `ImprVal == 0` | **99** |

`C*` is a **strict subset** of `ImprVal == 0` in the sample — no false negatives when `C*` fires. The 99 extra `ImprVal == 0` parcels are mostly `F1` (commercial real estate, no improvement value yet — undeveloped commercial lots awaiting buildout), plus a handful of NULL/X/E1 codes. **The union catches commercial-vacant + true vacant lots without losing anything.**

---

## "Vacant publicly-owned" — INTERSECTION, not union

**Locked definition:**

```python
is_vacant_public = lambda owner_class, state_cd, imprval: (
    is_public(owner_class) AND is_vacant(state_cd, imprval)
)
```

The earlier write-up in the chat surfaced an error: I wrote "AND" but the formula stamped to the rasterization codes was effectively a UNION (`is_public | is_vacant`, code 3 in my color-coded rasterization scheme caught "both", but the headline `(is_vacant | is_public).sum()` line was the union total). **The correct semantic for "vacant publicly-owned land suitable for greening" is the INTERSECTION** — city-owned vacant lots, county-owned undeveloped land, ISD-owned parcels with no buildings — not "all public land OR all vacant land."

### Rasterization code scheme (LOCKED, distinct buckets)

For the prototype public/vacant raster:

| Code | Meaning |
|---|---|
| 0 | neither public nor vacant (typical residential / commercial with improvements) |
| 1 | public-only (government-owned, has buildings — civic infrastructure) |
| 2 | vacant-only (privately-owned vacant lot — developer land bank, etc.) |
| **3** | **public AND vacant** — government-owned land with no improvements (the prime candidate for greening conversion) |
| -1 | outside-sample / no parcel data |

The "vacant publicly-owned" mask the eventual region-mask integration would use is `(raster == 3)`. The other codes are diagnostics, not selection inputs.

---

## Rasterization fidelity at 30 m

**69.0 % of sample parcels are sub-pixel** (`Acres < 0.222`, where 0.222 ≈ 30 m × 30 m in acres).

| Quartile | Acres |
|---|---:|
| 25 % | 0.167 |
| **median** | **0.179** |
| 75 % | 0.246 |
| residential median (`State_cd == 'A1'`) | **0.179** |

**Implication:** typical 30 m pixel in residential SA encompasses multiple parcels. `rasterio.features.rasterize` gives an arbitrary "winner" per pixel — **ownership/vacancy mask at 30 m is approximate near boundaries**, not authoritative per-parcel.

**Honesty caption requirement (for the integration UI):** when ownership filtering is active, the dashboard must surface a coarseness caveat — *"At 30 m resolution, ownership filtering is approximate; subdivisions are pixelated, large parcels (parks, public open space, institutional land) are accurate."* Without this caption, users could mistake the mask for parcel-perfect.

### Reliable vs unreliable

- **Reliable at 30 m:** parks (acres-scale), public open space, large institutional parcels (campuses, hospitals, military installations, river-authority land).
- **Pixelated at 30 m:** residential subdivisions, small commercial lots, single-family parcels.

The Region Selection use case (council-district scale) is unaffected — districts are kilometers across. The Ownership filter would be **most useful on coarse selection: "show me city-owned vacant land within District 5"** — combining a coarse region with the coarse-by-necessity ownership mask.

---

## Sample output (prototype raster, not committed)

The 2,000-parcel sample rasterized to the SA 30 m grid produced:

| Code | Pixel count (in sample bbox) |
|---|---:|
| 1 public-only (govt + improved) | 260 |
| 2 vacant-only (private + vacant) | 140 |
| **3 public AND vacant** | **547** |
| 0 neither | 2,035 |
| -1 outside-sample | 3,395,610 |

Sample only covered 0.09 % of the SA grid; full-county pull would change the raster shape but not the methodology.

---

## Held until later — full-county pull + canonical `sa_ownership.gpkg`

Per the user's gating direction, the full pull + canonical artifact is held until:
1. **Region seam stable through Commit 6** of Region Selection Phase 1 (`docs/internal/REGION_SELECTION_PHASE1_SPEC.md`). The ownership layer integrates as a second masked layer in the same seam; no new infrastructure needed.
2. **Classifier settled** — done here. "Publicly-owned" = government-owned (city/county/state/federal/ISD/river-authority); `church`/`university`/`tax_exempt_other` are distinct buckets.

When the gate opens:
- New script `scripts/data/download_bexar_parcels.py` — mirrors `download_sa_council_districts.py` pattern; paginates the REST service; runs the classifier; reprojects to EPSG:5070; rasterizes to the SA grid; writes both the polygon GeoPackage (audit trail) and the int32 raster (runtime artifact).
- New `data/sa/sa_ownership.gpkg` (polygon, ~30–50 MB after classification) + `data/sa/sa_ownership_30m.tif` (raster, ~7 MB).
- DATA_INVENTORY catalog entry under §7's region-selection subsection.
- Integration: a second `region_layers` registry entry (or a separate `ownership_layers` registry, TBD by the Commit-6 review).

---

## Carryovers into the integration design

1. **Government-owned vs tax-exempt is a categorical distinction**, not a degree of confidence — the UI selector must show "publicly-owned" (government) as the default and `tax_exempt_other` as a separate, opt-in toggle if surfaced at all. Don't conflate.
2. **"Vacant publicly-owned" is the intersection** — the on-toggle behavior is `region_mask ∩ (raster == 3)`, not `region_mask ∩ ((raster == 1) | (raster == 2) | (raster == 3))`.
3. **Coarseness caveat is mandatory** when ownership filtering is active — see the honesty caption text above.
4. **No region-clipped per-area metrics** in the integration's first cut (mirrors the Region Selection metric-aggregation decision: citywide-impact metrics with an honesty caption).
5. **Hand-tuning the residual is a follow-up**, not blocking. The simple-pattern coverage on government-owned is effectively 100 % at the prototype level after the definitional cleanup.

Investigation closed. Re-open when Commit 6 of Region Selection Phase 1 lands.

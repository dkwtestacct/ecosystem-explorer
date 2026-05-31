# Ownership data — Phase 0 investigation

**Audience:** Internal — research note
**Status:** Closed; full-county pull landed at Commit 0.5 of the Ownership Integration build (2026-05-31)
**Use this for:** The Bexar parcel data source + access pattern, the classifier crosswalk decisions, the locked vacancy methodology (exempt-keyed), the rasterization-loss measurement at 30 m, and the full-county classifier coverage
**Do not use this for:** Build sequence + UI design (→ `docs/internal/OWNERSHIP_INTEGRATION_SPEC.md`)
**Source of truth for:** The locked classifier + vacancy methodology, the empirical evidence behind them, and the full-county coverage numbers

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
| Actual total records (2026-05-31 pull) | **710,772** across 711 pages; EOD-confirmed (terminal page n=772) |

**Fields confirmed present (all 7 documented):** `Owner` (string, len 70), `State_cd` (string, len 10), `ImprVal` / `LandVal` / `TotVal` (double), `Exempts` (string, len 100), `Acres` (double). Field completion in the 2,000-parcel sample: 99 %+ on the value fields, 65 % on `Exempts` (null when no exemption applies).

**Phase 0 sample:** 2 × 1,000-parcel queries via spatial-envelope filter — one central-SA box (`-98.55, 29.38, -98.42, 29.46`), one downtown civic-core box (`-98.500, 29.415, -98.480, 29.435`). Outputs cached at `/tmp/bexar_parcels_sample/` during investigation (ephemeral; not committed).

**Access pattern (executed 2026-05-31):** paginate by `resultOffset` step 1000 until the terminal short page (n<1000). The Phase A fetch loop is resumable (skips pages already on disk) and the Phase B verify pass scans three completeness signals — MISSING offsets, SHORT pages (truncation), and EOD (terminal marker) — refetching anything that fails. The 2026-05-31 pull surfaced 1 MISSING page (offset 380000, recovered cleanly) and 1 mid-fetch socket DNS error (offset 701000, recovered on attempt 4); EOD confirmed at offset 710000 with n=772. Total wall-clock: ~22 min. Output GeoPackage = **281 MB** (above the 30–50 MB Phase 0 estimate because BCAD returns ~710 K parcels, ~18 % more than the ~600 K county-rough estimate).

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

### Classifier coverage at full county (2026-05-31, n = 710,772)

| Class | Parcels | % | In `is_public`? |
|---|---:|---:|---|
| **city** | 4,741 | 0.67% | ✅ |
| **county** | 914 | 0.13% | ✅ |
| **isd** | 463 | 0.07% | ✅ |
| **state** | 386 | 0.05% | ✅ |
| **federal** | 231 | 0.03% | ✅ |
| **river_auth** | 337 | 0.05% | ✅ |
| church | 1,760 | 0.25% | ❌ tax-exempt, not government |
| university | 229 | 0.03% | ❌ tax-exempt, not government |
| tax_exempt_other | 5,773 | 0.81% | ❌ catchall for charitable / EX exemption |
| private | 687,582 | 96.74% | ❌ |
| unknown (Owner NULL) | 8,356 | 1.18% | ❌ |

**Government-owned coverage (`is_public`):** **7,072 parcels (0.99 % by count)**. Audit signals at full scale:

- The 8,356 `unknown` parcels all have NULL `Owner` — no populated-name slip-through.
- Top 30 `private` owners by parcel count are unambiguously private (Continental Homes 2,439, Meritage 1,326, Pulte 851, Lennar 608, KB Home 555, M/I Homes 485, etc. — homebuilders the whole way down).
- A sweep of `private`-classified rows for suspicious tokens (HOSPITAL / AUTH / TRANSIT / UTIL / AIRPORT / VIA / CPS / SAWS) surfaced 273 hits; manual inspection confirmed every sampled one was a regex false positive ("HOSPITALITY" → "HOSPITAL", "GAUTHIER" → "AUTH"), **not** a classifier miss.

The 0.99 % gov-by-count is low but plausible — Bexar County is dominated by ~688 K residential subdivisions; public parcels concentrate as larger tracts (school campuses, county facilities, parks). Public-by-area lands at ~9 % of in-coverage pixels (see §raster). Hand-tuning the 8,356 NULL-owner parcels is a follow-up; <1.2 % impact, not blocking.

---

## Vacancy logic (LOCKED — exempt-keyed)

```python
is_totally_exempt = any token in Exempts matches EX-X[A-Z]
is_vacant = (str(state_cd).startswith('C')) or (~is_totally_exempt and imprval == 0)
```

**The discriminator is the total-exemption flag, not gov-vs-private.** Totally tax-exempt parcels carry `ImprVal == 0` because improvements aren't *assessed*, NOT because the land is empty. Applying the naive union (`C* OR ImprVal == 0`) to exempt parcels over-catches built civic land (schools, courthouses, churches, university buildings) as "vacant." Keying on the exemption flag fixes the over-catch symmetrically across all institutional owners.

### Empirical evidence — the EX-X* family marks "improvements unassessed"

For each `Exempts` token (parsed as the comma-separated set from the field), what fraction of parcels carrying that token have `ImprVal == 0` AND State_cd is NOT C* (i.e. the parcel is in a built category but reports zero improvement value)? High pct = "improvements unassessed"; low pct = "parcel still assessed despite the exemption."

| Token | n parcels | n built + ImprVal=0 | **% suspect** | Cluster |
|---|---:|---:|---:|---|
| EX-XI | 23 | 7 | **30.4 %** | total |
| EX-XD | 56 | 15 | **26.8 %** | total |
| EX-XJ | 253 | 53 | **20.9 %** | total |
| EX-XV | 13,803 | 2,389 | **17.3 %** | **total (the bulk)** |
| EX-XU | 108 | 17 | **15.7 %** | total |
| DVHS (100% disabled vet) | 32,881 | 128 | 0.4 % | partial |
| HS (homestead) | 405,782 | 708 | 0.2 % | partial |
| OV65 | 154,162 | 178 | 0.1 % | partial |
| DV4 (90-100% disabled vet) | 35,418 | 46 | 0.1 % | partial |

Two clean clusters: `EX-X[A-Z]` codes at 15–30 % built-but-zero (total exemption — parcel unassessed), and partial-exemption codes (HS, OV65, DV*, DP, etc.) at <0.5 %. Other token families (`EX-XG`, `EX-XL`, `EX-XR`) carry too few parcels for a meaningful suspect rate but follow the same naming pattern and are included for safety. The locked rule keys on `EX-X[A-Z]` and excludes homestead / partial-exemption codes — homestead residences are tax-exempt-partial but ARE assessed, so they correctly stay subject to the taxed-rule branch.

### Who the discriminator catches

`EX-XV` (the largest total-exemption token, 13,803 parcels) is carried by:

| owner_class | n with EX-XV |
|---|---:|
| tax_exempt_other | 5,388 |
| city | 4,725 |
| church | 1,410 |
| county | 801 |
| isd | 461 |
| state | 382 |
| river_auth | 334 |
| federal | 225 |
| university | 77 |

No `private` parcel carries `EX-XV`. So switching the discriminator from gov-vs-private to exempt-vs-taxed catches **exactly the right additional bucket** (church, university, tax_exempt_other) without false moves into `private`.

---

## "Vacant publicly-owned" — INTERSECTION (locked)

```python
is_vacant_public = is_public(owner_class) AND is_vacant(state_cd, imprval, exempts)
```

"Vacant publicly-owned land suitable for greening" is the intersection — city-owned vacant lots, county-owned undeveloped land, ISD-owned parcels with no buildings — not the union.

### Rasterization code scheme (LOCKED, distinct buckets)

| Code | Meaning |
|---|---|
| 0 | neither public nor vacant (typical residential / commercial with improvements) |
| 1 | public-only (government-owned, has buildings — civic infrastructure) |
| 2 | vacant-only (taxed parcel with `C*` State_cd or `ImprVal == 0`) |
| **3** | **public AND vacant** — government-owned land flagged vacant by the exempt-keyed rule (the prime candidate for greening conversion) |
| -1 | outside parcel coverage |

The "vacant publicly-owned" mask the integration uses is `(raster == 3)`. The other codes are diagnostics, not selection inputs.

---

## Rasterization fidelity at 30 m

**69.6 % of parcels are sub-pixel at full county** (`Acres < 0.222`, where 0.222 ≈ 30 m × 30 m in acres; n = 703,709 with Acres > 0).

| Quartile | Acres |
|---|---:|
| 25 % | 0.138 |
| **median** | **0.172** |
| 75 % | 0.250 |

**Implication:** typical 30 m pixel in residential SA encompasses multiple parcels. `rasterio.features.rasterize` gives an arbitrary "winner" per pixel — **ownership/vacancy mask at 30 m is approximate near boundaries**, not authoritative per-parcel.

**Honesty caption requirement (for the integration UI):** when ownership filtering is active, the dashboard must surface a coarseness caveat — *"At 30 m resolution, ownership filtering is approximate; subdivisions are pixelated, large parcels (parks, public open space, institutional land) are accurate."* Without this caption, users could mistake the mask for parcel-perfect.

### Reliable vs unreliable

- **Reliable at 30 m:** parks (acres-scale), public open space, large institutional parcels (campuses, hospitals, military installations, river-authority land).
- **Pixelated at 30 m:** residential subdivisions, small commercial lots, single-family parcels.

The Region Selection use case (council-district scale) is unaffected — districts are kilometers across. The Ownership filter would be **most useful on coarse selection: "show me city-owned vacant land within District 5"** — combining a coarse region with the coarse-by-necessity ownership mask.

---

## Full-county rasterized output (2026-05-31)

Rasterized to the SA 30 m grid (1713 × 1984, EPSG:5070):

| Code | Class | Pixels | % of in-coverage | Acres (≈) |
|---|---|---:|---:|---:|
| -1 | outside parcel coverage | 695,390 | — | — |
| 0 | neither public nor vacant | 1,631,164 | 60.3 % | 362,750 |
| 1 | public only (govt-built civic) | 195,374 | 7.2 % | 43,440 |
| 2 | vacant only (taxed + unimproved) | 823,391 | 30.5 % | 183,090 |
| **3** | **public AND vacant — the actionable headline** | **53,273** | **2.0 %** | **≈ 11,840** |

Parcel-count totals: 7,072 public (0.99 %), 80,005 vacant (11.26 %), **5,331 public AND vacant (0.75 %)**.

Public-by-area (codes 1+3 = 9.2 % of in-coverage) is meaningfully higher than public-by-count (0.99 %) because public parcels are systematically larger (school campuses, county facilities, parks). Vacant-by-area (codes 2+3 = 32.4 %) is similarly higher than vacant-by-count (11.3 %) — vacant parcels skew toward undeveloped tracts. Both ratios are plausible.

---

## Artifacts

| Artifact | Status |
|---|---|
| `scripts/data/download_bexar_parcels.py` | Committed at 0.5 (paginated fetch + verify-completeness + classify + rasterize, two-phase resumable) |
| `data/sa/sa_public_vacant_30m.tif` (~ 396 KB) | **Committed at 0.5** — runtime artifact (codes -1/0/1/2/3 on the SA grid) |
| `data/sa/sa_ownership.gpkg` (~281 MB polygon audit trail) | **Gitignored** — too heavy for Streamlit Cloud deploy weight. Archived outside the repo at `~/Desktop/ecosystem_explorer_archive/sa_ownership_bexar_2026-05-31.gpkg`; re-pull from BCAD via the script (~22 min wall) if a future parcel drill-down needs it |

---

## Carryovers into the integration design

1. **The discriminator for vacancy is exempt-vs-taxed, not gov-vs-private.** Captures gov + church + university + tax_exempt_other built parcels symmetrically. The locked rule is `EX-X[A-Z]` token match (verified via empirical per-token cluster above).
2. **"Vacant publicly-owned" is the intersection** — the on-toggle behavior is `region_mask ∩ (raster == 3)`, not `region_mask ∩ ((raster == 1) | (raster == 2) | (raster == 3))`.
3. **Government-owned vs tax-exempt is a categorical distinction** in the UI, not a degree of confidence — show "publicly-owned" (government) as the default; `tax_exempt_other` is a separate bucket if ever surfaced.
4. **Coarseness caveat is mandatory** when ownership filtering is active — see the honesty caption text above.
5. **No region-clipped per-area metrics** in the integration's first cut (mirrors the Region Selection metric-aggregation decision: citywide-impact metrics with an honesty caption).
6. **Hand-tuning candidates** (follow-up, non-blocking): the 8,356 NULL-`Owner` parcels (~1.2 %); the `EX-XG` / `EX-XL` / `EX-XR` small-count total-exemption variants (already caught by `EX-X[A-Z]`, no action needed); partial-match owner-name variants below the prefix-anchor regex.

Investigation closed; integrate per `OWNERSHIP_INTEGRATION_SPEC.md`.

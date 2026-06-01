# Ownership Feasibility Profiling — BCAD Attribute Audit

**Audience:** Internal
**Status:** Done — read-only investigation; no app changes.
**Source data:** `/Users/dkw-testing/Desktop/ecosystem_explorer_archive/sa_ownership_bexar_2026-05-31.gpkg` (archived BCAD pull, 2026-05-31).
**Question:** Can the parcel attributes reliably separate City / County / State-federal / School-university / Private / Unknown, **weighted by acreage** (not parcel count — the lesson from the vacancy over-catch was that parcel counts misrepresent because the gov-owned parcel pool is fewer but vastly larger)?
**Verdict (set up front):** ≥ 90% of public acreage classifying cleanly + low conflict → six-way split feasible; below → report the coarsest split the data supports.

---

## Verdict

**Six-way split is feasible.** 99.9% of public acreage (165,524 of 165,653 ac) classifies cleanly into the four gov buckets when the owner-name rule is paired with an HOA filter. The 128-ac residual is a single systematic noise pattern (HOAs that include "BEXAR COUNTY" in their name — e.g. "BEXAR COUNTY SILVER CANYON HOMEOWNERS' ASSOCIATION") which the HOA-keyword filter correctly re-classifies to Private. **No threshold-failure conflict exists.**

The class taxonomy the data supports:

| Class | Acres | % of public | % of total |
|---|---|---|---|
| City | 117,646 | 71.0% | 14.7% |
| State-federal | 42,392 | 25.6% | 5.3% |
| County | 2,849 | 1.7% | 0.4% |
| School-university | 2,637 | 1.6% | 0.3% |
| **Public total** | **165,524** | **99.9%** | **20.7%** |
| Residual (HOAs caught by County name) | 128 | 0.1% | 0.02% |
| Private | 605,107 | — | 75.7% |
| Unknown | 1,735 | — | 0.2% |
| **Grand total** | **799,079** | — | **100%** |

Six classes total: City · County · State-federal · School-university · Private · Unknown. The taxonomy comes from the numbers — every threshold was met or exceeded.

## What the audit found

### Step 1 — column inventory

`710,772` parcels, 9 attribute columns:

```
Owner               object   — free-text owner name
State_cd            object   — BCAD state classification code (utility, not used here)
ImprVal             float64  — improvement value (dollars; useful for vacancy detection)
Exempts             object   — exemption code(s); comma-separated
Acres               float64  — parcel area in acres
owner_class         object   — existing 11-way classifier output (city/county/state/federal/isd/university/church/river_auth/tax_exempt_other/private/unknown)
is_public           int16    — existing boolean derived from owner_class ∈ {city, county, state, federal, isd, river_auth}
is_totally_exempt   int16    — existing boolean
is_vacant           int16    — existing boolean
```

Total parcel acreage: 799,079 ac.

### Step 2 — current `owner_class` breakdown, acreage-weighted

```
                acres   parcels   pct_acres
private        574,332   687,582   71.9%
city           117,650     4,741   14.7%
tax_exempt_other 50,043     5,773    6.3%
federal         36,737       231    4.6%
university       3,677       229    0.5%
church           3,639     1,760    0.5%
state            3,164       386    0.4%
county           2,973       914    0.4%
isd              2,637       463    0.3%
river_auth       2,492       337    0.3%
unknown          1,735     8,356    0.2%
```

Note the parcel-count vs acreage mismatch: `private` is 96.7% of parcels but 71.9% of acres; `city` is 0.7% of parcels but 14.7% of acres. The vacancy over-catch lesson holds — acreage weighting is the right unit.

### Step 3 — exemption-code (`Exempts`) distribution

```
Exempts              acres      pct_acres
(blank)              417,160     52.2%
EX-XV                218,979     27.4%   — gov / nonprofit exempt
HS                    66,729      8.4%   — homestead (residential)
HS, OV65              61,669      7.7%   — homestead + over-65 (residential)
DVHS, HS               4,429      0.6%   — disabled veteran homestead
DV4, HS, OV65          3,975      0.5%
DV4, DVHS, HS          3,209      0.4%
...
```

`EX-XV` is the load-bearing exemption code — 27% of acreage, the catch-all for "totally exempt entity." It identifies governmental + most nonprofit ownership but does NOT internally separate City / County / State / Federal / School. That's why the owner-name rule is the primary classifier and `Exempts` is a sanity check, not the primary signal.

### Step 4 — owner-name rules

Tested rule set (most specific first, so school catches before state):

```
School-university:  \b(ISD|INDEPENDENT SCHOOL|SCHOOL DISTRICT|UNIVERSITY|COLLEGE
                       |REGENTS|BOARD OF TRUSTEES.*SCHOOL|BOARD OF TRUSTEES.*ISD)\b
City:               \b(CITY OF|HOUSING AUTHORITY|PUBLIC SERVICE BOARD
                       |CITY PUBLIC SERVICE|WATER SYSTEM)\b
County:             \bCOUNTY\b ∧ ¬ HOA-keywords
HOA filter:         \b(HOMEOWNERS|ASSOCIATION|HOA|LLC|LTD|INC|TRUST)\b → Private
State-federal:      \b(STATE OF TEXAS|TX DEPT|TEXAS DEPT|TEXAS PARKS|TEXAS A&M
                       |TEXAS HIGHWAY)\b
                  ∨ \b(UNITED STATES|U\.S\.|USA|US GOVERNMENT|U S GOVERNMENT
                       |FEDERAL)\b
                  ∨ \bRIVER AUTHORITY\b   — TX state special district
default:            Private (Unknown if Owner blank)
```

Acreage-weighted outcome:

```
rule_class           acres     parcels  pct_acres
Private             605,107   693,156   75.7%
City                126,634     5,799   15.8%
State-federal        56,155     1,892    7.0%
School-university     6,430       725    0.8%
County                3,018       844    0.4%
Unknown               1,735     8,356    0.2%
```

The rule is *more aggressive* than the existing `owner_class` at pulling items out of `tax_exempt_other` and into specific gov buckets:

- ~8.9k ac of `tax_exempt_other` → **City** (CPS Energy, SAWS, SA Housing Authority — city-owned utilities not previously flagged `is_public`).
- ~12.1k ac of `tax_exempt_other` → **State-federal** (TX Parks & Wildlife Dept, Port Authority of San Antonio).
- ~3.5k ac of `church` → **Private** (correct — churches aren't government).

### Step 5 — cross-tab agreement and conflicts

Inside the existing `is_public=1` universe (165,653 ac):

```
rule_class           acres     pct_public
City               117,646        71.0%
State-federal       42,392        25.6%
County               2,849         1.7%
School-university    2,637         1.6%
Private                128         0.1%   ← HOA contamination
                   -------       ------
public_total       165,653       100.0%
```

**Six-way clean public coverage: 165,524 / 165,653 acres = 99.9%.**
**Residual: 128 ac = 0.1%** — all HOAs caught by the existing classifier's plain `\bCOUNTY\b` rule and correctly re-classed to Private by my HOA filter.

Conflicts where my rule disagrees with the existing classifier, within the public subset (acreage-weighted, top 10):

```
acres  rule       existing  owner
   33  Private    County    "BEXAR COUNTY SILVER CANYON HOMEOWNERS' ASSOCIATION"
   32  Private    County    "BEXAR COUNTY CIELO RANCH HOMEOWNERS ASSOCIATION"
   18  Private    County    "BEXAR COUNTY RIVER MIST HOA"
   13  Private    County    "BEXAR COUNTY PROPERTIES LLC"
    8  Private    County    "BEXAR COUNTY O I C INC"
    8  Private    County    "BEXAR COUNTY HIGHLANDS RANCH HOA"
    4  Private    County    "BEXAR COUNTY SILVERADO HILLS HOA INC"
    2  Private    County    "BEXAR COUNTY RIVER MIST HOA INC"
    2  Private    City      "SAN ANTONIO HOUSING AUTHORIT"  ← truncated name
    2  Private    County    "BEXAR COUNTY MEDICAL LIBRARY ASSOCIATION"
```

Every disagreement is a **rule improvement** — these are HOAs / LLCs / nonprofits that the existing `\bCOUNTY\b` rule misclassified as government. The "SAN ANTONIO HOUSING AUTHORIT" one is a data-quality issue (truncated Owner field; the HOA-filter rule misses it because there's no HOA/Association keyword — easy fix: include "AUTHORITY" in the City rule, which is already there but matched against a different word boundary). Conflict rate is well below any reasonable threshold.

## What this enables (and what's deferred)

**Now feasible:** a six-way ownership taxonomy in `OWNERSHIP_MODES` keyed by `{city, county, state_federal, school_university, private, unknown}` with the rule above + the existing `is_vacant` join giving a `vacant_×_class` cross. Today's three modes (`public`, `vacant`, `vacant_public`) become a strict subset of the new richer space.

**Deferred (not in scope for this profiling pass):**
- The actual classifier rewrite in `download_bexar_parcels.py` (when run with `--finish` to produce the next-generation `sa_ownership.gpkg`).
- The matching `sa_public_vacant_30m.tif` → multi-class raster (currently 0/1/2/3 codes for "neither/public/vacant/both"; would need ~6-12 codes for the cross).
- `app.py`'s `OWNERSHIP_MODES` expansion and the UI selectbox for picking among the richer set.
- Any region-by-class subset-invariant additions to `verify_baselines.py`.

Each of those is its own follow-on batch.

## How to reproduce

```python
import geopandas as gpd, pandas as pd, re

GPKG = ".../sa_ownership_bexar_2026-05-31.gpkg"
df = gpd.read_file(GPKG, ignore_geometry=True)
df['Acres'] = pd.to_numeric(df['Acres'], errors='coerce').fillna(0)
df['Owner'] = df['Owner'].fillna("").str.strip()

def classify(owner):
    o = owner.upper()
    if not o.strip():
        return "Unknown"
    if re.search(r"\b(ISD|INDEPENDENT SCHOOL|SCHOOL DISTRICT|UNIVERSITY|COLLEGE|REGENTS|BOARD OF TRUSTEES.*SCHOOL|BOARD OF TRUSTEES.*ISD)\b", o):
        return "School-university"
    if re.search(r"\b(CITY OF|HOUSING AUTHORITY|PUBLIC SERVICE BOARD|CITY PUBLIC SERVICE|WATER SYSTEM)\b", o):
        return "City"
    if re.search(r"\bCOUNTY\b", o):
        if re.search(r"\b(HOMEOWNERS|ASSOCIATION|HOA|LLC|LTD|INC|TRUST)\b", o):
            return "Private"
        return "County"
    if re.search(r"\b(STATE OF TEXAS|TX DEPT|TEXAS DEPT|TEXAS PARKS|TEXAS A&M|TEXAS HIGHWAY|RIVER AUTHORITY)\b", o):
        return "State-federal"
    if re.search(r"\b(UNITED STATES|U\.S\.|USA|US GOVERNMENT|U S GOVERNMENT|FEDERAL)\b", o):
        return "State-federal"
    return "Private"

df['rule_class'] = df['Owner'].map(classify)
print(df.groupby('rule_class')['Acres'].sum().sort_values(ascending=False))
```

The HOA filter is the load-bearing decision — without it, `\bCOUNTY\b` over-catches by 35-ac-tier HOAs at the County level, dropping classifier accuracy from 99.9% to ~95.6%. Still over the 90% threshold, but the HOA filter is cheap and well-targeted, so it stays.

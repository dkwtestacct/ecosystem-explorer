# UNA Divergence — Spatial Case Study Material

**Audience:** Research
**Status:** Research — concluded (led to the Nature Quality Score removal)
**Use this for:** The UNA investigation record behind that decision
**Do not use this for:** Current UNA methodology (→ REFERENCE.md) or the decision itself (→ HISTORY.md)
**Historical record of:** UNA divergence case studies

---

Raw analytical material for `UNA_METRIC_COMPARISON.md`. Identifies specific
Minneapolis locations where the prototype's reachability proxy and canonical
InVEST 2SFCA diverge most, with thorough per-location characterization. **This
is raw material, not finished prose — the narrative case studies are Deborah's
to write.**

**Date:** 2026-05-21
**Source data:** `comparisons/una_diff_baseline_mn.tif` and the underlying
rasters regenerated through the Phase 1 pipeline (`compare_una_invest.py`):
the prototype access-score raster (`_compute_access_score_raster`), InVEST
`accessible_urban_nature.tif`, `urban_nature_supply_percapita.tif`,
`urban_nature_balance_totalpop.tif`, plus the MN cooling LULC, the Census-2020
population raster, and the rasterized InVEST UFR sample buildings.
**Method:** Quadrant analysis (Approach C). Each lulc-valid pixel is classified
on two *decision* thresholds — the prototype's own Nature Access threshold
(access score > 0.3) and InVEST's per-capita demand standard
(`urban_nature_supply_percapita` ≥ 250 m²/capita). Case-study locations were
picked by greedy spatial separation (≥ 600 m apart) ranked by
population × |normalized diff|. Coordinates are exact; neighborhood names are
**not** asserted (no geocoding source was available — locations are given as
coordinates + LULC-derived context + orientation within the AOI).

---

## Summary

**The divergence is one-directional and near-total.** Of the 70,868 lulc-valid
pixels, the quadrant split is:

| Quadrant | Pixels | Population | Pop share |
|---|---:|---:|---:|
| proxy says access / InVEST says inadequate | 27,359 | 60,555 | **90.5 %** |
| both say access / adequate | 43,509 | 6,390 | 9.5 % |
| both say no access / inadequate | 0 | 0 | 0 % |
| proxy says no access / InVEST adequate | 0 | 0 | 0 % |

The prototype assigns **zero** lulc-valid pixels an access score of 0 — every
valid pixel scores 0.5 or 1.0 (within the capped 1 km radius of *some* nature
class). So the proxy never reports "no access" for anyone in the valid extent;
the only question is whether InVEST agrees the supply is adequate, and for
90.5 % of residents it does not. The two "interesting" quadrants a quadrant
analysis usually hunts for (both-poor; proxy-pessimistic / InVEST-optimistic)
are **both empty** — a finding in itself: the proxy is *never* more pessimistic
than 2SFCA.

**The disagreement is about people, not pixels.** By pixel *area*, 61 % of the
valid extent (43,509 px) clears the 250 m²/capita demand; by *population* only
9.5 % do. The two quadrants differ 15× in density — the divergent quadrant
averages 2.21 people/pixel, the agreement quadrant 0.15. Residents cluster into
exactly the dense pixels where 2SFCA's per-capita arithmetic fails, while the
population-blind proxy returns "access" uniformly.

**A population/extent mismatch sits underneath the Phase 1 headline — see
Honest Gaps.** 56.6 % of the population raster falls on pixels the cooling LULC
marks nodata, where InVEST cannot model supply at all. The original "~70 % vs
~10 %" framing compared two different population denominators; this was
reconciled on 2026-05-21 (`compare_una_invest.py` restricted-extent columns;
REFERENCE.md "Official InVEST alignment — UNA" updated).

---

## Locations

All six locations are within the lulc-valid extent (rows 44–310, cols 44–314 of
the 356 × 360 grid; CRS EPSG:26915 / UTM 15N; 30 m pixels). "500 m context" =
a ~17-pixel-radius circular window. "Implied sharing population" =
`accessible_urban_nature ÷ supply_percapita` — a derived interpretive ratio
(roughly, how many people effectively share the reachable nature), not a direct
model output.

### Location 1 — dense built core, north-central AOI (proxy says access / InVEST inadequate)

- **Coordinates:** row=84, col=55; EPSG:26915 E=480403.8 N=4977459.3; lat=44.95030, lon=−93.24842
- **What's there:** pixel is NLCD 22 (Developed, Low). 500 m window is 51 % Dev Medium, 26 % Dev High, 14 % Dev Low, 8 % Dev Open Space — almost entirely built, no large green.
- **Population:** pixel = 21.0; 500 m window = 3,836 people; density ≈ 4,730 /km².
- **Buildings:** 40 % of pixels within 500 m carry a building footprint.
- **Nature proximity:** Dev Open Space 120 m · Herbaceous 210 m · Hay/Pasture 300 m · Deciduous Forest 2,181 m · Open Water 2,469 m. Only small scattered patches are near; all "real" green (forest, water) is >2 km away.
- **Metric values:** prototype access score **1.00** · InVEST accessible_urban_nature **31,283 m²** · supply_percapita **4.1 m²/capita** (demand 250) · balance_totalpop −5,164.5 · normalized diff **+0.983** · implied sharing population ≈ 7,686.
- **Why they diverge:** a Herbaceous patch (NLCD 71, `urban_nature`=1.0) sits 210 m away — inside the 1,000 m cap — so the proxy's `max(urban_nature × in_range)` returns 1.0 and never looks at how many people are nearby. InVEST 2SFCA spreads the 31,283 m² of reachable nature across an effective ~7,700 competitors → 4.1 m²/capita, **1.6 % of the 250 m² demand**. Same location, opposite verdict.

### Location 2 — high-intensity built, north-central AOI (proxy says access / InVEST inadequate)

- **Coordinates:** row=62, col=58; EPSG:26915 E=480493.8 N=4978119.3; lat=44.95624, lon=−93.24730
- **What's there:** pixel is NLCD 23 (Developed, Medium). 500 m window is 44 % Dev Medium, 39 % Dev High, 16 % Dev Low — the most uniformly high-intensity of the six, **0 % open space**.
- **Population:** pixel = 12.9; 500 m window = 3,905 people; density ≈ 4,816 /km².
- **Buildings:** 37 % of pixels within 500 m.
- **Nature proximity:** Dev Open Space 277 m · Herbaceous 725 m · Hay/Pasture 767 m · Deciduous Forest 2,101 m · Open Water 2,154 m.
- **Metric values:** prototype access score **1.00** · InVEST accessible_urban_nature **22,900 m²** · supply_percapita **3.3 m²/capita** · balance_totalpop −3,176.4 · normalized diff **+0.990** · implied sharing population ≈ 6,958.
- **Why they diverge:** this is the starkest "reachable but trivial" case — only 22,900 m² of nature is within range at all (a thin sliver of distant patches), and 2SFCA divides it down to **3.3 m²/capita (1.3 % of demand)**. The proxy still returns 1.0 because *something* with `urban_nature`=1.0 is within 1,000 m. The proxy cannot express "reachable but nowhere near enough."

### Location 3 — moderate-density built, east-central AOI (proxy says access / InVEST inadequate)

- **Coordinates:** row=96, col=108; EPSG:26915 E=481993.8 N=4977099.3; lat=44.94710, lon=−93.22825
- **What's there:** pixel is NLCD 22 (Developed, Low). 500 m window is 41 % Dev Medium, 34 % Dev Low, 25 % Dev High.
- **Population:** pixel = 8.9; 500 m window = 2,576 people; density ≈ 3,177 /km².
- **Buildings:** 36 % of pixels within 500 m.
- **Nature proximity:** Dev Open Space 210 m · Deciduous Forest 872 m · Open Water 1,445 m · Emergent Wetlands 1,681 m · Woody Wetlands 1,710 m. A real forest patch (NLCD 41) is 872 m away — inside the cap.
- **Metric values:** prototype access score **1.00** · InVEST accessible_urban_nature **160,073 m²** · supply_percapita **38.7 m²/capita** · balance_totalpop −1,891.1 · normalized diff **+0.890** · implied sharing population ≈ 4,136.
- **Why they diverge:** the useful contrast with Locations 1–2 — here there *is* substantial reachable nature (160,073 m², 5–7× more than L1/L2, because a deciduous-forest patch falls inside the radius). 2SFCA still reports only **38.7 m²/capita — 15 % of demand** — because ~4,100 people share it. The proxy reports 1.0 here exactly as it does at L1/L2: it has no resolution between "a sliver of nature" and "a forest," nor between "few neighbours" and "thousands."

### Location 4 — nature-rich southern AOI (both agree: access / adequate)

- **Coordinates:** row=237, col=166; EPSG:26915 E=483733.8 N=4972869.3; lat=44.90906, lon=−93.20606
- **What's there:** pixel is NLCD 24 (Developed, High), but the 500 m window mixes in real green — 38 % Dev Medium, 22 % Dev Low, 15 % Dev High, **11 % Deciduous Forest, 8 % Dev Open Space, 5 % Open Water**.
- **Population:** pixel = 7.5; 500 m window = 1,684 people; density ≈ 2,076 /km² (under half the divergent-quadrant density).
- **Buildings:** 27 % of pixels within 500 m.
- **Nature proximity:** Deciduous Forest 85 m · Dev Open Space 108 m · Woody Wetlands 201 m · Open Water 400 m · Emergent Wetlands 573 m — abundant nature, all very close.
- **Metric values:** prototype access score **1.00** · InVEST accessible_urban_nature **715,898 m²** · supply_percapita **313.1 m²/capita** (≥ 250 ✓) · balance_totalpop +475.4 · normalized diff +0.486 · implied sharing population ≈ 2,287.
- **Why they agree:** 715,898 m² of genuine reachable nature (23× Location 2) divided across ~2,300 effective competitors clears the demand at 313 m²/capita. Agreement happens when nature is abundant *and* population is moderate — both conditions hold here. Note the normalized diff is still positive (+0.49): the proxy is still "higher," but both metrics land on the same verdict.

### Location 5 — sparsely-populated southern fringe (both agree: access / adequate)

- **Coordinates:** row=271, col=180; EPSG:26915 E=484153.8 N=4971849.3; lat=44.89989, lon=−93.20071
- **What's there:** pixel is NLCD 23 (Developed, Medium). 500 m window is 33 % Dev Medium, 31 % Dev High, 17 % Dev Low, 13 % Dev Open Space, 7 % Deciduous Forest.
- **Population:** pixel = 7.0; 500 m window = **260 people**; density ≈ 320 /km² — by far the lowest of the six.
- **Buildings:** 23 % of pixels within 500 m.
- **Nature proximity:** Dev Open Space 30 m · Deciduous Forest 395 m · Woody Wetlands 524 m · Open Water 721 m · Emergent Wetlands 752 m.
- **Metric values:** prototype access score **1.00** · InVEST accessible_urban_nature **701,353 m²** · supply_percapita **1,290.1 m²/capita** · balance_totalpop +7,280.6 · normalized diff +0.497 · implied sharing population ≈ 544.
- **Why they agree:** this is agreement driven by the *absence of people*, not the abundance of nature. The reachable nature (701,353 m²) is essentially the same as Location 4, but only ~544 effective competitors share it → 1,290 m²/capita, 5× the demand. 2SFCA "agrees" here only because almost nobody is around to compete — which is itself the point 2SFCA exists to make.

### Location 6 — negative-diff tail, central AOI (NOT a true reversal — see note)

- **Coordinates:** row=178, col=131; EPSG:26915 E=482683.8 N=4974639.3; lat=44.92497, lon=−93.21942
- **What's there:** pixel is NLCD 23 (Developed, Medium). 500 m window is 53 % Dev Medium, 27 % Dev Low, 20 % Dev High, 1 % Dev Open Space.
- **Population:** pixel = 10.9; 500 m window = 2,464 people; density ≈ 3,039 /km².
- **Buildings:** 38 % of pixels within 500 m.
- **Nature proximity:** Dev Open Space 124 m · Deciduous Forest 1,020 m · Herbaceous 1,064 m · Open Water 1,101 m · Hay/Pasture 1,101 m — every `urban_nature`=1.0 class is just *outside* the 1,000 m cap.
- **Metric values:** prototype access score **0.50** · InVEST accessible_urban_nature **465,703 m²** · supply_percapita **95.1 m²/capita** (still < 250) · balance_totalpop −1,684.5 · normalized diff **−0.332** · implied sharing population ≈ 4,897.
- **Why the diff is negative — and why it is not a reversal:** the proxy returns **0.5**, not 1.0, because the only nature within the 1,000 m cap is Dev Open Space (NLCD 21, `urban_nature`=0.5); the nearest quality-1.0 classes all sit 20–100 m beyond the cap. A score of 0.5 normalizes to ~0.5, while 465,703 m² of accessible nature normalizes high on InVEST's continuous m² scale — so `norm_proto − norm_invest` goes negative. **But both metrics still say the supply is inadequate** (proxy 0.5 < 1.0; InVEST 95.1 < 250 m²/capita). The negative-diff tail (6,250 populated pixels, *all* of them proxy-score-0.5) is a normalization-scale artifact of the proxy's three-value discreteness, not a case of InVEST being more optimistic than the proxy.

---

## Aggregate patterns

- **Direction is uniform; the proxy is the optimistic one everywhere.** 69.9 % of valid-pixel population sits in `diff > 0` pixels, 30.1 % in `diff < 0` — but the entire `diff < 0` set is proxy-score-0.5 pixels (a normalization artifact, see Location 6), not genuine reversals. There is no populated pixel where InVEST reports adequate supply and the proxy does not.
- **Magnitude does not track the obvious drivers.** Correlation of |normalized diff| with population is **−0.002**; with distance-to-large-nature (water/forest/wetland) **−0.063**. The divergence is *pervasive and near-saturated*, not concentrated — the proxy returns 1.0 across most of the valid extent, so the diff sits near +0.9 almost everywhere regardless of local conditions. What varies is InVEST's continuous value; the proxy's discreteness flattens its side of the comparison.
- **The split is a density split, spatially.** The divergent quadrant's population centroid is row 149 / col 103 (north-central, 2.21 people/pixel); the agreement quadrant's is row 238 / col 130 (southern, 0.15 people/pixel). Agreement is confined to the low-density southern fringe of the AOI. There is no "one bad neighborhood" — the divergence covers essentially the entire populated core.
- **Pixels vs people invert the story.** Median `supply_percapita` over valid pixels is **804 m²/capita** — above demand — yet only **9.5 % of residents** clear it. Most pixels are near-empty and supply-rich; most people are in the dense minority of pixels that are supply-poor. The proxy, having no population term, reports on pixels; 2SFCA reports on people.
- **`supply_percapita` mean is not population-grounded.** The unweighted mean is 99,133 m²/capita (p90 = 353,614), inflated by near-empty pixels with tiny denominators. Only the population-weighted view (9.5 % adequate) is meaningful — consistent with the Phase 1 CSV caveat.

---

## Honest gaps

- **Population/extent mismatch — the most important caveat.** The cooling LULC
  (`data/cooling/land_use_2021.tif`) has valid data on only 70,868 of 128,160
  pixels (an irregular ~71k-pixel footprint inside a 356 × 360 frame); the
  Census-2020 population raster (`data/population/minneapolis_pop_2020.tif`)
  covers the full frame. **87,297 people — 56.6 % of the 154,242-person raster
  — fall on pixels the cooling LULC marks nodata.** InVEST UNA cannot model
  supply there at all (no land cover → no output), and the Phase 1 comparison
  excluded them. The prototype's `calculate_nature_access`, by contrast, runs
  its distance transform over the *full* grid and does score those pixels
  (46,797 of those people land on score-0 pixels >1 km from any nature; the
  rest score 0.5/1.0). Consequence: the Phase 1 "~70 % proxy vs ~10 % InVEST"
  headline compares **different population denominators** — proxy 69.7 % is
  over all 154,242 residents, InVEST 9.5 % is over the ~66,945 on valid
  pixels. On a common base (valid-extent residents only) the proxy scores
  **100 % "access" and InVEST 9.5 % "adequate"** — the divergence is *larger*,
  not smaller, than the headline suggested. **Reconciled 2026-05-21**
  (`compare_una_invest.py` now emits restricted-extent columns; REFERENCE.md
  updated): the mixed denominator was the measurement artifact — it *deflated*
  the proxy's headline (off-LULC residents scored 0), so removing them widens
  the gap to 100 % vs 9.5 %. The gap itself is a real property of the metrics,
  not an artifact.
- **"Implied sharing population" is interpretive.** `accessible ÷ supply_percapita`
  is a convenient proxy for "how many people effectively compete for the
  reachable nature," but 2SFCA's two-step distance-decayed catchment math does
  not reduce exactly to that ratio. Treat those figures as illustrative.
- **No neighborhood names.** No geocoding or Minneapolis place-name dataset was
  available; locations are coordinates + LULC context + AOI orientation only.
  Translating them to named neighborhoods/parks needs an external reference.
- **One scenario, one city, one InVEST configuration.** Baseline LULC only;
  Minneapolis downtown only; InVEST run with the published MN config (uniform
  1,000 m radius, exponential decay, 250 m²/capita demand). The demand standard
  is itself a policy choice — InVEST's own docs note "there is no set global
  standard for urban nature demand." A different demand value would move the
  9.5 % adequacy figure substantially; the *direction* of the divergence would
  not change.
- **Building footprints are the InVEST UFR downtown sample**, which covers the
  core but not the whole extent — "% buildings within 500 m" is a relative
  texture indicator, not a census of structures.
- **What this analysis cannot say:** whether residents in the divergent
  quadrant actually experience nature scarcity (ground-truthing, park quality,
  barriers, transit access, and demographics are all out of scope). It can only
  say the two metrics disagree, where, and by how much.

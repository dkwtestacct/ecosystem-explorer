# SA Cooling Biophysical Table — Source Provenance

This file documents the per-class tuning rationale for
`biophysical_table_urban_cooling_SA.csv`. The table was previously a
byte-identical copy of MN (humid continental). The values listed below
were tuned for San Antonio's Köppen BSh (hot semi-arid) climate as a
**medium-confidence interim calibration** — no SA-specific InVEST UCM
args run exists yet, so all values are anchored on published
non-SA measurements + first-principles direction.

Classes not listed here are unchanged from the MN table (developed
classes 22/23/24 are climate-agnostic impervious; rare classes 11, 31,
43, 71, 82, 90, 95 each cover <2 % of the SA AOI and are not material
to the aggregate ΔCC, so tuning was not attempted; class 21 was
intentionally left at MN's value — see note below).

## Class 21 — Developed, Open Space — INTENTIONALLY NOT TUNED

Class 21 was considered for tuning during Stage 2 but was intentionally
left at MN's value (shade 0.3, **Kc 0.516**, albedo 0.161) for SA.
Reason: the semantic divergence between MN's downtown "Developed Open
Space" (parks / cemeteries, intermittent canopy, cool-season mowed turf)
and SA's suburban "Developed Open Space" (residential lawn mix with
variable irrigation regimes, plus an unknown weed-lot fraction) is too
heterogeneous to justify a single defensible value without a
SA-calibrated InVEST UCM args run. A naive bump anchored on FAO-56
warm-season turf Kc would overweight the maintained-lawn case and bias
SA's baseline cooling upward — distorting *scenario delta* outputs
against an inflated baseline. Leaving the value at MN's 0.516 is the
honest choice given a 10%-of-AOI class with no better anchor than
guesswork.

## Methodology note

For the natural-vegetation classes (41, 42, 52, 81), Kc values were
preferentially anchored on **eddy-covariance measurements** of natural
vegetation per Pôças et al. (2017), which found that **measured Kc_mid
is systematically lower than FAO-56 theoretical values** for forest and
pasture. Where eddy-covariance measurements weren't available for the
specific climate / vegetation type (class 52 Shrub/Scrub), FAO-56
semi-arid ranges were used as the anchor instead.

## Per-class rationale

### Class 41 — Deciduous Forest
- **Old (MN copy):** shade 1.0, Kc 1.004, albedo 0.142
- **New (SA):**     shade 1.0, **Kc 0.60**, albedo 0.142
- **Confidence:** Medium-high
- **Rationale:** Eddy-covariance measurement of temperate-USA deciduous
  forest (Chestnut Ridge, Duke Forest) gives Kc_mid 0.43–0.51, well
  below FAO's theoretical 1.0. SA's deciduous trees (pecan, cedar elm,
  hackberry) cluster along riparian corridors with better water access
  than upland temperate deciduous, so above the empirical midpoint:
  0.60. Shade and albedo unchanged — leaf optical properties similar
  enough that adjustment is noise.
- **Source:** Pôças et al. 2017, "Assessing Crop Coefficients for
  Natural Vegetated Areas Using Satellite Data and Eddy Covariance
  Stations," _Sensors_ 17(11):2693. PMC5713072.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5713072/

### Class 42 — Evergreen Forest
- **Old (MN copy):** shade 1.0,  Kc 1.004, albedo 0.142
- **New (SA):**     **shade 0.85**, **Kc 0.50**, **albedo 0.16**
- **Confidence:** Medium-high (Kc), Medium (shade, albedo)
- **Rationale:** SA "Evergreen Forest" is predominantly Edwards Plateau
  juniper–oak woodland (ashe juniper, plateau live oak). Eddy-covariance
  Kc_mid for evergreen forest is 0.17–0.20 (Pôças et al. — Black Hills
  ponderosa pine) vs FAO 1.0. Live oak is broadleaf evergreen with
  year-round active photosynthesis and likely higher transpiration than
  ponderosa; juniper is needleleaf and water-limited. Splitting the
  difference between Black Hills empirical (~0.18) and FAO theoretical
  (1.0) for a broadleaf-evergreen / juniper SA mix: 0.50. Shade lowered
  to 0.85 because Hill Country juniper–oak savanna is more open-canopy
  than dense MN spruce-fir; structural inference, not from a measurement.
  Albedo bumped slightly because live oak / juniper waxy leaves are
  modestly more reflective than spruce needles.
- **Sources:** Pôças et al. 2017 (Kc). Stewart & Oke 2012 "Local Climate
  Zones" (albedo). Shade is a structural-inference judgment call.

### Class 52 — Shrub/Scrub
- **Old (MN copy):** shade 0.0,  Kc 0.968, albedo 0.189
- **New (SA):**     **shade 0.20**, **Kc 0.55**, **albedo 0.22**
- **Confidence:** Medium (Kc, albedo), Low (shade)
- **Rationale:** SA scrub is predominantly mesquite, yaupon, and
  cenizo on caliche substrate. FAO-56 "open vegetation with small
  shrubs" Kc_mid range is 0.50–0.85 for semi-arid; picking the lower
  end for SA's chronic-water-stress regime gives 0.55. The MN value of
  0.968 is implausibly high for xeric scrub. Shade ≠ 0 because mesquite
  / yaupon form partial canopy ≥2 m (InVEST's shade threshold), though
  considerably less than forest. Albedo elevated to reflect the
  exposed-caliche + dormant-brush mix, per Stewart & Oke semi-arid
  shrubland range 0.20–0.25.
- **Sources:** FAO-56 Chapter 6 (Kc range). Stewart & Oke 2012 (albedo).
  Shade value 0.20 is a low-confidence informed estimate.

### Class 81 — Hay/Pasture
- **Old (MN copy):** shade 0.0, Kc 0.932, albedo 0.171
- **New (SA):**     shade 0.0, **Kc 0.65**, **albedo 0.21**
- **Confidence:** Medium
- **Rationale:** SA pasture is predominantly unirrigated Bermuda /
  Kleingrass (warm-season grasses with summer-stress and a clear
  Nov-Feb dormancy). FAO-56 warm-season turf grass Kc_mid = 0.85
  (well-watered); SA's unirrigated dormant-fraction adjustment:
  annual-weighted ≈ 7.5 mo growing × 0.85 + 4.5 mo dormant × 0.30 ≈
  0.64, rounded to 0.65. Compares well with the alpine eddy-covariance
  pasture measurement (0.76–0.88) once you account for SA's drought-
  stress and dormancy. Albedo elevated for dormant warm-season grass
  (yellow-tan vs green sod) per Stewart & Oke 2012.
- **Sources:** FAO-56 Chapter 6 (warm-season turf Kc_mid). Pôças et al.
  2017 (alpine pasture eddy-covariance anchor). Stewart & Oke 2012
  (albedo).

## Future improvements

If SA-specific tuning is revisited:

1. **NatCap San Antonio Urban Agriculture project** — the same project
   that produced the food-forest yield benchmark (11,500 lbs/acre)
   may have published a calibrated SA InVEST UCM args JSON. If so,
   replace this entire table with their values and move from
   medium-confidence to high-confidence.
2. **USDA-ARS Texas Hill Country measurements** for ashe juniper /
   live oak transpiration would tighten class 42 specifically.
3. **Refine class 52 shade** with a remote-sensing-derived canopy
   cover map of Bexar County scrub. The current 0.20 is the
   lowest-confidence row.
4. **Classes 71, 82, 90** were left at MN values for scope reasons.
   Class 82 (Cultivated Crops, 6.8 % of SA AOI) is non-trivial and
   would be the next tuning target.

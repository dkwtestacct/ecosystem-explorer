# UNA Methodology Cross-Check

**Audience:** Research
**Status:** Research — concluded (led to the Nature Quality Score removal)
**Use this for:** The UNA investigation record behind that decision
**Do not use this for:** Current UNA methodology (→ REFERENCE.md) or the decision itself (→ HISTORY.md)
**Historical record of:** the UNA methodology cross-check

---

Diagnostic verification of the Phase 1 UNA reconciliation result (commit
5c7a6a4): on a common modelable-extent population base, the prototype reports
**100 % with access** vs InVEST's **9.5 % with adequate per-capita supply**. A
discrepancy this large usually signals a methodology mismatch — two methods
reading the same data differently — so before accepting it, this note checks
input-by-input whether the two implementations are equivalent.

**Date:** 2026-05-21
**Method:** read both implementations (`app.py` `_compute_access_score_raster`
+ `calculate_nature_access`; `natcap.invest.urban_nature_access` 3.16.2 source);
ran a diagnostic script that recomputes both with InVEST intermediates kept,
verifies the 100 % figure directly, and spot-checks one case-study pixel.

## Summary

**The 100 % vs 9.5 % result is real. It is not driven by an input or
parameter mismatch.** Both methods read the *same* LULC raster, the *same*
Census population raster (verified byte-identical), the *same* `urban_nature`
biophysical table, on the *same* 30 m grid. The two genuine differences —
(1) how each uses the `urban_nature` column and (2) binary vs exponential
distance decay — are **by design**: the prototype is a population-blind
*reachability* test, InVEST UNA is a population-normalized *2SFCA
supply-adequacy* ratio. They answer different questions, and both are
correctly implemented (spot-check below confirms it pixel-by-pixel). One
trivial input difference exists (class 81 search radius, ~147 px) and is
negligible. The gap is the signature of one metric having a demand/population
term and the other not — exactly the "Proxy" parity rating.

## Side-by-side comparison

### urban_nature interpretation

| | Prototype | InVEST UNA |
|---|---|---|
| Reads column | `urban_nature` (0 / 0.5 / 1.0) | `urban_nature` (same values) |
| Treats it as | a **0–1 quality multiplier** | the **proportion of pixel area that is nature** |
| Per-pixel use | `in_range × urban_nature` | `urban_nature_area = 900 m² × urban_nature` |
| Combine across classes | `max()` (highest single class) | sum of decay-weighted area (2SFCA) |

**Verified:** InVEST's `_reclassify_urban_nature_area` produces, per pixel,
`squared_pixel_area × urban_nature_proportion`. Diagnostic output confirms
lucode 11 (urban_nature 1.0) → 900 m², lucode 21 (0.5) → 450 m², lucode 81
(0.5) → 450 m², lucode 22/24 (0.0) → 0. InVEST's own spec calls the column
"a proportion 0-1 of how much of the pixel's area represents urban nature."
The prototype instead uses it as a reachability-quality weight.

**Equivalent? No — by design.** Same numbers, two semantics. Neither is a bug;
they are different metrics. This is the core of the divergence.

### Search radius

| | Prototype | InVEST UNA |
|---|---|---|
| Radius source | per-class `search_radius_m`, capped at `NATURE_RADIUS_CAP_M = 1000` | uniform `search_radius = 1000` (published MN config) |
| Effective radii | 1,000 m for every class **except class 81 Hay/Pasture = 500 m** | 1,000 m for all classes |
| Decay shape | **binary cutoff** — `in_range = distance ≤ radius` | **exponential** — `weight = e^(−dist / 1000)` |
| Weight at 1,000 m | 1.0 (still fully "in range") | e⁻¹ ≈ 0.368, kernel continues beyond |

**Equivalent? Nearly, with two caveats.** (a) The prototype's class 81 radius
is 500 m vs InVEST's 1,000 m — but class 81 is 147 pixels in the MN AOI, so
the effect is negligible. (b) Binary vs exponential decay changes the
*continuous* `accessible`/`supply` magnitudes, but not the *binary*
classifications that produce the headline percentages. Not a driver of the
100 % vs 9.5 % gap.

### Population data

Both methods receive the same Census-2020 raster. The diagnostic writes
`state.pop_count_raster` to the GeoTIFF handed to InVEST and confirms
`numpy.array_equal` against the array the prototype uses — **identical, same
356 × 360 grid, same units (per-pixel counts)**. The asymmetry is *where*
each method uses it: the prototype's per-pixel access score has **no
population term at all** (population enters only as a weight when
`calculate_nature_access` aggregates); InVEST's 2SFCA divides reachable nature
by decay-weighted population at the core of `supply_percapita`. Same data,
used at different points — by design.

### Modelable extent (common base)

The reconciliation restricts to `sup_valid` = pixels with valid LULC **and**
valid InVEST `accessible` **and** valid InVEST `supply_percapita`. Diagnostic
confirms all three coincide: `valid` = 70,868 px, `sup_valid` = 70,868 px —
no per-metric extent variation. The restricted proxy "% access" is computed
over exactly this set; no leakage. ✓

### Threshold semantics

| | Prototype | InVEST UNA |
|---|---|---|
| Cut | access score `> 0.3` (`NATURE_ACCESS_THRESHOLD`) | `supply_percapita ≥ 250 m²/capita` (`urban_nature_demand`) |
| What it means | "is *any* nature class reachable, weighted by quality" | "is per-capita supply adequate vs a demand standard" |
| Discriminating? | **No — non-binding** (see below) | Yes — a genuine adequacy cut |

These are **not comparable thresholds**. The 0.3 is a presence test; the 250
is a sufficiency standard. The 250 m²/capita demand is itself a policy choice
(InVEST docs: "no set global standard for urban nature demand").

## The "100 % access" claim — verified

Direct recomputation over the 70,868-pixel modelable extent:

| proxy threshold | % of pixels | % of population |
|---|---:|---:|
| score > 0.0 | 100.0000 % | 100.0000 % |
| **score > 0.3 (the app's threshold)** | **100.0000 %** | **100.0000 %** |
| score > 0.5 | 82.1711 % | 69.8852 % |
| score > 0.6 | 82.1711 % | 69.8852 % |
| score > 0.99 | 82.1711 % | 69.8852 % |

Score distribution over the modelable extent: **0 pixels at 0.0**, 12,635 at
0.5, 58,233 at 1.0.

**It is genuinely 100.0000 %, not a rounded 99.x %.** Not one modelable-extent
pixel is beyond reach of every nature class.

**The 0.3 threshold is non-binding; the radius is the binding constraint.**
Because per-pixel scores are only ever 0.5 or 1.0, *any* threshold in (0, 0.5]
yields 100 % and any threshold in (0.5, 1.0) yields the same 82 % / 70 %. The
0.3 cut does no discriminating work — "Nature Access %" is effectively a
binary "is any nature class within 1 km," and in dense downtown MN that is
true everywhere nature can be modelled. By contrast InVEST's `supply ≥ 250`
splits the same extent to 61.4 % of pixels / **9.5 % of population**.

## Spot check: Location 2 (case study) — manual computation

Pixel row 62, col 58 (LULC 23, Developed Medium). Case study reports proxy
score 1.0, InVEST supply 3.3 m²/capita.

**Prototype — manual, independent per-class distance transforms:**

| class | distance | radius | in range? | urban_nature | contributes |
|---|---:|---:|---|---:|---:|
| 11 Water | 2,154.2 m | 1,000 | no | 1.0 | 0.0 |
| 21 Open Space | 276.6 m | 1,000 | **yes** | 0.5 | 0.5 |
| 41 Decid Forest | 2,101.1 m | 1,000 | no | 1.0 | 0.0 |
| 71 Herbaceous | 725.0 m | 1,000 | **yes** | 1.0 | **1.0** |
| 81 Hay/Pasture | 766.6 m | 500 | no | 0.5 | 0.0 |
| (42, 43, 52, 90, 95 all > 2,400 m) | | | no | | 0.0 |

`max(contributions) = 1.0` — **matches the script's proto raster exactly.**
The 1.0 comes from a Herbaceous patch 725 m away. (Note class 81 at 766.6 m
falls *outside* its 500 m radius — the one place the per-class-radius
difference bites, and it doesn't change the result here.)

**InVEST — same pixel:**

- `urban_nature_area` (this pixel) = 0 m² (the pixel is LULC 23, not nature)
- `accessible_urban_nature` = 22,899.7 m²
- `distance_weighted_population` = 6,154.42
- `urban_nature_supply_percapita` = 3.2911 m²/capita
- `urban_nature_demand` = 3,218.75  →  pop (12.875) × 250 = 3,218.75 ✓
- `urban_nature_balance_totalpop` = −3,176.38  →  (3.2911 − 250) × 12.875 = −3,176.38 ✓

The 2SFCA `supply_percapita` is itself a convolution output (not hand-
computable), but every documented step *downstream* of it reproduces exactly:
demand = pop × 250 and balance = (supply − 250) × pop both match to the
penny. InVEST is computing what it documents.

**Verdict:** both implementations are correct at this pixel. The proxy says
"nature reachable → 1.0"; InVEST says "22,900 m² reachable, shared via 2SFCA
across ~6,150 decay-weighted competitors → 3.3 m²/capita, 1.3 % of the 250 m²
demand." Same pixel, both right, different questions.

## Honest conclusion

**The methodology is equivalent in inputs; the metrics genuinely diverge by
design. The 100 % vs 9.5 % result is real, not an artifact.**

- Same LULC, same population raster (verified identical), same `urban_nature`
  table, same grid, same modelable extent. No input or coverage mismatch.
- The two real differences are definitional: the prototype uses `urban_nature`
  as a quality weight and tests binary reachability with no demand term;
  InVEST uses `urban_nature` as area, runs 2SFCA, and divides by competing
  population. The prototype's "100 %" truthfully means "100 % of
  modelable-extent residents have some nature within 1 km"; InVEST's "9.5 %"
  truthfully means "9.5 % have ≥ 250 m²/capita after 2SFCA." Both statements
  are correct.
- The 0.3 threshold is non-binding — it could be any value ≤ 0.5 — so the
  proxy's "Nature Access %" is, on the modelable extent, a binary
  "any nature within 1 km" indicator. That is a real characterization of the
  proxy worth noting, but it is not a bug.
- One trivial input difference (class 81 Hay/Pasture radius 500 m vs 1,000 m,
  147 pixels) has no material effect.

This confirms the existing "Proxy" parity rating and the REFERENCE.md framing
(commit 5c7a6a4): closing the gap would mean adopting 2SFCA as the metric, not
correcting a mismatch. There is nothing to correct — the comparison is sound.

## Honest gaps

- The InVEST `supply_percapita` value itself was verified only by
  internal-consistency (demand and balance reproduce exactly from it), not by
  a hand-recomputed 2SFCA convolution — that is not feasible by hand.
- One scenario (baseline), one city (MN downtown), one InVEST config. A
  mixed-density city would exercise 2SFCA's supply/demand contrast harder; the
  binary-vs-exponential decay difference would also show up more in the
  continuous `accessible` magnitudes there.

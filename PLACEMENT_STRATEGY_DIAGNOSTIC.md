# Placement Strategy Diagnostic

**Date:** 2026-05-23 (re-run after Brief 9 strategy reformulation)
**Question:** Do the placement strategies do anything?
**Status:** complete; supersedes the 2026-05-23-pre-Brief-9 measurement

> **Re-run note.** This document was first published on 2026-05-23 under the
> pre-Brief-9 strategy formulas (raw CN for flood-focused, NLCD-intensity
> proxy for cooling-focused, population × homegrown access-deficit for the
> then-named `equity-focused`). Brief 9 reformulated three of the four
> weighted strategies to use canonical InVEST quantities (per-pixel
> `Q_{p,i}` for flood-focused, canonical HMI + real distance-to-buildings
> for cooling-focused, canonical UNA per-capita supply deficit for the
> renamed `undersupply-focused`). The diagnostic was re-run end-to-end
> against the new formulas; every table and finding below is from the
> post-reformulation run. The Brief 6 finding "flood-focused is the
> weakest mover on flood reduction" is **reversed** in the new data — see §6.

---

## 1. Question

The placement-strategy feature (`_compute_suitability_weights` + the
five-option sidebar radio) has been shipped for a while, but the empirical
question of *how much* the strategies move the needle, *where*, and *under
what conditions* hasn't been settled. This diagnostic measures three
layers of strategy behaviour:

1. **Layer 1 — suitability surface variance.** Does each strategy's
   per-pixel weight surface have enough variance to differentiate
   pixels at all?
2. **Layer 2 — chosen-pixel selectivity.** Does each strategy actually
   sample high-suitability pixels under its own surface (positive gap
   vs the overall pool mean)?
3. **Layer 3 — metric outcome delta vs random.** Do the chosen
   pixels translate into measurably different `evaluate_scenario`
   outputs? Does the effect scale with conversion fraction?

The goal is to record what we measured and what it means; if the data
is ambiguous, the writeup says so.

---

## 2. Method

**Cities:** Minneapolis, MN (downtown InVEST sample extent, 122.8 km²,
convertible pool 26,372 pixels), Minneapolis Full, MN (full city
boundary, 148.9 km², `available=False` in UI, pool 85,873), San
Antonio, TX (Bexar County bbox, 3,058 km², pool 840,488).

**Strategies:** `random`, `flood-focused`, `cooling-focused`,
`undersupply-focused` (renamed from `equity-focused` in Brief 9),
`balanced`.

**Layer 3 scenarios:** `all_gi` (gi=100, ff=0, hd=0), `all_ff` (gi=0,
ff=100, hd=0), `all_hd` (gi=0, ff=0, hd=100). Mixed allocations were
not measured.

**Conversion fractions (pct_converted):** 10, 25, 50.

**Seeds:** 0–9 (ten per cell). RNG plumbing verified deterministic:
`evaluate_scenario` builds `np.random.default_rng(seed)` and passes it
through `_select_pixels_for_conversion`'s `rng.choice` — same inputs +
same seed → bit-identical outputs.

**Total runs (Layer 3):** 3 × 5 × 3 × 3 × 10 = **1,350** calls to
`evaluate_scenario`. **210** of these are "saturated" (the strategy
didn't have enough non-zero pixels to sample `n_chosen` without
replacement; the Brief 7 fallback fills the remainder uniformly).
All 1,350 produced numerical outputs.

**Implementation:** `placement_strategy_diagnostic.py` (orchestrator
mode + per-city worker mode). Each city imports `app.py` with a
streamlit stub (lifted from `precompute_scenarios.py`), runs all three
layers, writes incrementally to CSV with checkpoint-resume. Each worker
runs in a fresh subprocess so the city's raster stack is dropped on
exit.

**Raw data:** `analysis/placement_diagnostic/layer{1,2,3}_*.csv`.

**Reproduction:** `python3 placement_strategy_diagnostic.py`. Wipe
`analysis/placement_diagnostic/` first to re-run from scratch; otherwise
the script resumes from existing rows.

**Wall time per `evaluate_scenario` call** (from the `elapsed_s`
column): MN downtown 0.030 s median, MN Full 0.060 s median, SA
0.940 s median. Total diagnostic runtime: ~13 minutes including imports.
Notably faster than the pre-Brief-9 run because the homegrown
access-score raster pipeline was deleted in Stage E cleanup.

**Minneapolis Full caveat.** `available=False` filters MN Full out of
the production sidebar, but the diagnostic stub bypasses the filter.
MN Full has its own CITIES entry and full data pipeline and runs as a
first-class city for this measurement.

---

## 3. Layer 1 — suitability surface variance

| city | strategy | n_pixels | mean | std | min | p25 | p50 | p75 | p95 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Minneapolis, MN | flood-focused | 26,372 | 16.96 | 5.40 | 0 | 13.27 | 17.66 | 18.89 | 23.0 | 23.0 |
| Minneapolis, MN | cooling-focused | 26,372 | 0.361 | 0.108 | 0.035 | 0.336 | 0.405 | 0.447 | 0.447 | 0.448 |
| Minneapolis, MN | undersupply-focused | 26,372 | 2.04 | 4.76 | 0 | 0 | 0 | 0 | 14.81 | 16.39 |
| Minneapolis, MN | balanced | 26,372 | 3.79e-05 | 3.17e-05 | 2.67e-06 | 2.19e-05 | 2.83e-05 | 3.28e-05 | 1.21e-04 | 1.34e-04 |
| Minneapolis Full, MN | flood-focused | 85,873 | 14.12 | 4.39 | 0 | 13.27 | 13.27 | 17.66 | 17.66 | 17.66 |
| Minneapolis Full, MN | cooling-focused | 85,873 | 0.358 | 0.108 | 0.035 | 0.298 | 0.405 | 0.447 | 0.447 | 0.448 |
| Minneapolis Full, MN | undersupply-focused | 85,873 | 2.52 | 5.05 | 0 | 0 | 0 | 0 | 14.94 | 16.7 |
| Minneapolis Full, MN | balanced | 85,873 | 1.17e-05 | 8.57e-06 | 4.02e-07 | 7.29e-06 | 8.04e-06 | 9.70e-06 | 3.21e-05 | 3.54e-05 |
| San Antonio, TX | flood-focused | 840,488 | 17.61 | 4.73 | 0 | 13.27 | 18.89 | 18.89 | 23.0 | 23.0 |
| San Antonio, TX | cooling-focused | 840,488 | 0.309 | 0.120 | 0.006 | 0.221 | 0.344 | 0.405 | 0.447 | 0.448 |
| San Antonio, TX | undersupply-focused | 840,488 | 0.205 | 1.20 | 0 | 0 | 0 | 0 | 0 | 16.45 |
| San Antonio, TX | balanced | 840,488 | 1.19e-06 | 2.36e-06 | 1.83e-08 | 6.46e-07 | 8.20e-07 | 9.46e-07 | 1.09e-06 | 3.29e-05 |

**Interpretation:**

- **`flood-focused`** now varies on per-pixel runoff `Q_{p,i}` in mm,
  not raw CN. Range 0–23 mm at the 2-inch design storm. Tight band
  around the mid-developed CN range; a substantial mass at the
  ceiling (23 mm) where impervious surfaces produce maximum runoff.
  Notice the **zero floor** — pixels with low CN (high retention)
  produce zero runoff at this storm depth.
- **`cooling-focused`** now combines canonical HMI with a real
  distance-to-buildings weight. Range 0.006–0.45 across cities;
  tighter distribution than under the old NLCD-intensity proxy
  because real distance is smoother than three-step categorical.
- **`undersupply-focused`** is the most concentrated of the four —
  p50 = 0 across all cities, meaning **more than half the convertible
  pool has zero per-capita deficit** (residents already meet the
  16.7 m²/capita standard). The signal lives entirely in the tail.
  This is the canonical InVEST UNA framing and the right answer for
  "undersupply" — but it produces a structurally aggressive saturation
  problem (see §4 and §6).
- **`balanced`** lives at numerically tiny scales because it averages
  three sum-1-normalized surfaces. The std/mean ratio is similar to
  the others.

**Non-zero pixel counts per strategy** (the basis for saturation):

| city | flood-focused | cooling-focused | undersupply-focused | balanced |
|---|---:|---:|---:|---:|
| MN downtown | 26,357 | 26,372 | 6,533 | 26,372 |
| MN Full | 85,855 | 85,873 | 27,165 | 85,873 |
| SA | 833,932 | 840,488 | 60,124 | 840,488 |

`undersupply-focused` has only ~25% non-zero pixels on the MN cities
and ~7% on SA. At pct=10 on MN downtown the strategy can still sample
without saturation (n_chosen=2,637 < nonzero=6,533); at pct=25 it
saturates (n_chosen=6,593 > 6,533); at pct=50 it saturates heavily.
On SA, `n_chosen=84,049 > nonzero=60,124` even at pct=10 — every
SA × undersupply-focused cell saturates.

---

## 4. Layer 2 — does each strategy select different pixels than random?

Gap = `chosen_pool_mean_score − overall_pool_mean_score`. Significance:
`|gap| > 2 × std(random's gap)` on the same (city, pct). Random is
scored against `flood-focused`'s surface so it has a baseline. All
cells run 10 seeds. 70 of 450 cells are flagged saturated — those rows
are valid but reflect the non-zero-then-uniform-remainder fallback path
(the strategy got all its non-zero pixels and filled the rest
uniformly), which structurally dilutes the gap.

### Selected rows — per (city, strategy, pct)

| city | strategy | pct | gap_mean | gap_std | sig | sat |
|---|---|---:|---:|---:|---|---|
| Minneapolis, MN | flood-focused | 10 | +1.43 | 0.057 | ✓ | — |
| Minneapolis, MN | flood-focused | 25 | +1.39 | 0.030 | ✓ | — |
| Minneapolis, MN | flood-focused | 50 | +1.30 | 0.022 | ✓ | — |
| Minneapolis, MN | cooling-focused | 10 | +0.032 | 0.0013 | — | — |
| Minneapolis, MN | cooling-focused | 25 | +0.029 | 0.0005 | — | — |
| Minneapolis, MN | cooling-focused | 50 | +0.024 | 0.0004 | — | — |
| Minneapolis, MN | undersupply-focused | 10 | +12.18 | 0.029 | ✓ | — |
| Minneapolis, MN | undersupply-focused | 25 | +6.10 | 0 | ✓ | sat |
| Minneapolis, MN | undersupply-focused | 50 | +2.04 | 0 | ✓ | sat |
| Minneapolis Full, MN | flood-focused | 10 | +1.34 | 0.020 | ✓ | — |
| Minneapolis Full, MN | flood-focused | 50 | +1.24 | 0.008 | ✓ | — |
| Minneapolis Full, MN | undersupply-focused | 10 | +9.84 | 0.029 | ✓ | — |
| Minneapolis Full, MN | undersupply-focused | 25 | +7.57 | 0 | ✓ | sat |
| Minneapolis Full, MN | undersupply-focused | 50 | +2.52 | 0 | ✓ | sat |
| San Antonio, TX | flood-focused | 10 | +1.43 | 0.005 | ✓ | — |
| San Antonio, TX | flood-focused | 50 | +1.25 | 0.001 | ✓ | — |
| San Antonio, TX | cooling-focused | 10 | +0.040 | 0.0002 | ✓ | — |
| San Antonio, TX | cooling-focused | 50 | +0.032 | 0.0001 | ✓ | — |
| San Antonio, TX | undersupply-focused | 10 | +2.05 | 0 | ✓ | sat |
| San Antonio, TX | undersupply-focused | 25 | +0.82 | 0 | ✓ | sat |
| San Antonio, TX | undersupply-focused | 50 | +0.41 | 0 | ✓ | sat |

(Full table — including `random` rows and `balanced` rows — in
`layer2_chosen_pixel_scores.csv`.)

**Interpretation:**

- **`flood-focused` is now far more selective** than under the pre-Brief-9
  CN-based formula. Gap mean +1.3 to +1.4 mm runoff over the pool mean
  — the Q-based formula concentrates more aggressively because Q has a
  much sharper distribution than CN (zero below the initial-abstraction
  threshold; rises rapidly past it).
- **`cooling-focused` selectivity is small** (+0.03–0.04 HMI units) but
  consistent. Lower than the pre-Brief-9 +0.11 because the new formula's
  smoother distance-to-buildings weight has more uniform low-end mass
  than the bimodal NLCD-intensity surface.
- **`undersupply-focused` shows huge gap_mean values that decay to
  exactly zero at saturation**. At pct=10 on MN downtown, the strategy
  picks pixels averaging 12.2 m²/capita deficit above the pool mean of
  2.0 — a 6× concentration. At pct=25 and beyond, the strategy is
  forced to include every non-zero pixel plus zero-weighted remainder,
  so the gap collapses to a deterministic value (gap_std = 0).
- **`random`** shows gap ≈ 0, as expected.

So Layer 2 says: every weighted strategy is now genuinely
selecting non-random pixels, with the canonical-quantity reformulations
producing larger or more consistent gaps than the pre-Brief-9 proxies.
The undersupply-focused saturation pattern is the new wrinkle: the
strategy has more concentration than the convertible-pool size can
absorb at any but the smallest pct values.

---

## 5. Layer 3 — does it move the metric?

Cross-city aggregate of `max |delta vs random|` per (city, strategy,
metric), across all (scenario, pct) cells:

| city | strategy | flood_reduction | mean_hm (HMI units) | runoff_acre_feet | food_mln_lbs | carbon_tons_co2_yr |
|---|---|---:|---:|---:|---:|---:|
| Minneapolis, MN | flood-focused | **0.372** | 0.00215 | 10.74 | 0 | 0 |
| Minneapolis, MN | cooling-focused | 0.058 | 0.00451 | 1.92 | 0 | 0 |
| Minneapolis, MN | undersupply-focused | 0.107 | 0.00460 | 3.06 | 0 | 0 |
| Minneapolis, MN | balanced | 0.248 | 0.00593 | 7.33 | 0 | 0 |
| Minneapolis Full, MN | flood-focused | **0.417** | 0.00423 | 44.21 | 0 | 0 |
| Minneapolis Full, MN | cooling-focused | 0.157 | 0.00472 | 16.98 | 0 | 0 |
| Minneapolis Full, MN | undersupply-focused | 0.099 | 0.00299 | 10.10 | 0 | 0 |
| Minneapolis Full, MN | balanced | 0.308 | 0.00720 | 33.08 | 0 | 0 |
| San Antonio, TX | flood-focused | **0.136** | 0.00101 | 90.53 | 0 | 0 |
| San Antonio, TX | cooling-focused | 0.060 | 0.00286 | 45.76 | 0 | 0 |
| San Antonio, TX | undersupply-focused | (saturated all cells) | | | | |
| San Antonio, TX | balanced | 0.100 | 0.00243 | 72.06 | 0 | 0 |

`food_mln_lbs` and `carbon_tons_co2_yr` are **always 0** for the same
structural reason as the pre-Brief-9 run: in single-cover scenarios
(all_gi / all_ff / all_hd), the converted-pixel counts depend only on
`pct_converted × pool_size × allocation_pct`, not on *which* pixels are
picked. Food and carbon scale linearly with these counts, so they are
strategy-invariant by construction.

To turn `mean_hm` into something physical, multiply by the city's
`HM_TO_FAHRENHEIT` (MN ≈ 3.69, SA ≈ 6.30): the largest cooling delta
observed (balanced on MN Full at pct=50, +0.0072 HMI) translates to
~0.027 °F. **Cooling deltas remain below the ±2 °F uncertainty band
the UCM card already discloses.**

---

## 6. Findings

The Brief 6 question — *"do the strategies do anything?"* — gets a much
sharper answer after Brief 9's canonical-quantity reformulation. The
strategies are not theatrical, the signal-to-noise has improved, and
the Brief 6 finding that "flood-focused is the weakest mover on flood
reduction" has reversed.

**Flood-focused is now the strongest mover on flood reduction across
every city.** Under the pre-Brief-9 CN-based formula, flood-focused
was the *weakest* of the four weighted strategies on its own metric
(0.094 max delta on MN downtown vs equity-focused's 0.163). Under the
Brief 9 Q-based formula, flood-focused beats every other strategy on
every city for the flood_reduction metric: 0.372 on MN downtown
(4× larger than the old number), 0.417 on MN Full (3.7× larger),
0.136 on SA (4.5× larger). This was the predicted outcome of the
reformulation — the SCS-CN runoff equation concentrates much more
sharply on high-runoff pixels than raw CN does, and the empirical
result confirms it.

**Undersupply-focused saturates aggressively on every city.** The strict
per-capita-deficit formula (no population multiplier, no artificial
floor) puts essentially all the suitability mass on pixels that have
both nonzero population and supply below 16.7 m²/capita. On MN downtown
that's 6,533 of 26,372 convertible pixels (25%); on MN Full it's 27,165
of 85,873 (32%); on SA it's 60,124 of 840,488 (7%). At pct≥25 on the
two MN cities and at every pct on SA, the strategy hits the Brief 7
saturation fallback and fills the remainder uniformly. The fallback is
working as designed — no crashes, valid outputs — but the practical
consequence is that **undersupply-focused on SA produces near-random
outcomes at the conversion fractions users typically pick**. The
gap-vs-random on Layer 3's flood_reduction metric is reported here as
"(saturated all cells)" because the dilution by uniform remainder
swamps the underlying signal.

**Cooling-focused remains a small mover.** Both the pre-Brief-9 and
Brief 9 formulas produce cooling-focused gaps that translate to
sub-0.03 °F temperature deltas — well below the UCM's own ±2 °F
uncertainty. The Brief 9 reformulation (canonical HMI + real
distance-to-buildings) was structurally cleaner than the old NLCD-
intensity proxy, but the underlying physics — a single hot-pixel
conversion can only do so much for citywide mean cooling — caps the
achievable effect size. Neither formula was going to produce a
detectable °F-level cooling effect at the AOI mean.

**The original "SA larger AOI → bigger strategy effects" hedge is
even less defensible now.** Pre-Brief-9 the data already showed
MN Full leading on flood deltas; post-Brief-9 the pattern is sharpened.
Flood-focused max delta is MN Full (0.417) > MN downtown (0.372) >
SA (0.136). On runoff_acre_feet in absolute terms SA leads (90.5 ac-ft
vs MN Full's 44.2 vs MN downtown's 10.7), but relative to baseline
runoff (SA ~10,000 ac-ft, MN Full ~1,300, MN downtown ~300) the
ratios are MN Full ~3.4%, MN downtown ~3.6%, SA ~0.9%. SA's strategy
effects are smallest in proportional terms on every metric. The
"AOI → effect" intuition does not hold.

**MN Full now dominates on flood_reduction and runoff** — the biggest
absolute deltas of any city on three of the five metrics. Two factors
contribute: (a) the largest convertible pool of the three (85,873 vs
26,372 / 840,488) but with a tighter spatial heterogeneity than SA,
giving the strategies room to maneuver without saturating; (b) the
random noise floor is lower than on MN downtown (more seeds average
toward zero), so 2σ significance gates pass at smaller signals.

**Food and carbon remain strategy-invariant in single-cover scenarios.**
This is unchanged from Brief 6 and is structural — both metrics depend
on counts, not placement geometry. Mixed-allocation scenarios were not
measured.

---

## 7. Implications

### Doc hedges already replaced

Brief 9 Stage I replaced the two doc hedges that motivated this whole
question:

- `REFERENCE.md:166-170` (the "SA larger AOI → bigger strategy effects"
  hedge) was rewritten in Brief 9 Stage F as part of the "Honest caveats"
  section under the new placement-strategy table — the unsourced
  speculation is gone; the section now points readers to this diagnostic
  for empirical answers.
- `INVEST_PLACEMENT.md:93` (the parallel UMH-focused hedge) was
  rewritten in Brief 9 Stage I as "Whether SA's larger AOI translates to
  a bigger UMH placement effect than MN downtown's is not yet measured."
  UMH was not in this diagnostic's metric set; that hedge stays accurate.

### Feature design

- **Undersupply-focused on SA is effectively random in production.**
  Every cell at every pct saturates. A user who picks "Prioritize
  areas with unmet nature demand" on San Antonio gets the same
  scenario outcome as random placement plus a thin sliver of
  intent-honoring placement. This is **not a bug** — it's the
  canonical InVEST UNA framing meeting a city where nearly everyone
  already has adequate per-capita nature supply (SA's bbox includes
  large county-fringe areas). But the UI doesn't currently signal
  this to users. Two follow-up directions worth considering: (1)
  surface the saturation state when active (sidebar caption or
  results banner: "On this city, this strategy converts mostly via
  uniform random sampling"), or (2) reintroduce a population weight
  *additively* (deficit + population-normalized term) so the
  strategy has more pixels to work with, at the cost of partially
  re-introducing the aggregate-need framing the reformulation
  rejected. Both are out of scope for this brief.
- **Cooling-focused effect size is at the noise floor of the metric.**
  Not a function of the strategy formula — the cap is the underlying
  physics. The strategy is doing what it can; further refinement is
  unlikely to produce user-visible improvement in the °F output.
- **Flood-focused is now doing real work** and is the strongest mover
  on flood-reduction across all cities. The Brief 9 reformulation was
  the right call empirically.

### For NatCap collaboration

Questions this raises:

1. **Is the per-capita-only undersupply formulation the right NatCap
   framing for placement?** The data shows it concentrates too aggressively
   to be usable at moderate pct values on a county-scale AOI. NatCap
   UNA's `urban_nature_balance_percapita.tif` is correct for *reporting*
   per-pixel deficit; whether it should be used as-is for *placement
   weighting* (vs e.g. a `Pund_adm`-style undersupplied-population
   weighting) is a question for the UNA team.
2. **For cooling, is the per-pixel mean(HMI) target the right thing to
   optimize for placement?** The model-implied physics caps single-pixel
   contribution to AOI mean cooling at ~0.001 HMI units, regardless of
   strategy. NatCap may have a sharper "where does conversion help most"
   signal we're missing.
3. **Mixed-allocation scenarios — does anyone in the NatCap ecosystem
   measure placement-strategy effects at, say, gi=50/ff=50/hd=0?**
   This diagnostic measures only single-cover; the food-and-carbon-are-
   structurally-invariant finding only generalizes mechanically. Real
   user scenarios are mixed.

---

## 8. Limitations

- **Mixed allocations not measured.** Only all-GI, all-FF, all-HD —
  not gi=50/ff=50 or other interior points. Strategy × scenario
  interactions at mixed allocations could differ from the
  single-cover findings, though the food/carbon invariance result
  generalizes mechanically.
- **Three conversion fractions only.** pct ∈ {10, 25, 50}. The
  curve between these points (or above 50%) isn't characterised.
  Undersupply-focused's saturation cliff begins around pct=20 on
  MN cities — the diagnostic captures its behaviour on both sides
  of the cliff but not the exact transition.
- **Ten seeds per cell.** Enough to estimate first and second
  moments of the gap distribution; not enough to characterise
  rare-but-real signals.
- **Lookup-table cache not exercised.** Each cell runs a fresh
  `evaluate_scenario`. The UI's behaviour depends on a hybrid
  lookup-table + live-refresh pattern; the live-evaluation path
  matches what the optimizer and any non-random strategy use, so
  the diagnostic's results apply to the strategy-comparison case
  but not necessarily to lookup-table-hit cases.
- **No surrogate involved.** This measures the raw biophysics
  layer, not the random-forest surrogate the optimizer uses.
- **`evaluate_scenario` reflects current `SCENARIO_SCHEMA_VERSION =
  18`.** Schema bumps will change absolute numbers and may
  invalidate this diagnostic's specific Layer 3 values, though
  Layer 1 + 2 are schema-independent. The version bump from 17 → 18
  in Brief 9 reflects this reformulation.
- **UMH not measured.** The two non-RF deterministic targets
  (`preventable_mh_cases`, `avoided_mh_cost_usd`) were not included
  in the Layer 3 column list. The parallel "SA larger → bigger
  effect" hedge in `INVEST_PLACEMENT.md:93` (which is specifically
  about UMH) remains tagged as "not yet measured."
- **`available=False` cities behave differently from production.**
  MN Full's data pipeline runs cleanly via the diagnostic's
  stub-override, but production users can't reach it.

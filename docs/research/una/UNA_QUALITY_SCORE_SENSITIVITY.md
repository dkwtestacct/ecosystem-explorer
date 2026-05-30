# Nature Quality Score Scenario Sensitivity at MN

**Audience:** Research
**Status:** Research — concluded (led to the Nature Quality Score removal)
**Use this for:** The UNA investigation record behind that decision
**Do not use this for:** Current UNA methodology (→ ../../../REFERENCE.md) or the decision itself (→ ../../archive/HISTORY.md)
**Historical record of:** Nature Quality Score sensitivity testing

---

Empirical test of whether Nature Quality Score varies meaningfully across
scenarios in MN downtown, or whether — like Nature Access — it is structurally
insensitive at this AOI scale.

**Date:** 2026-05-21
**Method:** ran `evaluate_scenario` (MN, seed=42, random placement) for the
baseline plus all-one-cover extremes at `pct_converted` 10/25/50. For each, the
per-pixel access-score raster was recomputed and population binned by score.
No shipped-code changes — diagnostics run from a scratch script.

## Summary

Nature Quality Score **is meaningfully more informative than Nature Access** —
it spans **0.569 to 0.741 across scenarios, a +29.5 % spread**, versus Nature
Access's 5-percentage-point range. But the response is a **step, not a
gradient**, with three sharp limitations:

1. **The entire jump happens in the first 10 % of conversion.** Any green
   conversion (GI or FF) moves Quality Score from the 0.572 baseline to ~0.73
   at `pct_converted=10`; from 10 % to 50 % it barely moves (0.732 → 0.741).
   In the interactive slider range users actually explore, Quality Score
   discriminates *"greening vs none"* but not *"how much greening."*
2. **It cannot distinguish Green Infrastructure from Food Forest.** Both
   convert developed land to an `urban_nature = 1.0` class (woody wetland 90 /
   deciduous forest 41); their Quality Score rows are *identical* at every
   conversion level.
3. **It is nearly dead to High Density.** All-HD scenarios move Quality Score
   only 0.572 → 0.569 — a 0.5 % nudge.

So Quality Score is effectively a **bimodal indicator** (~0.57 with no greening,
~0.73 with any greening) responsive to a single input dimension. It is better
than Nature Access, but it is not the smooth graded metric its "continuous
companion" framing implies.

## Test scenarios

`pop@X` columns are the share of the 154,242-person population whose per-pixel
access score is exactly X. (No values other than 0 / 0.5 / 1.0 occurred.)

| Scenario | Nature Access % | Quality Score | pop@0 | pop@0.5 | pop@1.0 |
|---|---:|---:|---:|---:|---:|
| Baseline | 69.7 | **0.572** | 30.34 % | 24.97 % | 44.69 % |
| All GI 10 % | 73.4 | **0.732** | 26.61 % | 0.40 % | 73.00 % |
| All FF 10 % | 73.4 | **0.732** | 26.61 % | 0.40 % | 73.00 % |
| All HD 10 % | 69.4 | **0.570** | 30.64 % | 24.67 % | 44.69 % |
| All GI 25 % | 73.9 | **0.739** | 26.06 % | 0.02 % | 73.92 % |
| All FF 25 % | 73.9 | **0.739** | 26.06 % | 0.02 % | 73.92 % |
| All HD 25 % | 69.1 | **0.569** | 30.87 % | 24.43 % | 44.69 % |
| All GI 50 % | 74.1 | **0.741** | 25.85 % | 0.00 % | 74.15 % |
| All FF 50 % | 74.1 | **0.741** | 25.85 % | 0.00 % | 74.15 % |
| All HD 50 % | 69.1 | **0.569** | 30.89 % | 24.41 % | 44.69 % |

**Spreads.** Quality Score: min 0.569, max 0.741, range **0.172** (+29.5 % vs
the 0.572 baseline; −0.5 % low). Nature Access: min 69.1, max 74.1, range
**5.0 pp**. Restricting to the interactive range (baseline + 10 %/25 % only)
barely changes the Quality Score span (0.569–0.739) — because the jump is
already complete by 10 %.

## Per-pixel score distribution shifts

The mechanism behind the step response is visible in the `pop@0.5` column:

- **Baseline:** 24.97 % of the population (38,507 people) sit at score 0.5 —
  pixels within 1 km of only a *0.5-quality* class (Developed Open Space,
  NLCD 21), with no `urban_nature = 1.0` class in range.
- **Any GI/FF conversion empties that bin almost instantly.** Random placement
  scatters new `urban_nature = 1.0` pixels across the AOI; at just
  `pct_converted = 10` essentially every former 0.5 pixel gains a 1.0-class
  neighbour within 1 km. `pop@0.5` collapses 38,507 → 611 (10 %) → 29 (25 %)
  → 2 (50 %). There is no gradient — the 0.5 cohort is gone after the first
  increment, which is exactly why Quality Score steps rather than ramps.
- **`pop@1.0`** jumps 68,938 → 112,593 at GI/FF 10 %, then is nearly flat
  (114,020 → 114,368). **`pop@0`** drops only modestly (46,797 → ~39,900):
  ~7,000 off-LULC residents near the AOI edge gain a 1.0-class neighbour, but
  most score-0 people are too far from any convertible land to be reached.
- **High Density** never changes `pop@1.0` (it stays exactly 68,938): HD
  conversion removes Developed Open Space (a 0.5 class), so it can only nudge
  people 0.5 → 0, never touch 1.0-class coverage. Hence the near-flat HD rows.

The persistent **~26 % of population stuck at score 0** under even All-GI 50 %
is the off-cooling-LULC population identified in the Phase 1 denominator work
(`UNA_METHODOLOGY_CROSS_CHECK.md`) — it caps Quality Score around 0.74.

## Honest conclusion

**Quality Score varies meaningfully across MN scenarios — a 0.172 range /
+29.5 % spread — and is clearly more informative than Nature Access, which
moves only 5 pp and is structurally degenerate (commit b946edf).** That is a
real difference: Quality Score does carry scenario signal where Nature Access
does not.

But "informative" needs the qualifier: Quality Score is a **step function, not
a graded response**. It answers one binary question well — *is there any green
conversion?* (0.572 → ~0.73) — and answers almost nothing else. It does not
grade greening intensity in the slider range users explore (10 %→50 % moves it
0.009), cannot tell Green Infrastructure from Food Forest, and is nearly inert
to High Density. Its "continuous companion metric" framing oversells it: the
underlying score only takes three values, and at MN the 0.5 cohort vanishes
after the first conversion increment, leaving an effectively two-state metric.

**Recommendation input (not a decision):** Quality Score is worth keeping over
Nature Access at MN — it is the less degenerate of the two — but it should not
be presented as a smooth gradient. If the goal is a metric that grades *how
much* or *what kind* of greening a scenario delivers, neither proxy does that
at this AOI scale; that remains the open 2SFCA-adoption question flagged in
`UNA_METHODOLOGY_CROSS_CHECK.md`. The keep / hide / remove decision should be
made with the step-function behaviour in view.

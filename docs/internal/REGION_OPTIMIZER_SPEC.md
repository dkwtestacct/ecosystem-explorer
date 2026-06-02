# Region-Constrained Optimizer — canonical spec

**Audience:** Internal
**Status:** Current (variant B — surrogate prefilter + engine-verify)
**Use this for:** Understanding why the region-active optimizer takes the shortlist-then-verify shape it does, and what the records / caption / assertions promise
**Do not use this for:** Citywide optimizer behavior (unchanged — see `surrogate.optimize_scenario` / `ARCHITECTURE.md` §5 Layer 3)
**Source of truth for:** The variant-B design, the records-vs-honesty contract, the two machine assertions

---

## 1. Grounding (do not re-measure)

| Measurement | Value | Source |
|---|---|---|
| Full-engine eval over a region | ~2.1 s, **flat regardless of region size** | `scripts/phase0_region_eval.py` |
| Citywide eval (no mask, reference) | ~1.73 s | same |
| Region overhead vs citywide | +24–30 % (region_local block) | same |
| Peak RSS (one eval) | ~912 MB | same |
| Surrogate (Fast, ~90 recipes) ranking ρ vs region engine | 0.83–0.98 across cooling / flood / runoff / carbon / food / nature-access | `scripts/phase0_5_surrogate_ranking.py` |
| Surrogate top-15 recall of engine top-5 | 1.00 across every metric | same |

These numbers fix the budget envelope. **Brute-force over a region does not get cheaper from a smaller region** — the engine computes citywide rasters then clips. The surrogate **does** rank region-scoped candidates cleanly, so a prefilter is signal.

**Implication:** the optimizer must minimize engine calls. K ≈ 30–50 is the working ceiling (K × 2.1 s ≈ 60–105 s); the surrogate's clean top-15 recall says a generous K won't drop true winners.

---

## 2. Mode switch — when this path fires

```
no region/ownership mask active   →  existing citywide surrogate optimizer (unchanged)
region or ownership mask active   →  region-prefilter + engine-verify path (this spec)
```

The new sidebar surface (weight sliders + "Optimize selected area" button) is **rendered only when a filter is active**. The existing min-target sliders + Optimize button are hidden in that case. This is a UI swap, not a mode flag — the presence/absence of the mask drives both branches.

No `KNOWN_DIVERGENCES` entry. The returned scenarios reproduce engine values (validated tier). Search-completeness is a caveat, not a divergence.

---

## 3. Pipeline

```
candidate grid (existing surrogate training set, ~90 recipes in Fast mode)
   │
   ▼
surrogate score every candidate (citywide, region-blind; ordering only)
   │
   ▼
Pareto-efficient set across the 5–6 objective metrics
   │  cap at K ≈ 40 (sized to ~2.1 s × K for a ~2-min target)
   │  if frontier > K, sample for spread (greedy maximin on knob space)
   ▼
engine-verify in-region: evaluate_scenario(recipe, selected_region_mask=combined)
   │  (i / K) progress, ~2.1 s per
   ▼
rank by weighted sum over min-max normalized engine region_local values
   │  weights from user sliders; direction-correct (invert cost + runoff)
   ▼
greedy dedup by knob-distance — top-5 distinct recipes
   │
   ▼
return: 5 records, engine-true values, weights captured, region+ownership stamped
```

**Why Pareto-shortlist, not top-K-by-current-weights.** The K is selected to be weight-robust — moving a weight slider re-ranks the already-engine-evaluated K instantly, no engine re-run. Every re-rank stays engine-true.

**Why K ≈ 40, not 15.** The weakest-ranked metric on Phase-0.5 was food (Spearman ρ 0.83). A tight top-15 would risk dropping a true winner on that axis. A roomy K=40 with the measured top-15 recall = 1.00 gives margin without busting the 2-min budget.

**Cost.** Recipe-deterministic — `compute_cost(n_wet, n_for, n_hd, cost_gi, cost_ff, cost_hd)`. Folded into the Pareto + the weighted ranking using the same machinery as the surrogate-predicted metrics. No surrogate prediction needed for cost (perfect rank correlation by construction).

---

## 4. Objective + normalization

**Weight sliders.** One per objective metric — cooling, flood, carbon, food, cost. (Nature-access optional — wire if it's already first-class in the citywide flow, skip otherwise.) Each 0.0–1.0, default 1.0 (equal-weight). All-zero is reset to equal-weight to avoid an undefined ranking.

**Normalization.** Min-max within the candidate set, direction-corrected:
- Higher-better metrics: `(v - v_min) / (v_max - v_min)`
- Lower-better metrics (cost, runoff): `(v_max - v) / (v_max - v_min)`

Two normalization rounds:
- **Prefilter normalization** runs over the full surrogate candidate grid (~90 recipes). Used only for Pareto identification.
- **Final-rank normalization** runs over the K engine-evaluated candidates. Used for the weighted-sum ranking that produces top-5.

Both rounds use the same direction-correction; the per-round v_min/v_max are different because the candidate sets are different.

**Greedy knob-distance dedup.** After ranking, sweep down the ranked K. Keep #1. For each next candidate, keep it only if its `(Δpct, ΔGI, ΔFF)` L1-distance from every already-kept record exceeds a threshold (suggested: 10 — i.e. at least one knob differs by ≥ 10 %). Stop when 5 distinct records are kept or candidates exhaust.

---

## 5. Records, provenance, honesty (the crux)

**Surfaced values are engine region-local only.** The surrogate produces citywide magnitudes — wrong for a region — used purely for ordering and never displayed.

**Per-record fields:**

```
{
  recipe:                  {pct_converted, gi_pct, ff_pct, hd_pct},
  placement_strategy:      'random',  # only random supported in this pass
  random_seed:             42,
  region_selection:        {layer, selected_ids},   # from session state
  ownership_filter:        OWNERSHIP_MODES key or composite dict,
  weights_used:            {flood, cooling, food, carbon, cost, ...},
  engine_metrics:          full region_local dict from evaluate_scenario,
  source:                  'region_optimized',
  validation:              'engine_verified',
  search_caveat:           'shortlist not exhaustive — top of a surrogate-Pareto K',
}
```

**Display framing.** Surface as **"Best tested mixes — selected area"** — best among the candidates the engine actually tested, not the optimum across all possible mixes. Values are engine-true; the search is over a surrogate shortlist, so global optimality is not claimed.

**Provenance.** A new constant, `PROVENANCE_REGION_OPTIMIZED = "region_optimizer_engine_verified"`, distinguishes region-optimized records from the citywide surrogate's `PROVENANCE_OPTIMIZER`. The two states could otherwise be confused — the citywide surrogate shows predicted values; the region path shows engine-true region-local values, with the surrogate's role limited to shortlisting. The rendered Source labels are correspondingly distinct: **"Engine-verified — region-optimized"** (region path) vs **"Surrogate-suggested"** (citywide path).

On the Apply path, the click handler sets `applied_from_region_optimizer=True` and clears `applied_from_optimizer` (the two flags can't co-fire). The main-panel rerun calls `evaluate_scenario` with the combined region∩ownership mask, producing engine-true values. The Save and Export branches read `applied_from_region_optimizer` BEFORE `applied_from_optimizer` so the new provenance wins. The auto-clear-on-slider-drift logic mirrors the existing citywide flag.

**No new `KNOWN_DIVERGENCES` entry.** The values reproduce the engine (validated tier). The only caveat is search-completeness, and it lives in the caption, not the divergence registry. Don't mislabel the badge — an engine-verified result carries the engine's normal validation status.

**Reuse the existing record infra.** Apply / Save / Export already handle records that carry the active scenario's provenance. No new save schema; no new export bundle field. The `source: 'region_optimized'` + `validation: 'engine_verified'` fields above are internal annotations on the returned shortlist, not new persisted columns.

---

## 6. Conditional caption (replaces `app.py:4883-4889`)

The existing caption ("Suggestions are citywide recommendations and ignore your current selection…") is replaced by a two-state caption keyed off `_filter_active`:

**No filter active:**
> Optimizer uses a fast citywide surrogate to search many candidate mixes.

**Filter active:**
> The fast model shortlists candidate mixes, then the full model evaluates the finalists on your selected area. The results shown are real (engine-verified); the shortlist may not be exhaustive.

Both versions own the truth (what the user sees is real) and the limit (search is surrogate-bounded). The filter-active version is the "pending optimizer caption" memo's resolution.

---

## 7. Runtime UX

- **Prefilter** is instant (surrogate predict over ~90 rows).
- **Engine pass** is K × ~2.1 s, ~60–105 s wall-clock for K = 30–50. Plus a one-time ~70 s cold start on the first run of a session (city load + ~app import).
- **Progress indicator** — `st.progress(i / K)` updated per engine eval. Not a frozen spinner.
- **Cancel.** Not in v1 — Streamlit's spinner/progress doesn't expose a cancel hook cleanly. Document as "ride out the K evals" (≤ 2 min).
- **Memory.** On a 1 GB Streamlit Cloud worker, stream/persist each engine result rather than accumulating K full result dicts (each carries ~5 full-AOI rasters). Strip `scenario_lulc / scenario_lulc_ucm / scenario_lulc_una / scenario_lulc_carbon` after extracting metrics — same pattern as `compute_scenario_grid`.

---

## 8. Machine assertions (the rigor)

Two new cells in `verify_baselines.py`, run once at the end alongside the existing matrix. Engine is untouched, so 40/40 remains byte-identical.

### 8.1 Subset invariant

For each record returned by the region optimizer, the locked subset relation must hold:

```
converted_mask = (BASELINE_LULC != record.scenario_lulc)
converted ⊆ eligible                                    (always)
converted ⊆ region_mask                                 (when region active)
converted ⊆ ownership_mask                              (when ownership active)
|converted| > 0                                          (non-empty — anti-vacuous)
```

The mask is composed via the production helper (`_build_ownership_mask` + region positional index), so a regression in the mask builder surfaces here.

### 8.2 Engine-verified reconciliation (the honesty guard)

For each record, a fresh direct `evaluate_scenario(recipe, selected_region_mask=combined)` call must reproduce the recorded engine metrics (rtol=1e-9 / atol=1e-9 across the standard metric list — same tolerance as the existing round-trip cell).

**Meta-test (load-bearing).** A separate cell injects a surrogate-predicted value into a record's `engine_metrics` field and asserts that the reconciliation cell **fails**. If the meta-test doesn't fail, the reconciliation guards nothing and the assertion is green-light theatre.

### 8.3 Provenance distinction (lock against collapse)

Three guards that the region-optimizer's record class can't silently collapse into the citywide surrogate's:

1. **Constants are distinct** — `PROVENANCE_REGION_OPTIMIZED != PROVENANCE_OPTIMIZER` at the source.
2. **Rendered Source labels are distinct** — `_PROVENANCE_HEADER_INFO[REGION_OPTIMIZED][0] != _PROVENANCE_HEADER_INFO[OPTIMIZER][0]`. The user-facing distinction is what the brief calls out, not just the constant.
3. **Record-shape distinguisher** — every region record carries `source='region_optimized'` + `validation='engine_verified'`; the citywide `optimize_scenario` DataFrame does NOT carry those columns. A regression that adds them to the citywide path would collapse the structural difference and trips the assertion.

All three cells are additive — they don't snapshot, they assert against a freshly-computed truth target.

---

## 9. Gating + discipline

- **Rendered-UI build → HOLD for eyeball, no commit, no push.** The author runs the eyeball checklist (below) and reports findings. The user reviews before any commit.
- **$-escape audit.** Enumerate every new interpolated label carrying `$` (caption, cost in objective, $-metrics in records). Convention: `\\$` inside Python string literals.
- **Orchestration only.** Reuse the existing candidate grid, mask helpers, `evaluate_scenario` region path, and record/Apply/Save/Export. Do not touch the engine. Gate once at the end; don't re-baseline.

**Eyeball checklist:**

1. With no filter active, the citywide optimizer renders identically to today (same sliders, same caption, same Tradeoff Analysis table).
2. With a region filter active, the new sidebar surface renders (weight sliders + "Optimize selected area").
3. The two-state caption reads correctly per state.
4. Engine-verify progress is visible (i / K), not a frozen spinner.
5. The returned top-5 are distinct (knob-distance dedup working).
6. Returned values are engine region-local (not surrogate predictions; no uncertainty bands).
7. `\\$` labels render as `$` (no LaTeX flip).
8. Apply on a returned record: sliders update, main panel re-evaluates engine-true (region+ownership respected), header flips to "Surrogate-suggested," Save / Export carries `PROVENANCE_OPTIMIZER`.

---

## 10. Out of scope (v1)

- **Placement-strategy weights.** Only `random` is searched. The strategy-focused options (cooling-focused, flood-focused, undersupply-focused) are not part of the K-grid.
- **Per-record uncertainty bands.** The shortlist values are engine-true, not surrogate quantiles — no 10th/90th band to show. The tradeoff chart's "diamonds with bars" stays on the citywide optimizer path.
- **High-resolution / Balanced surrogate.** Phase-0.5 validated **Fast** (~90, 100 trees) only. Other modes get the same treatment by reuse (the grid + estimator come from `_cached_train_surrogate`), but only Fast is the validated configuration.
- **Cancel / pause.** The K-eval pass runs to completion in v1.

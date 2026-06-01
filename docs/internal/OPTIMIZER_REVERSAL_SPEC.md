# Optimizer Reversal — Build Spec

**Audience:** Internal
**Status:** Building — reverses an existing guard; adds a sharper honesty caption + a new subset-invariant matrix cell. Zero human gates.
**Depends on:** Region Selection (live), Ownership Filter (SA live), Subset Invariants Pass (live — adds one matrix cell), Scenario Record Pass (live), surrogate optimizer (live).
**Builds:** Re-enable the optimizer when a region or ownership filter is active. The surrogate stays citywide-trained; the Apply path already evaluates suggestions through `evaluate_scenario` under the active masks (Phase 0 confirmed).
**Source of truth for:** the reversal decision, the honest mismatch framing, and the new subset-invariant cell that locks in correctness for optimizer-applied scenarios.

---

## Why

Today the Optimize button is disabled when `selected_region_mask` or `selected_ownership_mask` is set (`app.py:4576-4598`). The historical rationale: the surrogate is trained on citywide pixel pools, and a region-restricted scenario would diverge from its predicted metrics — so disable rather than mislead.

That's too cautious. The honest version is: **let the optimizer run**, surface its suggestions clearly as *citywide* recommendations, and tell the user that Apply re-evaluates under their filters and will produce different numbers. The user keeps optionality (the optimizer is still useful as "what conversion mix does a citywide-trained model think is efficient?"), and the honesty is sharper because it explains *why* the predicted numbers differ rather than just hiding behind "clear the region to use it."

The Apply path already routes through `evaluate_scenario` with `selected_region_mask=_combined_mask` (Phase 0 confirmed at `app.py:4751-4757`), so applying a suggestion under a region or ownership filter produces a *correctly masked* scenario. The optimizer's predicted metrics drift from the applied result — that's the surrogate-vs-engine + citywide-vs-region gap — but the post-Apply numbers are engine-validated.

## The change

### Remove the disable guard

`app.py:4576-4598`:

- Drop the `_optimizer_blocked_by_placement` boolean.
- Drop `disabled=_optimizer_blocked_by_placement` from `st.button("Optimize", ...)`.
- Drop the trailing `if _optimizer_blocked_by_placement: st.caption(...)` block.

### Replace with the honest mismatch caption

Below the Optimize button, render only when region or ownership is active:

> "Suggestions are citywide recommendations and ignore your current selection; predicted values won't match the region-applied result. Apply a suggestion to evaluate it under your filters."

Renders alongside (not replacing) the existing "predicted, not final" framing on the optimizer panel itself. The new caption sits next to the Optimize button (sidebar) so it's read before clicking Optimize; the optimizer-panel caption stays where it is for the Apply step.

## What stays

- **Surrogate stays citywide-trained.** No region/ownership parameter into `optimize_scenario`. Region-trained optimization is Phase 2.
- **Apply path** routes through `evaluate_scenario` with the active combined mask. No change.
- **OPTIMIZER provenance flag.** Apply path stamps `applied_from_optimizer=True`; comparison-table and audit rows continue to read it back as "Surrogate-suggested". No change.

## Honesty framing

The crux: **the optimizer's surrogate doesn't know about region or ownership.** Its 10,000-sample exploration scores pct/GI/FF combos against citywide-aggregate metric predictions. When the user has District 5 + vacant_public active, the surrogate's #1 might be a 30 % / 50 / 50 mix that would behave very differently inside that 1,514-pixel slice than it does citywide.

Two captions cover this:

- **Sidebar (next to Optimize button), conditional on filter active:** "Suggestions are citywide recommendations and ignore your current selection; predicted values won't match the region-applied result. Apply a suggestion to evaluate it under your filters."
- **Optimizer panel (Tradeoff Analysis tab), unchanged:** "These are surrogate model predictions. Click Apply to run a full validation against the prototype engine."

The first one teaches the *region-specific* mismatch; the second one teaches the *surrogate-vs-engine* gap. Both apply when filters are active.

## Verify

### Subset-invariant matrix — new cell

Add to `verify_baselines.py`'s subset-invariant matrix (SA only — only SA has ownership data):

- **Cell:** "SA / optimizer-applied recipe under region + ownership"
- **Setup:** Pick a representative optimizer-style recipe (e.g. pct=30 GI=60 FF=40 — high conversion, mixed GI/FF — the kind of mix the optimizer would actually recommend); SA District 5 + ownership=vacant_public.
- **Pass mask:** `region ∩ ownership` (combined, as the live app does).
- **Three subset checks:** converted ⊆ eligible, converted ⊆ region, converted ⊆ ownership. Same gate the other cells run.
- **Funnel reconciliation:** same as other cells.

Total matrix: SA gains one cell (now 7); MN unchanged at 4. Grand total: 11 cells.

### Existing gates

`verify_baselines.py` 40/40 byte-identical (no math touched), all existing assertions hold, extended matrix passes.

### Eyeball

- Pick SA, select District 5 + ownership=vacant_public.
- Optimize button is now **enabled**; click it.
- Tradeoff Analysis tab shows suggestions with the existing "predicted, not final" caption.
- Apply a suggestion.
- Scenario tab updates; provenance reads "Surrogate-suggested"; metric cards reflect the region+ownership-evaluated result; **the funnel's "Converted" row ≤ "After ownership filter" row** (subset invariant in action).
- Eligible/Converted acres on the audit expander align with the funnel.

## Out of scope

- **Region-trained surrogate / true region-constrained optimization** — Phase 2. Would require feeding the region+ownership pool into `optimize_scenario`'s sampling and re-training (or per-region surrogates).
- **Reporting predicted-vs-applied drift** — could be a future cell on the comparison table; not in this batch.
- **Disabling the optimizer for the no-feasible-pool degenerate case** (e.g. selected region has 0 convertible pixels). If that happens today the optimizer suggestion's pct%×0px=0 conversion is harmless; no special-case needed.

## Not touched

- `evaluate_scenario`, `optimize_scenario`, schema version, save handler, comparison table, audit expander, funnel renderer.
- The existing OPTIMIZER provenance / Source-line suffix machinery.

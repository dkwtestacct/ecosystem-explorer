# Open Questions

Outstanding asks, parked decisions, and external dependencies. Nothing here has
been sent or actioned externally — these are captured so the context isn't lost.

---

## NatCap data requests

### 1. Per-scenario compound LULC inputs · STATUS: PARKED, not sent

**Need.** The compound-encoded (NLCD×NLUD×tree) per-scenario LULC rasters —
`FF_20ac`, `FF_40ac`, `FF_MAX`, `UA_20ac`, `UA_40ac`, `UA_MAX` — that NatCap fed
to the carbon and urban-cooling models to produce the
`c_sequestration_<scenario>` and `avg_temp_f_<scenario>` cells in
`nootenboom_results/citywide_results_UPDATED.xlsx`.

**Why it matters.** Un-gates carbon/temperature reproduction — the only two
`natcap_published` metrics in `data/sa/natcap_reference_outputs.csv`. Without
these inputs, those two metrics stay at a **"NatCap published"** display (we show
NatCap's number), never **"✓ NatCap match — independently reproduced"** (we run
the prototype on NatCap's input and confirm the delta). The scenario rasters we
already have (`sa_lc_w_*_10m.tif`) are flood-encoded (NLCD×tree 3-tier), which
the Carbon/UCM/UNA compound-keyed tables cannot consume. See DESIGN_NOTES.md
"Brief B1".

**Status / resolution paths.**
- A local content-signature hunt (2026-05-29) across `~/Desktop`, `~/Downloads`,
  `~/Documents`, `/Volumes`, the Google Drive sync root, and the `_zip_archive`
  zips found **only baseline compound rasters** — no scenario variants. So the
  request is the path to un-gate, **but it is parked, not sent.** Send the draft
  below only on an explicit decision to reach out.

**Send-ready email draft (verbatim — do not send without a go-ahead):**

> **Subject:** SA urban-ag project — per-scenario LULC inputs behind the carbon & temperature results
>
> Hi [contact],
>
> We're building on the San Antonio urban-agriculture scenario work and independently reproducing a couple of the citywide results on our side as a cross-check. From the August 2024 drive share we have the baseline LULC overlay, the per-model biophysical tables, and the published citywide outputs in `nootenboom_results/citywide_results_UPDATED.xlsx`.
>
> What we're missing are the **per-scenario LULC rasters that were fed into the carbon and urban-cooling models** to produce the per-scenario values in that spreadsheet — i.e. the inputs behind the `c_sequestration_<scenario>` and `avg_temp_f_<scenario>` cells for:
>
> - **FF_20ac, FF_40ac, FF_MAX** (food-forest scenarios)
> - **UA_20ac, UA_40ac, UA_MAX** (urban-ag / garden scenarios)
>
> The scenario rasters we already have (`sa_lc_w_*_10m.tif`) look like the flood/NDR inputs — they don't carry the land-use dimension the carbon/cooling tables key on, so we can't reproduce the carbon/temperature numbers from them.
>
> Could you share:
> 1. The per-scenario LULC rasters used for the **carbon** and **urban-cooling** runs (the six scenarios above), **in whatever encoding and grid you ran them in** — we can reconcile CRS / resolution / encoding on our end as long as we know the scheme.
> 2. The matching **biophysical / carbon-pool table**, if it differs from the ones already in the August 2024 share (`carbon__nlcd_nlud_tree.csv`, `ucm__nlcd_nlud_tree.csv`).
>
> If those scenario LULCs sit under different internal names than the flood inputs, even a pointer to where they live in the project structure would help.
>
> Thanks!

---

### 2. Native NLCD×tree baseline flood raster · STATUS: open question (secondary, lower priority)

**Need.** NatCap's *baseline* LULC in the same native NLCD×tree 3-tier encoding
as the scenario rasters (the baseline the `sa_lc_w_*` scenarios were built from).
NatCap likely has one, since each scenario is that baseline with a small acreage
converted.

**Why it's lower priority.** Flood has **no NatCap published reference** (the
reference CSV's flood rows are all `aligned_method`, no value), so this is a
**UX-comparability** question, not a validation one. It surfaced from the Brief
B1 smoke test: NatCap's scenario rasters yield `mean_cn ≈ 81.4` (flood ≈ 18.6),
but the prototype's own SA baseline is `mean_cn 76.54` (flood 23.5) — a ~5-point
gap traced to the prototype's compound→NLCD×tree reduction (`tier = max(tree,1)`)
producing a different canopy-tier mix than the native raster. A fixed "food
forest" scenario therefore currently reads as *lower* flood retention than the
baseline, which would confuse users.

**Local fallbacks (no external dependency required):**
1. **Reconcile the derivation** — compute the prototype's SA baseline flood
   through the same native NLCD×tree path the loaded scenarios use, so baseline
   and scenarios are on one footing.
2. **Suppress the fixed-scenario flood delta** — display the scenario's flood as
   "≈ invariant" (matching NatCap's documented "essentially no difference"
   finding) rather than a signed delta vs the prototype baseline.

To be decided whenever a fixed-scenario flood card is built (see "B2 — Per-metric
validation markers · DEFERRED" below). See DESIGN_NOTES.md "Brief B1".

---

## Deferred briefs

### B2 — Per-metric validation markers · STATUS: DEFERRED (2026-05-29)

**What it was.** Per-metric validation badges on the dashboard cards: ✓ NatCap
match / × Diverged X% / ≈ Aligned method / Prototype, driven by
`data/sa/natcap_reference_outputs.csv` + `natcap_validation.py`.

**Why deferred.** The badges that were the point — per-scenario **Match /
Diverged** — require a prototype value *computed for a NatCap fixed scenario* to
compare against NatCap's published value. The available data doesn't support
that, and may never:
- The only two `natcap_published` metrics (`temp_change_f`, `carbon_tons_co2`)
  are compound-keyed (UCM, four-pool carbon); NatCap's scenario rasters are
  flood-encoded (NLCD×tree), so the prototype can't compute them for those
  scenarios. The compound inputs that would un-gate reproduction are parked
  (see "Per-scenario compound LULC inputs" above) and may not arrive.
- The other metrics are `aligned_method` / `prototype` with no per-scenario
  NatCap value, so they can only ever show a methodology label, not a match.
- `validation_status` is constant per metric across scenarios, so absent
  Match/Diverged the badges reduce to a **relabeling of the existing
  high/medium/prototype confidence captions** — and that relabel would lose the
  methodology-rigor distinction and mislabel the non-CSV cards (runoff, carbon-$,
  etc.). Not worth shipping (Phase 0 finding #4).

**Revisit only if** the parked compound scenario inputs arrive (un-gating
carbon/temp reproduction). Even then, the likely shape is a **reworked, smaller
surface** — baseline reproduction + a NatCap reference comparison table — **not**
the original per-card, per-scenario badge design. The B1 scaffolding
(`natcap_scenarios.py` loader + provenance taxonomy + flood helper) and
`natcap_validation.py` (lookup/compare helpers) remain in tree, ready to build on.

**No code shipped for B2.** Phase 0 only (investigation + this deferral record).

---

#### Preserved Phase 0 design work (reusable if B2 is reworked)

Captured so a future session doesn't redo it. All line numbers are as of commit
`436bffd` and will drift.

**(1) Card inventory + fixed-scenario classification.** The dashboard renders 16
`st.metric` cards (app.py ~3735–4292), each followed by `_confidence_caption(col,
tier)`. For a NatCap *fixed* scenario, each card was classified as **published**
(show NatCap's value from the reference CSV), **computed** (flood path via the B1
helper), or **unavailable** (compound-encoding-gated or no reference):

| card var | label | `results[...]` key | tier now | CSV metric → status | fixed-scenario |
|---|---|---|---|---|---|
| eco1 | Flood Retention | `flood_reduction` | high | `flood_reduction` → aligned | **computed** + reconcile |
| eco2 | Temperature Change | `temp_change_f` | high | `temp_change_f` → natcap_published | **published** (Δ) |
| eco3 | Runoff Volume | `runoff_acre_feet` | high | — | computed (flood path) / fold into flood |
| eco4 | Carbon | `carbon_tons_co2` | four_pool/proto | `carbon_tons_co2` → natcap_published | **published** (Δ) |
| eco5 | NDVI | `mean_ndvi` | prototype | — | **unavailable** |
| hs_na | Nature Access | `nature_access_pct` | medium | `nature_access_pct` → aligned | **unavailable** (compound) |
| hs3 | Preventable MH Cases | `preventable_mh_cases` | high | `preventable_mh_cases` → aligned | **unavailable** (compound/NDVI) |
| hs4 | Avoided MH Costs | `avoided_mh_cost_usd` | high | (pairs w/ MH) → aligned | **unavailable** |
| econ1 | Food Production | `food_mln_lbs` | prototype | `food_mln_lbs` → prototype | **unavailable** (code 998≠41; no ref) |
| econ2 | Est. Implementation Cost | `total_cost_mln` | medium | — | **unavailable** (slider/mix artifact) |
| econ3 | Flood Damage Avoided / Volume Reduction | `flood_damage_avoided_usd` / `flood_reduction` | medium | — | computed (SA → "Volume Reduction") + reconcile |
| econ4 | Cooling Energy Savings | `cooling_energy_savings_usd` | medium | `cooling_energy_savings_usd` → aligned | **unavailable** (UCM compound) |
| econ5 | Carbon Storage Value ($) | `carbon_value_usd` | medium | carbon × SC-CO2 → published-derived | **published** (derived) |
| ceff1–3 | Cost-Effectiveness ratios | `ce[...]` | medium | — | **unavailable** (inputs unavailable) |

Net per fixed scenario: ~4–5 cards carry a value (temp, carbon, carbon-$ →
published; flood, runoff → computed); ~11 are "not available."

**(2) Recommended architecture — (b2) dedicated reference-view.** `results` is
built once at app.py:3569–3606 (lookup+`_fresh`, else `evaluate_scenario`) then
consumed by the 16 cards + tradeoff plot + map with heavy inline delta/pill math
that would choke on `None`. Rather than populate every key with sentinels and
guard each card (pervasive, regression-prone, 11 grey cards), add a separate
`render_natcap_fixed_scenario_view(scenario_id)` that a sidebar **scenario-source
selector** (SA-only) routes to *instead of* the Explorer panel. It renders only
the meaningful cards (temp/carbon/carbon-$ published; flood/runoff computed) plus
a compact "not available for NatCap fixed scenarios (pending compound inputs):
Nature Access, Cooling Energy, Mental Health, Food, NDVI, Cost…" note. Explorer
path untouched; no optimizer/tradeoff/save (those are surrogate/slider-based).
Provenance + scenario_id come from `natcap_scenarios.py`; published values from
`natcap_validation.py` over `natcap_reference_outputs.csv`.

**Badge taxonomy (5 states).** ✓ NatCap match / × Diverged X% (both require
reproduction — gated), ≈ Aligned method, Prototype, and the interim **"NatCap
published"** (show NatCap's value, no match/diverged) for temp/carbon until
reproduction un-gates.

**(3) Three open decisions + recommendations** (for whoever reworks B2):
1. *Reference-view layout* — **(b2) compact dedicated view [recommended]** vs
   (b1) full 16-card grid with ~11 "not available" cards (lots of dead space).
2. *Confidence-badge replacement scope* — **(i) validation badges on the
   fixed-view only, keep confidence badges on Explorer [recommended]** vs (ii)
   unified hand-mapped taxonomy everywhere. (ii) collapses the high/medium
   distinction (e.g. MH "high"→"aligned") and mislabels non-CSV cards (runoff,
   carbon-$ fall to "Prototype" via the "no row" rule though InVEST-derived), so
   it needs a curated per-card map, not a raw CSV lookup.
3. *Flood reconcile (Brief B1 ~5-pt CN gap, native 81.4 vs prototype baseline
   76.54)* — **(i) suppress the fixed-scenario flood delta, show "≈ invariant"
   [recommended]** (matches NatCap's finding, low-risk) vs (ii) re-derive the SA
   baseline flood through the native NLCD×tree path (more correct, but a
   methodology change with wider blast radius).

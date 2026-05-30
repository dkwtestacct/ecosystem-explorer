# Open Questions

**Audience:** Internal
**Status:** Current
**Use this for:** Durable, unresolved questions and external blockers — only
**Do not use this for:** Meeting prep (→ DEMO_AND_COLLABORATION.md), code/UI tasks (→ GitHub issues), or decided design questions (→ DESIGN_NOTES.md)
**Source of truth for:** What still truly blocks progress

> Only durable, unresolved questions and external blockers belong here. Meeting prep → DEMO_AND_COLLABORATION.md. Code/UI tasks → GitHub issues. Design questions, once decided → DESIGN_NOTES.md.

---

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
  zips found **only baseline compound rasters** — no scenario variants.
- A Google Drive **connector** search (by name + `_3857` suffix) corroborates:
  the per-scenario compound overlays are **not in the shared Drive** either. What
  IS shared is the flood-encoded scenarios (`sa_lc_w_*_10m.tif`), the **compound
  baseline** (`lulc_overlay_3857.tif`), and its component layers
  (`nlcd_3857.tif`, `nlud_3857.tif`, `tree_3857.tif`). Conclusion: NatCap built
  the per-scenario **compound** LULCs as **unsaved pipeline intermediates** — they
  were generated, fed to Carbon/UCM, then not persisted.
- **Sharpened ask (parked, not sent):** either (a) the six per-scenario compound
  overlays if they can be regenerated/recovered, **or** (b) **Nootenboom's overlay
  script** that composes NLCD×NLUD×tree-canopy into the compound LULC — owner
  **cnootenb@umn.edu**. With the script we can rebuild the scenario overlays
  ourselves from the shared component layers.
- **Option 2 (local reconstruction) — possible, with a caveat.** We could rebuild
  the compound scenario overlays from `nlud_3857` + `tree_3857` + the flood
  scenario rasters. But the converted-pixel codes (`998` food forest / `999`
  garden) have no native compound class, so their NLUD×tree mapping would be
  **inferred**, not authoritative — a methodology assumption we'd have to own and
  document. Prefer (a)/(b) over reconstruction where possible.
- **Still parked, not sent.** Send the draft below only on an explicit decision.

**Impact if the compound scenario inputs are never obtained.**
The missing files are the per-scenario compound (NLCD×NLUD×tree) LULC rasters for
the six NatCap fixed alternative scenarios (FF_20ac, FF_40ac, FF_MAX, UA_20ac,
UA_40ac, UA_MAX). Without them, exactly one capability is foreclosed:

> The prototype cannot independently reproduce — and therefore cannot validate —
> its carbon and temperature outputs against NatCap's published per-scenario
> values for those six scenarios. Computing the prototype's own carbon/temp for a
> fixed scenario requires running the compound-keyed Carbon/UCM models, which need
> a compound scenario raster that does not exist on our side (NatCap built them as
> unsaved pipeline intermediates).

Downstream consequences (all the same fact):
- **B2** (per-metric validation badges): the per-scenario "✓ NatCap match (Δ X%)" /
  "× Diverged" states cannot exist for carbon/temp on the fixed scenarios. (B2 is
  deferred for this reason.)
- **B1 Phase 3** (verify prototype scenario carbon/temp deltas vs NatCap
  published): remains gated.
- **D1** (InVEST export): the six fixed alternative scenarios export flood-only
  (UFR args + flood-encoded source raster), with carbon/UCM/UNA args marked
  "unavailable" in metadata. Baseline, Explorer-generated, and Optimizer-suggested
  scenarios are unaffected and export the full five-model bundle.

**Ceiling even if obtained:** the files would only enable carbon + temperature
validation (the only two `natcap_published` metrics) on those six scenarios.
Nature access, cooling, and mental health have no NatCap published per-scenario
target, so they could not be validated against NatCap regardless.

Not affected (intact without the files):
- **Per-pixel parity vs canonical InVEST** — HMI MAE 0.0000 (Brief 28b), UMH
  MAE ≈ 0 (Brief B), measured against canonical InVEST 3.19.0. This is the
  validation credibility anchor.
- All **Explorer-generated and Optimizer-suggested** scenario exploration and full
  five-model InVEST export (the prototype builds their compound rasters internally
  via `evaluate_scenario`).
- Displaying NatCap's published per-scenario carbon/temp as **reference values**
  (the 14 figures are in `natcap_reference_outputs.csv`).
- Flood metric on the fixed scenarios.

**What is NOT reproducible from disk** (Brief B2 revised investigation, 2026-05-29):
NatCap's published *citywide absolute* baseline figures — `avg_temp_f` = 90.08 °F
and `c_sequestration` = 107.32M t CO2e. Their SA UCM `args.json` isn't shipped, and
`tot_c_cur.tif` doesn't aggregate to 107.32M by any standard interpretation. The
reproduction claim sits at **per-pixel parity**, not at citywide absolute.

**Validation story this leaves us with:** "the prototype reproduces canonical
InVEST per-pixel, uses canonical InVEST methodology, and surfaces NatCap's own
published scenario outcomes" — narrower than citywide-absolute reproduction, but
fully honest and intact.

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
> The scenario rasters we already have (`sa_lc_w_*_10m.tif`) look like the flood/NDR inputs — they don't carry the land-use dimension the carbon/cooling tables key on, so we can't reproduce the carbon/temperature numbers from them. The shared Drive has the compound *baseline* overlay (`lulc_overlay_3857.tif`) and its `nlcd_3857` / `nlud_3857` / `tree_3857` components, but not the per-scenario compound overlays — it looks like those were pipeline intermediates that weren't saved.
>
> Either of these would unblock us:
> 1. The six **per-scenario compound overlays** (the NLCD×NLUD×tree LULCs fed to carbon + urban-cooling), if they can be regenerated or recovered — in whatever encoding/grid you ran them; we can reconcile CRS/resolution on our end.
> 2. **The overlay script** that composes the compound LULC from the NLCD / NLUD / tree-canopy layers (Chris Nootenboom may own this) — with it we can rebuild the scenario overlays ourselves from the shared component layers.
>
> A pointer to either is plenty — thanks!

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

> **2026-05-29 update — partial unblock.** B2 was revised mid-session under a
> conservative-floor scope that DROPS the gated Match/Diverged states and
> delivers the ungated core: a three-state badge taxonomy (NatCap-anchored /
> NatCap method / Aligned method / Prototype), a dedicated SA fixed-scenario
> reference view, a cross-scenario comparison table, and a plain-line baseline
> validation claim. **That work landed** — see `DESIGN_NOTES.md` "Brief B2
> (revised)". The **original** match/diverged design described below remains
> deferred (compound-input-gated), and the preserved Phase-0 design lower in
> this entry stays a useful reference if the gated piece is ever rebuilt.

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

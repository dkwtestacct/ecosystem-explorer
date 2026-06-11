# STRATEGY.md — Ecosystem Explorer

**Audience:** Internal
**Status:** Current source of truth
**Use this for:** Big-picture framing, prioritization, value ladder, validation/exploration narrative
**Do not use this for:** Metric definitions, data file paths, or implementation details
**Source of truth for:** What Ecosystem Explorer is trying to be

---

*Strategic framing, prioritization, and working principles. Captured 2026-05-29; revised 2026-05-29 after the B2 validation investigation (see §4, §8).*

This document captures the conceptual scaffolding that the operational docs (../../CLAUDE.md, DESIGN_NOTES.md, NATCAP_ALIGNMENT.md, ../archive/HISTORY.md) reference but don't themselves articulate. If you're a future Claude or a future Deborah picking this up cold, read this first.

---

## 1. Center of gravity

**Ecosystem Explorer is a validated scenario exploration and discovery layer for urban InVEST analyses.**

It can:
1. Surface NatCap's published project-scenario outcomes as labeled reference anchors, and visualize them alongside prototype scenarios
2. Compare them in a common dashboard alongside user-generated scenarios
3. Use surrogate optimization to identify additional scenarios worth testing in future stakeholder conversations
4. Export selected scenarios as runnable canonical InVEST input bundles for full-resolution validation

**Tagline:** *Validated scenario exploration for urban ecosystem tradeoffs* — where "validated" means the engine matches canonical InVEST per-pixel (MAE≈0; §4), not that the prototype reproduces NatCap's published citywide numbers.

**Four-line narrative:**
> NatCap scenarios establish trust.
> Explorer scenarios expand the design space.
> The optimizer discovers promising alternatives.
> Full InVEST runs validate the candidates.

*"Establish trust" = the engine is validated against canonical InVEST and NatCap's own published values are shown as the labeled reference — not that the prototype independently reproduces NatCap's citywide figures. See §4 for the precise claim.*

---

## 2. What this app is NOT

Three things this app explicitly is not, and shouldn't drift toward:

- **Not a viewer of NatCap outputs.** That's the ceiling if validation becomes the whole identity. Validation grounds the optimizer, doesn't replace it.
- **Not a replacement for InVEST.** The optimizer suggests; canonical InVEST validates. The prototype is the discovery layer, not the source-of-truth.
- **Not a final decision engine.** The optimizer is a *scenario discovery engine*, surfacing options worth bringing into stakeholder conversations — not telling planners what to do.

---

## 3. Value ladder

Level 1 — **Viewer**: Show NatCap project outputs interactively.
Level 2 — **Comparator**: Compare baseline, project scenarios, and user-saved scenarios.
Level 3 — **Explorer**: Modify scenario assumptions and see tradeoffs.
Level 4 — **Scenario discovery engine**: Use surrogate search to identify promising alternatives.
Level 5 — **Workflow layer**: Feed selected candidates back into full InVEST runs, ROOT, stakeholder review.

The prototype now spans Levels 2–5: the cross-source comparison table makes L2 concrete, scenario provenance is surfaced throughout (L3), the optimizer is framed and plumbed as the L4 scenario-discovery engine, and Brief D1 (Export for InVEST) makes L5 partly real. Center of gravity: L4.

---

## 4. Validation taxonomy

Two questions have been conflated in this project, and keeping them separate is the entire point of being honest about validation:

1. **Provenance** — is the number on the card NatCap's own published value, or the prototype's own computation?
2. **What's been measured** — has the prototype's *engine* been checked against canonical InVEST, and (separately) has any prototype *output* ever been compared against a NatCap published value?

The dashboard surfaces this per metric × scenario via badges, with statuses recorded in `data/<city>/natcap_reference_outputs.csv`. The badge floor (B2-revised, 2026-05-29):

**Green — "NatCap published value"**
Reserved for the fixed-scenario reference view, where the card displays NatCap's own published number directly from the reference CSV. This is *not* a reproduction claim — we are showing NatCap's figure, not independently arriving at it.

**Blue — "≈ NatCap method"** (a `natcap_published`-class metric on the prototype's own computation: baseline / Explorer / Optimizer)
NatCap also publishes this metric, and the prototype computes it with canonical InVEST methodology — but the displayed value is the prototype's own, for a scenario with no NatCap anchor. Metric-aware tooltip:
- *Temperature* can cite measured per-pixel HMI parity vs canonical InVEST UCM (MAE 0.0000, r 1.0000 — Brief 28b).
- *Carbon* cites measured per-pixel parity too — the four-pool stock framework (Brief 30) is validated vs canonical InVEST 3.19.0 at MAE ≈ 0 / r 1.0 in matched units (Relay 69, `compare_carbon_sa_fourpool_invest.py`).

**Blue — "≈ Aligned method"** (`aligned_method`)
Canonical InVEST methodology, but no directly-comparable NatCap citywide reference, or the framing differs (statistic / scope / aggregation). Currently: SA UNA (per-block-group aggregation needed for citywide comparison), SA cooling energy (scope difference), SA flood (canonical UFR, no NatCap published value), UMH (canonical kernel parity at MAE≈0, but synthetic NDVI proxy).

**Gray — "Prototype"** (`prototype`)
Exploratory or proxy methodology, no canonical InVEST analog. Currently: food production (food-forest yield × area benchmark).

### What `natcap_published` does and does NOT mean

`natcap_published` is a **provenance marker** — "NatCap publishes a reference value for this metric" — *not* a verification result. As of the 2026-05-29 investigation:

- **The engine is validated against canonical InVEST** per-pixel (UCM/UNA/UMH at MAE≈0). This is real and measured.
- **No prototype output has ever been compared end-to-end against a NatCap published value.** `natcap_validation.compare_to_reference` exists and the CSV holds NatCap's numbers, but the only callers are a hardcoded smoke test. The two `natcap_published` metrics (temp, carbon) are exactly the ones gated by the unavailable compound scenario inputs, so the comparison was queued behind missing data the whole time. Status: **comparison-ready, never executed.**
- **NatCap's published citywide absolutes are not reproducible from what's on disk** (see §8) — so even the absolute baseline doesn't reproduce, for parameter/scope reasons, not methodology divergence.

The honest one-line claim: *the prototype's engine reproduces canonical InVEST per-pixel; it does not independently reproduce NatCap's published citywide figures; and it transparently displays NatCap's own published values where they exist.*

**Why this matters:** "validated where possible, exploratory where valuable" depends on per-metric honesty. A green badge means "this is NatCap's number," not "we matched NatCap." Letting `natcap_published` read as "verified against NatCap" is the precise overclaim this taxonomy exists to prevent.

---

## 5. Two-bars principle

Two changelog audiences, two bars:

**WHATS_NEW_SECTIONS** (in-app, strict bar) — Returning dashboard user would notice it; one line; no internal vocabulary or parameter values. Grouped into capability sections (e.g. "Interactive scenario placement", "Validation and handoff"). Reserved for user-visible feature changes, confidence-level changes, architectural shifts users see. Data/model-update implementation history (e.g. switching a per-city baseline source) lives in the changelog, not here.

**Collaborator-facing docs** (NATCAP_ALIGNMENT.md, DESIGN_NOTES.md, ../archive/HISTORY.md) — Looser bar. Methodology milestones, validation results, schema bumps, internal taxonomy. The institutional record.

**Operational implication:** A methodology change might be major-for-WHATS_NEW (because the user-visible numbers shift) but a confidence upgrade alone might not be (if the numbers don't move). CC's judgment per commit.

---

## 6. Scenario sources and provenance

Five scenario sources, all displayed in the same comparison dashboard with explicit provenance badges:

- **Baseline** (existing conditions)
- **NatCap project scenario** (provided LULC raster — SA: FF_20ac, FF_40ac, FF_MAX, UA_20ac, UA_40ac, UA_MAX)
- **Explorer-generated scenario** (slider parameters + placement strategy)
- **Optimized scenario** (surrogate-suggested, can be predicted or full-raster-evaluated)
- **Saved scenario** (user-named, derived from any of the above)

UX principle: **hide the seam (no mode toggle); expose scenario provenance (badges, not modes).** A planner shouldn't have to ask "am I in the right mode?" — they should see scenarios labeled by source.

---

## 7. Prioritization map

Tracks and their dependency order. Strikethrough = completed.

**Track A — Foundation**
- ~~A1: UMH validation harness + kernel fix~~ (Briefs A + B, pushed: db94098 + Brief B + 736756d)
- A2: SA UNA AOI investigation — landed as doc-only (736756d). The "AOI mismatch" turned out to be a per-block-group aggregation question, not a config swap. See `../research/una/SA_UNA_BIOPHYSICAL_EXTENT.md` (the durable Brief A2 single home); `NATCAP_ALIGNMENT.md` §4 + `CITY_PARITY.md` SA section carry the parity-claim summary.
- ~~A3: `natcap_reference_outputs.csv` schema and population~~ — **landed**. temp + carbon `natcap_published`; nature_access + cooling_energy `aligned_method`; flood + UMH `aligned_method`; food `prototype`. Built by `extract_natcap_reference_outputs.py`, read via `natcap_validation.py`. **Caveat (see §4):** `natcap_published` here is comparison-*ready*, never executed end-to-end — the temp/carbon metrics are gated by the unavailable compound inputs, so no prototype value has ever been pushed through `compare_to_reference` against a NatCap value.

**Track B — Keystone**
- ~~B1: NatCap fixed scenarios as first-class inputs~~ — **landed partial** (436bffd): loader + provenance taxonomy (`PROVENANCE_BASELINE`/`_NATCAP_FIXED`/`_EXPLORER`/`_OPTIMIZER`) + pure `flood_reduction_from_nlcd_tree` helper. Carbon/temp reproduction for the six fixed alternatives is gated (compound scenario inputs unavailable — NatCap built them as unsaved pipeline intermediates; see `OPEN_QUESTIONS.md`).
- ~~B2: Per-metric validation markers in dashboard~~ — **landed as B2-revised** (conservative floor; see `DESIGN_NOTES.md` §8.1 "Two-surface validation vocabulary"). Four-state badges wired across all metric cards, an SA fixed-scenario reference view, a cross-scenario comparison table, and a plain-line baseline validation claim. The original Match/Diverged design stays deferred — gated on the same unavailable compound inputs (see DESIGN_NOTES §11.5).
- ~~B2a: In-app validation-status note~~ — **landed** (b9d6600). Plain-language "validated vs displayed vs exploratory" note surfaced in the app; C1 recorded as frozen in §7 + §8 below.
- ~~B2b: Scenario provenance header~~ — **landed** (9fca481). Every scenario shows a Source + Validation header driven by the `PROVENANCE_*` taxonomy, wired into the main dashboard and the fixed-scenario reference view.
- ~~B2c: Cross-source comparison table~~ — **landed** (0dc4726). NatCap anchors + current + saved scenarios side by side at the top of the Tradeoff Analysis tab; mandatory Source/Validation columns, uniform Δ-vs-baseline basis, "—" for compound-gated NatCap cells.
- B3: Canonical flood-volume output for SA alongside index (needs investigation pass first).

**Track C — Payoff**
- C1: Parity validation across UCM/UNA/Carbon for fixed scenarios — **effectively closed; not a live track.** Reproducing the fixed alternatives needs the compound scenario inputs (unavailable), and NatCap's published citywide figures aren't recoverable from disk either: their UCM args aren't shipped, and the carbon aggregation behind the published 107.32M isn't either (see §8). Per-block-group UNA aggregation is the one piece that *could* be computed independently, but without the compound inputs there's no fixed-scenario parity to validate it against. Revisit only if NatCap shares the compound LULCs / args.

**Track D — Strategic addition**
- ~~D1: Export for InVEST workflow~~ — **landed**. Phase 3 verification passed: all five InVEST 3.19.0 urban models (UCM/UNA/UFR/Carbon/UMH) execute cleanly on the SA baseline bundle. Baseline / Explorer / Optimizer export the full five-model bundle (the prototype builds the compound raster internally via `evaluate_scenario`); the NatCap fixed alternatives export flood-only (compound inputs gated). Per-model `args.json`, polymorphic metadata block.
- ~~D2: Optimizer as scenario discovery~~ — **landed** (faa6d4d). "Find Best Scenario" reframed as discovery; an applied optimizer suggestion is labeled as such in the scenario header and records `optimizer_suggested` provenance in its InVEST export bundle. Verified end-to-end via Playwright.

**Track E — Optional**
- E1: NDR for fixed scenarios. Inputs concrete (ndr_biophysical_parameters_vNLCDTree_SA.csv, etc.). Only run for baseline + 20ac + 40ac, not arbitrary slider scenarios.
- E2: Status update to Yingjie — DROPPED. WHATS_NEW is the canonical record; if it's important enough to email, it should be in WHATS_NEW first.

**Track F — Defer / opportunistic** *(no longer "post-symposium" — see §11; there's no deadline forcing these)*
- Dormant `scenarios_dense_mpls_full.csv` regen
- MN four-pool Carbon upgrade (depends on NatCap data sharing)
- AlphaEarth integration (depends on Google AI proposal status)
- "Use as starting point" raster→parameters translation
- Per-block-group nature_balance_avg implementation (different math than per-pixel mean; needs careful design)
- Mortality-risk model (NatCap's headline equity metric; prototype gap)
- UA scenarios in Explorer mode (current sliders are FF-focused; UA = different conversion type)

---

## 8. Honest assessments

**What's validated rigorously (measured):**
- UCM, UNA, UMH, SA Carbon, and the UFR runoff-retention index all at MAE ≈ 0 vs canonical InVEST (UMH after Brief B kernel fix; SA Carbon four-pool per-pixel, r 1.0 vs InVEST 3.19.0 — Relay 69; UFR retention index `1 − Q/P`, r 1.0 vs UFRM 3.19.0 — Relay 71). This is per-pixel parity on the prototype's own grid — the real validated core. (For UFR this is the per-pixel retention index; the lumped Flood Index / Runoff Volume stay aligned-method.)

**What's methodology-aligned but NOT a measured match:**
- The **lumped** flood readings — Flood Index (`100 − mean_CN`) and Runoff Volume (lumped mean-CN) — are canonical SCS-CN but are scalar proxies, not per-pixel UFRM outputs, so they carry no per-pixel parity claim. *(The per-pixel UFR **runoff-retention index** moved up to measured — validated vs UFRM 3.19.0 at MAE ≈ 0 / r 1.0, Relay 71. SA Carbon likewise moved up in Relay 69; what remains unreproduced for carbon is NatCap's published citywide absolute, below.)*

**What is NOT established (2026-05-29 investigation, under a no-parameter-fitting guardrail):**
- **NatCap's published citywide absolutes are not reproducible from what's on disk.**
  - *Temperature.* No SA UCM `args` (T_ref / uhi_max) ship anywhere in the drive pull. The prototype's heat-wave args (T_ref = 35 °C) give a citywide mean ≈ 107 °F against NatCap's published 90.08 °F, which is evidently an average-day figure produced with parameters we don't have. `T_air_nomix.tif` exists, but back-solving T_ref from it is parameter-fitting and was declined.
  - *Carbon.* NatCap's published 107.32M t CO2e does not reconcile with their own `tot_c_cur.tif` — which their `report.html` documents as **76.27M Mg C** on the compound baseline (`lulc_overlay_3857.tif` + a four-pool compound pool table) — by any standard interpretation (per-ha → 25–33M depending on the area convention; per-pixel-total → 280M). The published number is a separate aggregation script that wasn't shipped.
- **No prototype-vs-NatCap comparison has run end-to-end** (`natcap_published` = comparison-ready only; see §4).

**What's known to diverge:**
- Flood retention index uses `100 - mean_CN` rather than canonical retention output; monotone with CN-based runoff but on a different scale. (NatCap's documented SA finding: flood is ~scenario-invariant under the design storm.)
- UMH uses synthetic per-NLCD NDVI proxy, not satellite-derived
- MN Carbon uses per-cover annual-rate proxy, not four-pool (no MN parameter table available)
- SA UNA citywide aggregation uses Bexar bbox extent (1.4% population overlap with NatCap's block-group framing)
- Cooling energy savings computed over typed-OSM building pixels (~29% coverage), not all buildings

**What hasn't been built yet:**
- NDR
- Mortality-risk model
- AlphaEarth NDVI integration
- Round-trip InVEST results re-import

**What I don't know:**
- The CRS/resolution-mismatch question (NatCap's 10m EPSG:3857 vs the prototype's 30m EPSG:5070) is moot for the NatCap comparison for now — that comparison was never run, since the inputs aren't available, so resampling noise in it is untested. It does *not* bear on the engine-vs-canonical-InVEST parity, which is measured per-pixel on the prototype's own grid.

---

## 9. Working principles (these are firm, in ../../CLAUDE.md too)

- **Align with NatCap canonical, per city.** Different cities can have different per-city parameters; what matters is matching the published methodology for each.
- **Investigate before refactoring.** Phase 0 with a hard stop-and-report gate catches wrong premises. Multiple briefs this session validated this pattern.
- **Stop-and-report sentinels in briefs.** When validation harness output deviates from expectation, pause for review before continuing.
- **Interface changes require auditing ALL consumers.** Renaming a metric isn't a label change; it cascades through display, dense CSV, baselines, surrogate training.
- **Single commit per concern.** Brief 4 template. Bundle the change and its announcement together.
- **Bump SCENARIO_SCHEMA_VERSION on math/output changes.** Schema bump cascades through baselines + dense CSVs + downstream tools.
- **Don't mutate shared environments.** App .venv and anaconda base are off-limits for new validation work; use isolated conda envs (e.g., `natcap_umh_validation`).
- **PROJ/rasterio env workaround**: `PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data`
- **Empirical proof beats reasoning.** When a metric switch or methodology change is proposed, measure it on real data before any code change. The nature_balance_avg brief was killed by Phase 0 empirical measurement (487 vs NatCap's 107) — reasoning had said it would work.
- **Per-pixel validation ≠ aggregation validation.** UNA at MAE≈0 per-pixel doesn't mean the citywide reductions are comparable. Different statistics on the same raster can produce wildly different answers.
- **Reproducing canonical InVEST ≠ reproducing NatCap's published numbers.** (2026-05-29.) The engine can match canonical InVEST per-pixel and still not reproduce NatCap's published citywide figures — different parameters (heat-wave vs average-day T_ref), extents, and aggregation scripts sit between the two. Keep the two claims separate everywhere: badges, tooltips, docs, commit messages.
- **A validation status must reflect a comparison that ran, not one that's merely wired.** (2026-05-29.) `natcap_published` sat as "comparison-ready" for the whole project because the inputs to run it never arrived, and it was easy to mistake the set-up for a result. Don't let a plumbed comparison read as a verified one.
- **Don't fit parameters to hit a published target.** (2026-05-29.) Recovering a parameter from a result raster so the numbers match is fitting, not reproduction. It was explicitly declined for temperature this session — the honest output was "not reproducible from what we have."

---

## 10. Conversational principles (working with Claude on this project)

- **"Do it right" preference**: When offered a choose-between-rigor-and-shortcut, Deborah consistently prefers rigor. Brief 4 sign refactor, Option A on UMH validation, the Brief B kernel fix — all examples.
- **Validate by measuring, not by arguing.** When in doubt, run the diagnostic.
- **Calibrated claims, neither direction.** Not overstating (no "we reproduce NatCap" without evidence) and not understating (per-pixel InVEST parity is a real, measured result — state it plainly, not as a consolation). Realistic and honest is the target.
- **WHATS_NEW is canonical.** No standalone status emails to Yingjie or others. If it's important enough to share, it should be in WHATS_NEW.
- **Per-city parameters, not universal defaults.** SA and MN may legitimately have different demand standards, search radii, etc. Brief 22 established the principle.

---

## 11. Positioning (the symposium is a venue, not a deadline)

The NatCap Symposium (June 29 – July 1, 2026) is a place to meet people and talk, **not a delivery gate** — Deborah isn't presenting. There's no MVP-by-a-date; the work is "finish when it's right." What matters is that whatever gets shown or discussed carries honest claims. Next-work prioritization lives in §7, not here.

**Honest claims the prototype can make:**
- The engine reproduces canonical InVEST per-pixel (UCM/UNA/UMH at MAE≈0). *The validated core.*
- It adopts NatCap's four-pool carbon framework (a methodology choice, not a measured match).
- It surfaces NatCap's own published scenario outcomes as labeled reference values.
- It explores a scenario design space and exports runnable canonical-InVEST bundles for full-resolution validation.

**What is explicitly NOT the claim:**
- "Validated against NatCap" without qualification — too strong; the engine is validated against *canonical InVEST*, and NatCap's published *citywide* figures are not independently reproduced.
- "Reproduces NatCap's results" — not established; the inputs to do so (compound per-scenario LULCs, UCM args, the carbon aggregation script) aren't available.
- "Replaces InVEST" — no; it's a discovery layer.
- "Decision engine" — no; it surfaces options for stakeholder conversation.

---

## 12. Related documents

This document is the internal source of truth for strategic framing: what the app is, what it is not, and how validation, exploration, optimization, and InVEST export fit together.

For metric definitions, see `../../REFERENCE.md`. For system architecture, see `ARCHITECTURE.md`. For current NatCap/InVEST alignment, see `NATCAP_ALIGNMENT.md`. For demo talking points, see `DEMO_AND_COLLABORATION.md`.

---

*This document supersedes nothing — operational truth lives in the code and other docs — but it captures the conceptual scaffolding decisions made through May 29, 2026 that aren't otherwise written down. The 2026-05-29 revision folds in the B2 validation investigation: the engine-vs-InVEST / NatCap-citywide-reproduction distinction (§4, §8), the comparison-ready-never-run status, and the removal of the symposium-as-deadline framing (§11).*

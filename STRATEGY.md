# STRATEGY.md — Ecosystem Explorer

*Strategic framing, prioritization, and working principles. Captured 2026-05-29.*

This document captures the conceptual scaffolding that the operational docs (CLAUDE.md, DESIGN_NOTES.md, NATCAP_ALIGNMENT.md, HISTORY.md) reference but don't themselves articulate. If you're a future Claude or a future Deborah picking this up cold, read this first.

---

## 1. Center of gravity

**Ecosystem Explorer is a validated scenario exploration and discovery layer for urban InVEST analyses.**

It can:
1. Reproduce and visualize NatCap project scenarios (validated anchors)
2. Compare them in a common dashboard alongside user-generated scenarios
3. Use surrogate optimization to identify additional scenarios worth testing in future stakeholder conversations
4. Export selected scenarios as runnable canonical InVEST input bundles for full-resolution validation

**Tagline:** *Validated scenario exploration for urban ecosystem tradeoffs*

**Four-line narrative:**
> NatCap scenarios establish trust.
> Explorer scenarios expand the design space.
> The optimizer discovers promising alternatives.
> Full InVEST runs validate the candidates.

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

The prototype is currently between Levels 3 and 4. Brief D1 (Export for InVEST) makes Level 5 partly real rather than aspirational.

---

## 4. Validation taxonomy

Three validation states. Each dashboard metric × scenario has one, surfaced via badges and recorded in `data/<city>/natcap_reference_outputs.csv`:

**✓ NatCap match** (`natcap_published`)
The prototype's value is comparable to a published NatCap value within tolerance. Strongest claim. Currently: SA temperature (city-wide °F), SA carbon (4-pool stock × 44/12).

**≈ Aligned method** (`aligned_method`)
The prototype uses canonical InVEST methodology, but no directly-comparable NatCap citywide reference exists, OR the framing differs (different summary statistic, scope, aggregation level). Currently: SA UNA (per-block-group aggregation needed for citywide comparison), SA cooling energy (scope difference), SA flood (canonical UFR method, no NatCap published value), UMH (canonical kernel parity, MAE≈0, but synthetic NDVI proxy).

**Prototype** (`prototype`)
Exploratory or proxy methodology with no canonical InVEST analog. Currently: food production (food-forest yield × area benchmark).

**Why the three states matter:** they make the validation claim honest. "Validated where possible, exploratory where valuable" depends on per-metric honesty, not blanket claims. The taxonomy makes the dashboard self-describing — a returning user can see which numbers are anchored to NatCap and which are prototype framing.

---

## 5. Two-bars principle

Two changelog audiences, two bars:

**WHATS_NEW_ENTRIES** (in-app, strict bar) — Returning dashboard user would notice it; one line; no internal vocabulary or parameter values. Reserved for user-visible feature changes, confidence-level changes, architectural shifts users see.

**Collaborator-facing docs** (NATCAP_ALIGNMENT.md, DESIGN_NOTES.md, HISTORY.md) — Looser bar. Methodology milestones, validation results, schema bumps, internal taxonomy. The institutional record.

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
- A2: SA UNA AOI investigation — landed as doc-only (736756d). The "AOI mismatch" turned out to be a per-block-group aggregation question, not a config swap. See `NATCAP_ALIGNMENT.md` "SA UNA / biophysical extent" and `DESIGN_NOTES.md` "Brief A2".
- A3: `natcap_reference_outputs.csv` schema and population — **landed**. temp + carbon `natcap_published` (compared as scenario−baseline deltas; tol 5%/0.1°F and 1%); nature_access + cooling_energy `aligned_method` (nature → per-block-group aggregation in Track C; cooling → typed-OSM scope caveat); flood + UMH `aligned_method` placeholders; food `prototype`. Built by `extract_natcap_reference_outputs.py`, read via `natcap_validation.py` (not yet wired into the dashboard — Brief B2).

**Track B — Keystone**
- B1: NatCap fixed scenarios as first-class inputs (7 scenarios: baseline, FF_20ac, FF_40ac, FF_MAX, UA_20ac, UA_40ac, UA_MAX)
- B2: Per-metric validation markers in dashboard (wires A3 helpers into card display)
- B3: Canonical flood-volume output for SA alongside index (needs investigation pass first)

**Track C — Payoff**
- C1: Parity validation across UCM/UNA/Carbon for fixed scenarios. UNA validation requires per-block-group aggregation (per A2). Flood has no NatCap published number so it's canonical-method-only.

**Track D — Strategic addition**
- D1: Export for InVEST workflow. Brief drafted (384 lines). Includes both source rasters for NatCap scenarios + prototype rasters; per-model args.json for all 5 InVEST urban models; polymorphic metadata block; Phase 3 verification required.

**Track E — Optional**
- E1: NDR for fixed scenarios. Inputs concrete (ndr_biophysical_parameters_vNLCDTree_SA.csv, etc.). Only run for baseline + 20ac + 40ac, not arbitrary slider scenarios.
- E2: Status update to Yingjie — DROPPED. WHATS_NEW is the canonical record; if it's important enough to email, it should be in WHATS_NEW first.

**Track F — Defer post-symposium**
- Dormant `scenarios_dense_mpls_full.csv` regen
- MN four-pool Carbon upgrade (depends on NatCap data sharing)
- AlphaEarth integration (depends on Google AI proposal status)
- "Use as starting point" raster→parameters translation
- Per-block-group nature_balance_avg implementation (different math than per-pixel mean; needs careful design)
- Mortality-risk model (NatCap's headline equity metric; prototype gap)
- UA scenarios in Explorer mode (current sliders are FF-focused; UA = different conversion type)

---

## 8. Honest assessments

**What's validated rigorously:**
- UCM, UNA, UMH all at MAE ≈ 0 vs canonical InVEST (UMH after Brief B kernel fix)
- SA Carbon: NatCap four-pool framework (Brief 30)

**What's known to diverge:**
- Flood retention index uses `100 - mean_CN` rather than canonical retention output; the index is monotone with CN-based runoff but on a different scale
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
- Whether your CRS/resolution mismatch between NatCap's 10m EPSG:3857 outputs and the prototype's 30m EPSG:5070 computations introduces meaningful noise in validation comparisons. The two clean `natcap_published` metrics (temp, carbon) are aggregates where resampling effects largely cancel, so this risk is bounded.

---

## 9. Working principles (these are firm, in CLAUDE.md too)

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

---

## 10. Conversational principles (working with Claude on this project)

- **"Do it right" preference**: When offered a choose-between-rigor-and-shortcut, Deborah consistently prefers rigor. Brief 4 sign refactor, Option A on UMH validation, the Brief B kernel fix — all examples.
- **Validate by measuring, not by arguing.** When in doubt, run the diagnostic.
- **WHATS_NEW is canonical.** No standalone status emails to Yingjie or others. If it's important enough to share, it should be in WHATS_NEW.
- **Per-city parameters, not universal defaults.** SA and MN may legitimately have different demand standards, search radii, etc. Brief 22 established the principle.

---

## 11. Symposium target

NatCap Symposium, June 29 – July 1, 2026.

Minimum-viable symposium-ready prototype:
- All three validation states surfaced in the dashboard via per-metric badges
- NatCap fixed scenarios loadable as first-class scenarios
- Side-by-side comparison of NatCap scenarios with Explorer scenarios
- At least one cleanly-validated metric per scenario (temp + carbon for SA)
- Export for InVEST capability demonstrable (zip download produces a runnable bundle)

Nice-to-haves if time:
- NDR for fixed scenarios (Track E1)
- Per-block-group UNA aggregation for direct NatCap comparability (Track C1)
- "Use as starting point" workflow (Track F)

What's *not* the symposium claim:
- "Validated against NatCap" without qualification (too strong; some metrics aren't directly comparable citywide)
- "Replaces InVEST" (no — we're a discovery layer)
- "Decision engine" (no — we surface options for stakeholder conversation)

---

## 12. Where to look next

If picking this up cold:

1. Read this file (you're here)
2. Read `CLAUDE.md` for operational conventions and pending-work pointers
3. Check `git log --oneline -20` for recent commit landings
4. Read `DESIGN_NOTES.md` "Brief A2" and "Brief B" for the two most-recent significant findings
5. Look at `data/sa/natcap_reference_outputs.csv` if it exists; if not, A3-impl is still pending
6. Look at `/mnt/user-data/outputs/` for any drafted-but-not-sent briefs

The drafted briefs as of this writing (May 29, 2026):
- `BRIEF_A3_IMPL.md` — populate natcap_reference_outputs.csv for SA (paused at Phase 0; resume instructions in conversation)
- `BRIEF_D1_EXPORT_INVEST.md` — Export for InVEST workflow (drafted, not sent)
- `BRIEF_NATURE_ACCESS_NTR_BAL.md` — KILLED by empirical measurement; do not send

---

*This document supersedes nothing — operational truth lives in the code and other docs — but it captures the conceptual scaffolding decisions made through May 29, 2026 that aren't otherwise written down.*

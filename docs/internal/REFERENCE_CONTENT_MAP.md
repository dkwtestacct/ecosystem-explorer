# REFERENCE.md restructure — content map (LOCKED)

**Audience:** Internal
**Status:** Locked — approve-before-rewrite gate passed; ready to drive Step 3 (the rewrite)
**Use this for:** Driving the REFERENCE.md §1–10 rewrite and the link-fix pass; the canonical source for which current content lands where
**Do not use this for:** Current REFERENCE methodology — this doc is scaffolding for the rewrite, not the dashboard guide itself
**Source of truth for:** The current → target mapping and locked resolutions for the REFERENCE rewrite

---

**Transient scaffolding.** Delete after the rewrite + link-fix passes have landed and the inbound-reference inventory is exhausted. Restructure **and** the stale-language fixes happen as **one mapped pass** for REFERENCE; the app.py expander rewrite and the ARCHITECTURE move are tracked here as separate commits.

---

## Target structure

1. What this tool is
2. How to read the dashboard
3. Scenario sources and provenance
4. Validation badges
5. Land-use scenarios
6. Metrics — **Ecological** (Flood Retention / Runoff Volume, Temperature Change, Carbon, NDVI) · **Human & Social** (Nature Access, Preventable MH Cases, Avoided / Added MH Costs) · **Economic** (Food Production, Implementation Cost, Energy Savings, Carbon Value, Cost-effectiveness ratios)
7. City-specific notes (Minneapolis / San Antonio)
8. What is InVEST-aligned vs prototype-specific
9. Known limitations
10. What to validate before decision use

Header block keeps the 5-field status header, with: **Use this for:** "Understanding dashboard metrics, data sources, validation badges, and limitations" / **Do not use this for:** "Internal task tracking, collaboration history, or implementation decisions."

---

## Global editorial rules (apply throughout)

- **Guide, not implementation log.** Strip `Brief NN` references, `recomputed YYYY-MM-DD` callouts, and "previously X, now Y" archaeology. Describe the *current* state. ("San Antonio uses NatCap's compound land-cover framework…", not "Brief 28b changed…")
- **History isn't deleted, it's already elsewhere.** Most of the stripped content (schema-version log, per-brief notes, retired-metric rationale) is already captured in HISTORY.md / DESIGN_NOTES.md / NATCAP_ALIGNMENT.md. So stripping = *deleting the duplicate* — but confirm each note is genuinely captured elsewhere before removing REFERENCE's copy, so we never drop the only record.
- **Metric mini-template** (every metric in §6): **What it shows** (plain English) · **How it is computed** (short method — fold the data source in here) · **Units** · **Validation status** (one of the §4 badges, with per-metric evidence) · **Main caveat** (one honest line).
- **Honest language, enforced:**
  - "validated against canonical InVEST where comparable inputs exist" — never "validated against NatCap."
  - NatCap published SA temperature/carbon → "displayed as NatCap reference values" — never "reproduced," unless the exact scenario inputs + aggregation path are available (they are not).
  - One-paragraph framing near the top of §1 (replaces any "simple tradeoff explorer" language): *"Ecosystem Explorer validates its modeling engine against canonical InVEST, displays NatCap project reference values where available, and lets users explore additional scenarios beyond the fixed project set — then export promising ones back to canonical InVEST for full validation."*

---

## Resolved findings (locked — replaces the prior `[VERIFY]` section)

### §4 — Badge vocabulary (LOCKED)

**§4 documents BOTH surfaces using the exact rendered strings.** No coined badge names. No "✓ InVEST validated" — that wording does not appear in the code.

**A — Per-card validation badge (per-metric × per-scenario context).** Renders inline under each metric card. Code: `_render_validation_caption` (`app.py:3297`); `render_validation_badge` (`natcap_validation.py:194`); colors `_VALIDATION_BADGE_COLOR_HEX` (`app.py:3290`). Comment at `app.py:3278` documents this as the replacement for the previous 3-tier system.

| Rendered text (verbatim) | Color | Fires when |
|---|---|---|
| `NatCap published value` | green | a `natcap_published`-class metric × the fixed-scenario reference view, where the card displays NatCap's own published number directly from `natcap_reference_outputs.csv`. The card surfaces NatCap's figure; the prototype does not claim independent reproduction. |
| `≈ NatCap method` | blue | a `natcap_published`-class metric × any other scenario context (Baseline / Explorer / Optimizer). The displayed value is the prototype's own computation; the methodology is aligned with NatCap's. Tooltip is metric-aware (temperature and carbon both cite measured per-pixel parity vs canonical InVEST 3.19.0). |
| `≈ Aligned method` | blue | an `aligned_method` metric (canonical InVEST methodology with no directly-comparable NatCap citywide reference) in any context. |
| `Prototype` | gray | a `prototype` metric (no canonical InVEST analog) in any context. |

**B — Scenario provenance header (per-scenario).** Renders as a prominent header above the metric cards. Code: `_render_scenario_provenance_header` (`app.py:3356`); table `_PROVENANCE_HEADER_INFO` (`app.py:3324`).

| Source (verbatim) | Validation line (verbatim) | Color |
|---|---|---|
| `Baseline` | engine verified vs canonical InVEST; absolute NatCap citywide figures not reproduced | blue |
| `NatCap published reference` | displayed from NatCap output; exact scenario raster / aggregation not available | green |
| `Explorer-generated` | canonical engine verified; scenario not NatCap-published | blue |
| `Surrogate-suggested` | engine-validated; full-raster evaluated — exploratory candidate for further validation | blue |

**Division of labor (§4 must explain this).**
- The **provenance header** answers: *whose scenario is this?* (per-scenario, four sources).
- The **per-card badge** answers: *how trustworthy is this specific number on this card right now?* (per-metric × per-context).
- The **context-switch:** `NatCap published value` fires only in the fixed-scenario reference view (when the dashboard surfaces NatCap's published number directly); in every other scenario context (Baseline / Explorer / Optimizer), a `natcap_published`-class metric is shown as `≈ NatCap method` because the displayed value is the prototype's own computation.
- **Per-metric validation evidence does NOT live in §4.** §4 documents only what the badges *mean* in the abstract. The per-metric "where the evidence comes from" detail (measured MAE numbers for UCM/UNA/UMH/SA-carbon; canonical methodology with no comparable NatCap reference for the aligned-method metrics) belongs in **§6 — Metrics**, in each metric's mini-template under **Validation status**.
- **§6 must explicitly distinguish:**
  - **Measured per-pixel parity** (temperature, nature access, MH cases) — concrete MAE numbers against `natcap.invest.*.execute()`.
  - **Method-adoption-without-parity** (SA carbon four-pool framework — methodology choice, no measured per-pixel comparison).
  - **Canonical methodology, no NatCap comparable** (the `≈ Aligned method` cards).

### §7 — UNA parameters (LOCKED)

UNA 2SFCA parameters are **per-city** in `config.py`'s `CITIES` dict. Both city configurations are NatCap-project-canonical adoptions.

The §7 UNA row reads (substantively, not as verbatim copy):

> **Per-city 2SFCA parameters + per-city biophysical table.** Minneapolis (both extents): 250 m²/capita demand, 1000 m search radius, exponential decay — NatCap MN-project canonical. San Antonio: 16.7 m²/capita, 800 m, dichotomy decay — NatCap SA-project canonical. Per-city biophysical table sources the `urban_nature` proportion for each LULC class. (Defined: `config.py:55–57` and `:274–276`; read in `app.py:1153–1155`.)

### `app.py:506–525` expander rewrite (LOCKED — tracked as separate commit)

The "How this prototype works" expander still describes the old **3-tier confidence system** (High / Medium / Prototype) even though the per-card render has switched to the §4-A four-state vocabulary. This is a real stale-language defect — it MUST be fixed for the rendered UI to be internally consistent.

**Scope:** tracked here, applied as **its own commit**, derived from the settled §4 text after the REFERENCE rewrite lands. Out of scope for the REFERENCE rewrite commit itself (touches app.py, not REFERENCE.md), but in scope for the same broader workstream so the dashboard's in-app docs match REFERENCE §4.

Commit body: *"Rewrites the in-app 'Confidence tiers' expander to match the four-state validation-badge vocabulary (NatCap published value / ≈ NatCap method / ≈ Aligned method / Prototype) and the four-source scenario provenance header. The previous 3-tier text predated the Brief B2-revised badge work; the per-card render has been on the new vocabulary since `5295a4d`-ish."*

### Inbound-reference inventory (LOCKED — drives the link-fix pass)

The rewrite must retarget every ref below to the new §-based anchors. **Sub-anchors must be stable** for the per-model alignment sections (UCM / UNA / UMH / Carbon) and for "Placement strategies" — these have the most inbound dependencies. Anchor name choices are the rewrite's call; the table below names what's currently cited.

#### External (`.md` + `.py` + `data/` — total: ~19 refs)

| File:line | Cited anchor | Action |
|---|---|---|
| `CLAUDE.md:194` | `"Cross-city Cooling Capacity comparison"` | **Name-drift fix:** retarget to the equivalent (the §7 SA notes pointer for the actual section "Cross-city Heat Mitigation Index comparison"). |
| `CLAUDE.md:220` | `"Official InVEST alignment"` (umbrella) | retarget to new §8 anchor |
| `CLAUDE.md:249` | `"Placement strategies"` | retarget to new §5 / §8 anchor |
| `CLAUDE.md:275` | `"Official InVEST alignment — UCM"` | retarget to per-model sub-anchor |
| `CLAUDE.md:556` | `"Official InVEST alignment — UMH"` | retarget to per-model sub-anchor |
| `CLAUDE.md:593` | `"Official InVEST alignment — UNA"` | retarget to per-model sub-anchor |
| `docs/internal/NATCAP_ALIGNMENT.md:223` | `"Official InVEST alignment"` (umbrella) | retarget |
| `docs/internal/DATA_INVENTORY.md:336` | `"Cross-city Heat Mitigation Index comparison"` | retarget (correct name already) |
| `docs/research/INVEST_PLACEMENT.md:333` | `"Placement strategies"` | retarget |
| `docs/research/una/UNA_DIVERGENCE_CASE_STUDIES.md:68` | `"Official InVEST alignment — UNA"` | retarget |
| `docs/archive/HISTORY.md:81` | `"Official InVEST alignment — UMH"` | retarget |
| `app.py:4553` | generic `"see REFERENCE.md"` | leave bare or anchor to §6 caveats |
| `app.py:5040` | `"Land-use alignment"` | retarget to new §5 anchor |
| `config.py:69` | generic `"see REFERENCE.md"` | leave bare |
| `config.py:100` | `"Option A buildings semantics"` | retarget to new §7 anchor |
| `export_invest_bundle.py:54` | `"Official InVEST alignment"` (umbrella) | retarget |
| `validation/compare_una_invest.py:10` | `"Official InVEST alignment"` (umbrella) | retarget |
| `validation/compare_ucm_invest.py:10` | `"Official InVEST alignment — UCM"` | retarget |
| `data/invest/cooling/UCM_AUDIT.md:47` | `"Official InVEST alignment"` (umbrella) | retarget |

#### Internal-self-refs (REFERENCE.md cites its own sections — total: ~10 hits, will be rebuilt by the rewrite)

| Cited anchor | At REFERENCE.md line(s) | New target |
|---|---|---|
| `"Official InVEST alignment"` (umbrella) | 32 | §8 anchor |
| `"Official InVEST alignment — UCM"` | 200, 408, 411, 777 | per-model sub-anchor |
| `"Official InVEST alignment — UNA"` | 489, 495, 923 | per-model sub-anchor |
| `"Official InVEST alignment — UMH"` | 516 | per-model sub-anchor |
| `"Cross-city Heat Mitigation Index comparison"` | 317 | §7 SA notes |
| `"Placement strategies"` | 365, 614, 636, 903 | §5 / §8 anchor |

**Anchor-stability requirement.** The five high-traffic anchors below MUST have stable, predictable sub-anchor names in the new structure (under §4 / §5 / §6 / §7 / §8 as appropriate):
- "Official InVEST alignment — UCM" (5 inbound)
- "Official InVEST alignment — UNA" (5 inbound)
- "Official InVEST alignment — UMH" (4 inbound)
- "Official InVEST alignment" umbrella (5 inbound)
- "Placement strategies" (5 inbound)

---

## Current → target mapping

| Current section (approx. lines) | → Target | Transformation / notes |
|---|---|---|
| Header status block (3–7) | Header | Update Use/Do-not-use wording as above. |
| Conceptual Overview (15–21) | §1 + §7 | New one-paragraph framing replaces the "comparative exploration / not precise prediction" line. Multi-city paragraph → §7 intro. |
| Division of labor (25–39) | §8 + §7 | Component-source table → §8. Per-city LULC-fidelity implications → §7. |
| Land-use alignment: MN / SA / fallback / closing (43–117) | §5 + §7 | Scenario-conversion mechanics (proxy mapping, fallback logic) → §5. Per-city baseline-LULC/code-system detail → §7. |
| Official InVEST alignment: UFR/UCM/UNA/UMH/Carbon/Crop (121–175) | §4 + §6 + §8 | Per-model **validation status + MAE evidence → §6 (per metric)** (keep the actual numbers — MAE=0 UCM, ≈0 UMH, UNA supply r=1.0 / ~5.5e-7 rel MAE). §4 documents only what the badges *mean*. "Placement-agnostic / prototype heuristic" framing → §8. |
| Intended Use (179–190) | §1 + §10 | "Designed for" → §1. "Not intended for" → §10. |
| Key Terms (194–202) | §6 / appendix | Fold into first-use in §6, or a short glossary appendix. Don't lose the definitions. |
| Methodology Notes (206–242) | §6 + §5 | CN formulas → §6 Flood; temp calibration → §6 Temperature; carbon rates → §6 Carbon; food yield → §6 Food. Placement strategies → §5 (and the "no InVEST parity" point → §8). |
| Computation Architecture (325–372) | §2 (gist) | Deep mechanics are absorbed by the **separate ARCHITECTURE refresh workstream**, which pulls this line range from git into ARCHITECTURE's new §5. **The REFERENCE rewrite does not edit ARCHITECTURE.** REFERENCE keeps only the reader-relevant gist in §2: "metrics run live per slider; some modes precompute supporting raster aggregates." Cross-ref ARCHITECTURE's existing three-layer section for now; the cross-ref retargets to the new §5 in the ARCHITECTURE refresh's link-fix sweep. |
| Metric Cards: Ecological/Human-Social/Economic/Cost-Eff (376–603) | §6 | The heart of §6. Reformat each to the mini-template. |
| Cross-city HMI comparison (415–429) | §7 (SA notes) | With a pointer from §6 Temperature. |
| Scenario Summary Text (605–614) | §2 | UI element. |
| Sidebar Controls + Example Scenario Buttons (618–679) | §2 | UI walkthrough. |
| Smart Scenario Search / Surrogate (683–749) | §8 (interpretive) + §2 (controls) | Deep surrogate mechanics are absorbed by the **separate ARCHITECTURE refresh workstream**, which pulls this line range from git into ARCHITECTURE's new §5. **The REFERENCE rewrite does not edit ARCHITECTURE.** §8 keeps the optimizer-as-discovery framing ("scenario search, not Pareto optimization in the ROOT sense"). Sidebar control walkthrough (just the controls' meaning) → §2. Cross-ref ARCHITECTURE's existing three-layer section for now; retarget in the ARCHITECTURE refresh's link-fix sweep. |
| Tradeoff Chart / Convex Hull / Saved / Pareto / Optimized / Input Influence / Best-by-Goal (753–858) | §2 | Dashboard-element references. Best-by-Goal + optimizer tie-ins cross-ref §8. |
| Baseline-vs-Scenario Comparison / Bar Charts / Map View (862–903) | §2 | UI elements. |
| Known Limitations (907–923) | §9 + §7 | General limits → §9. City-specific (Minneapolis Full hidden, SA = Bexar bbox, SA UNA characterization) → §7. |
| External Workflows / Export for InVEST (925–986) | §8 (own subsection) | Per the Export framing below. |

---

## Content moving OUT of REFERENCE

- **Computation-architecture internals + deep surrogate mechanics → separate ARCHITECTURE refresh workstream** (not this rewrite). The ARCHITECTURE refresh pulls REFERENCE.md:335–371 + 683–749 from git into ARCHITECTURE's new §5. **The REFERENCE rewrite does not edit ARCHITECTURE.** REFERENCE keeps gist + cross-ref.
- **Duplicated history** (schema-version log, brief-by-brief notes, retired-metric rationale) → already in HISTORY.md / DESIGN_NOTES.md → delete from REFERENCE after confirming capture.

---

## Stale-fixes folded into REFERENCE (from the audit)

- **UMH kernel** (current lines 497, 511 say `gaussian_filter`) → §6 MH "How it is computed" uses the canonical edge-corrected **buffer-mean / flat disk** (matches lines 161, 516). Keep the 300 m / 10 px radius — only the kernel *type* was wrong.
- **Nature Quality Score** (current 361 live-field list; 496 retired-row) → not in §6 at all (retired; history → HISTORY). Remove from the live-field description; drop the methodology-history row.
- **"Not yet shown / planned for Brief B2"** (current 303) → §4 states the badges + reference view are live.
- **NatCap-reproduction framing** (current 293–298) → §3/§4: displayed reference values; direct reproduction gated on unavailable per-scenario compound inputs.
- **Old Nature Access proxy / Wellbeing Score archaeology** → stripped (history lives elsewhere).
- **Three-tier confidence vocabulary in any REFERENCE residual** → replaced by the four-state §4 vocabulary throughout.

---

## Resolved decisions (bake in)

- **SA flood — move from headline to methodological note.** Per converted pixel, GI may be the most flood-effective land-cover change; but at the San Antonio / Bexar-County dashboard scale, the total flood metric is **nearly scenario-invariant** (developed land is a small share of the bbox), so a broad "GI is best for flood" can mislead users about the visible scenario outcome. Place as a precise note in §6 (Flood) and echo in §7 (SA). Replaces the general claim at current line 396.

---

## ARCHITECTURE refresh + REFERENCE rewrite — two commits

**No "Commit A" additive move.** `docs/internal/ARCHITECTURE.md` is being refreshed into a new structure by a separate workstream — pushing mechanical depth into the current ARCHITECTURE shape would just get reshuffled by that refresh.

### Owned by the separate ARCHITECTURE refresh workstream (NOT this rewrite)

The mechanical depth at the line ranges below is pulled from git into ARCHITECTURE's new §5 by the ARCHITECTURE refresh — recorded here so the source is unambiguous:

| REFERENCE source (in git, pre-rewrite) | Absorbed into |
|---|---|
| REFERENCE.md:335–352 (Model quality modes table + paragraph) | ARCHITECTURE refresh → new §5 |
| REFERENCE.md:354–371 (live-overwrite field list + `SCENARIO_SCHEMA_VERSION` mechanics) | ARCHITECTURE refresh → new §5 |
| REFERENCE.md:683–712 (Smart Scenario Search controls + Normal-slider paragraph + per-layer purpose table) | ARCHITECTURE refresh → new §5 |
| REFERENCE.md:716–749 ("Why use a Surrogate?", "How it thinks", optimizer mechanics) | ARCHITECTURE refresh → new §5 |
| REFERENCE.md:752 ("Advanced Settings" model-quality mode row) | ARCHITECTURE refresh → new §5 |

**The REFERENCE rewrite does not edit ARCHITECTURE.** After the rewrite, REFERENCE keeps only gist + cross-ref; the depth lives in git (and in the ARCHITECTURE refresh's eventual new §5).

### Commit A — REFERENCE rewrite (the §1–10 restructure + link-fix)

The deep mechanics get *stripped* from REFERENCE in this commit (replaced by gist + cross-ref) — not moved. The ARCHITECTURE refresh recovers them from git for its new §5.

- §2 two-sentence gist: "Metrics run live on every slider change. In *High resolution* mode some supporting raster aggregates are precomputed; the surrogate is only used when you explicitly run the optimizer." **Cross-ref ARCHITECTURE's existing three-layer section ("At a glance" + Layer 1/2/3) for now; the cross-ref retargets to the new §5 in the ARCHITECTURE refresh's link-fix sweep.**
- §6 metric mini-template's "How it is computed" — short method only.
- §8 InVEST-aligned vs prototype-specific — interpretive framing only; the optimizer-as-discovery reframe lives here. Cross-ref ARCHITECTURE for mechanics (same retarget note as above).
- Sidebar control walkthrough → §2 — controls' meaning + tooltip phrasing, not surrogate math.
- Link-fix pass: retargets all 19 external + 10 internal-self-refs per the Inbound-reference inventory above.

### Commit B (separate workstream) — app.py expander rewrite

`app.py:506–525` "Confidence tiers" expander → rewritten to match §4's four-state badge vocabulary + four-source provenance header. Derived from the settled §4 text after Commit A lands.

---

## Sequence after this map is approved

1. **Approved + map committed.** ← we are here.
2. **Commit A — REFERENCE §1–10 rewrite + link-fix pass.** Section-by-section drafts with stop-and-report before commit. Verifies every inbound ref (19 external + 10 internal-self-refs) resolves; every current-section row from the mapping table landed somewhere; nothing dropped. **Does not edit ARCHITECTURE.** Cross-refs to ARCHITECTURE point at the existing three-layer section; retargets to the new §5 happen in the ARCHITECTURE refresh's own link-fix sweep.
3. **Commit B — app.py expander rewrite.** Derived from §4's settled text; stop-and-report before commit.
4. **Delete this map file** after Commit B lands and the inbound-ref inventory is exhausted.

The separate **ARCHITECTURE refresh workstream** runs in parallel / later — it pulls REFERENCE.md:335–371 + 683–749 from git into the refreshed ARCHITECTURE's new §5. Not driven by this map.

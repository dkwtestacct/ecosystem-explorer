# DESIGN_NOTES.md refresh — content map

**Audience:** Internal
**Status:** In review — `[VERIFY]` 1–5 resolved; awaiting approval before rewrite
**Use this for:** Driving the DESIGN_NOTES.md §1–11 refresh, the per-decision currency sweep, the extract-not-delete discipline, and the HISTORY-capture check
**Do not use this for:** Current DESIGN_NOTES content — this doc is scaffolding for the refresh, not the decision log itself
**Source of truth for:** The current → target mapping and locked resolutions for the DESIGN_NOTES refresh

---

**Transient scaffolding.** Delete after the refresh + HISTORY-routing + cross-ref-fix passes land. Same discipline as `REFERENCE_CONTENT_MAP.md` / `ARCHITECTURE_CONTENT_MAP.md`.

**Purpose:** the current → target mapping for the DESIGN_NOTES refresh, turning a half-chronological append-only log into a durable technical decision journal without losing any decision. Sequence: CC finalizes the `[VERIFY]` items → we approve → CC rewrites §1–11 + routes the move-outs + verify, stop-and-report before commit.

**The governing risk here is different from REFERENCE/ARCHITECTURE.** Many durable decisions are embedded *inside* brief-numbered entries (the UMH buffer-mean kernel is in "Brief B"; the label-flip rule is in "Brief 1"; the validation-badge design is in "Brief B2-revised"). This refresh **extracts and preserves** those decisions (restructured to the template) and routes only the chronological narrative to HISTORY. Deleting a brief entry wholesale would lose its decision — so nothing is deleted until its durable content is confirmed captured in the new structure.

A second risk: the append-only log contains **superseded** decisions. The UNA section is the proven case — it records choosing 16.7 / 800 / dichotomy "for the Minneapolis prototype," but the live config is per-city (MN 250 / 1000 / exponential, SA 16.7 / 800 / dichotomy, per CC's `config.py` check). Every extracted decision must be confirmed against current code before it's enshrined as "current."

---

## Target structure

1. Documentation and naming conventions
2. City configuration and per-city parameters
3. Land-cover representation
4. Scenario generation and conversion logic
5. Placement strategy
6. Model evaluation design
7. Lookup table and surrogate optimizer
8. Validation and provenance design
9. Export for InVEST
10. UI communication decisions
11. Deferred alternatives

Header keeps the 5-field block, **rewritten away from "append-only / chronological"**: **Status:** "Current technical decision log" / **Use this for:** "Why a given implementation choice was made" / **Do not use this for:** "Metric definitions (→ REFERENCE.md), system structure (→ ARCHITECTURE.md), collaboration history, or brief chronology (→ HISTORY.md)" / **Source of truth for:** "Technical rationale and tradeoffs."

---

## Global editorial rules

- **Decision template for every entry:** Decision / Why / Alternatives considered / Consequences / Revisit if / Code touchpoints. Target ½–1 page each; the template enforces that.
- **Current-state only.** No brief numbers, no dates-as-headers, no chronology. "San Antonio uses NatCap's compound land-cover framework…", not "Brief 28b changed…".
- **DESIGN_NOTES = WHY.** Where a topic overlaps ARCHITECTURE (= HOW) — scenario sources, export, surrogate/optimizer, validation/provenance — keep the rationale here and the structure there, with cross-refs. The overlap-pairs are marked in the table; don't duplicate.
- **§8 vocabulary is shared** — the locked two-surface validation/provenance vocabulary, consistent with REFERENCE §4 and ARCHITECTURE §6.
- **Per-decision alternatives** live in the template's "Alternatives considered" field; §11 holds only cross-cutting deferred *approaches* not owned by one decision (PLUS/CLUE/LCM, ROOT).

---

## Current → target mapping

| Current section (line) | → Target | Transformation / extraction note |
|---|---|---|
| City-specific copy convention (13) | §1 | Keep — durable convention. Verify the `app.py ~3320–3333` reference hasn't drifted. |
| UNA parameters + 8 subsections (34) | §2 | **STALE — fix.** Collapse the 8 option-logs into one current per-city decision-note; the "16.7 for MN" reasoning → HISTORY. Use CC's confirmed per-city values. |
| Placement strategy: question / options / three-layer mask / suitability formulas / 4 strategies / decision principle (157) | §5 | Core durable decision, already well-shaped → template. Strip the "(2026-05-23 reformulation)" date. |
| — PLUS / CLUE / LCM deferred (224) | §11 | Deferred approach, compress. |
| — Wallpaper approach — "interpretation uncertain" (250) | OPEN_QUESTIONS / NATCAP_COLLABORATION | Unresolved ask, not a settled decision → active-ask doc (keep a one-line pointer in §5). |
| Land use / land cover sources: rasters / planned SA / NLCD-legacy (374) | §3 | Durable → template. File-path detail → cross-ref DATA_INVENTORY. Strip "(May 2026)". |
| UCM args alignment (432) | §6 | Durable model-eval decision. Strip date. |
| Per-city NatCap parameter framing (474) | §2 | **Anchor of §2** — explains "per-city because NatCap params are project-specific." Consolidate the UNA fix here. Strip briefs. |
| SA compound LULC integration (565) | §3 + §4 | CRS reproject + compound encoding → §3; conversion implications → §4. Strip brief. |
| SA UCM compound table (667) | §3 / §6 | Durable adoption → §3/§6. Note this is the **measured-parity** evidence for temperature (backs "≈ NatCap method" for temp). Strip brief. |
| SA UNA compound table (761) | §2 / §3 | Durable → §2/§3. Strip brief. |
| SA Carbon four-pool (845) | §6 | Durable carbon-framework decision. Note: **method adoption AND measured per-pixel parity** (validated vs InVEST 3.19.0, Relay 69). Strip brief. |
| SA AOI → ACS block-group (1003) | §2 | Durable AOI decision. Strip brief. |
| SA flood damage table — Path C + 4 paths + resolution (1070) | §6 (+ §10 xref) | Decision (embrace $0); the 4 paths = "Alternatives considered." "Explain on dashboard" → cross-ref §10. Strip brief. |
| Lookup-overlay safety contract (1234) | **§4 (owns)** + ARCHITECTURE §5 references it | Durable safety invariant — keep; future-you will forget it. **DESIGN_NOTES §4 owns the rationale + the 12 live-overwrite fields.** ARCHITECTURE §5 cross-refs §4 instead of restating the field list (avoids double-routing). |
| SA conversion-fallback instrumentation (1306) | §4 | Durable fallback-lucode logic. Strip brief. |
| NatCap ROOT as reference point (1386) | §11 (+ §7 xref) | Deferred approach → §11; cross-ref §7. Harmonize with ARCHITECTURE's why-not-ROOT (one pointer there, rationale here). |
| Signed metric cards — label-flip rule (Brief 1, 1444) | §10 + HISTORY | **Extract the label-flip rule** → §10 template; chronology → HISTORY. |
| Brief 2 — naming / labels (1479) | §1 / §10 + HISTORY | Extract durable naming rule; chronology → HISTORY. |
| Brief 4 — sign-convention refactor (1533) | §6 / §10 + HISTORY | Extract the `temp_change_f` sign convention; chronology → HISTORY. |
| Brief 5 — sidebar reorg + tooltips (1595) | §10 + HISTORY | Durable tooltip/layout decisions → §10; reorg chronology → HISTORY. |
| UMH validation vs InVEST 3.19.0 (1625) **+** Brief B — UMH kernel Gaussian → buffer-mean (1680) | **Consolidate into ONE §6 entry** + NATCAP_ALIGNMENT + HISTORY | These are two halves of the same decision (validation infrastructure + the canonical buffer-mean kernel that closed the gap). One §6 entry covers "UMH uses InVEST RR formula with canonical edge-corrected buffer-mean kernel (Gaussian was previously a divergence; corrected to match canonical) — see NATCAP_ALIGNMENT for MAE results." Validation-result narrative → NATCAP_ALIGNMENT. Chronology → HISTORY. Same decision REFERENCE §6 documents — keep consistent. |
| Brief A2 — SA UNA AOI investigation (1731) | docs/research/una/ | Investigation note → docs/research/una/; §2 keeps a one-line outcome pointer. |
| Brief B1 — NatCap fixed scenarios as inputs (1770) | §8 (+ ARCH §2 xref) | **WHY** fixed scenarios are first-class → §8; the HOW (source taxonomy) is ARCHITECTURE §2. Don't duplicate. Strip brief. |
| Brief D1 — Export workflow (1862) | §9 (+ ARCH §7 xref) | **WHY** export is designed this way → §9; the HOW (bundle structure) is ARCHITECTURE §7. Strip brief. |
| Brief B2-revised — validation badges + reference view (1939) | §8 | **Core badge-design rationale** → §8 (locked vocabulary). Chronology → HISTORY. Harmonize with REFERENCE §4 + ARCH §6. |
| Brief #3 — provenance header (2044) | §8 | Provenance-header design → §8. |
| Brief #4 — optimizer as discovery (2104) | §7 + §8 | Optimizer-as-discovery reframe → §7; the provenance bits → §8. |
| Brief #5 — cross-source comparison table (2176) | §8 + §10 | Split: Source/Validation columns + honest-display invariants → §8; presentation/placement → §10. |
| Topics not yet documented (2274) | keep (§11 tail or standalone) | Forward-pointer list — keep; trim any now-done. |

---

## Content moving OUT

- **Chronological / process narrative from every brief entry → HISTORY.md** — *only after* confirming HISTORY captures it (these entries may be the sole record).
- **UMH validation-result narrative → NATCAP_ALIGNMENT.**
- **Brief A2 UNA AOI investigation → docs/research/una/.**
- **Wallpaper "interpretation uncertain" → OPEN_QUESTIONS / NATCAP_COLLABORATION.**
- **Raster file-path detail → DATA_INVENTORY** (keep the decision in §3).

## Content coming IN

- **"Why numpy, not canonical natcap.invest" rationale → §6**, parked from the ARCHITECTURE refresh (latency, no `execute_from_arrays`, validation-not-replacement). This is its destination.

---

## Resolved findings (`[VERIFY]` 1–5)

### `[VERIFY] 1` — Per-decision currency sweep (partial — STALENESS CONFIRMED on UNA; sampled-only on others)

**Sampled-and-verified durable decisions (current — extract to template):**

| Section (line) | Decision | Code touchpoint | Status |
|---|---|---|---|
| City-specific copy convention (13) | `_PROVENANCE_HEADER_INFO` cross-ref → `app.py:3324` | confirmed at `app.py:3324` (per `[VERIFY] 1` of the ARCHITECTURE map) | **current** |
| Placement strategy (157) | Three-layer mask + 4 focused suitability formulas + Random + Balanced | helpers `_compute_suitability_weights`, `_select_pixels_for_conversion` in `app.py` | **current** |
| Land-use sources (374) | Per-city LULC framework (MN: NLCD; SA: NatCap compound) | matches REFERENCE §5 and `config.py` | **current** |
| UCM args alignment (432) | per-city UHI_MAX_C, dichotomy/exponential per city | matches `[VERIFY] 2` UNA findings: per-city | **current** |
| Per-city NatCap parameter framing (474) | "Per-city because NatCap params are project-specific" | `config.py:55–57` (MN), `:118–120` (MN Full), `:274–276` (SA) | **current — anchor of §2** |
| SA compound LULC (565) | EPSG:3857 → EPSG:5070 prep, NLCD×NLUD×tree compound | `data/sa/flood/land_use_compound_sa.tif` exists; loader reads compound | **current** |
| SA UCM compound (667) | adopted via NatCap's `ucm__nlcd_nlud_tree.csv` | `config.py:233` declares `ucm__nlcd_nlud_tree.csv` for SA | **current** |
| SA UNA compound (761) | adopted via NatCap's `una__nlcd_nlud_tree.csv` + per-city 800/16.7/dichotomy | `config.py:274–276` confirms per-city values | **current — fold UNA fix here** |
| SA Carbon four-pool (845) | NatCap compound `carbon__nlcd_nlud_tree.csv`; one-time stock | `app.py: _compute_carbon_four_pool` | **current — method adoption AND measured per-pixel parity (Relay 69)** |
| SA AOI block-group (1003) | ACS block-groups for SA per-tract aggregation | `config.py: tracts_file` = `acs_block_groups_3857.gpkg` for SA | **current** |
| SA flood damage Path C (1070) | embrace $0; label as "Flood Volume Reduction" | matches REFERENCE §6 Flood Damage Avoided card relabel | **current** |
| Lookup-overlay safety contract (1234) | live-overwrite invariant for ~12 fields after lookup hit | matches ARCHITECTURE map `[VERIFY] 4` live-overwrite list | **current — load-bearing** |
| SA conversion-fallback instrumentation (1306) | per-target fallback-pixel counts surfaced in dashboard | `app.py: ff_fellback_pixels` etc. | **current** |
| Brief 1 — label-flip rule (1444) | signed `_delta_pill` direction-of-good rule | `_delta_pill` at `app.py:4224` | **current** |
| Brief 4 — `temp_change_f` sign convention (1533) | positive=warmer, negative=cooler | `app.py:1492 _fmt_temp_change` + `app.py:1484 hm_to_temp_change_f` | **current — locked** |
| Brief B — UMH buffer-mean kernel (1680) | switched from Gaussian to canonical edge-corrected buffer-mean | matches REFERENCE §6 MH "How it is computed" | **current — locked** |
| Brief B2-revised — validation badge vocab (1939) | 4-state per-card + 4-source provenance header | locked in REFERENCE map § 4; ARCHITECTURE map § 6 | **current — locked** |
| Brief #3, #4, #5 — provenance / optimizer-as-discovery / cross-source table | all locked | covered in REFERENCE / ARCHITECTURE maps | **current** |

**STALE / SUPERSEDED (don't extract verbatim — restructure or move to HISTORY):**

| Section (line) | Why stale | Action |
|---|---|---|
| **UNA parameters (34)** — entire 8-subsection block | The decision-log frames itself as "for the Minneapolis prototype" choosing 16.7 / 800 / dichotomy. **Per current `config.py`: MN uses 250 / 1000 / exponential (Brief 22 NatCap MN-project canonical adoption); SA uses 16.7 / 800 / dichotomy (NatCap SA-project canonical).** The "16.7 for MN" rationale is fully superseded. | **Collapse** the 8 subsection log into one current per-city decision-note under §2 ("Per-city NatCap parameter framing — UNA"). Route the historical "considered options for the MN prototype" narrative → HISTORY. The UNA section becomes a clean per-city statement consistent with config.py and the `[VERIFY] 2` ARCH/REF agreement. |
| UMH validation result narrative (1625) | Validation results are NATCAP_ALIGNMENT's job, not DESIGN_NOTES's | Move the result narrative to NATCAP_ALIGNMENT; §6 keeps "validated — see NATCAP_ALIGNMENT." |

**Partial sweep — flagged for the rewrite step.** I sampled the durable decisions and confirmed the UNA staleness against `config.py`. **Sections 1444–2274 (the brief-numbered entries from Brief 1 onward)** mostly contain durable extractable decisions; for each, the rewrite must re-confirm against code at extraction time, not pattern-match. The map's "extract-not-delete + HISTORY-capture" discipline guards this.

### `[VERIFY] 2` — UNA per-city fix

Confirmed against `config.py:55–57` (MN), `:118–120` (MN Full), `:274–276` (SA):

| City | demand (m²/capita) | search_radius (m) | decay_function |
|---|---:|---:|---|
| Minneapolis (downtown) | **250** | **1000** | **exponential** |
| Minneapolis Full | 250 | 1000 | exponential |
| San Antonio | **16.7** | **800** | **dichotomy** |

Both MN and SA values are **NatCap-project canonical** (MN values from the InVEST UNA sample bundle / Brief 22 adoption; SA values from NatCap SA README). The §2 decision-note must state this per-city framework explicitly.

### `[VERIFY] 3` — HISTORY-capture check (partial — flagged)

`docs/archive/HISTORY.md` currently has section structures for: Schema version log (line 35), Retired infrastructure (line 64), Completed-workstream specifics (line 135), WHATS_NEW pruned (line 176). The brief-numbered narrative this refresh routes OUT of DESIGN_NOTES needs to land somewhere in HISTORY — either appended to existing sections or as new sub-sections.

**Risk:** HISTORY does NOT currently have per-brief narratives for Briefs 1, 2, 4, 5, B, A2, B1, D1, B2-revised, #3, #4, #5. If the rewrite extracts a durable decision to DESIGN_NOTES §6/§8/§10 but doesn't ALSO route the chronology to HISTORY, the brief narrative is dropped silently. **The HISTORY-capture step is a precondition for deleting the brief entry, not a bonus.**

**Recommendation for the rewrite:** for each brief entry, do two writes: (a) extract durable decision to DESIGN_NOTES §X (template-shaped); (b) append the chronological narrative to HISTORY (as a new sub-section under Completed-workstream specifics or as a per-brief stub). Only AFTER both writes confirm the content is captured, delete the DESIGN_NOTES brief entry. This is likely a paired commit (DESIGN_NOTES + HISTORY).

### `[VERIFY] 4` — §8 vocabulary alignment

The locked two-surface vocabulary (per-card badge: `NatCap published value` / `≈ NatCap method` / `≈ Aligned method` / `Prototype`; per-scenario provenance header: `Baseline` / `NatCap published reference` / `Explorer-generated` / `Surrogate-suggested`) is **locked** in both the REFERENCE map (§4 of the rewrite) and the ARCHITECTURE map (§6). DESIGN_NOTES §8 must use the same strings verbatim. The three docs do different jobs with the same words:

- DESIGN_NOTES §8 = **WHY this vocabulary** (the design rationale; the conservative-floor decision; the metric-aware tooltip rule)
- ARCHITECTURE §6 = **the badge / header is a system component** (it has a code home; it's rendered by `_render_validation_caption` / `_render_scenario_provenance_header`)
- REFERENCE §4 = **what the user sees** (the four states and what they mean for the dashboard reader)

### `[VERIFY] 5` — `app.py` line refs cited in DESIGN_NOTES

The City-specific copy convention section references `app.py ~3320–3333` as the runtime touchpoint. Reading that range today: lines 3324–3354 contain `_PROVENANCE_HEADER_INFO` (the provenance header table). The literal line numbers in DESIGN_NOTES drift over time — the rewrite should reference touchpoints by **stable symbol names** (e.g. `_PROVENANCE_HEADER_INFO` in `app.py`) rather than line numbers. The template's `Code touchpoints` field should use symbol references.

---

## Mapping rows worth flagging

1. **Lookup-overlay safety contract (line 1234) is load-bearing.** This is an *invariant about how the runtime works* (which fields are live-overwritten after a lookup hit). It overlaps directly with ARCHITECTURE map `[VERIFY] 4` (live-overwrite field list goes to ARCHITECTURE §5). **Risk of double-routing:** the ARCHITECTURE refresh pulls the invariant; the DESIGN_NOTES rewrite should keep ONLY the *why* (why we accept the overlay rather than redesigning the schema) and cross-ref ARCHITECTURE §5 for the field list. Calling this out so the two refreshes don't both list the 12 fields.
2. **NatCap ROOT (line 1386) overlaps ARCHITECTURE §10 ("Why not ROOT").** The mapping table correctly says §11 + §7 xref. The right split: ARCHITECTURE §10 has the one-paragraph "what ROOT is and why we don't use it" framing; DESIGN_NOTES §11 has the longer "Considered as a reference point; deferred because…" rationale with the alternatives that ROOT would have addressed.
3. **Topics not yet documented (line 2274) is forward-pointers, not decisions.** Keep as a tail-section in §11 or a separate appendix. Trim any items now done (the rewrite should mark which forward-pointers have landed since the list was last updated).
4. **The "UMH validation against canonical InVEST 3.19.0" section (1625) is the same decision the Brief B kernel-fix (1680) covers, just framed differently.** They're two halves of the same story (validation infrastructure + the buffer-mean kernel fix). The rewrite should consolidate them into a single §6 entry on the UMH model evaluation design + validation result, with NATCAP_ALIGNMENT holding the MAE numbers.

---

## Resolved decisions (bake in — LOCKED before rewrite)

- **Extraction-not-deletion** — brief entries hold original decisions; extract + restructure, route only narrative to HISTORY, delete nothing until its decision is captured.
- **HISTORY additions land first (or paired per brief) — never after the DESIGN_NOTES strip.** HISTORY lacks per-brief narratives for Briefs 1, 2, 4, 5, B, A2, B1, D1, B2-revised, #3, #4, #5 (per `[VERIFY] 3`). The strip step deletes those narratives from DESIGN_NOTES, so they must already live in HISTORY by the time the strip runs. Two acceptable orderings: (a) per-brief paired commits — extract decision + append HISTORY narrative + strip DESIGN_NOTES entry, one commit per brief; (b) a single HISTORY-additions commit landing the full set of brief narratives BEFORE the DESIGN_NOTES rewrite commit. Anything else risks silent narrative loss.
- **Stable symbol names in Code touchpoints, not line numbers (applies across all maps).** Cite `_PROVENANCE_HEADER_INFO`, `_compute_carbon_four_pool`, `_fmt_temp_change`, `_load_city_runtime_state`, etc. — not `app.py:3324`. Line numbers drift; symbol names don't (and grep finds them instantly).
- **UMH validation section + Brief B kernel-fix consolidate into one §6 entry.** They're two halves of the same decision (validation infrastructure + the canonical buffer-mean kernel). Consolidate; route the MAE-result narrative to NATCAP_ALIGNMENT; §6 keeps the design rationale.
- **Lookup-overlay safety contract is owned by DESIGN_NOTES §4** — it's a WHY-level safety invariant about the runtime. ARCHITECTURE §5 references it from DESIGN_NOTES §4 rather than restating the field list. (This flips the earlier route-out plan that risked double-listing the 12 live-overwrite fields.)
- DESIGN_NOTES = WHY; the ARCHITECTURE overlap-pairs (B1↔ARCH §2, D1↔ARCH §7, #4↔ARCH §5/§7) are why-here / how-there, cross-reffed not duplicated.
- §11 holds cross-cutting deferred approaches only; per-decision alternatives stay in the template field.

---

## Sequence after approval

1. **Approved + map committed.** ← awaiting approval.
2. **CC rewrites DESIGN_NOTES §1–11** (stop-and-report draft): extract each durable decision to the template, route narrative to HISTORY (after the capture check), place the incoming "why numpy" rationale in §6, §8 uses the locked vocabulary, **fix the UNA section against current per-city config**, mark the ARCHITECTURE cross-refs. Verify no durable decision dropped vs this map.
3. **Commit** — DESIGN_NOTES refresh as one concern; the HISTORY additions as a paired commit (per `[VERIFY] 3`).

**Coordination:** this refresh appends substantial narrative to HISTORY, which the REFERENCE and ARCHITECTURE refreshes also touch. Run it after their HISTORY touches (or coordinate the appends) so they don't collide — consistent with the suite sequence. Of the three docs, this is the last and largest restructure; it should land after REFERENCE and ARCHITECTURE so the cross-refs it points at already exist.

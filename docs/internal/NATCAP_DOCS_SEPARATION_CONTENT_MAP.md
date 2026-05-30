# NatCap docs separation — content map (ALIGNMENT · CITY_PARITY · COLLABORATION)

**Audience:** Internal
**Status:** Locked — approve-before-rewrite gate passed; ready to drive the three-way refactor
**Use this for:** Driving the three-way refactor of ALIGNMENT / CITY_PARITY / COLLABORATION and the deduplication contract
**Do not use this for:** Current alignment status — this doc is scaffolding for the refactor, not the alignment tracker itself
**Source of truth for:** The role contract, the single-home matrix, and the per-doc mappings for the NatCap-docs refactor

---

**Transient scaffolding.** Delete after all three refactors + link-fix passes land. Same discipline as REFERENCE / ARCHITECTURE / DESIGN_NOTES maps.

**Purpose:** map the three-way refactor that gives each NatCap doc one job and eliminates the duplication between them (the per-city parameter values currently live in all three). The single-home matrix below is the deduplication contract; the per-doc mappings execute it.

---

## Role contract (what each doc is for)

- **NATCAP_ALIGNMENT** — method/metric fidelity + validation truth. Organized by *model/metric*. Source of truth for: how each metric aligns with canonical InVEST, and the validation badge taxonomy.
- **CITY_PARITY** — per-city configuration parity. Organized by *city*. Source of truth for: per-city parameter/data/input parity.
- **NATCAP_COLLABORATION** — process log. Organized by *time/process*. Source of truth for: the collaboration history (asks, shared data, decisions, inferred priorities).
- **OPEN_QUESTIONS** (not refreshed here, but the boundary partner) — the *current-blocker dashboard*. COLLABORATION is the logbook; OPEN_QUESTIONS is the dashboard.

---

## Single-home matrix (the deduplication contract)

For each cross-cutting content type: exactly one home; everyone else cross-refs.

| Content | Single home | Everyone else |
|---|---|---|
| Per-city parameter **values** (UNA demand/radius/decay, UHI, rainfall, ET, biophysical tables) | **CITY_PARITY** | ALIGNMENT + COLLABORATION delete their copies → pointer |
| Validation **badge taxonomy** / validated–displayed–exploratory vocabulary | **NATCAP_ALIGNMENT §2** | REFERENCE §4 (minimal user restatement), ARCHITECTURE §6 (mechanics), DESIGN_NOTES §8 (rationale) cross-ref; app code is ground truth |
| **Method/metric fidelity** (is UCM canonical? UNA? carbon four-pool vs proxy?) | **NATCAP_ALIGNMENT §3** | CITY_PARITY references for per-city status |
| **Computed vs displayed** distinction | **NATCAP_ALIGNMENT §4** | REFERENCE §3/§4 + ARCHITECTURE §6 cross-ref |
| Per-city **data inputs / file-level parity** (MD5, paths) | **CITY_PARITY** | ALIGNMENT §2 deleted |
| Compound-LULC **structural inventory** (1,984×27 table internals) | **DATA_INVENTORY** | CITY_PARITY keeps the parity summary + pointer |
| **Live unresolved blockers** | **OPEN_QUESTIONS** | COLLABORATION keeps the logbook version |
| Collaboration **asks / shared data / decisions / priorities** | **NATCAP_COLLABORATION** | — |
| **Deferred alternatives** (PLUS/CLUE/LCM/ROOT rationale) | **DESIGN_NOTES §11** | ALIGNMENT + COLLABORATION point here |
| **Research-direction status** (NatCap-identified directions) | **NATCAP_COLLABORATION** (inferred priorities) | ALIGNMENT drops its §5 table |
| Brief chronology / retired metrics | **HISTORY** | — |

---

## NATCAP_ALIGNMENT — target structure + mapping

Target: §1 Alignment summary · §2 Validation badge taxonomy · §3 Metric methodology fidelity (per model) · §4 Computed vs displayed · §5 Known methodological divergences · §6 Export-to-InVEST validation boundary · §7 Link to city parity.

| Current section (line) | → | Note |
|---|---|---|
| Intro + "Validation story" (11–18, 99–106) | §1 | High-level summary; the per-pixel-parity-not-citywide-absolute framing. |
| §1 Metric Methodology Fidelity table + parity taxonomy (20–50) | §3 | **Trim per-city values out of the Nature Access + Carbon rows** (→ CITY_PARITY). Keep the method/parity claim. |
| "Validated reference outputs (SA)" badge block (52–88) | §2 + §4 | The badge taxonomy → §2 (**authoritative home**, keep the temp-can-cite-parity / carbon-cannot nuance). The natcap_published/aligned_method/prototype states + computed-vs-displayed → §4. |
| A3 "comparison-ready, never executed" (90–97) | §4 | The displayed-not-reproduced boundary. |
| Per-model validation in export bundle (108–112) | §6 | Validation travels with the bundle; "export ≠ validated." |
| §2 Data Source Alignment (114–142) | **CITY_PARITY** (delete) | Per-city data inputs — CITY_PARITY is the keeper. Split-config rationale → CITY_PARITY MN divergence note (or DESIGN_NOTES). |
| §3 Parameter Alignment (144–162) | **CITY_PARITY** (delete) | Pure per-city values — already in CITY_PARITY, in more detail. Replace with a §3 method statement + pointer. |
| §4 Spatial Fidelity (164–175) | **CITY_PARITY** (delete) | Per-city AOI → CITY_PARITY. |
| SA UNA / biophysical extent investigation, Brief A2 (177–204) | **docs/research/una/** | Investigation note; §3/§4 keep a one-line outcome + pointer (consolidate with the DESIGN_NOTES Brief-A2 routing — one home). |
| Research-direction synthesis + §5 Research Direction Status (206–261) | **COLLABORATION** + **DESIGN_NOTES §11** | NatCap-identified directions → COLLABORATION priorities; deferred-approach rationale (PLUS/CLUE/LCM/ROOT) → DESIGN_NOTES §11. |
| §6 Vocabulary and Reporting Alignment (262–282) | §3 + **HISTORY** | Durable canonical-term-per-metric → fold into §3 entries; "renamed on date" chronology → HISTORY. |
| Status legend / How to update (283–302) | keep (trimmed) | Update for the new section set. |

### §2 — authoritative badge taxonomy (locked spec) — **§2 reformat instruction**

This is the single written authority for the badge vocabulary (app code = ground truth above it; REFERENCE §4 / ARCHITECTURE §6 / DESIGN_NOTES §8 restate-minimally and point here). Keep it tight — this table plus a short note, not an essay.

**Rewrite instruction:** ALIGNMENT §2 currently exists as prose bullets at lines 79–88 of the live doc. The rewrite **reformats** it into the four-column table below + the two-point note. The badge strings (`NatCap published value` / `≈ NatCap method` / `≈ Aligned method` / `Prototype`) and color hints at lines 79–88 are confirmed correct against the locked vocabulary; this fold-in is about **format + completeness of the note**, not about changing the strings themselves.

| Badge | Meaning | What it claims | What it does not claim |
|---|---|---|---|
| NatCap published value | Displayed directly from NatCap reference output | This is NatCap's published number | Not independently reproduced |
| ≈ NatCap method | Computed with NatCap-aligned project method/data | Methodologically aligned | Not necessarily matched to a published scenario |
| ≈ Aligned method | InVEST-style/canonical method, no project-specific anchor | Comparable methodology | Not NatCap project-specific |
| Prototype | Exploratory proxy or assumption | Useful for exploration | Not a final quantitative result |

**Note below the table (both points, kept short):**
1. **Per-metric evidence varies within "≈ NatCap method."** Temperature can cite measured per-pixel parity (HMI MAE≈0); carbon is four-pool methodology adoption with no per-pixel parity measurement — do not imply parity for carbon.
2. **Badges are per-metric × per-context.** A `natcap_published` metric shows "NatCap published value" *only* in the fixed-scenario reference view; in baseline / Explorer / optimizer contexts the same metric shows "≈ NatCap method" (the prototype computed it). This is what prevents an Explorer-scenario number from reading as a NatCap-published one.

---

## CITY_PARITY — target structure + mapping

Target: §1 Principle (params are project-specific) · §2 Summary matrix (city × model) · §3 Minneapolis · §4 San Antonio · Minneapolis Full (dormant) · §5 Blocked parity items · §6 Link to methodology alignment. Largely **keep** — it's already organized by city and is the de-dup keeper.

| Current section | → | Note |
|---|---|---|
| Working principle (13) | §1 | Keep. |
| (new) | §2 | **Add the compact city × model summary matrix** (UCM/UNA/UMH/Carbon/Flood/Food/Export per city) — the per-model summaries exist; add the matrix on top. |
| Minneapolis + San Antonio per-model tables (24–262) | §3 / §4 | Keep. **Absorb** any ALIGNMENT-only detail (split-config buildings rationale → MN divergence note). |
| SA Compound LULC Framework — structural inventory (200–248) | **DATA_INVENTORY** | Move the 1,984×27 table internals; keep a parity summary ("SA uses NatCap compound tables, aligned") + pointer. |
| Minneapolis Full (dormant) (266–272) | keep | As-is. |
| Open questions about parity (276–285) | §5 + **OPEN_QUESTIONS** | City-specific parity Qs → §5; methodology-agnostic ones → OPEN_QUESTIONS. |

---

## NATCAP_COLLABORATION — target structure + mapping

Target: §1 Current collaboration summary · §2 Active asks · §3 Data received · §4 Decisions made because of NatCap input · §5 Gaps/blocked surfaced · §6 Questions to ask · §7 Closed/resolved · §8 Meeting notes & dated comms. **Demote to pure process log.**

| Current section | → | Note |
|---|---|---|
| (new) | §1 | **Short current summary** pointing to ALIGNMENT (alignment status), CITY_PARITY (per-city parity), OPEN_QUESTIONS (live blockers): "this file records how we got there." |
| Per-city parameter framing table (15–33) | §4 + pointer | Keep the **principle narrative** (project-specific by design — a real collaboration insight); the values table → pointer to CITY_PARITY. |
| Active asks (36–48) | §2 | Keep. |
| Inferred priorities (51–61) | §1/§4 | Keep; fold in the ALIGNMENT research-directions tracking. |
| Gaps (65–125) | §5 + **OPEN_QUESTIONS** | Logbook stays; **live** blockers → OPEN_QUESTIONS dashboard. |
| Open questions to raise, incl. Q12 flood-CN investigation (126–308) | §6/§7 + **HISTORY** + **OPEN_QUESTIONS** | Live Qs → OPEN_QUESTIONS (logbook copy in §6); resolved → §7; **Q12's heavy per-class CN tables → HISTORY** (Completed-workstream specifics — Q12 was resolved 2026-05-29 via the Mar_2023 pptx), summary + resolution stay in §7. "Resolve in-house" → §4. |
| Data NatCap has shared (311–322) | §3 | Keep. |
| Symposium and timeline (326–332) | §8 | Keep. |

---

## Resolved findings (`[VERIFY]` 1–5)

### `[VERIFY] 1` — De-dup coverage check (anti-loss): ALIGNMENT-only details that CITY_PARITY does NOT yet cover

Before deleting ALIGNMENT §2/§3/§4, **CITY_PARITY must first absorb these ALIGNMENT-only details**:

| ALIGNMENT-only detail | Current location | Goes into CITY_PARITY at |
|---|---|---|
| **Split-config buildings rationale** | ALIGNMENT lines 133–142 ("Placement-constraint inputs and model inputs serve different purposes…"; the `mask_buildings_file` vs `buildings_file` distinction; the *framing* that NatCap explicitly separates placement-constraint from model-input data) | Minneapolis section as a divergence/design note — currently CITY_PARITY mentions OSM + InVEST sample but does not carry the framing rationale |
| **SA UCM Köppen-BSh per-class tuning provenance** | ALIGNMENT §2 line 126 ("Köppen-BSh per-NLCD tuning retired; SA UCM consumes compound view directly") | SA §4 should carry the retirement note as a parity-history one-liner; the table provenance lives in `data/sa/cooling/biophysical_table_sources.md` (already cross-reffed) |
| **Population-overlap quantification for SA AOI** | ALIGNMENT §4 lines 184–192 (LULC-valid vs block-group pixel/area/population overlap, area IoU = 0.824, population overlap = 98.6 %) | SA AOI cell in the per-city §4 table — currently CITY_PARITY references the block-group polygons but does not carry the overlap numbers |
| **Per-model-row implementation detail** | ALIGNMENT §1 table cells (e.g., the UNA row says "MAE ≈ 0 + per-city parameters with values inline") | The cells already in CITY_PARITY are higher-level (✅/⚠️ summaries); the *method* claim (canonical 2SFCA, MAE numbers) needs to land somewhere — recommend leaving it in ALIGNMENT §3 (the *method* row), with CITY_PARITY referencing it for per-city status |

Once these absorptions land, ALIGNMENT §2/§3/§4 can be deleted cleanly. **Nothing deleted until confirmed captured.**

The dedup itself is real: the per-city UNA values (250/1000/exp MN; 16.7/800/dichotomy SA) currently live in THREE places — ALIGNMENT §3 lines 152–155, CITY_PARITY (UNA per-city tables), and COLLABORATION §1's per-city parameter framing table at lines 15–33. Per the matrix, CITY_PARITY is the keeper.

### `[VERIFY] 2` — Validation vocabulary single-source

**Confirmed: ALIGNMENT §2 (lines 79–88) is the right authoritative home.** The strings already match the locked vocabulary from the REFERENCE / ARCHITECTURE / DESIGN_NOTES maps:

- ALIGNMENT line 79: `"Green 'NatCap published value'"` — matches REFERENCE map's locked badge state #1
- ALIGNMENT line 82: `"Blue '≈ NatCap method'"` — matches state #2
- ALIGNMENT line 87: `"Blue '≈ Aligned method'"` — matches state #3
- ALIGNMENT line 88: `"Gray 'Prototype'"` — matches state #4

ALIGNMENT already carries the metric-aware tooltip nuance (line 85: "temperature CAN cite measured per-pixel HMI parity (Brief 28b); carbon must NOT") — this is the authoritative-spec detail the brief calls for. Restate-minimally and point here from REFERENCE §4 / ARCHITECTURE §6 / DESIGN_NOTES §8.

**Cross-doc anchor stability requirement:** the ALIGNMENT anchor `"Validated reference outputs (SA)"` is cited externally (CLAUDE.md:86; `app.py:3626` carries an in-app pointer). Either preserve this anchor under §2 in the refresh OR update those two external refs in the same commit.

### `[VERIFY] 3` — OPEN_QUESTIONS current state + COLLABORATION routing

**OPEN_QUESTIONS today** (`docs/internal/OPEN_QUESTIONS.md`) has two sections:
- **NatCap data requests** — (1) Per-scenario compound LULC inputs (PARKED), (2) Native NLCD×tree baseline flood raster (open, secondary)
- **Deferred briefs** — B2 — Per-metric validation markers (DEFERRED)

That is the full "current blockers" list today. It is NOT structured as a dashboard with consistent fields per blocker (status, owner, impact, ask). The brief's contract calls for it to BE the current-blocker dashboard — the rewrite needs to add that structure.

**COLLABORATION → OPEN_QUESTIONS routing required:**

The brief tells COLLABORATION's "Active asks" (lines 36–48) and "Gaps" (65–125) to route LIVE blockers to OPEN_QUESTIONS. I sampled COLLABORATION; concrete items to surface in OPEN_QUESTIONS:

| COLLABORATION location | Surface in OPEN_QUESTIONS as | Status today |
|---|---|---|
| §1 (lines 36–48) Active asks | Sharpen the per-scenario compound LULC ask (already in OPEN_QUESTIONS as parked); confirm the per-crop SA yield + MN Carbon four-pool ask there | Items 4a + 4b not yet in OPEN_QUESTIONS |
| Q1 — MN sample data still current? (line 132) | New OPEN_QUESTIONS entry | Not yet there |
| Q5 — SA NDR DEM + watersheds (line 140) | New OPEN_QUESTIONS entry | Not yet there |
| Q6 — Per-capita-only undersupply framing right for placement? (line 142) | New OPEN_QUESTIONS entry | Not yet there |
| Q11 — InVEST UNA edge handling at AOI boundary (line 144) | New OPEN_QUESTIONS entry | Not yet there |

**Net for the rewrite:** OPEN_QUESTIONS will gain ~5 new entries from COLLABORATION's live questions during the refactor. COLLABORATION keeps the logbook (the full narrative for each); OPEN_QUESTIONS holds the dashboard (concise status entry per blocker).

### `[VERIFY] 4` — Q12 home decision

**Q12 is RESOLVED 2026-05-29 via `Ben NDR and Flood Mar_2023.pptx`** (COLLABORATION lines 198–225). NatCap's CN values for SA reflect a documented design-storm-saturation framework; the prototype's behavior under the staged biophysical table matches NatCap's own modeled food-forest scenarios (+0.1 % to +1.1 % flood-volume increase). The deferral was reversed.

**Q12 is therefore a closed/completed workstream, not an open investigation.** The heavy detail (per-class NRCS-comparison CN table at lines 156–172, the framework explanation, the pptx-resolution narrative) is durable but historical reference content.

**Recommended home: HISTORY** (`docs/archive/HISTORY.md`'s "Completed-workstream specifics" section, which already houses similar resolved-workstream narratives like "Streamlit Cloud memory-fit workstream"). NOT `docs/research/` — that's for ongoing investigation notes, not resolved workstreams.

**What COLLABORATION keeps in §7 (Closed/resolved):**

A 3–5 line summary: *"Q12 — SA flood Curve Number table anomaly vs NRCS TR-55 (resolved 2026-05-29). Investigation found NatCap's CN values reflect a design-storm-saturation framework for SA's clay-rich D-soils; framework documented in `Ben NDR and Flood Mar_2023.pptx` slide 7. Full investigation + per-class CN comparison table → docs/archive/HISTORY.md "Completed-workstream specifics" → SA flood-CN investigation (2026-05-29)."*

**Anchor stability requirement:** The `"question 12"` anchor in COLLABORATION is cited by `config.py:172` and `CLAUDE.md:378, :710`. Two options:
- (a) keep the anchor in COLLABORATION's §7 (preferred — the cited content is a *summary + resolution pointer*, which §7 holds)
- (b) update the three external refs in the same commit

Recommend (a). The §7 summary preserves the anchor; the heavy detail moves to HISTORY without breaking the inbound refs.

### `[VERIFY] 5` — Inbound-reference inventory

#### NATCAP_ALIGNMENT.md (most-cited)

| File:line | Cited anchor / target | Action |
|---|---|---|
| `README.md:41`, `:58` | `docs/internal/NATCAP_ALIGNMENT.md` (Start here + Documentation map) | leave; full-path docs |
| `REFERENCE.md:366` | `Table 1` (per-metric alignment) | retarget → ALIGNMENT §3 (or whichever §3 sub-anchor holds the per-metric fidelity table) |
| `CLAUDE.md:86` | `"Validated reference outputs (SA)"` | **anchor must be preserved** in §2 or §4 |
| `app.py:3626` | generic "see NATCAP_ALIGNMENT.md" | leave bare |
| `natcap_validation.py:270` | generic | leave bare |
| `export_invest_bundle.py:66` | generic | leave bare |
| `docs/internal/DATA_INVENTORY.md:6, :13, :460, :481, :553` | generic + table-cell refs | leave |
| `docs/internal/DESIGN_NOTES.md:404` | "Tables 2, …" generic | retarget |
| `docs/internal/DESIGN_NOTES.md:1435` | `"Research-direction synthesis (~line 140)"` | **anchor will move** (Research-direction synthesis → COLLABORATION) — retarget |
| `docs/archive/SA_INTEGRATION_PLAN_2026-05.md:170` | archive | leave |

#### NATCAP_COLLABORATION.md

| File:line | Cited anchor / target | Action |
|---|---|---|
| `README.md:60, :87` | generic + Status section | leave |
| `CLAUDE.md:166, :378, :710` | `"question 12"` × 3 | **anchor must be preserved** in §7 (resolved) per VERIFY 4 |
| `config.py:172` | `"question 12"` | **anchor must be preserved** |
| `app.py:616` | generic | leave bare |
| `docs/internal/ARCHITECTURE.md:174` | generic | leave |
| `docs/archive/SA_INTEGRATION_PLAN_2026-05.md` | 3 refs, generic | leave |
| `docs/internal/DESIGN_NOTES.md:563, :906` | generic | leave |
| `docs/internal/NATCAP_ALIGNMENT.md:6` | header cross-ref | will be updated when ALIGNMENT header is rewritten |

#### CITY_PARITY.md

| File:line | Cited anchor / target | Action |
|---|---|---|
| `README.md:41, :59` | generic | leave |
| `docs/internal/ARCHITECTURE.md:68, :145, :172` | generic | leave |
| `docs/internal/DESIGN_NOTES.md:1081, :1117` | generic refs | leave |
| `docs/internal/DATA_INVENTORY.md:481` | generic | leave |
| `docs/archive/SA_INTEGRATION_PLAN_2026-05.md` | 2 refs, generic | leave |

#### Inter-doc cross-refs among the three NatCap docs

The three docs already cross-ref each other in their status headers (`NATCAP_ALIGNMENT.md:6` points to `CITY_PARITY.md` + `NATCAP_COLLABORATION.md`). When the refactor reorganizes content per the single-home matrix, every "see CITY_PARITY for per-city values" / "see ALIGNMENT for method status" / "see COLLABORATION for the conversation" cross-ref needs to be the cross-ref the doc was supposed to make all along but couldn't because the content was duplicated.

**Anchor-stability summary (across all 3 docs):**
- `"Validated reference outputs (SA)"` in ALIGNMENT — preserve in §2 or §4 (1 inbound: CLAUDE.md)
- `"Research-direction synthesis"` in ALIGNMENT — **retarget** the 1 inbound (DESIGN_NOTES.md:1435) to COLLABORATION
- `"question 12"` in COLLABORATION — preserve in §7 resolved (3 inbound: config.py + CLAUDE.md × 2)
- README's Documentation map rows for all three docs — leave (point at file, no anchor)

---

## Mapping rows worth flagging

1. **§5 Research Direction Status table in ALIGNMENT (lines 248–261) splits OUT to two destinations**, per the brief: NatCap-identified directions → COLLABORATION priorities; PLUS/CLUE/LCM/ROOT deferred-approach rationale → DESIGN_NOTES §11. The split is correct, but the row "San Antonio as full pilot" (line 257) is **neither** a research direction nor a deferred approach — it's an ongoing workstream. Recommend that row goes to COLLABORATION §4 (decisions made because of NatCap input) rather than into the priorities table.
2. **The "SA UNA / biophysical extent" investigation (ALIGNMENT lines 177–204) overlaps with DESIGN_NOTES Brief A2**, which the DESIGN_NOTES map already routes to `docs/research/una/`. **Consolidate** — one home for the investigation, not two. The DESIGN_NOTES routing wins (DESIGN_NOTES owns the decision-not-to-mask rationale); ALIGNMENT §3 keeps a one-line pointer.
3. **CITY_PARITY §2 (the new city × model summary matrix) is a content addition, not a routing decision.** The brief calls for adding it but doesn't specify its contents. Recommend rows = UCM, UNA, UMH, Carbon, Flood (UFR), Food, Export-availability; cols = MN downtown, MN Full (dormant), SA; cells = ✅ / ⚠️ / ❌ / ⏸️ short status from the existing per-city summaries. Compact, scannable, no detail.
4. **COLLABORATION §3 (Data received) is the right home for "data NatCap has shared" (current line 311), but the brief also routes "Decisions made because of NatCap input" to §4** — these overlap in the current doc (a shared dataset typically drives a downstream decision). Recommend: §3 lists *what* was received; §4 cross-refs §3 and adds *what changed because of it*. Don't double-list.
5. **The "Per-city parameter framing" narrative (COLLABORATION lines 15–33) and the principle in CITY_PARITY §1 are the same insight in two places.** Per the matrix, CITY_PARITY §1 is the principle home; COLLABORATION §4 keeps the *narrative* (project-specific by design — a real collaboration insight) but drops the values table (CITY_PARITY has it) and adds a pointer. Mapping table is right; flagging to make sure the narrative-vs-principle split is explicit.

---

## Resolved decisions (bake in — LOCKED before rewrite)

- **De-dup contract is the deletion gate.** Nothing deleted from ALIGNMENT §2/§3/§4 until CITY_PARITY confirms it has absorbed the rows AND the ALIGNMENT-only details listed in `[VERIFY] 1`. The split-config buildings rationale is the load-bearing example.
- **ALIGNMENT §2 is the authoritative badge-vocabulary spec, in four-column-table form.** The rewrite reformats §2 from the current prose bullets (lines 79–88) into the four-column table (Badge / Meaning / What it claims / What it does not claim) + the two-point note below the table. REFERENCE §4 / ARCHITECTURE §6 / DESIGN_NOTES §8 cross-ref §2 — they don't restate. App code is ground truth above §2.
- **Q12's heavy detail → HISTORY** ("Completed-workstream specifics"); summary + resolution stay in COLLABORATION §7. The `"question 12"` anchor preserved in §7 to keep the three external refs (`config.py:172`, `CLAUDE.md:378, :710`) stable. Q12 is RESOLVED 2026-05-29 — treat as completed-workstream content, not deferred.
- **"SA as full pilot" routes to COLLABORATION §4** (decisions made because of NatCap input) — NOT into the priorities table (it's an ongoing workstream, not a research direction).
- **"SA UNA / biophysical extent" investigation has ONE home** at `docs/research/una/` (per the DESIGN_NOTES Brief-A2 routing). ALIGNMENT §3 keeps a one-line pointer; no duplicate investigation note in ALIGNMENT.
- **OPEN_QUESTIONS gets a structural refresh, not an append.** When OPEN_QUESTIONS is eventually refactored, it (a) restructures every entry into a uniform dashboard format with fields `status / owner / impact / ask`, AND (b) imports the five live questions from COLLABORATION identified in `[VERIFY] 3` (Q1 MN-current, Q5 SA NDR DEM, Q6 per-capita-only undersupply, Q11 UNA edge handling, items 4a + 4b per-crop SA yield + MN Carbon four-pool). Both pieces — structural refresh AND the five imports — must happen in the same OPEN_QUESTIONS pass; merely appending new entries to the current narrative-section structure does NOT satisfy this decision.

---

## Sequence

1. **NATCAP_ALIGNMENT first** — it defines the validation language the app, README, REFERENCE, and demo all depend on. Establish §2 as the authoritative taxonomy; trim per-city values out; route research-directions + vocabulary. **Anchor preservation:** "Validated reference outputs (SA)" stays.
2. **CITY_PARITY** — absorb the ALIGNMENT-only details from `[VERIFY] 1` first, add the §2 matrix, move the structural inventory to DATA_INVENTORY, point to ALIGNMENT.
3. **NATCAP_COLLABORATION** — demote to process log, trim the param table to a pointer, route the 5 live blockers to OPEN_QUESTIONS, move Q12's heavy detail to HISTORY, **preserve `"question 12"` anchor in §7**.
4. **OPEN_QUESTIONS** gets its current-blocker list populated from COLLABORATION (the dashboard). Restructure to dashboard fields per entry (status/owner/impact/ask).
5. **HISTORY** receives the Q12 investigation under "Completed-workstream specifics."
6. Cross-ref sweep with the rest of the suite (REFERENCE §4 / ARCHITECTURE §6 / DESIGN_NOTES §8 retarget cross-refs to ALIGNMENT §2; `DESIGN_NOTES.md:1435` retargets to COLLABORATION).

Each step is its own commit; held local and batch-pushed with the suite.

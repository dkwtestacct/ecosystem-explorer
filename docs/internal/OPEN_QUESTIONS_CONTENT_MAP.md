# OPEN_QUESTIONS.md refresh — content map

**Audience:** Internal
**Status:** In review — `[VERIFY]` 1–5 resolved against the live docs + code; awaiting approval before rewrite
**Use this for:** Driving the OPEN_QUESTIONS structural refresh (narrative → dashboard) and the COLLABORATION-imports / B2-routing decisions
**Do not use this for:** Current blocker list — this doc is scaffolding for the refactor, not the dashboard itself
**Source of truth for:** The triage from candidate list → dashboard ceiling, and the per-item routing

---

**Transient scaffolding.** Delete after the refresh + cross-ref-fix pass land. Same discipline as the other content-maps.

**Purpose:** turn OPEN_QUESTIONS from a short narrative into a tight current-blocker **dashboard**. This is the last doc in the suite and it *receives* content from the NatCap refactor (the COLLABORATION→OPEN_QUESTIONS routing), so it runs after / with the NatCap trio.

**Governing rule:** the dashboard answers "what still blocks or meaningfully shapes future work?" — not "every open question." Target **5–8 items, 1–2 pages**. If it grows past that, it's a running log again. The discipline that keeps it short is the per-item **"Where details live"** pointer — the narrative lives in COLLABORATION / DESIGN_NOTES / a research note; the dashboard points.

---

## Header + structure

Header: **Status:** Current blocker list · **Use this for:** Durable unresolved questions, external dependencies, decisions that shape future work · **Do not use this for:** Running notes, meeting prep, historical investigations, or resolved questions · **Source of truth for:** What is still blocked or undecided. Plus the rule: *only durable unresolved questions belong here; resolved ones move to HISTORY / DESIGN_NOTES / COLLABORATION.*

Sections: §Summary (one paragraph) · §External data dependencies · §Methodology decisions not yet made · §Deferred research directions (pointers only) · §Recently closed (very short).

**Per-item template:** Status (Open / Parked / Blocked externally / Decision needed) · Why it matters · Current best understanding · What would resolve it · Where details live.

---

## The triage (the key step)

Two lists feed in and must be deduped against the 5–8 ceiling: the **keep-5** from the plan and the **5 imports** the NatCap refactor routes from COLLABORATION ([VERIFY] 3: MN-data-current, SA NDR DEM, undersupply-framing, UNA-edge-handling, carbon/yield). Deduped that's ~10 candidates — over the ceiling.

After triage + the `[VERIFY] 3` decision on region-selection (resolved below: speculative, not near-term), the dashboard lands at **6 items**, comfortably within the 5–8 ceiling:

**Dashboard (durable blockers / decisions):**

| # | Item | Section | Notes |
|---|---|---|---|
| 1 | **Reproduce NatCap's published SA citywide figures** | External data | **Consolidates the SA-reproduction trio.** One entry; under "What would resolve it" list the three missing artifacts: the six per-scenario compound LULCs (or the overlay script), the UCM temp args (t_ref / uhi_max / aggregation behind 90.08 °F), the carbon aggregation script (behind 107.32M). Phrase as a *reproduction boundary*, not an app blocker. **Anchor preservation** — see `[VERIFY] 5`: keep `"Per-scenario compound LULC inputs"` as a level-3 sub-anchor within this entry so the 5 inbound refs survive. |
| 2 | **MN four-pool carbon table** | External data | Blocks MN parity with SA's four-pool framing. (= keep-4 / import-4b.) |
| 3 | **SA NDR inputs (DEM + watersheds)** | External data | External data gating NDR. (= import Q5.) |
| 4 | **Per-crop SA food-forest yields** | External data | Per `[VERIFY] 4`: kept at low-priority status — COLLABORATION frames it as "modest fidelity upgrade," but it's a real data ask with NatCap, so dashboard it. (= keep-5 / import-4a.) |
| 5 | **Whether to implement NDR** | Methodology decision | Pairs with the NDR-data row but is a separate decision. |
| 6 | **Synthetic NDVI → satellite NDVI** | Methodology decision | Affects NDVI + UMH cards. |

**Route to Deferred research directions (pointers only):**
- **Region-selection design.** Per `[VERIFY] 3`: speculative, not near-term — only signal is ARCHITECTURE §11 ("no UI surface yet drives it"). Cross-ref ARCHITECTURE §11 from the Deferred-directions section.
- **AlphaEarth NDVI replacement.** Pointer to `docs/research/ALPHAEARTH_FEASIBILITY.md` + DESIGN_NOTES.
- **PLUS / CLUE / LCM / ROOT** (deferred alternatives). Pointer to DESIGN_NOTES §11.

**Route to COLLABORATION logbook (NOT the dashboard):**
- **"MN sample data still current?"** (import Q1) — a verify-with-NatCap, process not blocker.
- **UNA edge handling at AOI boundary** (import Q11) — technical / in-house item for the logbook unless it becomes a genuine pending decision.
- **Per-capita-only undersupply framing for placement** (import Q6) — same; technical item for the logbook.

---

## Move OUT

- **B2 deferral** (currently OPEN_QUESTIONS §2) — per `[VERIFY] 1`: the B2-revised conservative-floor work shipped (badges live in `natcap_validation.render_validation_badge`; commits `e0f5492` + `9fca481` + `0dc4726`). The original Match/Diverged per-scenario design remains gated on compound inputs but is captured by the consolidated reproduction entry above. Route: the *narrative* of the deferral → HISTORY ("Completed-workstream specifics"); the *preserved Phase 0 design work* (card inventory + b2 reference-view recommendation + three-open-decisions) → **DESIGN_NOTES §11, absorbed during the DESIGN_NOTES rewrite** (next-up, ahead of the trio). The OQ refresh just deletes the entry — the design artifact already lives there.
- **File-hunt / Drive-connector narratives** (currently in the SA-LULC parked block) — summarize to one sentence + pointer to COLLABORATION; the dashboard doesn't carry the search detail.
- **Send-ready email draft** (currently lines 111–131) — keep one short pointer in the consolidated reproduction entry's "Where details live"; the verbatim draft lives in COLLABORATION's "Active asks" section as the canonical home.
- **Second item: Native NLCD×tree baseline flood raster** — per `[VERIFY] 1`: fold into the consolidated reproduction entry as a fourth "What would resolve it" line (it's a secondary local-fallback option), OR keep as a sixth dashboard item — recommend FOLD (the reproduction-boundary framing already covers it).

---

## Resolved findings (`[VERIFY]` 1–5)

### `[VERIFY] 1` — Current OPEN_QUESTIONS inventory + per-entry routing

The live `docs/internal/OPEN_QUESTIONS.md` has three sections:

| Current section / item | Routing |
|---|---|
| **§1.1 Per-scenario compound LULC inputs (PARKED, not sent)** — lines 20–131. Includes: the need, the local content-signature hunt narrative, the Google Drive connector search, the sharpened ask, the Option 2 (local reconstruction) caveat, the impact-if-never-obtained block, the send-ready email draft. | **Folds into consolidated dashboard item #1.** Heavy narrative (hunt + Drive connector + email draft) → trim to one-sentence pointer to COLLABORATION's "Active asks." Send-ready email draft → COLLABORATION (canonical home). The "impact if never obtained" block → the consolidated entry's "Why it matters" + "Where details live" pointers. **Preserve the `"Per-scenario compound LULC inputs"` anchor** as a level-3 sub-anchor under the consolidated entry. |
| **§1.2 Native NLCD×tree baseline flood raster** — lines 134–161. Includes: the need (NatCap's *baseline* in same encoding), the local-fallback options (re-derive through compound→NLCD×tree reduction, or suppress fixed-scenario flood delta). | **Folds into consolidated dashboard item #1** as a fourth artifact under "What would resolve it." The local-fallback options are a tactical decision-not-a-blocker — route to DESIGN_NOTES if not already there, or HISTORY if already resolved-but-undocumented. |
| **§2 B2 — Per-metric validation markers · DEFERRED** — lines 166–273. Includes: the 2026-05-29 update note (partial unblock via B2-revised), the what-it-was, the why-deferred, the revisit-only-if conditions, and the preserved Phase 0 design work (lines 210–273: card inventory + (b2) reference-view recommendation + three open decisions + their recommendations). | **B2-revised shipped** — `natcap_validation.render_validation_badge` is live (lines 194+), `_render_validation_caption` wired into every metric card, the SA fixed-scenario reference view rendered. Original Match/Diverged design is **captured by the consolidated reproduction entry** (compound inputs are the gate). Routing: the deferral *narrative* → HISTORY "Completed-workstream specifics" (alongside Q12, streamlit-memory-fit). The *Preserved Phase 0 design work* → decision below in mapping rows worth flagging. |

### `[VERIFY] 2` — keep-5 vs 5 imports against the 5–8 ceiling

| Source | Item | Dashboard or routed |
|---|---|---|
| Keep-5 (plan) | Per-scenario compound LULC inputs | dashboard #1 (consolidated) |
| Keep-5 | Native NLCD×tree flood raster | folded into #1 |
| Keep-5 | B2 deferral | → HISTORY (shipped/resolved) |
| Keep-5 | MN four-pool carbon table | dashboard #2 |
| Keep-5 | Per-crop SA yields | dashboard #4 |
| NatCap import Q1 | MN sample data still current? | → COLLABORATION logbook |
| NatCap import Q5 | SA NDR DEM + watersheds | dashboard #3 |
| NatCap import Q6 | Per-capita-only undersupply framing | → COLLABORATION logbook |
| NatCap import Q11 | UNA edge handling at AOI boundary | → COLLABORATION logbook |
| NatCap import 4a + 4b | per-crop SA yields + MN Carbon | dedupes with keep-5 entries (already on dashboard as #2 + #4) |
| New decisions | Whether to implement NDR | dashboard #5 |
| New decisions | Synthetic NDVI → satellite NDVI | dashboard #6 |
| New decisions | Region-selection design | → Deferred directions (per `[VERIFY] 3`) |

**Dashboard total: 6 items.** Comfortably within the 5–8 ceiling. **COLLABORATION absorbs 3 new logbook entries** (Q1, Q6, Q11). HISTORY absorbs the B2-deferral narrative.

### `[VERIFY] 3` — Region-selection: speculative, not near-term

**Decision: Deferred directions, NOT the dashboard.**

Only signal in the repo for region-selection is ARCHITECTURE §11 ("Future architecture hooks"), which I just wrote — and the framing is explicit: *"no UI surface yet drives it."* No planning signal elsewhere — no roadmap mention, no WHATS_NEW entry, no in-flight branch, no DESIGN_NOTES decision. The conceptual seam exists in `evaluate_scenario`'s extensibility (`candidate_pixels = CONVERTIBLE_PIXELS ∩ selected_region_mask`) but a real "we're shipping region-selection in Q3" plan doesn't.

Route: **§Deferred research directions** as a one-line pointer to ARCHITECTURE §11. No bidirectional cross-ref needed since ARCHITECTURE §11 doesn't claim it's near-term either.

If a near-term region-selection plan materializes after this refresh: a new dashboard item replaces the Deferred-directions pointer; the question of what layers first (districts / block-groups / neighborhoods / drawn polygons) and how region-scenarios interact with the lookup table + optimizer becomes a real "Methodology decision not yet made" entry.

### `[VERIFY] 4` — Per-crop SA yields: keep on dashboard (low priority)

COLLABORATION line 137 frames the per-crop SA food-forest yields ask as a "Modest fidelity upgrade for SA" — explicitly *not* blocking, but a real data ask with NatCap.

**Decision: Keep on dashboard as item #4** with Status framing matching the data-not-blocker shape: *Open (Blocked externally — data not yet provided)* + a "Current best understanding" line that says the 8,500 lbs/acre placeholder is operational and produces directionally-correct results.

Demoting to "nice-to-have data" was the alternative; reject because the dashboard's contract is "external dependencies + methodology decisions that shape future work," and a per-crop yield ask is a data ask — exactly the shape the dashboard is for.

### `[VERIFY] 5` — Inbound refs to OPEN_QUESTIONS section anchors

**5 inbound refs all target one anchor:** `"Per-scenario compound LULC inputs"`.

| File:line | Cited anchor |
|---|---|
| `app.py:3563` | `"Per-scenario compound LULC inputs"` |
| `export_invest_bundle.py:20` | generic OPEN_QUESTIONS.md mention (no anchor) |
| `docs/internal/DESIGN_NOTES.md:1808` | `"Per-scenario compound LULC inputs"` |
| `docs/internal/ARCHITECTURE.md:70` | generic OPEN_QUESTIONS.md mention (no anchor — just-rewritten) |
| `docs/internal/NATCAP_ALIGNMENT.md:102` | `"Per-scenario compound LULC inputs"` |

**Anchor-stability requirement:** the consolidated reproduction entry (dashboard item #1) is titled `"Reproduce NatCap's published SA citywide figures"`. **The level-3 sub-anchor `"Per-scenario compound LULC inputs"` must be preserved within this entry** (as a `### Per-scenario compound LULC inputs` heading under the consolidated dashboard item) so the three explicit inbound refs (`app.py:3563`, `DESIGN_NOTES.md:1808`, `NATCAP_ALIGNMENT.md:102`) still resolve. The two generic refs (no anchor) survive automatically.

---

## Mapping rows worth flagging

1. **Preserved Phase 0 design work** (current OPEN_QUESTIONS lines 210–273) lands in **DESIGN_NOTES §11** as part of the DESIGN_NOTES rewrite — not as a separate later step. The DESIGN_NOTES rewrite (next-up, ahead of the trio) reads OPEN_QUESTIONS's current content and absorbs the Phase 0 design (card inventory, b2 reference-view recommendation, three-open-decisions) + the ROOT/PLUS/CLUE/LCM deferred rationale into a complete §11. By the time OPEN_QUESTIONS refreshes, the design artifact already lives in DESIGN_NOTES §11 — the OQ refresh just deletes the extracted B2 entry. The deferral chronology stays in HISTORY ("Completed-workstream specifics"), separate from the design artifact.
2. **The §Summary paragraph (one-paragraph open) needs to anchor the "what changed since last refresh" delta.** If the previous OPEN_QUESTIONS read as a narrative log, this rewrite's §Summary should explicitly say "this is now a dashboard; for the running narrative see COLLABORATION; for resolved questions see HISTORY." That's the wayfinding the new readers need.
3. **§Recently closed (very short) is bait for clutter.** Two options: (a) keep, but with a hard 3-item cap and a 30-day TTL — items rotate out into HISTORY/DESIGN_NOTES; (b) drop, and put the link "what was recently resolved? → HISTORY (Completed-workstream specifics)" in the §Summary. **Recommend (b)** — the section is an attractive nuisance otherwise.
4. **NatCap-trio coordination.** This refresh consumes content the NatCap trio routes outward — specifically Q1/Q6/Q11 → COLLABORATION logbook, the B2-deferral narrative → HISTORY, the Phase 0 design → DESIGN_NOTES. The cleanest order: NatCap-trio commits first (its rewrite establishes the destinations), then OPEN_QUESTIONS commits (the receiver). Or one coordinated commit pair. Don't run OPEN_QUESTIONS standalone; the destinations don't exist yet.

---

## Resolved decisions (bake in — LOCKED before rewrite)

- **Dashboard size — 6 items**, with section split: 4 External data (Reproduce SA citywide / MN four-pool / SA NDR inputs / Per-crop yields) + 2 Methodology decisions (Whether to implement NDR / Synthetic→satellite NDVI). Deferred directions section gets Region-selection + AlphaEarth + PLUS/CLUE/LCM/ROOT pointers; COLLABORATION logbook gets Q1 + Q6 + Q11.
- **Preserve `"Per-scenario compound LULC inputs"` as a level-3 sub-anchor** inside dashboard item #1 (Reproduce NatCap citywide). Three inbound refs depend on it.
- **B2-revised shipped, but the original Match/Diverged design stays gated.** The OPEN_QUESTIONS B2 entry exits the dashboard. The deferral narrative goes to HISTORY ("Completed-workstream specifics"); the Phase 0 design work goes to DESIGN_NOTES §11 — **absorbed during the DESIGN_NOTES rewrite (next-up), not as a separate later step**. When OPEN_QUESTIONS refreshes, the entry is just deleted; no second touch to DESIGN_NOTES §11.
- **Region-selection is Deferred, not near-term.** Pointer to ARCHITECTURE §11 in §Deferred research directions; not a dashboard item.
- **Per-crop SA yields stay on the dashboard at low priority.** External data ask, not "nice-to-have demotion."
- **Run after / with the NatCap trio.** The destinations COLLABORATION-logbook / HISTORY / DESIGN_NOTES need to exist before OPEN_QUESTIONS can route content there.
- **No §Recently closed section** — the surface invites clutter. §Summary links to HISTORY for "what was recently resolved."

---

## Sequence

Runs **after / with the NatCap trio**: NatCap commits first (establishes the COLLABORATION logbook / HISTORY destinations); OPEN_QUESTIONS commits second (routes content there). Then the cross-ref sweep + README reconciliation close the suite.

**Note on DESIGN_NOTES §11.** The full DESIGN_NOTES rewrite — next-up, ahead of the trio — lands §11 *complete* by absorbing OPEN_QUESTIONS's current B2 Phase 0 design + the ROOT/PLUS/CLUE/LCM deferred rationale at rewrite time. No standalone "DESIGN_NOTES §11" step exists in the back-half ordering. When OPEN_QUESTIONS refreshes, it just deletes the extracted B2 entry — the design artifact already lives in DESIGN_NOTES §11 from the earlier DESIGN_NOTES rewrite.

Concrete back-half commit order (DESIGN_NOTES rewrite happens BEFORE this):

1. **NATCAP_ALIGNMENT** refresh — establishes the §2 badge-vocabulary spec and trims per-city values.
2. **CITY_PARITY** refresh — absorbs the ALIGNMENT-only details + the §2 matrix.
3. **NATCAP_COLLABORATION** refresh — establishes the logbook destinations + the §7 Closed/resolved (Q12).
4. **HISTORY** paired commits — receives Q12 detail + B2-deferral narrative (+ the DESIGN_NOTES brief narratives, already queued from earlier).
5. **DATA_INVENTORY** refresh — keep-by-category + controlled status column (no by-city rebuild); dissolves §15 across OQ dashboard + catalog Notes + HISTORY; move-outs per the single-home matrix. Per-doc map ready at `/mnt/user-data/outputs/DATA_INVENTORY_CONTENT_MAP.md` (queued). Sits ahead of OPEN_QUESTIONS because §15 feeds OQ; doc-index → README; parity claims → CITY_PARITY; brief-chronology → HISTORY — all those upstream destinations need to exist first.
6. **OPEN_QUESTIONS refresh — this map.** All destinations now exist (DESIGN_NOTES §11 already holds the absorbed B2 design); route content + restructure to dashboard format. The B2 entry deletion is clean — the design artifact lives in DESIGN_NOTES §11 from the earlier DESIGN_NOTES rewrite.
7. **Cross-ref sweep + README reconciliation** — retarget remaining inbound refs across the suite; README touch-ups (title, repo-layout, doc-map row for CLAUDE.md at root) fold in here.

Each step is its own commit; held local and batch-pushed with the suite.

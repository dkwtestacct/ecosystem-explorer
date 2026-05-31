# Cross-ref sweep — running checklist

**Audience:** Internal
**Status:** Running checklist — accumulates through the rewrite phase; cleared at the closing-step cross-ref sweep commit
**Use this for:** Tracking inbound anchor cites + per-doc reconcile items that the cross-ref sweep needs to resolve
**Do not use this for:** Per-decision rationale (→ DESIGN_NOTES.md), or the doc rewrites themselves (each owns its own content map)
**Source of truth for:** What still needs to update across the suite when the cross-ref sweep runs

---

**Transient scaffolding.** Delete after the closing-step cross-ref sweep + README + CLAUDE.md trim land. Same discipline as the content-map files.

**Purpose.** As the rewrite phase strips and renames anchors across the suite, inbound cites from other docs / code / CLAUDE.md drift stale. Rather than co-update them in every per-doc commit (seven scattered updates per rewrite), defer to one closing-step sweep — but record each broken cite the moment it's surfaced so it isn't lost.

The closing-step ordering (per the suite plan): `DESIGN_NOTES rewrite` → `NATCAP trio` → `HISTORY paired commits` → `DATA_INVENTORY` → `OPEN_QUESTIONS` → **cross-ref sweep + README + CLAUDE.md trim (closing step)**.

---

## 1. Inbound anchor cites — DESIGN_NOTES rewrite (committed 2026-05-30)

The §1–11 template restructure renamed most anchors (chronology stripped, dates stripped, brief numbers stripped). Seven inbound cites point to old anchor names:

| Source | Line | Old anchor cited | Replace with |
|---|---|---|---|
| `docs/internal/OPEN_QUESTIONS.md` | :160 | `DESIGN_NOTES.md "Brief B1"` | (B1 narrative routed to HISTORY) DESIGN_NOTES §3 (compound LULC) + §4 (conversion logic) + §8 (validation/provenance) **or** `../archive/HISTORY.md` "Brief narrative chronology — Brief B1" |
| `docs/internal/DATA_INVENTORY.md` | :482 | `DESIGN_NOTES.md "NLCD legacy vs Annual NLCD"` | DESIGN_NOTES §3.1 |
| `docs/internal/DATA_INVENTORY.md` | :484 | `DESIGN_NOTES.md "SA flood damage table — resolved (Path C, Brief 33)"` | DESIGN_NOTES §6.5 |
| `CLAUDE.md` | :163 | `DESIGN_NOTES.md "Brief 4 — \`cooling_f\` → \`temp_change_f\` sign-convention refactor"` | DESIGN_NOTES §10.1 |
| `CLAUDE.md` | :528 | `DESIGN_NOTES.md "SA Carbon four-pool framework adoption"` | DESIGN_NOTES §6.4 |
| `app.py` (code comment) | :1409 | `DESIGN_NOTES.md "Brief B — UMH NE kernel: Gaussian → buffer-mean"` | DESIGN_NOTES §6.3 (anchor preserved: `UMH validation against canonical InVEST 3.19.0`) |
| `app.py` (code comment) | :3288 | `DESIGN_NOTES.md "Brief B2 (revised)"` | DESIGN_NOTES §8.1 (or §8.1 + §11.4 for the deferred Match/Diverged) |

**One inbound cite survives intact:** `app.py:4013 → DESIGN_NOTES.md "Lookup-overlay safety contract"` — anchor preserved at §4.4. No edit needed.

**CONTRIBUTING.md anchor preserved:** `docs/dev/CONTRIBUTING.md:70 → DESIGN_NOTES.md "UMH validation against canonical InVEST 3.19.0"` — the §6.3 anchor was explicitly preserved (the rewrite carries a `> **Anchor preserved:** ...` callout). No edit needed.

---

## 2. Per-doc reconcile items surfaced during the rewrite phase

### 2.1 DATA_INVENTORY §9.3 — stale UNA-demand claim (surfaced 2026-05-30)

**Current text** (line 383): *"Shared per-city scalars: demand `UNA_DEMAND_M2_PER_CAPITA = 16.7` (constant in app.py — per-city values match in current configs)."*

**Why stale.** Doubly wrong:
- `UNA_DEMAND_M2_PER_CAPITA` is not a constant — it's bound from `city_cfg['una_demand_m2_per_capita']` at `app.py:1153`, so at runtime it's 250 for MN, 16.7 for SA.
- Per-city values do NOT match: `config.py` declares MN 250 m²/capita (line 55), MN Full 250 (line 118), SA 16.7 (line 274).

**Fix during DATA_INVENTORY refresh.** Replace with a per-city pointer: *"Per-city `urban_nature_demand` (MN 250 / SA 16.7), `search_radius_m` (MN 1000 / SA 800), and `decay_function` (MN exponential / SA dichotomy) — values declared in `config.py` per the per-city framework principle (DESIGN_NOTES §2.1); current per-city values + parity status in CITY_PARITY.md UNA rows."* No value table here.

**Single-home anchor.** CITY_PARITY.md UNA rows (MN section lines 62–76, SA section lines 155–168) are the source of truth for the values. DESIGN_NOTES §2.2 also points at CITY_PARITY for values (no value table in DESIGN_NOTES either) — consistent.

---

## 3. NatCap trio — items accumulating

### 3.1 ALIGNMENT chronology strip — Commit 4 (trim)

Strip `"(Brief B2 revised, 2026-05-29)"` chronology from the lead-in sentence at NATCAP_ALIGNMENT.md line 77: *"Surfaced in the dashboard via per-metric validation badges (Brief B2 revised, 2026-05-29):"* → *"Surfaced in the dashboard via per-metric validation badges:"*. The §2 badge taxonomy is the authoritative current-state spec; the brief-number prefix belongs in HISTORY chronology, not in the spec lead-in. Catch as part of the trim's chronology pass.

### 3.2 SA compound-LULC structural inventory — CITY_PARITY → DATA_INVENTORY **consolidation** (not raw move)

Currently CITY_PARITY lines 222+ (`### SA Compound LULC Framework (structural inventory)` — the 1,984×27 `lulc_crosswalk` + ucm/una/carbon table column counts + `urban_nature` distribution 976/48/960 + four-pool max values + LULC raster comparison + integration implications).

**The DATA_INVENTORY refresh must consolidate this with §2 Land cover and land use, not append a duplicate.** DATA_INVENTORY §2 already holds substantial compound-LULC catalog content (`land_use_compound_sa.tif` entry at line 108; the raw NatCap-curated source rasters table at lines 113–121 including `lulc_overlay_3857.tif` + the NLCD/NLUD/tree component layers; the 1,984-lucode space described at line 124; the committed `data/sa/natcap_2024/` files inventory at lines 126–141 listing the three biophysical tables + `lulc_crosswalk.csv` + the canopy-QA docs).

What CITY_PARITY's structural inventory adds that §2 doesn't yet have: per-table column counts (27/21/27), the `urban_nature` distribution (976 / 48 / 960 rows), the per-pool max values (c_above 105.7, c_below 8.0, c_soil 259.0, c_dead 14.4), and the parity-style LULC raster comparison table (prototype 1984×1713 EPSG:5070 vs NatCap 2106×2218 EPSG:3857).

**Refresh action:** merge those specifics into the existing §2 entries (per-table rows in the committed-files table; per-pool maxes into a Carbon biophysical detail; LULC raster comparison into the SA dual-raster pipeline narrative); CITY_PARITY drops its structural-inventory section, replaced with a one-line pointer to DATA_INVENTORY §2.

The NatCap-trio refactor explicitly retains the CITY_PARITY content through Commit 4 (ALIGNMENT trim); the DATA_INVENTORY refresh consolidates afterward.

Note: the original Commit-2 framing referenced "ALIGNMENT → DATA_INVENTORY", but the structural inventory actually lives in CITY_PARITY, not ALIGNMENT. ALIGNMENT only carries a 1-row Table 2 entry (line 122) referencing the compound LULC raster.

### 3.3 Dual-home cross-references at OPEN_QUESTIONS + DATA_INVENTORY refreshes

The same underlying question can legitimately live in three docs at once, with **distinct framings**:

- **NATCAP_COLLABORATION §6 (Open questions to raise)** — ask-framing: how to phrase the question for a NatCap conversation. Logbook copy.
- **OPEN_QUESTIONS** — blocker-framing: status / owner / impact / ask fields, dashboard form. Live state.
- **DATA_INVENTORY §15 (Open questions)** — data-availability framing: which file is missing or which dataset hasn't been received.

**At the OPEN_QUESTIONS refresh:** when importing the five live questions identified in the trio map (Q1 MN-current, Q5 SA NDR DEM, Q6 per-capita-only undersupply, Q11 UNA edge handling, items 4a + 4b per-crop SA yield + MN Carbon four-pool), each entry should carry an explicit `see COLLABORATION §6 #N for the conversation framing` pointer so the two homes stay aligned. The COLLABORATION logbook copy stays — they serve different purposes.

**At the DATA_INVENTORY refresh:** when §15 mentions a missing dataset that's also tracked in COLLABORATION/OPEN_QUESTIONS as a question, add a cross-ref pointer in the same direction (`see OPEN_QUESTIONS dashboard for status, COLLABORATION §6 #N for the NatCap framing`).

**Question-numbering schemes are independent.** COLLABORATION uses Q1/Q4a/Q4b/Q5/Q6/Q7/Q8/Q9/Q10/Q11 (gaps + chronological original numbers). OPEN_QUESTIONS will get its own dashboard-form numbering at the refresh. DATA_INVENTORY §15 numbers from 1. Don't conflate the three schemes — each home owns its own numbering, and cross-refs use the source doc's number explicitly (e.g. "COLLABORATION Q5").

### 3.4 ALIGNMENT trim — inbound cites broken by the §1–§7 restructure

The Commit 4 trim deleted four sub-sections and one labeled "Table 1" reference. Four inbound cites surface as stale; one (REFERENCE.md:366) was already in the original [VERIFY] 5 inventory.

| Source | Line | Cite | Retarget to |
|---|---|---|---|
| `REFERENCE.md` | :366 | `NATCAP_ALIGNMENT.md Table 1` | NATCAP_ALIGNMENT.md §3 (per-metric fidelity table is now §3's content; "Table 1" label is gone) |
| `docs/internal/DATA_INVENTORY.md` | :460 | `Per-surface alignment status with NatCap recommendations. Six tables.` | Update to reflect new structure: badge taxonomy (§2) + per-metric fidelity (§3) + methodological divergences (§5) + export-bundle validation (§6) — 4 tables, not 6. Or remove the table count and just describe the doc. |
| `docs/archive/HISTORY.md` | :336 | `NATCAP_ALIGNMENT.md "SA UNA / biophysical extent" for the parity-claim implication` | The old `### SA UNA / biophysical extent — investigation (Brief A2, 2026-05-29)` sub-heading is gone. Retarget either to (a) `../research/una/` (the Brief-A2 single home per DESIGN_NOTES routing), or (b) CITY_PARITY's "SA biophysical extent vs ACS block-group polygons" callout (which now holds the IoU + pop-overlap numbers). |
| `docs/internal/STRATEGY.md` | :130 | same `NATCAP_ALIGNMENT.md "SA UNA / biophysical extent"` anchor | Same retarget as HISTORY:336. |

All inbound cites to ALIGNMENT §2 / §3 / §4 / §5 / §6 / §7 by number resolve correctly (the new structure preserves these numbered sections; the content differs but the headings exist). Anchor cite `"Validated reference outputs (SA)"` from CLAUDE.md:86 preserved as the §2 sub-heading at NATCAP_ALIGNMENT.md line 38.

DESIGN_NOTES inbound cites to ALIGNMENT (DESIGN_NOTES.md:11, :414, :516) all resolve correctly with the new structure.

### 3.5 CLAUDE.md trim — inbound-cite + orphan resolution (RESOLVED before push)

The CLAUDE.md trim (710 → 121 lines, 59,782 → 12,255 bytes) deleted the `## Blocked / pending work` section. Pre-push fixes applied (`<commit-tbd>`):

| Item | Status |
|---|---|
| `docs/internal/DESIGN_NOTES.md` Stratified Impervious Siting bullet cite to CLAUDE.md "Blocked / pending work" | **Resolved (option b — absorb)**: full proposal absorbed into DESIGN_NOTES §11.4 as a proper deferred-alternative entry; §11.5 → §11.6 renumbered; the broken-cite bullet in the tail "Topics not yet documented" list removed (the topic now has its own entry) |
| Track C1 frozen content | **Already homed**: STRATEGY.md §7 Track C row + §8. No action needed. |
| Heat Vulnerability Index pending | **Already homed**: DESIGN_NOTES §11.6 (Topics tail), ARCHITECTURE §11, REFERENCE §7, NATCAP_COLLABORATION methodology gaps, app.py:902 TODO + app.py:5845 UI tooltip. No action needed. |
| Minneapolis Full hidden, `load_data` parameterization | **Already homed**: HISTORY "Full Minneapolis extent" + "load_data parameterization (2026-05-09)". No action needed. |
| SA flood damage Path C + SA flood biophysical integrated | **Already homed**: DESIGN_NOTES §6.5 + CITY_PARITY + NATCAP_COLLABORATION Q12. No action needed. |
| SA-as-default-test-bed memory-fit | **Already homed**: HISTORY "Streamlit Cloud memory-fit workstream". No action needed. |

All four anchor-named cites from `app.py` (lines 1232, 1958, 2209, 2575) resolve correctly — `"Pure-variant helpers"`, `"Interface changes require auditing all consumers"`, `"Buildings — typing"` are all preserved as `###` headings in the trimmed CLAUDE.md.

**Residual sweep work:** STRATEGY.md §7 line 136 references *"C1 recorded as frozen in `../../CLAUDE.md`"* — that reference no longer resolves to a CLAUDE.md section. Retarget at sweep time to point at DESIGN_NOTES (or just rephrase to "recorded as frozen in STRATEGY.md §7 + §8" since the home is already self-referential).

## 4. Forward-looking — items expected to land here during the remaining rewrites

Items added as rewrites surface them. The standing categories:

- **Inbound anchor cites** that break when a rewrite renames anchors (the §1 pattern).
- **Cross-doc factual reconcile items** like §2.1 — claims in one doc contradicted by current code or by another doc's authoritative section.
- **Outbound cite tweaks** — e.g. if a doc's pointer goes to "see DESIGN_NOTES Brief B1" and Brief-B1 narrative now lives in HISTORY, the pointer text needs the doc rename.

When a cross-ref item is genuinely small and clearly scoped to *one* doc rewrite, it can fold into that commit instead of waiting for the sweep. Default is "queue for the sweep" — scattering increases risk of dropping items.

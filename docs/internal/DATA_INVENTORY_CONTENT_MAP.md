# DATA_INVENTORY.md refresh — content map

**Audience:** Internal
**Status:** In review — `[VERIFY]` 1–5 resolved against the live docs + code; awaiting approval before rewrite
**Use this for:** Driving the DATA_INVENTORY targeted dedup + status pass and the §15 four-way split
**Do not use this for:** Current data catalog — this doc is scaffolding for the refresh, not the catalog itself
**Source of truth for:** The dedup contract, the controlled-status column spec, and the §15 split routing

---

**Transient scaffolding.** Delete after the refresh + cross-ref-fix pass land. Same discipline as the other content-maps.

**Purpose:** turn DATA_INVENTORY into a boring, status-driven **file catalog**. The current doc is already ~70 % tabular — this is a *targeted dedup + status pass*, not a rebuild.

**Job of the doc:** what data exists, what's active / superseded / missing, and what uses it. NOT why it was chosen (→ DESIGN_NOTES), how aligned it is (→ CITY_PARITY / NATCAP_ALIGNMENT), what NatCap shared and when (→ COLLABORATION / HISTORY), or how metrics are computed (→ REFERENCE).

---

## Two structural decisions

1. **KEEP by-category** (§2 LULC / §3 soil / §4 buildings / §5 roads / §6 population / §7 tracts / §8 ET / §9 biophysical tables). It already answers "where are all the X files" and gives shared inputs (InVEST samples, raw OSM) a clean home. A by-city rebuild is high-churn, low-payoff; the wins below are orthogonal to it. **Do not reorganize by city.**
2. **ADD a controlled Status column** to every catalog table. The doc already *has* the status info, scattered in prose — promote it to a column.

**Status vocabulary (controlled):** `active` · `active_optional` · `derived` · `reference_only` · `superseded` · `retired` · `missing` · `external_not_committed`. Status call to get right: the SA per-scenario compound LULCs are **`missing`** (likely unsaved pipeline intermediates), not `external_not_committed`; the gitignored NatCap rasters + OSM zips are **`external_not_committed`**.

---

## The big dedup: §15 "Open questions" must dissolve

§15 is a parallel open-questions list that collides with the OPEN_QUESTIONS dashboard. Split four ways:

| Current §15 content | Destination |
|---|---|
| Resolved (Q1, Q4, Q5, Q6 — struck-through) | HISTORY (or delete; they're done) |
| Live blockers (SA-reproduction inputs; NLUD provenance if it gates work) | **Don't restate** — point to the OQ dashboard |
| Provenance-uncertainties (Q8 MN pop Census-vs-WorldPop; Q2 NatCap UNA demand value; Q7 NLUD source) | Fold into the relevant catalog entry's **Notes** as a `status=active` caveat — "what is this file / where's it from" is catalog territory |
| Acquisition tasks (Q10 Drive-folder triage) | COLLABORATION logbook |
| CRS-mismatch decision (Q9) | Resolved in practice (SA runs EPSG:5070) → state as a catalog fact, or → OQ if still a live decision |

**Missing-data summary** = the *catalog-side* view only: `absent file / substitute data used instead / status=missing / → OQ`. NOT a duplicate of OQ's "why it matters / what would resolve it."

---

## Cross-doc move-outs (single-home contract)

| Content | Out of DATA_INVENTORY → |
|---|---|
| §12 "Documentation files" index | **README** owns the doc-map. Delete §12, point to README. |
| Brief-chronology ("Brief 27 dual-raster…", "Brief 28b shifted baseline_hm 0.2866→0.3937", "Brief 9 retired", "Brief 33"…) | **HISTORY.** Keep current-state catalog facts; the when/why → HISTORY. |
| Methodology-of-use (§9.2 tree-canopy shade=0.66 dominance; §9.4 `_CARBON_IS_STOCK` behavior; §10 `HM_TO_FAHRENHEIT` derivation) | **REFERENCE / DESIGN_NOTES.** Catalog says what the file is + key + status + what reads it, then points. |
| MD5 / "byte-identical to InVEST sample" (§2 MN LULC) | **CITY_PARITY** owns parity claims. DATA_INVENTORY records source + points. |
| §13 "Status of data integrations" | Trim to catalog-status (active/dormant/adopted); shed Brief-chronology (→HISTORY) + parity overlap (→CITY_PARITY). |

**Boundary with CITY_PARITY:** DATA_INVENTORY = the *catalog* (path / type / CRS-grid / used-by / source / status). CITY_PARITY = *parity claims that reference the catalog* (this file MD5-matches NatCap's; this config aligns). The path may appear in both, but CITY_PARITY points to the catalog for the file's existence and adds the parity assertion on top — it does not re-catalog.

---

## Target column sets

- **Active inputs:** `Path | Type | CRS/grid | Used by | Source | Status | Notes`
- **Derived/precomputed:** `Derived file | Source files | Script | Rebuild command | Status` — the *rebuild command* makes it actionable.
- **Missing/wanted:** `Needed data | City | Substitute used | Status | → OQ`

Cleanup rule (verbatim from the plan): no long methodology. E.g. `land_use_compound_sa.tif` — active compound LULC for the SA stack; see CITY_PARITY for parameter alignment, DESIGN_NOTES for the compound-LULC decision. That's enough.

---

## Resolved findings (`[VERIFY]` 1–5)

### `[VERIFY] 1` — §15 per-item status against live code/docs

Read `docs/internal/DATA_INVENTORY.md` lines 518–540. **Six of the ten items resolved (four explicitly struck-through; two partially-resolved that the prose doesn't flag clearly).** Per-item routing:

| Q | Topic | Status | Route |
|---|---|---|---|
| Q1 | NLCD vintage of NatCap SA data | ~~✅ RESOLVED 2026-05-24~~ — explicitly struck | HISTORY (or delete; it's done) |
| Q2 | NatCap UNA demand parameter (SA) | **Open** — needs reading `urban_nature_demand.tif`'s single non-zero value | **Catalog Notes** on the SA UNA biophysical entry as `status=active` provenance caveat |
| Q3 | NatCap UNA search radius (SA) | **Partially resolved** — `kernel_800.0.tif` confirms 800 m; full args.json absent | **Catalog Notes** ("800 m confirmed via kernel intermediate; full args.json not shipped") |
| Q4 | NatCap UHI_MAX_C for SA | ~~✅ RESOLVED 2026-05-24 Brief 14~~ — explicitly struck | HISTORY |
| Q5 | `et0_annual_cgiar_3857.tif` resolution/extent | ~~✅ RESOLVED 2026-05-24~~ — explicitly struck | HISTORY |
| Q6 | SA buildings — typed? | ~~✅ Resolved~~ — explicitly resolved | HISTORY |
| Q7 | NLUD provenance | **Open** — provenance-uncertainty | **Catalog Notes** on NLUD-related entries (`status=active` caveat) |
| Q8 | MN downtown population source (Census vs WorldPop) | **Open** — provenance-uncertainty | **Catalog Notes** (per `[VERIFY] 3` below) |
| Q9 | CRS + grid mismatch (NatCap 3857 vs SA stack 5070) | **Resolved in practice** — SA stack runs EPSG:5070; NatCap data reprojected at prep time | **Catalog fact** in §2 SA LULC entry; NOT a live decision |
| Q10 | Other Drive-shared folders triage | **Open** — acquisition task | **COLLABORATION logbook** ("Active asks" or new "Data not yet triaged" sub-section) |

**§15 dissolves into 5 destinations:** HISTORY (4 resolved), catalog Notes (3 provenance-uncertainties + 1 catalog-fact for Q9), COLLABORATION logbook (1 acquisition task). No item is genuinely a "live blocker" that belongs in OPEN_QUESTIONS — the open ones are either provenance-uncertainties (catalog) or acquisition tasks (logbook).

### `[VERIFY] 2` — `data/sa/flood/lulc_nlcd_2021_sa.tif` duplicate

Cross-referenced DATA_INVENTORY §2 line 111 against CLAUDE.md:54:
- CLAUDE.md:54 documents the file as: *"Raw NLCD 2021 clipped to SA bbox via MRLC WCS (EPSG:5070, 30 m, 1984×1713 px)"* — the raw NLCD download.
- DATA_INVENTORY:111 calls it: *"A second copy ... (provenance unclear; sibling of the canonical NLCD file)."*

**Resolution:** This is the **raw NLCD download** preserved for provenance. The live SA stack consumes the dual-raster pipeline outputs (`land_use_compound_sa.tif` for compound-keyed models; the canonical NLCD-only raster for flood/CN). The raw clip is read once during pipeline setup, then never at runtime.

**Status: `reference_only`** with Notes: "Raw NLCD 2021 clip via MRLC WCS, preserved for provenance. Not read at runtime — the live SA flood path consumes the dual-raster pipeline outputs." The §2 prose at line 111 ("provenance unclear") gets deleted; the table row replaces it.

(NOT `superseded` — superseded implies a replacement file exists. The dual-raster outputs aren't replacements for the raw download; they're derived from it. NOT deletable either — it's the audit trail for the SA NLCD source.)

### `[VERIFY] 3` — MN pop Census-vs-WorldPop (Q8)

Read DATA_INVENTORY §6 lines 252–262: The caveat says *"the on-disk file's provenance cannot be determined from the file alone; if WorldPop was the source, totals would diverge from the Census reference."* Q8 (line 536) frames the resolution: *"Running the Census pipeline to re-verify would settle this."*

**Resolution: Fold into catalog Notes as `status=active` provenance caveat** — does NOT need a separate OPEN_QUESTIONS entry. The §6 MN-downtown catalog entry's Notes field reads:

> "Both `download_census_pop.py` (Census, canonical) and `clip_worldpop.py` (WorldPop, alternative) target the same output path. On-disk provenance is not encoded in the file. **Rebuild command** (resolves provenance): `python scripts/data/download_census_pop.py`."

The "Rebuild command" makes it actionable — anyone who cares can re-run the canonical pipeline and resolve it. Catalog Notes treatment + a Rebuild command is sufficient; this is provenance-uncertainty, not a blocker.

### `[VERIFY] 4` — Move-IN sources

| Source | Has catalog-shaped content? | Routing |
|---|---|---|
| `CLAUDE.md` (lines 40–48 data-path table + line 54) | **Yes** — 8 row-by-row file paths with purpose strings for MN downtown's input files, plus SA's `lulc_nlcd_2021_sa.tif`. Catalog territory. | **Move IN** to DATA_INVENTORY catalog entries; CLAUDE.md keeps only operational principles, points at DATA_INVENTORY for the file inventory. |
| `docs/archive/SA_INTEGRATION_PLAN_2026-05.md` | **Already archived** — confirmed at `docs/archive/SA_INTEGRATION_PLAN_2026-05.md` (one of the three archive renames from Commit 1 of the docs migration). The archived plan is historical, not a live data-facts source. | **Do NOT move IN.** Any current data facts that the archived plan referenced are already in the live catalog or DESIGN_NOTES; archive content stays archived. |

So the move-IN scope is **CLAUDE.md's data-path tables only**. After the move, CLAUDE.md row entries for `data/...` paths get deleted; CLAUDE.md's relevant section points readers at DATA_INVENTORY.

### `[VERIFY] 5` — Inbound refs to DATA_INVENTORY

5 hits — **none cite specific section anchors**, all are generic file-level references:

| File:line | Citation form |
|---|---|
| `README.md:61` | doc-map row, generic |
| `docs/research/ALPHAEARTH_FEASIBILITY.md:6` | status header cross-ref ("→ ../internal/DATA_INVENTORY.md"), generic |
| `docs/internal/ARCHITECTURE.md:55` | "authoritative source for the per-city data files", generic |
| `docs/internal/ARCHITECTURE.md:277` | "See DATA_INVENTORY.md", generic |
| `docs/internal/NATCAP_COLLABORATION.md:317` | "See `DATA_INVENTORY.md` for full file list", generic |

**Anchor-stability is NOT a constraint** for the rewrite — no inbound ref depends on a `§X` anchor name. The refresh can restructure sections freely.

---

## Mapping rows worth flagging

1. **§13 "Status of data integrations" overlaps heavily with CITY_PARITY's per-city summary tables AND the new Status column added per Decision 2.** Trim §13 hard: drop everything that's now redundant against per-row Status, drop everything that's a parity claim (→ CITY_PARITY), drop brief-chronology (→ HISTORY). What's left is at most a short "data-integration status snapshot" — and even that may be deletable if §1 (Top-level tree) gets a status summary header.
2. **§15 Q9 (CRS mismatch) is a catalog fact, not a live decision.** The SA stack runs EPSG:5070 by deliberate choice (equal-area for area-based math, per ARCHITECTURE.md §3 "CRS handling"); NatCap rasters get reprojected at prep time. The catalog should state this as a one-liner where the affected files appear (the SA LULC and population sections). Don't carry it forward as an unresolved question.
3. **The "Rebuild command" column** (Derived/precomputed target schema) needs a real example to show what good looks like — recommend `data/precomputed/<city>/nature_distance_<lucode>.npy` since those have an actual delete-and-regen story: *Rebuild command: delete the directory; the loader re-creates on next module load.* Without that example column the rewrite reads as documentation-shape rather than catalog-shape.
4. **Boundary with CITY_PARITY needs a concrete cross-ref pattern.** Per-city MD5-match claims today live in the catalog ("byte-identical to InVEST sample"). After the move-out, the cross-ref pattern is: catalog row's Notes says "Parity claim: see CITY_PARITY §3 Minneapolis UNA biophysical row." CITY_PARITY's row says "MD5-matches NatCap's MN bundle (see DATA_INVENTORY §9 for path)." Bidirectional, one assertion in each, no duplication.
5. **§15 has NO inbound external refs** (per `[VERIFY] 5`) — the rewrite can dissolve the section without anchor-preservation concerns.

---

## Resolved decisions (bake in — LOCKED before rewrite)

- **Keep by-category structure**; do NOT reorganize by city.
- **Controlled Status column** added to every catalog table — vocabulary: `active` / `active_optional` / `derived` / `reference_only` / `superseded` / `retired` / `missing` / `external_not_committed`.
- **§15 dissolves four ways** (per `[VERIFY] 1`): 4 resolved struck-through items → HISTORY; 3 provenance-uncertainties (Q2, Q7, Q8) → catalog Notes; 1 catalog-fact (Q9) → §2 SA LULC; 1 acquisition task (Q10) → COLLABORATION logbook. **No item routes to OPEN_QUESTIONS** — none is a genuine live blocker.
- **`lulc_nlcd_2021_sa.tif`** = `reference_only` (raw NLCD download preserved for provenance).
- **MN Census-vs-WorldPop (Q8)** = catalog Notes caveat + Rebuild command. NOT a dashboard item.
- **Move IN from CLAUDE.md** = the data-path table at lines 40–48 + the SA `lulc_nlcd_2021_sa.tif` row at line 54. After move, CLAUDE.md points at DATA_INVENTORY.
- **Move IN from SA_INTEGRATION_PLAN** = nothing (already archived; archive content stays archived).
- **No anchor-stability constraint** — 5 inbound refs are all generic file-level, no `§X` anchor cited.
- **§12 doc-index deleted** — README owns the doc-map.
- **Brief-chronology stripped** — HISTORY is the home; catalog says current-state only.
- **Methodology-of-use moved** to REFERENCE / DESIGN_NOTES; catalog says what + key + status + what reads it, then points.
- **MD5 / parity claims moved** to CITY_PARITY; catalog records the path + source, parity assertion lives next door with a bidirectional cross-ref.

---

## Sequencing — joins the back-half cluster

DATA_INVENTORY is a dedup *partner* of the docs being refreshed now, so it slots into the back half, not standalone:
- **after CITY_PARITY** (so parity claims have landed there to point to),
- **coordinated with OPEN_QUESTIONS** (the §15 split feeds OQ's external-data items; decide the catalog↔dashboard routing once, like the COLLABORATION↔OQ split),
- **after README** exists as the doc-map home (for the §12 deletion — but the README touch-ups are scheduled for the cross-ref sweep, so the existing README's doc-map is enough),
- **with Brief-chronology landing in HISTORY first** (same precondition as DESIGN_NOTES).

Slot in the back-half commit-ordering: NATCAP_ALIGNMENT → CITY_PARITY → COLLABORATION → HISTORY → **DATA_INVENTORY** → OPEN_QUESTIONS → cross-ref sweep + README. Its own commit. This matches the slot the OPEN_QUESTIONS map already locked in (the back-half map already places DATA_INVENTORY at step 5, after HISTORY and before OPEN_QUESTIONS).

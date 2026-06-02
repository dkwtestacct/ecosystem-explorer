# Open Questions

**Audience:** Internal
**Status:** Current dashboard
**Use this for:** The live state of every parked data request, methodology question awaiting external input, and planned feature in development
**Do not use this for:** Meeting prep (→ `DEMO_AND_COLLABORATION.md`), code/UI tasks (→ task tracker), per-decision rationale (→ `DESIGN_NOTES.md`), or the collaboration logbook (→ `NATCAP_COLLABORATION.md`)
**Source of truth for:** What still truly blocks progress, and what's planned

---

> **Dashboard, not logbook.** Each entry below uses the same four-field shape: **Status / Owner / Impact / Ask**. The full conversation framing for items imported from NatCap collaboration lives in `NATCAP_COLLABORATION.md §6` — this doc holds the *current state*, not the conversation history. Cross-refs are explicit (`see COLLABORATION §6 #N`).

---

## 1. Active NatCap data dependencies

External-data requests where NatCap has the missing piece. These unblock specific reproduction / parity / model-implementation work.

### 1.1 Per-scenario compound LULC inputs

- **Status:** Parked (draft email ready; not sent)
- **Owner:** NatCap (Chris Nootenboom — `cnootenb@umn.edu` — or the SA project team)
- **Impact:** Un-gates carbon / temperature reproduction for the six NatCap fixed alternative scenarios (FF_20ac, FF_40ac, FF_MAX, UA_20ac, UA_40ac, UA_MAX) — the only two `natcap_published` metrics in `data/sa/natcap_reference_outputs.csv`. Without these, those two metrics stay at "NatCap published" display, never "✓ NatCap match — independently reproduced".
- **Ask:** Either (a) the six per-scenario **compound** LULC overlays NatCap fed to Carbon + UCM; or (b) **Nootenboom's overlay script** that composes NLCD × NLUD × tree-canopy → compound LULC, so the prototype can rebuild the scenarios from the shared component layers.

#### Detail

A local content-signature hunt (2026-05-29) across `~/Desktop`, `~/Downloads`, `~/Documents`, `/Volumes`, the Google Drive sync root, and the `_zip_archive` zips found **only baseline compound rasters** — no scenario variants. A Google Drive **connector** search by name + `_3857` suffix corroborates: the per-scenario compound overlays are **not in the shared Drive** either. What IS shared is the flood-encoded scenarios (`sa_lc_w_*_10m.tif`), the **compound baseline** (`lulc_overlay_3857.tif`), and its component layers (`nlcd_3857.tif`, `nlud_3857.tif`, `tree_3857.tif`). Conclusion: NatCap built the per-scenario compound LULCs as **unsaved pipeline intermediates** — generated, fed to Carbon / UCM, then not persisted.

**Local reconstruction (Option 2) — possible, with a caveat.** We could rebuild the compound scenario overlays from `nlud_3857` + `tree_3857` + the flood scenario rasters. But the converted-pixel codes (`998` food forest / `999` garden) have no native compound class, so their NLUD × tree mapping would be **inferred**, not authoritative — a methodology assumption we'd have to own and document. Prefer (a) / (b) over reconstruction where possible.

**Impact if the inputs are never obtained.** Exactly one capability is foreclosed: the prototype cannot independently reproduce — and therefore cannot validate — its carbon and temperature outputs against NatCap's published per-scenario values for those six scenarios. Downstream consequences (same fact in three places):

- **B2** (per-scenario Match / Diverged validation badges): per-card "✓ NatCap match (Δ X %)" / "× Diverged" states cannot exist for carbon/temp on the fixed scenarios. B2 stays deferred. See `DESIGN_NOTES.md` §11.5.
- **B1 Phase 3** (verify prototype scenario carbon/temp deltas vs NatCap published): remains gated.
- **D1** (InVEST export): the six fixed alternatives export flood-only (UFR args + flood-encoded source raster), with carbon/UCM/UNA args marked "unavailable" in metadata. Baseline, Explorer-generated, and Optimizer-suggested scenarios are unaffected and export the full five-model bundle.

**Ceiling even if obtained.** Files only enable carbon + temperature validation (the only two `natcap_published` metrics) on those six scenarios. Nature access, cooling, MH have no NatCap published per-scenario target — they could not be validated against NatCap regardless.

**Not affected** (intact without the files):
- Per-pixel parity vs canonical InVEST — HMI MAE 0.0000 (Brief 28b), UMH MAE ≈ 0 (Brief B). The validation credibility anchor.
- All Explorer-generated + Optimizer-suggested scenarios + full five-model InVEST export (the prototype builds their compound rasters internally via `evaluate_scenario`).
- Displaying NatCap's published per-scenario carbon/temp as **reference values** (the 14 figures are in `natcap_reference_outputs.csv`).
- Flood metric on the fixed scenarios.

**What is NOT reproducible from disk** (Brief B2 revised investigation, 2026-05-29): NatCap's published *citywide absolute* baseline figures — `avg_temp_f` = 90.08 °F and `c_sequestration` = 107.32M t CO2e. Their SA UCM `args.json` isn't shipped, and `tot_c_cur.tif` doesn't aggregate to 107.32M by any standard interpretation. The reproduction claim sits at **per-pixel parity**, not citywide absolute.

#### Send-ready draft (verbatim — do not send without a go-ahead)

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
> The scenario rasters we already have (`sa_lc_w_*_10m.tif`) look like the flood / NDR inputs — they don't carry the land-use dimension the carbon / cooling tables key on, so we can't reproduce the carbon/temperature numbers from them. The shared Drive has the compound *baseline* overlay (`lulc_overlay_3857.tif`) and its `nlcd_3857` / `nlud_3857` / `tree_3857` components, but not the per-scenario compound overlays — it looks like those were pipeline intermediates that weren't saved.
>
> Either of these would unblock us:
> 1. The six **per-scenario compound overlays** (the NLCD × NLUD × tree LULCs fed to carbon + urban-cooling), if they can be regenerated or recovered — in whatever encoding/grid you ran them; we can reconcile CRS/resolution on our end.
> 2. **The overlay script** that composes the compound LULC from the NLCD / NLUD / tree-canopy layers (Chris Nootenboom may own this) — with it we can rebuild the scenario overlays ourselves from the shared component layers.
>
> A pointer to either is plenty — thanks!

### 1.2 Native NLCD × tree baseline flood raster

- **Status:** Open (lower priority — UX comparability, not validation)
- **Owner:** NatCap
- **Impact:** Resolves the ~5-point CN gap between the prototype's SA baseline (`mean_cn 76.54`) and NatCap's fixed scenarios (`mean_cn ≈ 81.4`). Without resolution, fixed-scenario flood cards risk reading as *lower* retention than baseline — a UI confusion, not a validation defect.
- **Ask:** NatCap's SA *baseline* LULC in the same native NLCD × tree 3-tier encoding as the scenario rasters.

**Local fallbacks (no external dependency required):**
1. **Reconcile the derivation** — compute the prototype's SA baseline flood through the same native NLCD × tree path the loaded scenarios use, so baseline and scenarios are on one footing.
2. **Suppress the fixed-scenario flood delta** — display the scenario's flood as "≈ invariant" (matching NatCap's documented "essentially no difference" finding) rather than a signed delta vs the prototype baseline. **Adopted in Brief B2-revised.**

Flood has **no NatCap published reference** (the reference CSV's flood rows are all `aligned_method`, no value), so this is a UX-comparability question, not a validation one. See `DESIGN_NOTES.md` §11.5 + the Brief B1 smoke-test trace.

### 1.3 SA NDR — DEM + watersheds

- **Status:** Open (never asked — paths suggest they live on a NatCap internal machine)
- **Owner:** NatCap (SA project team)
- **Impact:** Unblocks SA NDR (Nutrient Delivery Ratio) implementation — the missing 6th model from NatCap's six-model SA scope.
- **Ask:** `sa_dem_3m_proj.tif` and `San_Antonio_TX_buffer_mod.shp` (the NatCap SA README references these with `E:/GIS/` paths, suggesting they're on a NatCap internal machine, not shared).

See `NATCAP_COLLABORATION.md §6` Q5 for the conversation framing.

### 1.4 Per-crop SA food forest yield (CoSA)

- **Status:** Open (would resolve the SA Food Production prototype default)
- **Owner:** NatCap (SA Urban Agriculture project team)
- **Impact:** Replaces the SA `FOOD_FOREST_LBS_ACRE = 8,500` placeholder with per-crop NatCap-aligned numbers, lifting the Food Production card from `prototype` to `≈ NatCap method` for SA.
- **Ask:** Per-crop yield numbers from NatCap's SA Urban Agriculture project (`CoSA_Crop_production_ESModeling` referenced in the meeting note).

See `NATCAP_COLLABORATION.md §6` Q4a for the conversation framing.

### 1.5 MN Carbon four-pool bundle

- **Status:** Open (narrowed scope — methodology now clear)
- **Owner:** NatCap (or in-house — see Detail)
- **Impact:** Brings MN to parity with SA's post-Brief-30 four-pool stock framework — closes the one remaining methodology gap between MN and SA Carbon.
- **Ask:** Either (a) MN-specific four-pool table, or (b) guidance on whether to apply the Spawn et al. parameterization (NatCap's SA basis) to MN ourselves.

**Methodology now clear** (per `Notes on NASA Urban LULC overlay QA/QC`, paras 140–142): SA's four-pool comes from Spawn et al. data parameterized by NatCap (Lingling), refactored to omit embedded emissions / embedded storage / annual emissions for this project — matching the prototype's one-time-storage framing. So the path forward could be in-house if NatCap signs off on applying Spawn et al. to MN.

See `NATCAP_COLLABORATION.md §6` Q4b for the conversation framing.

---

## 2. Open methodology questions

External input wanted but not strictly blocking — methodology calls awaiting NatCap perspective.

### 2.1 MN sample data current vs SA-project framing

- **Status:** Open (no urgency — adoption already applied per per-city framing principle)
- **Owner:** NatCap
- **Impact:** Confirmation that the MN-project values Briefs 22 + 23 adopted (UNA demand 250, radius 1000, exponential decay; rainfall 100 mm) are still NatCap-current, not superseded by the SA-project framing that arrived later.
- **Ask:** Have the MN sample data values been superseded by the SA-project framing, or are both still considered project-canonical?

The MN UNA bundle is dated March 2026 (3 months before the SA README). Reversible if NatCap flags MN as superseded — the per-city `city_cfg` config makes a value swap a one-line change.

See `NATCAP_COLLABORATION.md §6` Q1.

### 2.2 Per-capita-only undersupply formulation for placement weighting

- **Status:** Open (empirical question)
- **Owner:** NatCap UNA team
- **Impact:** Could justify a placement-weighting refinement away from the current `max(0, demand − supply)` per-capita deficit formula, which Brief 9 found saturates (100 % SA, 67 % MN) too aggressively for moderate `pct_converted` values.
- **Ask:** Canonical framing for *reporting* (`urban_nature_balance_percapita`) may differ from canonical for *placement* — does NatCap have a recommended placement weighting?

See `NATCAP_COLLABORATION.md §6` Q6.

### 2.3 InVEST UNA edge handling at AOI boundary

- **Status:** Open (architectural)
- **Owner:** NatCap (Vibrant Land team)
- **Impact:** Resolves whether the prototype's zero-padded UNA convolution at AOI boundaries (matching `convolve_2d(ignore_nodata_and_edges=False)`) is the canonical choice or whether Vibrant Land buffered the AOI to avoid the edge bias. Particularly relevant for SA's post-Brief-31 ACS-block-groups extent (Mission Reach + Government Canyon sit just outside).
- **Ask:** Did Vibrant Land accept the edge bias, or buffer the AOI? If buffering, what radius?

See `NATCAP_COLLABORATION.md §6` Q11.

---

## 3. In development (planned features)

Planned work that's been scoped but not yet shipped. Not blockers — forward-looking.

### 3.1 Region Selection (Phase 1) — Shipped 2026-05

- **Status:** Shipped. Live sidebar Region Selection radio + region-layer dropdown + multiselect drive a `selected_region_mask: np.ndarray | None` parameter on `evaluate_scenario`. Composes with the SA Eligible-land ownership filter (mask = `selected_region_mask ∩ ownership_mask`). The Map View tab carries an interactive click-to-select polygon picker on top of the sidebar selector (originally listed as out-of-Phase-1; promoted in during build).
- **Polygon sources live:** SA council districts (primary) + Bexar census tracts (fallback); MN downtown census tracts.
- **Region-local metrics shipped beyond original Phase 1 scope:** a Selected-region impact table on the Scenario tab pairs region-clipped values with citywide for every decomposable metric (reconciliation contract: full-AOI region-local = citywide, machine-checked by `verify_baselines.py`). Two locked caveats render in-app: closed-form SCS-CN flood routing, and reach-model spillover (UCM ~600 m / UNA 800 m / UMH 300 m).
- **Optimizer integration:** the selected-area path runs surrogate-shortlist + engine-verify against the composed mask and emits `PROVENANCE_REGION_OPTIMIZED` ("Engine-verified — region-optimized"), distinct from the citywide `PROVENANCE_OPTIMIZER`. See `DESIGN_NOTES.md` §7.3 and `REGION_OPTIMIZER_SPEC.md`.
- **Still out of scope:** freehand polygons, parcel editing, region-specific lookup tables / region-aware surrogate (the engine integrates over the full raster — see §11 of `ARCHITECTURE.md`).

---

## 4. Deferred briefs

Decisions made; resume only if an upstream gate clears.

### 4.1 B2 — Per-metric Match / Diverged validation badges

- **Status:** Deferred (compound-input-gated)
- **Owner:** Resume when §1.1 (per-scenario compound LULC inputs) lifts
- **Impact:** Per-scenario "✓ NatCap match (Δ X %)" / "× Diverged" badge states for carbon + temp on the six NatCap fixed alternative scenarios — currently unreachable.
- **Ask:** Wait for §1.1 to resolve. The reworked surface will likely be smaller than the original (baseline reproduction + NatCap reference comparison table), not the per-card per-scenario badge design — see `DESIGN_NOTES.md` §11.5.

**Conservative-floor scope already shipped (Brief B2-revised, 2026-05-29):** four-state badge taxonomy as badges, fixed-scenario reference view (`_render_natcap_fixed_scenario_view`), cross-scenario comparison table, plain-line baseline-validation claim. The deferred piece is the Match/Diverged states alone.

---

## Maintenance

Add an entry when:
- A new external-data dependency surfaces (→ §1 or §2)
- A planned feature gets scoped and queued (→ §3)
- A brief defers pending an upstream gate (→ §4)

Update an entry's **Status** when:
- An ask gets sent (Parked → Open)
- A response arrives (Open → Resolved — strikethrough or move to `NATCAP_COLLABORATION.md` Closed/resolved)
- A planned feature ships (§3 → strikethrough; the WHATS_NEW + Underway discipline moves the visible side to `app.py`)
- A deferred brief resumes (§4 → reactivated and tracked in the task tracker)

Pair with `NATCAP_COLLABORATION.md` (logbook) and `CITY_PARITY.md` / `DATA_INVENTORY.md` (current state) updates when the same finding affects multiple docs.

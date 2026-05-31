# NatCap Collaboration

**Audience:** Internal — not shared with NatCap
**Status:** Current (living logbook)
**Use this for:** The collaboration record — what NatCap asked, implied, sent, or decided, over time
**Do not use this for:** Current alignment status (→ NATCAP_ALIGNMENT.md), per-city parameter values (→ CITY_PARITY.md), or the live blocker dashboard (→ OPEN_QUESTIONS.md)
**Source of truth for:** The history of the NatCap collaboration

---

**Naming:** Refer to all NatCap collaborators as "NatCap" — no individual names.

---

## Current collaboration summary

This file is the **logbook** — what was asked, what arrived, what was decided, when. The current-state companions:

- **NATCAP_ALIGNMENT.md** — model/metric fidelity + validation taxonomy (current state).
- **CITY_PARITY.md** — per-city parameter values + per-city data parity (current state).
- **OPEN_QUESTIONS.md** — live blocker dashboard (what's parked / pending right now).

This file records *how we got there.*

---

## Active asks

What NatCap has explicitly requested. Status as of 2026-05-29.

| Ask | When | Status | Notes |
|---|---|---|---|
| Adopt the curated SA dataset (NLCD + NLUD + tree-canopy compound LULC + matched biophysical tables) | Received 2026-05-23 | ✅ Done 2026-05-25 (Briefs 27/28b/29/30/31). LULC raster + crosswalk + UCM/UNA/Carbon biophysical tables + ACS block-group AOI all adopted; all three SA biophysical models compound-keyed. | Data folder in `data/sa/natcap_2024/`. README: "please update all input data to what is contained here." Multi-brief workstream tracked in `../archive/SA_INTEGRATION_PLAN_2026-05.md`. |
| Use InVEST canonical models where available | Throughout | ✅ Mostly done | UCM validated MAE = 0. UFR uses canonical CN-based runoff. UNA uses canonical 2SFCA. UMH uses canonical formula + buffer-mean NE kernel (MAE ≈ 0 post-Brief B). SA Carbon uses canonical four-pool stock framework (Brief 30); MN Carbon still uses the single-rate annual proxy (per-city framing — MN doesn't have four-pool NatCap data). |
| Implement InVEST Nutrient Delivery Ratio (NDR) model for SA | Meeting note (April 2026) lists NDR as part of NatCap's six-model SA scope | ⏸️ Not started | Inputs documented in `README_San_Antonio_InVEST_model_inputs.docx`: biophysical table, SA DEM at 3 m, runoff_proxy at 32 inches precipitation, watersheds shp. New model implementation, not a parameter tweak. SA DEM + watersheds pending NatCap (see OPEN_QUESTIONS). |
| Implement Urban Mental Health model | Earlier session | ✅ Done | InVEST UMH v3.19.0 integrated. Cards on dashboard. Validated against canonical execute() at MAE ≈ 0 post-Brief B. |
| Use canonical Heat Mitigation Index, not approximation | Earlier session | ✅ Done | `_compute_hmi_raster` validated against `natcap.invest.urban_cooling_model.execute()` at MAE = 0. |
| Separate placement-constraint inputs from model-input data | Earlier session | ✅ Done | Comprehensive OSM building mask added; UFR sample buildings still drive damage metrics. Split-config rationale in CITY_PARITY.md MN section. |

---

## Data NatCap has shared

Inventory of curated data delivered. File-level parity (MD5, paths) per city → CITY_PARITY.md.

| Item | Received | Status | Notes |
|---|---|---|---|
| SA NLCD + NLUD + tree LULC overlay + matched UCM/UNA/Carbon biophysical tables + pre-computed InVEST results | 2026-05-23 | ✅ Adopted | At `data/sa/natcap_2024/`. See `DATA_INVENTORY.md` for full file list and `CITY_PARITY.md` SA section for per-model parity. |
| `README_San_Antonio_InVEST_model_inputs.docx` | 2026-05-24 reviewed | ✅ Read | Documents NatCap's InVEST args for SA across UCM/Carbon/UNA/UFR/NDR. Settled four high-priority open questions (UNA demand, UNA search radius, UHI value, NLCD vintage). Triggered Brief 14 (SA UHI fix). |
| `Ecosystem_Explorer_-_Meeting_Note.docx` | 2026-05-24 reviewed | ✅ Read | Establishes project context: Natural Capital Symposium June 29 – July 1, 2026, Google AI for Science proposal, full six-model SA project scope (revealing NDR as a missing model in the prototype). |
| `Minneapolis/building footprints/` (Drive subfolder) | 2026-05-24 inspected | ⏸️ Reviewed, no download needed | Single ESRI shapefile bundle `gis_osm_buildings_a_fre_MN.*` — Geofabrik's unmodified OSM extract. Same source as the prototype's `download_osm_minneapolis.py`. No new data. |
| `Minneapolis/roads/` (Drive subfolder) | 2026-05-24 inspected | ⏸️ Reviewed, no download needed | Single ESRI shapefile bundle `gis_osm_roads_free_1_MN.*` — same Geofabrik source. No new data. |
| `Minneapolis/Urban model sample data same AOI Minneapolis/` (3 ZIPs of canonical InVEST sample data for MN AOI) | 2026-05-24 downloaded + inspected | ✅ Adopted | At `data/invest/mn_sample_data_natcap_2026/`. Three args.json files extracted and compared against prototype values. Confirmed MN UCM full alignment (uhi_max = 2.05, all distances match). Surfaced that the prototype's MN UNA was using SA-project values rather than MN-project values — a real per-city misalignment, resolved in Brief 22 (see Decisions below). |
| `Ben NDR and Flood Mar_2023.pptx` | 2026-05-29 located | ✅ Read | Slide 7 documents NatCap's SA design-storm-saturation framework for the flood model. Resolved Question 12 (see Closed/resolved below). |
| `Notes on NASA Urban LULC overlay QA/QC` (in the LULC and Parameters August 2024 bundle) | 2026-05-29 read | ✅ Read | Paragraphs 123–138 document the canopy-weighted parameter framework underlying the compound UCM / UNA / Carbon biophysical tables. Resolved the canopy-tier mapping question (in-house). Paras 140–142 documented SA Carbon four-pool methodology (Spawn et al. data parameterized by NatCap). |

---

## Decisions made because of NatCap input

What changed in the prototype because of something NatCap asked, sent, or modeled. Per-city parameter *values* live in **CITY_PARITY.md** (the single home); the entries below capture *why* a decision was made and *what NatCap context drove it*.

### Per-city parameter framing (the meta-decision, 2026-05-24)

NatCap parameters are **project-specific by design.** Different city projects use different parameter values, reflecting team-by-team and project-by-project policy framings. There is no single "NatCap canonical UNA demand" or "NatCap canonical UFR rainfall" — each project parameterizes for its own context. SA's project is keyed to a heat-wave-day scenario and a WHO-minimum-green-space demand; MN's project is keyed to a moderate-summer day and an aspirational green-space target. Both internally coherent for their own cities' analyses.

**Implication for the prototype's working principle.** "Align with NatCap canonical" works fine — the qualifier is **per-city**. MN-side parameters should match the MN project; SA-side parameters should match the SA project. The per-city UNA / UCM / UFR values live in **CITY_PARITY.md** under each city's `### UNA`, `### UCM`, and `### UFR` parameter tables. Rationale lives in DESIGN_NOTES §2.1.

**MN UNA misalignment resolved 2026-05-24 (Brief 22).** The prototype's MN UNA was using SA-project values (`demand = 16.7`, `radius = 800`, `decay = dichotomy`) rather than MN-project values (`demand = 250`, `radius = 1000`, `decay = exponential`). Brief 22 migrated the three UNA parameters to per-city `city_cfg` entries and implemented the exponential-decay kernel canonically. MN nature-access metrics dropped substantially (baseline ~43 % → 9.5 % under the 15× higher demand) — expected magnitude. SA-side UNA unchanged.

### NatCap research directions (the prototype's response)

NatCap collaborator notes (`data/sa/natcap_2024/Ecosystem_Explorer_-_Meeting_Note.docx`) identify three broad research directions; the prototype addresses each as follows.

| NatCap direction | What it means | Prototype response |
|---|---|---|
| **Multi-model integration** | Bring multiple urban InVEST models into a single decision-support context | ✅ Five InVEST urban models live (UCM, UFR, UNA, UMH, Carbon) for both cities. NDR pending — blocked on SA DEM + watersheds from NatCap. |
| **Spatially realistic scenario generation** | Encode where interventions can plausibly happen, not just how much area changes | ✅ Three-layer non-convertible mask (buildings + roads + existing nature); SA additionally preserves each pixel's (NLUD, tree-canopy) context through conversion via NatCap's compound crosswalk. See DESIGN_NOTES §4 + §5. |
| **Optimization and cost-effectiveness** | Surface tradeoffs and efficient scenarios, not just per-scenario metrics | ✅ Random Forest surrogate trained on a precomputed scenario grid drives Pareto search; cost-effectiveness ratios alongside biophysical metrics. ROOT considered as a reference point, deferred (DESIGN_NOTES §11.3). |

**Per-direction status on NatCap-identified workstreams:**

| Direction | NatCap mention | Status | Notes |
|---|---|---|---|
| OSM buildings + roads as placement constraints | "Simpler approaches" in NatCap document | ✅ Implemented | Roads + comprehensive OSM building footprints unioned into the non-convertible mask for every city (MN via the split-config `mask_buildings_file`). See DESIGN_NOTES §5. |
| Wallpaper approach | "Simpler approaches" in NatCap document | ⏸️ Interpretation unclear; to discuss with NatCap | Working interpretation in DESIGN_NOTES §11.2. |
| AlphaEarth satellite embeddings (NDVI replacement) | Identified as future direction | 🔵 Feasibility researched, not implemented | See `../research/ALPHAEARTH_FEASIBILITY.md`. |
| Land ownership layer | "Future consideration" in NatCap document | 🔵 Not pursued | Out of scope for current prototype. |
| San Antonio as full pilot (six-model SA project scope) | Active NatCap research direction | 🔄 In progress | SA NatCap data integration (Briefs 27–31) is complete for five of six models; NDR pending DEM + watersheds. The prototype is positioned as the SA pilot for NatCap's six-model framing. |
| Carbon Storage and Sequestration (deeper four-pool) | Listed as additional model for consideration | ✅ SA done (Brief 30, four-pool stock); MN remains single-rate proxy (no NatCap MN four-pool data) |
| Urban Mental Health model | Listed as additional model for consideration | ✅ Implemented | InVEST UMH v3.19.0; canonical buffer-mean kernel post-Brief B. |
| PLUS / CLUE / LCM land-use simulation | "Existing models" in NatCap document | 🔵 Considered, deferred — see DESIGN_NOTES §11.1 |
| ROOT (Restoration Opportunities Optimization Tool) | Mentioned in NatCap document | 🔵 Not pursued for in-app optimization — see DESIGN_NOTES §11.3 |

### Inferred priorities

What NatCap probably wants based on documents, project framing, and engagement style. Not explicitly asked but consistent with their direction.

- **The prototype should look and feel like an InVEST model run, not a separate methodology.** The "align with canonical" principle was inferred from NatCap's general posture about model fidelity. Per the per-city framing above, "canonical" is scoped per-city-project.
- **Per-capita supply/demand is the right framing for UNA.** Their canonical output is `urban_nature_balance_percapita.tif`. The aggregate-need framing (population × deficit) is not in their vocabulary.
- **The SA Urban Agriculture project is the primary SA use case.** The NatCap-curated SA data is keyed to this project; the food forest yield estimate is from it (8,500 lbs/acre placeholder pending project-report numbers).
- **Tree canopy matters more than NLCD class alone.** The compound NLCD + NLUD + tree LULC overlay treats tree canopy as a dominant per-pixel signal — any pixel with high canopy gets a shade boost regardless of NLCD class.
- **The prototype is positioned as an early example of the Google AI for Science proposal pitch.** The proposal describes an "AI-augmented InVEST platform" with agentic systems, dynamic data (AlphaEarth), multi-model integration, scenario generation, optimization. The prototype implements pieces of all of these.

### Per-decision log — choices made without confirmation

Choices made based on NatCap canonical output, not explicit NatCap confirmation. Recorded for later confirmation at the Natural Capital Symposium or future conversations.

| Decision | Date | Rationale | Confirmation path |
|---|---|---|---|
| SA UCM aligned to NatCap's heat-wave-day scenario (`uhi_max = 11`, `t_ref` documented but not used) | 2026-05-24 (Brief 14) | NatCap's SA README states these values explicitly; aligning per per-city framing. SA temperature deltas now ~3× larger than before. | NatCap doesn't need to confirm — already documented. |
| Prototype MN UNA migrated to MN-project canonical values | 2026-05-24 (Brief 22) | NatCap's MN sample data (March 2026, recent) documents these values explicitly. The per-city framing principle (Brief 14 for SA UCM) applies: align MN with the MN project, not blend with SA-project values. | NatCap confirmation at the symposium would close this; not blocking. Reversible if NatCap flags the MN-project values as superseded. |
| `DESIGN_STORM_INCHES` migrated from global (2.0″) to per-city | 2026-05-24 (Brief 23) | Same per-city framing as Brief 22. NatCap's MN args.json (100 mm) and SA README (157 mm) document the project-specific values. The prototype's previous 2-inch global was a "typical minor storm" plausibility default with no NatCap or InVEST source. | NatCap confirmation at the symposium would close this. Reversible. |
| SA LULC raster migrated from NLCD-only to NatCap compound (`land_use_compound_sa.tif`, reprojected EPSG:3857 → EPSG:5070 with nearest-neighbor at 30 m) | 2026-05-24 (Brief 27) | Foundational adoption of NatCap's curated SA dataset per `../archive/SA_INTEGRATION_PLAN_2026-05.md`. The reprojection choice (5070 over 3857) preserves area-based metric accuracy. 97.91 % pixel-wise agreement with prior `land_use_2021_sa.tif`; SA baselines drift < 0.5 % on every headline. MN untouched. | Three NatCap-facing questions surfaced in the integration plan: EPSG:5070 over EPSG:3857 acceptable? conversion-mapping rule (preserve NLUD + tree-canopy) consistent with intent? `code` column encoding — is `lucode` the only intended join key? |
| SA UCM biophysical table swapped from prototype's Köppen-BSh-tuned per-NLCD table to NatCap's compound `ucm__nlcd_nlud_tree.csv` | 2026-05-24 (Brief 28b) | The shift uncovered a meaningful finding: `baseline_hm` jumped 0.2866 → 0.3937 (+37 %) because the compound table captures tree-canopy variation on developed land that per-NLCD couldn't. `cooling_energy_savings_usd` dropped 77–86 % as a downstream amplification. The Köppen tuning was overstating cooling leverage by understating baseline canopy. MN UCM untouched. | **Worth flagging at the symposium:** per-NLCD biophysical tuning may systematically overstate cooling-intervention dollar leverage in cities with non-trivial existing tree canopy on developed land. The compound framework's tree-canopy bin resolves this. |
| SA UNA biophysical table swapped to NatCap's compound `una__nlcd_nlud_tree.csv` | 2026-05-24 (Brief 29) | SA baseline `nature_access_pct` shifted 89.7 → 94.2 (+5.0 %, +4.5 pp). Same direction as Brief 28b's UCM finding (per-NLCD biases against existing canopy). The compound framework's NLUD + tree-canopy bins resolve this; the borrowed-from-MN per-NLCD table had treated all developed-class pixels at `urban_nature = 0`. MN UNA untouched. | **Worth flagging alongside Brief 28b's UCM finding:** per-NLCD `urban_nature` scoring systematically understates baseline accessibility in cities with non-trivial tree canopy or natural-managed NLUD context on developed land. |
| SA Carbon model swapped to NatCap's canonical InVEST four-pool stock framework via the compound `carbon__nlcd_nlud_tree.csv` | 2026-05-25 (Brief 30) | The methodology decision aligns *directly* with NatCap's own published SA work — the 2023 "Vibrant Land" report (Guerry et al.) uses identical InVEST Carbon model methodology (four-pool stock × SC-CO2 with no NPV) for the same SA AOI. **Methodology matches; SC-CO2 vintage intentionally differs** — the prototype's `EPA_SOCIAL_COST_CARBON = $190/t @ 2 % discount` is EPA 2023 final rule (more current vintage of the same US-government standard); Vibrant Land used IWG 2021 ($53/t @ 3 %). SA dollar values run ~3.6× Vibrant Land's reported figures on equivalent stock magnitudes — methodology aligned, vintage differs. | **Confirmatory question for NatCap:** are you planning to update Vibrant Land's reported figures to EPA 2023 SC-CO2 ($190/t), or do you intend to keep the IWG 2021 framing? The prototype's constant can move either way. Not blocking. |
| Use per-capita supply deficit (no population multiplier) for undersupply-focused placement | 2026-05-23 (Brief 9) | Matches InVEST UNA's `urban_nature_balance_percapita.tif` framing. Aggregate-need form was a homegrown proxy. | Could surface to NatCap with empirical findings — Brief 9 saturation (100 % SA, 67 % MN) suggests the canonical framing may not be usable as-is for placement on county-scale AOIs. |
| Rename `equity-focused` → `undersupply-focused` | 2026-05-23 (Brief 9) | InVEST UNA reserves "equity" for demographic-group stratification. | Vocabulary change; no expected NatCap pushback. |
| Use per-pixel runoff Q from SCS-CN equation for flood-focused, not raw CN | 2026-05-23 (Brief 9) | Q is canonical UFR output. | Routine alignment. |
| Rename "Cooling Capacity / CC" → "Heat Mitigation Index / HMI" in UI | 2026-05-23 (Brief 8) | Reported value was already canonical HMI; label was stale. | Vocabulary cleanup. |
| Default to gitignoring NatCap-curated SA rasters; commit only small CSVs/docs | 2026-05-24 | Avoid large files in git; data is reproducible from NatCap's source. | Pragmatic. |
| Keep "Balanced" placement strategy as app-specific heuristic, no InVEST analog | Throughout | No InVEST model prescribes balanced placement; ROOT does weighted-sum LP. | Documented in REFERENCE with pointer to ROOT. |
| Surrogate optimizer is app-specific (random-forest); not ROOT | Throughout | Different optimization framework than ROOT's LP. | Documented with pointer to ROOT. |

### Resolved in-house (not asked of NatCap)

Items answered ourselves rather than asking NatCap — listed here for follow-through tracking.

- **A. EPSG:5070 reprojection of compound LULC raster** — methodologically standard for CONUS area-based metrics; no NatCap confirmation needed. Rationale in DESIGN_NOTES §3.3.
- **B. SC-CO2 vintage divergence (EPA 2023 $190/t vs. Vibrant Land's IWG 2021 $53/t)** — EPA 2023 final rule is the more current US-government standard; decision already made. Documented in DESIGN_NOTES §6.4. May surface at symposium as FYI, not as a question.
- **C. Conversion-mapping rule for `COMPOUND_AFTER_*` arrays (preserve NLUD + tree-canopy when changing NLCD)** — already empirically validated by Brief 30's investigate-first (HD < FF < GI four-pool carbon ordering is land-cover-plausible). No NatCap blessing needed.
- **D. `code` column encoding (is `lucode` truly the only intended join key?)** — trivially verifiable with `grep`; `lucode` is the only join key consumed by `load_data` routines, implicitly confirmed by Briefs 28b/29/30 each successfully loading their compound biophysical tables.

---

## Gaps surfaced

Where the prototype currently diverges from NatCap asks or inferred priorities. Each gap has a reason. **Live blockers (parked, pending) live in OPEN_QUESTIONS.md; this section is the historical logbook record.**

### Parameter divergences (resolved)

The prototype's MN UNA and MN UFR parameters had been matching the NatCap **SA project** rather than the NatCap **MN project**. Resolved 2026-05-24 (Briefs 22 + 23) per the per-city framing principle.

| Gap | Prototype | NatCap MN project | Reason / status |
|---|---|---|---|
| ~~MN UNA `urban_nature_demand_per_capita`~~ | ~~16.7 m²/capita (SA-project value)~~ | ~~250 m²/capita~~ | ✅ Resolved 2026-05-24 (Brief 22). |
| ~~MN UNA `search_radius`~~ | ~~800 m (SA-project value)~~ | ~~1000 m~~ | ✅ Resolved 2026-05-24 (Brief 22). |
| ~~MN UNA `decay_function`~~ | ~~dichotomy (SA-project value)~~ | ~~exponential~~ | ✅ Resolved 2026-05-24 (Brief 22). |
| ~~MN UFR `rainfall_depth`~~ | ~~50.8 mm (2 inches)~~ | ~~100 mm~~ | ✅ Resolved 2026-05-24 (Brief 23). |

### Methodology gaps (acknowledged, documented)

| Gap | Reason | Status |
|---|---|---|
| MN Carbon is single-rate proxy, not four-pool InVEST | No NatCap MN four-pool data; per-city framing keeps each city on its own published methodology | Open; on roadmap pending MN four-pool data |
| Cooling Energy Savings uses per-pixel aggregation, not per-building T_air sampling | Methodology gap acknowledged in REFERENCE | Documented divergence |
| Food Forest yield uses single per-city benchmark; NatCap uses InVEST Crop Production with per-crop parameterization (`CoSA_Crop_production_ESModeling`) | Different methodology framework; per-crop data not yet obtained | Open — would require CoSA model integration |
| Flood mitigation methodology divergence: NatCap pre-computes UFR over two alternative LULCs (20-acre + 40-acre food-forest scenarios at 10 m); prototype runs UFR live per slider position | Different workflow framework. Both defensible. | Documented divergence |
| Flood Damage Avoided produces dollar values; InVEST UFR's `serv_blt` is officially an indicator only | Documented in REFERENCE tooltip | Documented divergence |
| No formal Heat Vulnerability Index (CDC / ATSDR HVI) — using NLCD-intensity proxy | Lower-priority methodology improvement | Open; on roadmap |
| No Annual NLCD migration (prototype stays on legacy 21-class) | InVEST sample data + biophysical tables are calibrated to legacy. NatCap's curated SA data confirmed using legacy NLCD 2021 (Brief 12). | Open question for NatCap. |

### Data gaps (NatCap also has them)

| Gap | Reason | Status |
|---|---|---|
| SA NDR model not implemented | Outside original prototype scope; SA DEM + watersheds not in shared Drive | Open — see Active asks + OPEN_QUESTIONS |
| SA Flood Damage Avoided degrades to $0 — no per-building damage rates | NatCap also leaves the damage loss table blank in their SA setup (per the README) | **Resolved (Brief 33, Path C)** for Flood Damage: SA dashboard now renders "Flood Volume Reduction" as percent volume reduction, matching NatCap's Vibrant Land (Guerry et al. 2023) reporting. Reversible if future NatCap conversation surfaces SA-specific damage values (Path A). See DESIGN_NOTES §6.5. |
| Cooling Energy Savings (SA) still uses the OSM building footprint with partial type coverage | Untyped OSM polygons can't carry the per-typed-building dollar metric | Documented caveat surfaced on the card when `BUILDINGS_TYPE_COVERAGE < 0.95`. |
| SA UCM weights (shade = 0.6, albedo = 0.2, et = 0.2) verified to match | NatCap README specifies; prototype matches | ✅ Closed |
| SA UCM `air_blending_distance = 600`, `maximum_cooling_distance = 450` verified to match | NatCap README specifies; prototype matches | ✅ Closed |

---

## Open questions to raise with NatCap

Live questions to raise at the symposium or future conversations. Concise — fuller context in OPEN_QUESTIONS.md (live blocker dashboard) or DESIGN_NOTES.md.

### Highest priority — per-city alignment

1. **Are the MN sample data values still current, or have they been superseded by the SA-project framing?** The MN UNA bundle is dated March 2026 (3 months ago); the SA README is from later. Briefs 22 + 23 adopted the MN-project values pending this confirmation; reversible if NatCap flags MN as superseded.

### High priority — operational

4. **Data ask: per-crop CoSA + MN Carbon four-pool bundle.** Two pieces of the same kind of ask.
   - **#4a — Per-crop SA food forest yield.** Currently using 8,500 lbs/acre placeholder. NatCap's SA Urban Agriculture project should have per-crop numbers.
   - **#4b — MN Carbon four-pool bundle.** Would bring MN to parity with SA's post-Brief-30 framing. Methodology now clear (SA's four-pool comes from Spawn et al. data parameterized by NatCap). What's outstanding is narrowed: either the MN-specific four-pool table, or guidance on whether to apply the Spawn et al. parameterization to MN ourselves.

5. **For SA NDR integration: are watershed and DEM files in the shared Drive folder somewhere?** The README references `sa_dem_3m_proj.tif` and `San_Antonio_TX_buffer_mod.shp` with `E:/GIS/` paths suggesting they're on a NatCap internal machine, not shared.

6. **Is the per-capita-only undersupply formulation right for placement weighting?** Brief 9's saturation finding (100 % on SA, 67 % on MN) shows that strict per-capita deficit concentrates too aggressively to be usable at moderate pct values. Canonical framing for *reporting* may differ from canonical for *placement*.

11. **InVEST UNA edge handling at AOI boundary.** `_una_convolve` matches InVEST UNA's `convolve_2d(ignore_nodata_and_edges=False)` — edges are zero-padded, not edge-corrected. UCM does edge-correct. Residents near the AOI boundary have under-counted nature access because off-AOI green space is treated as absent. Particularly relevant for SA's post-Brief-31 ACS-block-groups extent, where Mission Reach and Government Canyon sit just outside the AOI. Did Vibrant Land accept this edge bias, or buffer the AOI?

### Medium priority

7. **For mixed-allocation scenarios (gi = 50 / ff = 50 / hd = 0), does anyone in the NatCap ecosystem measure placement-strategy effects?** Diagnostic only measured single-cover.

### Low priority

8. **Should the prototype migrate to Annual NLCD once NatCap's own data does?** No urgency unless they signal a migration.

9. **What's the right way to validate a placement strategy from NatCap's perspective?** The three-layer diagnostic (variance / selectivity / outcome delta) might not be how they think about it.

10. **Building damage rates per-city?** SA has no per-building type codes; downstream metrics degrade to $0. Also a NatCap gap, not just a prototype gap.

### To share (finding, not a question)

- **Per-NLCD biophysical tuning systematically biases against existing canopy on developed land.** Brief 28b's UCM swap shifted SA `baseline_hm` 0.2866 → 0.3937 (+37 %); Brief 29's UNA swap shifted baseline `nature_access_pct` 89.7 → 94.2 (+5 %). The compound table credits per-pixel tree-canopy variation the per-NLCD framework couldn't represent. Potentially useful signal for any other prototype team migrating per-NLCD → compound.

---

## Closed / resolved

Items that were open at session start and have since been resolved.

### Resolved through NatCap delivered data (2026-05-24)

- ~~NatCap NLCD vintage~~ → Legacy NLCD 2021 (per `gdalinfo -hist`, Brief 12)
- ~~NatCap UNA demand for SA~~ → 16.7 m²/capita (per SA README)
- ~~NatCap UNA search radius for SA~~ → 800 m uniform (per SA README)
- ~~NatCap UCM `uhi_max` for SA~~ → 11 °C (per SA README, applied in Brief 14)
- ~~MN UNA: switch from SA-project values to MN-project canonical?~~ → ✅ Done in Brief 22 (demand 16.7 → 250, radius 800 → 1000, decay dichotomy → exponential).
- ~~UFR rainfall depth — should the prototype use per-city values?~~ → ✅ Done in Brief 23 (MN 3.94″ / 100 mm; SA 6.18″ / 157 mm; migrated from global `DESIGN_STORM_INCHES = 2.0`).
- ~~"Wallpaper approach" meaning~~ → Working interpretation: uniform tiling of conversions; equivalent to prototype's `random` placement strategy
- ~~NatCap UCM weights for SA~~ → shade = 0.6, albedo = 0.2, et = 0.2 (per SA README; prototype matches)
- ~~NatCap UCM `uhi_max` for MN~~ → 2.05 °C (per MN args.json; prototype matches)
- ~~NatCap UCM blending and cooling distances for MN + SA~~ → 600 m / 450 m (consistent across both, prototype matches)

### Question 12 — SA flood Curve Number table: systematic anomaly vs NRCS TR-55

**Resolved 2026-05-29 via `Ben NDR and Flood Mar_2023.pptx`.** NatCap's CN values for SA reflect a documented **design-storm-saturation framework**: under the 24-hour 100-year storm on SA's clay-rich D-soils, soil infiltration capacity is exceeded across most vegetated surfaces, so even wetlands and forests rank as runoff-generating. Slide 7 of the pptx explicitly addresses the finding ("essentially no difference between garden, food forest, park, or vacant vegetated space"). NatCap's own modeled food-forest scenarios show +0.1 % to +1.1 % increase in flood volume vs baseline — matching the prototype's behavior when wired to the staged biophysical table. The integration was correct; the conservative deferral was reversed.

Full investigation + per-class CN comparison vs NRCS TR-55 + framework explanation → `../archive/HISTORY.md` "Completed-workstream specifics" → SA flood-CN investigation (Q12, resolved 2026-05-29).

### Canopy-tier mapping for flood CN table — resolved in-house (2026-05-29)

`Notes on NASA Urban LULC overlay QA/QC` (paragraphs 123–138) documents that NatCap's UCM / UNA / Carbon parameter framework uses **continuous weighted averaging** of base × tree-canopy parameters based on fractional canopy cover — not a 3-tier discretization. The flood CN table is the only NatCap table that bins into discrete tiers (211 / 212 / 213), likely because CN values are non-linear and cannot be meaningfully averaged. NatCap does not have a 4 → 3 canopy mapping to recover; the prototype's `tier = max(tree, 1)` (None + Low → tier 1, Medium → tier 2, High → tier 3) is a documented prototype-side discretization driven by the flood CN table's structure, not a NatCap methodology gap. The conservative wet-side framing (None and Low both treated as lowest canopy) is retained.

### UCM / UNA / Carbon canopy-weighted parameter framework — methodology note (2026-05-29)

The compound NLCD × NLUD × tree-canopy biophysical tables for UCM, UNA, and Carbon are not 4-tier discretizations but rather materializations of a *continuous weighted-averaging* methodology that NatCap documents in `Notes on NASA Urban LULC overlay QA/QC`:

- **UCM Shade:** "weighted average (base parameter, tree parameter) based on the amount of tree canopy. No impact of zoning." (para 126)
- **UCM kC:** "weighted average … Then increased by up to 10 % based on expected irrigation use (as identified by NLUD)." (paras 128–129)
- **UCM Albedo:** same framework as kC (paras 131–132)
- **UCM Green Area:** "any areas with greater than 50 % tree canopy cover (i.e. 'High') were considered Green Areas no matter their underlying NLCD/NLUD classification." (paras 134–135)
- **Carbon (all pools):** "weighted average (base parameter, tree parameter) based on the amount of tree canopy" (paras 137–138)

The prototype consumes the compound table directly (`lucode 0–1983`), which already encodes the weighted-averaging result per realistic combination. No prototype-side weighted-averaging logic is needed. The flood CN table is the exception that uses 3-tier discrete binning (see Question 12).

---

## Meeting notes & dated comms

**Natural Capital Symposium 2026:** June 29 – July 1, 2026. User is attending but **not presenting**. Prototype will be visible/discussed informally with NatCap collaborators.

**Implication for pacing:** No formal deliverable deadline, but having the prototype in good shape — known divergences documented, NatCap alignment current, recent integrations stable — is the working target for the symposium window.

**Google AI for Science proposal:** Active funding pitch (link in `Ecosystem_Explorer_-_Meeting_Note.docx`). Describes an "AI-augmented InVEST platform" with agentic systems, dynamic data, multi-model integration. The prototype is positioned as an early example of this vision. Status of the proposal itself: unknown to user; track via NatCap conversations.

---

## Maintenance

This doc gets updated when:

- NatCap explicitly asks for something new (add to Active asks)
- Daniel infers a new priority (add to Inferred priorities under Decisions)
- A new gap is identified (add to Gaps surfaced)
- A decision is made without checking with NatCap first (add to Decisions per-decision log)
- A new question to raise comes up (add to Open questions to raise)
- NatCap shares new data (add to Data NatCap has shared)
- An ask gets delivered, a gap closes, a question gets answered (update status, don't delete history — strikethrough or move to Closed/resolved)

Pair with `NATCAP_ALIGNMENT.md` and `CITY_PARITY.md` updates. Same discipline as `WHATS_NEW`.

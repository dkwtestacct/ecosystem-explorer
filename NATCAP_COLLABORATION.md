# NatCap Collaboration

**Purpose:** Track what NatCap is asking for, what they probably want (read between the lines), and where the prototype's current state diverges from either.

**Audience:** Daniel and future Claude sessions. Not shared with NatCap.

**Naming:** Refer to all NatCap collaborators as "NatCap" — no individual names.

**Relationship to other docs:**

- `NATCAP_ALIGNMENT.md` — per-surface alignment status (six tables: methodology, parameters, AOI, research directions, vocabulary). The *result*.
- This doc — the *process*. Asks, inferences, gaps, decisions made without confirmation, open questions.

---

## Per-city parameter framing (2026-05-24)

NatCap parameters are **project-specific by design.** Different city projects use different parameter values, reflecting team-by-team and project-by-project policy framings. There is no single "NatCap canonical UNA demand" or "NatCap canonical UFR rainfall" — each project parameterizes for its own context. Comparing the MN InVEST sample data bundles (received 2026-05-24) with the SA README (received 2026-05-23) makes this concrete:

| Parameter | NatCap MN project | NatCap SA project |
|---|---|---|
| UNA `urban_nature_demand_per_capita` | 250 m²/capita | 16.7 m²/capita |
| UNA `search_radius` | 1000 m | 800 m |
| UNA `decay_function` | exponential | dichotomy |
| UFR `rainfall_depth` | 100 mm | 157 mm |
| UCM `uhi_max` | 2.05 °C | 11 °C |
| UCM `t_ref` | 23.2 °C | 35 °C |

These reflect different framings, not drift between similar values. SA's project is keyed to a heat-wave-day scenario and a WHO-minimum-green-space demand; MN's project is keyed to a moderate-summer day and what appears to be an aspirational green-space target. Both internally coherent for their own cities' analyses.

**Implication for the prototype's working principle.** "Align with NatCap canonical" works fine — the qualifier is **per-city**. MN-side parameters should match the MN project; SA-side parameters should match the SA project. Brief 14 followed this for SA (adopted `uhi_max=11`); the prototype's MN UCM already matches the MN project.

**MN UNA misalignment resolved 2026-05-24 (Brief 22).** The prototype's MN UNA was using SA-project values (`demand=16.7`, `radius=800`, `decay=dichotomy`) rather than MN-project values (`demand=250`, `radius=1000`, `decay=exponential`). Brief 22 migrated the three UNA parameters to per-city `city_cfg` entries and implemented the exponential-decay kernel canonically (matching `natcap.invest.urban_nature_access`'s call to `pygeoprocessing.kernels.exponential_decay_kernel` with `max_distance = ceil(search_radius_in_pixels) * 2 + 1`, `expected_distance = search_radius_in_pixels`). MN nature-access metrics dropped substantially (baseline ~43% → 9.5% under the 15× higher demand) — expected magnitude. SA-side UNA unchanged. See Gaps "Closed" subsection.

---

## Active asks

What NatCap has explicitly requested. Status as of 2026-05-24.

| Ask | When | Status | Notes |
|---|---|---|---|
| Adopt the curated SA dataset (NLCD + NLUD + tree-canopy compound LULC + matched biophysical tables) | Received 2026-05-23 | 🔄 Integration queued | Data folder in `data/sa/natcap_2024/`. README: "please update all input data to what is contained here." Multi-brief workstream (Brief 16+ after current alignment work). |
| Use InVEST canonical models where available | Throughout | ✅ Mostly done | UCM validated MAE=0. UFR uses canonical CN-based runoff. UNA uses canonical 2SFCA. UMH uses canonical formula. Carbon is single-rate proxy — pending. |
| Implement InVEST Nutrient Delivery Ratio (NDR) model for SA | Meeting note (April 2026) lists NDR as part of NatCap's six-model SA scope | ⏸️ Not started | Inputs documented in `README_San_Antonio_InVEST_model_inputs.docx`: biophysical table, SA DEM at 3 m, runoff_proxy at 32 inches precipitation, watersheds shp. New model implementation, not a parameter tweak. |
| Implement Urban Mental Health model | Earlier session | ✅ Done | InVEST UMH v3.19.0 integrated. Cards on dashboard. |
| Use canonical Heat Mitigation Index, not approximation | Earlier session | ✅ Done | `_compute_hmi_raster` validated against `natcap.invest.urban_cooling_model.execute()` at MAE=0. |
| Separate placement-constraint inputs from model-input data | Earlier session | ✅ Done | Comprehensive OSM building mask added; UFR sample buildings still drive damage metrics. |

---

## Inferred priorities

What NatCap probably wants based on documents, project framing, and how they've engaged. Not explicitly asked but consistent with their direction.

- **The prototype should look and feel like an InVEST model run, not a separate methodology.** The "align with canonical" principle Daniel adopted (2026-05-23) was inferred from NatCap's general posture about model fidelity. Per the "Per-city parameter framing" section above, "canonical" is scoped per-city-project: align MN-side with the MN project's parameters and SA-side with the SA project's parameters, rather than picking one set and applying it across both cities.
- **Per-capita supply/demand is the right framing for UNA.** Their canonical output is `urban_nature_balance_percapita.tif`. The aggregate-need framing (population × deficit) is not in their vocabulary.
- **The SA Urban Agriculture project is the primary SA use case.** The NatCap-curated SA data is keyed to this project; the food forest yield estimate is from it (8,500 lbs/acre placeholder pending project-report numbers).
- **Tree canopy matters more than NLCD class alone.** The compound NLCD+NLUD+tree LULC overlay treats tree canopy as the dominant signal (any pixel with high canopy gets shade=0.66 regardless of NLCD class).
- **The prototype is positioned as an early example of the Google AI for Science proposal pitch.** The proposal (referenced in meeting note) describes an "AI-augmented InVEST platform" with agentic systems, dynamic data (AlphaEarth), multi-model integration, scenario generation, optimization. The prototype implements pieces of all of these. AlphaEarth integration remains research-only (ALPHAEARTH_FEASIBILITY.md).
- **ROOT exists but is not being pursued for this prototype.** Mentioned in meeting note as Deborah's planned future investigation. The prototype's surrogate-based optimizer is acknowledged as a different (simpler) approach.

---

## Gaps

Where the prototype currently diverges from NatCap asks or inferred priorities. Each gap has a reason.

### Parameter divergences (prototype uses SA-project values for MN)

The prototype's MN UNA and MN UFR parameters happen to match the NatCap **SA project**, not the NatCap **MN project**. This is a real misalignment — the right move is per-city: match MN-project values for MN, SA-project values for SA.

| Gap | Prototype | NatCap MN project | Reason / status |
|---|---|---|---|
| ~~MN UNA `urban_nature_demand_per_capita`~~ | ~~16.7 m²/capita (SA-project value)~~ | ~~250 m²/capita~~ | ✅ Resolved 2026-05-24 (Brief 22). Switched to 250 m²/capita per MN-project args.json. |
| ~~MN UNA `search_radius`~~ | ~~800 m (SA-project value)~~ | ~~1000 m~~ | ✅ Resolved 2026-05-24 (Brief 22). Switched to 1000 m per MN-project args.json. |
| ~~MN UNA `decay_function`~~ | ~~dichotomy (SA-project value)~~ | ~~exponential~~ | ✅ Resolved 2026-05-24 (Brief 22). Switched to exponential per MN-project args.json; canonical InVEST exponential-decay kernel implemented (`exp(-d / expected_distance)`, max_distance = `ceil(radius_px) * 2 + 1`). |
| MN UFR `rainfall_depth` | 50.8 mm (2 inches) | 100 mm | Prototype's 2-inch design storm is its own choice, not the SA project's 157 mm either. So this isn't a "wrong city" issue — it's a prototype-specific divergence from both NatCap projects. Worth raising with NatCap which they'd consider appropriate. |

### Methodology gaps (acknowledged, not fixable)

| Gap | Reason | Status |
|---|---|---|
| Carbon is single-rate proxy, not four-pool InVEST | Methodology upgrade | Open; on roadmap |
| Cooling Energy Savings uses per-pixel aggregation, not per-building T_air sampling | Methodology gap acknowledged in REFERENCE.md | Documented divergence |
| Food Forest yield uses single per-city benchmark; NatCap uses InVEST Crop Production with per-crop parameterization (`CoSA_Crop_production_ESModeling`) | Different methodology framework; per-crop data not yet obtained | Open — would require CoSA model integration |
| Flood mitigation methodology divergence: NatCap pre-computes UFR over two alternative LULCs (20-acre and 40-acre food-forest expansion scenarios at 10 m resolution); prototype runs UFR live per slider position | Different workflow framework. Both defensible. | Open as methodology divergence; not a "fix" |
| Flood Damage Avoided produces dollar values; InVEST UFR's `serv_blt` is officially an indicator only | Documented in REFERENCE.md tooltip | Documented divergence |
| No formal Heat Vulnerability Index (CDC/ATSDR HVI) — using NLCD-intensity proxy | Lower-priority methodology improvement | Open; on roadmap |
| No Annual NLCD migration (prototype stays on legacy 21-class) | InVEST sample data + biophysical tables are calibrated to legacy. Migrating would require revalidating everything. NatCap's curated SA data confirmed using legacy NLCD 2021 (Brief 12). | Open question for NatCap. |

### Data gaps (NatCap also has them)

| Gap | Reason | Status |
|---|---|---|
| SA NDR model not implemented | Outside original prototype scope | Open — see Active asks |
| SA Cooling Energy Savings and Flood Damage Avoided degrade to $0 — no per-building damage rates | NatCap also leaves the damage loss table blank in their SA setup (per the README). The data gap is real, not a prototype shortcoming | Persistent — would require independent SA damage estimation |
| SA UCM weights (shade=0.6, albedo=0.2, et=0.2) verified to match | NatCap README specifies; prototype matches | ✅ Closed |
| SA UCM `air_blending_distance=600`, `maximum_cooling_distance=450` verified to match | NatCap README specifies; prototype matches | ✅ Closed |

---

## Decisions made without confirmation

Choices made based on Daniel's reading of canonical NatCap output, not explicit NatCap input. Recorded for later confirmation.

| Decision | Date | Rationale | Confirmation path |
|---|---|---|---|
| SA UCM aligned to NatCap's heat-wave-day scenario (`uhi_max=11`, `t_ref documented but not used`) | 2026-05-24 (Brief 14) | NatCap's SA README states these values explicitly; aligning per working principle. SA temperature deltas now ~3× larger than before. | NatCap doesn't need to confirm — they've already documented this. |
| Prototype MN UNA migrated to MN-project canonical values (`demand=250`, `radius=1000`, `decay=exponential`) | 2026-05-24 (Brief 22) | NatCap's MN sample data (March 2026, recent) documents these values explicitly. The per-city framing principle (Brief 14 for SA UCM) applies: align MN with the MN project, not blend with SA-project values. MN nature_access_pct dropped from ~43% to ~9.5% baseline — expected magnitude given 15× demand increase. | NatCap confirmation at the symposium would close this; not blocking. The decision is reversible if NatCap flags the MN-project values as superseded. |
| Use per-capita supply deficit (no population multiplier) for undersupply-focused placement | 2026-05-23 (Brief 9) | Matches InVEST UNA's `urban_nature_balance_percapita.tif` framing. Aggregate-need form was a homegrown proxy. | Could surface to NatCap with empirical findings — Brief 9 saturation (100% SA, 67% MN) suggests the canonical framing may not be usable as-is for placement on county-scale AOIs. |
| Rename "equity-focused" → "undersupply-focused" | 2026-05-23 (Brief 9) | InVEST UNA reserves "equity" for demographic-group stratification. | Vocabulary change; no expected NatCap pushback. |
| Use per-pixel runoff Q from SCS-CN equation for flood-focused, not raw CN | 2026-05-23 (Brief 9) | Q is canonical UFR output. | Routine alignment. |
| Rename "Cooling Capacity / CC" → "Heat Mitigation Index / HMI" in UI | 2026-05-23 (Brief 8) | Reported value was already canonical HMI; label was stale. | Vocabulary cleanup. |
| Default to gitignoring NatCap-curated SA rasters; commit only small CSVs/docs | 2026-05-24 | Avoid large files in git; data is reproducible from NatCap's source. | Pragmatic. |
| Keep "Balanced" placement strategy as app-specific heuristic, no InVEST analog | Throughout | No InVEST model prescribes balanced placement; ROOT does weighted-sum LP. | Documented in REFERENCE.md with pointer to ROOT. |
| Surrogate optimizer is app-specific (random-forest over ~90 pre-computed runs); not ROOT | Throughout | Different optimization framework than ROOT's LP. | Documented with pointer to ROOT. |

---

## Open questions to raise with NatCap

Grouped by priority. Things to ask next time there's a chance to.

### Highest priority — per-city alignment

1. **Same per-city question for UFR `rainfall_depth`.** NatCap MN uses 100 mm; NatCap SA uses 157 mm; prototype uses 50.8 mm for both. The prototype's 2-inch design storm doesn't match either project. Should the prototype switch to per-city rainfall (100 mm MN / 157 mm SA), or is its design-storm choice defensible as a different framing?

2. **Are the MN sample data values still current, or have they been superseded by the SA-project framing?** The MN UNA bundle is dated March 2026 (3 months ago); the SA README is from later. NatCap may have updated their thinking and not retroactively republished MN. Brief 22 adopted the MN-project values pending this confirmation; reversible if NatCap flags MN as superseded.

### High priority — operational

4. **Per-crop SA food forest yield?** Currently using 8,500 lbs/acre placeholder for hot semi-arid. NatCap's SA Urban Agriculture project should have per-crop numbers.

5. **For SA NDR integration: are watershed and DEM files in the shared Drive folder somewhere?** The README references `sa_dem_3m_proj.tif` and `San_Antonio_TX_buffer_mod.shp` with `E:/GIS/` paths suggesting they're on a NatCap internal machine, not shared. Need to obtain to implement NDR.

6. **Is the per-capita-only undersupply formulation right for placement weighting?** Brief 9's saturation finding (100% on SA, 67% on MN) shows that strict per-capita deficit concentrates too aggressively to be usable at moderate pct values. Canonical framing for *reporting* (`urban_nature_balance_percapita`) may differ from canonical for *placement*. Worth asking the UNA team.

### Medium priority

7. **For mixed-allocation scenarios (gi=50/ff=50/hd=0), does anyone in the NatCap ecosystem measure placement-strategy effects?** Diagnostic only measured single-cover.

### Low priority

8. **Should the prototype migrate to Annual NLCD once NatCap's own data does?** No urgency unless they signal a migration.

9. **What's the right way to validate a placement strategy from NatCap's perspective?** The three-layer diagnostic (variance / selectivity / outcome delta) might not be how they think about it.

10. **Building damage rates per-city?** SA has no per-building type codes; downstream metrics degrade to $0. Also a NatCap gap, not just a prototype gap.

### Closed questions

The following were open at session start (2026-05-24) and have been resolved by the meeting note + README + MN sample data audit:

- ~~NatCap NLCD vintage~~ → Legacy NLCD 2021 (per `gdalinfo -hist`, Brief 12)
- ~~NatCap UNA demand for SA~~ → 16.7 m²/capita (per SA README)
- ~~NatCap UNA search radius for SA~~ → 800 m uniform (per SA README)
- ~~NatCap UCM `uhi_max` for SA~~ → 11 °C (per SA README, applied in Brief 14)
- ~~MN UNA: switch from SA-project values to MN-project canonical?~~ → ✅ Done in Brief 22 (demand 16.7→250, radius 800→1000, decay dichotomy→exponential). Reversible if NatCap flags MN-project framing as superseded.
- ~~"Wallpaper approach" meaning~~ → Uniform tiling of conversions; equivalent to prototype's `random` placement strategy
- ~~NatCap UCM weights for SA~~ → shade=0.6, albedo=0.2, et=0.2 (per SA README; prototype matches)
- ~~NatCap UCM `uhi_max` for MN~~ → 2.05 °C (per MN args.json; prototype matches)
- ~~NatCap UCM blending and cooling distances for MN+SA~~ → 600 m / 450 m (consistent across both, prototype matches)

---

## Data NatCap has shared

Inventory of curated data from NatCap that's been delivered.

| Item | Received | Status | Notes |
|---|---|---|---|
| SA NLCD+NLUD+tree LULC overlay + matched UCM/UNA/Carbon biophysical tables + pre-computed InVEST results | 2026-05-23 | Downloaded, integration queued | At `data/sa/natcap_2024/`. See `DATA_INVENTORY.md` for full file list. |
| `README_San_Antonio_InVEST_model_inputs.docx` (loose doc) | 2026-05-24 reviewed | ✅ Downloaded and read | Documents NatCap's InVEST args for SA across UCM/Carbon/UNA/UFR/NDR. Settled four high-priority open questions (UNA demand, UNA search radius, UHI value, NLCD vintage). Triggered Brief 14 (SA UHI fix). |
| `Ecosystem_Explorer_-_Meeting_Note.docx` (loose doc) | 2026-05-24 reviewed | ✅ Downloaded and read | Establishes project context: Natural Capital Symposium June 29–July 1, 2026, Google AI for Science proposal, full six-model SA project scope (revealing NDR as a missing model in the prototype). |
| `Minneapolis/building footprints/` (Drive subfolder) | 2026-05-24 inspected | ⏸️ Reviewed, no download needed | Single ESRI shapefile bundle `gis_osm_buildings_a_fre_MN.*` — Geofabrik's unmodified OSM extract for Minnesota. Same source as the prototype's `download_osm_minneapolis.py`. No new data. |
| `Minneapolis/roads/` (Drive subfolder) | 2026-05-24 inspected | ⏸️ Reviewed, no download needed | Single ESRI shapefile bundle `gis_osm_roads_free_1_MN.*` — Geofabrik's unmodified OSM extract. Same source as the prototype's `download_osm_minneapolis.py` and `process_osm_expanded.py`. No new data. |
| `Minneapolis/Urban model sample data same AOI Minneapolis/` (3 ZIPs of canonical InVEST sample data for MN AOI) | 2026-05-24 downloaded + inspected | ✅ Downloaded; surfaced major findings | At `data/invest/mn_sample_data_natcap_2026/`. Three args.json files extracted and compared against prototype values. Confirmed MN UCM full alignment (uhi_max=2.05, all distances match). Surfaced that the prototype's MN UNA uses SA-project values (`demand=16.7`, `radius=800`, `decay=dichotomy`) rather than MN-project values (`demand=250`, `radius=1000`, `decay=exponential`) — a real per-city misalignment. See "Per-city parameter framing" section. |

---

## Symposium and timeline

**Natural Capital Symposium 2026:** June 29 – July 1, 2026 (~5 weeks from 2026-05-24). User is attending but **not presenting**. Prototype will be visible/discussed informally with NatCap collaborators.

**Implication for pacing:** No formal deliverable deadline, but having the prototype in good shape — known divergences documented, NatCap alignment current, recent integrations stable — is the working target for the symposium window. The current sequence (Brief 14 UHI fix → Brief 15 collab doc update → Brief 16 commit MN findings → Briefs 17+ SA NatCap data integration) is paced for that.

**Google AI for Science proposal:** Active funding pitch (link in `Ecosystem_Explorer_-_Meeting_Note.txt`). Describes an "AI-augmented InVEST platform" with agentic systems, dynamic data, multi-model integration. The prototype is positioned as an early example of this vision. Status of the proposal itself: unknown to Daniel; track via NatCap conversations.

---

## Maintenance

This doc gets updated when:

- NatCap explicitly asks for something new (add to Active asks)
- Daniel infers a new priority (add to Inferred priorities)
- A new gap is identified (add to Gaps)
- A decision is made without checking with NatCap first (add to Decisions made without confirmation)
- A new question to raise comes up (add to Open questions)
- NatCap shares new data (add to Data NatCap has shared)
- An ask gets delivered, a gap closes, a question gets answered (update status, don't delete history — strikethrough or move to a "Closed" subsection)

Pair with `NATCAP_ALIGNMENT.md` updates. Same discipline as `WHATS_NEW`.

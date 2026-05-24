# NatCap Collaboration

**Purpose:** Track what NatCap is asking for, what they probably want (read between the lines), and where the prototype's current state diverges from either.

**Audience:** Daniel and future Claude sessions. Not shared with NatCap.

**Naming:** Refer to all NatCap collaborators as "NatCap" — no individual names.

**Relationship to other docs:**

- `NATCAP_ALIGNMENT.md` — per-surface alignment status (six tables: methodology, parameters, AOI, research directions, vocabulary). The *result*.
- This doc — the *process*. Asks, inferences, gaps, decisions made without confirmation, open questions.

---

## Active asks

What NatCap has explicitly requested. Status as of 2026-05-24.

| Ask | When | Status | Notes |
|---|---|---|---|
| Adopt the curated SA dataset (NLCD + NLUD + tree-canopy compound LULC + matched biophysical tables) | Received 2026-05-23 | 🔄 Integration queued | Data folder in `data/sa/natcap_2024/`. README says "please update all input data to what is contained here." Integration is multi-brief workstream (Brief 12+). |
| Implement InVEST Nutrient Delivery Ratio (NDR) model for SA | Meeting note (April 2026) lists NDR as part of NatCap's SA Urban Agriculture project scope | ⏸️ Not started | Inputs documented in `data/sa/natcap_2024/README_San_Antonio_InVEST_model_inputs.txt`: `ndr_biophysical_parameters_vNLCDTree_SA.csv`, SA DEM at 3 m (`sa_dem_3m_proj.tif`), `runoff_proxy_path` at 32 inches precipitation, watersheds shp. Currently no NDR-related code in the prototype. Real new model implementation, not a parameter tweak. The DEM and watersheds paths in the README are `E:/GIS/…` (NatCap internal machine) — see new open question. |
| Use InVEST canonical models where available | Throughout | ✅ Mostly done | UCM validated MAE=0 against canonical. UFR uses canonical CN-based runoff. UNA uses canonical 2SFCA. UMH uses canonical formula. Carbon is still single-rate proxy — pending. |
| Implement Urban Mental Health model | Earlier session | ✅ Done | InVEST UMH v3.19.0 integrated. Cards on dashboard. |
| Use canonical Heat Mitigation Index, not approximation | Earlier session | ✅ Done | `_compute_hmi_raster` validated against `natcap.invest.urban_cooling_model.execute()` at MAE=0. |
| Separate placement-constraint inputs from model-input data | Earlier session | ✅ Done | Comprehensive OSM building mask added; UFR sample buildings still drive damage metrics. |

---

## Inferred priorities

What NatCap probably wants based on documents, project framing, and how they've engaged. Not explicitly asked but consistent with their direction.

- **The prototype should look and feel like an InVEST model run, not a separate methodology.** The "align to canonical" principle Daniel adopted (2026-05-23) was inferred from NatCap's general posture about model fidelity. Confirmed implicitly by the SA data dump being a complete reparameterization rather than a parameter patch.
- **Per-capita supply/demand is the right framing for UNA.** Their canonical output is `urban_nature_balance_percapita.tif`. The aggregate-need framing (population × deficit) is not in their vocabulary.
- **The SA Urban Agriculture project is the primary SA use case.** The NatCap-curated SA data is keyed to this project; the food forest yield estimate is from it (8,500 lbs/acre placeholder pending project-report numbers).
- **Tree canopy matters more than NLCD class alone.** The new LULC overlay framework treats tree canopy as the dominant signal (any pixel with high canopy gets shade=0.66 regardless of NLCD class).
- **ROOT exists but is not being pursued for this prototype.** Mentioned in their project doc; the prototype's surrogate-based optimizer is acknowledged as a different (simpler) approach.
- **The prototype is positioned as an early example of NatCap's "AI-augmented InVEST platform" pitch.** The Google AI for Science proposal referenced in the meeting note describes an agentic-systems-driven InVEST platform with dynamic input data (AlphaEarth), multi-model integration, scenario generation, and trade-off/optimization analysis. The prototype implements pieces of this: multi-model integration (UFR/UCM/UNA/UMH/Carbon), scenario generation (Conversion Mix sliders), trade-off analysis (Tradeoff Analysis tab), surrogate-based optimization (Find Best Scenario). AlphaEarth integration is research-only (`ALPHAEARTH_FEASIBILITY.md`).
- **NatCap's SA Urban Agriculture project is the canonical SA reference.** Six InVEST models (Crop Production, UCM, Carbon, UNA, UFR, NDR), with parameter tables tuned to the project's compound NLCD+NLUD+tree LULC framework. The prototype implements 5 of these 6.

---

## Gaps

Where the prototype currently diverges from NatCap asks or inferred priorities. Each gap has a reason.

| Gap | Reason | Status |
|---|---|---|
| SA still uses independent NLCD + tuned cooling table, not the curated dataset | Just received the data 2026-05-23; integration is multi-brief | Briefs 12-16 queued |
| Carbon is single-rate proxy, not four-pool InVEST | Lower-priority methodology upgrade | Open; on roadmap |
| Cooling Energy Savings uses per-pixel aggregation, not InVEST's per-building T_air sampling | Methodology gap acknowledged in REFERENCE.md | Documented divergence |
| Food forest yield is single per-city benchmark, not per-crop NatCap project values | Waiting on SA project-report numbers | Pending |
| No formal Heat Vulnerability Index (CDC/ATSDR HVI) — using NLCD-intensity proxy for the heat overlay | Lower-priority methodology improvement | Open; on roadmap |
| Flood Damage Avoided produces dollar values; InVEST UFR's `serv_blt` is officially an indicator only | Documented in REFERENCE.md tooltip | Documented divergence |
| No Annual NLCD migration (prototype stays on legacy 21-class) | InVEST sample data + biophysical tables are calibrated to legacy. Migrating would require revalidating everything. | Open question for NatCap — would they recommend migrating once their own data does? |
| Nutrient Delivery Ratio (NDR) not implemented | Outside original prototype scope; NatCap's SA project includes it as one of six models | Open — see new Active ask above |
| Food forest yield uses single per-city benchmark (MN 11,500 lbs/acre, SA 8,500 placeholder); NatCap uses InVEST Crop Production with per-crop parameterization (`CoSA_Crop_production_ESModeling`) | Different methodology framework; per-crop data not yet obtained | Open — would require CoSA model integration |
| Flood mitigation methodology divergence: NatCap pre-computes UFR over two alternative LULCs (20-acre and 40-acre food-forest expansion scenarios at 10 m resolution); prototype runs UFR live per slider position | Different workflow framework. The prototype's live-conversion approach is more user-interactive; NatCap's pre-computed approach is more aligned with their InVEST workflow toolkit | Open as methodology divergence; not a "fix" — both approaches are defensible |
| SA Cooling Energy Savings and Flood Damage Avoided degrade to $0 — no per-building damage rates | NatCap also leaves the damage loss table blank in their SA setup (per `README_San_Antonio_InVEST_model_inputs.txt`: "damage loss table (csv): (leaving blank)"). The data gap is real, not a prototype shortcoming | Persistent — would require independent SA damage estimation |

### Closed (resolved gaps)

| Gap | Reason | Status |
|---|---|---|
| SA UHI parameter (was 3.5; NatCap canonical is 11 for heat-wave-day scenario) | Was a placeholder pending NatCap's calibrated value | ✅ Resolved 2026-05-24 (Brief 14); SA temperature deltas now ~3× larger |

---

## Decisions made without confirmation

Choices made based on Daniel's reading of canonical NatCap output, not explicit NatCap input. Recorded for later confirmation.

| Decision | Date | Rationale | Confirmation path |
|---|---|---|---|
| Use per-capita supply deficit (no population multiplier) for undersupply-focused placement | 2026-05-23 (Brief 9) | Matches InVEST UNA's `urban_nature_balance_percapita.tif` framing exactly. Aggregate-need form was a homegrown proxy. | Could surface to NatCap with empirical findings — current SA saturation (100%) suggests the canonical framing may not be usable as-is for placement on county-scale AOIs. |
| Rename "equity-focused" → "undersupply-focused" | 2026-05-23 (Brief 9) | InVEST UNA reserves "equity" for demographic-group stratification. | Vocabulary change; no expected NatCap pushback. |
| Use per-pixel runoff Q from SCS-CN equation for flood-focused, not raw CN | 2026-05-23 (Brief 9) | Q is canonical UFR output (`Q_mm.tif`). | Routine alignment; no expected pushback. |
| Rename "Cooling Capacity / CC" → "Heat Mitigation Index / HMI" in UI | 2026-05-23 (Brief 8) | Reported value was already canonical HMI; label was stale. | Vocabulary cleanup. |
| Default to gitignoring NatCap-curated SA rasters; commit only small CSVs/docs | 2026-05-24 | Avoid large files in git; data is reproducible from NatCap's source. | Pragmatic; would only matter if NatCap requires the rasters in the repo. |
| Keep "Balanced" placement strategy as app-specific heuristic, no InVEST analog | Throughout | No InVEST model prescribes balanced placement; ROOT does weighted-sum LP. | Documented in REFERENCE.md with pointer to ROOT. |
| Surrogate optimizer is app-specific (random-forest over ~90 pre-computed runs); not ROOT | Throughout | Different optimization framework than ROOT's LP approach. | Documented with pointer to ROOT. |
| SA UCM aligned to NatCap's heat-wave-day scenario (`uhi_max=11`, `t_ref=35`) rather than average-summer-day estimate | 2026-05-24 (Brief 14) | NatCap's README states these values explicitly; aligning per working principle. Both choices are methodologically defensible; alignment beats independent calibration. Note: the prototype reports pure deltas (no absolute T_air calculation), so `t_ref` has no analog in the codebase to update — only `uhi_max_c` changed. | NatCap doesn't need to confirm — they've already documented this in their README. The prototype now matches their published config. |

---

## Open questions to raise with NatCap

Things to ask next time there's a chance to. Grouped by priority.

### High priority

1. **Is per-capita supply deficit the right NatCap framing for placement weighting?** The empirical finding from Brief 9 is that pure per-capita deficit saturates aggressively (100% of cells on SA, 67% on MN). Canonical framing for *reporting* is `urban_nature_balance_percapita`; canonical framing for *placement* may be different. Worth asking the UNA team.

2. **What `t_ref` (reference air temperature) does NatCap consider canonical for Minneapolis?** The InVEST UCM sample args.json for MN (`data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/invest_urban_cooling_model_args_MN.json`) specifies `t_ref = 23.2 °C` — a moderate-summer reference. NatCap's curated SA value is `t_ref = 35 °C` — a heat-wave-day reference. These are very different scenarios. The prototype reports pure deltas (no absolute T_air), so `t_ref` doesn't currently affect any user-facing output — but if NatCap considers MN's canonical scenario to be heat-wave-day too, the framing inconsistency is worth surfacing. Worth verifying against the InVEST UCM sample args.json that the prototype was originally seeded from.

3. **For SA NDR integration: are watershed delineation and DEM data in the NatCap-curated folder?** The README references `sa_dem_3m_proj.tif` and `San_Antonio_TX_buffer_mod.shp` watersheds path — but the paths are `E:/GIS/_natcap/san_antonio/…` (NatCap internal machine), not in the shared Drive folder. If we adopt NDR we need these files; raising with NatCap is necessary.

### Medium priority

4. **What's the per-crop SA food forest yield?** Currently using 8,500 lbs/acre placeholder for hot semi-arid. NatCap's SA Urban Agriculture project (`CoSA_Crop_production_ESModeling`, referenced in meeting note) should have actual numbers.

5. **For mixed-allocation scenarios (gi=50/ff=50/hd=0), does anyone in the NatCap ecosystem measure placement-strategy effects?** Diagnostic only measured single-cover.

### Low priority

6. **Should the prototype migrate to Annual NLCD once NatCap's own data does?** No urgency unless they signal a migration.

7. **What's the right way to validate a placement strategy from NatCap's perspective?** The three-layer diagnostic (variance / selectivity / outcome delta) might not be how they think about it.

8. **Building damage rates per-city?** SA has no per-building type codes; downstream metrics degrade to $0. NatCap also leaves this blank in their SA setup (per `README_San_Antonio_InVEST_model_inputs.txt`) — so this is a shared data gap, not a prototype-specific issue. A NatCap-provided damage table or a typed buildings shapefile would unblock.

### Closed (resolved questions)

- ~~**What `urban_nature_demand` did NatCap use for SA?**~~ → ✅ Resolved 2026-05-24: **16.7 m²/capita**, matches prototype (from `README_San_Antonio_InVEST_model_inputs.txt`: "urban nature demand per capita (number) (m²): 16.7").
- ~~**What `search_radius` did NatCap use for SA UNA?**~~ → ✅ Resolved 2026-05-24: **800 m uniform**, matches prototype (from README: "uniform search radius (number) (m): 800").
- ~~**What `uhi_max` did NatCap use for SA?**~~ → ✅ Resolved 2026-05-24: **11 °C** (from README: "UHI effect: 11"); applied in Brief 14.
- ~~**Legacy NLCD vs Annual NLCD in NatCap SA data?**~~ → ✅ Resolved 2026-05-24: **Legacy NLCD 2021** — 16 unique values from the legacy 21-class set, confirmed via `gdalinfo -hist` in Brief 12. Aligns with the prototype.
- ~~**What does "wallpaper approach" mean?**~~ → ✅ Resolved 2026-05-24: Per the meeting note, listed as one of the "Simpler approaches" alongside "Road layer + building layer + existing tree layer." Refers to uniform tiling of conversions across the landscape; equivalent to the prototype's `random` placement strategy (with the placement mask applied). NatCap doesn't view this as a problem — it's a legitimate baseline approach.

---

## Data NatCap has shared

Inventory of curated data from NatCap that's been delivered.

| Folder | Received | Status | Notes |
|---|---|---|---|
| SA NLCD+NLUD+tree LULC overlay + matched UCM/UNA/Carbon biophysical tables + pre-computed InVEST results | 2026-05-23 | Downloaded, integration queued | At `data/sa/natcap_2024/`. See `DATA_INVENTORY.md` for full file list. |
| Minneapolis | Earlier (separate "Minneapolis" folder in Drive's "Shared with me") | Not yet downloaded | Probably the canonical NatCap MN dataset; would pair with the SA data for parity. |
| roads | Earlier | Not yet downloaded | Unknown contents. |
| building footprints | Earlier | Not yet downloaded | May contain typed SA buildings — would unblock SA Cooling Energy Savings and Flood Damage Avoided dollar metrics. |
| Urban model sample data same AOI Minneapolis | Earlier | Not yet downloaded | Unknown contents. |
| README_San Antonio InVEST model inputs | Earlier | Not yet downloaded | Loose doc. |

---

## Symposium and timeline

**Natural Capital Symposium 2026:** June 29 – July 1, 2026 (~5 weeks from
2026-05-24). User is attending but **not presenting**. Prototype will be
visible/discussed informally with NatCap collaborators.

**Implication for pacing:** No formal deliverable deadline, but having
the prototype in good shape — known divergences documented, NatCap
alignment current, recent integrations stable — is the working target
for the symposium window. The current sequence (Brief 14 UHI fix →
Brief 15 collab doc update → Briefs 16+ SA NatCap data integration) is
paced for that.

**Google AI for Science proposal:** Active funding pitch (link in
`Ecosystem_Explorer_-_Meeting_Note.txt`). Describes an "AI-augmented
InVEST platform" with agentic systems, dynamic data, multi-model
integration. The prototype is positioned as an early example of this
vision. Status of the proposal itself: unknown to Daniel; track via
NatCap conversations.

---

## Maintenance

This doc gets updated when:

- NatCap explicitly asks for something new (add to Active asks)
- Daniel infers a new priority (add to Inferred priorities)
- A new gap is identified (add to Gaps)
- A decision is made without checking with NatCap first (add to Decisions made without confirmation)
- A new question to raise comes up (add to Open questions)
- NatCap shares new data (add to Data NatCap has shared)
- An ask gets delivered, a gap closes, a question gets answered (update status, don't delete history — strikethrough or move to a "Closed" section)

Pair with `NATCAP_ALIGNMENT.md` updates. Same discipline as `WHATS_NEW`.

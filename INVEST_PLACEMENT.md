# InVEST Placement Methodology Research

A research note investigating how each InVEST model the app builds on
handles spatial placement of land use changes. Informs future
implementation of placement strategies in the Ecosystem Explorer's
`evaluate_scenario` function.

**Date:** 2026-05-20
**Sources:** InVEST 3.19.0 user guides (6 models), InVEST Scenario Generator:
Proximity Based guide, NatCap SA Urban Agriculture project (2023) via
secondary reporting (project pages, news coverage — the PDF is image-based
and not machine-parseable; see SA section for source list).

---

## Summary

All six InVEST models the app builds on are **placement-agnostic**: they
accept user-provided LULC rasters and compute outputs from them, with no
built-in logic for deciding where land use changes should occur. Scenario
construction — the "where" question — is explicitly outside the model
boundary. InVEST does ship a **Scenario Generator: Proximity Based** tool,
but it is designed for habitat-fragmentation studies (converting habitat
nearest-to or farthest-from existing edges) and has no urban-specific
mode. A second **rule-based scenario generator** (transition-probability
driven) is referenced in the docs but has no dedicated guide page. Neither
tool produces the kind of suitability-weighted urban placement the app
needs. The closest published precedent — the NatCap San Antonio Urban
Agriculture project (2023) — used a simple eligibility filter (publicly
owned + underutilized parcels) rather than a weighted suitability surface,
with InVEST applied for evaluation after placement, not to drive it. The
biggest gap between the app's current random sampling and canonical InVEST
practice is not that random is "wrong" — InVEST has no canonical placement
— but that the app lacks the eligibility and suitability layers (land
ownership, flood plains, food deserts, equity indices) that real-world
NatCap projects use to constrain the convertible pool before running the
models.

---

## Per-model findings

### Urban Flood Risk Mitigation (UFR)

| Question | Answer |
|----------|--------|
| **Does InVEST recommend a placement strategy?** | No. The model is placement-agnostic. |
| **Strategy description** | N/A. The guide describes the SCS Curve Number method and per-watershed service indicators but provides no guidance on where to site green infrastructure for flood mitigation. The model computes retention indices and a `serv_blt` service indicator, but the guide explicitly calls these "only an indicator of service, it is not an actual measure of either damage or savings." Outputs describe where services occur, not where to invest. |
| **Inputs that could inform placement** | The model requires a soil hydrologic group raster and a CN biophysical table. Together these identify which pixels have the highest runoff potential (high CN on D-class soils) — a natural suitability input for targeting green infrastructure. The model also takes building footprints for the damage indicator, which could inform placement near high-value structures. |
| **Does the app have those inputs?** | Yes. Both MN and SA have SSURGO soil rasters and CN tables. Building footprints are available for both cities (InVEST sample for MN downtown, OSM for SA). The app could compute a per-pixel "runoff reduction potential" from existing data. |
| **Gap from random placement** | Random placement treats all developed pixels as equally valuable for flood mitigation. In reality, converting a high-CN pixel on D-class clay soil yields far more runoff reduction per acre than converting a low-CN pixel on A-class sandy soil. The app already has the data to compute this differential but doesn't use it for siting. |

**Source:** [InVEST UFR User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_flood_mitigation.html)

---

### Urban Cooling Model (UCM)

| Question | Answer |
|----------|--------|
| **Does InVEST recommend a placement strategy?** | No. The model is placement-agnostic. |
| **Strategy description** | N/A. The guide describes CC computation, the HMI park-proximity step, air temperature interpolation, and energy savings, but contains no siting guidance for cooling interventions. The 2-hectare park threshold and `d_cool` distance parameter describe how the model *propagates* cooling effects from green spaces, not where to *create* them. The guide notes that `d_cool` and the air-blending radius `r` are "difficult to derive from the literature as they vary with vegetation properties, climate (effect of large green spaces), and wind patterns (air mixing)." |
| **Inputs that could inform placement** | The biophysical table's `green_area` flag (binary 0/1) classifies which LULC types count as green space. The 2-ha threshold means that interventions creating contiguous green areas ≥ 2 ha would trigger the park-cooling-plume effect. Building footprints with per-type energy consumption rates identify where cooling saves the most AC dollars. Population density identifies where cooling benefits the most people. |
| **Does the app have those inputs?** | Partially. Building footprints and energy consumption tables exist. Population rasters exist. The app does not currently compute a "cooling benefit potential" surface, but the baseline CC raster identifies the coldest and hottest pixels — hot pixels adjacent to buildings are the highest-value targets. |
| **Gap from random placement** | Random placement scatters green conversions uniformly, preventing the formation of contiguous ≥ 2 ha patches that would trigger the park-cooling-plume effect in canonical InVEST. It also ignores proximity to buildings (where cooling saves AC energy) and to high-population pixels (where cooling benefits the most people). The existing heat-priority mode partially addresses this by weighting toward high-intensity development, but doesn't consider building proximity or contiguity. |

**Source:** [InVEST UCM User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html)

---

### Urban Nature Access (UNA)

| Question | Answer |
|----------|--------|
| **Does InVEST recommend a placement strategy?** | No, but the model's balance output implicitly serves as a placement guide. |
| **Strategy description** | The UNA model computes supply, demand, and balance per administrative unit. The balance metric identifies areas with a surplus or deficit of urban nature relative to population demand. The guide tracks "the number of people in an administrative unit with an urban nature deficit." While the model doesn't prescribe placement, negative-balance areas are the obvious candidates for new nature. The guide also notes that supply/demand/balance "can be summarized to different groups within the population (e.g., by different age groups, levels of income, race or ethnicity, etc.), which may be important for equity considerations" — explicitly connecting the output to equity-informed placement. |
| **Inputs that could inform placement** | Population raster, administrative boundaries, per-group demographic data (optional), search radii per nature type (capturing mobility differences — "people who own cars may travel further to recreate than people who rely on public transit"). |
| **Does the app have those inputs?** | Partially. Population rasters and census tracts exist for both cities. The app computes a nature access score per pixel. It does not compute the 2SFCA supply-demand balance, and it does not have per-group demographic breakdowns. But the existing access score could identify deficit areas. |
| **Gap from random placement** | Random placement ignores where nature is most needed. A pixel in a nature-surplus area gets the same conversion probability as one in a deficit area. The app's existing nature access score could be inverted into a siting weight: pixels with low access scores (far from existing nature, high population) would be prioritized. This is the model where placement matters most — the whole point of UNA is to identify where nature is undersupplied. |

**Source:** [InVEST UNA User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_nature_access.html)

---

### Urban Mental Health (UMH)

| Question | Answer |
|----------|--------|
| **Does InVEST recommend a placement strategy?** | No. |
| **Strategy description** | N/A. The guide describes the preventable-cases formula (`PC = (1 − RR) × BIR × POP`) and notes that "land use scenarios are key to understanding how alternative land use change and associated greenery change might impact mental health benefits." But it provides no methodology for identifying optimal greenspace placement. The model accepts baseline + alternate LULC (or NDVI) rasters and quantifies health outcomes from the difference. |
| **Inputs that could inform placement** | Population count raster (where more people live, more cases are preventable), baseline NDVI (where NDVI is lowest, the marginal gain from greening is largest — the RR formula is exponential in ΔNE, so gains are largest where baseline NE is low). |
| **Does the app have those inputs?** | Yes. Population rasters and baseline NE rasters exist for both cities. The app could compute a "mental health benefit potential" surface: `population × (1 − exp(ln(RR) × 10 × expected_ΔNE))` per pixel, where `expected_ΔNE` is the NDVI gain from converting that pixel. |
| **Gap from random placement** | The RR formula means that greening a high-population, low-NDVI pixel prevents more cases than greening a low-population, high-NDVI pixel. Random placement averages over this gradient. The gap is largest in cities with high spatial variance in population density and baseline NDVI — SA (1.9M people, large suburban-to-urban gradient) likely has a bigger placement effect than MN downtown (154K, more uniform). |

**Source:** [InVEST UMH User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_mental_health.html)

---

### Carbon Storage and Sequestration

| Question | Answer |
|----------|--------|
| **Does InVEST recommend a placement strategy?** | No. The model is purely placement-agnostic. |
| **Strategy description** | N/A. The guide describes a pixel-by-pixel stock comparison between baseline and alternate LULC maps. It explicitly frames the model as "simply applied to the baseline landscape and a real or projected alternate landscape, and the difference in storage is calculated, pixel by pixel." No spatial optimization, no siting guidance, no soil-type consideration for placement. |
| **Inputs that could inform placement** | The model uses a per-LULC-class carbon pool table (4 pools: above-ground, below-ground, soil, dead organic matter). In principle, soil organic carbon potential varies spatially, but the model doesn't use spatial soil data — it applies uniform per-class rates. |
| **Does the app have those inputs?** | N/A. The app uses a simpler single-rate-per-class approach that doesn't benefit from spatial placement optimization. A pixel converted to food forest sequesters the same rate regardless of location. |
| **Gap from random placement** | Minimal for the app's current implementation. Since the app uses flat per-class rates (3.5 tons CO2e/acre for food forest, 2.0 for GI), placement doesn't affect carbon outcomes — only the count of converted pixels matters. A future upgrade to spatially-varying carbon rates (e.g., from soil organic carbon maps) would make placement relevant, but that's beyond both the current app and the canonical InVEST carbon model. |

**Source:** [InVEST Carbon User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/carbonstorage.html)

---

### Crop Production

| Question | Answer |
|----------|--------|
| **Does InVEST recommend a placement strategy?** | No. |
| **Strategy description** | N/A. The guide describes climate-binned yield distributions for 172 staple crops and fertilizer-response regressions for 10 crops. It explicitly acknowledges the model cannot capture landscape heterogeneity: "A rocky hill slope and a fertile river valley, if they share the same climate, would be assigned the same yield in the current model." The model does not support food forests, agroforestry, or polyculture systems. |
| **Inputs that could inform placement** | Climate bin rasters provide coarse spatial yield variation. No soil suitability data is incorporated. |
| **Does the app have those inputs?** | Not applicable. The app uses a flat yield benchmark (11,500 lbs/acre MN, 8,500 SA) rather than the InVEST climate-binned approach. Like carbon, placement doesn't affect the output. |
| **Gap from random placement** | None for the app's current implementation. The flat yield benchmark means every food forest pixel produces the same amount regardless of where it's placed. A future upgrade to spatially-varying yields (soil quality, microclimate) would make placement relevant, but that's a different model entirely. |

**Source:** [InVEST Crop Production User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/crop_production.html)

---

## Cross-cutting findings

### Unified approach vs. per-model strategies

There is **no unified InVEST-canonical placement approach**. All six models
are evaluation tools: they score a given LULC scenario, they don't generate
one. InVEST does ship two scenario-generation tools, but neither is
designed for urban green-infrastructure siting:

1. **Scenario Generator: Proximity Based** — converts habitat
   nearest-to or farthest-from existing habitat edges, designed for
   studying how fragmentation patterns affect biodiversity. Inputs:
   focal LULC codes, convertible codes, replacement code, number of
   steps. Single replacement type per run. No urban-specific mode, no
   suitability weighting, no population or equity inputs.
   ([Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/scenario_gen_proximity.html))

2. **Rule-Based Scenario Generator** — creates LULC change maps from
   user-assigned transition probabilities. Referenced in the proximity
   generator's docs as a complement but has no dedicated guide page in
   the current InVEST documentation. Designed for forecasting land-use
   transitions, not for optimizing ecosystem service outcomes.

**Implication for implementation:** since InVEST provides no canonical
urban placement strategy, the app is free to design its own. The
constraint is methodological honesty — the placement strategy should
be documented as an app-specific feature, not attributed to InVEST.

### Cross-model dependencies

**No InVEST model's placement depends on another model's outputs.** Each
model takes a LULC raster independently. However, the research reveals
natural complementarities that a multi-objective placement strategy could
exploit:

- **UFR × UCM:** Pixels with high runoff potential (high CN, D-class
  soil) that are also near buildings (high cooling-energy-savings
  potential) are high-value targets for both flood and cooling.
- **UNA × UMH:** Pixels in nature-deficit areas with high population
  density maximize both nature access and mental health outcomes.
- **UCM contiguity:** The 2-ha park threshold in UCM means that
  clustered conversions are more effective for cooling than scattered
  ones — a constraint that's orthogonal to the other models' pixel-
  independent logic.

These complementarities suggest a **weighted multi-objective suitability
surface** rather than per-model placement strategies.

### SA Urban Agriculture report findings

The NatCap San Antonio Urban Agriculture project report ("Vibrant Land:
The Benefits of Food Forests and Urban Farms in San Antonio," May 2023)
used InVEST for **evaluation, not placement**. The scenario workflow:

1. **Eligibility filter:** Only publicly owned, underutilized parcels
   were candidates for conversion (~16,800 acres across San Antonio).
   Private land was excluded entirely.

2. **Siting criteria (coarse):** Land ownership (public only),
   utilization status (underutilized/vacant), and parcel size. No
   pixel-level suitability surface. Flood plains were flagged as
   particularly suitable because they are less attractive to other
   development. SNAP enrollment data identified food-insecure
   districts (Districts 3 and 5) as priority areas.

3. **Scenarios tested:** At least six — 2 land use types (food forest
   vs. urban farm) × 3 parcel-cap variants (uncapped, max 40
   acres/parcel, max 20 acres/parcel). Three site-specific case
   studies at existing food forests (Tamox Talom at Padre Park, Villa
   Coronado Park, Garcia Street Urban Farm) calibrated yield
   assumptions.

4. **InVEST role:** Models (UCM, UFR, Carbon, Nutrient Delivery, UNA)
   were run on the modified LULC rasters to quantify co-benefits
   ($3.5M/yr cooling services, ~600 lives/yr saved, etc.). InVEST
   outputs did not feed back into placement decisions.

5. **Stakeholder-driven, not optimization-driven:** The project was a
   co-production between NatCap and SA city departments/Food Policy
   Council. City officials identified priorities; NatCap inventoried
   eligible parcels and ran models. No formal multi-criteria GIS
   suitability analysis produced a ranked surface.

**Key insight:** the SA project's "simple eligibility filter" approach is
closer to the app's current NLCD 21–24 + building/road exclusion than to
a sophisticated suitability model. The main difference is that the SA
project filtered on **land ownership** (public only) — a dimension the
app doesn't have. The equity dimension (SNAP enrollment) was applied at
district level, not pixel level.

*SA report URL was not directly parseable (image-based PDF). Findings
synthesized from NatCap project page, Natural Capital Alliance news
release, Phys.org, BioCycle, Texas Public Radio, and San Antonio Current
coverage of the 2023 report. See the agent research log for full source
list.*

---

## Implications for the Ecosystem Explorer

### What the research found

1. **InVEST has no canonical placement strategy.** The models are
   evaluators, not planners. The app's placement logic is entirely its
   own design space.

2. **Four of six models have placement-sensitive outputs** (UFR, UCM,
   UNA, UMH). Two don't (Carbon, Crop Production — both use flat
   per-class rates that are location-independent).

3. **The data for smarter placement already exists in the app.** Soil
   groups (runoff potential), baseline CC raster (heat exposure),
   building footprints (cooling-energy value), population rasters
   (nature access + mental health impact), and baseline NE rasters
   (mental health marginal benefit) are all loaded at startup.

4. **The SA precedent validates "eligibility filter + evaluation"** as
   a real NatCap workflow. The app already does this (NLCD 21–24 minus
   buildings minus roads). What the app lacks is the suitability
   weighting within the eligible pool.

### Architecture recommendation

A **single multi-objective suitability surface** rather than per-model
placement strategies. Rationale:

- Per-model strategies would require users to choose which model's
  placement to prioritize — flooding, cooling, or access — which
  reintroduces the tradeoff question the app is designed to explore.
- A weighted suitability surface can combine signals from all four
  placement-sensitive models into a single per-pixel score.
- The weights can be exposed as user controls (or tied to the
  optimizer's objective weights) without requiring separate placement
  modes per model.

**Candidate suitability components** (all derivable from existing data):

| Component | Source data | What it captures |
|-----------|-----------|-----------------|
| Runoff reduction potential | CN table + soil raster | Pixels where GI yields the most flood benefit |
| Heat exposure | Baseline CC raster (inverted) | Pixels where greening yields the most cooling |
| Building proximity | Distance to BUILDINGS_RASTER | Pixels where cooling saves the most AC energy |
| Nature deficit | Baseline access score (inverted) | Pixels where nature is most needed |
| Population density | Pop raster | Pixels where benefits reach the most people |
| MH benefit potential | Pop × (1 − baseline NE) | Pixels where greening prevents the most MH cases |

The UCM contiguity concern (2-ha patches for park-cooling plume) is a
second-order effect that could be addressed via a post-sampling clustering
step rather than a suitability weight.

### What's not needed

- Per-model placement strategies (too complex, forces users to choose).
- InVEST's Scenario Generator: Proximity Based (designed for habitat
  fragmentation, not urban siting).
- Land ownership data (would be ideal per the SA precedent, but is a
  new data-sourcing effort not in the current pipeline).

---

## Recommended next steps

1. **Implement a single suitability-weighted placement mode** as an
   alternative to uniform random sampling. The mode computes a per-pixel
   suitability score from the components listed above, normalizes it to
   a probability distribution, and uses it as the sampling weight in
   `np.random.choice`. This replaces the existing heat-priority mode
   (which is a single-signal version of the same idea) with a
   multi-signal version.

2. **Expose suitability weights in the sidebar** — either as individual
   sliders (runoff, cooling, access, population, MH) or as a preset
   menu ("Flood-focused," "Cooling-focused," "Equity-focused,"
   "Balanced"). The preset approach is simpler and avoids a slider
   explosion; individual sliders are more flexible but harder to
   explain.

3. **Keep uniform random as the default.** The suitability mode is an
   opt-in upgrade, not a replacement. Users exploring general tradeoffs
   benefit from seeing average-case outcomes; users testing specific
   interventions benefit from targeted placement.

4. **Do not attribute the suitability strategy to InVEST.** Document it
   as an app-specific placement heuristic informed by InVEST model
   structure but not part of any InVEST model.

5. **Validate before shipping.** Run the suitability-weighted mode
   against the existing baseline test suite (`verify_baselines.py`) to
   confirm that targeted placement produces meaningfully different
   metric outputs. If it doesn't (because the AOI is too small or too
   homogeneous for placement to matter), the feature isn't worth the
   complexity.

6. **Defer land-ownership and food-desert layers.** These would make
   the suitability surface more realistic (per the SA precedent) but
   require new data sourcing. Treat as a future enhancement, not a
   blocker for the initial implementation.

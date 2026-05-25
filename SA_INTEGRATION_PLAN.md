# SA NatCap Data Integration Plan

**Purpose:** Plan the multi-brief integration of NatCap's curated SA dataset
into the prototype. Establishes three foundational decisions (CRS/extent,
conversion-target mapping, sequence), enumerates the brief sequence with
scope per brief, and surfaces open questions for future NatCap conversations.

**Audience:** Daniel and future Claude sessions. Not user-facing.

**Relationship to other docs:**

- `CITY_PARITY.md` — per-city alignment matrix. The "SA Compound LULC
  Framework" subsection in the SA section has the structural inventory
  (Brief 24). This plan sits on top of it.
- `NATCAP_COLLABORATION.md` — running collaboration log. SA integration
  is documented there as an active workstream.
- `DESIGN_NOTES.md` — internal design decisions. Per-city framing
  (Briefs 22+23) established the alignment principle this plan executes
  against.

---

## Scope

What this plan covers: adopting NatCap's curated SA data — compound LULC
raster (`lulc_overlay_3857.tif`), three biophysical tables (`ucm__nlcd_nlud_tree.csv`,
`una__nlcd_nlud_tree.csv`, `carbon__nlcd_nlud_tree.csv`), and the LULC
crosswalk (`lulc_crosswalk.csv`). Possibly also the block-group AOI
(`acs_block_groups_3857.gpkg`).

What this plan does NOT cover:

- **NDR model implementation.** NatCap's SA project includes Nutrient
  Delivery Ratio as the 6th model; the prototype implements 5/6. NDR is
  a separate workstream blocked on NatCap sharing DEM and watersheds
  (see NATCAP_COLLABORATION.md).
- **AlphaEarth integration.** Future LULC pipeline upgrade. Documented
  in `ALPHAEARTH_FEASIBILITY.md`; deferred pending this SA integration.
- **CoSA per-crop yield data.** Replaces the SA food forest placeholder.
  Open ask in NATCAP_COLLABORATION.md, blocked on NatCap sharing.

---

## The three foundational decisions

### Decision 1: CRS and extent

**The situation.**

Prototype's SA stack is EPSG:5070 (NAD83 Conus Albers — equal-area, the
prototype's canonical CRS for SA). Every SA raster (`land_use_2021_sa.tif`,
`soil_group_SA.tif`, `et_annual_sa.tif`, `sa_pop_2020.tif`) sits on the
same 1984×1713 grid at 30 m.

NatCap's SA LULC (`lulc_overlay_3857.tif`) is EPSG:3857 (Web Mercator)
at 34.5 m. Dimensions 2106×2218. Geographic extent overlaps but extends
farther N/S/W than the prototype.

**The options:**

| Option | Description | Pro | Con |
|---|---|---|---|
| A. Reproject NatCap → 5070, clip to prototype extent | Keep prototype's existing grid; resample NatCap's raster with nearest-neighbor (categorical) to align | Minimal disturbance to existing pipeline; matches all other SA rasters | Loses NatCap's extended-coverage pixels; small (~15%) resampling artifacts on categorical data |
| B. Reproject NatCap → 5070, keep NatCap's extent | Extend prototype grid to match NatCap's larger extent; re-fetch / re-clip all other SA rasters to the new extent | Preserves NatCap's coverage; methodologically faithful | Requires regenerating soil, population, ET, buildings, roads rasters at the new extent; touches more code |
| C. Migrate prototype → 3857 | All SA work moves to EPSG:3857 to match NatCap directly | Aligns prototype's SA with NatCap exactly; no reprojection of source data | Web Mercator is poor for area-based metrics (acres, runoff volumes per pixel). Distortion at SA's 29°N latitude is ~15% per-pixel area variance. Compromises area calculations on which most metrics depend. |

**The decision: Option A.**

Rationale: Equal-area projection (5070) is the right methodological
choice for the prototype's area-based metrics (acres converted, runoff
volume, population per pixel). NatCap shipping in 3857 was likely
operational convenience (web display), not a methodology preference;
their MN sample data uses an equal-area projection too. Option C would
compromise the area accuracy the entire metric calculation depends on.

Option A loses some pixels at the extent edges but preserves the
prototype's canonical grid and avoids regenerating five rasters. Option
B is methodologically purer but doesn't add enough value to justify
the regeneration cost given that SA's analysis is already constrained
to roughly Bexar County.

**Implementation note for the first integration brief:** the reprojection
will use `gdalwarp` with `-r near` (nearest-neighbor) for categorical
LULC data, `-r bilinear` for any continuous rasters. Output dimensions
should match the prototype's existing 1984×1713 grid via `-te` (target
extent) flag matching prototype's bounding box.

**For NatCap conversation:** "We reprojected your SA data to EPSG:5070
to preserve area-based metric accuracy. The clip to our existing extent
loses ~15% of your coverage at the edges. Comfortable with that, or
would you prefer we extend our grid to match yours?"

### Decision 2: Conversion-target lucode mapping

**The situation.**

The prototype's UI lets users convert developed pixels to one of three
target land covers: green infrastructure (`CODE_GREEN_INFRA = 90`),
food forest (`CODE_FOOD_FOREST = 41`), high-density development
(`CODE_HIGH_DENSITY = 24`). These are NLCD lucodes.

In compound LULC, every pixel has a compound lucode that encodes
NLCD × NLUD × tree-canopy. When the slider converts a pixel "to food
forest," we need to assign a *compound* lucode. Brief 24 confirmed the
compound `code` column isn't positionally derived; the serial `lucode`
is the join key, looked up via `lulc_crosswalk.csv`.

**The options:**

| Option | Description | Pro | Con |
|---|---|---|---|
| A. Preserve NLUD + tree-canopy | When converting pixel P to food forest, find the compound lucode where (NLCD=41, NLUD=P.NLUD, tree=P.tree). Use that compound code. | Least presumptuous — only the land cover changes; land use and canopy state preserved | Multiple rows can match if NLUD/tree-canopy bins have multiple variations; need a tie-breaker |
| B. Use a "default food forest" compound code | Pick one specific compound lucode that represents an idealized food forest (NLCD=41, NLUD=managed natural, tree=high). Use it for all conversions. | Simpler — single lookup, no per-pixel logic | Loses pixel-specific context; converts a residential street to the same compound code as a converted woodland |
| C. Domain-specific mapping rules | Build a small rule table: "if pixel is residential developed, food-forest target is X; if pixel is commercial developed, food-forest target is Y" | Reflects real urban-design intuition | Adds another layer of methodology decisions, each of which needs justification |

**The decision: Option A with explicit tie-breaker.**

Rationale: Option A is the most faithful to the compound framework's
intent — the compound lucode captures multiple signals; conversion
should change only the signal that's actually changing. Option B
flattens valuable spatial information. Option C adds methodology weight
the project hasn't earned.

**The tie-breaker.** Brief 24 found 820 of the 1,984 possible compound
lucodes are actually present in NatCap's SA raster. For some (NLCD,
NLUD, tree-canopy) tuples, multiple `lucode` rows may exist in the
crosswalk that match the target NLCD with the source's NLUD + tree-bin.
When that happens, prefer the row marked `is_realistic_to_create = True`
or `is_realistic_to_paint = True` (per Brief 24's crosswalk inventory).
If both flag columns disambiguate, use the first matching row by
`lucode` ascending.

**Edge case.** When the source pixel's NLUD or tree-canopy combination
doesn't have a matching row for the target NLCD (e.g., converting an
NLCD=23 developed pixel where the specific (NLUD, tree-canopy) combo
doesn't appear with NLCD=41), fall back to a "default food forest"
compound lucode chosen at integration time — flagged as `DEFAULT_FF_LUCODE`,
`DEFAULT_GI_LUCODE`, `DEFAULT_HD_LUCODE` in config. Document the choice.

**For NatCap conversation:** "When converting pixels, we preserve the
NLUD and tree-canopy signals — only the NLCD changes. Does that match
how you'd model the transition? Should newly-planted food forests
inherit the source pixel's NLUD or start with a 'transitional' code
instead?"

### Decision 3: Integration sequence

**The situation.**

Four pieces of NatCap data to adopt: compound LULC raster, UCM table,
UNA table, Carbon table. Plus optionally the block-group AOI. These
pieces are intertwined (the tables are keyed on the LULC raster's
lucodes) but not all at once (each model uses its own table).

**The options:**

| Option | Description | Pro | Con |
|---|---|---|---|
| Big-bang | Adopt LULC + 3 tables + AOI in one brief | Single commit, single baseline regenerate | High blast radius; harder to isolate which change caused which output shift |
| LULC-first then per-model | Brief 1: adopt LULC raster + crosswalk, route through reduction to existing per-NLCD tables (no metric change). Brief 2: UCM table. Brief 3: UNA table. Brief 4: Carbon. Brief 5 (optional): AOI. | Each brief is testable; revertable; baseline regenerate is isolated | More briefs; intermediate state has compound LULC routing through per-NLCD tables (works but inelegant) |
| All-models, no-AOI | Brief 1: adopt LULC + 3 tables together. Brief 2 (optional): AOI. | Avoids intermediate "compound LULC + per-NLCD tables" state; all model changes land together | Larger blast radius than per-model; hard to bisect output shifts to a single model |

**The decision: LULC-first then per-model.**

Rationale: Each brief is independently testable and revertable. The
intermediate state (compound LULC routing through per-NLCD tables) is
inelegant but functional — the LULC adoption is the foundational change,
proving the reprojection + crosswalk routing works; then each model
table swap is a clean atomic update on top.

**The brief sequence:**

| Brief | Scope | Output impact |
|---|---|---|
| 27 | ✅ Done 2026-05-24. Adopt compound LULC raster + crosswalk. Reproject to 5070 + clip. Add `lulc_crosswalk.csv` loading. Add `DEFAULT_FF/GI/HD_LUCODE` config (1310 / 122 / 341, picked from `is_realistic_to_create=yes` rows by descending frequency). Route compound lucodes through to existing per-NLCD tables via crosswalk's `nlcd` column. | Minimal — 97.91 % pixel-wise agreement with prior `land_use_2021_sa.tif`; SA baselines drift <0.5 % on every headline metric (mean_hm, mean_cn, flood_reduction, runoff_acre_feet, nature_access_pct). MN untouched. |
| 28 | ✅ Done 2026-05-24. Switched SA UCM biophysical table to `ucm__nlcd_nlud_tree.csv`. SA UCM consumers now index the compound LULC raster directly (UFR + UNA still route through compound→NLCD reduction pending Briefs 29-30). Köppen-BSh tuning retired (file preserved on disk for historical reference). | Substantial — `baseline_hm` 0.2866 → 0.3937 (+37%) because the compound table captures tree-canopy variation on developed land that the per-NLCD table couldn't. `cooling_energy_savings_usd` dropped 77-86% across SA scenarios (mechanically explained — see DESIGN_NOTES.md). MN untouched (0 divergences across 20 MN baselines). |
| 29 | ✅ Done 2026-05-24. Switched SA UNA biophysical table to `una__nlcd_nlud_tree.csv` (1,984 rows; `urban_nature` ∈ {0.0, 0.5, 1.0} at 960/48/976). SA UNA consumers now index the compound LULC raster directly via a new `scenario_lulc_una` view threaded through `evaluate_scenario`. The previous Python-dict-iteration lookup over `URBAN_NATURE_PROPORTION` was replaced with a vectorized `urban_nature_arr[scenario_lulc_una]` indexed read. Only Carbon still routes through compound→NLCD reduction (pending Brief 30). | Modest — SA baseline `nature_access_pct` 89.7 → 94.2 (+5.0%, +4.5 pp); baseline people-with-access 1,710,167 → 1,794,653 (+84,486). Random-strategy scenarios shifted ~0.3–6 pp depending on whether they remove urban-nature pixels (high-density: +5.9 pp) or add saturated ones (food_forest already near 100%). Undersupply-focused + balanced placements also shift downstream metrics because the baseline UNA raster feeds suitability weights. MN untouched (0 value divergences across 20 MN baselines — only the new `scenario_lulc_una__md5` field added). |
| 30 | ✅ Done 2026-05-25. Switched SA Carbon to NatCap's four-pool stock framework via `carbon__nlcd_nlud_tree.csv` (1,984 rows × 27 cols; pools c_above/c_below/c_soil/c_dead in tons C/ha). SA Carbon consumers index `cooling_lulc_compound` directly via a new `scenario_lulc_carbon` view threaded through `evaluate_scenario`. Field rename: `carbon_tons_co2_yr` → `carbon_tons_co2` (unified key; semantics differ per city — annual flow MN, one-time stock SA). Dollar metric reframe: `avoided_carbon_cost_usd` → `carbon_value_usd` with city-conditional dashboard label ("Avoided Carbon Cost"/yr for MN, "Carbon Storage Value" one-time for SA). Methodology matches NatCap's Vibrant Land (Guerry et al. 2023); `EPA_SOCIAL_COST_CARBON=$190/t` kept untouched (EPA 2023 vintage vs Vibrant Land's IWG 2021 $53/t — methodology aligns, SC-CO2 vintage differs intentionally). | Substantial — SA Carbon stock numerically ~30× the prior annual proxy (category-error correction, not a value shift). E.g. SA food_forest_random: 65,264.9 t CO2/yr → 1,936,072 t CO2 stock; green_infrastructure_balanced: 37,294.3 → 4,375,912; high_density now shows negative stock (–849,262, nature loss) where the prior proxy clipped at $0. MN baselines unchanged (zero value divergence across 20 baselines — only the new `scenario_lulc_carbon__md5` field added; hash matches `scenario_lulc__md5` for MN). Order-of-magnitude check vs Vibrant Land's 340,000 t citywide reference: within plausible bounds given different AOI extent + "full conversion" definition. |
| 31 (optional) | Switch SA AOI from Bexar County bbox to `acs_block_groups_3857.gpkg` block-group polygons. | Affects per-block-group reporting; doesn't change global metrics. |

Each brief: schema bump, MN-untouched baseline regeneration (SA only),
WHATS_NEW entry, NATCAP_ALIGNMENT.md row update, CITY_PARITY.md row
flip.

---

## Conversion semantics in compound LULC

After Brief 27 adopts the compound LULC framework, the meaning of
"convert 30% to food forest" shifts slightly. Worth being explicit:

**Pre-integration (current):** All developed pixels (NLCD 21/22/23)
are converted by setting their lucode to 41 (food forest NLCD).
Every converted pixel has the same lucode.

**Post-Brief-27 (compound framework):** All developed pixels (where the
crosswalk row's `nlcd` is 21/22/23) are convertible. For each one,
Decision 2's lookup rule assigns a compound lucode based on its
existing NLUD + tree-canopy. Converted pixels can have different
compound lucodes depending on their pre-conversion context.

**User-facing change:** None — the slider semantics are identical;
the underlying pixel selection is just slightly more nuanced.

**Diagnostic change:** Placement strategy effects may shift slightly
because compound lucode variation creates more variation in biophysical
properties. Brief 9's placement-strategy diagnostic may want re-running
after Brief 28-30 land.

---

## What this plan does NOT decide

- **Exact `DEFAULT_FF/GI/HD_LUCODE` values.** Chosen at Brief 27 execution
  time after surveying the crosswalk's "realistic to create" flags.
- **Whether the four-pool Carbon framework also rolls out to MN.** Out of
  scope for this plan (which is SA-focused); MN Carbon stays at single-
  rate proxy until NatCap shares MN-equivalent tables or until methodology
  parity is independently prioritized.
- **AOI switch timing.** Brief 31 is optional; could be deferred indefinitely
  if Bexar County bbox keeps working.

---

## Open questions for NatCap

Bundled with the existing NATCAP_COLLABORATION.md open questions; not
duplicating here. The integration-specific ones are:

1. Are we right that EPSG:5070 (Conus Albers) is the appropriate CRS for
   the prototype's area-based metrics, even though your SA data ships in
   EPSG:3857?
2. Is the conversion-mapping rule (preserve NLUD + tree-canopy; change
   NLCD) consistent with how you'd model land-cover transitions?
3. Should newly-converted food-forest pixels inherit the source's NLUD
   code or start with a "transitional" NLUD code?
4. The compound `code` column in `lulc_crosswalk.csv` doesn't appear
   positionally encoded — is there an undocumented scheme, or is `lucode`
   genuinely the only intended join key?

---

## Maintenance

Update when:

- An integration brief lands (mark its row Done in the brief sequence)
- A decision changes (rare — would need re-justification)
- A new SA NatCap data file arrives that affects the plan
- An open question is answered by NatCap

Pair updates with `NATCAP_COLLABORATION.md` and `CITY_PARITY.md` when
relevant.

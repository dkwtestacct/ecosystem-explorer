# Ecosystem Explorer — Current Capabilities

Urban land-use tradeoff prototype · San Antonio + Minneapolis · canonical-InVEST-aligned engine with export handoff

## Cities
- **San Antonio, TX** — full spatial workflow: ownership/parcel-derived filters, council districts, NatCap reference scenarios
- **Minneapolis, MN** — scenarios + models; no ownership layer
- City switcher

## Scenario building
- Convert a selected % of developed land
- Allocate converted land among **green infrastructure / food forest / high-density development**
- Quick Start presets: Balanced / Green Infrastructure / Food Forest / High Density
- Placement strategies: random · prioritize flood-prone areas · prioritize hot areas near buildings · prioritize areas with unmet nature demand
- Editable implementation costs
- City-specific carbon / valuation assumptions

## Spatial targeting
- Select regions: council districts or census tracts; conversions confined to the selected area when active
- Always excluded: buildings, roads, existing natural land
- San Antonio ownership filters: None / Public land / Vacant land / Vacant + public / School land / College / university land / Custom
- Custom ownership classes: city / county / state-federal / school / university, plus vacant-only overlay
- *Caveat:* school and university classes are planning-screen filters, not title-verified ownership
- Eligibility breakdown: selected area → developed land → eligible land → converted acres
- **School-related scenarios** — restrict conversions to school-related parcels where identified, then evaluate nature access, **children's nature access**, cooling, and validated Urban Mental Health outcomes

## Outcome models
- **Urban Cooling** — temperature change, °F cooling, cooling energy savings
- **Urban Flood** — flood retention, runoff volume, flood damage avoided where valuation inputs exist
- **Carbon** — carbon storage / sequestration change, carbon value
- **Urban Nature Access** — nature access %, nature exposure / NDVI indicators
- **Urban Mental Health** — validated InVEST UMH estimates of preventable depression/anxiety cases and avoided costs
- **Children's Nature Access** — under-18 share of nature access (Census 2020 PL 94-171 under-18; access share child-weighted, supply stays on total pop)
- **Food** — annual production, people fed
- **Cost-effectiveness** — $/acre-foot runoff · $/°F cooling · $/1,000 people fed

## Scenario discovery
- **Citywide surrogate search** — predicted suggestions; apply to verify with full engine
- **Selected-area full-engine search** — best tested mixes under active region/ownership filters; not guaranteed global optima
- Goal weights / minimum-target thresholds
- Results table: rank · mix · score · converted acres · cooling · flood · carbon · food · cost · apply

## NatCap reference scenarios
- San Antonio published reference scenarios: baseline + food-forest / urban-ag variants
- Side-by-side published values for temperature, carbon stock, and derived carbon value
- Baseline shown as absolutes; alternatives shown as change from NatCap baseline
- Clearly labeled as displayed NatCap values, not fully recomputed Explorer scenarios
- Metrics that cannot be recomputed from available inputs are flagged

## Compare & analyze
- Tradeoff-space scatter plot
- Current + saved scenario comparison table
- CSV download
- Neighborhood breakdown
- Selected-region impact table: selected-area vs citywide

## Provenance / honesty surface
- Scenario source: Explorer-generated / NatCap reference / optimizer-suggested / selected-area optimized
- Validation status shown at scenario and per-metric levels
- Scenario audit records source, area, ownership filter, placement, seed, validation state
- Locked badge vocabulary: NatCap published value · ≈ NatCap method · ≈ Aligned method · Prototype
- UI principle: visible text = what · tooltip/expander = how · docs = why

## Validation / rigor
- Regression / snapshot harness — 40 byte-identical baseline checks across city / scenario / strategy outputs (guards drift)
- Canonical-InVEST parity checks where comparable inputs exist; baseline engine-verified vs canonical InVEST
- Urban Mental Health: per-pixel parity vs canonical InVEST 3.19.0 — algorithm validated; NDVI input is a synthetic land-cover proxy
- Per-metric validation badges (locked vocab): NatCap published value · ≈ NatCap method · ≈ Aligned method · Prototype (temperature & carbon = published; cooling / nature / flood = aligned-method)
- Subset-invariant tests: converted pixels stay within eligible / region / ownership masks
- Static lints for key UI and provenance invariants

## Export / handoff
- Export current San Antonio scenario as a runnable canonical InVEST 3.19.0 input bundle
- Includes rasters, AOIs, biophysical tables, per-model args, metadata
- CSV download for comparison table

## Not yet / on the radar
- **School-point targeting** — use individual school locations to prioritize interventions near schools (beyond today's school-land filter), and make children's nature access an optimization target rather than only a reported metric
- AlphaEarth-derived land-cover inputs
- Pixel-level spatial optimization
- NDR / nutrient retention, pending canonical inputs

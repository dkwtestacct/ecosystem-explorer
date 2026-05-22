# UNA Implementation Notes — Parameter Choices

Working log of InVEST UNA parameter choices for the Minneapolis prototype. For
each parameter: the options considered, the value chosen, and why. To revisit
as the implementation evolves and as collaborators provide feedback.

## `urban_nature_demand` (m²/capita)

The per-capita supply standard. Pixels where supply ≥ demand are "adequately
supplied"; the headline metric reports the share of population that meets this.

**Options considered:**
- 250 m²/capita — InVEST default (used in Phase 1, gave 9.5% adequately supplied)
- 16.7 m²/capita — NatCap SA-study value
- Other published thresholds (e.g., WHO 9 m²/capita) — not investigated

**Chosen: 16.7 m²/capita.**

NatCap-validated for the SA Urban Agriculture project. No formal MN benchmark
exists, so adopting a NatCap-applied value is more defensible than the InVEST
generic default. Applicability of an SA value to MN's different urban context
is an open question.

## `search_radius_mode`

How the search radius is configured.

**Options considered:**
- `'uniform radius'` — single value for all nature classes
- `'urban_nature_table'` — per-class radii from the biophysical table
- `'population_group_radii_table'` — per-population-group radii

**Chosen: `'uniform radius'`.**

Matches NatCap SA-study practice and simplifies parameter exposition. Per-class
radii would be appropriate if the biophysical table had well-justified per-class
values for the MN context; it doesn't.

## `search_radius` (m)

The radius defining what nature is "reachable."

**Options considered:**
- 1000 m — used in Phase 1 (roughly 12-min walk)
- 800 m — NatCap SA-study value (roughly 10-min walk)
- 500 m — common in walkability literature (5-min walk)

**Chosen: 800 m.**

NatCap SA-study practice. Reasonable urban-planning walking distance.

## `decay_function`

How reachability falls off with distance from nature.

**Options considered:**
- `'dichotomy'` — binary in/out within the radius
- `'exponential'` — exponential decay (InVEST default)
- `'gaussian'` — gaussian decay

**Chosen: `'dichotomy'`.**

Matches NatCap SA-study practice. Simpler to explain than exponential.
Exponential is more theoretically grounded; revisit if dichotomy produces
output that's too coarse for the prototype's use case.

## `aggregate_by_pop_group`

Whether to compute aggregates separately for population subgroups.

**Options considered:**
- `False` — single aggregate over all population
- `True` — per-subgroup aggregates (requires `population_group_radii_table`)

**Chosen: `False`.**

The prototype's population raster doesn't have subgroup breakdowns. Subgroup
analysis isn't a current prototype goal.

## `urban_nature_lulc_table`

The biophysical table mapping LULC classes to `urban_nature` values and
per-class search radii.

**Options considered:**
- InVEST MN sample table (`LULC_attribute_table_UNA.csv`, already in repo)
- A NatCap-curated MN-specific table (existence unknown)
- A custom table designed for the prototype

**Chosen: InVEST MN sample table.**

Already in repo, already validated in Phase 1 comparison work. If a
NatCap-curated MN-specific table exists, would warrant adoption.

## `population_raster_path`

The population raster used as demand input.

**Options considered:**
- Existing Census 2020 raster (`pop_count_raster`, already in repo)
- A NatCap-provided alternative (existence unknown)

**Chosen: Existing Census 2020 raster.**

Already in repo. Standard data source.

## Population denominator (in headline reporting)

How to handle the 56.6% of MN population sitting on cooling-LULC nodata pixels
that InVEST cannot model.

**Options considered:**
- Report % of modelable-extent population (InVEST native output)
- Report % of total city population (count off-LULC residents as inadequate)
- Report both numbers separately

**Chosen: % of modelable-extent population, with tooltip clearly framing the
denominator.**

Matches InVEST's native output. Tooltip honesty addresses the denominator
ambiguity. Whether this is the right framing for a planner-facing dashboard
is open.

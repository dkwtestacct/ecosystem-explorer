# Urban Ecosystem Tradeoff Explorer

**Audience:** External
**Status:** Current
**Use this for:** First stop — what the app is, the live link, how to run it, where to read more
**Do not use this for:** Methodology detail (→ REFERENCE.md) or internal direction (→ docs/internal/)
**Source of truth for:** The public entry point and documentation index

---

A Streamlit prototype for exploring tradeoffs in urban land-use scenarios — how different allocations across green infrastructure, food forests, and high-density development affect flood risk, cooling, food production, mental health, carbon, and cost.

Built in collaboration with NatCap (Stanford). Currently supports Minneapolis (downtown) and San Antonio.

**Live app:** [ecosystem-explorer.streamlit.app](https://ecosystem-explorer.streamlit.app/)

---

## Running locally

```bash
git clone https://github.com/dkwtestacct/ecosystem-explorer.git
cd ecosystem-explorer
pip install -r requirements.txt
streamlit run app.py
```

The first run trains the surrogate model (a few seconds) and loads the lookup table. Subsequent runs are faster.

---

## Where to start

Different paths through the docs depending on what you need:

- **Using the app and curious about a metric** → `REFERENCE.md`
- **Understanding how the app is built** → `ARCHITECTURE.md`
- **Picking up the project after a break** → this README, then `ARCHITECTURE.md`, then `DESIGN_NOTES.md`
- **NatCap collaborator** → `NATCAP_ALIGNMENT.md`, then `CITY_PARITY.md`
- **Future Claude session** → `CLAUDE.md`, then `DESIGN_NOTES.md`, then `ARCHITECTURE.md`

---

## Documentation index

### Core docs

| File | Purpose |
|---|---|
| `README.md` | This file. Repo overview and doc index. |
| `ARCHITECTURE.md` | Three-layer system overview (raster simulations → lookup table → surrogate). Read this to understand how the prototype is built. |
| `REFERENCE.md` | User-facing methodology. What each metric means, which model produced it, where the data comes from. |
| `DESIGN_NOTES.md` | Internal design decisions. Options considered, chosen, why. Audience: future Claude sessions and Daniel. |
| `CLAUDE.md` | Working principles for Claude sessions. |
| `SPEC.md` | Original design specification. |

### NatCap collaboration

| File | Purpose |
|---|---|
| `NATCAP_ALIGNMENT.md` | Per-surface alignment status against NatCap canonical. Six tables (methodology, parameters, AOI, research directions, vocabulary). The methodology view. |
| `NATCAP_COLLABORATION.md` | Running conversation log. Asks, inferred priorities, gaps, decisions made without confirmation, open questions. The process view. |
| `CITY_PARITY.md` | Per-city alignment matrix. How closely the prototype matches NatCap's published configurations for each specific city. The city view. |
| `SA_INTEGRATION_PLAN.md` | Multi-brief plan for adopting NatCap's curated SA dataset (compound LULC + three biophysical tables). Foundational CRS/extent, conversion-mapping, and sequencing decisions ahead of Briefs 27+. |

### Data

| File | Purpose |
|---|---|
| `DATA_INVENTORY.md` | Every external data source the prototype consumes. Per-city, per-category, with provenance. |

### Investigations and analysis

| File | Purpose |
|---|---|
| `INVEST_PLACEMENT.md` | Per-InVEST-model placement-strategy analysis. |
| `PLACEMENT_STRATEGY_DIAGNOSTIC.md` | Empirical measurements of placement-strategy effect sizes (Brief 6 baseline + Brief 9 reformulation). |
| `ALPHAEARTH_FEASIBILITY.md` | Research on AlphaEarth Foundations as future LULC source. |
| `UNA_DIVERGENCE_CASE_STUDIES.md`, `UNA_METHODOLOGY_CROSS_CHECK.md`, `UNA_QUALITY_SCORE_SENSITIVITY.md`, `UNA_LULC_INVESTIGATION.md` | UNA-specific investigations leading to the "Nature Quality Score temporarily removed" decision. |

### Summary documents

| File | Purpose |
|---|---|
| `SUMMARY.md` (+ `.docx`, `.pdf`) | High-level prototype summary. |

---

## Repo layout

```
ecosystem_explorer/
├── app.py                    # Streamlit app + all scenario evaluation
├── config.py                 # Per-city configuration (paths, scalars)
├── surrogate.py              # Random-forest training + Pareto filtering
├── precompute_scenarios.py   # Generates the lookup table
├── verify_baselines.py       # Regression test gate (40 baselines)
├── *.md                      # Documentation (see index above)
├── data/                     # Source rasters, biophysical tables, etc.
├── tests/                    # Baselines + tests
├── analysis/                 # Diagnostic outputs (e.g., placement_diagnostic)
└── download_*.py, process_*.py  # Data pipeline scripts
```

`DATA_INVENTORY.md` has the detailed per-script breakdown of what each `download_*.py` / `process_*.py` produces.

---

## Status

Prototype, actively developed. The NatCap collaboration is ongoing — see `NATCAP_COLLABORATION.md` for the running log of asks, gaps, and open questions.

Active workstream as of 2026-05-24: alignment with NatCap's curated San Antonio dataset (compound NLCD+NLUD+tree-canopy LULC framework + matched biophysical tables for UCM/UNA/Carbon). Integration is multi-brief; see NATCAP_COLLABORATION.md for queue.

# Urban Ecosystem Tradeoff Explorer

**Audience:** External
**Status:** Current
**Use this for:** First stop — what the app is, the live link, how to run it, where to read more
**Do not use this for:** Methodology detail (→ REFERENCE.md) or internal direction (→ docs/internal/)
**Source of truth for:** The public entry point and documentation index

---

A Streamlit tool for exploring tradeoffs in urban land-use scenarios — how reallocating developed land across green infrastructure, food forests, and high-density development affects flood risk, cooling, food production, mental health, carbon, and cost.

It is built on a model engine validated against canonical InVEST (the core urban models match per-pixel), displays NatCap's published San Antonio project values as labeled reference points, lets you explore new scenarios beyond the fixed project set, and exports promising candidates back to canonical InVEST for a full run. Built in collaboration with NatCap (Stanford). Currently supports Minneapolis (downtown) and San Antonio.

For what each number means — and what's *validated* vs *displayed* vs *exploratory* — see **REFERENCE.md**.

**Live app:** [ecosystem-explorer.streamlit.app](https://ecosystem-explorer.streamlit.app/)

---

## Running locally

```bash
git clone https://github.com/dkwtestacct/ecosystem-explorer.git
cd ecosystem-explorer
pip install -r requirements.txt
streamlit run app.py
```

The first run trains the surrogate model (a few seconds) and loads the lookup table; later runs are faster.

---

## Start here

Pick the path that matches what you need:

- **Using the app, wondering what a metric means or how grounded it is** → `REFERENCE.md`
- **Understanding how the system is built** → `ARCHITECTURE.md`
- **The big picture and where it's headed** → `STRATEGY.md`
- **NatCap collaborator checking alignment** → `NATCAP_ALIGNMENT.md`, then `CITY_PARITY.md`
- **Setting up or running the validation harness** → `CONTRIBUTING.md`
- **Picking the project back up after a break** → this file, then `ARCHITECTURE.md`

---

## Documentation

Two docs are **external** — written to stand on their own:

- **`README.md`** (this file) — entry point and map
- **`REFERENCE.md`** — what each dashboard number means, where the data comes from, and how the models align with InVEST/NatCap (with caveats)

Everything else is **internal** working documentation, currently at the repo root, organized here by role:

- **Internal** — `STRATEGY.md` (north star) · `ARCHITECTURE.md` · `DESIGN_NOTES.md` (decision log) · `DATA_INVENTORY.md` · `NATCAP_ALIGNMENT.md` (validation status) · `CITY_PARITY.md` (per-city parameter parity) · `OPEN_QUESTIONS.md` (current blockers) · `NATCAP_COLLABORATION.md` (collaboration log) · `docs/internal/DEMO_AND_COLLABORATION.md` (demo/meeting runbook)
- **Developer** — `CONTRIBUTING.md` (setup + validation harness). `CLAUDE.md` also at the repo root, where tooling expects it.
- **Research (feasibility + investigation)** — `ALPHAEARTH_FEASIBILITY.md`, `INVEST_PLACEMENT.md`, `PLACEMENT_STRATEGY_DIAGNOSTIC.md`, and the four `UNA_*.md` notes
- **Archive (superseded / historical)** — `HISTORY.md`, `SPEC.md` (original spec), `SUMMARY.md`, `SA_INTEGRATION_PLAN.md`

Every doc carries a status header (Audience · Status · Source of truth for) declaring its role, so you can tell at a glance whether it's current truth or a historical record.

---

## Repo layout

```
app.py                   Streamlit app + scenario evaluation
config.py                Per-city configuration (paths, scalars)
surrogate.py             Random-forest surrogate + optimizer
precompute_scenarios.py  Lookup-table generation
verify_baselines.py      Regression test gate
docs/                    Documentation (see map above)
data/                    Source rasters, biophysical tables, config
```

---

## Status

Prototype, actively developed. The NatCap collaboration is ongoing — see `NATCAP_COLLABORATION.md` for the running log and `OPEN_QUESTIONS.md` for current blockers.

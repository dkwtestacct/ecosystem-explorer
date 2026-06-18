# Ecosystem Explorer

**Audience:** External
**Status:** Current
**Use this for:** First stop — what the app is, the live link, how to run it, where to read more
**Do not use this for:** Methodology detail (→ REFERENCE.md) or internal direction (→ docs/internal/)
**Source of truth for:** The public entry point and documentation index

---

A prototype of a reusable workflow for exploring tradeoffs in urban land-use scenarios — how reallocating developed land across green infrastructure, food forests, and high-density development affects flood risk, cooling, food production, mental health, carbon, and cost. It ships as a Streamlit app, but the pattern is the point: **scenario definition → spatial placement → model-aligned evaluation → validation/provenance → comparison → export/handoff.**

It is built on a model engine validated against canonical InVEST (the core urban models match per-pixel), displays NatCap's published San Antonio project values as labeled reference points, lets you explore new scenarios beyond the fixed project set, and exports promising candidates back to canonical InVEST for a full run. Currently supports Minneapolis (downtown) and San Antonio.

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
- **Understanding how the system is built** → `docs/internal/ARCHITECTURE.md`
- **The big picture and where it's headed** → `docs/internal/STRATEGY.md`
- **NatCap collaborator checking alignment** → `docs/internal/NATCAP_ALIGNMENT.md`, then `docs/internal/CITY_PARITY.md`
- **Setting up or running the validation harness** → `docs/dev/CONTRIBUTING.md`
- **Picking the project back up after a break** → this file, then `docs/internal/ARCHITECTURE.md`

---

## Documentation map

Each question has one source of truth — if you're unsure where something belongs, it goes in the doc that owns that question, and nowhere else.

| Question | Source of truth |
|---|---|
| What is this app? | `README.md` |
| What do the dashboard metrics mean? | `REFERENCE.md` |
| What is the internal strategic framing? | `docs/internal/STRATEGY.md` |
| How does the system work? | `docs/internal/ARCHITECTURE.md` |
| Why was a design choice made? | `docs/internal/DESIGN_NOTES.md` |
| How aligned is this with InVEST/NatCap? | `docs/internal/NATCAP_ALIGNMENT.md` |
| How aligned is each city configuration? | `docs/internal/CITY_PARITY.md` |
| What is the running log of NatCap asks, gaps, and decisions? | `docs/internal/NATCAP_COLLABORATION.md` |
| What data exists? | `docs/internal/DATA_INVENTORY.md` |
| What is still unresolved? | `docs/internal/OPEN_QUESTIONS.md` |
| What do I say in a demo or meeting? | `docs/internal/DEMO_AND_COLLABORATION.md` |
| How do I set up and run validation/contribution workflows? | `docs/dev/CONTRIBUTING.md` |
| How should a Claude session work in this repo? | `CLAUDE.md` (repo root) |
| What did I investigate historically? | `docs/research/` |
| What is superseded history? | `docs/archive/` |

---

## Repo layout

```
app.py                   Streamlit app + scenario evaluation
config.py                Per-city configuration (paths, scalars)
surrogate.py             Random-forest surrogate + optimizer
precompute_scenarios.py  Lookup-table generation
verify_baselines.py      Regression test gate
validation/              Canonical-InVEST parity comparators
diagnostics/             Standalone investigation + diagnostic scripts
scripts/                 Data-pipeline and utility scripts
docs/                    Documentation (see map above)
data/                    Source rasters, biophysical tables, config
CLAUDE.md                Operating manual for Claude sessions (repo root)
```

---

## Status

Prototype, actively developed. The NatCap collaboration is ongoing — see `docs/internal/NATCAP_COLLABORATION.md` for the running log and `docs/internal/OPEN_QUESTIONS.md` for current blockers.

# CLAUDE.md trim — content map

**Audience:** Internal
**Status:** In review — `[VERIFY]` 1–5 resolved against the live `CLAUDE.md`; awaiting approval before rewrite
**Use this for:** Driving the CLAUDE.md trim from a ~700-line mini-encyclopedia to a lean agent operating manual
**Do not use this for:** Current Claude-Code operating instructions — this doc is scaffolding for the trim, not the manual itself
**Source of truth for:** Keep/cut decisions, contract-preservation discipline, and inbound-anchor inventory for the trim

---

**Transient scaffolding.** Delete after the trim + cross-ref-fix pass land. Same discipline as the other content-maps.

**Purpose:** cut CLAUDE.md from a ~700-line mini-encyclopedia (current: 710 lines) to a lean **agent operating manual** — "how should an AI assistant work in this repo?"

**Overriding constraint:** CLAUDE.md is CC's *own* operating manual. The trim must **protect the behavioral contract while cutting the encyclopedia.** Cut hard, not into bone. **Stays at repo root** (Claude Code auto-discovery is root-only — do not move to docs/dev/).

---

## KEEP — the durable contract + orientation (the load-bearing core)

- **Behavioral contract** (the rails that have governed every commit): Phase-0 investigate-first · hard stop-and-report before any commit · single-commit-per-concern, **no `Co-Authored-By`** · bump `SCENARIO_SCHEMA_VERSION` on any math/output change · `verify_baselines.py` 40/40 as the regression gate · the single-home doc contract (one question, one home).
- **What not to break (invariants, stated as rules — not explained):** the CRS assertion (every `rasterio.open` asserts the city's canonical CRS or crashes loudly) · per-city scalars are config-driven · the live-override-on-load baseline mechanism (don't hardcode-and-trust).
- **Agent read-order:** short — "read this, then ARCHITECTURE for the system, then the task's owning doc." Points to **README's source-of-truth table** for the full doc-map (do not duplicate the 15-row table here).
- **Key commands:** `streamlit run app.py` · `verify_baselines.py [--update]` · `precompute_scenarios.py`.
- **Where current truths live:** pointer to README doc-map + the single-home matrix.

---

## MOVE OUT / DELETE

| Current content | Disposition |
|---|---|
| Baseline-values table (`BASELINE_HM`, `BASELINE_RUNOFF_ACRE_FEET`, …) | **Delete + point.** Recomputed at load ("documentation only"); mechanism is ARCHITECTURE §3. Don't enshrine volatile values. |
| Cost defaults (values) | **Delete + point to `config.py`** (the source of truth). |
| Long data inventories / data-path tables / file lists | → **DATA_INVENTORY** (already mapped to receive them). |
| Old city-config descriptions | → **DATA_INVENTORY / CITY_PARITY / config.py**. |
| Detailed methodology *explanations* (e.g. why equal-area matters, SA flood-CN derivation) | → **REFERENCE / DESIGN_NOTES**. Keep the *invariant* as a rule (see KEEP); cut the prose. |
| History / Brief-chronology | → **HISTORY**. |
| Embedded doc-map / full source-of-truth table | → point to **README** (one canonical map). |

**The keep/cut line for methodology:** invariant the agent must preserve → KEEP as a rule. Explanation of why → MOVE. A blind strip of methodology-adjacent text would take the rules with the prose.

---

## Resolved findings (`[VERIFY]` 1–5)

### `[VERIFY] 1` — Contract-item presence in CLAUDE.md (some are session conventions, NOT in doc)

| Contract item from the KEEP list | Currently in CLAUDE.md? | Action in the trim |
|---|---|---|
| Phase-0 investigate-first | ✅ Present ("investigate-first" — 2 hits) | preserve as durable rule |
| Stop-and-report before commit | ✅ Present (1 hit) | preserve |
| Bump `SCENARIO_SCHEMA_VERSION` on math/output changes | ✅ Present (in "Cached functions use path params as cache keys" coding-convention) | promote to a top-level rule (not buried in caching) |
| `verify_baselines.py` 40/40 regression gate | ✅ Present (2 hits) | preserve, surface as the gate |
| Per-city scalars config-driven ("No bare globals for city data") | ✅ Present in Coding conventions | preserve as a rule |
| Live-override-on-load baseline mechanism ("Dynamic baselines") | ✅ Present in Architecture notes section | preserve as a rule + point at ARCHITECTURE §3 for mechanism |
| **Single-commit-per-concern** | ❌ NOT in CLAUDE.md — session convention only | **ADD as a rule** in the trim |
| **No `Co-Authored-By`** | ❌ NOT in CLAUDE.md — session convention only | **ADD as a rule** |
| **Single-home doc contract** (one question, one home) | ❌ NOT in CLAUDE.md — emerged in the doc-suite refactor | **ADD as a rule** with pointer to README's source-of-truth table |
| **CRS assertion as a rule** (`_assert_raster_crs` at every `rasterio.open`) | ❌ NOT in CLAUDE.md as an explicit rule (only mentioned indirectly) | **ADD as a rule** with pointer to ARCHITECTURE §3 CRS handling |

**Five contract items currently live as session conventions, not durable doc text.** The trim must ADD them, not just preserve. Without this addition, the trim would silently strip the explicit Phase-0 / 40-of-40 / config-driven rules while *not gaining* the unwritten contract — net regression for the agent's operating context.

### `[VERIFY] 2` — Move-out destinations all exist

| Destination | Status | Receives |
|---|---|---|
| `DATA_INVENTORY` | live (and queued for refresh; map is committed at `/docs/internal/DATA_INVENTORY_CONTENT_MAP.md`) | data-path tables (CLAUDE.md lines 40–48 + line 54), file lists, city-config data descriptions |
| `HISTORY` | live (at `docs/archive/HISTORY.md` post-migration) | brief-chronology, retired-metric narratives |
| `CITY_PARITY` | live | MD5 / parity claims that CLAUDE.md currently carries in passing |
| `REFERENCE` | live (just rewritten at commit `8d0788e`) | methodology *explanations* (the why) |
| `DESIGN_NOTES` | live (rewrite queued; map committed) | per-decision rationale |
| `ARCHITECTURE` | live (just rewritten at commit `6819d7d`) | system mechanism (CRS handling, the three layers, caching) |
| `README` | live | the source-of-truth table (CLAUDE.md should point, not duplicate) |
| `config.py` | live (cost defaults moved there in the migration) | actual constants |

All destinations exist. No stranding risk.

### `[VERIFY] 3` — Baseline-values + cost-defaults are safe to delete-with-pointer

**Baseline values** (`BASELINE_HM`, `BASELINE_CN`, `BASELINE_NDVI`, `BASELINE_RUNOFF_ACRE_FEET`) — confirmed in code: live-recomputed inside `_load_city_runtime_state` (ARCHITECTURE §3) on every module load. The `CITIES[city]['baseline_*']` values are **documentation-only** (explicitly noted in the current CLAUDE.md and the live `config.py` comments). **Safe to delete from CLAUDE.md** + point readers at ARCHITECTURE §3 for the mechanism.

**Cost defaults** (`DEFAULT_COST_GI`, `DEFAULT_COST_FF`, `DEFAULT_COST_HD`) — confirmed in `config.py` post-migration (`from config import CITIES, DEFAULT_COST_GI, DEFAULT_COST_FF, DEFAULT_COST_HD` in `app.py`). **Safe to delete from CLAUDE.md** + point at `config.py` as the source of truth.

No value lives only in CLAUDE.md text without being either live-computed or config-sourced. No values get orphaned by deletion.

### `[VERIFY] 4` — Cited anchors: outbound from CLAUDE.md AND inbound TO CLAUDE.md

**Outbound cites at the line numbers the brief named** — these preserve content + the cite, not a CLAUDE.md anchor:

| Line | What's cited (preserve the cite) |
|---|---|
| :86 | "See docs/internal/NATCAP_ALIGNMENT.md `Validated reference outputs (SA)`" |
| :194 | "Full breakdown in REFERENCE.md `Cross-city Heat Mitigation Index comparison`" (already name-fixed by commit `8d0788e`) |
| :378 | "Methodology documented in `docs/internal/NATCAP_COLLABORATION.md` question 12" |
| :710 | "`docs/internal/DESIGN_NOTES.md` ... and `docs/internal/NATCAP_COLLABORATION.md`" (Brief 30 pattern guidance) |

**Inbound CITES TO CLAUDE.md anchors — these anchor names must be preserved by the trim OR co-updated in the same commit:**

| CLAUDE.md anchor | Cited from |
|---|---|
| **`"Pure-variant helpers"`** | `app.py:1216`, `app.py:2559` (2 cites) |
| **`"Interface changes require auditing all consumers"`** | `app.py:1942` |
| **`"Buildings — typing"`** | `app.py:2193` |
| **`"OSM road exclusion"`** | `docs/internal/DESIGN_NOTES.md:203` |

**4 anchors in CLAUDE.md must survive the trim** — they're the only inbound-cited targets. Two of them ("Pure-variant helpers", "OSM road exclusion") describe coding-convention *invariants* the agent must know about — they belong in the trimmed CLAUDE.md as rules (KEEP). The other two ("Interface changes require auditing all consumers" = a session-convention guard rule; "Buildings — typing" = a coding-convention SA-specific note) are also rules, not encyclopedic prose — they survive.

If a rule's anchor name changes during the trim (e.g. consolidating "Pure-variant helpers" into a broader "Coding rules" subsection), update the 5 inbound cites in the same commit.

### `[VERIFY] 5` — Read-order accuracy for the final doc set

The trimmed read-order names docs as they'll be **after** the suite lands. Recommend exact phrasing:

> *"For any task: read this file → read `docs/internal/ARCHITECTURE.md` (system structure) → consult the task's owning doc per `README.md`'s **Documentation map** table (one question, one home)."*

Constraints:
- **No 15-row table inside CLAUDE.md** — that's README's job (one canonical map).
- **No doc-by-doc summaries** — agent reads each doc when needed.
- **ARCHITECTURE first after CLAUDE** because the system structure orients every subsequent read.
- **"task's owning doc"** is the runtime resolution — pulled from the table, not enumerated here.

---

## Mapping rows worth flagging

1. **Five contract items live as session conventions, not doc text** (per `[VERIFY] 1`). The trim must **ADD** them, not just preserve the existing 5. Without this, the trim is a silent regression for the agent's operating context. Single-commit-per-concern, no-Co-Authored-By, single-home-doc-contract, CRS-assertion-as-a-rule, and the explicit "verify_baselines 40/40 as the gate" framing all need explicit doc text.
2. **The four inbound-cited anchors define the minimum surface** the trim can't eliminate. Cutting "Pure-variant helpers" / "Interface changes require auditing all consumers" / "Buildings — typing" / "OSM road exclusion" would break 5 inbound cites. They survive (they're real coding-convention rules, exactly the shape the trimmed manual should hold). They may move into a single "Coding rules" subsection, but their anchor names need to be either preserved verbatim or co-updated in the same commit.
3. **Architecture-notes section (currently lines 224–355) contains both load-bearing invariants and encyclopedic prose** — needs row-by-row triage. The "Dynamic baselines" subsection is an invariant (KEEP); the long "City runtime state (`CityState` + `_load_city_runtime_state`)" prose is ARCHITECTURE §3 territory (MOVE). A clean split would shrink Architecture notes from ~130 lines to ~15.
4. **CLAUDE.md's current `_CARBON_IS_STOCK` discussion (in Coding conventions and elsewhere) duplicates REFERENCE §6 Carbon Sequestration framing** — strip from CLAUDE.md, point at REFERENCE §6. The invariant ("Carbon is per-city semantics; the flag drives card labels") is a rule the agent should know; the carbon-rates table + per-pool details are not. Same shape as the baseline-values disposition: keep the rule, cut the data.
5. **CLAUDE.md mentions the Wellbeing Score retirement, the Nature Quality Score retirement, the 2026-05-22 2SFCA restoration** — pure chronology. Goes to HISTORY (already exists post-migration). The agent doesn't need these for current operating context.

---

## Resolved decisions (bake in — LOCKED before rewrite)

- **Stays at repo root.** Claude Code auto-discovery is root-only. Do NOT move to `docs/dev/`.
- **ADD five missing contract items** (per `[VERIFY] 1`): single-commit-per-concern, no-Co-Authored-By, single-home-doc-contract, CRS-assertion-as-a-rule, 40-of-40 framing. These join the five that are already present, for a 10-item durable behavioral contract.
- **Preserve four inbound-cited anchor names** (per `[VERIFY] 4`): `"Pure-variant helpers"`, `"Interface changes require auditing all consumers"`, `"Buildings — typing"`, `"OSM road exclusion"`. Either keep verbatim or co-update the 5 inbound cites in the same commit.
- **Baseline-values + cost-defaults DELETE-with-pointer** (per `[VERIFY] 3`): all values are live-computed or config-sourced; no orphan risk. Pointers go to ARCHITECTURE §3 (mechanism) and `config.py` respectively.
- **Read-order is 3 docs only** (CLAUDE → ARCHITECTURE → task's owning doc per README). No 15-row table inside CLAUDE.md.
- **Preserve the 4 outbound cites** at lines 86 / 194 / 378 / 710 (the cite content survives — it just may move to a different position in the trimmed doc).
- **Trim is the LAST step** of the suite (closing step alongside README and the cross-ref sweep). Its move-outs depend on every other doc having settled first.

---

## Sequencing — last, with README

CLAUDE.md and README are the two orientation docs (agent / human). Both settle **last**, in the closing step, because their maps must reflect the final state of everything they point at and their move-outs need destinations that already exist.

**Closing step:** cross-ref sweep + **README + CLAUDE.md** together → batch push. Its own commit.

Full back-half order (with this trim slotted in):

```
DESIGN_NOTES rewrite (next-up, ahead of trio — lands §11 complete)
    ↓
NATCAP_ALIGNMENT refresh → CITY_PARITY refresh → NATCAP_COLLABORATION refresh
    ↓
HISTORY paired commits (Q12 + B2-deferral + DESIGN_NOTES brief narratives + chronology from CLAUDE.md)
    ↓
DATA_INVENTORY refresh (receives CLAUDE.md's data-path tables)
    ↓
OPEN_QUESTIONS refresh
    ↓
Cross-ref sweep + README reconciliation + **CLAUDE.md trim** ← closing step
    ↓
Batch push to origin
```

Each step is its own commit; held local; batch-pushed at the closing step.

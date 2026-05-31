# Cross-ref sweep — running checklist

**Audience:** Internal
**Status:** Running checklist — accumulates through the rewrite phase; cleared at the closing-step cross-ref sweep commit
**Use this for:** Tracking inbound anchor cites + per-doc reconcile items that the cross-ref sweep needs to resolve
**Do not use this for:** Per-decision rationale (→ DESIGN_NOTES.md), or the doc rewrites themselves (each owns its own content map)
**Source of truth for:** What still needs to update across the suite when the cross-ref sweep runs

---

**Transient scaffolding.** Delete after the closing-step cross-ref sweep + README + CLAUDE.md trim land. Same discipline as the content-map files.

**Purpose.** As the rewrite phase strips and renames anchors across the suite, inbound cites from other docs / code / CLAUDE.md drift stale. Rather than co-update them in every per-doc commit (seven scattered updates per rewrite), defer to one closing-step sweep — but record each broken cite the moment it's surfaced so it isn't lost.

The closing-step ordering (per the suite plan): `DESIGN_NOTES rewrite` → `NATCAP trio` → `HISTORY paired commits` → `DATA_INVENTORY` → `OPEN_QUESTIONS` → **cross-ref sweep + README + CLAUDE.md trim (closing step)**.

---

## 1. Inbound anchor cites — DESIGN_NOTES rewrite (committed 2026-05-30)

The §1–11 template restructure renamed most anchors (chronology stripped, dates stripped, brief numbers stripped). Seven inbound cites point to old anchor names:

| Source | Line | Old anchor cited | Replace with |
|---|---|---|---|
| `docs/internal/OPEN_QUESTIONS.md` | :160 | `DESIGN_NOTES.md "Brief B1"` | (B1 narrative routed to HISTORY) DESIGN_NOTES §3 (compound LULC) + §4 (conversion logic) + §8 (validation/provenance) **or** `../archive/HISTORY.md` "Brief narrative chronology — Brief B1" |
| `docs/internal/DATA_INVENTORY.md` | :482 | `DESIGN_NOTES.md "NLCD legacy vs Annual NLCD"` | DESIGN_NOTES §3.1 |
| `docs/internal/DATA_INVENTORY.md` | :484 | `DESIGN_NOTES.md "SA flood damage table — resolved (Path C, Brief 33)"` | DESIGN_NOTES §6.5 |
| `CLAUDE.md` | :163 | `DESIGN_NOTES.md "Brief 4 — \`cooling_f\` → \`temp_change_f\` sign-convention refactor"` | DESIGN_NOTES §10.1 |
| `CLAUDE.md` | :528 | `DESIGN_NOTES.md "SA Carbon four-pool framework adoption"` | DESIGN_NOTES §6.4 |
| `app.py` (code comment) | :1409 | `DESIGN_NOTES.md "Brief B — UMH NE kernel: Gaussian → buffer-mean"` | DESIGN_NOTES §6.3 (anchor preserved: `UMH validation against canonical InVEST 3.19.0`) |
| `app.py` (code comment) | :3288 | `DESIGN_NOTES.md "Brief B2 (revised)"` | DESIGN_NOTES §8.1 (or §8.1 + §11.4 for the deferred Match/Diverged) |

**One inbound cite survives intact:** `app.py:4013 → DESIGN_NOTES.md "Lookup-overlay safety contract"` — anchor preserved at §4.4. No edit needed.

**CONTRIBUTING.md anchor preserved:** `docs/dev/CONTRIBUTING.md:70 → DESIGN_NOTES.md "UMH validation against canonical InVEST 3.19.0"` — the §6.3 anchor was explicitly preserved (the rewrite carries a `> **Anchor preserved:** ...` callout). No edit needed.

---

## 2. Per-doc reconcile items surfaced during the rewrite phase

### 2.1 DATA_INVENTORY §9.3 — stale UNA-demand claim (surfaced 2026-05-30)

**Current text** (line 383): *"Shared per-city scalars: demand `UNA_DEMAND_M2_PER_CAPITA = 16.7` (constant in app.py — per-city values match in current configs)."*

**Why stale.** Doubly wrong:
- `UNA_DEMAND_M2_PER_CAPITA` is not a constant — it's bound from `city_cfg['una_demand_m2_per_capita']` at `app.py:1153`, so at runtime it's 250 for MN, 16.7 for SA.
- Per-city values do NOT match: `config.py` declares MN 250 m²/capita (line 55), MN Full 250 (line 118), SA 16.7 (line 274).

**Fix during DATA_INVENTORY refresh.** Replace with a per-city pointer: *"Per-city `urban_nature_demand` (MN 250 / SA 16.7), `search_radius_m` (MN 1000 / SA 800), and `decay_function` (MN exponential / SA dichotomy) — values declared in `config.py` per the per-city framework principle (DESIGN_NOTES §2.1); current per-city values + parity status in CITY_PARITY.md UNA rows."* No value table here.

**Single-home anchor.** CITY_PARITY.md UNA rows (MN section lines 62–76, SA section lines 155–168) are the source of truth for the values. DESIGN_NOTES §2.2 also points at CITY_PARITY for values (no value table in DESIGN_NOTES either) — consistent.

---

## 3. NatCap trio — items accumulating

### 3.1 ALIGNMENT chronology strip — Commit 4 (trim)

Strip `"(Brief B2 revised, 2026-05-29)"` chronology from the lead-in sentence at NATCAP_ALIGNMENT.md line 77: *"Surfaced in the dashboard via per-metric validation badges (Brief B2 revised, 2026-05-29):"* → *"Surfaced in the dashboard via per-metric validation badges:"*. The §2 badge taxonomy is the authoritative current-state spec; the brief-number prefix belongs in HISTORY chronology, not in the spec lead-in. Catch as part of the trim's chronology pass.

### 3.2 SA compound-LULC structural inventory — CITY_PARITY → DATA_INVENTORY move

Currently CITY_PARITY lines 200–248 (`### SA Compound LULC Framework (structural inventory)` — the 1,984×27 lulc_crosswalk + ucm/una/carbon table internals + LULC raster comparison + integration implications). Per the single-home matrix, this is DATA_INVENTORY territory. The NatCap-trio refactor explicitly retains it in CITY_PARITY through Commit 4 (ALIGNMENT trim); the DATA_INVENTORY refresh later absorbs it and CITY_PARITY drops it with a one-line pointer.

Note: the user's Commit-2 framing said "ALIGNMENT→DATA_INVENTORY" but the structural inventory actually lives in CITY_PARITY, not ALIGNMENT. ALIGNMENT only carries a 1-row Table 2 entry (line 122) referencing the compound LULC raster.

## 4. Forward-looking — items expected to land here during the remaining rewrites

Items added as rewrites surface them. The standing categories:

- **Inbound anchor cites** that break when a rewrite renames anchors (the §1 pattern).
- **Cross-doc factual reconcile items** like §2.1 — claims in one doc contradicted by current code or by another doc's authoritative section.
- **Outbound cite tweaks** — e.g. if a doc's pointer goes to "see DESIGN_NOTES Brief B1" and Brief-B1 narrative now lives in HISTORY, the pointer text needs the doc rename.

When a cross-ref item is genuinely small and clearly scoped to *one* doc rewrite, it can fold into that commit instead of waiting for the sweep. Default is "queue for the sweep" — scattering increases risk of dropping items.

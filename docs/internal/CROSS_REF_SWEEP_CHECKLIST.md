# Cross-ref sweep — running checklist (RESOLVED)

**Audience:** Internal
**Status:** RESOLVED — all accumulated items executed in the Track 3 batch (commit `<pending>`); this file is now scaffolding for deletion at the closing-step delete pass.
**Use this for:** Confirmation that every inbound-cite break surfaced by the doc-suite rewrites has been resolved, and an audit trail for the sweep
**Do not use this for:** Per-decision rationale (→ DESIGN_NOTES.md) — this is sweep state, not design
**Source of truth for:** Sweep completion + what was retargeted to what

---

**Transient scaffolding.** Delete after the pre-push review approves the batch and the push lands. Same discipline as the other content-maps + sweep-tracking files.

**Resolution batch (Track 3 sweep commit):**

| Item | Action taken | Resolution |
|---|---|---|
| §1 — DESIGN_NOTES anchor breaks from the §1–11 rewrite | Audited each cite against current state | RESOLVED — see table below |
| §2.1 — DATA_INVENTORY §9.3 UNA-demand stale claim | Reconciled to CITY_PARITY section-anchor pointer | RESOLVED in DI-1 (`7a44253`) |
| §3.1 — ALIGNMENT chronology strip (`Brief B2 revised` lead-in) | Stripped in NatCap-trio Commit 4 | RESOLVED in trio (`f315082`) |
| §3.2 — SA compound-LULC consolidation | Absorbed into DATA_INVENTORY §2 + §9.4; CITY_PARITY breadcrumb pointer added | RESOLVED in DI-2 (`f944b5f`) |
| §3.3 — Dual-home cross-refs (OQ ↔ COLLABORATION ↔ DATA_INVENTORY) | 6 OQ entries each carry an explicit `see COLLABORATION §6 #N` cross-ref; DATA_INVENTORY §9.5 + §12 already point at COLLABORATION asks 4a/4b/5 | RESOLVED in OQ refresh (`eca0ea3`) |
| §3.4 — ALIGNMENT trim inbound cites | Audited + retargeted; see table below | RESOLVED in this commit |
| §3.5 — CLAUDE.md trim inbound cite + orphan resolution | Stratified Impervious Siting absorbed; other items confirmed homed | RESOLVED in fix-before-push (`802640a`) |

---

## §1 — DESIGN_NOTES inbound cites — resolution audit

| Cite location | Final state |
|---|---|
| `OPEN_QUESTIONS:160 → "Brief B1"` | Resolved during OQ refresh — old §160 deleted in the structural rewrite; OQ §1.1 (Per-scenario compound LULC inputs) now points at `DESIGN_NOTES §11.5` for the B2 deferral context |
| `DATA_INVENTORY:482 → "NLCD legacy vs Annual NLCD"` | Resolved during DI-1 — the §13 prose row referencing this anchor was deleted; the §12 status-snapshot row references `DESIGN_NOTES §3.1` |
| `DATA_INVENTORY:484 → "SA flood damage table — resolved"` | Resolved during DI-1 — §12 status-snapshot row references `DESIGN_NOTES §6.5` |
| `CLAUDE.md:163 → "Brief 4 cooling_f → temp_change_f"` | Resolved during CLAUDE.md trim — anchor + surrounding constants table deleted (live values in `config.py`) |
| `CLAUDE.md:528 → "SA Carbon four-pool framework adoption"` | Resolved during CLAUDE.md trim — anchor surfaced via the "Methodology matches; constant differs" coding rule, retargeted to `DESIGN_NOTES §6.4` |
| `app.py:1409 → "Brief B — UMH NE kernel"` (now line 1425 post-rewrites) | **RETARGETED this commit** → `DESIGN_NOTES §6.3 "UMH validation against canonical InVEST 3.19.0"` |
| `app.py:3288 → "Brief B2 (revised)"` (now line 3304 post-rewrites) | **RETARGETED this commit** → `DESIGN_NOTES §8.1 "Two-surface validation vocabulary — locked"` |

**Survives intact (no edit):** `app.py:4029 → DESIGN_NOTES "Lookup-overlay safety contract"` — anchor preserved at §4.4. `CONTRIBUTING.md:70 → DESIGN_NOTES "UMH validation against canonical InVEST 3.19.0"` — anchor preserved at §6.3.

## §3.4 — ALIGNMENT trim inbound cites — resolution audit

| Cite location | Final state |
|---|---|
| `REFERENCE.md:366 → "NATCAP_ALIGNMENT.md Table 1"` | **RETARGETED this commit** → `NATCAP_ALIGNMENT.md §3 "Metric methodology fidelity"` |
| `DATA_INVENTORY:460 → "Six tables"` | Resolved during DI-1 — old §12 doc-index deleted; no replacement reference needed |
| `HISTORY:336 → "SA UNA / biophysical extent"` | **RETARGETED this commit** → `docs/research/una/SA_UNA_BIOPHYSICAL_EXTENT.md` (the durable Brief A2 single home) + summary at `NATCAP_ALIGNMENT.md §4` + `CITY_PARITY.md` SA section |
| `STRATEGY.md:130 → same anchor` | **RETARGETED this commit** → same target as HISTORY:336 |
| `STRATEGY.md:135 → "Brief B2 (revised)"` | **RETARGETED this commit** → `DESIGN_NOTES §8.1` + cross-ref to §11.5 for the deferred Match/Diverged piece |
| `STRATEGY.md:136 → "C1 recorded as frozen in CLAUDE.md"` | **RETARGETED this commit** → self-reference to STRATEGY §7 + §8 (CLAUDE.md's `Blocked / pending work` section was trimmed; the Track-C-frozen context lives in STRATEGY's own Tracks discussion) |

## Other code-comment cite retargets in this commit

| Cite location | Final state |
|---|---|
| `natcap_scenarios.py:23 → "Brief B1"` | **RETARGETED this commit** → `HISTORY.md "Brief B1 (2026-05-29) — NatCap fixed scenarios as first-class inputs"` + `DESIGN_NOTES §11.5` for the deferred design |
| `HISTORY:59 → "Brief B — UMH NE kernel"` (schema log) | **RETARGETED this commit** → `DESIGN_NOTES §6.3 "UMH validation against canonical InVEST 3.19.0"` |

---

## Intentional provenance reference (NOT a broken cite)

`DESIGN_NOTES.md:777` retains the phrasing *"Originally captured in CLAUDE.md 'Blocked / pending work' (pre-trim); absorbed here so the proposal has a durable home"* — this is intentional provenance documenting where the Stratified Impervious Siting content came from historically. Not a broken cite; not a sweep target.

---

## Post-sweep audit

Post-sweep grep across all `*.py` + `*.md` files (excluding this checklist):

- Old `DESIGN_NOTES.md "Brief X"` anchor cites: **0 remaining**
- Old `NATCAP_ALIGNMENT.md "SA UNA / biophysical extent"` cites: **0 remaining**
- Old `NATCAP_ALIGNMENT.md Table N` cites: **0 remaining**
- Old `CLAUDE.md "Blocked / pending work"` cites: **0 remaining** (the one DESIGN_NOTES:777 hit is intentional provenance per above)
- `app.py` and `natcap_scenarios.py` both parse after the comment edits (syntax check confirmed)

**Sweep complete.** This file is ready for deletion at the closing-step delete pass; until then it carries the audit trail.

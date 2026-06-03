# Honesty-Surface Pass — Build Spec

**Audience:** Internal
**Status:** Ready to build LAST — after Ownership and Region-Local close. It's a consolidation over the finished surface, so it runs once at the end, not per-feature.
**Depends on:** everything that puts a value or claim on screen (validation, region selection, ownership, region-local).
**Builds:** a single audit of the displayed honesty surface + an enriched export metadata block. No new capabilities, no new vocabulary.
**Source of truth for:** the audit scope, the metadata schema additions, and where the divergences review lands (zero gates — see build sequence).

---

## Purpose

Capabilities were built incrementally, each labeling its own provenance against the LOCKED 4-state badge vocabulary ("NatCap published value" / "≈ NatCap method" / "≈ Aligned method" / "Prototype"). This pass audits the *whole* surface at once — catching any drift, gap, or inconsistency only visible across features — and enriches the export so a downloaded scenario carries its own honest provenance. Doing it once at the end, rather than re-auditing after each feature, is the entire point.

## The two parts

**1. Badge / label audit (verification, not invention).** Sweep every displayed value and confirm it carries the correct badge from the locked vocabulary; confirm region, ownership, and region-local each state provenance honestly; confirm the ownership coarseness caption and the region-local spillover caveat are present wherever those data appear. This is confirm-and-fix — **NOT** new vocabulary. If a label seems to need a new state, stop and raise it rather than inventing one.

**2. Metadata enrichment.** Enrich exported `metadata.json` with: git commit (provenance), raster lineage (source + pull date + methodology per raster — e.g., the BCAD 2026-05-31 pull and the EX-X* exempt-keyed vacancy rule), generator params, and a **known-divergences** section: the honest, plain-language list of where the prototype diverges from canonical or published values.

## Build sequence + gate tiers (zero gates)

The whole pass is batch with no required human touch. The known divergences are machine-enforced (Commit 4 asserts each seed entry is present, so they can't be dropped), and every divergence we've identified — including the emergent compound-uncertainty one — is promoted into the seed list rather than left for a person to re-notice. Each was vetted when its feature was built (the doc reframe, the 0.5 gate, the spillover caveat), so this pass is consolidation, not fresh disclosure. The only thing a machine can't check is an *unknown* divergence nobody has thought to list — and that doesn't want a dedicated stop either; it rides whatever look you take before pushing, entirely optional.

**Commit 1 — badge/label audit (batch).** Run the sweep, fix any mismatches, report findings (what was checked, what was off, what was fixed). Batch because the vocabulary is locked and labels were built per-feature — this is verification. *Exception:* if the sweep finds a serious mislabel — an overclaim — stop and raise it; that one becomes a gate.

**Commit 2 — known-divergences disclosure (batch).** Compose the known-divergences metadata section and report the assembled list. The seed list below is **machine-enforced** by the Commit 4 assertion (each entry must be present — they can't be silently dropped), so there's no human review of the known ones. As new divergences get identified, add them to the seed list so they're asserted too — that's the mechanism, rather than relying on anyone to re-notice them. Seed list (machine-enforced):
- citywide SA figures not reproduced (data-blocked — per-scenario LULCs were unsaved intermediates);
- ownership rasterization coarseness (sub-pixel parcels unreliable; reliable for large tracts);
- ownership vacancy = EX-X* exempt-keyed (a deliberate methodology choice, not a canonical definition);
- region-local spillover (reach models — UNA/UCM/UMH);
- region-local over an ownership-filtered region compounds the spillover *and* coarseness uncertainties (neither feature's own note captures the stack);
- the displayed / validated / exploratory taxonomy itself.

**Commit 3 — metadata lineage + params (batch).** git commit, raster lineage, generator params into `metadata.json`. Mechanical.

**Commit 4 — testing (batch).** export carries the full block; schema bump; **assertion that the known-divergences section contains every entry in the locked seed list** (machine-enforced completeness, so the pre-vetted divergences can't be dropped) and that the audit's fixes hold.

## Scope

**In:** auditing existing labels; enriching export metadata; composing the divergences disclosure.
**Out:** new badge vocabulary (locked — raise, don't invent); new capabilities; re-validating models (validation states are already set); changing any displayed *value* (this pass changes labels and metadata, not numbers).

## Honesty contract

This is the honesty surface's final consolidation, so its own bar is high: the divergences list must be the one a skeptical NatCap collaborator would write, not a softened version. The audit confirms claims are correctly qualified; the disclosure makes those qualifications portable — they travel with the export. If anything on screen can't be honestly badged within the locked four states, that's a finding to surface, not paper over.

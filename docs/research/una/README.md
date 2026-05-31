# UNA research notes — index

**Audience:** Internal — research-note index
**Status:** Live index
**Use this for:** Locating the UNA-specific investigations behind current-state decisions
**Do not use this for:** Current UNA methodology (→ `../../../REFERENCE.md`), current parity claims (→ `../../internal/CITY_PARITY.md`), or per-decision rationale (→ `../../internal/DESIGN_NOTES.md`)
**Source of truth for:** Which UNA investigation answered which question, and where its conclusion now lives

---

These five documents capture UNA-specific investigations the prototype ran while integrating canonical InVEST UNA (2SFCA). Most concluded into a single durable decision and were archived as research notes; this index points each one's conclusion at its current-state home.

| Doc | Question answered | Conclusion | Status | Where the conclusion lives now |
|---|---|---|---|---|
| [`SA_UNA_BIOPHYSICAL_EXTENT.md`](SA_UNA_BIOPHYSICAL_EXTENT.md) | Does the prototype's SA UNA computation footprint match NatCap's ACS block-group AOI? | Area IoU 0.824; population overlap 98.6 %; +27,457 exurban people in the bbox outside block groups. Sub-1 % population effect; don't mask the UNA path. | Live reference | `CITY_PARITY.md` SA UNA "biophysical extent" callout; `DESIGN_NOTES.md` §2.5; `NATCAP_ALIGNMENT.md` §4; `HISTORY.md` Brief A2 narrative |
| [`UNA_LULC_INVESTIGATION.md`](UNA_LULC_INVESTIGATION.md) | Does the prototype's MN cooling LULC differ from the InVEST UNA sample LULC? | The two rasters are byte-identical (MD5 `56d1080…`). | Superseded (decision held; raster identity locked in) | `CITY_PARITY.md` MN UNA LULC row (MD5 verified identical); `DATA_INVENTORY.md` §2 MN entry |
| [`UNA_METHODOLOGY_CROSS_CHECK.md`](UNA_METHODOLOGY_CROSS_CHECK.md) | Does the prototype's canonical 2SFCA implementation match `natcap.invest.urban_nature_access.execute()` on matched inputs? | MAE ≈ 0 / Pearson r = 1.0 against canonical execute(). | Superseded (UNA 2SFCA shipped as canonical) | `NATCAP_ALIGNMENT.md` §3 Nature Access row; `REFERENCE.md` UNA section; `validation/compare_una_invest.py` is the live harness |
| [`UNA_DIVERGENCE_CASE_STUDIES.md`](UNA_DIVERGENCE_CASE_STUDIES.md) | What scenario shapes drive divergence between the old Nature Access proxy and canonical 2SFCA at MN? | The old proxy behaved as a two-state "greening vs none" indicator rather than a continuous gradient — drove the Quality Score removal. | Superseded (led to Nature Quality Score retirement) | `HISTORY.md` "Nature Quality Score card (retired)" |
| [`UNA_QUALITY_SCORE_SENSITIVITY.md`](UNA_QUALITY_SCORE_SENSITIVITY.md) | Does the Nature Quality Score respond meaningfully across MN scenario space? | No — behaves as a two-state indicator, not a continuous quality gradient. | Superseded (led to Nature Quality Score retirement) | `HISTORY.md` "Nature Quality Score card (retired)" |

---

## Why these exist

These investigations grounded specific UNA-related decisions: the canonical-2SFCA adoption (replacing the prior proxy access score) and the Nature Quality Score retirement (2026-05-22). The SA biophysical extent measurement (Brief A2, 2026-05-29) is the most recent addition — it grounds the don't-mask decision for SA's raster-only UNA computation, and is the single home for the IoU 0.824 / 98.6 % population overlap measurement that several other docs reference.

Live-reference docs (currently informing decisions) are kept here. Superseded docs (the decision shipped and is now in the current-state docs) are also kept here as the audit trail — they're not deleted because the measurement detail behind a decision is the kind of thing future sessions need to verify before changing the decision.

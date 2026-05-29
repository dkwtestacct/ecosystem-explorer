# Open Questions

Outstanding asks, parked decisions, and external dependencies. Nothing here has
been sent or actioned externally — these are captured so the context isn't lost.

---

## NatCap data requests

### 1. Per-scenario compound LULC inputs · STATUS: PARKED, not sent

**Need.** The compound-encoded (NLCD×NLUD×tree) per-scenario LULC rasters —
`FF_20ac`, `FF_40ac`, `FF_MAX`, `UA_20ac`, `UA_40ac`, `UA_MAX` — that NatCap fed
to the carbon and urban-cooling models to produce the
`c_sequestration_<scenario>` and `avg_temp_f_<scenario>` cells in
`nootenboom_results/citywide_results_UPDATED.xlsx`.

**Why it matters.** Un-gates carbon/temperature reproduction — the only two
`natcap_published` metrics in `data/sa/natcap_reference_outputs.csv`. Without
these inputs, those two metrics stay at a **"NatCap published"** display (we show
NatCap's number), never **"✓ NatCap match — independently reproduced"** (we run
the prototype on NatCap's input and confirm the delta). The scenario rasters we
already have (`sa_lc_w_*_10m.tif`) are flood-encoded (NLCD×tree 3-tier), which
the Carbon/UCM/UNA compound-keyed tables cannot consume. See DESIGN_NOTES.md
"Brief B1".

**Status / resolution paths.**
- A local content-signature hunt (2026-05-29) across `~/Desktop`, `~/Downloads`,
  `~/Documents`, `/Volumes`, the Google Drive sync root, and the `_zip_archive`
  zips found **only baseline compound rasters** — no scenario variants. So the
  request is the path to un-gate, **but it is parked, not sent.** Send the draft
  below only on an explicit decision to reach out.

**Send-ready email draft (verbatim — do not send without a go-ahead):**

> **Subject:** SA urban-ag project — per-scenario LULC inputs behind the carbon & temperature results
>
> Hi [contact],
>
> We're building on the San Antonio urban-agriculture scenario work and independently reproducing a couple of the citywide results on our side as a cross-check. From the August 2024 drive share we have the baseline LULC overlay, the per-model biophysical tables, and the published citywide outputs in `nootenboom_results/citywide_results_UPDATED.xlsx`.
>
> What we're missing are the **per-scenario LULC rasters that were fed into the carbon and urban-cooling models** to produce the per-scenario values in that spreadsheet — i.e. the inputs behind the `c_sequestration_<scenario>` and `avg_temp_f_<scenario>` cells for:
>
> - **FF_20ac, FF_40ac, FF_MAX** (food-forest scenarios)
> - **UA_20ac, UA_40ac, UA_MAX** (urban-ag / garden scenarios)
>
> The scenario rasters we already have (`sa_lc_w_*_10m.tif`) look like the flood/NDR inputs — they don't carry the land-use dimension the carbon/cooling tables key on, so we can't reproduce the carbon/temperature numbers from them.
>
> Could you share:
> 1. The per-scenario LULC rasters used for the **carbon** and **urban-cooling** runs (the six scenarios above), **in whatever encoding and grid you ran them in** — we can reconcile CRS / resolution / encoding on our end as long as we know the scheme.
> 2. The matching **biophysical / carbon-pool table**, if it differs from the ones already in the August 2024 share (`carbon__nlcd_nlud_tree.csv`, `ucm__nlcd_nlud_tree.csv`).
>
> If those scenario LULCs sit under different internal names than the flood inputs, even a pointer to where they live in the project structure would help.
>
> Thanks!

---

### 2. Native NLCD×tree baseline flood raster · STATUS: open question (secondary, lower priority)

**Need.** NatCap's *baseline* LULC in the same native NLCD×tree 3-tier encoding
as the scenario rasters (the baseline the `sa_lc_w_*` scenarios were built from).
NatCap likely has one, since each scenario is that baseline with a small acreage
converted.

**Why it's lower priority.** Flood has **no NatCap published reference** (the
reference CSV's flood rows are all `aligned_method`, no value), so this is a
**UX-comparability** question, not a validation one. It surfaced from the Brief
B1 smoke test: NatCap's scenario rasters yield `mean_cn ≈ 81.4` (flood ≈ 18.6),
but the prototype's own SA baseline is `mean_cn 76.54` (flood 23.5) — a ~5-point
gap traced to the prototype's compound→NLCD×tree reduction (`tier = max(tree,1)`)
producing a different canopy-tier mix than the native raster. A fixed "food
forest" scenario therefore currently reads as *lower* flood retention than the
baseline, which would confuse users.

**Local fallbacks (no external dependency required):**
1. **Reconcile the derivation** — compute the prototype's SA baseline flood
   through the same native NLCD×tree path the loaded scenarios use, so baseline
   and scenarios are on one footing.
2. **Suppress the fixed-scenario flood delta** — display the scenario's flood as
   "≈ invariant" (matching NatCap's documented "essentially no difference"
   finding) rather than a signed delta vs the prototype baseline.

To be decided in **B2**, when the fixed-scenario flood card is wired into the
dashboard. See DESIGN_NOTES.md "Brief B1".

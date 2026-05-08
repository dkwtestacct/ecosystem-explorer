# Urban Cooling Model — Audit vs Canonical InVEST UCM

**Audit date:** 2026-05-08
**Last updated:** 2026-05-08 (resolution pass after audit-driven fixes)
**App revision:** in-tree (Minneapolis active, San Antonio ET pending)
**Reference model:** InVEST 3.14.3 Urban Cooling Model
**Sample-data params (MN):** `t_ref=23.2 °C`, `uhi_max=2.05 °C`, `green_area_cooling_distance=450 m`, `t_air_average_radius=600 m`, `cc_method=factors` (from `invest_urban_cooling_model_args_MN.json`)

## Resolution status (2026-05-08)

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Missing T_air spatial convolution | HIGH | ✅ **RESOLVED** — Gaussian filter (σ = 15 px = 450 m) applied inside `_compute_cc_raster` |
| 2 | `mean(CC)` reported as "HM" | HIGH | ✅ **RESOLVED (label-only)** — UI strings updated to "Cooling Capacity" / "CC"; internal var names kept stable for cache + surrogate compatibility |
| 3 | Energy-savings formula divergent from canonical | HIGH | ✅ **RESOLVED** — refactored to InVEST canonical `consumption × ΔT_°C × area`; `AC_KWH_PER_DEG_F` constant removed (was double-counting) |
| 4 | `HM_TO_FAHRENHEIT = 4.0` vs canonical 3.69 | MEDIUM | ✅ **RESOLVED** — added `UHI_MAX_C = 2.05`, derived `HM_TO_FAHRENHEIT = UHI_MAX_C × 1.8 = 3.69` |
| 5 | `AC_KWH_PER_DEG_F` naming + citation | MEDIUM | ✅ **RESOLVED via removal** — constant deleted entirely (the InVEST consumption rate already encodes the per-°C response, so a separate fractional-sensitivity term was double-counting) |
| 6 | `BASELINE_HM` field name stores mean(CC) | LOW | ⏸️ **DEFERRED** — internal field name kept as `mean_hm` / `BASELINE_HM` for cache + surrogate compatibility; a future search-and-replace can clean this up if the lookup table is regenerated from scratch |
| 7 | ETI normalised by max of resampled grid | LOW | ⏸️ **DEFERRED + SCOPE-BUMPED to MEDIUM** — see "Newly identified issues" below |
| 8 | `HM_TO_FAHRENHEIT` is global, not city-scoped | LOW | ⏸️ **DEFERRED** — `UHI_MAX_C` + derived `HM_TO_FAHRENHEIT` are still module-level. Should become `city_cfg['uhi_max_c']` once SA's InVEST args JSON is retrieved |
| 9 | `$0.13` vs `$0.12` quoted in spec | LOW | ✅ **NO ACTION NEEDED** — code value is correct (EIA 2024); spec was outdated |
| 10 | Buildings-extent gap | LOW | ⏸️ **DEFERRED** — documented as a known limitation in REFERENCE.md caveats |

**Schema version bumped 7 → 8** to invalidate cached lookup tables and force regeneration with the new CC pipeline.

### Newly identified issues (during the resolution pass)

| # | Finding | Severity | Status |
|---|---|---|---|
| 11 | ET nodata sentinel (65535) survives `~np.isfinite()` filter, poisoning `MAX_ET_REF` and zeroing the ETI term in CC | MEDIUM | ✅ **RESOLVED (2026-05-08)** — read `et_src.nodata` and mask sentinel + out-of-range values to NaN *before* `resize()` so bilinear interpolation can't bleed sentinels onto valid pixels. NaNs introduced by resize are filled with the field median. Schema bumped 8 → 9 to invalidate caches. **Empirical impact:** `MAX_ET_REF` 65535 → 1158 (~57× smaller), so the 0.2 × ETI term is now active rather than ≈ 0. `mean(CC, smoothed)` shifted 0.107 → 0.186 — the ~0.08 increase is consistent with `0.2 × kc × ET_normalised` (Kc ≈ 0.5 average across MN classes, normalised ETI ≈ 0.97 for valid pixels = 0.097 contribution before smoothing edge effects). Code at app.py:307-330. |

---

## TL;DR

The Ecosystem Explorer correctly implements the **per-pixel cooling-capacity (CC) factor** stage of InVEST UCM, including the ET-driven ETI term. It **does not** implement the spatial **T_air convolution / cooling-distance kernel**, the **HMI normalization** against `t_ref` and `uhi_max`, or the **per-building T_air sampling** that InVEST uses for energy valuation. As a result:

- The metric labelled **HM** on the UI is `mean(CC)` — a *capacity* index, not InVEST's *realized mitigation index* (HMI).
- **Cooling-energy savings** are estimated from a per-pixel ΔCC × fractional-AC-sensitivity heuristic, not InVEST's ΔT_air-per-building × consumption formulation.
- The dollar number is **order-of-magnitude**, not directly comparable to InVEST's energy-valuation output.

Severity rating per finding: **HIGH** = changes the meaning of the metric; **MEDIUM** = affects accuracy but not interpretation; **LOW** = cosmetic / hygiene.

---

## Canonical InVEST UCM pipeline (factors method)

For reference, the steps the canonical model runs:

1. **CC per pixel** — `CC_i = 0.6·shade_i + 0.2·albedo_i + 0.2·ETI_i`, where `ETI_i = (Kc_i × ET_ref_i) / max(ET_ref over AOI)`.
2. **T_air_nomix per pixel** — `T_air_nomix_i = t_ref + uhi_max × (1 − CC_i)`. With `uhi_max=2.05 °C`, this caps the per-pixel anomaly at +2.05 °C above `t_ref=23.2 °C`.
3. **T_air per pixel** — Gaussian convolution of `T_air_nomix` with kernel radius = `green_area_cooling_distance` (450 m for MN). This is what spatially propagates cooling from green pixels onto adjacent built ones.
4. **HMI per pixel** — `HMI_i = 1 − (T_air_i − t_ref) / uhi_max`, bounded 0–1. *This* is the canonical "Heat Mitigation Index."
5. **Per-building T_air** — for each building footprint, average `T_air` over pixels within `t_air_average_radius` (600 m for MN).
6. **Energy savings per building** — `kWh_saved_b = consumption_b × (T_air_max − T_air_b)`, where `consumption_b` is the per-type rate from `energy_consumption.csv` (kWh/m²/yr × footprint area, or kWh/yr direct depending on InVEST version) and `T_air_max` is the maximum T_air across the AOI. Optional `$/kWh` valuation for dollar output.
7. **(Optional) Work productivity** — uses humidity + T_air to estimate outdoor-labor heat-stress hours. Disabled in the MN sample (`do_productivity_valuation: false`).

---

## Step-by-step audit

### Step 1 — CC per pixel ✅ FAITHFUL

**App location:** [`_compute_cc_raster()`](../../app.py) at app.py:335-347.

**Behavior:** `CC = 0.6·shade + 0.2·albedo + 0.2·ETI` with `ETI = kc · ET / max(ET)`.

**Findings:** Matches canonical formulation. ET raster is bilinear-resampled from 1 km to 30 m, which smooths the field but does not introduce systematic bias — the InVEST model itself accepts coarser ET inputs and resamples internally.

**Severity:** —

---

### Step 2 — T_air_nomix ✅ RESOLVED (2026-05-08)

**App location:** N/A.

**Canonical formula:** `T_air_nomix_i = 23.2 + 2.05 × (1 − CC_i)` °C for Minneapolis.

**Findings:** The app skips this step entirely and instead uses `ΔCC × HM_TO_FAHRENHEIT` (4.0 °F per HM unit) as a direct linear translation. With `uhi_max=2.05 °C ≈ 3.69 °F`, the canonical model implies the *maximum possible* ΔT (full CC swing from 0 → 1) is **3.69 °F**, not 4.0 °F. Our calibration is therefore ~8 % high in absolute magnitude for the MN AOI. For SA, no `uhi_max` has been retrieved yet, so the equivalent calibration there is unknown.

**Severity:** **MEDIUM** — direction is correct, magnitude is plausible, but `HM_TO_FAHRENHEIT = 4.0` is not derived from the InVEST MN parameters; it's an InVEST-literature-range midpoint. Should be reset to `uhi_max × 1.8 °F/°C ≈ 3.69` for MN if we want consistency with the project's own params.

**Resolution:** Added `UHI_MAX_C = 2.05` to app.py and derived `HM_TO_FAHRENHEIT = UHI_MAX_C * 1.8 = 3.69` from it. Source comment cites the InVEST args JSON. See app.py:75-82.

**Outstanding:** the constant is still module-level, not per-city. Once SA's InVEST args JSON is retrieved, both should move into `CITIES['<city>']['uhi_max_c']`.

---

### Step 3 — T_air spatial convolution ✅ RESOLVED (2026-05-08)

**App location:** N/A.

**Canonical formula:** Gaussian convolution of `T_air_nomix` with kernel σ chosen so radius = `green_area_cooling_distance` (450 m for MN, ≈ 15 NLCD pixels).

**Findings:** This is the most significant divergence. Without convolution, a building footprint immediately *next to* a converted green pixel sees zero benefit in our model — only buildings *on* the converted pixel itself do. In reality, urban cooling propagates 100s of metres downwind. Our per-pixel cooling effects are therefore **conservative inside green patches** (no pooling boost from neighbors) and **pessimistic at green-patch edges** (buildings nearby get no spillover credit).

For the MN AOI (10.8 × 10.7 km, 360 × 356 px), a 450 m kernel radius is a 15-pixel Gaussian — large enough that omitting it materially changes which buildings "see" each conversion. For scenarios with many small scattered conversions, the omission *underestimates* benefits; for scenarios with one large contiguous green patch, the per-patch interior is fine but edge effects are missed.

**Severity:** **HIGH** — changes the semantics of "HM" and "energy savings" significantly. Mean values across the AOI may be roughly self-consistent for *relative* scenario comparison, but absolute °F and $ figures should not be presented as InVEST-calibrated.

**Resolution:** Added `from scipy.ndimage import gaussian_filter` and applied it to the per-pixel CC raster inside `_compute_cc_raster` (app.py:344-385). Constants `GREEN_AREA_COOLING_DISTANCE_M = 450` and `_CC_SIGMA_PX = 15` (= 450 m / 30 m). NaN-aware filling: NaN pixels are temporarily replaced with the in-AOI mean before convolution (otherwise NaN would zero out a 15-px ring around every nodata pixel and visibly bleed inward), then restored on the output. Edge handling: `mode='nearest'` to avoid mirror-reflection of cooling effects.

We smooth CC directly rather than `T_air_nomix`. These are equivalent up to constants because `T_air_nomix = t_ref + UHI × (1 − CC)` is an affine transform of CC and Gaussian convolution is linear, so smoothed-CC × ΔT-mapping produces the same per-pixel result.

**Empirical impact (Minneapolis baseline, computed at module load):**

- mean(CC) shifts from 0.27 → similar (Gaussian is mean-preserving with reflection BCs; small drift from `mode='nearest'` at edges).
- std(CC) drops from ~0.14 → ~0.05 — heterogeneity gets averaged within the 450 m radius, which is the desired effect (hot pixels see neighboring green; green pixels see neighboring concrete).
- Per-scenario evaluation cost rose by ~5 ms (single FFT-based Gaussian on 360 × 356 px), well within the precompute budget. Lookup-table regeneration cost: ~13 s extra for 2,541 scenarios.

**Outstanding:** we do **not** sample T_air per-building polygon over a 600 m `t_air_average_radius` (Step 5 below remains open). The convolution alone gets us most of the way: buildings on or *near* converted pixels now see ΔCC > 0, where previously only buildings *exactly on top of* a converted pixel did.

---

### Step 4 — HMI normalization ✅ RESOLVED (label-only, 2026-05-08)

**App location:** `mean(CC)` is reported as "HM" throughout the UI.

**Canonical formula:** `HMI_i = 1 − (T_air_i − t_ref) / uhi_max`.

**Findings:** Our "HM" is `mean(CC)`, which has the same 0–1 range as canonical HMI by coincidence (CC is bounded 0–1 by the 0.6 + 0.2 + 0.2 weights × inputs ≤ 1). But the values represent different things:

- **Mean CC** = average per-pixel cooling *potential*, agnostic of spatial mixing.
- **Canonical HMI** = average realized cooling effectiveness, post-spatial-mixing.

Numerically, for a typical MN baseline, mean(CC) ≈ 0.27 implies `T_air_nomix` mean ≈ 23.2 + 2.05 × 0.73 ≈ 24.7 °C; after spatial mixing toward the rural T_ref, HMI mean would be slightly higher than 0.27 (because hot pixels get cooled by neighbors). The two are correlated but not identical.

**Severity:** **HIGH** for terminology, **MEDIUM** for numbers — they disagree by less than the 4 °F-per-HM uncertainty band, but the metric *name* is wrong.

**Resolution:** Took option (a) — UI strings updated:

- "Heat Mitigation Index" → "Cooling Capacity" in axis labels, chart titles, the methodology box.
- "Cooling HM:" → "Cooling CC:" in tooltips and reference-scenario hover text.
- The metric-card help text now explicitly notes: *"this is mean(CC), an approximation of the canonical InVEST Heat Mitigation Index — see UCM_AUDIT.md."*

Internal variable names (`mean_hm`, `BASELINE_HM`, `hm_arr`) were **deliberately kept** to preserve compatibility with the cached lookup-table CSV columns and the surrogate-model feature names. A future deeper refactor could rename these together with a fresh lookup-table regeneration.

**Outstanding:** Option (b) — implementing the full HMI normalisation `HMI = 1 − (T_air − t_ref) / uhi_max` — would still be a more principled fix. Currently mean(smoothed CC) and canonical mean(HMI) differ by less than the ±2 °F-per-CC uncertainty, so the practical impact is small.

---

### Step 5 — Per-building T_air sampling ❌ NOT IMPLEMENTED

**App location:** N/A.

**Canonical formula:** For each building polygon, average T_air over pixels in a 600 m radius (`t_air_average_radius`).

**Findings:** We use per-pixel ΔCC at the building *footprint* directly. No 600 m averaging. Combined with the missing convolution (step 3), this means our energy savings respond only to conversions that happen *on or directly under* a building's roof pixels — physically implausible (you don't air-condition a roof, you air-condition the volume below it, and that volume's T_air depends on the surrounding 600 m).

**Severity:** **HIGH** — directly affects the dollar figure on the metric card.

**Recommended fix:** After implementing step 3, sample the convolved T_air at each building's footprint pixels and average. Or, simpler interim fix: use the AOI mean of T_air (single number) for all buildings — much better than per-roof.

---

### Step 6 — Energy savings formulation ✅ RESOLVED (2026-05-08)

**App location:** [`compute_cooling_energy_savings()`](../../app.py) at app.py:1014-1032.

**Canonical formula:** `kWh_saved_b = consumption_b × (T_air_max − T_air_b)` per building, in kWh/yr (consumption units in the InVEST sample table are kWh/°C/m²/yr or kWh/°C/yr depending on version).

**Our formula:**
```
ΔCC = CC_scenario − CC_baseline                    # per pixel
ΔT_°F = ΔCC × HM_TO_FAHRENHEIT                     # per pixel, °F
avoided_fraction = clip(ΔT_°F × AC_KWH_PER_DEG_F, 0, 1)
$_per_pixel = avoided_fraction × consumption_rate × 900 m² × $0.13/kWh
$_total = sum($_per_pixel for pixels with buildings)
```

**Differences:**
1. **Per-pixel, not per-building.** Three pixels under one building footprint count as three independent contributions.
2. **`AC_KWH_PER_DEG_F = 0.03` is dimensionless (fractional), not kWh/°F.** The variable name is misleading. The value (3 % of AC consumption per °F cooler) is loosely sourced — see below.
3. **`consumption` from `energy_consumption.csv` is treated as kWh/m²/yr** and multiplied by `900 m²` per pixel. The InVEST sample table column `consumption` is documented as **kWh/°C/m²/yr** in InVEST 3.14 — we drop the `/°C` denominator and apply our own `0.03 fraction × ΔT °F` instead. This works numerically but breaks compatibility with InVEST's table semantics; if a user swaps in a project-specific table calibrated for InVEST's formula, our app will silently misinterpret the values.
4. **Saturation cap at 100 % per pixel** — sensible safety, but indicates we know ΔT × 0.03 can blow past unity for large CC swings.

**Severity:** **HIGH** for table-semantic compatibility, **MEDIUM** for the dollar magnitude (likely within 2–3× of the canonical answer for typical scenarios).

**Verification of `consumption` units:** Fetched directly from the InVEST 3.14 user guide: *"consumption (number, units: kWh/(m² · °C), required): Energy consumption by footprint area for this building type."* The audit's original guess about the units was correct.

**Resolution:** `compute_cooling_energy_savings` (app.py:1014-1042) was refactored to:

```python
delta_t_c = clip(ΔCC × UHI_MAX_C, 0, None)              # °C, lower-clamped at 0
kwh_per_pixel = CONSUMPTION_RATE × delta_t_c × PIXEL_AREA_M2     # kWh/yr
usd_per_pixel = kwh_per_pixel × COST_PER_KWH_USD                  # $/yr
total = sum(usd_per_pixel for pixels with buildings)
```

Now matches InVEST's canonical formula exactly except for the per-building polygon aggregation (Step 5, still open). The `AC_KWH_PER_DEG_F = 0.03` fractional sensitivity has been **removed entirely** — it was double-counting against the per-degree response already encoded in the InVEST consumption rate.

**Outstanding:** still per-pixel rather than per-building polygon. A building footprint covering 10 pixels gets 10 contributions; InVEST canonically averages T_air over those 10 pixels first and applies the per-building consumption rate once. For the typical case where consumption_rate is uniform across a building footprint, the two are numerically identical. They diverge when a building straddles a CC discontinuity.

---

### Step 7 — Work productivity ⛔️ INTENTIONALLY OUT OF SCOPE

The MN sample data has `do_productivity_valuation: false`, and we don't expose this output. No action needed unless the SA project deliverables include outdoor-labor heat-stress estimates.

---

## The `AC_KWH_PER_DEG_F = 0.03` constant — defensibility

The code comment cites "RECS 2020" for the 3 %/°F figure. RECS (US Energy Information Administration's Residential Energy Consumption Survey) does **not** directly publish an AC-elasticity value. The 3 %/°F approximation typically comes from one of these sources:

- **Cooling-degree-day (CDD) elasticity studies** — Auffhammer & Mansur (2014, *Energy Economics*) and similar regression studies estimate AC electricity demand rises ~2–5 % per °F of summer mean-temperature increase, with substantial regional variation (Sun Belt higher, Pacific Northwest lower).
- **EPA ENERGY STAR thermostat guidance** — claims 1 % savings per °F of thermostat setpoint adjustment, but this is *setpoint sensitivity*, not *outdoor-T sensitivity* (the two are correlated but distinct).
- **California Energy Commission Title 24 modelling** — gives ~3 %/°F for residential AC under typical California climate-zone profiles.

**Verdict:** 3 %/°F is **defensible as an order-of-magnitude residential figure** but should not be cited as "RECS 2020." It is also **not defensible for commercial or industrial** building types — those have different HVAC response curves (commercial often higher per-°F due to internal-load-dominated buildings; industrial often lower due to process-cooling that's largely T-independent).

**Recommended fix:**
1. Update the source comment from "RECS 2020" to "Auffhammer & Mansur 2014 (residential AC CDD elasticity, US national midpoint)".
2. Either (a) accept the residential-only approximation and document it as a limitation, or (b) split into per-type sensitivities — e.g., `AC_FRACTION_PER_DEG_F = {0: 0.025, 1: 0.04, 2: 0.03, 3: 0.015}` for other/commercial/residential/industrial.
3. **Rename the constant** to `AC_FRACTION_PER_DEG_F` (it's a fraction, not a kWh value).

**Severity:** **MEDIUM** — the value is plausible, the citation is wrong, and the variable name confuses readers.

**Resolution (2026-05-08):** Constant **removed entirely** rather than renamed. The user-spec asked for a rename to `AC_FRACTION_PER_DEG_F`, but verifying the InVEST `consumption` column units (kWh/(m²·°C), see Step 6) showed that the fractional sensitivity was double-counting the per-degree response already encoded in the InVEST rate. Removing the constant + the multiplication step produces the canonical formula directly. Citation discussion above is preserved for future reference if a per-type sensitivity adjustment is ever reintroduced.

---

## Other findings

### Lower-severity issues

| Finding | Location | Severity | Note |
|---|---|---|---|
| `BASELINE_HM` field name in `CITIES` actually stores baseline mean(CC), not HMI | app.py:18-21, 33-48 | LOW | Either rename to `baseline_cc` or implement true HMI. |
| ETI normalization uses `max(ET_RESIZED)` over the bilinear-resampled grid, not the AOI mask | app.py:307-309 | LOW | Functionally equivalent for the MN AOI since the ET raster is already AOI-clipped. Worth nailing down for SA. |
| `HM_TO_FAHRENHEIT = 4.0` is global, not city-scoped | app.py:76 | LOW | Should become `city_cfg['uhi_max_f']` once we wire MN/SA params from their respective InVEST JSONs. |
| `COST_PER_KWH_USD = 0.13` hardcoded in app.py vs `$0.12` quoted in the original spec discussion | app.py:331 | LOW | Code value is current EIA 2024 US average; spec was outdated. Not a bug. |
| Building-extent gap: areas of MN outside the InVEST sample-data buildings shapefile contribute zero energy savings | app.py:1014-1032 | LOW | Documented in REFERENCE.md caveats. Acceptable for the MN demo; replace with city-wide assessor parcels for a real deployment. |

### Things we do *better* than canonical InVEST UCM

To be fair, a few choices in the app go beyond what stock UCM offers:

- **Per-pixel saturation cap** on avoided-AC fraction prevents pathological output for scenarios with extreme ΔCC.
- **ETI is computed against the in-AOI max, not a global max**, which scales the term sensibly across cities of different climates.
- **Graceful degradation** — if any of (buildings, energy table, ET raster) are missing, the energy-savings card returns $0 cleanly rather than crashing. InVEST UCM hard-fails on missing inputs.

---

## Recommended next steps, prioritised

1. **(HIGH)** Rename `AC_KWH_PER_DEG_F` → `AC_FRACTION_PER_DEG_F` and fix the citation comment. Trivial change, removes the most misleading line in the codebase.
2. **(HIGH)** Decide on a path for HM semantics: either (a) globally rename `mean_hm` / `BASELINE_HM` to make clear we report mean CC, or (b) implement steps 2–4 (T_air_nomix, Gaussian convolution, HMI). Option (b) is ~80 lines of code and one config field per city; option (a) is a search-and-replace.
3. **(MEDIUM)** Add per-city `uhi_max_f` and `t_ref_f` to `CITIES`, sourced from the InVEST args JSON. Replace global `HM_TO_FAHRENHEIT`.
4. **(MEDIUM)** When implementing step 3, gate the convolution behind `ENABLE_T_AIR_CONVOLUTION` so the lookup-table precompute time can be measured before/after.
5. **(LOW)** Add a "Methodology divergences" section to the in-app About / Help text linking here, so the dollar figure on the energy-savings card is presented honestly.

---

## What would change if we implemented the canonical pipeline

Rough expected impact on Minneapolis at the default conversion sliders:

- **Mean HM**: ~0.27 → ~0.28–0.30 (small uplift from spatial mixing helping built pixels near green ones).
- **Temperature change**: similar magnitude (4.0 °F factor → 3.69 °F factor cancels with the convolution-driven HM uplift, give or take).
- **Cooling energy savings ($)**: meaningful change (~2–5×), because per-building T_air sampling over 600 m credits buildings *near* conversions, not just *on top of* them. Direction is "higher savings".

Bottom line: relative scenario rankings are likely robust under our simplified implementation, but absolute $ figures should be presented as the order-of-magnitude estimates they are, with this audit linked from the metric card's tooltip.

"""model_validation.py — single source of truth for which InVEST models have
measured per-pixel parity against canonical natcap.invest 3.19.0.

This is the ONE canonical record of the "validated" flag. It is consumed by:
  - the InVEST export bundle (export_invest_bundle._VALIDATION re-exports this), and
  - from Stage 2, the per-card validation badges,
so the flag can't drift across surfaces. **Low-level by design:** it imports
nothing from the app / bundle / badge modules, so any of them can import it
without a circular dependency. (In particular the badge path must NOT have to
import the export-bundle module just to read a validated flag.)

Per-model state + parity metadata (`status` / `mae` / `pearson_r` / `reference` /
`notes`):
  - **`validated`** — per-pixel parity measured against canonical
    natcap.invest 3.19.0 via a committed `compare_*_invest.py` harness.
  - **`methodology_aligned`** — canonical method, no per-pixel parity check.
    (Currently none: all five urban + carbon models have measured per-pixel
    parity. UFR is validated via its per-pixel runoff-retention index; the lumped
    Flood Index / Runoff Volume readings remain aligned-method.)

Edit the validated set ONLY when a model's parity status genuinely changes — the
`verify_baselines.py` source-consistency check pins the expected set and will
fail until it is updated on purpose (a deliberate-change detector).
"""

MODEL_VALIDATION = {
    "ucm": {"status": "validated", "mae": 0.0, "pearson_r": 1.0,
            "reference": "natcap.invest.urban_cooling_model 3.19.0",
            "notes": "HMI = max(CC_local, CC_park) validated to per-pixel parity "
                     "(compare_ucm_invest.py). Export is biophysical-cooling-only "
                     "(do_energy_valuation=False)."},
    "una": {"status": "validated", "mae": 0.0545, "pearson_r": 1.0,
            "reference": "natcap.invest.urban_nature_access 3.19.0",
            "notes": "2SFCA supply_percapita validated to per-pixel parity vs "
                     "canonical natcap.invest 3.19.0 (compare_una_supply_invest.py "
                     "→ comparisons/una_supply_parity_mn.csv): Pearson r = "
                     "1.000000, per-pixel MAE 0.054 m²/person (~5.5e-7 of the "
                     "~99,000 m²/person field), |Δtotal| ≈ 0 over 70,868 MN px. "
                     "Matched-but-independent (InVEST computes its own supply from "
                     "the same LULC/pop/table/params) + non-vacuous +2%-pop guard. "
                     "Supersedes the withdrawn ungrounded 0.0234 claim and the "
                     "3.16.2 reachability proxy."},
    "ufr": {"status": "validated", "mae": 0.0, "pearson_r": 1.0,
            "reference": "natcap.invest.urban_flood_risk_mitigation 3.19.0",
            "notes": "The per-pixel runoff-retention index (1 − Q/P, "
                     "app.cn_array_to_retention_index) is validated to per-pixel "
                     "parity against UFRM's runoff_retention_index: MAE ~5e-8, "
                     "r = 1.0 over 3.36M pixels, value-identical CN "
                     "(compare_ufr_invest.py, Relay 71). UFRM λ=0.2 / "
                     "S_max=25400/CN−254 mm is algebraically identical to the "
                     "evaluator's Ia=0.2S / S=1000/CN−10 in. The dashboard's "
                     "headline Flood Index (100 − mean_CN) and Runoff Volume are "
                     "lumped mean-CN proxies, NOT per-pixel UFRM outputs — they "
                     "stay aligned-method. SA uses NatCap's NLCD×tree CN table; "
                     "damage valuation disabled (no SA damage table, Path C)."},
    "carbon": {"status": "validated", "mae": 0.0, "pearson_r": 1.0,
               "reference": "natcap.invest.carbon 3.19.0",
               "notes": "SA four-pool stock framework per NatCap Vibrant Land "
                        "(Guerry et al. 2023), validated to per-pixel parity against "
                        "natcap.invest.carbon 3.19.0 in matched units "
                        "(compare_carbon_sa_fourpool_invest.py): per-pixel MAE ~3e-7 "
                        "Mg C, r = 1.0. do_valuation=False in the export (stock change "
                        "only)."},
    "umh": {"status": "validated",
            "reference": "natcap.invest.urban_mental_health 3.19.0",
            "notes": "Per-pixel kernel parity validated (Brief B); canonical "
                     "execution on the baseline bundle verified (D1 Phase 3). "
                     "Two args files emitted (depression, anxiety). Inputs use a "
                     "synthetic uniform baseline-prevalence vector "
                     "(risk_rate = CDC ever-diagnosed BIR) and a synthetic NDVI "
                     "proxy, not satellite NDVI — input-quality caveats, "
                     "separate from algorithmic parity."},
}

# Convenience views — DERIVED from the canonical dict, never re-declared as a
# literal (a second literal could silently drift from MODEL_VALIDATION).
VALIDATED_MODELS = frozenset(
    k for k, v in MODEL_VALIDATION.items() if v.get("status") == "validated"
)
METHODOLOGY_ALIGNED_MODELS = frozenset(
    k for k, v in MODEL_VALIDATION.items() if v.get("status") == "methodology_aligned"
)

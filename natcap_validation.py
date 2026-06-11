"""natcap_validation.py — load + compare prototype outputs against NatCap reference values.

Reads `data/<city>/natcap_reference_outputs.csv` (the source of truth for what
NatCap publishes for the SA project scenarios; built by
`extract_natcap_reference_outputs.py`).

NOT wired into the dashboard yet — Brief B2 will use these helpers to render
per-metric validation markers on the cards. This module just provides the
lookup + comparison primitives.

Comparison model: reference values for `natcap_published` metrics are stored
ABSOLUTE with explicit baseline rows, but the prototype reports those metrics as
DELTAS (`temp_change_f` = ΔT vs baseline; `carbon_tons_co2` = stock change). So
`compare_to_reference` compares the prototype's delta to
`(ref_scenario − ref_baseline)`. `aligned_method` and `prototype` rows carry no
tolerance and return `status='no_reference'`. (If a future `natcap_published`
metric is an *absolute* the prototype produces directly, this module would need
a per-metric comparison-mode flag — today both published metrics are deltas.)
"""
from __future__ import annotations
import os
from functools import lru_cache

import pandas as pd

import model_validation  # Stage 1 — single source of truth for the validated set.

# Per-city reference CSV. MN has no nootenboom coverage yet (deferred).
_CITY_CSV = {"SA": "data/sa/natcap_reference_outputs.csv"}

# Card-metric → InVEST model, for the cards that CAN render "InVEST-validated".
# Validation is read from model_validation.VALIDATED_MODELS (the Stage-1 source) —
# this map only says which model backs each metric; it never re-declares which are
# validated. Only the validated-5 cards appear here; everything else is aligned /
# prototype. carbon_value_usd is deliberately absent (a dollar valuation, not the
# per-pixel stock output), as are the lumped Flood Index / Runoff Volume.
_METRIC_TO_MODEL = {
    "temp_change_f":       "ucm",
    "nature_access_pct":   "una",
    "preventable_mh_cases": "umh",
    "runoff_retention_idx": "ufr",
    "carbon_tons_co2":     "carbon",
}


def _city_key(city: str) -> str:
    return "SA" if str(city).startswith(("San Antonio", "SA")) else str(city)


@lru_cache(maxsize=4)
def load_reference_outputs(city: str):
    """Return the reference DataFrame for `city`, or None if no reference file exists."""
    path = _CITY_CSV.get(_city_key(city))
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def lookup_reference(city: str, scenario_id: str, metric_name: str):
    """Return the matching reference row as a dict, or None."""
    df = load_reference_outputs(city)
    if df is None:
        return None
    m = df[(df["scenario_id"] == scenario_id) & (df["metric_name"] == metric_name)]
    return m.iloc[0].to_dict() if len(m) else None


def _num(v):
    """Coerce a CSV cell to float, or None for blank/NaN/non-numeric."""
    try:
        if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def compare_to_reference(city: str, scenario_id: str, metric_name: str, prototype_value: float):
    """Compare a prototype scenario value against its NatCap reference.

    For `natcap_published` metrics the stored reference is absolute, so the
    comparison is on the delta: prototype_value (already a delta) vs
    `(ref_scenario − ref_baseline)`. Returns a dict with:
        status            'match' | 'diverged' | 'no_reference'
        within_tolerance  bool | None
        delta_abs         prototype_delta − reference_delta  (None if no ref)
        delta_pct         100 × delta_abs / |reference_delta|  (None if ref_delta≈0)
        reference_delta, prototype_delta, validation_status, tolerance_pct, tolerance_abs
    `within_tolerance` is True if the absolute OR the percentage gate passes
    (temp's tiny deltas make the absolute gate the practical one).
    """
    row = lookup_reference(city, scenario_id, metric_name)
    if row is None:
        return {"status": "no_reference", "reason": "no matching reference row",
                "within_tolerance": None, "delta_abs": None, "delta_pct": None}
    vstatus = row.get("validation_status")
    if vstatus != "natcap_published":
        return {"status": "no_reference", "reason": f"validation_status={vstatus}",
                "validation_status": vstatus, "within_tolerance": None,
                "delta_abs": None, "delta_pct": None, "notes": row.get("notes")}

    ref_scn = _num(row.get("natcap_value"))
    base_row = lookup_reference(city, "baseline", metric_name)
    ref_base = _num(base_row.get("natcap_value")) if base_row else None
    if ref_scn is None or ref_base is None:
        return {"status": "no_reference", "reason": "missing reference value or baseline row",
                "within_tolerance": None, "delta_abs": None, "delta_pct": None}

    ref_delta = ref_scn - ref_base
    proto_delta = float(prototype_value)
    delta_abs = proto_delta - ref_delta
    delta_pct = (100.0 * delta_abs / abs(ref_delta)) if abs(ref_delta) > 1e-12 else None
    tol_pct, tol_abs = _num(row.get("tolerance_pct")), _num(row.get("tolerance_abs"))

    ok = False
    if tol_abs is not None and abs(delta_abs) <= tol_abs:
        ok = True
    if tol_pct is not None and delta_pct is not None and abs(delta_pct) <= tol_pct:
        ok = True

    return {"status": "match" if ok else "diverged", "within_tolerance": ok,
            "delta_abs": delta_abs, "delta_pct": delta_pct,
            "reference_delta": ref_delta, "prototype_delta": proto_delta,
            "validation_status": vstatus, "tolerance_pct": tol_pct, "tolerance_abs": tol_abs}


def published_delta(city: str, scenario_id: str, metric_name: str):
    """Return (scenario_absolute, baseline_absolute, delta) from the reference
    CSV for a `natcap_published` metric. Used by the fixed-scenario reference
    view to display NatCap's per-scenario delta. Returns (None, None, None) if
    the metric isn't natcap_published or if any value is missing.
    """
    row_scn  = lookup_reference(city, scenario_id, metric_name)
    row_base = lookup_reference(city, "baseline", metric_name)
    if row_scn is None or row_base is None:
        return None, None, None
    if row_scn.get("validation_status") != "natcap_published":
        return None, None, None
    sv = _num(row_scn.get("natcap_value"))
    bv = _num(row_base.get("natcap_value"))
    if sv is None or bv is None:
        return None, None, None
    return sv, bv, sv - bv


# ── Brief B2 (revised): scenario-aware validation badges ─────────────────────
# The CSV's `validation_status` is a per-METRIC property (constant across
# scenarios — temp/carbon = `natcap_published`; nature/cooling/flood/UMH =
# `aligned_method`; food = `prototype`). The badge text reflects the metric
# class; the *tooltip* is scenario-aware so the badge never overclaims —
# especially for `natcap_published` metrics on non-baseline scenarios, where
# "we match NatCap" would be flatly wrong.
#
# The original B2's per-scenario "✓ Match (Δ X%)" / "× Diverged" states are
# intentionally OUT of scope here — they need prototype reproduction for the
# NatCap fixed alternative scenarios, which is gated on compound scenario
# inputs (see docs/internal/OPEN_QUESTIONS.md). When/if those arrive, this helper grows a
# 4th and 5th state.

# Active-scenario contexts. The caller passes one based on session_state /
# provenance (see natcap_scenarios.PROVENANCE_*). These drive the tooltip
# wording so the badge stays honest per context.
SCENARIO_CONTEXT_BASELINE     = "baseline"
SCENARIO_CONTEXT_NATCAP_FIXED = "natcap_fixed_scenario"
SCENARIO_CONTEXT_EXPLORER     = "explorer_generated"
SCENARIO_CONTEXT_OPTIMIZER    = "optimizer_suggested"

ALL_SCENARIO_CONTEXTS = (
    SCENARIO_CONTEXT_BASELINE,
    SCENARIO_CONTEXT_NATCAP_FIXED,
    SCENARIO_CONTEXT_EXPLORER,
    SCENARIO_CONTEXT_OPTIMIZER,
)


def _natcap_method_tooltip_for_metric(metric_name):
    """Per-metric tooltip body for the blue '≈ NatCap method' state.

    Temperature cites measured per-pixel HMI parity (Brief 28b). Carbon now
    also cites measured per-pixel parity — its four-pool stock framework
    (Brief 30) is validated against canonical InVEST 3.19.0 at MAE ≈ 0 / r 1.0
    in matched units (Relay 69). Other natcap_published-class metrics fall back
    to a generic tooltip with no parity claim.
    """
    if metric_name == "temp_change_f":
        return (
            "Canonical InVEST UCM methodology. The displayed value is the "
            "prototype's own computation; per-pixel HMI parity vs canonical "
            "InVEST UCM is measured (MAE 0.0000, r 1.0000 — Brief 28b). "
            "NatCap's published citywide T_air baseline isn't reproducible "
            "from disk (UCM args not shipped); see docs/internal/OPEN_QUESTIONS.md."
        )
    if metric_name in ("carbon_tons_co2", "carbon_value_usd"):
        return (
            "NatCap's InVEST four-pool stock framework, adopted per "
            "Vibrant Land (Guerry et al. 2023, Brief 30). The displayed "
            "value is the prototype's own computation; per-pixel parity vs "
            "canonical InVEST Carbon IS measured (MAE ≈ 0, r 1.0 vs "
            "natcap.invest 3.19.0 in matched units — Relay 69). NatCap's "
            "published citywide baseline carbon isn't reproducible from "
            "disk (their aggregation script isn't shipped); see "
            "docs/internal/OPEN_QUESTIONS.md."
        )
    return (
        "NatCap-aligned methodology (canonical InVEST). The displayed "
        "value is the prototype's own computation; per-pixel parity vs "
        "canonical InVEST is not measured for this metric."
    )


def _badge_tooltip_for_metric(metric_name, vstatus):
    """Existing per-metric badge tooltip body, factored so the InVEST-validated
    and InVEST-aligned branches reuse it unchanged (tooltip *bodies* are rewritten
    in Slice 3, not here)."""
    if vstatus == "natcap_published":
        return _natcap_method_tooltip_for_metric(metric_name)
    if vstatus == "aligned_method" and metric_name == "runoff_retention_idx":
        return (
            "Canonical InVEST UFR runoff-retention index (1 − Q/P). The "
            "displayed value is the prototype's own computation; per-pixel "
            "parity vs canonical InVEST UFRM IS measured (MAE ≈ 0, r 1.0 vs "
            "natcap.invest 3.19.0 in matched units — Relay 71). No published "
            "NatCap SA flood value exists to match against."
        )
    return (
        "Canonical InVEST methodology. No directly-comparable NatCap citywide "
        "reference exists, or the framing differs (different summary statistic, "
        "scope, or aggregation level). See docs/internal/NATCAP_ALIGNMENT.md."
    )


def render_validation_badge(metric_name: str, scenario_context: str,
                            city: str = "San Antonio, TX",
                            explicit_status: str = None,
                            validated_path: bool = True) -> dict:
    """Return the badge state for a (metric, scenario-context) pair.

    Returns dict:
        text     — short label shown under the card
        tooltip  — scenario-aware, metric-aware; never overclaims
        color    — "green" / "blue" / "gray" — Streamlit display hint
        state    — "natcap_anchored" / "natcap_method" / "aligned_method" /
                   "prototype" / "unknown_metric"

    **Floor (B2-revised, 2026-05-29 — conservative posture):**

    - **Green** is reserved for the **fixed-scenario reference view** when the
      card displays NatCap's own published value directly — text reads
      **"NatCap published value"**. The card surfaces NatCap's number from the
      reference CSV; we don't claim reproduction.
    - **Blue ≈ NatCap method** for `natcap_published`-class metrics in **every
      other context** (BASELINE / EXPLORER / OPTIMIZER) — the displayed value is
      the prototype's own computation. Tooltip is METRIC-AWARE: temp cites
      measured per-pixel HMI parity (Brief 28b); carbon cites measured per-pixel
      parity too (four-pool framework validated vs InVEST 3.19.0, Relay 69).
    - **Blue ≈ Aligned method** for `aligned_method` metrics regardless of
      scenario context.
    - **Gray Prototype** for `prototype` metrics.

    Investigation under guardrails confirmed NatCap's published citywide
    absolutes (temp `avg_temp_f` 90.08 °F, carbon 107.32M t CO2e) aren't
    reproducible from disk — UCM args and carbon aggregation scripts aren't
    shipped. Per-pixel InVEST parity (Brief B / Brief 28b) is the validated
    claim. See `docs/internal/OPEN_QUESTIONS.md`.

    `explicit_status` lets the caller override the CSV lookup with a hand-
    curated status — used for non-CSV cards (runoff, NDVI, cost-effectiveness,
    carbon-$ on SA, etc.). See `docs/internal/DESIGN_NOTES.md` §8.1 "Two-surface
    validation vocabulary — locked" for the curated non-CSV-card status map.
    """
    if scenario_context not in ALL_SCENARIO_CONTEXTS:
        raise ValueError(
            f"unknown scenario_context {scenario_context!r}; "
            f"known: {ALL_SCENARIO_CONTEXTS}"
        )

    if explicit_status is not None:
        vstatus = explicit_status
    else:
        # Look up the metric's validation_status. It's per-metric-constant
        # across scenarios, so the baseline row is canonical.
        row = lookup_reference(city, "baseline", metric_name)
        vstatus = row.get("validation_status") if row else None

    # Fixed reference view: surfaces NatCap's published number directly (green).
    if (vstatus == "natcap_published"
            and scenario_context == SCENARIO_CONTEXT_NATCAP_FIXED):
        tooltip = (
            "Displayed value is **NatCap's published value for this "
            "scenario**, sourced from `natcap_reference_outputs.csv` "
            "(originally `nootenboom_results/citywide_results_UPDATED.xlsx`). "
            "The prototype does not independently reproduce it (compound "
            "scenario inputs are unavailable; see docs/internal/OPEN_QUESTIONS.md)."
        )
        return {"text": "NatCap published value", "tooltip": tooltip,
                "color": "green", "state": "natcap_anchored"}

    # Everyday view — InVEST-validated iff the card's model has measured per-pixel
    # parity (read from the Stage-1 canonical set, never re-hardcoded) AND it is on
    # the validated compute path (validated_path — e.g. carbon only when SA
    # four-pool stock, not the MN proxy). This extends the committed-reproducer
    # rule to the badge: no card claims validated unless the source says so.
    _model = _METRIC_TO_MODEL.get(metric_name)
    if (_model in model_validation.VALIDATED_MODELS and validated_path
            and scenario_context != SCENARIO_CONTEXT_NATCAP_FIXED):
        return {"text": "InVEST-validated",
                "tooltip": _badge_tooltip_for_metric(metric_name, vstatus),
                "color": "teal", "state": "invest_validated"}

    # Canonical-method basis, but parity not measured for THIS output (lumped
    # Flood Index / Runoff Volume, the natcap-published metrics off their fixed
    # view, and any aligned-method card whose model isn't validated).
    if vstatus in ("natcap_published", "aligned_method"):
        return {"text": "InVEST-aligned",
                "tooltip": _badge_tooltip_for_metric(metric_name, vstatus),
                "color": "blue", "state": "invest_aligned"}

    if vstatus == "prototype":
        tooltip = (
            "Exploratory metric — no canonical InVEST analog. Useful as a "
            "directional signal, not a validated number."
        )
        return {"text": "Prototype", "tooltip": tooltip,
                "color": "gray", "state": "prototype"}

    return {"text": None, "tooltip": None, "color": None,
            "state": "unknown_metric"}


if __name__ == "__main__":
    df = load_reference_outputs("San Antonio, TX")
    assert df is not None and len(df) == 49, f"expected 49 rows, got {0 if df is None else len(df)}"
    print("loaded rows:", len(df), "| statuses:", df["validation_status"].value_counts().to_dict())

    r = lookup_reference("San Antonio, TX", "FF_MAX", "temp_change_f")
    print("lookup temp FF_MAX:", r["natcap_value"], r["units"], r["validation_status"])

    # temp FF_MAX ref delta = 89.9646 − 90.0816 = −0.117 °F. Prototype −0.12 → match (|Δ|=0.003 ≤ 0.1).
    m1 = compare_to_reference("San Antonio, TX", "FF_MAX", "temp_change_f", -0.12)
    print("temp FF_MAX proto=-0.12:", m1["status"], f"Δabs={m1['delta_abs']:.4f}")
    assert m1["status"] == "match", m1
    # warmer outlier → diverged
    m2 = compare_to_reference("San Antonio, TX", "FF_MAX", "temp_change_f", 0.5)
    print("temp FF_MAX proto=+0.5:", m2["status"], f"Δabs={m2['delta_abs']:.4f}")
    assert m2["status"] == "diverged", m2
    # carbon FF_MAX ref delta ≈ +1.239e6 t CO2. Prototype 1.24e6 → match (~0.1% ≤ 1%).
    m3 = compare_to_reference("San Antonio, TX", "FF_MAX", "carbon_tons_co2", 1.239e6)
    print("carbon FF_MAX proto=1.239e6:", m3["status"], f"Δpct={m3['delta_pct']:.3f}%")
    assert m3["status"] == "match", m3
    # aligned_method → no_reference
    m4 = compare_to_reference("San Antonio, TX", "FF_MAX", "nature_access_pct", 94.0)
    print("nature FF_MAX:", m4["status"])
    assert m4["status"] == "no_reference", m4

    # ── badge helper smoke test ──
    print("\n=== render_validation_badge — three states × four contexts ===")
    metrics = ["temp_change_f", "carbon_tons_co2", "nature_access_pct",
               "preventable_mh_cases", "food_mln_lbs", "runoff_acre_feet"]
    for ctx in ALL_SCENARIO_CONTEXTS:
        print(f"\n  context = {ctx}")
        for m in metrics:
            b = render_validation_badge(m, ctx)
            txt = b["text"] or "(no row)"
            print(f"    {m:24s} → {txt:24s} state={b['state']}")

    # Stage 2 badge-tier checks:
    #  - natcap_fixed + natcap_published → GREEN "NatCap published value".
    #  - everyday + model in the Stage-1 validated set + validated path → TEAL
    #    "InVEST-validated" (temp/nature/MH/runoff-retention/SA-carbon).
    #  - everyday, canonical method but parity not measured for this output →
    #    BLUE "InVEST-aligned".
    #  - proxy / no canonical analog → GRAY "Prototype".
    b_base  = render_validation_badge("temp_change_f", SCENARIO_CONTEXT_BASELINE)
    b_fixed = render_validation_badge("temp_change_f", SCENARIO_CONTEXT_NATCAP_FIXED)
    b_expl  = render_validation_badge("temp_change_f", SCENARIO_CONTEXT_EXPLORER)
    b_opt   = render_validation_badge("temp_change_f", SCENARIO_CONTEXT_OPTIMIZER)

    # Fixed-scenario reference view: green "NatCap published value".
    assert b_fixed["state"] == "natcap_anchored", b_fixed
    assert b_fixed["color"] == "green", b_fixed
    assert b_fixed["text"] == "NatCap published value", b_fixed
    assert "natcap's published value" in b_fixed["tooltip"].lower(), b_fixed

    # Everyday view: temperature (UCM) is in the Stage-1 validated set → teal
    # "InVEST-validated" in baseline / explorer / optimizer.
    for _b in (b_base, b_expl, b_opt):
        assert _b["state"] == "invest_validated" and _b["color"] == "teal", _b
        assert _b["text"] == "InVEST-validated", _b
    # Tooltip still cites the measured HMI parity (bodies rewritten in Slice 3).
    assert "brief 28b" in b_base["tooltip"].lower(), b_base
    assert "hmi parity" in b_base["tooltip"].lower(), b_base

    # SA carbon (four-pool stock, validated path) → InVEST-validated.
    b_carbon = render_validation_badge("carbon_tons_co2", SCENARIO_CONTEXT_EXPLORER)
    assert b_carbon["state"] == "invest_validated", b_carbon
    assert "four-pool" in b_carbon["tooltip"].lower(), b_carbon
    assert "is measured" in b_carbon["tooltip"].lower(), b_carbon
    assert "per-pixel hmi parity" not in b_carbon["tooltip"].lower(), b_carbon
    # City-split: MN carbon is the proxy path (validated_path=False) → Prototype,
    # NOT validated — gated on the stock/proxy condition, not the city name.
    b_carbon_mn = render_validation_badge(
        "carbon_tons_co2", SCENARIO_CONTEXT_EXPLORER,
        explicit_status="prototype", validated_path=False)
    assert b_carbon_mn["state"] == "prototype" and b_carbon_mn["text"] == "Prototype", b_carbon_mn

    # carbon-$ is NOT in the validated map (a dollar valuation, not the per-pixel
    # stock output): everyday → InVEST-aligned; fixed view → green.
    b_cd_fixed = render_validation_badge("carbon_value_usd", SCENARIO_CONTEXT_NATCAP_FIXED,
                                         explicit_status="natcap_published")
    b_cd_expl  = render_validation_badge("carbon_value_usd", SCENARIO_CONTEXT_EXPLORER,
                                         explicit_status="natcap_published")
    assert b_cd_fixed["state"] == "natcap_anchored" and b_cd_fixed["color"] == "green", b_cd_fixed
    assert b_cd_expl["state"]  == "invest_aligned"  and b_cd_expl["color"]  == "blue",  b_cd_expl
    assert "four-pool" in b_cd_expl["tooltip"].lower(), b_cd_expl

    # Runoff Retention (UFR, validated set, non-CSV) → InVEST-validated; tooltip
    # cites the measured UFRM parity. Runoff Volume / Flood Index are lumped
    # proxies (model not validated for that output) → InVEST-aligned.
    b_ret = render_validation_badge("runoff_retention_idx", SCENARIO_CONTEXT_EXPLORER,
                                    explicit_status="aligned_method")
    assert b_ret["state"] == "invest_validated" and b_ret["text"] == "InVEST-validated", b_ret
    assert "is measured" in b_ret["tooltip"].lower(), b_ret
    assert "ufrm" in b_ret["tooltip"].lower(), b_ret
    b_runoff = render_validation_badge("runoff_acre_feet", SCENARIO_CONTEXT_EXPLORER,
                                       explicit_status="aligned_method")
    assert b_runoff["state"] == "invest_aligned" and b_runoff["text"] == "InVEST-aligned", b_runoff
    assert "is measured" not in b_runoff["tooltip"].lower(), b_runoff  # lumped proxy
    # Nature Access (UNA) + Preventable MH (UMH) are in the validated set too.
    for _m in ("nature_access_pct", "preventable_mh_cases"):
        _bm = render_validation_badge(_m, SCENARIO_CONTEXT_EXPLORER,
                                      explicit_status="aligned_method")
        assert _bm["state"] == "invest_validated" and _bm["color"] == "teal", (_m, _bm)
    # Flood Index is a lumped proxy (its model UFR is validated, but flood_reduction
    # isn't the per-pixel output) → InVEST-aligned, never validated.
    b_flood = render_validation_badge("flood_reduction", SCENARIO_CONTEXT_BASELINE)
    assert b_flood["state"] == "invest_aligned", b_flood
    # A prototype metric stays Prototype.
    b_food = render_validation_badge("food_mln_lbs", SCENARIO_CONTEXT_EXPLORER)
    assert b_food["state"] == "prototype", b_food

    # Non-CSV metric without explicit_status → unknown_metric.
    b_unk = render_validation_badge("runoff_acre_feet", SCENARIO_CONTEXT_EXPLORER)
    assert b_unk["state"] == "unknown_metric" and b_unk["text"] is None, b_unk
    print("\nsmoke test OK")

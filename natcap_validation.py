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

# Per-city reference CSV. MN has no nootenboom coverage yet (deferred).
_CITY_CSV = {"SA": "data/sa/natcap_reference_outputs.csv"}


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
    print("smoke test OK")

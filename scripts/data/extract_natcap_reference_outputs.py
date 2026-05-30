#!/usr/bin/env python3
"""extract_natcap_reference_outputs.py — build data/sa/natcap_reference_outputs.csv

Reads NatCap's published San Antonio citywide outputs and writes the prototype's
reference-validation CSV (the source of truth for "what does NatCap publish for
the SA project scenarios?"). Idempotent — re-running produces identical output.

Source (external NatCap deliverable, not tracked in this repo):
    ~/Desktop/natcap_drive_pull/nootenboom_results/citywide_results_UPDATED.xlsx
    (one sheet, long format: `<metric>_<scenario>` label → Value / Change).

Output (long format, one row per prototype-metric × scenario):
    data/sa/natcap_reference_outputs.csv

Schema columns:
    city, scenario_id, metric_name, natcap_value, units, tolerance_pct,
    tolerance_abs, validation_status, source_file, source_cell_or_row, notes

Validation states:
    natcap_published — NatCap published a directly-comparable reference value
                       (temp, carbon). Has tolerance.
    aligned_method   — canonical methodology, but the NatCap value isn't a
                       clean apples-to-apples comparison (nature: different
                       summary statistic; cooling: building-coverage scope
                       mismatch) OR NatCap published no SA reference (flood,
                       UMH). No tolerance check applies.
    prototype        — prototype-only metric, no canonical InVEST analog (food).

Values are stored ABSOLUTE with explicit baseline rows; the validation helper
derives deltas where the prototype reports a delta (temp_change_f,
carbon_tons_co2 stock change, cooling savings).

Usage:
    python extract_natcap_reference_outputs.py [--xlsx PATH] [--out PATH]
Requires pandas + openpyxl (e.g. anaconda base `python`).
"""
from __future__ import annotations
import argparse
import csv
import os

DEFAULT_XLSX = os.path.expanduser(
    "~/Desktop/natcap_drive_pull/nootenboom_results/citywide_results_UPDATED.xlsx")
DEFAULT_OUT = "data/sa/natcap_reference_outputs.csv"
SOURCE_FILE = "nootenboom_results/citywide_results_UPDATED.xlsx"

SCENARIOS = ["baseline", "FF_20ac", "FF_40ac", "FF_MAX", "UA_20ac", "UA_40ac", "UA_MAX"]
C_TO_CO2 = 44.0 / 12.0

# Prototype metrics with a NatCap reference value. Each: prototype metric_name,
# xlsx label builder, units, value transform, validation_status, tol_pct,
# tol_abs, note.
MAPPED = [
    dict(metric="temp_change_f", label=lambda s: f"avg_temp_f_{s}", units="deg_F",
         transform=lambda v: v, status="natcap_published", tol_pct=5.0, tol_abs=0.1,
         note="NatCap mean air temperature (absolute, °F). Prototype temp_change_f "
              "is a delta — compare to (scenario − baseline); lower = cooler."),
    dict(metric="carbon_tons_co2", label=lambda s: f"c_sequestration_{s}", units="tons_CO2",
         transform=lambda v: v * C_TO_CO2, status="natcap_published", tol_pct=1.0, tol_abs=None,
         note="NatCap four-pool carbon stock (tons C) × 44/12 → tons CO2 (absolute). "
              "Prototype carbon_tons_co2 is a stock change — compare to (scenario − baseline)."),
    dict(metric="nature_access_pct", label=lambda s: f"ntr_bal_avg_{s}", units="nature_balance_index",
         transform=lambda v: v, status="aligned_method", tol_pct=None, tol_abs=None,
         note="Different metric — NatCap reports a per-block-group balance aggregate "
              "(mean supply − demand ≈ 107); prototype reports pct-meeting-demand. A "
              "citywide per-pixel mean does NOT reproduce NatCap's value (per-pixel "
              "supply blows up at low-population pixels). Per-block-group prototype "
              "aggregation in Track C provides the comparable framing (see "
              "NATCAP_ALIGNMENT.md 'SA UNA / biophysical extent')."),
    dict(metric="cooling_energy_savings_usd", label=lambda s: f"annual_cdd_cost_{s}_sum",
         units="USD_per_yr", transform=lambda v: v, status="aligned_method",
         tol_pct=None, tol_abs=None,
         note="Scope difference — NatCap citywide all-buildings AC spend; prototype "
              "computes savings over typed-OSM buildings (~29% coverage). Absolute "
              "spend stored; savings derivable as (baseline − scenario)."),
]

# Prototype metrics with NO NatCap SA reference value (placeholder rows).
PLACEHOLDERS = [
    dict(metric="flood_reduction", units="index_0_100", status="aligned_method",
         note="No NatCap SA flood reference — NatCap used InVEST UFRM without damage "
              "valuation and published no flood metric in nootenboom_results."),
    dict(metric="preventable_mh_cases", units="cases_per_yr", status="aligned_method",
         note="UMH not in NatCap's SA project; prototype UMH is validated vs canonical "
              "InVEST 3.19.0 (Brief B, MAE≈0) but has no NatCap SA reference value."),
    dict(metric="food_mln_lbs", units="million_lbs_per_yr", status="prototype",
         note="No canonical InVEST analog — prototype food-forest yield benchmark."),
]

HEADER = ["city", "scenario_id", "metric_name", "natcap_value", "units",
          "tolerance_pct", "tolerance_abs", "validation_status",
          "source_file", "source_cell_or_row", "notes"]


def _fmt(v):
    if v is None:
        return ""
    return f"{v:.6g}"


def build_rows(xlsx_path):
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name="citywide_results")
    lookup = dict(zip(df["Unnamed: 0"].astype(str), df["Value"]))
    rows = []
    for m in MAPPED:
        for scn in SCENARIOS:
            label = m["label"](scn)
            if label not in lookup:
                raise KeyError(f"missing xlsx row {label!r} for metric {m['metric']}")
            val = m["transform"](float(lookup[label]))
            rows.append(["SA", scn, m["metric"], _fmt(val), m["units"],
                         _fmt(m["tol_pct"]), _fmt(m["tol_abs"]), m["status"],
                         SOURCE_FILE, label, m["note"]])
    for m in PLACEHOLDERS:
        for scn in SCENARIOS:
            rows.append(["SA", scn, m["metric"], "", m["units"], "", "",
                         m["status"], "", "", m["note"]])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    rows = build_rows(args.xlsx)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows "
          f"({sum(1 for r in rows if r[7]=='natcap_published')} natcap_published, "
          f"{sum(1 for r in rows if r[7]=='aligned_method')} aligned_method, "
          f"{sum(1 for r in rows if r[7]=='prototype')} prototype)")


if __name__ == "__main__":
    main()

"""
analyze_placement_diagnostic.py — turn the three CSVs from
placement_strategy_diagnostic.py into the tables and summaries that
populate docs/research/PLACEMENT_STRATEGY_DIAGNOSTIC.md.

Run after the diagnostic has finished:

    python3 diagnostics/analyze_placement_diagnostic.py

Prints markdown-ready tables to stdout.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("analysis/placement_diagnostic")

L1 = pd.read_csv(OUT_DIR / "layer1_suitability_variance.csv")
L2 = pd.read_csv(OUT_DIR / "layer2_chosen_pixel_scores.csv")
L3 = pd.read_csv(OUT_DIR / "layer3_metric_outcomes.csv")


def hdr(s):
    print(f"\n\n### {s}\n")


def md_table(df, fmt=None):
    """Print a DataFrame as a markdown table. `fmt` maps column -> format string."""
    cols = list(df.columns)
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("—")
            elif fmt and c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


# ── Layer 1: suitability surface variance ─────────────────────────────────────
hdr("Layer 1 — suitability surface variance")
l1_view = L1[["city", "strategy", "n_pixels", "mean", "std", "min",
              "p25", "p50", "p75", "p95", "max"]].copy()
md_table(l1_view, fmt={
    "n_pixels": "{:,}", "mean": "{:.4g}", "std": "{:.4g}",
    "min": "{:.4g}", "p25": "{:.4g}", "p50": "{:.4g}",
    "p75": "{:.4g}", "p95": "{:.4g}", "max": "{:.4g}",
})


# ── Layer 2: chosen-pool vs overall-pool gap ──────────────────────────────────
hdr("Layer 2 — chosen-pool mean score vs overall-pool mean (gap, averaged over 10 seeds)")
# `saturated` is now a marker (Brief 7 fix to _select_pixels_for_conversion),
# not a skip flag. Saturated rows have valid chosen-pool scores but reflect
# the non-zero-then-uniform-remainder fallback path. Surface the count for
# context, then include saturated rows in the gap computation.
sat = L2[L2["saturated"] == True]  # noqa: E712
if len(sat):
    print(f"\n_Saturated combos (strategy fell into the non-zero-then-"
          f"uniform-remainder fallback because n_chosen exceeded the "
          f"non-zero pixel count):_ {len(sat)} of {len(L2)}. Values are "
          f"valid but dilution by the uniform remainder shrinks the gap.")

ok = L2  # include all rows; saturated is informational, not a skip signal
g = ok.groupby(["city", "strategy", "pct"]).agg(
    gap_mean=("gap", "mean"),
    gap_std=("gap", "std"),
    chosen_mean=("chosen_pool_mean_score", "mean"),
    overall_mean=("overall_pool_mean_score", "first"),
    surface=("scoring_surface", "first"),
    n_seeds=("seed", "count"),
).reset_index()

# Random's std on the gap (over 10 seeds) is the noise floor — anything
# within 2× of it is not significantly different from random.
random_std = (g[g["strategy"] == "random"]
              .set_index(["city", "pct"])["gap_std"]
              .to_dict())
def is_significant(row):
    rs = random_std.get((row["city"], row["pct"]), np.nan)
    if pd.isna(rs) or rs == 0:
        return ""
    return "✓" if abs(row["gap_mean"]) > 2 * rs else "—"

g["sig"] = g.apply(is_significant, axis=1)
md_table(g[["city", "strategy", "pct", "n_seeds", "gap_mean", "gap_std",
            "chosen_mean", "overall_mean", "surface", "sig"]],
         fmt={"gap_mean": "{:+.4g}", "gap_std": "{:.4g}",
              "chosen_mean": "{:.4g}", "overall_mean": "{:.4g}"})


# ── Layer 3: metric outcomes — strategy delta vs random ───────────────────────
hdr("Layer 3 — metric outcomes (delta vs random, averaged over 10 seeds)")
sat3 = L3[L3["saturated"] == True]  # noqa: E712
if len(sat3):
    print(f"\n_Saturated combos:_ {len(sat3)} of {len(L3)}.")

ok3 = L3[L3["saturated"] == False].copy()  # noqa: E712
metrics = ["flood_reduction", "mean_hm", "food_mln_lbs",
           "carbon_tons_co2", "runoff_acre_feet"]

for m in metrics:
    hdr(f"Layer 3 — {m}")
    # Mean per (city, strategy, scenario, pct)
    grp = ok3.groupby(["city", "strategy", "scenario", "pct"])[m].agg(["mean", "std"]).reset_index()
    # Pivot so random's mean is alongside each weighted strategy's mean
    rand = grp[grp["strategy"] == "random"].set_index(["city", "scenario", "pct"])
    rand_mean = rand["mean"].to_dict()
    rand_std = rand["std"].to_dict()
    grp["random_mean"] = grp.apply(
        lambda r: rand_mean.get((r["city"], r["scenario"], r["pct"]), np.nan), axis=1)
    grp["random_std"] = grp.apply(
        lambda r: rand_std.get((r["city"], r["scenario"], r["pct"]), np.nan), axis=1)
    grp["delta"] = grp["mean"] - grp["random_mean"]
    grp["sig"] = grp.apply(
        lambda r: "" if r["strategy"] == "random" else
                  ("—" if pd.isna(r["random_std"]) or r["random_std"] == 0
                   else ("✓" if abs(r["delta"]) > 2 * r["random_std"] else "—")),
        axis=1)
    show = grp[grp["strategy"] != "random"][
        ["city", "strategy", "scenario", "pct", "mean", "random_mean",
         "delta", "random_std", "sig"]
    ]
    md_table(show, fmt={
        "mean": "{:.4g}", "random_mean": "{:.4g}",
        "delta": "{:+.4g}", "random_std": "{:.4g}",
    })


# ── Cross-city aggregate: max delta per (city, strategy) across all scenarios ─
hdr("Cross-city aggregate — max |delta vs random| per (city, strategy, metric)")
roll_rows = []
for m in metrics:
    grp = ok3.groupby(["city", "strategy", "scenario", "pct"])[m].agg(["mean"]).reset_index()
    rand = grp[grp["strategy"] == "random"].set_index(["city", "scenario", "pct"])["mean"].to_dict()
    grp["delta"] = grp.apply(
        lambda r: r["mean"] - rand.get((r["city"], r["scenario"], r["pct"]), np.nan), axis=1)
    g2 = grp[grp["strategy"] != "random"].groupby(["city", "strategy"])["delta"].apply(
        lambda s: s.abs().max()).reset_index()
    g2["metric"] = m
    roll_rows.append(g2)
roll = pd.concat(roll_rows, ignore_index=True)
roll = roll.pivot(index=["city", "strategy"], columns="metric", values="delta").reset_index()
md_table(roll, fmt={
    "flood_reduction": "{:.3g}", "mean_hm": "{:.4g}",
    "food_mln_lbs": "{:.4g}", "carbon_tons_co2": "{:.4g}",
    "runoff_acre_feet": "{:.4g}",
})


# ── Per-call timing summary ──────────────────────────────────────────────────
hdr("Per-call wall time (seconds, from elapsed_s column)")
tim = ok3.groupby("city")["elapsed_s"].agg(["count", "mean", "median", "max"]).reset_index()
md_table(tim, fmt={"mean": "{:.3f}", "median": "{:.3f}", "max": "{:.3f}"})

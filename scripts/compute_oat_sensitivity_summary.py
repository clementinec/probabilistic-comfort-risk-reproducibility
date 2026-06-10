#!/usr/bin/env python3
"""Recompute the compact annual OAT sensitivity summaries.

This script reads the archived OAT elasticity matrix reported in the
Supplementary Material and verifies the max-elasticity ranking. If a raw
run-level table named ``annual_oat_sweep_results.csv`` is placed next to the
summary CSVs, the script also recomputes the EUI coefficient of variation from
that table. Otherwise it reports the archived EUI CV summary.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev


METRICS = ["wDDE", "PMV_Vio", "AC_Vio", "Kh", "EUI"]


def find_base() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, here / "summary_outputs", here.parent / "summary_outputs"]
    for candidate in candidates:
        if (candidate / "annual_oat_elasticity_summary.csv").exists():
            return candidate
    raise FileNotFoundError("annual_oat_elasticity_summary.csv not found")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verify_elasticity_summary(base: Path) -> list[dict[str, object]]:
    rows = read_rows(base / "annual_oat_elasticity_summary.csv")
    verified: list[dict[str, object]] = []
    for row in rows:
        metric_values = {metric: float(row[metric]) for metric in METRICS}
        max_metric, max_value = max(metric_values.items(), key=lambda item: item[1])
        verified.append(
            {
                "parameter": row["parameter"],
                "max_metric": max_metric,
                "computed_max_elasticity": round(max_value, 2),
                "archived_max_elasticity": float(row["max_elasticity"]),
                "archived_rank": int(row["sensitivity_rank"]),
            }
        )

    ranked = sorted(verified, key=lambda row: row["computed_max_elasticity"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["computed_rank"] = rank
    return ranked


def eui_cv(base: Path) -> tuple[str, float]:
    raw = base / "annual_oat_sweep_results.csv"
    if raw.exists():
        rows = read_rows(raw)
        eui_values = [float(row["EUI"]) for row in rows if row.get("EUI") not in (None, "")]
        if not eui_values:
            raise ValueError("annual_oat_sweep_results.csv contains no EUI values")
        return "raw_run_table", 100.0 * pstdev(eui_values) / mean(eui_values)

    archived = read_rows(base / "annual_oat_eui_cv_summary.csv")
    return "archived_summary", float(archived[0]["coefficient_of_variation_pct"])


def main() -> None:
    base = find_base()
    print(f"Using OAT summary directory: {base}")
    print("\nAnnual OAT max-elasticity ranking:")
    for row in verify_elasticity_summary(base):
        print(
            f"  rank {row['computed_rank']}: {row['parameter']} "
            f"max={row['computed_max_elasticity']:.2f} via {row['max_metric']} "
            f"(archived rank {row['archived_rank']})"
        )

    source, cv = eui_cv(base)
    print(f"\nEUI coefficient of variation: {cv:.1f}% ({source})")


if __name__ == "__main__":
    main()

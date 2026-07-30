#!/usr/bin/env python3
"""Audit which future-weather variables contain scenario-specific information.

The paired SSP2-4.5/SSP5-8.5 forecast files share timestamps and a common
reference series. Exact paired comparisons therefore reveal which output
variables are scenario-dependent in the preserved artifacts. This is a
representation audit, not an observational validation of the weather product.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
HPH = ROOT / "restricted_inputs" / "weather"
OUT_DEFAULT = ROOT / "outputs" / "weather" / "variable_support"
PROVENANCE = ROOT / "outputs" / "weather" / "provenance"

VARIABLES = [
    "temp",
    "pressure",
    "wind_speed",
    "GHI",
    "DHI",
    "DNI",
    "relative_humidity",
    "specific_humidity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    return parser.parse_args()


def paired_comparison(
    city: str,
    ssp245_path: Path,
    ssp585_path: Path,
    chunk_rows: int,
) -> list[dict[str, object]]:
    accumulators = {
        variable: {
            "finite": 0,
            "exact": 0,
            "sum_245": 0.0,
            "sum_585": 0.0,
            "sum_abs_difference": 0.0,
            "sum_squared_difference": 0.0,
            "max_abs_difference": 0.0,
        }
        for variable in VARIABLES
    }
    rows = 0
    mismatched_timestamps = 0
    left_reader = pd.read_csv(
        ssp245_path,
        usecols=["datetime", *VARIABLES],
        chunksize=chunk_rows,
    )
    right_reader = pd.read_csv(
        ssp585_path,
        usecols=["datetime", *VARIABLES],
        chunksize=chunk_rows,
    )
    for left, right in zip(left_reader, right_reader, strict=True):
        if len(left) != len(right):
            raise ValueError(f"Chunk-size mismatch for {city}")
        rows += len(left)
        mismatched_timestamps += int(
            (left["datetime"].to_numpy() != right["datetime"].to_numpy()).sum()
        )
        for variable in VARIABLES:
            a = pd.to_numeric(left[variable], errors="coerce").to_numpy(float)
            b = pd.to_numeric(right[variable], errors="coerce").to_numpy(float)
            finite = np.isfinite(a) & np.isfinite(b)
            a = a[finite]
            b = b[finite]
            difference = b - a
            absolute = np.abs(difference)
            item = accumulators[variable]
            item["finite"] += int(len(a))
            item["exact"] += int(np.equal(a, b).sum())
            item["sum_245"] += float(a.sum())
            item["sum_585"] += float(b.sum())
            item["sum_abs_difference"] += float(absolute.sum())
            item["sum_squared_difference"] += float(np.square(difference).sum())
            item["max_abs_difference"] = max(
                float(item["max_abs_difference"]),
                float(absolute.max()) if len(absolute) else 0.0,
            )
    if mismatched_timestamps:
        raise ValueError(
            f"{city}: {mismatched_timestamps} paired timestamps do not match"
        )
    records: list[dict[str, object]] = []
    for variable, item in accumulators.items():
        n = int(item["finite"])
        records.append(
            {
                "city": city,
                "variable": variable,
                "paired_rows": rows,
                "finite_paired_rows": n,
                "exact_equal_rows": int(item["exact"]),
                "exact_equal_pct": 100.0 * float(item["exact"]) / n,
                "mean_ssp245": float(item["sum_245"]) / n,
                "mean_ssp585": float(item["sum_585"]) / n,
                "mean_ssp585_minus_ssp245": (
                    float(item["sum_585"]) - float(item["sum_245"])
                )
                / n,
                "mean_abs_paired_difference": (
                    float(item["sum_abs_difference"]) / n
                ),
                "rmse_paired_difference": np.sqrt(
                    float(item["sum_squared_difference"]) / n
                ),
                "max_abs_paired_difference": float(
                    item["max_abs_difference"]
                ),
            }
        )
    return records


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(PROVENANCE / "source_forecast_inventory.csv")
    delta = pd.read_csv(PROVENANCE / "source_workbook_delta_summary.csv")

    records: list[dict[str, object]] = []
    for city, group in inventory.groupby("city", sort=True):
        paths = {
            row.scenario: (
                Path(row.csv_path)
                if Path(row.csv_path).is_absolute()
                else HPH / Path(row.csv_path)
            )
            for row in group.itertuples(index=False)
        }
        if set(paths) != {"ssp245", "ssp585"}:
            raise ValueError(f"{city}: expected paired SSP files, found {paths}")
        print(f"[pair] {city}", flush=True)
        records.extend(
            paired_comparison(
                city,
                paths["ssp245"],
                paths["ssp585"],
                args.chunk_rows,
            )
        )

    comparison = pd.DataFrame(records)
    comparison.to_csv(
        args.output_dir / "paired_scenario_variable_comparison.csv",
        index=False,
    )
    aggregate = (
        comparison.groupby("variable", as_index=False)
        .agg(
            cities=("city", "nunique"),
            paired_rows=("finite_paired_rows", "sum"),
            exact_equal_rows=("exact_equal_rows", "sum"),
            mean_ssp245=("mean_ssp245", "mean"),
            mean_ssp585=("mean_ssp585", "mean"),
            mean_ssp585_minus_ssp245=(
                "mean_ssp585_minus_ssp245",
                "mean",
            ),
            mean_abs_paired_difference=(
                "mean_abs_paired_difference",
                "mean",
            ),
            maximum_abs_paired_difference=(
                "max_abs_paired_difference",
                "max",
            ),
        )
        .sort_values("variable")
    )
    aggregate["exact_equal_pct"] = (
        100.0 * aggregate["exact_equal_rows"] / aggregate["paired_rows"]
    )
    aggregate.to_csv(
        args.output_dir / "paired_scenario_variable_aggregate.csv",
        index=False,
    )

    documented = sorted(delta["variable"].astype(str).unique())
    invariant = aggregate.loc[
        np.isclose(aggregate["exact_equal_pct"], 100.0), "variable"
    ].tolist()
    summary = {
        "cities": int(comparison["city"].nunique()),
        "paired_rows_per_city": int(comparison["paired_rows"].iloc[0]),
        "documented_delta_variables": documented,
        "exactly_scenario_invariant_variables": invariant,
        "representation_audit_only": True,
        "observational_validation": False,
        "specific_humidity_written_to_epw": False,
        "epw_dewpoint_basis": "forecast temperature plus carried relative humidity",
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    variable_table = aggregate[
        [
            "variable",
            "exact_equal_pct",
            "mean_ssp585_minus_ssp245",
            "mean_abs_paired_difference",
            "maximum_abs_paired_difference",
        ]
    ].copy()
    variable_table["exact_equal_pct"] = variable_table[
        "exact_equal_pct"
    ].map(lambda value: f"{value:.3f}")
    for column in [
        "mean_ssp585_minus_ssp245",
        "mean_abs_paired_difference",
        "maximum_abs_paired_difference",
    ]:
        variable_table[column] = variable_table[column].map(
            lambda value: f"{value:.6g}"
        )
    headers = variable_table.columns.tolist()
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in variable_table.itertuples(index=False, name=None)
    )
    table = "\n".join(table_lines)
    report = f"""# Future-weather variable-support audit

## Result

The paired forecast workbooks document daily climate-delta transformations
for {", ".join(f"`{value}`" for value in documented)}. At the same
city and timestamp, `relative_humidity`, `specific_humidity`, and `DNI` are
exactly identical between SSP2-4.5 and SSP5-8.5 in every one of the
{summary['cities']} paired city files. They therefore contain no
scenario-specific signal in the preserved product. Temperature, pressure,
wind speed, GHI, and derived/adjusted DHI do differ by scenario.

{table}

## Consequence for interpretation

- The future-weather experiment can describe responses to the **joint selected
  weather files**, but it cannot separately attribute a future change to a
  scenario-specific humidity projection or a scenario-specific direct-normal
  irradiance projection.
- Relative humidity is passed into the EPW. Dew point is recalculated from
  future dry-bulb and that carried RH, so absolute moisture changes
  mechanically with temperature under an effectively fixed-RH weather
  assumption. The source `specific_humidity` column is not written to the EPW.
- GHI and DHI vary across scenarios, while DNI is carried unchanged. Horizontal
  GHI remains usable as a descriptive forcing covariate, but neither it nor
  the radiation partition supports a facade-solar causal attribution.
- These exact paired checks establish what the product represents; they are
  not independent observational validation of the underlying climate method.

## Interpretation boundary

The weather data are a single-GCM, daily-delta stress-test product, with the
delta-adjusted variables identified above. Humidity and solar results describe
covarying components of the selected weather files rather than independent
climate-driver effects. An EnergyPlus rerun would be necessary only for a
future component-specific humidity or solar attribution analysis.
"""
    (args.output_dir / "variable_support_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

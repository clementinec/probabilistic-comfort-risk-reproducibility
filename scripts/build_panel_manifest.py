#!/usr/bin/env python3
"""Build the weather-case manifest from the selected EPW panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_INPUTS = REPO_ROOT / "restricted_inputs" / "weather"
DEFAULT_PANEL = RESTRICTED_INPUTS / "cmip_weather_panel.csv"
DEFAULT_WEATHER_ROOT = RESTRICTED_INPUTS / "selected_epws"
DEFAULT_OUT = REPO_ROOT / "data" / "panel_manifest.csv"

SMOKE_CASES = {
    ("Phoenix", "ssp585", "baseline_2020s", "typical"),
    ("Phoenix", "ssp585", "late_2080s", "heatwave_extreme"),
    ("Kolkata", "ssp245", "baseline_2020s", "typical"),
    ("Guangzhou", "ssp585", "mid_2050s", "hot"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--weather-root", type=Path, default=DEFAULT_WEATHER_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def build_epw_path(row: pd.Series, weather_root: Path) -> Path:
    city = str(row["city"])
    scenario = str(row["scenario_raw"])
    time_slice = str(row["time_slice"])
    severity = str(row["severity"])
    year = int(row["weather_year"])
    filename = f"{city.lower()}_{scenario}_{time_slice}_{severity}_{year}.epw"
    return weather_root / scenario / city / filename


def make_case_id(row: pd.Series) -> str:
    return "_".join(
        [
            str(row["city"]).lower(),
            str(row["scenario_raw"]),
            str(row["time_slice"]),
            str(row["severity"]),
            str(int(row["weather_year"])),
        ]
    )


def main() -> int:
    args = parse_args()
    panel = pd.read_csv(args.panel)
    panel["epw_path"] = panel.apply(lambda row: str(build_epw_path(row, args.weather_root)), axis=1)
    panel["case_id"] = panel.apply(make_case_id, axis=1)
    panel["stage_smoke"] = panel.apply(
        lambda row: (
            str(row["city"]),
            str(row["scenario_raw"]),
            str(row["time_slice"]),
            str(row["severity"]),
        )
        in SMOKE_CASES,
        axis=1,
    )
    panel["stage_typical"] = panel["severity"].eq("typical")
    panel["stage_full"] = True
    panel["epw_exists"] = panel["epw_path"].map(lambda path: Path(path).exists())

    missing = panel.loc[~panel["epw_exists"], ["case_id", "epw_path"]]
    if not missing.empty:
        preview = "\n".join(f"{r.case_id}: {r.epw_path}" for r in missing.itertuples())
        raise FileNotFoundError(f"Missing EPW files:\n{preview}")

    ordered_cols = [
        "case_id",
        "country",
        "city",
        "climate_role",
        "scenario",
        "scenario_raw",
        "time_slice",
        "year_window",
        "severity",
        "weather_year",
        "epw_path",
        "stage_smoke",
        "stage_typical",
        "stage_full",
        "mean_T_out",
        "CDD18_hourly",
        "max_T_out",
        "hours_temp_ge_35",
        "humidity_metric",
        "selector",
        "source",
        "notes",
    ]
    out = panel[ordered_cols].sort_values(
        ["city", "scenario_raw", "time_slice", "severity", "weather_year"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[manifest] wrote {len(out)} cases: {args.output}")
    print(f"[manifest] smoke cases: {int(out['stage_smoke'].sum())}")
    print(f"[manifest] typical cases: {int(out['stage_typical'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

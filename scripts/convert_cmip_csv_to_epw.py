#!/usr/bin/env python
"""Convert one year of CMIP forecast CSV weather to minimal EPW files.

The CMIP CSVs provide the weather variables needed by the Medium Office
diagnostic rerun: dry-bulb temperature, pressure, wind speed, GHI, DHI, DNI,
and relative humidity. Missing EPW fields are filled with the same conservative
placeholders used by the existing HPH selected-year EPWs.
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CMIP_ROOT = ROOT / "HPH_Carbon_Entitlement" / "CMIPs"
DEFAULT_OUT = ROOT / "Probabilities_ENB" / "generated_epw_diagnostic"


@dataclass(frozen=True)
class CityMeta:
    name: str
    latitude: float
    longitude: float
    timezone: float
    elevation_m: float


CITY_META = {
    "Phoenix": CityMeta("Phoenix", 33.4484, -112.0740, -7.0, 331.0),
    "Beijing": CityMeta("Beijing", 39.9042, 116.4074, 8.0, 44.0),
    "Houston": CityMeta("Houston", 29.7604, -95.3698, -6.0, 13.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="ssp585", choices=["ssp245", "ssp585"])
    parser.add_argument("--model-tag", default="CMIP6_MPI_0515")
    parser.add_argument("--year", type=int, default=2038)
    parser.add_argument("--cities", nargs="+", default=["Phoenix", "Beijing", "Houston"])
    parser.add_argument("--cmip-root", type=Path, default=DEFAULT_CMIP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def source_path(cmip_root: Path, model_tag: str, scenario: str, city: str) -> Path:
    return (
        cmip_root
        / f"{model_tag}_{scenario}"
        / city
        / f"forecast_{city}_{model_tag}_{scenario}.csv"
    )


def dewpoint_c(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    rh = np.clip(rh_pct, 0.1, 100.0) / 100.0
    a = 17.625
    b = 243.04
    gamma = np.log(rh) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def epw_header(meta: CityMeta, scenario: str, year: int, model_tag: str) -> list[str]:
    first_weekday = calendar.day_name[pd.Timestamp(year=year, month=1, day=1).weekday()]
    return [
        (
            f"LOCATION,{meta.name},,Synthetic,{model_tag},{year},"
            f"{meta.latitude:.4f},{meta.longitude:.4f},{meta.timezone:.1f},{meta.elevation_m:.0f}"
        ),
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,Generated from CMIP forecast CSV for probability-necessity diagnostic.",
        f"COMMENTS 2,{meta.name} {scenario} direct CMIP forecast year {year}.",
        f"DATA PERIODS,1,1,Data,{first_weekday},1/1,12/31",
    ]


def convert_city(
    *,
    city: str,
    scenario: str,
    model_tag: str,
    year: int,
    cmip_root: Path,
    output_dir: Path,
) -> Path:
    if city not in CITY_META:
        raise ValueError(f"Missing city metadata for {city}")
    src = source_path(cmip_root, model_tag, scenario, city)
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src, parse_dates=["datetime"])
    df = df[df["datetime"].dt.year == year].copy()
    if len(df) != 8760:
        raise ValueError(f"{src} year {year} has {len(df)} rows, expected 8760.")
    df = df.sort_values("datetime")

    temp = pd.to_numeric(df["temp"], errors="coerce").to_numpy(float)
    rh = pd.to_numeric(df["relative_humidity"], errors="coerce").to_numpy(float)
    pressure = pd.to_numeric(df["pressure"], errors="coerce").to_numpy(float)
    wind_speed = pd.to_numeric(df["wind_speed"], errors="coerce").to_numpy(float)
    ghi = pd.to_numeric(df["GHI"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    dni = pd.to_numeric(df["DNI"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    dhi = pd.to_numeric(df["DHI"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)

    temp = np.nan_to_num(temp, nan=float(np.nanmedian(temp)))
    rh = np.clip(np.nan_to_num(rh, nan=float(np.nanmedian(rh))), 0.0, 100.0)
    pressure = np.nan_to_num(pressure, nan=float(np.nanmedian(pressure)))
    if np.nanmedian(pressure) < 2000.0:
        pressure = pressure * 100.0
    pressure = np.clip(pressure, 50000.0, 110000.0)
    wind_speed = np.clip(np.nan_to_num(wind_speed, nan=0.0), 0.0, 40.0)
    dew = dewpoint_c(temp, rh)

    meta = CITY_META[city]
    out_dir = output_dir / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{city.lower()}_{scenario}_cmip_direct_{year}.epw"

    lines = epw_header(meta, scenario, year, model_tag)
    for row_idx, row in enumerate(df.itertuples(index=False)):
        dt = row.datetime
        hour = int(dt.hour) + 1
        line = [
            year,
            int(dt.month),
            int(dt.day),
            hour,
            60,
            "?9?9?9?9",
            f"{temp[row_idx]:.2f}",
            f"{dew[row_idx]:.2f}",
            int(round(rh[row_idx])),
            int(round(pressure[row_idx])),
            9999,
            9999,
            9999,
            int(round(ghi[row_idx])),
            int(round(dni[row_idx])),
            int(round(dhi[row_idx])),
            999999,
            999999,
            999999,
            9999,
            180,
            f"{wind_speed[row_idx]:.2f}",
            9,
            9,
            9999,
            99999,
            9,
            999999999,
            999,
            0.999,
            0,
            88,
            0.2,
            0,
            0,
        ]
        lines.append(",".join(map(str, line)))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    args = parse_args()
    outputs = []
    for city in args.cities:
        path = convert_city(
            city=city,
            scenario=args.scenario,
            model_tag=args.model_tag,
            year=args.year,
            cmip_root=args.cmip_root,
            output_dir=args.output_dir,
        )
        outputs.append(path)
        print(f"[write] {path}")
    print(f"[done] wrote {len(outputs)} EPW files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

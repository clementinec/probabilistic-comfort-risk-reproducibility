#!/usr/bin/env python3
"""Build a zone-size mosaic heat map for Paper A zone aggregation diagnostics."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "runs" / "diagnostic_reference_zone_raw_full" / "traces"
EPLUS_DIR = ROOT / "runs" / "diagnostic_reference_zone_raw_full" / "energyplus"
OUT = ROOT / "diagnostics" / "zone_size_mosaic"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAIL_DIAGNOSTIC = 0.20

CITY_ORDER = ["Ahmedabad", "Beijing", "Guangzhou", "Houston", "Kolkata", "Phoenix"]
SCENARIO_ORDER = ["ssp245", "ssp585"]
TIME_ORDER = ["baseline_2020s", "near_2030s", "mid_2050s", "late_2080s"]
SEVERITY_ORDER = ["typical", "hot", "heatwave_extreme"]

ZONE_ORDER = [
    "core_bottom",
    "perimeter_bot_zn_1",
    "perimeter_bot_zn_2",
    "perimeter_bot_zn_3",
    "perimeter_bot_zn_4",
    "core_mid",
    "perimeter_mid_zn_1",
    "perimeter_mid_zn_2",
    "perimeter_mid_zn_3",
    "perimeter_mid_zn_4",
    "core_top",
    "perimeter_top_zn_1",
    "perimeter_top_zn_2",
    "perimeter_top_zn_3",
    "perimeter_top_zn_4",
]

ZONE_LABELS = {
    "core_bottom": "Core bottom",
    "core_mid": "Core middle",
    "core_top": "Core top",
    "perimeter_bot_zn_1": "Bottom P1",
    "perimeter_bot_zn_2": "Bottom P2",
    "perimeter_bot_zn_3": "Bottom P3",
    "perimeter_bot_zn_4": "Bottom P4",
    "perimeter_mid_zn_1": "Middle P1",
    "perimeter_mid_zn_2": "Middle P2",
    "perimeter_mid_zn_3": "Middle P3",
    "perimeter_mid_zn_4": "Middle P4",
    "perimeter_top_zn_1": "Top P1",
    "perimeter_top_zn_2": "Top P2",
    "perimeter_top_zn_3": "Top P3",
    "perimeter_top_zn_4": "Top P4",
}

FALLBACK_AREAS_M2 = {
    "core_bottom": 983.54,
    "core_mid": 983.54,
    "core_top": 983.54,
    "perimeter_bot_zn_1": 207.34,
    "perimeter_bot_zn_2": 131.26,
    "perimeter_bot_zn_3": 207.34,
    "perimeter_bot_zn_4": 131.25,
    "perimeter_mid_zn_1": 207.34,
    "perimeter_mid_zn_2": 131.26,
    "perimeter_mid_zn_3": 207.34,
    "perimeter_mid_zn_4": 131.25,
    "perimeter_top_zn_1": 207.34,
    "perimeter_top_zn_2": 131.26,
    "perimeter_top_zn_3": 207.34,
    "perimeter_top_zn_4": 131.25,
}

WEATHER_RE = re.compile(
    r"^(?P<city>ahmedabad|beijing|guangzhou|houston|kolkata|phoenix)_"
    r"(?P<scenario_raw>ssp245|ssp585)_"
    r"(?P<time_slice>baseline_2020s|near_2030s|mid_2050s|late_2080s)_"
    r"(?P<severity>typical|hot|heatwave_extreme)_"
    r"(?P<weather_year>\d{4})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--energyplus-dir", type=Path, default=EPLUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=TAIL_DIAGNOSTIC,
        help="High-tail cutoff applied to each zone probability.",
    )
    return parser.parse_args()


def slug_from_zone_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_weather_metadata(weather: str) -> dict[str, object]:
    match = WEATHER_RE.match(str(weather))
    if not match:
        raise ValueError(f"Could not parse weather id: {weather}")
    data = match.groupdict()
    return {
        "city": data["city"].title(),
        "scenario_raw": data["scenario_raw"],
        "time_slice": data["time_slice"],
        "severity": data["severity"],
        "weather_year": int(data["weather_year"]),
    }


def sort_key(meta: pd.Series) -> tuple[int, int, int, int, int]:
    return (
        CITY_ORDER.index(meta["city"]),
        SCENARIO_ORDER.index(meta["scenario_raw"]),
        TIME_ORDER.index(meta["time_slice"]),
        SEVERITY_ORDER.index(meta["severity"]),
        int(meta["weather_year"]),
    )


def discover_zone_probability_columns(trace_path: Path) -> list[str]:
    header = pd.read_csv(trace_path, nrows=0).columns
    cols = [
        col
        for col in header
        if col.startswith("zone_")
        and col.endswith("_p_disc")
        and not col.startswith("zone_heating")
        and not col.startswith("zone_cooling")
    ]
    expected = [f"zone_{zone}_p_disc" for zone in ZONE_ORDER]
    missing = [col for col in expected if col not in cols]
    if missing:
        raise ValueError(f"Missing expected zone probability columns: {missing}")
    return expected


def read_zone_areas(energyplus_dir: Path) -> dict[str, float]:
    eio_paths = sorted(energyplus_dir.glob("*/diagnostic_reference/eplusout.eio"))
    if not eio_paths:
        return FALLBACK_AREAS_M2.copy()

    lines = eio_paths[0].read_text(errors="ignore").splitlines()
    header = next((line for line in lines if line.startswith("! <Zone Information>")), "")
    header_parts = [part.strip() for part in header.split(",")]
    try:
        area_idx = next(i for i, part in enumerate(header_parts) if part.startswith("Floor Area"))
        total_area_idx = next(
            i for i, part in enumerate(header_parts) if part.startswith("Part of Total Building Area")
        )
    except StopIteration:
        return FALLBACK_AREAS_M2.copy()

    areas: dict[str, float] = {}
    for line in lines:
        if not line.startswith(" Zone Information,"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) <= max(area_idx, total_area_idx):
            continue
        if parts[total_area_idx].lower() != "yes":
            continue
        zone = slug_from_zone_name(parts[1])
        if zone in ZONE_ORDER:
            areas[zone] = float(parts[area_idx])

    for zone, area in FALLBACK_AREAS_M2.items():
        areas.setdefault(zone, area)
    return {zone: areas[zone] for zone in ZONE_ORDER}


def pct(mask: pd.Series | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean() * 100.0)


def build_case_rows(
    trace_dir: Path, p_cols: list[str], areas: dict[str, float], threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    if not paths:
        raise FileNotFoundError(f"No diagnostic trace files found in {trace_dir}")

    usecols = ["weather", "occupied", "discomfort_probability", *p_cols]
    zone_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    area_vector = np.array([areas[zone] for zone in ZONE_ORDER], dtype=float)

    for path in paths:
        df = pd.read_csv(path, usecols=usecols)
        df = df[df["occupied"]].copy()
        if df.empty:
            continue
        weather = str(df["weather"].iloc[0])
        meta = parse_weather_metadata(weather)
        p = df[p_cols].astype(float)
        p_arr = p.to_numpy()
        high = p_arr >= threshold
        area_weighted_high = np.average(high.astype(float), axis=1, weights=area_vector)
        area_weighted_p = np.average(p_arr, axis=1, weights=area_vector)

        for zone, col in zip(ZONE_ORDER, p_cols, strict=True):
            values = p[col]
            row = {
                **meta,
                "weather": weather,
                "zone": zone,
                "zone_label": ZONE_LABELS[zone],
                "area_m2": areas[zone],
                "area_share_pct": areas[zone] / float(area_vector.sum()) * 100.0,
                "mean_p_tail": float(values.mean()),
                "p95_p_tail": float(values.quantile(0.95)),
                "high_tail_pct": pct(values.ge(threshold)),
                "occupied_records": int(len(values)),
            }
            zone_rows.append(row)

        mean_high = df["discomfort_probability"].ge(threshold)
        any_zone_high = high.any(axis=1)
        max_zone_p = p.max(axis=1)
        case_rows.append(
            {
                **meta,
                "weather": weather,
                "occupied_records": int(len(df)),
                "mean_zone_high_tail_pct": pct(mean_high),
                "any_zone_high_tail_pct": pct(any_zone_high),
                "hidden_any_zone_high_tail_pct": pct((~mean_high.to_numpy()) & any_zone_high),
                "area_weighted_zone_time_high_tail_pct": float(area_weighted_high.mean() * 100.0),
                "area_weighted_mean_p_tail": float(area_weighted_p.mean()),
                "mean_zone_mean_p_tail": float(df["discomfort_probability"].mean()),
                "any_zone_mean_p_tail_max": float(max_zone_p.mean()),
                "zone_time_high_tail_pct_unweighted": float(high.mean() * 100.0),
            }
        )

    zone_df = pd.DataFrame(zone_rows)
    case_df = pd.DataFrame(case_rows)
    if zone_df.empty or case_df.empty:
        raise ValueError("No occupied case-zone data were collected.")
    case_df["sort_key"] = case_df.apply(sort_key, axis=1)
    zone_df = zone_df.merge(case_df[["weather", "sort_key"]], on="weather", how="left")
    zone_df = zone_df.sort_values(["sort_key", "zone"]).drop(columns=["sort_key"])
    case_df = case_df.sort_values("sort_key").drop(columns=["sort_key"])
    return zone_df, case_df


def zone_summary(zone_df: pd.DataFrame, areas: dict[str, float]) -> pd.DataFrame:
    grouped = (
        zone_df.groupby(["zone", "zone_label"], sort=False)
        .agg(
            area_m2=("area_m2", "first"),
            area_share_pct=("area_share_pct", "first"),
            mean_high_tail_pct=("high_tail_pct", "mean"),
            p95_case_high_tail_pct=("high_tail_pct", lambda s: float(s.quantile(0.95))),
            max_case_high_tail_pct=("high_tail_pct", "max"),
            mean_p_tail=("mean_p_tail", "mean"),
            p95_p_tail=("p95_p_tail", "mean"),
        )
        .reset_index()
    )
    grouped["zone_order"] = grouped["zone"].map({zone: i for i, zone in enumerate(ZONE_ORDER)})
    grouped = grouped.sort_values("zone_order").drop(columns=["zone_order"])
    grouped["area_rank_large_to_small"] = grouped["zone"].map(
        {
            zone: rank
            for rank, zone in enumerate(
                sorted(ZONE_ORDER, key=lambda z: areas[z], reverse=True), start=1
            )
        }
    )
    return grouped


def markdown_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    headers = [str(c) for c in out.columns]
    rows = [[str(v) for v in row] for row in out.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def time_label(value: str) -> str:
    return {
        "baseline_2020s": "2020s",
        "near_2030s": "2030s",
        "mid_2050s": "2050s",
        "late_2080s": "2080s",
    }[value]


def scenario_label(value: str) -> str:
    return {"ssp245": "S245", "ssp585": "S585"}[value]


def make_mosaic(
    zone_df: pd.DataFrame,
    case_df: pd.DataFrame,
    areas: dict[str, float],
    out_dir: Path,
    width_mode: str,
    threshold: float,
) -> Path:
    if width_mode == "area":
        widths = np.array([areas[zone] for zone in ZONE_ORDER], dtype=float)
        width_note = "column width proportional to zone floor area"
        suffix = "area"
    elif width_mode == "sqrt_area":
        widths = np.sqrt(np.array([areas[zone] for zone in ZONE_ORDER], dtype=float))
        width_note = "column width proportional to sqrt(zone floor area)"
        suffix = "sqrt_area"
    else:
        widths = np.ones(len(ZONE_ORDER), dtype=float)
        width_note = "equal-width zone columns"
        suffix = "equal"

    x_edges = np.concatenate([[0.0], np.cumsum(widths)])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    city_frames = {city: case_df[case_df["city"] == city].copy() for city in CITY_ORDER}
    vmax = min(
        100.0,
        max(45.0, float(np.ceil(zone_df["high_tail_pct"].quantile(0.995) / 5.0) * 5.0)),
    )

    fig, axes = plt.subplots(
        len(CITY_ORDER),
        1,
        figsize=(12.6, 12.8),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"hspace": 0.08},
    )
    mesh = None
    cmap = plt.get_cmap("YlOrRd")

    zone_pivot = zone_df.pivot_table(
        index="weather", columns="zone", values="high_tail_pct", aggfunc="first"
    )
    for ax, city in zip(axes, CITY_ORDER, strict=True):
        cdf = city_frames[city]
        matrix = zone_pivot.loc[cdf["weather"], ZONE_ORDER].to_numpy(float)
        y_edges = np.arange(matrix.shape[0] + 1)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            linewidth=0.15,
            edgecolors="#f2f2f2",
        )
        ax.invert_yaxis()
        ax.set_ylabel(city, rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(axis="y", length=0, labelsize=7)

        group_centers: list[float] = []
        group_labels: list[str] = []
        for scenario in SCENARIO_ORDER:
            for time_slice in TIME_ORDER:
                mask = (cdf["scenario_raw"] == scenario) & (cdf["time_slice"] == time_slice)
                idx = np.flatnonzero(mask.to_numpy())
                if len(idx):
                    group_centers.append(float(idx.mean() + 0.5))
                    group_labels.append(f"{scenario_label(scenario)} {time_label(time_slice)}")
        ax.set_yticks(group_centers)
        ax.set_yticklabels(group_labels)

        for y in range(3, matrix.shape[0], 3):
            ax.axhline(y, color="#d7d7d7", linewidth=0.35)
        for y in range(12, matrix.shape[0], 12):
            ax.axhline(y, color="#8a8a8a", linewidth=0.75)

        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xticks(x_centers)
    axes[-1].set_xticklabels([ZONE_LABELS[zone] for zone in ZONE_ORDER], rotation=45, ha="right")
    axes[-1].tick_params(axis="x", labelsize=8)

    for boundary in [5, 10]:
        x = x_edges[boundary]
        for ax in axes:
            ax.axvline(x, color="#555555", linewidth=0.9)

    fig.suptitle(
        "Zone-size mosaic of high-tail exposure across 144 fixed-reference runs",
        fontsize=13,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.01,
        0.012,
        f"Cell color: occupied records with zone p_tail >= {threshold:.2f}. "
        f"Rows within each scenario-time group are typical, hot, heatwave-extreme. "
        f"{width_note}.",
        fontsize=8,
    )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.72])
    assert mesh is not None
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("Zone high-tail exposure (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.965, bottom=0.09)

    png = out_dir / f"zone_size_mosaic_heatmap_{suffix}.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def write_summary(
    zone_df: pd.DataFrame,
    case_df: pd.DataFrame,
    zsum: pd.DataFrame,
    areas: dict[str, float],
    out_dir: Path,
    threshold: float,
) -> Path:
    all_case = case_df.agg(
        {
            "mean_zone_high_tail_pct": "mean",
            "any_zone_high_tail_pct": "mean",
            "hidden_any_zone_high_tail_pct": "mean",
            "area_weighted_zone_time_high_tail_pct": "mean",
            "zone_time_high_tail_pct_unweighted": "mean",
        }
    )
    city_summary = (
        case_df.groupby("city", sort=False)
        .agg(
            mean_zone_high_tail_pct=("mean_zone_high_tail_pct", "mean"),
            any_zone_high_tail_pct=("any_zone_high_tail_pct", "mean"),
            hidden_any_zone_high_tail_pct=("hidden_any_zone_high_tail_pct", "mean"),
            area_weighted_zone_time_high_tail_pct=(
                "area_weighted_zone_time_high_tail_pct",
                "mean",
            ),
        )
        .reset_index()
    )
    top_zones = zsum.sort_values("mean_high_tail_pct", ascending=False).head(8)
    area_total = sum(areas.values())

    path = out_dir / "zone_size_mosaic_summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Zone-Size Mosaic Diagnostic\n\n")
        f.write(
            f"Source traces: `{TRACE_DIR}`. High-tail cutoff: "
            f"`p_tail >= {threshold:.2f}`.\n\n"
        )
        f.write("## Headline\n\n")
        f.write(
            "- Mean-zone high-tail exposure averaged "
            f"{all_case.mean_zone_high_tail_pct:.2f}% across cases; any-zone exposure averaged "
            f"{all_case.any_zone_high_tail_pct:.2f}%.\n"
        )
        f.write(
            "- Area-weighted zone-time high-tail exposure averaged "
            f"{all_case.area_weighted_zone_time_high_tail_pct:.2f}%, versus "
            f"{all_case.zone_time_high_tail_pct_unweighted:.2f}% when all zones are weighted equally.\n"
        )
        f.write(
            "- Conditioned floor area represented in the mosaic is "
            f"{area_total:.2f} m2 across {len(ZONE_ORDER)} zones.\n"
        )
        f.write("\n## City Summary\n\n")
        f.write(markdown_table(city_summary))
        f.write("\n\n## Highest-Risk Zones\n\n")
        f.write(
            markdown_table(
                top_zones[
                    [
                        "zone",
                        "zone_label",
                        "area_m2",
                        "area_share_pct",
                        "mean_high_tail_pct",
                        "p95_case_high_tail_pct",
                        "mean_p_tail",
                    ]
                ]
            )
        )
        f.write("\n\n## Zone Area and Risk Summary\n\n")
        f.write(markdown_table(zsum))
        f.write("\n")
    return path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trace_paths = sorted(args.trace_dir.glob("*_diagnostic_reference.csv"))
    if not trace_paths:
        raise FileNotFoundError(f"No diagnostic traces found in {args.trace_dir}")
    p_cols = discover_zone_probability_columns(trace_paths[0])
    areas = read_zone_areas(args.energyplus_dir)

    zone_df, case_df = build_case_rows(args.trace_dir, p_cols, areas, args.threshold)
    zsum = zone_summary(zone_df, areas)

    zone_df.to_csv(args.output_dir / "zone_case_mosaic_values.csv", index=False)
    case_df.to_csv(args.output_dir / "zone_size_case_summary.csv", index=False)
    zsum.to_csv(args.output_dir / "zone_size_summary.csv", index=False)

    pdfs = [
        make_mosaic(zone_df, case_df, areas, args.output_dir, "area", args.threshold),
        make_mosaic(zone_df, case_df, areas, args.output_dir, "sqrt_area", args.threshold),
        make_mosaic(zone_df, case_df, areas, args.output_dir, "equal", args.threshold),
    ]
    summary = write_summary(zone_df, case_df, zsum, areas, args.output_dir, args.threshold)

    print(f"[zone-size-mosaic] wrote {args.output_dir}")
    for pdf in pdfs:
        print(f"[zone-size-mosaic] figure {pdf}")
    print(f"[zone-size-mosaic] summary {summary}")
    print(
        case_df[
            [
                "mean_zone_high_tail_pct",
                "any_zone_high_tail_pct",
                "hidden_any_zone_high_tail_pct",
                "area_weighted_zone_time_high_tail_pct",
            ]
        ]
        .mean()
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

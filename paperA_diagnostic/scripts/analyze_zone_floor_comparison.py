#!/usr/bin/env python3
"""Compare bottom, middle, and top zone tail exposure for Paper A."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "runs" / "diagnostic_reference_zone_raw_full" / "traces"
OUT = ROOT / "diagnostics" / "zone_floor_comparison"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAIL_DIAGNOSTIC = 0.20
FLOORS = ["bottom", "middle", "top"]
POSITIONS = ["core", "P1", "P2", "P3", "P4"]
POSITION_NAMES = {
    "core": "Core",
    "P1": "P1 south",
    "P2": "P2 east",
    "P3": "P3 north",
    "P4": "P4 west",
}

ZONE_BY_FLOOR = {
    "bottom": [
        "core_bottom",
        "perimeter_bot_zn_1",
        "perimeter_bot_zn_2",
        "perimeter_bot_zn_3",
        "perimeter_bot_zn_4",
    ],
    "middle": [
        "core_mid",
        "perimeter_mid_zn_1",
        "perimeter_mid_zn_2",
        "perimeter_mid_zn_3",
        "perimeter_mid_zn_4",
    ],
    "top": [
        "core_top",
        "perimeter_top_zn_1",
        "perimeter_top_zn_2",
        "perimeter_top_zn_3",
        "perimeter_top_zn_4",
    ],
}

ZONE_ORDER = [zone for floor in FLOORS for zone in ZONE_BY_FLOOR[floor]]

ZONE_AREAS_M2 = {
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
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--threshold", type=float, default=TAIL_DIAGNOSTIC)
    return parser.parse_args()


def zone_floor(zone: str) -> str:
    if zone.endswith("bottom") or "_bot_" in zone:
        return "bottom"
    if zone.endswith("mid") or "_mid_" in zone:
        return "middle"
    if zone.endswith("top") or "_top_" in zone:
        return "top"
    raise ValueError(zone)


def zone_position(zone: str) -> str:
    match = re.search(r"_zn_(\d)$", zone)
    return "core" if not match else f"P{match.group(1)}"


def zone_kind(zone: str) -> str:
    return "core" if zone.startswith("core_") else "perimeter"


def parse_weather(weather: str) -> dict[str, object]:
    match = WEATHER_RE.match(weather)
    if not match:
        return {"city": "", "scenario_raw": "", "time_slice": "", "severity": "", "weather_year": np.nan}
    data = match.groupdict()
    return {
        "city": data["city"].title(),
        "scenario_raw": data["scenario_raw"],
        "time_slice": data["time_slice"],
        "severity": data["severity"],
        "weather_year": int(data["weather_year"]),
    }


def pct(mask: pd.Series | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean() * 100.0)


def build_summaries(trace_dir: Path, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    if not paths:
        raise FileNotFoundError(f"No diagnostic traces found in {trace_dir}")

    usecols = ["occupied"]
    for zone in ZONE_ORDER:
        usecols.extend(
            [
                f"zone_{zone}_p_disc",
                f"zone_{zone}_warm_tail",
                f"zone_{zone}_cold_tail",
                f"zone_{zone}_expected_tsv",
                f"zone_{zone}_ta_c",
                f"zone_{zone}_tr_c",
            ]
        )

    zone_acc = {
        zone: {
            "n": 0,
            "high": 0,
            "sum_p": 0.0,
            "sum_warm": 0.0,
            "sum_cold": 0.0,
            "sum_mu": 0.0,
            "sum_ta": 0.0,
            "sum_tr": 0.0,
            "warm_dominant_high": 0,
            "cold_dominant_high": 0,
        }
        for zone in ZONE_ORDER
    }
    case_rows: list[dict[str, object]] = []

    for path in paths:
        weather = path.name.removesuffix("_diagnostic_reference.csv")
        meta = parse_weather(weather)
        df = pd.read_csv(path, usecols=usecols)
        df = df[df["occupied"]].copy()
        row: dict[str, object] = {"weather": weather, **meta}

        for floor in FLOORS:
            zones = ZONE_BY_FLOOR[floor]
            p = df[[f"zone_{zone}_p_disc" for zone in zones]].to_numpy(float)
            high = p >= threshold
            weights = np.array([ZONE_AREAS_M2[zone] for zone in zones], dtype=float)
            row[f"{floor}_area_weighted_high_tail_pct"] = float(
                np.average(high.astype(float), axis=1, weights=weights).mean() * 100.0
            )
            row[f"{floor}_unweighted_high_tail_pct"] = float(high.mean() * 100.0)
            row[f"{floor}_any_zone_high_tail_pct"] = float(high.any(axis=1).mean() * 100.0)

        for zone in ZONE_ORDER:
            p_col = f"zone_{zone}_p_disc"
            warm_col = f"zone_{zone}_warm_tail"
            cold_col = f"zone_{zone}_cold_tail"
            p = df[p_col].astype(float)
            warm = df[warm_col].astype(float)
            cold = df[cold_col].astype(float)
            high = p.ge(threshold)
            acc = zone_acc[zone]
            acc["n"] += len(df)
            acc["high"] += int(high.sum())
            acc["sum_p"] += float(p.sum())
            acc["sum_warm"] += float(warm.sum())
            acc["sum_cold"] += float(cold.sum())
            acc["sum_mu"] += float(df[f"zone_{zone}_expected_tsv"].sum())
            acc["sum_ta"] += float(df[f"zone_{zone}_ta_c"].sum())
            acc["sum_tr"] += float(df[f"zone_{zone}_tr_c"].sum())
            acc["warm_dominant_high"] += int((high & warm.ge(cold)).sum())
            acc["cold_dominant_high"] += int((high & cold.gt(warm)).sum())

        for position in ["P1", "P2", "P3", "P4"]:
            for floor in FLOORS:
                tag = {"bottom": "bot", "middle": "mid", "top": "top"}[floor]
                zone = f"perimeter_{tag}_zn_{position[-1]}"
                row[f"{position}_{floor}_high_tail_pct"] = pct(
                    df[f"zone_{zone}_p_disc"].ge(threshold)
                )
        case_rows.append(row)

    zone_rows = []
    for zone, acc in zone_acc.items():
        n = acc["n"]
        high = acc["high"]
        zone_rows.append(
            {
                "zone": zone,
                "floor": zone_floor(zone),
                "kind": zone_kind(zone),
                "position": zone_position(zone),
                "position_name": POSITION_NAMES[zone_position(zone)],
                "area_m2": ZONE_AREAS_M2[zone],
                "high_tail_pct": high / n * 100.0,
                "mean_p_tail": acc["sum_p"] / n,
                "mean_warm_tail": acc["sum_warm"] / n,
                "mean_cold_tail": acc["sum_cold"] / n,
                "mean_expected_tsv": acc["sum_mu"] / n,
                "mean_air_temp_c": acc["sum_ta"] / n,
                "mean_radiant_temp_c": acc["sum_tr"] / n,
                "warm_dominant_high_tail_pct": acc["warm_dominant_high"] / high * 100.0
                if high
                else float("nan"),
                "cold_dominant_high_tail_pct": acc["cold_dominant_high"] / high * 100.0
                if high
                else float("nan"),
            }
        )

    return pd.DataFrame(zone_rows), pd.DataFrame(case_rows)


def floor_summary(zone_df: pd.DataFrame, case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for floor in FLOORS:
        sub = zone_df[zone_df["floor"] == floor]
        weights = sub["area_m2"].to_numpy(float)
        case_aw = case_df[f"{floor}_area_weighted_high_tail_pct"]
        case_uw = case_df[f"{floor}_unweighted_high_tail_pct"]
        case_any = case_df[f"{floor}_any_zone_high_tail_pct"]
        rows.append(
            {
                "floor": floor,
                "area_weighted_high_tail_pct": float(np.average(sub["high_tail_pct"], weights=weights)),
                "unweighted_zone_time_high_tail_pct": float(sub["high_tail_pct"].mean()),
                "any_zone_within_floor_high_tail_pct": float(case_any.mean()),
                "mean_warm_tail_area_weighted": float(np.average(sub["mean_warm_tail"], weights=weights)),
                "mean_cold_tail_area_weighted": float(np.average(sub["mean_cold_tail"], weights=weights)),
                "mean_expected_tsv_area_weighted": float(
                    np.average(sub["mean_expected_tsv"], weights=weights)
                ),
                "mean_air_temp_area_weighted_c": float(np.average(sub["mean_air_temp_c"], weights=weights)),
                "mean_radiant_temp_area_weighted_c": float(
                    np.average(sub["mean_radiant_temp_c"], weights=weights)
                ),
                "cases_floor_has_highest_area_weighted_tail": int(
                    (
                        case_aw
                        == case_df[[f"{f}_area_weighted_high_tail_pct" for f in FLOORS]].max(axis=1)
                    ).sum()
                ),
                "cases_floor_has_highest_unweighted_tail": int(
                    (
                        case_uw
                        == case_df[[f"{f}_unweighted_high_tail_pct" for f in FLOORS]].max(axis=1)
                    ).sum()
                ),
                "cases_floor_has_highest_any_zone_tail": int(
                    (
                        case_any
                        == case_df[[f"{f}_any_zone_high_tail_pct" for f in FLOORS]].max(axis=1)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(zone_df: pd.DataFrame, floor_df: pd.DataFrame, case_df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "zone_floor_comparison_summary.md"
    position = (
        zone_df.pivot(index="position_name", columns="floor", values="high_tail_pct")
        .loc[[POSITION_NAMES[pos] for pos in POSITIONS], FLOORS]
        .reset_index()
    )
    with path.open("w", encoding="utf-8") as f:
        f.write("# Zone Floor Comparison\n\n")
        f.write("P1 is the south perimeter zone, P2 east, P3 north, and P4 west.\n\n")
        f.write("## Headline\n\n")
        f.write(
            "- Area-weighted high-tail exposure by floor is "
            f"{floor_df.loc[floor_df.floor == 'bottom', 'area_weighted_high_tail_pct'].iloc[0]:.2f}% "
            "for bottom, "
            f"{floor_df.loc[floor_df.floor == 'middle', 'area_weighted_high_tail_pct'].iloc[0]:.2f}% "
            "for middle, and "
            f"{floor_df.loc[floor_df.floor == 'top', 'area_weighted_high_tail_pct'].iloc[0]:.2f}% "
            "for top.\n"
        )
        f.write(
            "- Middle floor has the highest area-weighted floor exposure in "
            f"{int(floor_df.loc[floor_df.floor == 'middle', 'cases_floor_has_highest_area_weighted_tail'].iloc[0])} "
            f"of {len(case_df)} cases.\n"
        )
        f.write(
            "- Bottom P1 is higher than middle and top P1 in most cases, but this is a south-perimeter/ground-floor result, not a general bottom-floor result.\n"
        )
        f.write(
            "- High-tail states are warm-dominant in every zone under the current threshold.\n\n"
        )
        f.write("## Floor Summary\n\n")
        f.write(markdown_table(floor_df))
        f.write("\n\n## Position-by-Floor High-Tail Exposure\n\n")
        f.write(markdown_table(position))
        f.write("\n")
    return path


def markdown_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    lines = [
        "| " + " | ".join(out.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in out.to_numpy())
    return "\n".join(lines)


def make_plot(zone_df: pd.DataFrame, floor_df: pd.DataFrame, out_dir: Path) -> Path:
    heat = (
        zone_df.pivot(index="position", columns="floor", values="high_tail_pct")
        .loc[POSITIONS, FLOORS]
        .to_numpy(float)
    )
    warm = floor_df.set_index("floor").loc[FLOORS, "mean_warm_tail_area_weighted"].to_numpy(float)
    cold = floor_df.set_index("floor").loc[FLOORS, "mean_cold_tail_area_weighted"].to_numpy(float)
    aw = floor_df.set_index("floor").loc[FLOORS, "area_weighted_high_tail_pct"].to_numpy(float)
    any_floor = floor_df.set_index("floor").loc[FLOORS, "any_zone_within_floor_high_tail_pct"].to_numpy(float)

    fig = plt.figure(figsize=(11.0, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.09,
        right=0.94,
        top=0.86,
        bottom=0.16,
        wspace=0.34,
        hspace=0.55,
    )
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_aw = fig.add_subplot(gs[0, 1])
    ax_tail = fig.add_subplot(gs[1, 1])

    im = ax_heat.imshow(heat, cmap="YlOrRd", vmin=0, vmax=max(35, float(np.ceil(heat.max() / 5) * 5)))
    ax_heat.set_xticks(range(len(FLOORS)))
    ax_heat.set_xticklabels(["Bottom", "Middle", "Top"])
    ax_heat.set_yticks(range(len(POSITIONS)))
    ax_heat.set_yticklabels([POSITION_NAMES[pos] for pos in POSITIONS])
    ax_heat.set_title("Zone-position high-tail exposure (%)", fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax_heat.text(
                j,
                i,
                f"{heat[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if heat[i, j] >= 22 else "#222222",
                fontweight="bold" if heat[i, j] >= 20 else "normal",
            )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label("High-tail exposure (%)")

    x = np.arange(len(FLOORS))
    floor_labels = ["Bottom", "Middle", "Top"]
    ax_aw.bar(x - 0.18, aw, width=0.36, color="#4c78a8", label="Area-weighted")
    ax_aw.bar(x + 0.18, any_floor, width=0.36, color="#d45b43", label="Any zone on floor")
    ax_aw.set_xticks(x)
    ax_aw.set_xticklabels(floor_labels)
    ax_aw.set_ylabel("Exposure (%)")
    ax_aw.set_title("Floor-level exposure summaries", fontweight="bold")
    ax_aw.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax_aw.set_ylim(0, max(any_floor.max(), aw.max()) * 1.18)
    ax_aw.legend(frameon=False, fontsize=8)
    for xi, val in zip(x - 0.18, aw, strict=True):
        ax_aw.text(xi, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, val in zip(x + 0.18, any_floor, strict=True):
        ax_aw.text(xi, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax_tail.bar(x, warm, color="#e4572e", label="Warm tail")
    ax_tail.bar(x, cold, bottom=warm, color="#3b73b9", label="Cold tail")
    ax_tail.set_xticks(x)
    ax_tail.set_xticklabels(floor_labels)
    ax_tail.set_ylabel("Mean probability")
    ax_tail.set_title("Area-weighted tail components", fontweight="bold")
    ax_tail.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax_tail.set_ylim(0, (warm + cold).max() * 1.22)
    ax_tail.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Floor comparison separates level and perimeter orientation",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.09,
        0.055,
        "P1--P4 are perimeter orientations: P1 south, P2 east, P3 north, P4 west. "
        "High-tail exposure is the share of occupied records with zone p_tail >= 0.20.",
        fontsize=8,
    )
    png = out_dir / "zone_floor_position_comparison.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zone_df, case_df = build_summaries(args.trace_dir, args.threshold)
    floor_df = floor_summary(zone_df, case_df)

    zone_df.to_csv(args.output_dir / "zone_floor_position_summary.csv", index=False)
    case_df.to_csv(args.output_dir / "zone_floor_case_summary.csv", index=False)
    floor_df.to_csv(args.output_dir / "zone_floor_summary.csv", index=False)
    fig = make_plot(zone_df, floor_df, args.output_dir)
    md = write_markdown(zone_df, floor_df, case_df, args.output_dir)

    print(f"[zone-floor] wrote {args.output_dir}")
    print(f"[zone-floor] figure {fig}")
    print(f"[zone-floor] summary {md}")
    print(floor_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

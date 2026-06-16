#!/usr/bin/env python3
"""Audit how p_tail threshold choice changes exposure summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "diagnostics"
    / "zone_aggregation_zone_raw"
    / "zone_aggregation_rows_compact.csv"
)
DEFAULT_OUT = ROOT / "diagnostics" / "tail_threshold_sensitivity"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[round(x, 2) for x in np.arange(0.05, 0.401, 0.05)],
    )
    return parser.parse_args()


def pct(mask: pd.Series | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean() * 100.0)


def load_rows(path: Path) -> pd.DataFrame:
    usecols = [
        "city",
        "scenario_raw",
        "time_slice",
        "severity",
        "discomfort_probability",
        "warm_discomfort_probability",
        "cold_discomfort_probability",
        "zone_p_disc_max",
        "zone_p_disc_p90",
    ]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    needed = [
        "discomfort_probability",
        "warm_discomfort_probability",
        "cold_discomfort_probability",
        "zone_p_disc_max",
        "zone_p_disc_p90",
    ]
    for col in needed:
        if col not in df:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=needed).copy()


def summarize_group(df: pd.DataFrame, thresholds: list[float], group: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tau in thresholds:
        rows.append(
            {
                **group,
                "threshold": float(tau),
                "rows": int(len(df)),
                "mean_zone_high_tail_pct": pct(df["discomfort_probability"].ge(tau)),
                "p90_zone_high_tail_pct": pct(df["zone_p_disc_p90"].ge(tau)),
                "any_zone_high_tail_pct": pct(df["zone_p_disc_max"].ge(tau)),
                "hidden_any_zone_pct": pct(
                    df["discomfort_probability"].lt(tau) & df["zone_p_disc_max"].ge(tau)
                ),
                "mean_warm_tail_high_pct": pct(df["warm_discomfort_probability"].ge(tau)),
                "mean_cold_tail_high_pct": pct(df["cold_discomfort_probability"].ge(tau)),
            }
        )
    return rows


def summarize(df: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = summarize_group(df, thresholds, {"scope": "all"})
    city_rows: list[dict[str, object]] = []
    for city, sdf in df.groupby("city", sort=True):
        city_rows.extend(summarize_group(sdf, thresholds, {"scope": "city", "city": city}))
    return pd.DataFrame(global_rows), pd.DataFrame(city_rows)


def write_plot(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=180)
    specs = [
        ("mean_zone_high_tail_pct", "Mean zone", "#365f91", "-"),
        ("p90_zone_high_tail_pct", "90th percentile zone", "#6d7f3f", "-"),
        ("any_zone_high_tail_pct", "Any zone", "#a84831", "-"),
        ("hidden_any_zone_pct", "Hidden any-zone", "#6a4c93", "--"),
    ]
    for col, label, color, ls in specs:
        ax.plot(
            summary["threshold"],
            summary[col],
            marker="o",
            ms=4.0,
            lw=1.8,
            color=color,
            linestyle=ls,
            label=label,
        )
    ax.axvline(0.20, color="#333333", lw=0.8, ls=":")
    ax.set_xlabel(r"High-tail screen, $\tau$")
    ax.set_ylabel("Occupied records crossing screen (%)")
    ax.set_xlim(0.045, 0.405)
    ax.set_ylim(bottom=0)
    ax.grid(color="#dddddd", lw=0.5)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    pdf = out_dir / "p_tail_threshold_sensitivity.pdf"
    png = out_dir / "p_tail_threshold_sensitivity.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return pdf


def write_markdown(summary: pd.DataFrame, out_dir: Path, plot: Path) -> Path:
    path = out_dir / "tail_threshold_sensitivity_summary.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# p_tail Threshold Sensitivity\n\n")
        handle.write(
            "Exposure shares are computed over occupied records. "
            "The 0.20 threshold used in the manuscript is a reporting screen; "
            "this audit checks whether the aggregation gap persists when the screen changes.\n\n"
        )
        handle.write("## Global summary\n\n")
        handle.write("```csv\n")
        handle.write(summary.to_csv(index=False))
        handle.write("```\n\n")
        handle.write("## Figure\n\n")
        handle.write(f"- `{plot.name}`\n")
    return path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.input)
    global_summary, city_summary = summarize(df, args.thresholds)
    global_path = args.output_dir / "p_tail_threshold_sensitivity_global.csv"
    city_path = args.output_dir / "p_tail_threshold_sensitivity_city.csv"
    global_summary.to_csv(global_path, index=False)
    city_summary.to_csv(city_path, index=False)
    plot = write_plot(global_summary, args.output_dir)
    md = write_markdown(global_summary, args.output_dir, plot)
    print(f"[write] {global_path}")
    print(f"[write] {city_path}")
    print(f"[write] {plot}")
    print(f"[write] {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

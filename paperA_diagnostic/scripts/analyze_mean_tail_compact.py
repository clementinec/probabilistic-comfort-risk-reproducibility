#!/usr/bin/env python3
"""Compact Paper A mean-tail diagnostics for the full 144-case panel."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TRACE_DIR = ROOT / "runs" / "diagnostic_reference" / "traces"
DEFAULT_OUT = ROOT / "diagnostics" / "mean_tail_compact"
USECOLS = {
    "strategy",
    "weather",
    "month",
    "day",
    "current_time",
    "occupied",
    "expected_tsv",
    "discomfort_probability",
    "warm_discomfort_probability",
    "cold_discomfort_probability",
    "mean_pmv",
    "mean_operative_temp_c",
    "outdoor_temp_c",
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
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--tail-threshold", type=float, default=0.20)
    parser.add_argument("--direction-threshold", type=float, default=0.10)
    parser.add_argument("--plot-sample-rows", type=int, default=300_000)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def parse_weather(stem: str) -> dict[str, object]:
    match = WEATHER_RE.match(str(stem))
    if not match:
        return {
            "city": np.nan,
            "scenario_raw": np.nan,
            "time_slice": np.nan,
            "severity": np.nan,
            "weather_year": np.nan,
        }
    data = match.groupdict()
    data["weather_year"] = int(data["weather_year"])
    data["city"] = data["city"].title()
    return data


def load_rows(trace_dir: Path) -> pd.DataFrame:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    paths = [path for path in paths if path.name != "medium_office_control_traces.csv"]
    if not paths:
        raise FileNotFoundError(f"No per-case diagnostic traces found in {trace_dir}")
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=lambda col: col in USECOLS)
        occupied = bool_series(df["occupied"])
        probability = df["expected_tsv"].notna() & df["discomfort_probability"].notna()
        frames.append(df.loc[occupied & probability].copy())
    rows = pd.concat(frames, ignore_index=True)
    for col in USECOLS - {"strategy", "weather", "occupied"}:
        if col in rows:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["p_tail"] = rows["discomfort_probability"]
    rows["d_tail"] = rows["warm_discomfort_probability"] - rows["cold_discomfort_probability"]
    rows["abs_mu"] = rows["expected_tsv"].abs()
    rows["abs_d_tail"] = rows["d_tail"].abs()
    meta = pd.DataFrame([parse_weather(stem) for stem in rows["weather"]], index=rows.index)
    for col in meta.columns:
        rows[col] = meta[col]
    return rows


def pct(mask: pd.Series | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean() * 100.0)


def summarize_group(label: dict[str, object], df: pd.DataFrame, eps: float, tail: float, direction: float) -> dict[str, object]:
    near_mean = df["abs_mu"].lt(eps)
    high_tail = df["p_tail"].ge(tail)
    directional = df["abs_d_tail"].ge(direction)
    mu_sign = np.sign(np.where(df["abs_mu"].ge(eps), df["expected_tsv"], 0.0))
    d_sign = np.sign(np.where(df["abs_d_tail"].ge(direction), df["d_tail"], 0.0))
    comparable = (mu_sign != 0) & (d_sign != 0)
    conflict = comparable & (mu_sign != d_sign)
    out = dict(label)
    out.update(
        {
            "rows": int(len(df)),
            "weather_cases": int(df["weather"].nunique()),
            "mean_mu": float(df["expected_tsv"].mean()),
            "mean_abs_mu": float(df["abs_mu"].mean()),
            "mean_p_tail": float(df["p_tail"].mean()),
            "p95_p_tail": float(df["p_tail"].quantile(0.95)),
            "high_tail_pct": pct(high_tail),
            "near_mean_pct": pct(near_mean),
            "near_mean_high_tail_pct": pct(near_mean & high_tail),
            "near_mean_directional_tail_pct": pct(near_mean & directional),
            "warm_tail_dominant_pct": pct(df["d_tail"].gt(direction)),
            "cold_tail_dominant_pct": pct(df["d_tail"].lt(-direction)),
            "sign_conflict_pct_of_comparable": float(conflict.sum() / comparable.sum() * 100.0)
            if comparable.sum()
            else float("nan"),
            "sign_comparable_rows": int(comparable.sum()),
        }
    )
    return out


def build_summaries(rows: pd.DataFrame, eps: float, tail: float, direction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = [summarize_group({"scope": "all"}, rows, eps, tail, direction)]
    group_specs = [
        ("city", ["city"]),
        ("scenario", ["scenario_raw"]),
        ("time_slice", ["time_slice"]),
        ("severity", ["severity"]),
        ("city_time_slice", ["city", "time_slice"]),
        ("city_scenario_time_slice", ["city", "scenario_raw", "time_slice"]),
    ]
    for scope, keys in group_specs:
        for values, group in rows.groupby(keys, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            label = {"scope": scope}
            label.update(dict(zip(keys, values)))
            summary_rows.append(summarize_group(label, group, eps, tail, direction))
    case_rows = []
    for weather, group in rows.groupby("weather", sort=True):
        meta = parse_weather(weather)
        label = {"weather": weather, **meta}
        case_rows.append(summarize_group(label, group, eps, tail, direction))
    return pd.DataFrame(summary_rows), pd.DataFrame(case_rows)


def build_mean_bin_spread(rows: pd.DataFrame, tail: float, bin_width: float = 0.05) -> pd.DataFrame:
    work = rows[["expected_tsv", "p_tail"]].dropna().copy()
    work["mu_bin_left"] = np.floor(work["expected_tsv"] / bin_width) * bin_width
    grouped = work.groupby("mu_bin_left", sort=True)
    out = grouped["p_tail"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack()
    out.columns = ["p_tail_q05", "p_tail_q25", "p_tail_median", "p_tail_q75", "p_tail_q95"]
    out["rows"] = grouped.size()
    out["high_tail_pct"] = grouped["p_tail"].apply(lambda s: float(s.ge(tail).mean() * 100.0))
    out = out.reset_index()
    out["mu_bin_right"] = out["mu_bin_left"] + bin_width
    out["mu_bin_center"] = out["mu_bin_left"] + bin_width / 2.0
    out["p_tail_iqr"] = out["p_tail_q75"] - out["p_tail_q25"]
    out["p_tail_90pct_spread"] = out["p_tail_q95"] - out["p_tail_q05"]
    return out[["mu_bin_left", "mu_bin_right", "mu_bin_center", "rows", "p_tail_q05", "p_tail_q25", "p_tail_median", "p_tail_q75", "p_tail_q95", "p_tail_iqr", "p_tail_90pct_spread", "high_tail_pct"]]


def find_examples(rows: pd.DataFrame, eps: float, tail: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "weather",
        "city",
        "scenario_raw",
        "time_slice",
        "severity",
        "weather_year",
        "month",
        "day",
        "current_time",
        "outdoor_temp_c",
        "mean_operative_temp_c",
        "expected_tsv",
        "p_tail",
        "d_tail",
        "warm_discomfort_probability",
        "cold_discomfort_probability",
    ]
    near_high = rows[rows["abs_mu"].lt(eps) & rows["p_tail"].ge(tail)].copy()
    near_high["example_score"] = near_high["p_tail"] - near_high["abs_mu"]
    near_high = near_high.sort_values("example_score", ascending=False)[cols + ["example_score"]].head(100)

    work = rows.copy()
    work["mu_bin"] = (work["expected_tsv"] / 0.02).round() * 0.02
    pairs = []
    for _, group in work.groupby("mu_bin", sort=False):
        if len(group) < 2:
            continue
        low = group.loc[group["p_tail"].idxmin()]
        high = group.loc[group["p_tail"].idxmax()]
        gap = float(high["p_tail"] - low["p_tail"])
        if gap <= 0:
            continue
        pairs.append((gap, low.name, high.name))
    pairs = sorted(pairs, reverse=True)[:50]
    pair_rows = []
    for pair_id, (gap, low_idx, high_idx) in enumerate(pairs, start=1):
        for member, idx in [("low_tail", low_idx), ("high_tail", high_idx)]:
            rec = rows.loc[idx, cols].to_dict()
            rec["pair_id"] = pair_id
            rec["member"] = member
            rec["pair_tail_gap"] = gap
            pair_rows.append(rec)
    return near_high, pd.DataFrame(pair_rows)


def write_plots(rows: pd.DataFrame, bin_spread: pd.DataFrame, out_dir: Path, max_rows: int, tail: float) -> list[Path]:
    plot_rows = rows.sample(n=max_rows, random_state=42) if len(rows) > max_rows else rows
    paths = []
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    sc = ax.scatter(
        plot_rows["expected_tsv"],
        plot_rows["p_tail"],
        c=plot_rows["d_tail"],
        cmap="coolwarm",
        s=8,
        alpha=0.55,
        linewidths=0,
    )
    ax.axvline(0, color="#444444", lw=0.8)
    ax.axhline(tail, color="#7a2d42", lw=0.9, ls="--")
    ax.axvspan(-0.15, 0.15, color="#d9d9d9", alpha=0.25, lw=0)
    ax.set_xlabel(r"Expected TSV, $\mu_{\mathrm{TSV}}=\sum_k k p_k$")
    ax.set_ylabel(r"Discomfort-tail probability, $p_{\mathrm{tail}}$")
    ax.set_title("Expected TSV versus discomfort-tail risk", weight="bold")
    ax.grid(color="#dddddd", lw=0.6)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"Directional tail dominance, $d_{\mathrm{tail}}$")
    fig.tight_layout()
    path = out_dir / "mean_vs_tail_scatter_sample.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    plot_bins = bin_spread.sort_values("mu_bin_center")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    x = plot_bins["mu_bin_center"].to_numpy(float)
    ax.fill_between(x, plot_bins["p_tail_q05"], plot_bins["p_tail_q95"], color="#b8c7e6", alpha=0.38, label="5-95%")
    ax.fill_between(x, plot_bins["p_tail_q25"], plot_bins["p_tail_q75"], color="#4f6f9f", alpha=0.34, label="IQR")
    ax.plot(x, plot_bins["p_tail_median"], color="#243f73", lw=1.8, label="Median")
    ax.axhline(tail, color="#7a2d42", lw=0.9, ls="--")
    ax.axvline(0, color="#444444", lw=0.75)
    ax.set_xlabel(r"Expected TSV bin center, $\mu_{\mathrm{TSV}}$")
    ax.set_ylabel(r"Discomfort-tail probability, $p_{\mathrm{tail}}$")
    ax.set_title("Within-bin spread of discomfort-tail risk", weight="bold")
    ax.grid(color="#dddddd", lw=0.55)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    path = out_dir / "tail_spread_by_mean_bin.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def write_markdown(summary: pd.DataFrame, case_summary: pd.DataFrame, out_dir: Path) -> Path:
    all_row = summary[summary["scope"].eq("all")].iloc[0]
    worst_cases = case_summary.sort_values("high_tail_pct", ascending=False).head(10)
    path = out_dir / "mean_tail_compact_summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Mean-Tail Diagnostic Summary\n\n")
        f.write(f"- Occupied probability rows: {int(all_row.rows):,}\n")
        f.write(f"- Weather cases: {int(all_row.weather_cases)}\n")
        f.write(f"- Mean `p_tail`: {all_row.mean_p_tail:.3f}; 95th percentile `p_tail`: {all_row.p95_p_tail:.3f}\n")
        f.write(f"- High-tail exposure (`p_tail >= 0.20`): {all_row.high_tail_pct:.1f}%\n")
        f.write(f"- Near-mean high-tail states (`|mu_TSV| < 0.15` and `p_tail >= 0.20`): {all_row.near_mean_high_tail_pct:.1f}%\n")
        f.write("\n## Highest High-Tail Cases\n\n")
        f.write("```csv\n")
        f.write(
            worst_cases[
                [
                    "weather",
                    "rows",
                    "high_tail_pct",
                    "mean_p_tail",
                    "p95_p_tail",
                    "near_mean_high_tail_pct",
                ]
            ].to_csv(index=False)
        )
        f.write("```\n")
        f.write("\n")
    return path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.trace_dir)
    print(f"[load] occupied probability rows: {len(rows)}")
    print(f"[load] weather cases: {rows['weather'].nunique()}")
    summary, case_summary = build_summaries(rows, args.eps, args.tail_threshold, args.direction_threshold)
    bin_spread = build_mean_bin_spread(rows, args.tail_threshold)
    near_high, pairs = find_examples(rows, args.eps, args.tail_threshold)

    summary.to_csv(args.output_dir / "mean_tail_summary.csv", index=False)
    case_summary.to_csv(args.output_dir / "mean_tail_case_summary.csv", index=False)
    bin_spread.to_csv(args.output_dir / "mean_bin_tail_spread.csv", index=False)
    near_high.to_csv(args.output_dir / "near_mean_high_tail_examples.csv", index=False)
    pairs.to_csv(args.output_dir / "same_mean_tail_contrast_examples.csv", index=False)
    plot_paths = write_plots(rows, bin_spread, args.output_dir, args.plot_sample_rows, args.tail_threshold)
    md = write_markdown(summary, case_summary, args.output_dir)

    print(f"[write] {args.output_dir / 'mean_tail_summary.csv'}")
    print(f"[write] {args.output_dir / 'mean_tail_case_summary.csv'}")
    print(f"[write] {args.output_dir / 'mean_bin_tail_spread.csv'}")
    print(f"[write] {args.output_dir / 'near_mean_high_tail_examples.csv'}")
    print(f"[write] {args.output_dir / 'same_mean_tail_contrast_examples.csv'}")
    for path in plot_paths:
        print(f"[write] {path}")
    print(f"[write] {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare scalar PMV and mean TSV summaries with discomfort-tail exposure."""

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


DEFAULT_TRACE_DIR = ROOT / "runs" / "diagnostic_reference_zone_raw_full" / "traces"
DEFAULT_OUT = ROOT / "diagnostics" / "scalar_tail_comparison"
WEATHER_RE = re.compile(
    r"^(?P<city>ahmedabad|beijing|guangzhou|houston|kolkata|phoenix)_"
    r"(?P<scenario_raw>ssp245|ssp585)_"
    r"(?P<time_slice>baseline_2020s|near_2030s|mid_2050s|late_2080s)_"
    r"(?P<severity>typical|hot|heatwave_extreme)_"
    r"(?P<weather_year>\d{4})$"
)
USECOLS = ["weather", "occupied", "mean_pmv", "expected_tsv", "discomfort_probability"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tail-threshold", type=float, default=0.20)
    parser.add_argument("--sample-rows", type=int, default=300_000)
    return parser.parse_args()


def parse_weather(weather: str) -> dict[str, object]:
    match = WEATHER_RE.match(str(weather))
    if not match:
        return {
            "city": str(weather).split("_")[0].title(),
            "scenario_raw": "",
            "time_slice": "",
            "severity": "",
            "weather_year": np.nan,
        }
    out = match.groupdict()
    out["city"] = out["city"].title()
    out["weather_year"] = int(out["weather_year"])
    return out


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def trace_paths(trace_dir: Path) -> list[Path]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    paths = [p for p in paths if p.name != "medium_office_control_traces.csv"]
    if not paths:
        raise FileNotFoundError(f"No per-case traces found in {trace_dir}")
    return paths


def load_rows(trace_dir: Path) -> pd.DataFrame:
    frames = []
    for path in trace_paths(trace_dir):
        df = pd.read_csv(path, usecols=lambda col: col in USECOLS)
        occ = bool_series(df["occupied"])
        df = df.loc[occ, ["weather", "mean_pmv", "expected_tsv", "discomfort_probability"]].copy()
        for col in ["mean_pmv", "expected_tsv", "discomfort_probability"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["mean_pmv", "expected_tsv", "discomfort_probability"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def threshold_metrics(values: np.ndarray, high_tail: np.ndarray, threshold: float) -> dict[str, float]:
    pred = values >= threshold
    n = len(values)
    tp = int(np.sum(pred & high_tail))
    tn = int(np.sum(~pred & ~high_tail))
    fp = int(np.sum(pred & ~high_tail))
    fn = int(np.sum(~pred & high_tail))
    return {
        "threshold": float(threshold),
        "disagreement_pct": float((fp + fn) / n * 100.0),
        "false_negative_pct": float(fn / n * 100.0),
        "false_positive_pct": float(fp / n * 100.0),
        "sensitivity_pct": float(tp / max(tp + fn, 1) * 100.0),
        "specificity_pct": float(tn / max(tn + fp, 1) * 100.0),
    }


def best_threshold(values: np.ndarray, high_tail: np.ndarray) -> dict[str, float]:
    qs = np.linspace(0.001, 0.999, 999)
    thresholds = np.unique(np.quantile(values[np.isfinite(values)], qs))
    best = None
    for threshold in thresholds:
        rec = threshold_metrics(values, high_tail, float(threshold))
        if best is None or rec["disagreement_pct"] < best["disagreement_pct"]:
            best = rec
    assert best is not None
    return best


def summarize(df: pd.DataFrame, tail_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["abs_pmv"] = df["mean_pmv"].abs()
    df["abs_mu_tsv"] = df["expected_tsv"].abs()
    df["high_tail"] = df["discomfort_probability"] >= tail_threshold
    rows = []
    for label, values, standard_threshold in [
        ("abs_mean_pmv", df["abs_pmv"].to_numpy(float), 0.5),
        ("abs_mu_tsv", df["abs_mu_tsv"].to_numpy(float), 0.5),
    ]:
        high_tail = df["high_tail"].to_numpy(bool)
        best = best_threshold(values, high_tail)
        standard = threshold_metrics(values, high_tail, standard_threshold)
        rows.append(
            {
                "scalar": label,
                "rows": int(len(df)),
                "tail_threshold": tail_threshold,
                "pearson_r": float(pd.Series(values).corr(df["discomfort_probability"], method="pearson")),
                "spearman_r": float(pd.Series(values).corr(df["discomfort_probability"], method="spearman")),
                "standard_threshold": standard_threshold,
                "standard_disagreement_pct": standard["disagreement_pct"],
                "standard_false_negative_pct": standard["false_negative_pct"],
                "standard_false_positive_pct": standard["false_positive_pct"],
                "standard_sensitivity_pct": standard["sensitivity_pct"],
                "standard_specificity_pct": standard["specificity_pct"],
                "best_threshold": best["threshold"],
                "best_disagreement_pct": best["disagreement_pct"],
                "best_false_negative_pct": best["false_negative_pct"],
                "best_false_positive_pct": best["false_positive_pct"],
                "best_sensitivity_pct": best["sensitivity_pct"],
                "best_specificity_pct": best["specificity_pct"],
            }
        )

    meta = pd.DataFrame([parse_weather(w) for w in df["weather"]])
    city_df = pd.concat([df.reset_index(drop=True), meta], axis=1)
    city_rows = []
    for city, group in city_df.groupby("city", sort=True):
        high_tail = group["high_tail"].to_numpy(bool)
        for label, col, standard_threshold in [
            ("abs_mean_pmv", "abs_pmv", 0.5),
            ("abs_mu_tsv", "abs_mu_tsv", 0.5),
        ]:
            values = group[col].to_numpy(float)
            standard = threshold_metrics(values, high_tail, standard_threshold)
            city_rows.append(
                {
                    "city": city,
                    "scalar": label,
                    "rows": int(len(group)),
                    "high_tail_pct": float(high_tail.mean() * 100.0),
                    "pearson_r": float(pd.Series(values).corr(group["discomfort_probability"], method="pearson")),
                    "standard_disagreement_pct": standard["disagreement_pct"],
                    "standard_false_negative_pct": standard["false_negative_pct"],
                    "standard_false_positive_pct": standard["false_positive_pct"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(city_rows)


def write_plot(df: pd.DataFrame, out_dir: Path, sample_rows: int, tail_threshold: float) -> Path:
    sample = df.sample(n=min(sample_rows, len(df)), random_state=42).copy()
    sample["abs_pmv"] = sample["mean_pmv"].abs()
    sample["abs_mu_tsv"] = sample["expected_tsv"].abs()
    x_max = 3.0
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=180, sharey=True)
    specs = [
        (axes[0], "abs_mu_tsv", r"$|\mu_{\mathrm{TSV}}|$"),
        (axes[1], "abs_pmv", r"$|\mathrm{PMV}|$"),
    ]
    for ax, col, xlabel in specs:
        ax.hexbin(
            sample[col],
            sample["discomfort_probability"],
            gridsize=58,
            mincnt=1,
            cmap="viridis",
            linewidths=0,
        )
        ax.axhline(tail_threshold, color="#7a2d42", lw=0.9, ls="--")
        ax.axvline(0.5, color="#666666", lw=0.8, ls=":")
        ax.set_xlim(0.0, x_max)
        ax.set_xlabel(xlabel)
        ax.grid(color="#dddddd", lw=0.5)
    axes[0].set_ylabel(r"$p_{\mathrm{tail}}$")
    axes[0].set_title("Mean TSV scalar", weight="bold")
    axes[1].set_title("PMV scalar", weight="bold")
    fig.tight_layout()
    path = out_dir / "scalar_tail_comparison_hexbin.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_markdown(out_dir: Path, global_summary: pd.DataFrame, city_summary: pd.DataFrame, plot: Path) -> Path:
    path = out_dir / "scalar_tail_comparison_summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Scalar-vs-Tail Diagnostic Summary\n\n")
        f.write("This diagnostic compares two scalar comfort summaries against the discomfort-tail probability threshold. It is a representation audit, not a controller comparison.\n\n")
        f.write("## Global Summary\n\n")
        f.write("```csv\n")
        f.write(global_summary.to_csv(index=False))
        f.write("```\n\n")
        f.write("## City Summary\n\n")
        f.write("```csv\n")
        f.write(city_summary.to_csv(index=False))
        f.write("```\n\n")
        f.write("## Figure\n\n")
        f.write(f"- `{plot.name}`\n")
    return path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.trace_dir)
    global_summary, city_summary = summarize(df, args.tail_threshold)
    global_summary.to_csv(args.output_dir / "scalar_tail_global_summary.csv", index=False)
    city_summary.to_csv(args.output_dir / "scalar_tail_city_summary.csv", index=False)
    plot = write_plot(df, args.output_dir, args.sample_rows, args.tail_threshold)
    md = write_markdown(args.output_dir, global_summary, city_summary, plot)
    print(f"[write] {args.output_dir / 'scalar_tail_global_summary.csv'}")
    print(f"[write] {args.output_dir / 'scalar_tail_city_summary.csv'}")
    print(f"[write] {plot}")
    print(f"[write] {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

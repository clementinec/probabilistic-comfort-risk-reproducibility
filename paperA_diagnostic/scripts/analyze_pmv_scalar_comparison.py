#!/usr/bin/env python3
"""Compare scalar PMV and expected-TSV summaries against p_tail diagnostics."""

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
DEFAULT_OUT = ROOT / "diagnostics" / "pmv_scalar_comparison"
TAIL_THRESHOLD = 0.20

USECOLS = {
    "weather",
    "occupied",
    "expected_tsv",
    "discomfort_probability",
    "mean_pmv",
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
    parser.add_argument("--tail-threshold", type=float, default=TAIL_THRESHOLD)
    parser.add_argument("--plot-sample-rows", type=int, default=250_000)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def parse_weather(weather: str) -> dict[str, object]:
    match = WEATHER_RE.match(str(weather))
    if not match:
        return {
            "city": np.nan,
            "scenario_raw": np.nan,
            "time_slice": np.nan,
            "severity": np.nan,
            "weather_year": np.nan,
        }
    data = match.groupdict()
    data["city"] = data["city"].title()
    data["weather_year"] = int(data["weather_year"])
    return data


def load_rows(trace_dir: Path) -> pd.DataFrame:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    if not paths:
        raise FileNotFoundError(f"No diagnostic reference traces found in {trace_dir}")
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=lambda col: col in USECOLS)
        keep = (
            bool_series(df["occupied"])
            & df["expected_tsv"].notna()
            & df["discomfort_probability"].notna()
            & df["mean_pmv"].notna()
        )
        frames.append(df.loc[keep].copy())
    rows = pd.concat(frames, ignore_index=True)
    rows["p_tail"] = pd.to_numeric(rows["discomfort_probability"], errors="coerce")
    rows["abs_mu_tsv"] = pd.to_numeric(rows["expected_tsv"], errors="coerce").abs()
    rows["abs_pmv"] = pd.to_numeric(rows["mean_pmv"], errors="coerce").abs()
    meta = pd.DataFrame([parse_weather(w) for w in rows["weather"]], index=rows.index)
    for col in meta.columns:
        rows[col] = meta[col]
    return rows.dropna(subset=["p_tail", "abs_mu_tsv", "abs_pmv"])


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator * 100.0)


def best_threshold(rows: pd.DataFrame, scalar: str, tail_threshold: float) -> dict[str, object]:
    y = rows["p_tail"].ge(tail_threshold).to_numpy(bool)
    x = rows[scalar].to_numpy(float)
    grid = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 501)))
    best = None
    for threshold in grid:
        pred = x >= threshold
        disagreements = pred != y
        rate = disagreements.mean()
        if best is None or rate < best["disagreement_rate"]:
            fp = pred & ~y
            fn = ~pred & y
            best = {
                "threshold": float(threshold),
                "disagreement_rate": float(rate),
                "false_positive_rows": int(fp.sum()),
                "false_negative_rows": int(fn.sum()),
                "flagged_rows": int(pred.sum()),
                "high_tail_rows": int(y.sum()),
            }
    assert best is not None
    out = {
        "scalar": scalar,
        "threshold": best["threshold"],
        "disagreement_pct": best["disagreement_rate"] * 100.0,
        "false_negative_pct_all": pct(best["false_negative_rows"], len(rows)),
        "false_positive_pct_all": pct(best["false_positive_rows"], len(rows)),
        "missed_high_tail_pct_of_high_tail": pct(best["false_negative_rows"], best["high_tail_rows"]),
        "false_alarm_pct_of_flagged": pct(best["false_positive_rows"], best["flagged_rows"]),
        "flagged_pct_all": pct(best["flagged_rows"], len(rows)),
        "high_tail_pct_all": pct(best["high_tail_rows"], len(rows)),
    }
    return out


def threshold_metrics(rows: pd.DataFrame, scalar: str, threshold: float, tail_threshold: float, label: str) -> dict[str, object]:
    y = rows["p_tail"].ge(tail_threshold).to_numpy(bool)
    pred = rows[scalar].to_numpy(float) >= threshold
    fp = pred & ~y
    fn = ~pred & y
    return {
        "scalar": label,
        "threshold": threshold,
        "disagreement_pct": pct((pred != y).sum(), len(rows)),
        "false_negative_pct_all": pct(fn.sum(), len(rows)),
        "false_positive_pct_all": pct(fp.sum(), len(rows)),
        "missed_high_tail_pct_of_high_tail": pct(fn.sum(), y.sum()),
        "false_alarm_pct_of_flagged": pct(fp.sum(), pred.sum()),
        "flagged_pct_all": pct(pred.sum(), len(rows)),
        "high_tail_pct_all": pct(y.sum(), len(rows)),
    }


def summarize_by_city(rows: pd.DataFrame, tail_threshold: float) -> pd.DataFrame:
    out = []
    for city, group in rows.groupby("city", sort=True):
        y = group["p_tail"].ge(tail_threshold)
        pmv_neutral = group["abs_pmv"].le(0.5)
        out.append(
            {
                "city": city,
                "rows": int(len(group)),
                "high_tail_pct": pct(y.sum(), len(group)),
                "pmv_neutral_pct": pct(pmv_neutral.sum(), len(group)),
                "high_tail_inside_pmv_neutral_pct_all": pct((y & pmv_neutral).sum(), len(group)),
                "high_tail_inside_pmv_neutral_pct_of_high_tail": pct((y & pmv_neutral).sum(), y.sum()),
                "mean_abs_pmv": float(group["abs_pmv"].mean()),
                "mean_abs_mu_tsv": float(group["abs_mu_tsv"].mean()),
            }
        )
    return pd.DataFrame(out)


def write_markdown(out_dir: Path, rows: pd.DataFrame, scalar_summary: pd.DataFrame, city_summary: pd.DataFrame) -> None:
    path = out_dir / "pmv_scalar_comparison_summary.md"
    corr = {
        "pearson_abs_mu_p_tail": rows["abs_mu_tsv"].corr(rows["p_tail"], method="pearson"),
        "spearman_abs_mu_p_tail": rows["abs_mu_tsv"].corr(rows["p_tail"], method="spearman"),
        "pearson_abs_pmv_p_tail": rows["abs_pmv"].corr(rows["p_tail"], method="pearson"),
        "spearman_abs_pmv_p_tail": rows["abs_pmv"].corr(rows["p_tail"], method="spearman"),
    }
    lines = [
        "# PMV Scalar Comparison Summary",
        "",
        "This diagnostic compares scalar PMV and expected-TSV summaries against the same `p_tail >= 0.20` high-tail flag used in the Paper A diagnostic.",
        "",
        f"- Occupied rows with PMV and probability outputs: {len(rows):,}",
        f"- `|mu_TSV|` Pearson/Spearman with `p_tail`: {corr['pearson_abs_mu_p_tail']:.3f} / {corr['spearman_abs_mu_p_tail']:.3f}",
        f"- `|PMV|` Pearson/Spearman with `p_tail`: {corr['pearson_abs_pmv_p_tail']:.3f} / {corr['spearman_abs_pmv_p_tail']:.3f}",
        "",
        "## Scalar Threshold Summary",
        "",
        "```csv",
        scalar_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## City PMV Neutral-Band Summary",
        "",
        "```csv",
        city_summary.to_csv(index=False).strip(),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot(rows: pd.DataFrame, out_dir: Path, sample_rows: int) -> None:
    sample = rows.sample(n=min(sample_rows, len(rows)), random_state=20260616)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, x_col, label in [
        (axes[0], "abs_mu_tsv", r"$|\mu_{\mathrm{TSV}}|$"),
        (axes[1], "abs_pmv", r"$|\mathrm{PMV}|$"),
    ]:
        ax.scatter(sample[x_col], sample["p_tail"], s=2, alpha=0.08, linewidths=0)
        ax.axhline(0.20, color="#444444", linewidth=1.1, linestyle="--")
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$p_{\mathrm{tail}}$")
    axes[1].axvline(0.5, color="#b03a2e", linewidth=1.1, linestyle=":")
    axes[1].text(0.52, 0.93, r"$|\mathrm{PMV}|=0.5$", transform=axes[1].get_xaxis_transform(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pmv_vs_expected_tsv_scalar_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.trace_dir)
    scalar_rows = [
        best_threshold(rows, "abs_mu_tsv", args.tail_threshold),
        best_threshold(rows, "abs_pmv", args.tail_threshold),
        threshold_metrics(rows, "abs_pmv", 0.5, args.tail_threshold, "abs_pmv_standard_0p5"),
    ]
    scalar_summary = pd.DataFrame(scalar_rows)
    city_summary = summarize_by_city(rows, args.tail_threshold)
    scalar_summary.to_csv(args.output_dir / "pmv_scalar_threshold_summary.csv", index=False)
    city_summary.to_csv(args.output_dir / "pmv_scalar_city_summary.csv", index=False)
    plot(rows, args.output_dir, args.plot_sample_rows)
    write_markdown(args.output_dir, rows, scalar_summary, city_summary)
    print(f"[write] {args.output_dir / 'pmv_scalar_comparison_summary.md'}")


if __name__ == "__main__":
    main()

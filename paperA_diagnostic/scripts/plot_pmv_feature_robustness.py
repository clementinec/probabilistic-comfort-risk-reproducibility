#!/usr/bin/env python3
"""Compare PMV-scalar relationship for primary and no-PMV TSV predictors."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = ROOT / "runs" / "diagnostic_reference_zone_raw_full" / "traces"
DEFAULT_NO_PMV = ROOT / "runs" / "diagnostic_reference_zone_raw_no_pmv" / "traces"
DEFAULT_OUT = ROOT / "diagnostics" / "pmv_feature_robustness"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

USECOLS = ["weather", "occupied", "mean_pmv", "expected_tsv", "discomfort_probability"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-trace-dir", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--no-pmv-trace-dir", type=Path, default=DEFAULT_NO_PMV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-rows", type=int, default=300_000)
    parser.add_argument("--tail-threshold", type=float, default=0.20)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def trace_paths(trace_dir: Path) -> list[Path]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    paths = [path for path in paths if path.name != "medium_office_control_traces.csv"]
    if not paths:
        raise FileNotFoundError(f"No per-case traces found in {trace_dir}")
    return paths


def load_rows(trace_dir: Path, label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in trace_paths(trace_dir):
        df = pd.read_csv(path, usecols=lambda col: col in USECOLS)
        occ = bool_series(df["occupied"])
        df = df.loc[occ, ["weather", "mean_pmv", "expected_tsv", "discomfort_probability"]].copy()
        for col in ["mean_pmv", "expected_tsv", "discomfort_probability"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["mean_pmv", "expected_tsv", "discomfort_probability"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["model"] = label
    out["abs_pmv"] = out["mean_pmv"].abs()
    out["abs_mu_tsv"] = out["expected_tsv"].abs()
    return out


def summarize(df: pd.DataFrame, tail_threshold: float) -> pd.DataFrame:
    rows = []
    for model, sdf in df.groupby("model", sort=False):
        high_tail = sdf["discomfort_probability"].ge(tail_threshold)
        rows.append(
            {
                "model": model,
                "rows": int(len(sdf)),
                "mean_p_tail": float(sdf["discomfort_probability"].mean()),
                "high_tail_pct": float(high_tail.mean() * 100.0),
                "pmv_pearson_r": float(
                    sdf["abs_pmv"].corr(sdf["discomfort_probability"], method="pearson")
                ),
                "pmv_spearman_r": float(
                    sdf["abs_pmv"].corr(sdf["discomfort_probability"], method="spearman")
                ),
                "mu_pearson_r": float(
                    sdf["abs_mu_tsv"].corr(sdf["discomfort_probability"], method="pearson")
                ),
                "mu_spearman_r": float(
                    sdf["abs_mu_tsv"].corr(sdf["discomfort_probability"], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def write_plot(df: pd.DataFrame, out_dir: Path, sample_rows: int, tail_threshold: float) -> Path:
    samples = []
    per_model = max(sample_rows // max(df["model"].nunique(), 1), 1)
    for model, sdf in df.groupby("model", sort=False):
        samples.append(sdf.sample(n=min(per_model, len(sdf)), random_state=42).copy())
    sample = pd.concat(samples, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), dpi=180, sharex=True, sharey=True)
    specs = [
        ("Primary predictor", "Primary TSV probabilities", "viridis"),
        ("No-PMV predictor", "No-PMV TSV probabilities", "magma"),
    ]
    for ax, (model, title, cmap) in zip(axes, specs):
        sdf = sample[sample["model"] == model]
        hb = ax.hexbin(
            sdf["abs_pmv"],
            sdf["discomfort_probability"],
            gridsize=58,
            mincnt=1,
            cmap=cmap,
            linewidths=0,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Sampled records per bin", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.axhline(tail_threshold, color="#7a2d42", lw=0.9, ls="--")
        ax.axvline(0.5, color="#666666", lw=0.8, ls=":")
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title, weight="bold", fontsize=10)
        ax.set_xlabel(r"$|\mathrm{PMV}|$")
        ax.grid(color="#dddddd", lw=0.5)
    axes[0].set_ylabel(r"$p_{\mathrm{tail}}$ from primary model")
    axes[1].set_ylabel(r"$p_{\mathrm{tail}}$ from no-PMV model")
    fig.tight_layout()
    pdf = out_dir / "pmv_feature_robustness_hexbin.pdf"
    png = out_dir / "pmv_feature_robustness_hexbin.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return pdf


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary = load_rows(args.primary_trace_dir, "Primary predictor")
    no_pmv = load_rows(args.no_pmv_trace_dir, "No-PMV predictor")
    df = pd.concat([primary, no_pmv], ignore_index=True)
    summary = summarize(df, args.tail_threshold)
    summary_path = args.output_dir / "pmv_feature_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot = write_plot(df, args.output_dir, args.sample_rows, args.tail_threshold)
    print(f"[write] {summary_path}")
    print(f"[write] {plot}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

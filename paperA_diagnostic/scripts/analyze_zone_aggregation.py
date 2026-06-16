#!/usr/bin/env python3
"""Build Paper A diagnostics for zone aggregation of discomfort-tail risk."""

from __future__ import annotations

import os
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "runs" / "diagnostic_reference" / "traces" / "medium_office_control_traces.csv"
TRACE_DIR = ROOT / "runs" / "diagnostic_reference" / "traces"
OUT = ROOT / "diagnostics" / "zone_aggregation"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISC_LOW = 0.065
DISC_UP = 0.35
TAIL_DIAGNOSTIC = 0.20
MU_SIGN_EPS = 0.05
DTAIL_SIGN_EPS = 0.01

FIXED_COLUMNS = {
    "weather",
    "strategy",
    "month",
    "day",
    "hour",
    "current_time",
    "occupied",
    "expected_tsv",
    "discomfort_probability",
    "warm_discomfort_probability",
    "cold_discomfort_probability",
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
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def pct(mask: pd.Series | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean() * 100.0)


def parse_weather_metadata(weather: str) -> dict[str, object]:
    match = WEATHER_RE.match(str(weather))
    if not match:
        return {
            "city": str(weather).split("_")[0].title(),
            "scenario_raw": "",
            "time_slice": "",
            "severity": "",
            "year": np.nan,
        }
    data = match.groupdict()
    return {
        "city": data["city"].title(),
        "scenario_raw": data["scenario_raw"],
        "time_slice": data["time_slice"],
        "severity": data["severity"],
        "year": int(data["weather_year"]),
    }


def gate_category(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(arr >= DISC_UP, 2, np.where(arr > DISC_LOW, 1, 0))


def sign(values: pd.Series | np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(arr > eps, 1, np.where(arr < -eps, -1, 0))


def trace_usecols(col: str) -> bool:
    return col in FIXED_COLUMNS or (
        col.startswith("zone_")
        and (
            col.endswith("_p_disc")
            or col.endswith("_expected_tsv")
            or col.endswith("_d_tail")
        )
    )


def load_trace(trace: Path, trace_dir: Path) -> pd.DataFrame:
    if trace.exists():
        df = pd.read_csv(trace, usecols=trace_usecols)
    elif trace_dir.exists():
        paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
        if not paths:
            raise FileNotFoundError(f"No diagnostic traces found in {trace_dir}")
        frames = [pd.read_csv(path, usecols=trace_usecols) for path in paths]
        df = pd.concat(frames, ignore_index=True)
    else:
        raise FileNotFoundError(trace)
    df = df[df["occupied"]].copy()
    df = df[df["discomfort_probability"].notna()].copy()
    meta = pd.DataFrame([parse_weather_metadata(stem) for stem in df["weather"]], index=df.index)
    for col in meta.columns:
        df[col] = meta[col]
    return df


def add_aggregation_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    p_cols = [
        c
        for c in df.columns
        if c.startswith("zone_")
        and c.endswith("_p_disc")
        and c not in {"zone_heating_rate_w", "zone_cooling_rate_w"}
    ]
    mu_cols = [c for c in df.columns if c.startswith("zone_") and c.endswith("_expected_tsv")]
    d_cols = [c for c in df.columns if c.startswith("zone_") and c.endswith("_d_tail")]
    if not (p_cols and mu_cols and d_cols):
        raise ValueError("Missing zone-level probability columns.")

    work = df.copy()
    p = work[p_cols]
    mu = work[mu_cols]
    d_tail = work[d_cols]
    work["zone_p_disc_max"] = p.max(axis=1)
    work["zone_p_disc_p90"] = p.quantile(0.90, axis=1)
    work["zone_p_disc_min"] = p.min(axis=1)
    work["zone_p_disc_spread"] = work["zone_p_disc_max"] - work["zone_p_disc_min"]
    work["zone_p_disc_sd"] = p.std(axis=1)
    work["zones_above_disc_low"] = p.gt(DISC_LOW).sum(axis=1)
    work["zones_above_tail_020"] = p.ge(TAIL_DIAGNOSTIC).sum(axis=1)
    work["zones_above_disc_up"] = p.ge(DISC_UP).sum(axis=1)
    work["mean_tail_category"] = gate_category(work["discomfort_probability"])
    work["max_zone_tail_category"] = gate_category(work["zone_p_disc_max"])
    work["p90_zone_tail_category"] = gate_category(work["zone_p_disc_p90"])
    work["max_zone_more_severe"] = work["max_zone_tail_category"] > work["mean_tail_category"]
    work["p90_zone_more_severe"] = work["p90_zone_tail_category"] > work["mean_tail_category"]

    work["mean_d_tail"] = work["warm_discomfort_probability"] - work["cold_discomfort_probability"]
    work["zone_d_tail_max_abs"] = d_tail.abs().max(axis=1)
    work["zone_mu_max_abs"] = mu.abs().max(axis=1)
    work["zone_mu_spread"] = mu.max(axis=1) - mu.min(axis=1)
    work["zone_d_tail_spread"] = d_tail.max(axis=1) - d_tail.min(axis=1)

    mu_sign = sign(work["expected_tsv"], MU_SIGN_EPS)
    d_sign = sign(work["mean_d_tail"], DTAIL_SIGN_EPS)
    comparable = (mu_sign != 0) & (d_sign != 0)
    work["mu_d_tail_sign_comparable"] = comparable
    work["mu_d_tail_sign_conflict"] = comparable & (mu_sign != d_sign)

    zone_mu_sign = np.sign(mu.to_numpy(float))
    zone_d_sign = np.sign(d_tail.to_numpy(float))
    work["zone_mu_has_both_signs"] = (zone_mu_sign.min(axis=1) < 0) & (
        zone_mu_sign.max(axis=1) > 0
    )
    work["zone_d_tail_has_both_signs"] = (zone_d_sign.min(axis=1) < 0) & (
        zone_d_sign.max(axis=1) > 0
    )
    return work, p_cols, mu_cols, d_cols


def summarize(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("All", "All", work)]
    groups.extend((city, "All", sdf) for city, sdf in work.groupby("city", sort=True))
    groups.extend(
        (city, str(year), sdf)
        for (city, year), sdf in work.groupby(["city", "year"], sort=True)
    )
    for city, year, sdf in groups:
        comp = sdf[sdf["mu_d_tail_sign_comparable"]]
        rows.append(
            {
                "city": city,
                "year": year,
                "occupied_probability_steps": int(len(sdf)),
                "mean_p_disc_mean": float(sdf["discomfort_probability"].mean()),
                "max_zone_p_disc_mean": float(sdf["zone_p_disc_max"].mean()),
                "p90_zone_p_disc_mean": float(sdf["zone_p_disc_p90"].mean()),
                "mean_high_tail_020_pct": pct(sdf["discomfort_probability"].ge(TAIL_DIAGNOSTIC)),
                "max_zone_high_tail_020_pct": pct(sdf["zone_p_disc_max"].ge(TAIL_DIAGNOSTIC)),
                "p90_zone_high_tail_020_pct": pct(sdf["zone_p_disc_p90"].ge(TAIL_DIAGNOSTIC)),
                "hidden_tail_020_pct": pct(
                    sdf["discomfort_probability"].lt(TAIL_DIAGNOSTIC)
                    & sdf["zone_p_disc_max"].ge(TAIL_DIAGNOSTIC)
                ),
                "mean_low_but_any_zone_above_disc_low_pct": pct(
                    sdf["discomfort_probability"].le(DISC_LOW)
                    & sdf["zone_p_disc_max"].gt(DISC_LOW)
                ),
                "mean_below_disc_up_but_any_zone_above_disc_up_pct": pct(
                    sdf["discomfort_probability"].lt(DISC_UP)
                    & sdf["zone_p_disc_max"].ge(DISC_UP)
                ),
                "max_zone_more_severe_pct": pct(sdf["max_zone_more_severe"]),
                "p90_zone_more_severe_pct": pct(sdf["p90_zone_more_severe"]),
                "mean_zone_p_disc_spread": float(sdf["zone_p_disc_spread"].mean()),
                "p95_zone_p_disc_spread": float(sdf["zone_p_disc_spread"].quantile(0.95)),
                "mean_zones_above_tail_020": float(sdf["zones_above_tail_020"].mean()),
                "zone_mu_both_signs_pct": pct(sdf["zone_mu_has_both_signs"]),
                "zone_d_tail_both_signs_pct": pct(sdf["zone_d_tail_has_both_signs"]),
                "mu_d_tail_sign_conflict_pct_of_comparable": pct(
                    comp["mu_d_tail_sign_conflict"]
                )
                if len(comp)
                else float("nan"),
                "mu_d_tail_sign_comparable_steps": int(len(comp)),
                "mean_abs_mu": float(sdf["expected_tsv"].abs().mean()),
                "mean_abs_d_tail": float(sdf["mean_d_tail"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_tail_direction_subsets(work: pd.DataFrame) -> pd.DataFrame:
    subsets = [
        ("all_occupied", work),
        ("above_low_tail_reference", work[work["discomfort_probability"].gt(DISC_LOW)]),
        (
            "between_low_and_upper_tail_reference",
            work[
                work["discomfort_probability"].gt(DISC_LOW)
                & work["discomfort_probability"].lt(DISC_UP)
            ],
        ),
        ("above_upper_tail_reference", work[work["discomfort_probability"].ge(DISC_UP)]),
        (
            "diagnostic_high_tail_p_disc_ge_020",
            work[work["discomfort_probability"].ge(TAIL_DIAGNOSTIC)],
        ),
    ]
    rows = []
    for subset, sdf in subsets:
        groups = [("All", sdf)]
        groups.extend((city, cdf) for city, cdf in sdf.groupby("city", sort=True))
        for city, cdf in groups:
            comp = cdf[cdf["mu_d_tail_sign_comparable"]]
            rows.append(
                {
                    "subset": subset,
                    "city": city,
                    "steps": int(len(cdf)),
                    "comparable_steps": int(len(comp)),
                    "share_of_all_occupied_pct": (len(cdf) / len(work) * 100.0)
                    if len(work)
                    else float("nan"),
                    "mu_d_tail_sign_conflict_pct_of_comparable": pct(
                        comp["mu_d_tail_sign_conflict"]
                    )
                    if len(comp)
                    else float("nan"),
                    "mu_d_tail_sign_conflict_pct_of_subset": pct(
                        cdf["mu_d_tail_sign_conflict"]
                    )
                    if len(cdf)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def zone_ranking(work: pd.DataFrame, p_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in p_cols:
        zone = col.removeprefix("zone_").removesuffix("_p_disc")
        rows.append(
            {
                "zone": zone,
                "mean_p_disc": float(work[col].mean()),
                "p95_p_disc": float(work[col].quantile(0.95)),
                "high_tail_020_pct": pct(work[col].ge(TAIL_DIAGNOSTIC)),
                "above_disc_up_pct": pct(work[col].ge(DISC_UP)),
            }
        )
    return pd.DataFrame(rows).sort_values("high_tail_020_pct", ascending=False)


def write_top_hidden_examples(work: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "city",
        "year",
        "month",
        "day",
        "hour",
        "current_time",
        "outdoor_temp_c",
        "expected_tsv",
        "discomfort_probability",
        "zone_p_disc_max",
        "zone_p_disc_p90",
        "zone_p_disc_spread",
        "zones_above_tail_020",
        "mean_d_tail",
        "zone_d_tail_max_abs",
    ]
    hidden = work[
        work["discomfort_probability"].lt(TAIL_DIAGNOSTIC)
        & work["zone_p_disc_max"].ge(TAIL_DIAGNOSTIC)
    ].copy()
    hidden["gap_max_minus_mean"] = hidden["zone_p_disc_max"] - hidden["discomfort_probability"]
    cols.append("gap_max_minus_mean")
    return hidden.sort_values("gap_max_minus_mean", ascending=False)[cols].head(50)


def make_plot(summary: pd.DataFrame, work: pd.DataFrame) -> Path:
    city_year = summary[(summary["year"] == "All") & (summary["city"] != "All")].copy()
    city_year["label"] = city_year["city"]
    city_year = city_year.sort_values(["city"])

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.5), constrained_layout=True)

    x = np.arange(len(city_year))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        city_year["mean_high_tail_020_pct"],
        width,
        label="Mean-zone aggregation",
        color="#4c78a8",
    )
    axes[0].bar(
        x + width / 2,
        city_year["max_zone_high_tail_020_pct"],
        width,
        label="Any-zone maximum",
        color="#d45b43",
    )
    axes[0].set_ylabel("High-tail exposure (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(city_year["label"], rotation=35, ha="right")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis="y", color="#d8d8d8", linewidth=0.6)

    axes[1].bar(
        x,
        city_year["hidden_tail_020_pct"],
        color="#8c6bb1",
    )
    axes[1].set_ylabel("Hidden any-zone\nhigh-tail states (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(city_year["label"], rotation=35, ha="right")
    axes[1].grid(axis="y", color="#d8d8d8", linewidth=0.6)

    plot_data = [
        work.loc[work["city"] == city, "zone_p_disc_spread"].dropna().to_numpy()
        for city in sorted(work["city"].unique())
    ]
    axes[2].boxplot(
        plot_data,
        labels=sorted(work["city"].unique()),
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#b7d7d8", "edgecolor": "#3e6b70"},
        medianprops={"color": "#1f3437", "linewidth": 1.4},
    )
    axes[2].set_ylabel("Within-timestep\nzone p_disc spread")
    axes[2].grid(axis="y", color="#d8d8d8", linewidth=0.6)

    fig.suptitle(
        "Zone aggregation diagnostic for Medium Office future-weather traces",
        fontsize=13,
        fontweight="bold",
    )
    out = OUT / "zone_aggregation_diagnostic.png"
    pdf = out.with_suffix(".pdf")
    fig.savefig(out, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return out


def write_markdown(
    summary: pd.DataFrame, direction_summary: pd.DataFrame, zone_rank: pd.DataFrame
) -> Path:
    all_row = summary[(summary["city"] == "All") & (summary["year"] == "All")].iloc[0]
    city_rows = summary[summary["year"] == "All"].query("city != 'All'")
    direction_all = direction_summary[
        (direction_summary["city"] == "All")
        & direction_summary["subset"].isin(
            [
                "above_low_tail_reference",
                "above_upper_tail_reference",
                "diagnostic_high_tail_p_disc_ge_020",
            ]
        )
    ]
    low_reference_city = direction_summary[
        (direction_summary["subset"] == "above_low_tail_reference")
        & (direction_summary["city"] != "All")
    ]
    path = OUT / "zone_tail_diagnostic_summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Zone Aggregation Diagnostic\n\n")
        f.write("Source trace: `runs/diagnostic_reference/traces/medium_office_control_traces.csv`\n\n")
        f.write("## Key Results\n\n")
        f.write(
            "- Across all occupied probability timesteps, mean aggregation gives "
            f"{all_row.mean_high_tail_020_pct:.1f}% high-tail exposure at "
            "`p_disc >= 0.20`, while any-zone aggregation gives "
            f"{all_row.max_zone_high_tail_020_pct:.1f}%.\n"
        )
        f.write(
            "- Mean aggregation hides at least one zone above `p_disc >= 0.20` in "
            f"{all_row.hidden_tail_020_pct:.1f}% of occupied probability timesteps.\n"
        )
        f.write(
            "- A max-zone aggregation would assign a more severe descriptive tail category in "
            f"{all_row.max_zone_more_severe_pct:.1f}% of occupied probability timesteps; "
            f"a p90-zone aggregation would do so in {all_row.p90_zone_more_severe_pct:.1f}%.\n"
        )
        f.write(
            "- Aggregate `mu_TSV` and aggregate `d_tail` signs conflict in "
            f"{all_row.mu_d_tail_sign_conflict_pct_of_comparable:.2f}% of comparable timesteps "
            f"(n={int(all_row.mu_d_tail_sign_comparable_steps)}), showing where expected sensation "
            "and directional tail exposure carry different information.\n"
        )
        f.write(
            "- Among states above the low reference threshold (`p_tail > 0.065`), the same sign "
            "conflict is "
            f"{direction_all.loc[direction_all['subset'] == 'above_low_tail_reference', 'mu_d_tail_sign_conflict_pct_of_comparable'].iloc[0]:.2f}% "
            "of comparable states.\n"
        )
        f.write("\n## City-Level Summary\n\n")
        f.write(
            markdown_table(
                city_rows[
                    [
                        "city",
                        "mean_high_tail_020_pct",
                        "max_zone_high_tail_020_pct",
                        "hidden_tail_020_pct",
                        "max_zone_more_severe_pct",
                        "mu_d_tail_sign_conflict_pct_of_comparable",
                    ]
                ]
            )
        )
        f.write("\n\n## Tail-Direction Subsets\n\n")
        f.write(markdown_table(direction_all))
        f.write("\n\n### Low-reference subset by city\n\n")
        f.write(markdown_table(low_reference_city))
        f.write("\n\n## Highest-Risk Zones\n\n")
        f.write(markdown_table(zone_rank.head(8)))
        f.write("\n")
    return path


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


def main() -> int:
    args = parse_args()
    global OUT
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_trace(args.trace, args.trace_dir)
    work, p_cols, _, _ = add_aggregation_fields(df)
    summary = summarize(work)
    direction_summary = summarize_tail_direction_subsets(work)
    zone_rank = zone_ranking(work, p_cols)
    hidden_examples = write_top_hidden_examples(work)

    work_out = OUT / "zone_aggregation_rows_compact.csv"
    cols = [
        "city",
        "year",
        "weather",
        "month",
        "day",
        "hour",
        "current_time",
        "occupied",
        "expected_tsv",
        "discomfort_probability",
        "warm_discomfort_probability",
        "cold_discomfort_probability",
        "mean_d_tail",
        "zone_p_disc_max",
        "zone_p_disc_p90",
        "zone_p_disc_spread",
        "zones_above_disc_low",
        "zones_above_tail_020",
        "zones_above_disc_up",
        "mean_tail_category",
        "max_zone_tail_category",
        "p90_zone_tail_category",
        "max_zone_more_severe",
        "p90_zone_more_severe",
        "zone_mu_has_both_signs",
        "zone_d_tail_has_both_signs",
        "mu_d_tail_sign_comparable",
        "mu_d_tail_sign_conflict",
    ]
    work[cols].to_csv(work_out, index=False)
    summary.to_csv(OUT / "zone_aggregation_summary.csv", index=False)
    direction_summary.to_csv(OUT / "tail_direction_subset_summary.csv", index=False)
    zone_rank.to_csv(OUT / "zone_risk_ranking.csv", index=False)
    hidden_examples.to_csv(OUT / "hidden_zone_tail_examples.csv", index=False)
    make_plot(summary, work)
    write_markdown(summary, direction_summary, zone_rank)
    print(f"[diagnostic] wrote {OUT}")
    print(summary[(summary["city"] == "All") & (summary["year"] == "All")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

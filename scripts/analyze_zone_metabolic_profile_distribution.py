#!/usr/bin/env python3
"""Summarize zone-resolved tail risk across metabolic profiles."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_metabolic_spread as mean_met
import legacy_control_pipeline as runner

TRACE_DIR = ROOT / "restricted_inputs" / "simulation_traces"
MODEL_PATH = ROOT / "models" / "tsv_predictor_bundle.joblib"
OUT = ROOT / "outputs" / "metabolic_profile"

TAIL_THRESHOLD = 0.20
PROFILE_ORDER = ["COMP-Lo", "S-real", "COMP-Med", "EQ-Max", "COMP-Hi", "LEGACY"]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--predictor", choices=["ordinal", "nominal"], default="ordinal")
    parser.add_argument(
        "--bsa-mode",
        choices=["paper_convention", "fixed_1p80"],
        default="paper_convention",
    )
    parser.add_argument("--tail-threshold", type=float, default=TAIL_THRESHOLD)
    parser.add_argument("--max-cases", type=int, default=None)
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
    if zone.startswith("core_"):
        return "core"
    return "P" + zone[-1]


def zone_position_name(zone: str) -> str:
    return {
        "core": "Core",
        "P1": "P1 south",
        "P2": "P2 east",
        "P3": "P3 north",
        "P4": "P4 west",
    }[zone_position(zone)]


def trace_paths(trace_dir: Path, max_cases: int | None) -> list[Path]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    paths = [path for path in paths if path.name not in runner.COMBINED_TRACE_NAMES]
    if not paths:
        raise FileNotFoundError(f"No zone-raw traces found in {trace_dir}")
    return paths[:max_cases] if max_cases is not None else paths


def zone_raw_columns() -> list[str]:
    cols: list[str] = []
    for slug in runner.ZONE_FIELD_NAMES:
        cols.extend([f"zone_{slug}_ta_c", f"zone_{slug}_tr_c", f"zone_{slug}_rh_pct"])
    return cols


def load_trace(path: Path) -> pd.DataFrame:
    usecols = {
        "weather",
        "occupied",
        "outdoor_temp_c",
        "running_mean_outdoor_c",
        *zone_raw_columns(),
    }
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    raw_cols = set(zone_raw_columns())
    missing = sorted(raw_cols - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing zone raw columns: {', '.join(missing[:8])}")
    valid = mean_met.bool_series(df["occupied"])
    for col in raw_cols:
        valid &= df[col].notna()
    df = df.loc[valid].copy()
    for col in usecols - {"weather", "occupied"}:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def flatten_zone_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ta = df[[f"zone_{slug}_ta_c" for slug in runner.ZONE_FIELD_NAMES]].to_numpy(float)
    tr = df[[f"zone_{slug}_tr_c" for slug in runner.ZONE_FIELD_NAMES]].to_numpy(float)
    rh = df[[f"zone_{slug}_rh_pct" for slug in runner.ZONE_FIELD_NAMES]].to_numpy(float)
    return ta, tr, rh


def predict_zone_tail(bundle, df: pd.DataFrame, scenario: mean_met.MetScenario, predictor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ta, tr, rh = flatten_zone_inputs(df)
    n, z = ta.shape
    rm = df["running_mean_outdoor_c"].where(
        df["running_mean_outdoor_c"].notna(), df["outdoor_temp_c"]
    ).to_numpy(float)
    features = runner.build_features_from_arrays(
        ta=ta.reshape(-1),
        tr=tr.reshape(-1),
        v=np.full(n * z, 0.10),
        rh=rh.reshape(-1),
        met=np.full(n * z, scenario.met),
        clo=np.full(n * z, 0.65),
        bsa=np.full(n * z, scenario.bsa_m2),
        rm_out=np.repeat(rm, z),
        spec=bundle.spec,
    )
    probs = bundle.predict_ordinal(features) if predictor == "ordinal" else bundle.predict_nominal(features)
    probs = probs.reshape(n, z, -1)
    cold = probs[:, :, [0, 1]].sum(axis=2)
    warm = probs[:, :, [5, 6]].sum(axis=2)
    return cold + warm, warm, cold


def process_trace(path: Path, bundle, scenarios: list[mean_met.MetScenario], args: argparse.Namespace) -> list[dict[str, object]]:
    df = load_trace(path)
    if df.empty:
        return []
    weather = str(df["weather"].iloc[0])
    meta = mean_met.parse_weather(weather)
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        p_tail, warm, cold = predict_zone_tail(bundle, df, scenario, args.predictor)
        for idx, zone in enumerate(runner.ZONE_FIELD_NAMES):
            values = p_tail[:, idx]
            row = {
                "weather": weather,
                **meta,
                "scenario": scenario.scenario,
                "watts_person": scenario.watts_person,
                "met": scenario.met,
                "bsa_m2": scenario.bsa_m2,
                "zone": zone,
                "zone_label": ZONE_LABELS[zone],
                "floor": zone_floor(zone),
                "position": zone_position(zone),
                "position_name": zone_position_name(zone),
                "rows": int(len(values)),
                "mean_p_tail": float(values.mean()),
                "median_p_tail": float(np.median(values)),
                "p90_p_tail": float(np.quantile(values, 0.90)),
                "p95_p_tail": float(np.quantile(values, 0.95)),
                "high_tail_pct": float((values >= args.tail_threshold).mean() * 100.0),
                "mean_warm_tail": float(warm[:, idx].mean()),
                "mean_cold_tail": float(cold[:, idx].mean()),
            }
            rows.append(row)
    return rows


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(float), weights=weights.to_numpy(float)))


def summarize(case_zone: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_zone = (
        case_zone.groupby(
            [
                "scenario",
                "watts_person",
                "met",
                "zone",
                "zone_label",
                "floor",
                "position",
                "position_name",
            ],
            sort=False,
            observed=True,
        )
        .agg(
            cases=("weather", "nunique"),
            rows=("rows", "sum"),
            mean_p_tail=("mean_p_tail", "mean"),
            p95_case_mean_p_tail=("mean_p_tail", lambda s: float(s.quantile(0.95))),
            mean_high_tail_pct=("high_tail_pct", "mean"),
            median_case_high_tail_pct=("high_tail_pct", "median"),
            p95_case_high_tail_pct=("high_tail_pct", lambda s: float(s.quantile(0.95))),
            mean_warm_tail=("mean_warm_tail", "mean"),
            mean_cold_tail=("mean_cold_tail", "mean"),
        )
        .reset_index()
    )

    pivot = case_zone.pivot_table(
        index=["weather", "zone", "zone_label", "floor", "position", "position_name"],
        columns="scenario",
        values=["high_tail_pct", "mean_p_tail"],
        aggfunc="first",
        observed=True,
    )
    pivot.columns = [f"{metric}__{scenario}" for metric, scenario in pivot.columns]
    pivot = pivot.reset_index()
    high_cols = [f"high_tail_pct__{scenario}" for scenario in PROFILE_ORDER]
    mean_cols = [f"mean_p_tail__{scenario}" for scenario in PROFILE_ORDER]
    pivot["high_tail_pct_profile_spread"] = pivot[high_cols].max(axis=1) - pivot[high_cols].min(axis=1)
    pivot["mean_p_tail_profile_spread"] = pivot[mean_cols].max(axis=1) - pivot[mean_cols].min(axis=1)
    pivot["sreal_high_tail_gap_to_max_profile"] = pivot[high_cols].max(axis=1) - pivot["high_tail_pct__S-real"]
    pivot["sreal_mean_p_tail_gap_to_max_profile"] = pivot[mean_cols].max(axis=1) - pivot["mean_p_tail__S-real"]

    spread_zone = (
        pivot.groupby(
            ["zone", "zone_label", "floor", "position", "position_name"],
            sort=False,
            observed=True,
        )
        .agg(
            cases=("weather", "nunique"),
            mean_high_tail_pct_profile_spread=("high_tail_pct_profile_spread", "mean"),
            median_high_tail_pct_profile_spread=("high_tail_pct_profile_spread", "median"),
            p95_high_tail_pct_profile_spread=("high_tail_pct_profile_spread", lambda s: float(s.quantile(0.95))),
            mean_sreal_high_tail_gap_to_max_profile=("sreal_high_tail_gap_to_max_profile", "mean"),
            p95_sreal_high_tail_gap_to_max_profile=("sreal_high_tail_gap_to_max_profile", lambda s: float(s.quantile(0.95))),
            mean_p_tail_profile_spread=("mean_p_tail_profile_spread", "mean"),
            p95_mean_p_tail_profile_spread=("mean_p_tail_profile_spread", lambda s: float(s.quantile(0.95))),
        )
        .reset_index()
    )
    return scenario_zone, pivot, spread_zone


def make_plots(case_zone: pd.DataFrame, scenario_zone: pd.DataFrame, spread_case_zone: pd.DataFrame, spread_zone: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    profile_labels = [
        f"{row.scenario}\n{row.watts_person:.0f} W"
        for row in case_zone[["scenario", "watts_person"]]
        .drop_duplicates()
        .set_index("scenario")
        .loc[PROFILE_ORDER]
        .reset_index()
        .itertuples()
    ]

    heat = scenario_zone.pivot_table(
        index="scenario",
        columns="zone",
        values="mean_high_tail_pct",
        aggfunc="first",
        observed=True,
    ).loc[PROFILE_ORDER, ZONE_ORDER]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 9.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.95, 1.35]},
    )
    vmax = max(40.0, float(np.ceil(heat.to_numpy().max() / 5.0) * 5.0))
    im = axes[0].imshow(heat.to_numpy(float), cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    axes[0].set_yticks(range(len(PROFILE_ORDER)))
    axes[0].set_yticklabels(profile_labels)
    axes[0].set_xticks(range(len(ZONE_ORDER)))
    axes[0].set_xticklabels([ZONE_LABELS[zone] for zone in ZONE_ORDER], rotation=55, ha="right")
    axes[0].set_title("Mean high-tail exposure by metabolic profile and zone", fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = float(heat.iloc[i, j])
            axes[0].text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > vmax * 0.55 else "#222222",
            )
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.015)
    cbar.set_label("High-tail exposure (%)")

    spread_quantiles = []
    for zone in ZONE_ORDER:
        values = spread_case_zone.loc[
            spread_case_zone["zone"] == zone, "high_tail_pct_profile_spread"
        ].to_numpy(float)
        spread_quantiles.append(
            {
                "zone": zone,
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "mean": float(np.mean(values)),
            }
        )
    spread_df = pd.DataFrame(spread_quantiles)
    y = np.arange(len(ZONE_ORDER))
    axes[1].hlines(y, spread_df["q05"], spread_df["q95"], color="#bdbdbd", linewidth=1.2, label="5th-95th percentile")
    axes[1].hlines(y, spread_df["q25"], spread_df["q75"], color="#525252", linewidth=5.0, alpha=0.88, label="Interquartile range")
    axes[1].plot(spread_df["median"], y, "o", color="#111111", ms=4.2, label="Median")
    axes[1].plot(
        spread_df["mean"],
        y,
        marker="D",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="#111111",
        markeredgewidth=1.0,
        ms=4.2,
        label="Mean",
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([ZONE_LABELS[zone] for zone in ZONE_ORDER])
    axes[1].invert_yaxis()
    spread_max = float(np.ceil(spread_df["q95"].max() / 5.0) * 5.0)
    axes[1].set_xlim(0.0, spread_max)
    axes[1].set_xlabel("Spread across metabolic profiles (percentage points)")
    axes[1].set_title("Within-zone profile spread summarized over weather cases", fontweight="bold")
    axes[1].grid(axis="x", color="#d9d9d9", linewidth=0.6)
    axes[1].legend(frameon=False, ncol=4, loc="lower right", fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.suptitle("Zone-resolved metabolic sensitivity of tail-risk exposure", fontweight="bold")
    png = out_dir / "zone_metabolic_profile_summary.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    paths.append(pdf)

    heat = scenario_zone.pivot_table(
        index="scenario",
        columns="zone",
        values="mean_high_tail_pct",
        aggfunc="first",
        observed=True,
    ).loc[PROFILE_ORDER, ZONE_ORDER]
    fig, ax = plt.subplots(figsize=(12.6, 4.8), constrained_layout=True)
    vmax = max(50.0, float(np.ceil(heat.to_numpy().max() / 5.0) * 5.0))
    im = ax.imshow(heat.to_numpy(float), cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(PROFILE_ORDER)))
    ax.set_yticklabels(profile_labels)
    ax.set_xticks(range(len(ZONE_ORDER)))
    ax.set_xticklabels([ZONE_LABELS[zone] for zone in ZONE_ORDER], rotation=55, ha="right")
    ax.set_title("Mean high-tail exposure by metabolic profile and zone", fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = float(heat.iloc[i, j])
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > vmax * 0.55 else "#222222",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("High-tail exposure (%)")
    png = out_dir / "zone_metabolic_profile_zone_heatmap.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    paths.append(pdf)
    return paths


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


def write_summary(case_zone: pd.DataFrame, scenario_zone: pd.DataFrame, spread_case_zone: pd.DataFrame, spread_zone: pd.DataFrame, plot_paths: list[Path], out_dir: Path) -> Path:
    profile = (
        case_zone.groupby(["scenario", "watts_person", "met"], sort=False, observed=True)
        .agg(
            case_zone_rows=("high_tail_pct", "size"),
            mean_case_zone_high_tail_pct=("high_tail_pct", "mean"),
            median_case_zone_high_tail_pct=("high_tail_pct", "median"),
            p75_case_zone_high_tail_pct=("high_tail_pct", lambda s: float(s.quantile(0.75))),
            p95_case_zone_high_tail_pct=("high_tail_pct", lambda s: float(s.quantile(0.95))),
        )
        .reset_index()
    )
    top_spread = spread_zone.sort_values(
        "mean_high_tail_pct_profile_spread", ascending=False
    ).head(8)
    sreal = scenario_zone[scenario_zone["scenario"] == "S-real"].sort_values(
        "mean_high_tail_pct", ascending=False
    ).head(8)
    all_spread = spread_case_zone["high_tail_pct_profile_spread"]
    sreal_gap = spread_case_zone["sreal_high_tail_gap_to_max_profile"]
    path = out_dir / "zone_metabolic_profile_distribution_summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Zone Metabolic Profile Distribution\n\n")
        f.write("This diagnostic summarizes case-zone tail exposure across the six metabolic profiles.\n\n")
        f.write("## Headline\n\n")
        f.write(
            "- Case-zone high-tail exposure spread across profiles has mean "
            f"{all_spread.mean():.2f} percentage points and p95 {all_spread.quantile(0.95):.2f} percentage points.\n"
        )
        f.write(
            "- The gap between S-real and the highest profile has mean "
            f"{sreal_gap.mean():.2f} percentage points and p95 {sreal_gap.quantile(0.95):.2f} percentage points.\n"
        )
        f.write(
            "- Largest mean profile spreads occur in "
            + ", ".join(f"{row.zone_label} ({row.mean_high_tail_pct_profile_spread:.1f} pp)" for row in top_spread.head(4).itertuples())
            + ".\n\n"
        )
        f.write("## Profile Distribution\n\n")
        f.write(markdown_table(profile))
        f.write("\n\n## Highest S-real Zone Exposures\n\n")
        f.write(
            markdown_table(
                sreal[
                    [
                        "zone_label",
                        "floor",
                        "position_name",
                        "mean_high_tail_pct",
                        "p95_case_high_tail_pct",
                        "mean_p_tail",
                    ]
                ]
            )
        )
        f.write("\n\n## Largest Zone Profile Spreads\n\n")
        f.write(
            markdown_table(
                top_spread[
                    [
                        "zone_label",
                        "floor",
                        "position_name",
                        "mean_high_tail_pct_profile_spread",
                        "p95_high_tail_pct_profile_spread",
                        "mean_sreal_high_tail_gap_to_max_profile",
                    ]
                ]
            )
        )
        f.write("\n\n## Figures\n\n")
        for plot in plot_paths:
            f.write(f"- `{plot.name}`\n")
    return path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = mean_met.load_bundle(args.model_path)
    scenarios = mean_met.build_met_scenarios(args.bsa_mode)
    paths = trace_paths(args.trace_dir, args.max_cases)
    print(f"[load] model: {args.model_path}", flush=True)
    print(f"[load] traces: {len(paths)} cases from {args.trace_dir}", flush=True)

    rows: list[dict[str, object]] = []
    for i, path in enumerate(paths, start=1):
        rows.extend(process_trace(path, bundle, scenarios, args))
        if i == 1 or i % 12 == 0 or i == len(paths):
            print(f"[progress] processed {i}/{len(paths)} traces", flush=True)

    case_zone = pd.DataFrame(rows)
    if case_zone.empty:
        raise ValueError("No case-zone-profile rows were collected.")
    case_zone["scenario"] = pd.Categorical(case_zone["scenario"], PROFILE_ORDER, ordered=True)
    case_zone["zone"] = pd.Categorical(case_zone["zone"], ZONE_ORDER, ordered=True)
    case_zone = case_zone.sort_values(["weather", "scenario", "zone"])
    scenario_zone, spread_case_zone, spread_zone = summarize(case_zone)

    case_zone.to_csv(args.output_dir / "zone_metabolic_profile_case_zone_summary.csv", index=False)
    scenario_zone.to_csv(args.output_dir / "zone_metabolic_profile_zone_summary.csv", index=False)
    spread_case_zone.to_csv(args.output_dir / "zone_metabolic_profile_case_zone_spread.csv", index=False)
    spread_zone.to_csv(args.output_dir / "zone_metabolic_profile_zone_spread_summary.csv", index=False)
    plot_paths = make_plots(case_zone, scenario_zone, spread_case_zone, spread_zone, args.output_dir)
    summary = write_summary(case_zone, scenario_zone, spread_case_zone, spread_zone, plot_paths, args.output_dir)

    print(f"[write] {args.output_dir / 'zone_metabolic_profile_case_zone_summary.csv'}")
    print(f"[write] {args.output_dir / 'zone_metabolic_profile_zone_summary.csv'}")
    print(f"[write] {args.output_dir / 'zone_metabolic_profile_case_zone_spread.csv'}")
    print(f"[write] {args.output_dir / 'zone_metabolic_profile_zone_spread_summary.csv'}")
    for path in plot_paths:
        print(f"[write] {path}")
    print(f"[write] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Corrected same-state panel uncertainty for the endpoint-only TSV event.

Input must be the validated 144-case ``endpoint_case_summary.csv`` emitted by
``run_endpoint_nominal_robustness.py``. The event is
``P(TSV=-3) + P(TSV=+3)``. The analysis preserves selector-role pairing and
separately gives each of the 119 distinct source-year trajectories equal weight.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

import panel_cluster_uncertainty as panel_boot

warnings.simplefilter("ignore", PerformanceWarning)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ANALYSIS_ROOT
    / "outputs/robustness_endpoint_model/endpoint_case_summary.csv"
)
DEFAULT_OUT = (
    ANALYSIS_ROOT
    / "outputs/uncertainty_panel_endpoint_corrected_same_state"
)
SCREENS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
KEYS = [
    "weather",
    "city",
    "scenario_raw",
    "time_slice",
    "severity",
    "weather_year",
]
SOURCE_KEY = ["city", "scenario_raw", "time_slice", "weather_year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-case-summary", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=panel_boot.N_BOOT)
    parser.add_argument("--seed", type=int, default=panel_boot.SEED)
    return parser.parse_args()


def token(screen: float) -> str:
    return str(screen).replace(".", "p")


def metric_columns(screen: float) -> dict[str, str]:
    suffix = f"_at_{token(screen)}"
    return {
        f"mean_zone_high_pct__tau_{token(screen)}": (
            "equal_zone_mean_high_pct" + suffix
        ),
        f"any_zone_high_pct__tau_{token(screen)}": (
            "any_zone_high_pct" + suffix
        ),
        f"hidden_any_zone_pct__tau_{token(screen)}": (
            "hidden_any_zone_pct" + suffix
        ),
        f"area_weighted_zone_time_high_pct__tau_{token(screen)}": (
            "area_weighted_zone_time_high_pct" + suffix
        ),
    }


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data = pd.read_csv(path)
    if "scenario_raw" not in data and "scenario" in data:
        data = data.rename(columns={"scenario": "scenario_raw"})
    required = {
        *KEYS,
        "equal_zone_mean_mean",
        "area_weighted_mean_mean",
    }
    all_mappings: dict[str, str] = {}
    for screen in SCREENS:
        all_mappings.update(metric_columns(screen))
    required.update(all_mappings.values())
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            "Endpoint case summary is incomplete: " + ", ".join(missing)
        )
    if len(data) != 144:
        raise ValueError(f"Expected 144 endpoint role rows, found {len(data)}")

    out = data[sorted(required)].copy()
    out = out.rename(
        columns={
            "equal_zone_mean_mean": "mean_endpoint_probability",
            "area_weighted_mean_mean": (
                "area_weighted_mean_endpoint_probability"
            ),
            **{source: target for target, source in all_mappings.items()},
        }
    )
    metrics = [
        "mean_endpoint_probability",
        "area_weighted_mean_endpoint_probability",
    ]
    for screen in SCREENS:
        names = list(metric_columns(screen))
        mean_name, any_name, _, _ = names
        gap_name = f"any_minus_mean_zone_pp__tau_{token(screen)}"
        out[gap_name] = out[any_name] - out[mean_name]
        metrics.extend([*names, gap_name])

    duplicate = out[out.duplicated(SOURCE_KEY, keep=False)]
    for metric in metrics:
        spread = duplicate.groupby(SOURCE_KEY, sort=False)[metric].agg(
            lambda values: values.max() - values.min()
        )
        if not np.allclose(spread.to_numpy(float), 0.0, atol=1e-10, rtol=0.0):
            raise ValueError(
                f"Repeated roles disagree for corrected endpoint metric {metric}"
            )
    unique = out.drop_duplicates(SOURCE_KEY).copy()
    if len(unique) != 119:
        raise ValueError(f"Expected 119 endpoint source years, found {len(unique)}")
    return out, unique, metrics


def fmt(row: pd.Series) -> str:
    return (
        f"{row['estimate']:.2f} "
        f"[{row['ci_lower_2_5']:.2f}, {row['ci_upper_97_5']:.2f}]"
    )


def write_summary(output_dir: Path, results: pd.DataFrame, n_boot: int, seed: int) -> None:
    role = results[
        (results["dataset"] == "endpoint_role_weighted_144")
        & (results["analysis"] == "overall")
    ].set_index("metric")
    unique = results[
        (results["dataset"] == "endpoint_unique_source_year_119")
        & (results["analysis"] == "overall")
    ].set_index("metric")
    time_contrasts = results[
        (results["dataset"] == "endpoint_role_weighted_144")
        & (results["analysis"] == "time_slice_contrast")
        & (results["contrast"] == "late_2080s minus baseline_2020s")
    ].set_index("metric")
    lines = [
        "# Corrected same-state endpoint panel uncertainty",
        "",
        "- Event: `P(TSV=-3) + P(TSV=+3)`.",
        "- Timing provenance: corrected same-state inference from each recorded row.",
        f"- Bootstrap draws: {n_boot:,}",
        f"- Seed: `{seed}`",
        (
            "- Role-weighted resampling: 48 city×scenario×time-slice blocks, "
            "with all three selector roles retained."
        ),
        (
            "- Unique-year sensitivity: 119 distinct source years receive equal "
            "weight and remain nested within the 48 parent design blocks."
        ),
        "",
        "## Spatial screen results",
        "",
        "| Endpoint probability screen | Mean-zone exceedance (%) | Any-zone exceedance (%) | Any-minus-mean gap (pp) | Hidden any-zone (%) | Gap lower bound > 0? |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for screen in SCREENS:
        t = token(screen)
        mean_row = role.loc[f"mean_zone_high_pct__tau_{t}"]
        any_row = role.loc[f"any_zone_high_pct__tau_{t}"]
        gap_row = role.loc[f"any_minus_mean_zone_pp__tau_{t}"]
        hidden_row = role.loc[f"hidden_any_zone_pct__tau_{t}"]
        lines.append(
            f"| {screen:.3f} | {fmt(mean_row)} | {fmt(any_row)} | "
            f"{fmt(gap_row)} | {fmt(hidden_row)} | "
            f"{'yes' if gap_row['ci_lower_2_5'] > 0 else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Equal-source-year check",
            "",
            "| Screen | Role-weighted gap (pp) | Equal-source-year gap (pp) |",
            "|---:|---:|---:|",
        ]
    )
    for screen in SCREENS:
        name = f"any_minus_mean_zone_pp__tau_{token(screen)}"
        lines.append(
            f"| {screen:.3f} | {fmt(role.loc[name])} | {fmt(unique.loc[name])} |"
        )
    lines.extend(
        [
            "",
            "## Late-2080s minus baseline contrast",
            "",
            "| Screen | Mean-zone contrast (pp) | Any-zone contrast (pp) |",
            "|---:|---:|---:|",
        ]
    )
    for screen in SCREENS:
        t = token(screen)
        mean_name = f"mean_zone_high_pct__tau_{t}"
        any_name = f"any_zone_high_pct__tau_{t}"
        lines.append(
            f"| {screen:.3f} | {fmt(time_contrasts.loc[mean_name])} | "
            f"{fmt(time_contrasts.loc[any_name])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "This is a support-limited bounding sensitivity. The conditional "
                "block-bootstrap intervals quantify variation across the selected "
                "case design; they do not turn endpoint probability into observed "
                "dissatisfaction or quantify climate-model, building, or occupant "
                "structural uncertainty."
            ),
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    role, unique, metrics = load_data(args.endpoint_case_summary)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, object]] = []
    draw_frames: list[pd.DataFrame] = []
    result, draws = panel_boot.block_mean_bootstrap(
        role,
        metrics,
        ["city", "scenario_raw", "time_slice"],
        rng,
        args.n_bootstrap,
        "endpoint_role_weighted_144",
    )
    rows.extend(result)
    draw_frames.append(draws)
    result, draws = panel_boot.variable_size_block_bootstrap(
        unique,
        metrics,
        ["city", "scenario_raw", "time_slice"],
        rng,
        args.n_bootstrap,
        "endpoint_unique_source_year_119",
    )
    rows.extend(result)
    draw_frames.append(draws)

    # Paired future-slice contrasts for the spatial metrics at each screen.
    for dataset, frame in [
        ("endpoint_role_weighted_144", role),
        ("endpoint_unique_source_year_119", unique),
    ]:
        time_metrics = []
        for screen in SCREENS:
            time_metrics.extend(metric_columns(screen))
            time_metrics.append(
                f"any_minus_mean_zone_pp__tau_{token(screen)}"
            )
        result, draws = panel_boot.paired_factor_bootstrap(
            frame,
            time_metrics,
            ["city", "scenario_raw"],
            "time_slice",
            panel_boot.TIME_LEVELS,
            rng,
            args.n_bootstrap,
            dataset,
        )
        rows.extend(result)
        draw_frames.append(draws)

    results = pd.DataFrame(rows)
    results.to_csv(
        args.output_dir / "endpoint_panel_clustered_intervals.csv",
        index=False,
    )
    all_draws = draw_frames[0]
    for frame in draw_frames[1:]:
        all_draws = all_draws.merge(frame, on="draw", validate="one_to_one")
    all_draws.to_csv(
        args.output_dir / "endpoint_panel_bootstrap_draws.csv.gz",
        index=False,
        compression="gzip",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "interval": "percentile_2.5_97.5",
                "timing_provenance": "corrected_same_state",
                "endpoint_case_summary": str(args.endpoint_case_summary),
                "event_classes": [-3, 3],
                "screens": SCREENS,
                "role_rows": len(role),
                "unique_source_years": len(unique),
                "global_resampling_blocks": 48,
                "paired_time_slice_blocks": 12,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_summary(
        args.output_dir,
        results,
        args.n_bootstrap,
        args.seed,
    )
    panel_boot.write_output_checksums(args.output_dir)
    print(args.output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

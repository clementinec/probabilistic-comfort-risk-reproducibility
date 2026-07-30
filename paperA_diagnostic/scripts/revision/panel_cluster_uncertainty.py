#!/usr/bin/env python3
"""Clustered uncertainty audit for Paper A's case-level panel summaries.

The script deliberately starts from one row per selection-role case. It never
resamples 15-minute records as independent observations.

Primary role-weighted intervals preserve the three weather-selector roles within
each city×scenario×time-slice block. Paired time-slice, scenario, and severity
contrasts use larger design blocks so that the compared levels travel together.
A separate sensitivity gives each of the 119 distinct source-year trajectories
one row and one vote while retaining those years within their 48 parent design
blocks during resampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ANALYSIS_ROOT.parents[1]
DEFAULT_MANIFEST = WORKSPACE_ROOT / "paperA_rebuild/data/panel_manifest.csv"
DEFAULT_MEAN = (
    WORKSPACE_ROOT
    / "paperA_rebuild/diagnostics/mean_tail_compact/mean_tail_case_summary.csv"
)
DEFAULT_ZONE = (
    WORKSPACE_ROOT
    / "paperA_rebuild/diagnostics/zone_size_mosaic/zone_size_case_summary.csv"
)
SEED = 20260729
N_BOOT = 10_000
SOURCE_KEY = ["city", "scenario_raw", "time_slice", "weather_year"]
JOIN_KEY = ["city", "scenario_raw", "time_slice", "severity", "weather_year"]
TIME_LEVELS = ["baseline_2020s", "near_2030s", "mid_2050s", "late_2080s"]
SCENARIO_LEVELS = ["ssp245", "ssp585"]
SEVERITY_LEVELS = ["typical", "hot", "heatwave_extreme"]

OVERALL_METRICS = [
    "mean_p_tail",
    "high_tail_pct",
    "mean_zone_high_tail_pct",
    "any_zone_high_tail_pct",
    "hidden_any_zone_high_tail_pct",
    "area_weighted_zone_time_high_tail_pct",
    "area_weighted_mean_p_tail",
    "any_minus_mean_zone_pct_points",
    "area_weighted_minus_mean_zone_pct_points",
]
FACTOR_METRICS = [
    "mean_p_tail",
    "high_tail_pct",
    "any_zone_high_tail_pct",
    "hidden_any_zone_high_tail_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mean-summary", type=Path, default=DEFAULT_MEAN)
    parser.add_argument("--zone-summary", type=Path, default=DEFAULT_ZONE)
    parser.add_argument(
        "--corrected-headline-summary",
        type=Path,
        help=(
            "Corrected same-state 144-case summary from "
            "run_endpoint_nominal_robustness.py. When supplied, it replaces "
            "the legacy mean/zone summaries."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timing-provenance",
        required=True,
        choices=[
            "legacy_begin_probability_end_environment",
            "corrected_same_state",
        ],
        help=(
            "Required trace-timing label. Output directories must be kept "
            "separate across provenance states."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_case_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.corrected_headline_summary is not None:
        if args.timing_provenance != "corrected_same_state":
            raise ValueError(
                "--corrected-headline-summary requires "
                "--timing-provenance corrected_same_state"
            )
        corrected = pd.read_csv(args.corrected_headline_summary)
        if "scenario_raw" not in corrected and "scenario" in corrected:
            corrected = corrected.rename(columns={"scenario": "scenario_raw"})
        source_columns = {
            "corrected_equal_zone_mean_mean": "mean_p_tail",
            "corrected_equal_zone_mean_high_pct": "high_tail_pct",
            "corrected_any_zone_high_pct": "any_zone_high_tail_pct",
            "corrected_hidden_any_zone_pct": "hidden_any_zone_high_tail_pct",
            "corrected_area_weighted_zone_time_high_pct": (
                "area_weighted_zone_time_high_tail_pct"
            ),
            "corrected_area_weighted_mean_mean": "area_weighted_mean_p_tail",
        }
        required = [*JOIN_KEY, *source_columns]
        missing = sorted(set(required).difference(corrected.columns))
        if missing:
            raise ValueError(
                "Corrected headline summary is missing columns: "
                + ", ".join(missing)
            )
        data = corrected[required].rename(columns=source_columns).copy()
        data["mean_zone_high_tail_pct"] = data["high_tail_pct"]
    else:
        if args.timing_provenance == "corrected_same_state":
            raise ValueError(
                "Corrected provenance requires --corrected-headline-summary; "
                "legacy summaries cannot be relabelled as corrected."
            )
        manifest = pd.read_csv(args.manifest)
        mean = pd.read_csv(args.mean_summary)
        zone = pd.read_csv(args.zone_summary)
        data = manifest.merge(
            mean[
                JOIN_KEY
                + [
                    "mean_p_tail",
                    "high_tail_pct",
                ]
            ],
            on=JOIN_KEY,
            validate="one_to_one",
        )
        data = data.merge(
            zone[
                JOIN_KEY
                + [
                    "mean_zone_high_tail_pct",
                    "any_zone_high_tail_pct",
                    "hidden_any_zone_high_tail_pct",
                    "area_weighted_zone_time_high_tail_pct",
                    "area_weighted_mean_p_tail",
                ]
            ],
            on=JOIN_KEY,
            validate="one_to_one",
        )
    if len(data) != 144:
        raise ValueError(f"Expected 144 role-labelled cases, found {len(data)}")

    data["source_year_id"] = data[SOURCE_KEY].astype(str).agg("|".join, axis=1)
    data["any_minus_mean_zone_pct_points"] = (
        data["any_zone_high_tail_pct"] - data["mean_zone_high_tail_pct"]
    )
    data["area_weighted_minus_mean_zone_pct_points"] = (
        data["area_weighted_zone_time_high_tail_pct"]
        - data["mean_zone_high_tail_pct"]
    )

    duplicate = data[data.duplicated(SOURCE_KEY, keep=False)]
    for metric in OVERALL_METRICS:
        spread = duplicate.groupby(SOURCE_KEY, sort=False)[metric].agg(
            lambda values: values.max() - values.min()
        )
        if not np.allclose(spread.to_numpy(float), 0.0, atol=1e-10, rtol=0.0):
            raise ValueError(
                f"Repeated selector roles disagree for {metric}:\n"
                f"{spread[spread.abs() > 1e-10].head()}"
            )

    unique = data.drop_duplicates(SOURCE_KEY).copy()
    if len(unique) != 119:
        raise ValueError(f"Expected 119 unique source years, found {len(unique)}")
    return data, unique


def summarize_draws(
    draws: np.ndarray,
    estimate: float,
    *,
    dataset: str,
    analysis: str,
    metric: str,
    resampling_unit: str,
    n_units: int,
    level: str = "",
    contrast: str = "",
) -> dict[str, object]:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        lower = upper = boot_mean = boot_sd = np.nan
    else:
        lower, upper = np.quantile(finite, [0.025, 0.975])
        boot_mean = finite.mean()
        boot_sd = finite.std(ddof=1)
    return {
        "dataset": dataset,
        "analysis": analysis,
        "metric": metric,
        "level": level,
        "contrast": contrast,
        "estimate": float(estimate),
        "ci_lower_2_5": float(lower),
        "ci_upper_97_5": float(upper),
        "bootstrap_mean": float(boot_mean),
        "bootstrap_sd": float(boot_sd),
        "n_bootstrap_valid": int(len(finite)),
        "resampling_unit": resampling_unit,
        "n_resampling_units": int(n_units),
    }


def block_mean_bootstrap(
    data: pd.DataFrame,
    metrics: Iterable[str],
    block_columns: list[str],
    rng: np.random.Generator,
    n_boot: int,
    dataset: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    metrics = list(metrics)
    blocks = data.groupby(block_columns, sort=True)[metrics].mean().reset_index()
    n_units = len(blocks)
    sampled = rng.integers(0, n_units, size=(n_boot, n_units))
    rows: list[dict[str, object]] = []
    draw_frame = pd.DataFrame({"draw": np.arange(n_boot, dtype=int)})
    for metric in metrics:
        values = blocks[metric].to_numpy(float)
        draws = values[sampled].mean(axis=1)
        rows.append(
            summarize_draws(
                draws,
                data[metric].mean(),
                dataset=dataset,
                analysis="overall",
                metric=metric,
                resampling_unit=" × ".join(block_columns),
                n_units=n_units,
            )
        )
        draw_frame[f"{dataset}__overall__{metric}"] = draws
    return rows, draw_frame


def variable_size_block_bootstrap(
    data: pd.DataFrame,
    metrics: Iterable[str],
    block_columns: list[str],
    rng: np.random.Generator,
    n_boot: int,
    dataset: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    metrics = list(metrics)
    grouped = data.groupby(block_columns, sort=True)
    block_counts = grouped.size().to_numpy(float)
    block_sums = grouped[metrics].sum().reset_index(drop=True)
    n_units = len(block_sums)
    sampled = rng.integers(0, n_units, size=(n_boot, n_units))
    sampled_denominator = block_counts[sampled].sum(axis=1)
    rows: list[dict[str, object]] = []
    draw_frame = pd.DataFrame({"draw": np.arange(n_boot, dtype=int)})
    for metric in metrics:
        values = block_sums[metric].to_numpy(float)
        draws = values[sampled].sum(axis=1) / sampled_denominator
        rows.append(
            summarize_draws(
                draws,
                data[metric].mean(),
                dataset=dataset,
                analysis="overall",
                metric=metric,
                resampling_unit=(
                    " × ".join(block_columns)
                    + " block containing distinct source-year trajectories"
                ),
                n_units=n_units,
            )
        )
        draw_frame[f"{dataset}__overall__{metric}"] = draws
    return rows, draw_frame


def paired_factor_bootstrap(
    data: pd.DataFrame,
    metrics: Iterable[str],
    unit_columns: list[str],
    factor_column: str,
    levels: list[str],
    rng: np.random.Generator,
    n_boot: int,
    dataset: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Bootstrap paired design units while retaining every row in a sampled unit.

    Sums and counts are stored by unit×factor level. This permits unequal counts
    in the unique-source-year sensitivity while retaining its equal-year target.
    """

    metrics = list(metrics)
    group_columns = unit_columns + [factor_column]
    grouped = data.groupby(group_columns, sort=True)
    counts = grouped.size().rename("n").reset_index()
    sums = grouped[metrics].sum().reset_index()
    aggregate = counts.merge(sums, on=group_columns, validate="one_to_one")

    units = aggregate[unit_columns].drop_duplicates().reset_index(drop=True)
    units["unit_index"] = np.arange(len(units), dtype=int)
    aggregate = aggregate.merge(units, on=unit_columns, validate="many_to_one")
    n_units = len(units)

    expected = pd.MultiIndex.from_product(
        [range(n_units), levels], names=["unit_index", factor_column]
    )
    observed = pd.MultiIndex.from_frame(aggregate[["unit_index", factor_column]])
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(
            f"Missing paired {factor_column} levels for {dataset}: "
            f"{list(missing[:5])}"
        )

    count_matrix = (
        aggregate.pivot(index="unit_index", columns=factor_column, values="n")
        .reindex(index=range(n_units), columns=levels)
        .to_numpy(float)
    )
    sampled = rng.integers(0, n_units, size=(n_boot, n_units))
    sampled_counts = count_matrix[sampled].sum(axis=1)

    rows: list[dict[str, object]] = []
    draw_frame = pd.DataFrame({"draw": np.arange(n_boot, dtype=int)})
    point_by_level = data.groupby(factor_column, sort=False)[metrics].mean()

    for metric in metrics:
        sum_matrix = (
            aggregate.pivot(index="unit_index", columns=factor_column, values=metric)
            .reindex(index=range(n_units), columns=levels)
            .to_numpy(float)
        )
        level_draws = sum_matrix[sampled].sum(axis=1) / sampled_counts
        for level_index, level in enumerate(levels):
            values = level_draws[:, level_index]
            rows.append(
                summarize_draws(
                    values,
                    point_by_level.loc[level, metric],
                    dataset=dataset,
                    analysis=f"{factor_column}_level",
                    metric=metric,
                    level=level,
                    resampling_unit=" × ".join(unit_columns),
                    n_units=n_units,
                )
            )
            draw_frame[
                f"{dataset}__{factor_column}__{metric}__{level}"
            ] = values

        reference = levels[0]
        reference_values = level_draws[:, 0]
        for level_index, level in enumerate(levels[1:], start=1):
            contrast_draws = level_draws[:, level_index] - reference_values
            estimate = (
                point_by_level.loc[level, metric]
                - point_by_level.loc[reference, metric]
            )
            rows.append(
                summarize_draws(
                    contrast_draws,
                    estimate,
                    dataset=dataset,
                    analysis=f"{factor_column}_contrast",
                    metric=metric,
                    contrast=f"{level} minus {reference}",
                    resampling_unit=" × ".join(unit_columns),
                    n_units=n_units,
                )
            )
            draw_frame[
                f"{dataset}__{factor_column}_contrast__{metric}__"
                f"{level}_minus_{reference}"
            ] = contrast_draws

        if factor_column == "severity":
            hot_index = levels.index("hot")
            extreme_index = levels.index("heatwave_extreme")
            contrast_draws = (
                level_draws[:, extreme_index] - level_draws[:, hot_index]
            )
            estimate = (
                point_by_level.loc["heatwave_extreme", metric]
                - point_by_level.loc["hot", metric]
            )
            rows.append(
                summarize_draws(
                    contrast_draws,
                    estimate,
                    dataset=dataset,
                    analysis="severity_contrast",
                    metric=metric,
                    contrast="heatwave_extreme minus hot",
                    resampling_unit=" × ".join(unit_columns),
                    n_units=n_units,
                )
            )
            draw_frame[
                f"{dataset}__severity_contrast__{metric}__"
                "heatwave_extreme_minus_hot"
            ] = contrast_draws

    return rows, draw_frame


def fmt_interval(row: pd.Series, digits: int = 2) -> str:
    return (
        f"{row['estimate']:.{digits}f} "
        f"[{row['ci_lower_2_5']:.{digits}f}, "
        f"{row['ci_upper_97_5']:.{digits}f}]"
    )


def write_summary(
    out_dir: Path,
    results: pd.DataFrame,
    n_boot: int,
    seed: int,
    timing_provenance: str,
) -> None:
    role_overall = results[
        (results["dataset"] == "role_weighted_144")
        & (results["analysis"] == "overall")
    ].set_index("metric")
    unique_overall = results[
        (results["dataset"] == "unique_source_year_119")
        & (results["analysis"] == "overall")
    ].set_index("metric")
    role_time = results[
        (results["dataset"] == "role_weighted_144")
        & (results["analysis"] == "time_slice_contrast")
        & (results["metric"] == "high_tail_pct")
    ].set_index("contrast")
    role_scenario = results[
        (results["dataset"] == "role_weighted_144")
        & (results["analysis"] == "scenario_raw_contrast")
        & (results["metric"] == "high_tail_pct")
    ].set_index("contrast")

    any_gap = role_overall.loc["any_minus_mean_zone_pct_points"]
    hidden = role_overall.loc["hidden_any_zone_high_tail_pct"]
    late = role_time.loc["late_2080s minus baseline_2020s"]
    mid = role_time.loc["mid_2050s minus baseline_2020s"]
    near = role_time.loc["near_2030s minus baseline_2020s"]
    ssp = role_scenario.loc["ssp585 minus ssp245"]

    lines = [
        "# Clustered panel uncertainty audit",
        "",
        f"- Bootstrap draws: {n_boot:,}",
        f"- Seed: `{seed}`",
        f"- Timing provenance: `{timing_provenance}`",
        "- Intervals: empirical 2.5th and 97.5th percentiles.",
        (
            "- Primary global/spatial resampling unit: 48 "
            "city×scenario×time-slice blocks; all three selector roles travel "
            "together."
        ),
        (
            "- Paired time-slice resampling unit: 12 city×scenario blocks; all "
            "four slices and selector roles travel together."
        ),
        (
            "- Separate sensitivity: 119 distinct source-year trajectories with "
            "equal source-year weight, retained within 48 parent design blocks "
            "during resampling."
        ),
        "",
        "## Headline role-weighted results",
        "",
        (
            "- Mean-zone high-tail exceedance (%): "
            f"{fmt_interval(role_overall.loc['mean_zone_high_tail_pct'])}."
        ),
        (
            "- Any-zone high-tail exceedance (%): "
            f"{fmt_interval(role_overall.loc['any_zone_high_tail_pct'])}."
        ),
        (
            "- Any-zone minus mean-zone gap (percentage points): "
            f"{fmt_interval(any_gap)}."
        ),
        (
            "- Hidden any-zone exceedance (%): "
            f"{fmt_interval(hidden)}."
        ),
        "",
        "## Paired contrasts in high-tail exceedance",
        "",
        (
            "- Near-2030s minus baseline (percentage points): "
            f"{fmt_interval(near)}."
        ),
        (
            "- Mid-2050s minus baseline (percentage points): "
            f"{fmt_interval(mid)}."
        ),
        (
            "- Late-2080s minus baseline (percentage points): "
            f"{fmt_interval(late)}."
        ),
        (
            "- SSP5-8.5 minus SSP2-4.5 (percentage points): "
            f"{fmt_interval(ssp)}. The scenario interval resamples only six "
            "paired city blocks and must not be read as regional climate "
            "uncertainty."
        ),
        "",
        "## Equal-source-year sensitivity",
        "",
        (
            "- Mean-zone high-tail exceedance (%): "
            f"{fmt_interval(unique_overall.loc['mean_zone_high_tail_pct'])}."
        ),
        (
            "- Any-zone high-tail exceedance (%): "
            f"{fmt_interval(unique_overall.loc['any_zone_high_tail_pct'])}."
        ),
        (
            "- Hidden any-zone exceedance (%): "
            f"{fmt_interval(unique_overall.loc['hidden_any_zone_high_tail_pct'])}."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "These intervals quantify sensitivity to which existing case blocks "
            "or source years receive weight. They do not include climate-model "
            "ensemble spread, weather-file construction error, building-model "
            "structural error, predictor transport error, or future occupant "
            "adaptation."
        ),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_output_checksums(output_dir: Path) -> None:
    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "output_checksums.csv":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    pd.DataFrame(records).to_csv(
        output_dir / "output_checksums.csv",
        index=False,
    )


def main() -> int:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    role, unique = load_case_data(args)
    rng = np.random.default_rng(args.seed)

    result_rows: list[dict[str, object]] = []
    draw_frames: list[pd.DataFrame] = []

    rows, draws = block_mean_bootstrap(
        role,
        OVERALL_METRICS,
        ["city", "scenario_raw", "time_slice"],
        rng,
        args.n_bootstrap,
        "role_weighted_144",
    )
    result_rows.extend(rows)
    draw_frames.append(draws)

    rows, draws = variable_size_block_bootstrap(
        unique,
        OVERALL_METRICS,
        ["city", "scenario_raw", "time_slice"],
        rng,
        args.n_bootstrap,
        "unique_source_year_119",
    )
    result_rows.extend(rows)
    draw_frames.append(draws)

    for dataset, frame in [
        ("role_weighted_144", role),
        ("unique_source_year_119", unique),
    ]:
        rows, draws = paired_factor_bootstrap(
            frame,
            FACTOR_METRICS,
            ["city", "scenario_raw"],
            "time_slice",
            TIME_LEVELS,
            rng,
            args.n_bootstrap,
            dataset,
        )
        result_rows.extend(rows)
        draw_frames.append(draws)

        rows, draws = paired_factor_bootstrap(
            frame,
            FACTOR_METRICS,
            ["city"],
            "scenario_raw",
            SCENARIO_LEVELS,
            rng,
            args.n_bootstrap,
            dataset,
        )
        result_rows.extend(rows)
        draw_frames.append(draws)

    rows, draws = paired_factor_bootstrap(
        role,
        FACTOR_METRICS,
        ["city", "scenario_raw", "time_slice"],
        "severity",
        SEVERITY_LEVELS,
        rng,
        args.n_bootstrap,
        "role_weighted_144",
    )
    result_rows.extend(rows)
    draw_frames.append(draws)

    results = pd.DataFrame(result_rows)
    results.to_csv(args.output_dir / "clustered_interval_summary.csv", index=False)

    all_draws = draw_frames[0]
    for frame in draw_frames[1:]:
        all_draws = all_draws.merge(frame, on="draw", validate="one_to_one")
    all_draws.to_csv(
        args.output_dir / "clustered_bootstrap_draws.csv.gz",
        index=False,
        compression="gzip",
    )

    inventory = {
        "role_labelled_cases": int(len(role)),
        "role_global_blocks": int(
            role[["city", "scenario_raw", "time_slice"]].drop_duplicates().shape[0]
        ),
        "paired_time_slice_blocks": int(
            role[["city", "scenario_raw"]].drop_duplicates().shape[0]
        ),
        "paired_scenario_blocks": int(role[["city"]].drop_duplicates().shape[0]),
        "paired_severity_blocks": int(
            role[["city", "scenario_raw", "time_slice"]].drop_duplicates().shape[0]
        ),
        "unique_source_years": int(len(unique)),
        "overlapping_role_rows": int(len(role) - len(unique)),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "interval": "percentile_2.5_97.5",
                "timing_provenance": args.timing_provenance,
                "manifest": str(args.manifest),
                "mean_summary": str(args.mean_summary),
                "zone_summary": str(args.zone_summary),
                "corrected_headline_summary": (
                    str(args.corrected_headline_summary)
                    if args.corrected_headline_summary is not None
                    else None
                ),
                "inventory": inventory,
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
        args.timing_provenance,
    )
    write_output_checksums(args.output_dir)
    print(args.output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

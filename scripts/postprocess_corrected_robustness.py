#!/usr/bin/env python3
"""Build manuscript-facing artifacts from synchronized diagnostic probability arrays.

This script is deliberately downstream of ``run_endpoint_nominal_robustness.py``.
It does not fit a model, rerun EnergyPlus, modify source traces, or edit the
manuscript.  It consumes the immutable, same-state NPZ products and creates:

* corrected inputs and renderings for manuscript Figures 1--3;
* exact scalar-versus-tail summaries and threshold diagnostics;
* role-weighted and unique-environmental-state summaries;
* zone and floor probability summaries; and
* a legacy-to-corrected reported-value audit map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "core"
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_inputs"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TAIL_SCREEN = 0.20
NEAR_MEAN_EPS = 0.15
DIRECTION_EPS = 0.10
CATEGORY_LOW = 0.065
CATEGORY_HIGH = 0.35
PLOT_SAMPLE_ROWS = 300_000
RANDOM_SEED = 42
EXPECTED_SCHEMA = "paperA_corrected_same_state_v3"
EXPECTED_CASES = 144
EXPECTED_UNIQUE_STATES = 119
EXPECTED_STEPS_PER_CASE = 16_704

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

FLOOR_BY_ZONE = {
    "core_bottom": "bottom",
    "perimeter_bot_zn_1": "bottom",
    "perimeter_bot_zn_2": "bottom",
    "perimeter_bot_zn_3": "bottom",
    "perimeter_bot_zn_4": "bottom",
    "core_mid": "middle",
    "perimeter_mid_zn_1": "middle",
    "perimeter_mid_zn_2": "middle",
    "perimeter_mid_zn_3": "middle",
    "perimeter_mid_zn_4": "middle",
    "core_top": "top",
    "perimeter_top_zn_1": "top",
    "perimeter_top_zn_2": "top",
    "perimeter_top_zn_3": "top",
    "perimeter_top_zn_4": "top",
}

ZONE_LABEL = {
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

REQUIRED_NPZ_FIELDS = {
    "schema_version",
    "source_row_index",
    "month",
    "day",
    "hour",
    "current_time",
    "zone_names",
    "zone_pmv",
    "zone_expected_tsv",
    "zone_cold_tail",
    "zone_warm_tail",
    "zone_cold_endpoint",
    "zone_warm_endpoint",
    "zone_no_pmv_expected_tsv",
    "zone_no_pmv_cold_tail",
    "zone_no_pmv_warm_tail",
}

CITY_ORDER = ["Ahmedabad", "Beijing", "Guangzhou", "Houston", "Kolkata", "Phoenix"]
SCENARIO_ORDER = ["ssp245", "ssp585"]
TIME_ORDER = ["baseline_2020s", "near_2030s", "mid_2050s", "late_2080s"]
SEVERITY_ORDER = ["typical", "hot", "heatwave_extreme"]
ZONE_PLOT_ORDER = [
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
FLOOR_ORDER = ["bottom", "middle", "top"]
POSITION_ORDER = ["core", "P1", "P2", "P3", "P4"]
POSITION_NAMES = {
    "core": "Core",
    "P1": "P1 south",
    "P2": "P2 east",
    "P3": "P3 north",
    "P4": "P4 west",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rows", type=int, default=PLOT_SAMPLE_ROWS)
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).mean() * 100.0)


def tail_category(values: np.ndarray) -> np.ndarray:
    """Legacy three-level reporting category used by the zone audit."""
    arr = np.asarray(values, dtype=float)
    return np.where(arr >= CATEGORY_HIGH, 2, np.where(arr > CATEGORY_LOW, 1, 0))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def threshold_metrics(
    values: np.ndarray, high_tail: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = np.asarray(values) >= threshold
    high = np.asarray(high_tail, dtype=bool)
    n = len(pred)
    tp = int(np.sum(pred & high))
    tn = int(np.sum(~pred & ~high))
    fp = int(np.sum(pred & ~high))
    fn = int(np.sum(~pred & high))
    return {
        "threshold": float(threshold),
        "disagreement_pct": float((fp + fn) / n * 100.0),
        "false_negative_pct": float(fn / n * 100.0),
        "false_positive_pct": float(fp / n * 100.0),
        "sensitivity_pct": float(tp / max(tp + fn, 1) * 100.0),
        "specificity_pct": float(tn / max(tn + fp, 1) * 100.0),
    }


def best_threshold(values: np.ndarray, high_tail: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    thresholds = np.unique(np.quantile(finite, np.linspace(0.001, 0.999, 999)))
    best: dict[str, float] | None = None
    for threshold in thresholds:
        record = threshold_metrics(values, high_tail, float(threshold))
        if best is None or record["disagreement_pct"] < best["disagreement_pct"]:
            best = record
    assert best is not None
    return best


def scalar_record(
    model: str,
    scalar: str,
    values: np.ndarray,
    probability: np.ndarray,
    high_tail: np.ndarray,
    standard_threshold: float = 0.5,
) -> dict[str, object]:
    standard = threshold_metrics(values, high_tail, standard_threshold)
    best = best_threshold(values, high_tail)
    return {
        "model": model,
        "scalar": scalar,
        "rows": int(len(values)),
        "tail_threshold": TAIL_SCREEN,
        "pearson_r": float(pd.Series(values).corr(pd.Series(probability), method="pearson")),
        "spearman_r": float(pd.Series(values).corr(pd.Series(probability), method="spearman")),
        "standard_threshold": standard_threshold,
        **{f"standard_{key}": value for key, value in standard.items() if key != "threshold"},
        "best_threshold": best["threshold"],
        **{f"best_{key}": value for key, value in best.items() if key != "threshold"},
    }


def group_mean_tail(
    label: dict[str, object],
    mu: np.ndarray,
    tail: np.ndarray,
    warm: np.ndarray,
    cold: np.ndarray,
) -> dict[str, object]:
    d_tail = warm - cold
    abs_mu = np.abs(mu)
    high = tail >= TAIL_SCREEN
    near = abs_mu < NEAR_MEAN_EPS
    directional = np.abs(d_tail) >= DIRECTION_EPS
    mu_sign = np.sign(np.where(abs_mu >= NEAR_MEAN_EPS, mu, 0.0))
    d_sign = np.sign(np.where(np.abs(d_tail) >= DIRECTION_EPS, d_tail, 0.0))
    comparable = (mu_sign != 0) & (d_sign != 0)
    conflict = comparable & (mu_sign != d_sign)
    return {
        **label,
        "rows": int(len(mu)),
        "mean_mu": float(np.mean(mu)),
        "mean_abs_mu": float(np.mean(abs_mu)),
        "mean_p_tail": float(np.mean(tail)),
        "p95_p_tail": float(np.quantile(tail, 0.95)),
        "high_tail_pct": pct(high),
        "near_mean_pct": pct(near),
        "near_mean_high_tail_pct": pct(near & high),
        "near_mean_directional_tail_pct": pct(near & directional),
        "mean_warm_tail": float(np.mean(warm)),
        "mean_cold_tail": float(np.mean(cold)),
        "warm_tail_gt_cold_tail_pct": pct(warm > cold),
        "warm_tail_gt_cold_tail_among_high_pct": (
            pct((warm > cold)[high]) if high.any() else math.nan
        ),
        "warm_tail_minus_cold_tail_gt_0p10_pct": pct(d_tail > DIRECTION_EPS),
        "cold_tail_minus_warm_tail_gt_0p10_pct": pct(d_tail < -DIRECTION_EPS),
        "sign_conflict_pct_of_comparable": (
            float(conflict.sum() / comparable.sum() * 100.0)
            if comparable.any()
            else math.nan
        ),
        "sign_comparable_rows": int(comparable.sum()),
    }


def scalar_band_summary(
    pmv: np.ndarray, mu: np.ndarray, tail: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, values in [("abs_mean_pmv", np.abs(pmv)), ("abs_mu_tsv", np.abs(mu))]:
        for lower, upper in [(0.20, 0.30), (0.30, 0.40), (0.40, 0.60)]:
            mask = (tail >= lower) & (tail < upper)
            rows.append(
                {
                    "scalar": name,
                    "p_tail_lower": lower,
                    "p_tail_upper": upper,
                    "rows": int(mask.sum()),
                    "q05": float(np.quantile(values[mask], 0.05)) if mask.any() else math.nan,
                    "q50": float(np.quantile(values[mask], 0.50)) if mask.any() else math.nan,
                    "q95": float(np.quantile(values[mask], 0.95)) if mask.any() else math.nan,
                }
            )
    return pd.DataFrame(rows)


def build_mean_bins(mu: np.ndarray, tail: np.ndarray, width: float = 0.05) -> pd.DataFrame:
    work = pd.DataFrame({"mu": mu, "p_tail": tail})
    work["mu_bin_left"] = np.floor(work["mu"] / width) * width
    grouped = work.groupby("mu_bin_left", sort=True)["p_tail"]
    out = grouped.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack()
    out.columns = ["p_tail_q05", "p_tail_q25", "p_tail_median", "p_tail_q75", "p_tail_q95"]
    out["rows"] = grouped.size()
    out["high_tail_pct"] = grouped.apply(lambda x: pct(x.to_numpy() >= TAIL_SCREEN))
    out = out.reset_index()
    out["mu_bin_right"] = out["mu_bin_left"] + width
    out["mu_bin_center"] = out["mu_bin_left"] + width / 2.0
    out["p_tail_iqr"] = out["p_tail_q75"] - out["p_tail_q25"]
    out["p_tail_90pct_spread"] = out["p_tail_q95"] - out["p_tail_q05"]
    return out


def role_local_index(global_indices: np.ndarray, steps_per_role: int) -> tuple[np.ndarray, np.ndarray]:
    return global_indices // steps_per_role, global_indices % steps_per_role


def write_figures(sample: pd.DataFrame, bins: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    scatter = ax.scatter(
        sample["expected_tsv"],
        sample["p_tail"],
        c=sample["warm_tail"] - sample["cold_tail"],
        cmap="coolwarm",
        s=8,
        alpha=0.55,
        linewidths=0,
    )
    ax.axvline(0, color="#444444", lw=0.8)
    ax.axhline(TAIL_SCREEN, color="#7a2d42", lw=0.9, ls="--")
    ax.axvspan(-NEAR_MEAN_EPS, NEAR_MEAN_EPS, color="#d9d9d9", alpha=0.25, lw=0)
    ax.set_xlabel(r"Expected TSV, $\mu_{\mathrm{TSV}}=\sum_k k p_k$")
    ax.set_ylabel(r"TSV-tail probability, $p_{\mathrm{tail}}$")
    ax.set_title("Expected TSV versus TSV-tail probability", weight="bold")
    ax.grid(color="#dddddd", lw=0.6)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Directional tail dominance, $p_{\mathrm{warm}}-p_{\mathrm{cold}}$")
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_vs_tail_scatter_sample_corrected.png", bbox_inches="tight")
    fig.savefig(fig_dir / "mean_vs_tail_scatter_sample_corrected.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    x = bins["mu_bin_center"].to_numpy(float)
    ax.fill_between(x, bins["p_tail_q05"], bins["p_tail_q95"], color="#b8c7e6", alpha=0.38, label="5--95%")
    ax.fill_between(x, bins["p_tail_q25"], bins["p_tail_q75"], color="#4f6f9f", alpha=0.34, label="IQR")
    ax.plot(x, bins["p_tail_median"], color="#243f73", lw=1.8, label="Median")
    ax.axhline(TAIL_SCREEN, color="#7a2d42", lw=0.9, ls="--")
    ax.axvline(0, color="#444444", lw=0.75)
    ax.set_xlabel(r"Expected TSV bin center, $\mu_{\mathrm{TSV}}$")
    ax.set_ylabel(r"TSV-tail probability, $p_{\mathrm{tail}}$")
    ax.set_title("Within-bin spread of TSV-tail probability", weight="bold")
    ax.grid(color="#dddddd", lw=0.55)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "tail_spread_by_mean_bin_corrected.png", bbox_inches="tight")
    fig.savefig(fig_dir / "tail_spread_by_mean_bin_corrected.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=180, sharey=True)
    for ax, col, title, xlabel in [
        (axes[0], "abs_mu_tsv", "Expected-TSV scalar", r"$|\mu_{\mathrm{TSV}}|$"),
        (axes[1], "abs_pmv", "PMV scalar", r"$|\mathrm{PMV}|$"),
    ]:
        ax.hexbin(
            sample[col],
            sample["p_tail"],
            gridsize=58,
            mincnt=1,
            cmap="viridis",
            linewidths=0,
        )
        ax.axhline(TAIL_SCREEN, color="#7a2d42", lw=0.9, ls="--")
        ax.axvline(0.5, color="#666666", lw=0.8, ls=":")
        ax.set_xlim(0.0, 3.0)
        ax.set_xlabel(xlabel)
        ax.grid(color="#dddddd", lw=0.5)
        ax.set_title(title, weight="bold")
    axes[0].set_ylabel(r"$p_{\mathrm{tail}}$")
    fig.tight_layout()
    fig.savefig(fig_dir / "scalar_tail_comparison_hexbin_corrected.png", bbox_inches="tight")
    fig.savefig(fig_dir / "scalar_tail_comparison_hexbin_corrected.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), dpi=180, sharex=True, sharey=True)
    for ax, col, title, cmap in [
        (axes[0], "p_tail", "Ordinal model with PMV feature", "viridis"),
        (axes[1], "no_pmv_p_tail", "Ordinal model without PMV feature", "magma"),
    ]:
        hb = ax.hexbin(
            sample["abs_pmv"],
            sample[col],
            gridsize=58,
            mincnt=1,
            cmap=cmap,
            linewidths=0,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Sampled records per bin", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.axhline(TAIL_SCREEN, color="#7a2d42", lw=0.9, ls="--")
        ax.axvline(0.5, color="#666666", lw=0.8, ls=":")
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title, weight="bold", fontsize=10)
        ax.set_xlabel(r"$|\mathrm{PMV}|$")
        ax.grid(color="#dddddd", lw=0.5)
    axes[0].set_ylabel(r"$p_{\mathrm{tail}}$ from model with PMV")
    axes[1].set_ylabel(r"$p_{\mathrm{tail}}$ from model without PMV")
    fig.tight_layout()
    fig.savefig(fig_dir / "pmv_feature_robustness_hexbin_corrected.png", bbox_inches="tight")
    fig.savefig(fig_dir / "pmv_feature_robustness_hexbin_corrected.pdf", bbox_inches="tight")
    plt.close(fig)


def zone_position(zone: str) -> str:
    if zone.startswith("core_"):
        return "core"
    return f"P{zone.rsplit('_', 1)[-1]}"


def aggregate_zone_aggregation_cases(case_df: pd.DataFrame) -> pd.DataFrame:
    """Pool case-level zone-aggregation metrics with explicit timestep weights."""
    metric_columns = [
        "mean_p_tail",
        "p90_zone_p_tail",
        "max_zone_p_tail",
        "mean_zone_high_tail_pct",
        "p90_zone_high_tail_pct",
        "any_zone_high_tail_pct",
        "hidden_any_zone_pct",
        "max_zone_more_severe_category_pct",
        "p90_zone_more_severe_category_pct",
        "warm_tail_gt_cold_tail_pct",
    ]
    specs = [
        ("global", []),
        ("city", ["city"]),
        ("time_slice", ["time_slice"]),
        ("city_time_slice", ["city", "time_slice"]),
        ("scenario_time_slice", ["scenario", "time_slice"]),
    ]
    records: list[dict[str, object]] = []
    for scope, keys in specs:
        groups: Iterable[tuple[object, pd.DataFrame]]
        if keys:
            groups = case_df.groupby(keys, sort=True, dropna=False)
        else:
            groups = [("all", case_df)]
        for values, group in groups:
            if keys and not isinstance(values, tuple):
                values = (values,)
            weights = group["n_steps"].to_numpy(float)
            record: dict[str, object] = {
                "group_scope": scope,
                "n_steps": int(weights.sum()),
                "n_cases": int(len(group)),
            }
            if keys:
                record.update(dict(zip(keys, values)))
            for column in metric_columns:
                record[column] = float(
                    np.average(group[column].to_numpy(float), weights=weights)
                )
            records.append(record)
    return pd.DataFrame(records)


def time_label(value: str) -> str:
    return {
        "baseline_2020s": "2020s",
        "near_2030s": "2030s",
        "mid_2050s": "2050s",
        "late_2080s": "2080s",
    }[value]


def scenario_label(value: str) -> str:
    return {"ssp245": "S245", "ssp585": "S585"}[value]


def make_corrected_zone_mosaic(zone_case: pd.DataFrame, figure_dir: Path) -> None:
    widths = np.array([ZONE_AREAS_M2[zone] for zone in ZONE_PLOT_ORDER], dtype=float)
    x_edges = np.concatenate([[0.0], np.cumsum(widths)])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    vmax = min(
        100.0,
        max(
            45.0,
            float(np.ceil(zone_case["high_tail_pct"].quantile(0.995) / 5.0) * 5.0),
        ),
    )
    case_meta = (
        zone_case[
            ["weather", "city", "scenario", "time_slice", "severity", "weather_year"]
        ]
        .drop_duplicates()
        .copy()
    )
    case_meta["city"] = pd.Categorical(case_meta["city"], CITY_ORDER, ordered=True)
    case_meta["scenario"] = pd.Categorical(
        case_meta["scenario"], SCENARIO_ORDER, ordered=True
    )
    case_meta["time_slice"] = pd.Categorical(
        case_meta["time_slice"], TIME_ORDER, ordered=True
    )
    case_meta["severity"] = pd.Categorical(
        case_meta["severity"], SEVERITY_ORDER, ordered=True
    )
    case_meta = case_meta.sort_values(
        ["city", "scenario", "time_slice", "severity", "weather_year"]
    )
    zone_pivot = zone_case.pivot_table(
        index="weather", columns="zone", values="high_tail_pct", aggfunc="first"
    )

    fig, axes = plt.subplots(
        len(CITY_ORDER),
        1,
        figsize=(12.6, 12.8),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"hspace": 0.08},
    )
    mesh = None
    for ax, city in zip(axes, CITY_ORDER):
        city_cases = case_meta[case_meta["city"].astype(str).eq(city)]
        matrix = zone_pivot.loc[
            city_cases["weather"].astype(str), ZONE_PLOT_ORDER
        ].to_numpy(float)
        mesh = ax.pcolormesh(
            x_edges,
            np.arange(matrix.shape[0] + 1),
            matrix,
            cmap=plt.get_cmap("YlOrRd"),
            vmin=0,
            vmax=vmax,
            linewidth=0.15,
            edgecolors="#f2f2f2",
        )
        ax.invert_yaxis()
        ax.set_ylabel(city, rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(axis="y", length=0, labelsize=7)
        group_centers: list[float] = []
        group_labels: list[str] = []
        for scenario in SCENARIO_ORDER:
            for time_slice in TIME_ORDER:
                mask = (
                    city_cases["scenario"].astype(str).eq(scenario)
                    & city_cases["time_slice"].astype(str).eq(time_slice)
                )
                indices = np.flatnonzero(mask.to_numpy())
                if len(indices):
                    group_centers.append(float(indices.mean() + 0.5))
                    group_labels.append(
                        f"{scenario_label(scenario)} {time_label(time_slice)}"
                    )
        ax.set_yticks(group_centers)
        ax.set_yticklabels(group_labels)
        for y in range(3, matrix.shape[0], 3):
            ax.axhline(y, color="#d7d7d7", linewidth=0.35)
        for y in range(12, matrix.shape[0], 12):
            ax.axhline(y, color="#8a8a8a", linewidth=0.75)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xticks(x_centers)
    axes[-1].set_xticklabels(
        [ZONE_LABEL[zone] for zone in ZONE_PLOT_ORDER], rotation=45, ha="right"
    )
    axes[-1].tick_params(axis="x", labelsize=8)
    for boundary in [5, 10]:
        x = x_edges[boundary]
        for ax in axes:
            ax.axvline(x, color="#555555", linewidth=0.9)
    fig.suptitle(
        "Zone-size mosaic of high-tail screened share across 144 fixed-reference runs",
        fontsize=13,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.01,
        0.012,
        "Cell color: occupied records with synchronized zone p_tail >= 0.20. "
        "Rows within each scenario-time group are typical, hot, heatwave-extreme. "
        "Column width proportional to zone floor area.",
        fontsize=8,
    )
    assert mesh is not None
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.72])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("Zone high-tail screened share (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.965, bottom=0.09)
    fig.savefig(figure_dir / "zone_size_mosaic_heatmap_area.png", dpi=260)
    fig.savefig(figure_dir / "zone_size_mosaic_heatmap_area.pdf")
    plt.close(fig)


def make_corrected_floor_plot(
    zone_summary: pd.DataFrame, floor_summary: pd.DataFrame, figure_dir: Path
) -> None:
    zone_plot = zone_summary.copy()
    zone_plot["position"] = zone_plot["zone"].map(zone_position)
    heat = (
        zone_plot.pivot(index="position", columns="floor", values="mean_high_tail_pct")
        .loc[POSITION_ORDER, FLOOR_ORDER]
        .to_numpy(float)
    )
    floor_index = floor_summary.set_index("floor").loc[FLOOR_ORDER]
    warm = floor_index["mean_warm_tail_area_weighted"].to_numpy(float)
    cold = floor_index["mean_cold_tail_area_weighted"].to_numpy(float)
    area_weighted = floor_index["area_weighted_high_tail_pct"].to_numpy(float)
    any_floor = floor_index["any_zone_within_floor_high_tail_pct"].to_numpy(float)

    fig = plt.figure(figsize=(11.0, 7.2), constrained_layout=False)
    grid = fig.add_gridspec(
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
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_floor = fig.add_subplot(grid[0, 1])
    ax_tail = fig.add_subplot(grid[1, 1])

    image = ax_heat.imshow(
        heat,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(35, float(np.ceil(heat.max() / 5.0) * 5.0)),
    )
    ax_heat.set_xticks(range(len(FLOOR_ORDER)))
    ax_heat.set_xticklabels(["Bottom", "Middle", "Top"])
    ax_heat.set_yticks(range(len(POSITION_ORDER)))
    ax_heat.set_yticklabels([POSITION_NAMES[position] for position in POSITION_ORDER])
    ax_heat.set_title("Zone-position high-tail screened share (%)", fontweight="bold")
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
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.03)
    colorbar.set_label("High-tail screened share (%)")

    x = np.arange(len(FLOOR_ORDER))
    labels = ["Bottom", "Middle", "Top"]
    ax_floor.bar(
        x - 0.18,
        area_weighted,
        width=0.36,
        color="#4c78a8",
        label="Area-weighted",
    )
    ax_floor.bar(
        x + 0.18,
        any_floor,
        width=0.36,
        color="#d45b43",
        label="Any zone on floor",
    )
    ax_floor.set_xticks(x)
    ax_floor.set_xticklabels(labels)
    ax_floor.set_ylabel("Screened share (%)")
    ax_floor.set_title("Floor-level screening summaries", fontweight="bold")
    ax_floor.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax_floor.set_ylim(0, max(any_floor.max(), area_weighted.max()) * 1.18)
    ax_floor.legend(frameon=False, fontsize=8)
    for location, value in zip(x - 0.18, area_weighted):
        ax_floor.text(
            location, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=8
        )
    for location, value in zip(x + 0.18, any_floor):
        ax_floor.text(
            location, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=8
        )

    ax_tail.bar(x, warm, color="#e4572e", label="Warm tail")
    ax_tail.bar(x, cold, bottom=warm, color="#3b73b9", label="Cold tail")
    ax_tail.set_xticks(x)
    ax_tail.set_xticklabels(labels)
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
        "Screened share is the percentage of occupied records with synchronized zone p_tail >= 0.20.",
        fontsize=8,
    )
    fig.savefig(figure_dir / "zone_floor_position_comparison.png", dpi=260)
    fig.savefig(figure_dir / "zone_floor_position_comparison.pdf")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column).replace("|", r"\|") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in display.itertuples(index=False, name=None):
        cells = [
            ("" if pd.isna(value) else str(value)).replace("|", r"\|").replace("\n", " ")
            for value in values
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def lookup_curve(
    curve: pd.DataFrame, model: str, event: str, threshold: float
) -> pd.Series:
    row = curve[
        curve["model"].eq(model)
        & curve["event"].eq(event)
        & np.isclose(curve["threshold"], threshold)
    ]
    require(len(row) == 1, f"Expected one curve row for {model}/{event}/{threshold}")
    return row.iloc[0]


def write_reported_value_map(
    input_dir: Path,
    output_dir: Path,
    mean_summary: pd.DataFrame,
    scalar_global: pd.DataFrame,
    unique_global: pd.DataFrame,
    zone_summary: pd.DataFrame,
    floor_summary: pd.DataFrame,
) -> None:
    headline = pd.read_csv(input_dir / "corrected_headline_case_summary.csv")
    curve = pd.read_csv(input_dir / "global_threshold_curves.csv")
    grouped_cont = pd.read_csv(input_dir / "grouped_continuous_summary.csv")
    grouped_curve = pd.read_csv(input_dir / "grouped_threshold_curves.csv")
    endpoint_cont = pd.read_csv(input_dir / "global_continuous_summary.csv")
    nominal_pair = pd.read_csv(input_dir / "global_paired_model_thresholds.csv")

    corrected = lookup_curve(curve, "ordinal", "broad_tail", TAIL_SCREEN)
    stored = lookup_curve(curve, "stored_ordinal", "broad_tail", TAIL_SCREEN)
    nominal = lookup_curve(curve, "nominal", "broad_tail", TAIL_SCREEN)
    no_pmv = lookup_curve(curve, "no_pmv_ordinal", "broad_tail", TAIL_SCREEN)

    old_mean = float(stored["equal_zone_mean_high_pct"])
    old_any = float(stored["any_zone_high_pct"])
    old_hidden = float(stored["hidden_any_zone_pct"])
    new_mean = float(corrected["equal_zone_mean_high_pct"])
    new_any = float(corrected["any_zone_high_pct"])
    new_hidden = float(corrected["hidden_any_zone_pct"])

    legacy_scalar_path = (
        ROOT
        / "restricted_inputs"
        / "legacy_summaries"
        / "scalar_tail_global_summary.csv"
    )
    legacy_scalar = pd.read_csv(legacy_scalar_path) if legacy_scalar_path.exists() else pd.DataFrame()
    old_unique_path = (
        ROOT
        / "restricted_inputs"
        / "legacy_summaries"
        / "unique_weather_global_comparison.csv"
    )
    old_unique = pd.read_csv(old_unique_path) if old_unique_path.exists() else pd.DataFrame()
    legacy_zone_path = (
        ROOT / "restricted_inputs" / "legacy_summaries" / "zone_size_summary.csv"
    )
    legacy_zone = pd.read_csv(legacy_zone_path) if legacy_zone_path.exists() else pd.DataFrame()
    legacy_floor_path = (
        ROOT / "restricted_inputs" / "legacy_summaries" / "zone_floor_summary.csv"
    )
    legacy_floor = pd.read_csv(legacy_floor_path) if legacy_floor_path.exists() else pd.DataFrame()

    def old_unique_value(metric: str) -> float:
        if old_unique.empty:
            return math.nan
        rows = old_unique[old_unique["metric"].eq(metric)]
        return float(rows.iloc[0]["unique_year_value"]) if len(rows) else math.nan

    unique_row = unique_global.iloc[0]
    primary_scalar = scalar_global[scalar_global["model"].eq("ordinal")]
    no_pmv_scalar = scalar_global[scalar_global["model"].eq("no_pmv_ordinal")]
    zone_aggregation_group = pd.read_csv(
        output_dir / "corrected_zone_aggregation_group_summary.csv"
    )
    zone_global = zone_aggregation_group[
        zone_aggregation_group["group_scope"].eq("global")
    ].iloc[0]

    replacement_rows: list[dict[str, object]] = [
        {
            "location_or_object": "Abstract/results/conclusion: mean-zone screened share",
            "legacy": old_mean,
            "corrected": new_mean,
            "unit": "%",
            "source": "global_threshold_curves.csv",
        },
        {
            "location_or_object": "Results/conclusion: any-zone screened share",
            "legacy": old_any,
            "corrected": new_any,
            "unit": "%",
            "source": "global_threshold_curves.csv",
        },
        {
            "location_or_object": "Abstract/results/conclusion: mean-hidden any-zone share",
            "legacy": old_hidden,
            "corrected": new_hidden,
            "unit": "%",
            "source": "global_threshold_curves.csv",
        },
        {
            "location_or_object": "Full-panel mean p_tail",
            "legacy": float(headline["stored_equal_zone_mean_mean"].mean()),
            "corrected": float(headline["corrected_equal_zone_mean_mean"].mean()),
            "unit": "probability",
            "source": "corrected_headline_case_summary.csv",
        },
        {
            "location_or_object": "Unique-source mean-zone screened share",
            "legacy": old_unique_value("mean_zone_high_tail_pct"),
            "corrected": float(unique_row["equal_zone_mean_high_pct"]),
            "unit": "%",
            "source": "corrected_unique_state_global_summary.csv",
        },
        {
            "location_or_object": "Unique-source any-zone screened share",
            "legacy": old_unique_value("any_zone_high_tail_pct"),
            "corrected": float(unique_row["any_zone_high_pct"]),
            "unit": "%",
            "source": "corrected_unique_state_global_summary.csv",
        },
        {
            "location_or_object": "Unique-source mean-hidden any-zone share",
            "legacy": old_unique_value("hidden_any_zone_high_tail_pct"),
            "corrected": float(unique_row["hidden_any_zone_pct"]),
            "unit": "%",
            "source": "corrected_unique_state_global_summary.csv",
        },
        {
            "location_or_object": "No-PMV mean-zone screened share",
            "legacy": 13.847232199872286,
            "corrected": float(no_pmv["equal_zone_mean_high_pct"]),
            "unit": "%",
            "source": "global_threshold_curves.csv",
        },
        {
            "location_or_object": "No-PMV any-zone screened share",
            "legacy": 37.52,
            "corrected": float(no_pmv["any_zone_high_pct"]),
            "unit": "%",
            "source": "global_threshold_curves.csv",
        },
        {
            "location_or_object": "Nominal mean-zone screened share",
            "legacy": math.nan,
            "corrected": float(nominal["equal_zone_mean_high_pct"]),
            "unit": "%",
            "source": "global_threshold_curves.csv",
        },
        {
            "location_or_object": "Nominal any-zone screened share",
            "legacy": math.nan,
            "corrected": float(nominal["any_zone_high_pct"]),
            "unit": "%",
            "source": "global_threshold_curves.csv",
        },
        {
            "location_or_object": "Nominal mean-hidden any-zone share",
            "legacy": math.nan,
            "corrected": float(nominal["hidden_any_zone_pct"]),
            "unit": "%",
            "source": "global_threshold_curves.csv",
        },
        {
            "location_or_object": "Max-zone more-severe reporting category",
            "legacy": 31.686439043209873,
            "corrected": float(zone_global["max_zone_more_severe_category_pct"]),
            "unit": "%",
            "source": "corrected_zone_aggregation_group_summary.csv",
        },
        {
            "location_or_object": "P90-zone more-severe reporting category",
            "legacy": 20.94138296881652,
            "corrected": float(zone_global["p90_zone_more_severe_category_pct"]),
            "unit": "%",
            "source": "corrected_zone_aggregation_group_summary.csv",
        },
        {
            "location_or_object": "Warm-tail probability exceeds cold-tail probability",
            "legacy": 78.9,
            "corrected": float(zone_global["warm_tail_gt_cold_tail_pct"]),
            "unit": "%",
            "source": "corrected_zone_aggregation_group_summary.csv",
        },
    ]

    if not legacy_scalar.empty:
        for scalar, label in [
            ("abs_mean_pmv", "|PMV|"),
            ("abs_mu_tsv", "|mu_TSV|"),
        ]:
            old_row = legacy_scalar[legacy_scalar["scalar"].eq(scalar)].iloc[0]
            new_row = primary_scalar[primary_scalar["scalar"].eq(scalar)].iloc[0]
            for field, friendly in [
                ("pearson_r", "Pearson r"),
                ("spearman_r", "Spearman rho"),
                ("best_threshold", "optimized threshold"),
                ("best_disagreement_pct", "optimized disagreement"),
                ("standard_disagreement_pct", "0.5-band disagreement"),
            ]:
                replacement_rows.append(
                    {
                        "location_or_object": f"Scalar table: {label} {friendly}",
                        "legacy": float(old_row[field]),
                        "corrected": float(new_row[field]),
                        "unit": "%" if field.endswith("_pct") else "",
                        "source": "corrected_scalar_global_summary.csv",
                    }
                )

    # Grouped scenario/slice/city/severity values.
    old_mean_path = (
        ROOT / "restricted_inputs" / "legacy_summaries" / "mean_tail_summary.csv"
    )
    old_mean_table = pd.read_csv(old_mean_path) if old_mean_path.exists() else pd.DataFrame()
    for _, corrected_group in mean_summary[
        mean_summary["scope"].isin(["city", "scenario", "time_slice", "severity"])
    ].iterrows():
            scope = str(corrected_group["scope"])
            value = str(corrected_group["level"])
            legacy_mean = legacy_pct = math.nan
            if not old_mean_table.empty:
                old_rows = old_mean_table[old_mean_table["scope"].eq(scope)].copy()
                old_key = "scenario_raw" if scope == "scenario" else scope
                if old_key in old_rows:
                    old_rows = old_rows[old_rows[old_key].astype(str).eq(value)]
                if len(old_rows) == 1:
                    legacy_mean = float(old_rows.iloc[0]["mean_p_tail"])
                    legacy_pct = float(old_rows.iloc[0]["high_tail_pct"])
            replacement_rows.extend(
                [
                    {
                        "location_or_object": f"Panel table: {scope}={value} mean p_tail",
                        "legacy": legacy_mean,
                        "corrected": float(corrected_group["mean_p_tail"]),
                        "unit": "probability",
                        "source": "corrected_mean_tail_summary.csv",
                    },
                    {
                        "location_or_object": f"Panel table: {scope}={value} screened share",
                        "legacy": legacy_pct,
                        "corrected": float(corrected_group["high_tail_pct"]),
                        "unit": "%",
                        "source": "corrected_mean_tail_summary.csv",
                    },
                ]
            )

    # Zone/floor table replacements.
    if not legacy_zone.empty:
        for _, new_row in zone_summary.iterrows():
            old_rows = legacy_zone[legacy_zone["zone"].eq(new_row["zone"])]
            replacement_rows.append(
                {
                    "location_or_object": f"Zone summary: {new_row['zone_label']} high-tail share",
                    "legacy": float(old_rows.iloc[0]["mean_high_tail_pct"]) if len(old_rows) else math.nan,
                    "corrected": float(new_row["mean_high_tail_pct"]),
                    "unit": "%",
                    "source": "corrected_zone_summary.csv",
                }
            )
    if not legacy_floor.empty:
        for _, new_row in floor_summary.iterrows():
            old_rows = legacy_floor[legacy_floor["floor"].eq(new_row["floor"])]
            for field in [
                "area_weighted_high_tail_pct",
                "unweighted_zone_time_high_tail_pct",
                "any_zone_within_floor_high_tail_pct",
            ]:
                replacement_rows.append(
                    {
                        "location_or_object": f"Floor summary: {new_row['floor']} {field}",
                        "legacy": float(old_rows.iloc[0][field]) if len(old_rows) else math.nan,
                        "corrected": float(new_row[field]),
                        "unit": "%",
                        "source": "corrected_floor_summary.csv",
                    }
                )

    endpoint_rows = endpoint_cont[
        endpoint_cont["model"].eq("ordinal")
        & endpoint_cont["event"].eq("outermost")
    ]
    endpoint_metric = (
        endpoint_rows.set_index("metric")["mean"].to_dict()
        if len(endpoint_rows)
        else {}
    )
    endpoint_screen = lookup_curve(curve, "ordinal", "outermost", 0.05)
    for metric in [
        "equal_zone_mean",
        "area_weighted_mean",
        "zone_p90",
        "any_zone",
        "cold_equal",
        "warm_equal",
    ]:
        if metric in endpoint_metric:
            replacement_rows.append(
                {
                    "location_or_object": f"Endpoint sensitivity: mean {metric}",
                    "legacy": math.nan,
                    "corrected": float(endpoint_metric[metric]),
                    "unit": "probability",
                    "source": "global_continuous_summary.csv",
                }
            )
    for field in [
        "equal_zone_mean_high_pct",
        "area_weighted_mean_high_pct",
        "zone_p90_high_pct",
        "any_zone_high_pct",
        "hidden_any_zone_pct",
    ]:
        replacement_rows.append(
            {
                "location_or_object": f"Endpoint sensitivity at 0.05: {field}",
                "legacy": math.nan,
                "corrected": float(endpoint_screen[field]),
                "unit": "%",
                "source": "global_threshold_curves.csv",
            }
        )

    replacement = pd.DataFrame(replacement_rows)
    replacement["difference_corrected_minus_legacy"] = (
        pd.to_numeric(replacement["corrected"], errors="coerce")
        - pd.to_numeric(replacement["legacy"], errors="coerce")
    )
    replacement.to_csv(output_dir / "reported_values.csv", index=False)

    paired_at_020 = nominal_pair[
        nominal_pair["event"].eq("broad_tail")
        & np.isclose(nominal_pair["threshold"], TAIL_SCREEN)
    ]
    no_pmv_rows = no_pmv_scalar[
        no_pmv_scalar["scalar"].isin(["abs_mean_pmv", "abs_mu_tsv"])
    ]
    figure_status = pd.DataFrame(
        [
            {
                "object": "Figure 1: expected TSV versus p_tail",
                "source status": "SYNCHRONIZED",
                "basis": "Both axes were recomputed on the synchronized end-of-step environmental state.",
                "source artifact": "figures/mean_vs_tail_scatter_sample_corrected.pdf",
            },
            {
                "object": "Figure 2: scalar-tail comparison",
                "source status": "SYNCHRONIZED",
                "basis": "PMV, expected TSV, and p_tail were evaluated on the same recorded state.",
                "source artifact": "figures/scalar_tail_comparison_hexbin_corrected.pdf",
            },
            {
                "object": "Figure 3: PMV-feature robustness",
                "source status": "SYNCHRONIZED",
                "basis": "Corrected PMV and both ordinal outputs are synchronized and explicitly labelled.",
                "source artifact": "figures/pmv_feature_robustness_hexbin_corrected.pdf",
            },
            {
                "object": "Threshold-sensitivity figure",
                "source status": "SYNCHRONIZED",
                "basis": "The figure uses synchronized global_threshold_curves.csv; legacy coordinates are retained only for audit.",
                "source artifact": "../core/robustness_threshold_curves.pdf",
            },
            {
                "object": "Zone-size mosaic figure",
                "source status": "SYNCHRONIZED",
                "basis": "Every cell was recomputed from synchronized zone probabilities.",
                "source artifact": "figures/zone_size_mosaic_heatmap_area.pdf",
            },
            {
                "object": "Zone/floor position figure",
                "source status": "SYNCHRONIZED",
                "basis": "Zone, floor, warm-tail, and cold-tail summaries use synchronized probabilities.",
                "source artifact": "figures/zone_floor_position_comparison.pdf",
            },
            {
                "object": "Held-out model calibration/validation figures",
                "source status": "RETAINED",
                "basis": "These use the held-out occupant dataset, not the callback-timed simulation trace.",
                "source artifact": "None",
            },
            {
                "object": "Metabolic sensitivity figures",
                "source status": "SEPARATE PROVENANCE",
                "basis": "Their inference was recomputed from recorded environmental fields under the metabolic-analysis provenance.",
                "source artifact": "See metabolic-analysis provenance.",
            },
        ]
    )
    figure_status.to_csv(output_dir / "figure_source_status.csv", index=False)

    lines = [
        "# Reported Value Audit",
        "",
        "All corrected quantities below come from the synchronized end-of-zone-timestep inference. "
        "The legacy trace files remain unchanged. `legacy` values are retained for audit, not for reuse.",
        "",
        "## Headline quantities",
        "",
        markdown_table(replacement.head(12)),
        "",
        "## Nominal-model robustness at the 0.20 screen",
        "",
        markdown_table(paired_at_020),
        "",
        "## Corrected scalar audit",
        "",
        markdown_table(primary_scalar),
        "",
        "## Corrected no-PMV audit",
        "",
        markdown_table(no_pmv_rows),
        "",
        "## Unique-environmental-state sensitivity",
        "",
        markdown_table(unique_global),
        "",
        "## Figure source status",
        "",
        markdown_table(figure_status),
        "",
        "## Reported-value ledger",
        "",
        "The full old-to-corrected ledger is `reported_values.csv`; it includes "
        "scenario, time-slice, city, severity, zone, floor, scalar, nominal, and endpoint entries.",
        "",
        "## Interpretation and reuse notes",
        "",
        "- The manuscript, tables, captions, and figures use the corrected quantities.",
        "- Do not mix stored callback-timed PMV/probabilities with corrected end-state quantities.",
        "- Describe the nominal and endpoint analyses as robustness/sensitivity checks, not as external validation.",
        "- Endpoint-only results are support-limited because TSV -3 and +3 are sparse in held-out data.",
        "- Keep the 144 role-weighted panel and the 119 unique-state sensitivity distinct.",
        "",
    ]
    (output_dir / "reported_value_map.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = input_dir / "corrected_figure_input_manifest.csv"
    require(manifest_path.exists(), f"Missing final manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    require(len(manifest) == EXPECTED_CASES, f"Expected {EXPECTED_CASES} roles, found {len(manifest)}")
    require(
        manifest["analysis_state_sha256"].nunique() == EXPECTED_UNIQUE_STATES,
        f"Expected {EXPECTED_UNIQUE_STATES} unique states, found "
        f"{manifest['analysis_state_sha256'].nunique()}",
    )
    require(
        (manifest["occupied_steps"] == EXPECTED_STEPS_PER_CASE).all(),
        "One or more roles do not contain 16,704 occupied steps",
    )
    manifest = manifest.sort_values("weather").reset_index(drop=True)
    total_rows = int(manifest["occupied_steps"].sum())
    rng = np.random.default_rng(RANDOM_SEED)
    sample_global = np.sort(
        rng.choice(total_rows, size=min(args.sample_rows, total_rows), replace=False)
    )
    sample_roles, sample_local = role_local_index(sample_global, EXPECTED_STEPS_PER_CASE)

    primary_mu_parts: list[np.ndarray] = []
    primary_tail_parts: list[np.ndarray] = []
    warm_parts: list[np.ndarray] = []
    cold_parts: list[np.ndarray] = []
    no_pmv_mu_parts: list[np.ndarray] = []
    no_pmv_tail_parts: list[np.ndarray] = []
    pmv_parts: list[np.ndarray] = []
    city_code_parts: list[np.ndarray] = []
    sample_parts: list[pd.DataFrame] = []
    city_names = sorted(manifest["city"].unique())
    city_to_code = {city: index for index, city in enumerate(city_names)}

    unique_case_rows: list[dict[str, object]] = []
    zone_aggregation_case_rows: list[dict[str, object]] = []
    zone_case_rows: list[dict[str, object]] = []
    floor_case_rows: list[dict[str, object]] = []
    loaded_cache: dict[str, dict[str, np.ndarray]] = {}
    validated_hashes: set[str] = set()
    zone_names_reference: list[str] | None = None
    weights_reference: np.ndarray | None = None

    for role_index, row in manifest.iterrows():
        state_hash = str(row["analysis_state_sha256"])
        npz_path = Path(row["corrected_zone_npz"])
        require(npz_path.exists(), f"Missing NPZ: {npz_path}")
        if state_hash not in validated_hashes and not args.skip_hash_check:
            observed_hash = file_sha256(npz_path)
            require(
                observed_hash == str(row["corrected_zone_npz_sha256"]),
                f"NPZ checksum mismatch: {npz_path}",
            )
            validated_hashes.add(state_hash)
        if state_hash not in loaded_cache:
            with np.load(npz_path, allow_pickle=False) as archive:
                require(
                    REQUIRED_NPZ_FIELDS.issubset(archive.files),
                    f"NPZ schema fields missing: {npz_path}",
                )
                require(
                    str(archive["schema_version"].item()) == EXPECTED_SCHEMA,
                    f"Unexpected NPZ schema in {npz_path}",
                )
                loaded_cache[state_hash] = {
                    field: np.asarray(archive[field])
                    for field in archive.files
                    if field != "schema_version"
                }
        data = loaded_cache[state_hash]
        zone_names = [str(value) for value in data["zone_names"]]
        if zone_names_reference is None:
            zone_names_reference = zone_names
            require(set(zone_names) == set(ZONE_AREAS_M2), "Unexpected zone-name set")
            weights_reference = np.array([ZONE_AREAS_M2[name] for name in zone_names], dtype=float)
            weights_reference /= weights_reference.sum()
        else:
            require(zone_names == zone_names_reference, "Zone order changes across NPZ files")
        assert weights_reference is not None

        zone_tail = (
            data["zone_cold_tail"].astype(float)
            + data["zone_warm_tail"].astype(float)
        )
        zone_warm = data["zone_warm_tail"].astype(float)
        zone_cold = data["zone_cold_tail"].astype(float)
        zone_mu = data["zone_expected_tsv"].astype(float)
        zone_pmv = data["zone_pmv"].astype(float)
        zone_no_pmv_tail = (
            data["zone_no_pmv_cold_tail"].astype(float)
            + data["zone_no_pmv_warm_tail"].astype(float)
        )
        zone_no_pmv_mu = data["zone_no_pmv_expected_tsv"].astype(float)
        n = zone_tail.shape[0]
        require(n == EXPECTED_STEPS_PER_CASE, f"Unexpected row count in {npz_path}")

        primary_tail = zone_tail.mean(axis=1)
        primary_mu = zone_mu.mean(axis=1)
        warm = zone_warm.mean(axis=1)
        cold = zone_cold.mean(axis=1)
        no_pmv_tail = zone_no_pmv_tail.mean(axis=1)
        no_pmv_mu = zone_no_pmv_mu.mean(axis=1)
        mean_pmv = zone_pmv.mean(axis=1)

        primary_tail_parts.append(primary_tail.astype(np.float32))
        primary_mu_parts.append(primary_mu.astype(np.float32))
        warm_parts.append(warm.astype(np.float32))
        cold_parts.append(cold.astype(np.float32))
        no_pmv_tail_parts.append(no_pmv_tail.astype(np.float32))
        no_pmv_mu_parts.append(no_pmv_mu.astype(np.float32))
        pmv_parts.append(mean_pmv.astype(np.float32))
        city_code_parts.append(
            np.full(n, city_to_code[str(row["city"])], dtype=np.int8)
        )

        local_for_role = sample_local[sample_roles == role_index]
        if len(local_for_role):
            sample_parts.append(
                pd.DataFrame(
                    {
                        "global_role_row": role_index * EXPECTED_STEPS_PER_CASE + local_for_role,
                        "weather": str(row["weather"]),
                        "city": str(row["city"]),
                        "scenario": str(row["scenario"]),
                        "time_slice": str(row["time_slice"]),
                        "severity": str(row["severity"]),
                        "weather_year": int(row["weather_year"]),
                        "source_row_index": data["source_row_index"][local_for_role],
                        "month": data["month"][local_for_role],
                        "day": data["day"][local_for_role],
                        "hour": data["hour"][local_for_role],
                        "current_time": data["current_time"][local_for_role],
                        "mean_pmv": mean_pmv[local_for_role],
                        "abs_pmv": np.abs(mean_pmv[local_for_role]),
                        "expected_tsv": primary_mu[local_for_role],
                        "abs_mu_tsv": np.abs(primary_mu[local_for_role]),
                        "p_tail": primary_tail[local_for_role],
                        "warm_tail": warm[local_for_role],
                        "cold_tail": cold[local_for_role],
                        "no_pmv_expected_tsv": no_pmv_mu[local_for_role],
                        "no_pmv_p_tail": no_pmv_tail[local_for_role],
                    }
                )
            )

        high = zone_tail >= TAIL_SCREEN
        zone_p90 = np.quantile(zone_tail, 0.90, axis=1)
        any_zone = zone_tail.max(axis=1)
        area_mean = zone_tail @ weights_reference
        equal_mean = primary_tail
        mean_category = tail_category(equal_mean)
        p90_category = tail_category(zone_p90)
        any_category = tail_category(any_zone)
        zone_aggregation_case_rows.append(
            {
                "weather": str(row["weather"]),
                "city": str(row["city"]),
                "scenario": str(row["scenario"]),
                "time_slice": str(row["time_slice"]),
                "severity": str(row["severity"]),
                "weather_year": int(row["weather_year"]),
                "analysis_state_sha256": state_hash,
                "n_steps": n,
                "mean_p_tail": float(equal_mean.mean()),
                "p90_zone_p_tail": float(zone_p90.mean()),
                "max_zone_p_tail": float(any_zone.mean()),
                "mean_zone_high_tail_pct": pct(equal_mean >= TAIL_SCREEN),
                "p90_zone_high_tail_pct": pct(zone_p90 >= TAIL_SCREEN),
                "any_zone_high_tail_pct": pct(any_zone >= TAIL_SCREEN),
                "hidden_any_zone_pct": pct(
                    (equal_mean < TAIL_SCREEN) & (any_zone >= TAIL_SCREEN)
                ),
                "max_zone_more_severe_category_pct": pct(
                    any_category > mean_category
                ),
                "p90_zone_more_severe_category_pct": pct(
                    p90_category > mean_category
                ),
                "warm_tail_gt_cold_tail_pct": pct(warm > cold),
                "warm_tail_gt_cold_tail_among_high_pct": (
                    pct((warm > cold)[equal_mean >= TAIL_SCREEN])
                    if np.any(equal_mean >= TAIL_SCREEN)
                    else math.nan
                ),
            }
        )
        unique_case_rows.append(
            {
                "analysis_state_sha256": state_hash,
                "representative_weather": str(row["weather"]),
                "n_steps": n,
                "equal_zone_mean_mean": float(equal_mean.mean()),
                "area_weighted_mean_mean": float(area_mean.mean()),
                "zone_p90_mean": float(zone_p90.mean()),
                "any_zone_mean": float(any_zone.mean()),
                "equal_zone_mean_high_pct": pct(equal_mean >= TAIL_SCREEN),
                "area_weighted_mean_high_pct": pct(area_mean >= TAIL_SCREEN),
                "zone_p90_high_pct": pct(zone_p90 >= TAIL_SCREEN),
                "any_zone_high_pct": pct(any_zone >= TAIL_SCREEN),
                "hidden_any_zone_pct": pct(
                    (equal_mean < TAIL_SCREEN) & (any_zone >= TAIL_SCREEN)
                ),
                "area_weighted_zone_time_high_pct": float(
                    (high.astype(float) @ weights_reference).mean() * 100.0
                ),
                "unweighted_zone_time_high_pct": pct(high),
            }
        )
        # Keep only one row per unique state in the unique sensitivity table.
        if sum(record["analysis_state_sha256"] == state_hash for record in unique_case_rows) > 1:
            unique_case_rows.pop()

        for zone_index, zone in enumerate(zone_names):
            zone_case_rows.append(
                {
                    "weather": str(row["weather"]),
                    "city": str(row["city"]),
                    "scenario": str(row["scenario"]),
                    "time_slice": str(row["time_slice"]),
                    "severity": str(row["severity"]),
                    "weather_year": int(row["weather_year"]),
                    "analysis_state_sha256": state_hash,
                    "zone": zone,
                    "zone_label": ZONE_LABEL[zone],
                    "floor": FLOOR_BY_ZONE[zone],
                    "area_m2": ZONE_AREAS_M2[zone],
                    "high_tail_pct": pct(high[:, zone_index]),
                    "mean_p_tail": float(zone_tail[:, zone_index].mean()),
                    "mean_warm_tail": float(zone_warm[:, zone_index].mean()),
                    "mean_cold_tail": float(zone_cold[:, zone_index].mean()),
                    "mean_expected_tsv": float(zone_mu[:, zone_index].mean()),
                }
            )
        for floor in ["bottom", "middle", "top"]:
            indices = np.array(
                [i for i, zone in enumerate(zone_names) if FLOOR_BY_ZONE[zone] == floor]
            )
            floor_weights = weights_reference[indices]
            floor_weights /= floor_weights.sum()
            floor_tail = zone_tail[:, indices]
            floor_high = high[:, indices]
            floor_case_rows.append(
                {
                    "weather": str(row["weather"]),
                    "analysis_state_sha256": state_hash,
                    "floor": floor,
                    "area_weighted_high_tail_pct": float(
                        (floor_high.astype(float) @ floor_weights).mean() * 100.0
                    ),
                    "unweighted_zone_time_high_tail_pct": pct(floor_high),
                    "any_zone_within_floor_high_tail_pct": pct(floor_high.any(axis=1)),
                    "mean_p_tail_area_weighted": float(
                        (floor_tail @ floor_weights).mean()
                    ),
                    "mean_warm_tail_area_weighted": float(
                        (zone_warm[:, indices] @ floor_weights).mean()
                    ),
                    "mean_cold_tail_area_weighted": float(
                        (zone_cold[:, indices] @ floor_weights).mean()
                    ),
                    "mean_expected_tsv_area_weighted": float(
                        (zone_mu[:, indices] @ floor_weights).mean()
                    ),
                }
            )

        # Cache only the most recently used state; selector duplicates are adjacent
        # in the sorted role manifest, so this bounds memory while retaining reuse.
        if role_index + 1 < len(manifest):
            next_hash = str(manifest.iloc[role_index + 1]["analysis_state_sha256"])
            if next_hash != state_hash:
                loaded_cache.pop(state_hash, None)
        else:
            loaded_cache.pop(state_hash, None)
        if (role_index + 1) % 24 == 0:
            print(f"[postprocess] {role_index + 1}/{len(manifest)} roles")

    mu = np.concatenate(primary_mu_parts).astype(float)
    tail = np.concatenate(primary_tail_parts).astype(float)
    warm = np.concatenate(warm_parts).astype(float)
    cold = np.concatenate(cold_parts).astype(float)
    no_pmv_mu = np.concatenate(no_pmv_mu_parts).astype(float)
    no_pmv_tail = np.concatenate(no_pmv_tail_parts).astype(float)
    pmv = np.concatenate(pmv_parts).astype(float)
    city_codes = np.concatenate(city_code_parts)
    require(len(mu) == total_rows, "Role-weighted scalar row count mismatch")
    require(np.all(np.isfinite(np.column_stack([mu, tail, warm, cold, no_pmv_mu, no_pmv_tail, pmv]))), "Nonfinite scalar output")

    sample = pd.concat(sample_parts, ignore_index=True).sort_values("global_role_row")
    require(len(sample) == min(args.sample_rows, total_rows), "Plot sample row count mismatch")
    sample.to_csv(output_dir / "corrected_scalar_plot_sample_seed42.csv.gz", index=False)

    mean_rows = [
        group_mean_tail({"scope": "all", "level": "full_panel"}, mu, tail, warm, cold)
    ]
    for city, code in city_to_code.items():
        mask = city_codes == code
        mean_rows.append(
            group_mean_tail({"scope": "city", "level": city}, mu[mask], tail[mask], warm[mask], cold[mask])
        )
    for group_name, values in [
        ("scenario", manifest["scenario"].unique()),
        ("time_slice", manifest["time_slice"].unique()),
        ("severity", manifest["severity"].unique()),
    ]:
        for value in sorted(values):
            role_mask = manifest[group_name].astype(str).eq(str(value)).to_numpy()
            row_mask = np.repeat(role_mask, EXPECTED_STEPS_PER_CASE)
            mean_rows.append(
                group_mean_tail(
                    {"scope": group_name, "level": value},
                    mu[row_mask],
                    tail[row_mask],
                    warm[row_mask],
                    cold[row_mask],
                )
            )
    mean_summary = pd.DataFrame(mean_rows)
    mean_summary.to_csv(output_dir / "corrected_mean_tail_summary.csv", index=False)

    high_tail = tail >= TAIL_SCREEN
    scalar_rows = [
        scalar_record("ordinal", "abs_mean_pmv", np.abs(pmv), tail, high_tail),
        scalar_record("ordinal", "abs_mu_tsv", np.abs(mu), tail, high_tail),
        scalar_record(
            "no_pmv_ordinal",
            "abs_mean_pmv",
            np.abs(pmv),
            no_pmv_tail,
            no_pmv_tail >= TAIL_SCREEN,
        ),
        scalar_record(
            "no_pmv_ordinal",
            "abs_mu_tsv",
            np.abs(no_pmv_mu),
            no_pmv_tail,
            no_pmv_tail >= TAIL_SCREEN,
        ),
    ]
    scalar_global = pd.DataFrame(scalar_rows)
    scalar_global.to_csv(output_dir / "corrected_scalar_global_summary.csv", index=False)

    scalar_city_rows = []
    for city, code in city_to_code.items():
        mask = city_codes == code
        for model, scalar, values, probability in [
            ("ordinal", "abs_mean_pmv", np.abs(pmv[mask]), tail[mask]),
            ("ordinal", "abs_mu_tsv", np.abs(mu[mask]), tail[mask]),
            ("no_pmv_ordinal", "abs_mean_pmv", np.abs(pmv[mask]), no_pmv_tail[mask]),
            ("no_pmv_ordinal", "abs_mu_tsv", np.abs(no_pmv_mu[mask]), no_pmv_tail[mask]),
        ]:
            high = probability >= TAIL_SCREEN
            standard = threshold_metrics(values, high, 0.5)
            scalar_city_rows.append(
                {
                    "city": city,
                    "model": model,
                    "scalar": scalar,
                    "rows": int(mask.sum()),
                    "high_tail_pct": pct(high),
                    "pearson_r": float(pd.Series(values).corr(pd.Series(probability), method="pearson")),
                    "spearman_r": float(pd.Series(values).corr(pd.Series(probability), method="spearman")),
                    **{f"standard_{key}": value for key, value in standard.items()},
                }
            )
    pd.DataFrame(scalar_city_rows).to_csv(
        output_dir / "corrected_scalar_city_summary.csv", index=False
    )

    extra_scalar = pd.DataFrame(
        [
            {
                "rows": total_rows,
                "mean_p_tail": float(tail.mean()),
                "p95_p_tail": float(np.quantile(tail, 0.95)),
                "max_abs_mu_tsv": float(np.abs(mu).max()),
                "abs_mu_tsv_ge_1_pct": pct(np.abs(mu) >= 1.0),
                "max_abs_pmv": float(np.abs(pmv).max()),
                "pmv_ge_0p5_below_tail_screen_pct_of_pmv_ge_0p5": pct(
                    tail[np.abs(pmv) >= 0.5] < TAIL_SCREEN
                ),
                "pmv_ge_1_below_tail_screen_pct_of_pmv_ge_1": pct(
                    tail[np.abs(pmv) >= 1.0] < TAIL_SCREEN
                ),
                "mu_nonpositive_high_tail_pct": pct((mu <= 0) & high_tail),
                "mu_gt_1_high_tail_pct_of_mu_gt_1": pct(
                    high_tail[mu > 1]
                ) if np.any(mu > 1) else math.nan,
                "no_pmv_mean_p_tail": float(no_pmv_tail.mean()),
                "no_pmv_high_tail_pct": pct(no_pmv_tail >= TAIL_SCREEN),
            }
        ]
    )
    extra_scalar.to_csv(output_dir / "corrected_scalar_extended_summary.csv", index=False)
    scalar_band_summary(pmv, mu, tail).to_csv(
        output_dir / "corrected_scalar_p_tail_band_quantiles.csv", index=False
    )

    mean_bins = build_mean_bins(mu, tail)
    mean_bins.to_csv(output_dir / "corrected_mean_bin_tail_spread.csv", index=False)
    write_figures(sample, mean_bins, output_dir)

    unique_case = pd.DataFrame(unique_case_rows).sort_values("analysis_state_sha256")
    require(len(unique_case) == EXPECTED_UNIQUE_STATES, "Unique-state summary count mismatch")
    unique_case.to_csv(output_dir / "corrected_unique_state_case_summary.csv", index=False)
    unique_global = pd.DataFrame(
        [
            {
                "scope": "119_unique_environmental_states_equal_weight",
                "unique_states": len(unique_case),
                **{
                    column: float(unique_case[column].mean())
                    for column in [
                        "equal_zone_mean_mean",
                        "area_weighted_mean_mean",
                        "zone_p90_mean",
                        "any_zone_mean",
                        "equal_zone_mean_high_pct",
                        "area_weighted_mean_high_pct",
                        "zone_p90_high_pct",
                        "any_zone_high_pct",
                        "hidden_any_zone_pct",
                        "area_weighted_zone_time_high_pct",
                        "unweighted_zone_time_high_pct",
                    ]
                },
            }
        ]
    )
    unique_global.to_csv(
        output_dir / "corrected_unique_state_global_summary.csv", index=False
    )

    zone_case = pd.DataFrame(zone_case_rows)
    zone_case.to_csv(output_dir / "corrected_zone_case_summary.csv", index=False)
    zone_summary_rows = []
    total_area = sum(ZONE_AREAS_M2.values())
    for zone, group in zone_case.groupby("zone", sort=True):
        zone_summary_rows.append(
            {
                "zone": zone,
                "zone_label": group.iloc[0]["zone_label"],
                "floor": group.iloc[0]["floor"],
                "area_m2": ZONE_AREAS_M2[zone],
                "area_share_pct": ZONE_AREAS_M2[zone] / total_area * 100.0,
                "mean_high_tail_pct": float(group["high_tail_pct"].mean()),
                "p95_case_high_tail_pct": float(group["high_tail_pct"].quantile(0.95)),
                "max_case_high_tail_pct": float(group["high_tail_pct"].max()),
                "mean_p_tail": float(group["mean_p_tail"].mean()),
                "mean_warm_tail": float(group["mean_warm_tail"].mean()),
                "mean_cold_tail": float(group["mean_cold_tail"].mean()),
                "mean_expected_tsv": float(group["mean_expected_tsv"].mean()),
            }
        )
    zone_summary = pd.DataFrame(zone_summary_rows)
    zone_summary.to_csv(output_dir / "corrected_zone_summary.csv", index=False)

    floor_case = pd.DataFrame(floor_case_rows)
    floor_case.to_csv(output_dir / "corrected_floor_case_summary.csv", index=False)
    highest_area = (
        floor_case.loc[floor_case.groupby("weather")["area_weighted_high_tail_pct"].idxmax()]
        ["floor"]
        .value_counts()
    )
    highest_unweighted = (
        floor_case.loc[floor_case.groupby("weather")["unweighted_zone_time_high_tail_pct"].idxmax()]
        ["floor"]
        .value_counts()
    )
    highest_any = (
        floor_case.loc[floor_case.groupby("weather")["any_zone_within_floor_high_tail_pct"].idxmax()]
        ["floor"]
        .value_counts()
    )
    floor_summary_rows = []
    for floor, group in floor_case.groupby("floor", sort=True):
        floor_summary_rows.append(
            {
                "floor": floor,
                **{
                    column: float(group[column].mean())
                    for column in [
                        "area_weighted_high_tail_pct",
                        "unweighted_zone_time_high_tail_pct",
                        "any_zone_within_floor_high_tail_pct",
                        "mean_p_tail_area_weighted",
                        "mean_warm_tail_area_weighted",
                        "mean_cold_tail_area_weighted",
                        "mean_expected_tsv_area_weighted",
                    ]
                },
                "cases_floor_has_highest_area_weighted_tail": int(highest_area.get(floor, 0)),
                "cases_floor_has_highest_unweighted_tail": int(highest_unweighted.get(floor, 0)),
                "cases_floor_has_highest_any_zone_tail": int(highest_any.get(floor, 0)),
            }
        )
    floor_summary = pd.DataFrame(floor_summary_rows)
    floor_summary.to_csv(output_dir / "corrected_floor_summary.csv", index=False)

    zone_aggregation_case = pd.DataFrame(zone_aggregation_case_rows)
    zone_aggregation_case.to_csv(
        output_dir / "corrected_zone_aggregation_case_summary.csv", index=False
    )
    zone_aggregation_group = aggregate_zone_aggregation_cases(zone_aggregation_case)
    zone_aggregation_group.to_csv(
        output_dir / "corrected_zone_aggregation_group_summary.csv", index=False
    )
    figure_dir = output_dir / "figures"
    make_corrected_zone_mosaic(zone_case, figure_dir)
    make_corrected_floor_plot(zone_summary, floor_summary, figure_dir)

    write_reported_value_map(
        input_dir,
        output_dir,
        mean_summary,
        scalar_global,
        unique_global,
        zone_summary,
        floor_summary,
    )

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "roles": len(manifest),
        "unique_states": manifest["analysis_state_sha256"].nunique(),
        "role_weighted_rows": total_rows,
        "npz_schema": EXPECTED_SCHEMA,
        "tail_screen": TAIL_SCREEN,
        "plot_sample_rows": len(sample),
        "random_seed": RANDOM_SEED,
        "hash_check": not args.skip_hash_check,
        "script_sha256": file_sha256(Path(__file__)),
        "source_manifest_sha256": file_sha256(manifest_path),
    }
    (output_dir / "postprocess_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    checksums = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "output_checksums.csv":
            checksums.append(
                {
                    "file": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    pd.DataFrame(checksums).to_csv(output_dir / "output_checksums.csv", index=False)
    print(f"[done] wrote {len(checksums) + 1} postprocessed artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

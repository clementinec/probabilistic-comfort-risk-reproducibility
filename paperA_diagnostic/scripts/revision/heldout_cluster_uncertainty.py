#!/usr/bin/env python3
"""Contributor-clustered intervals for Paper A's exact held-out tail metrics.

The exact saved predictor bundles and exact random seed/split logic from the
baseline validation are reused. Bootstrap draws sample contributors, not
individual TSV records. Consequently, all held-out records from a selected
contributor receive the same multiplicity in a draw.

These intervals quantify variation in the composition of the pooled held-out
contributors. They do not convert the record-level split into an external or
contributor-held-out validation.
"""

from __future__ import annotations

import argparse
import __main__
import json
import os
import sys
from pathlib import Path
from typing import Any

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ANALYSIS_ROOT.parents[1]
DRAFTS_ROOT = WORKSPACE_ROOT.parent
PANEL_SCRIPTS = WORKSPACE_ROOT / "paperA_R01/01_baseline_submission/scripts"
DEFAULT_DATA = DRAFTS_ROOT / "TCN/newin_with_bmr.csv"
DEFAULT_MODEL_DIR = (
    WORKSPACE_ROOT
    / "paperA_rebuild/runs/diagnostic_reference_zone_raw_full/models"
)
DEFAULT_OUT = ANALYSIS_ROOT / "outputs/uncertainty_heldout"

os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ANALYSIS_ROOT / ".mplconfig"))
os.environ.setdefault("FC_CACHEDIR", str(ANALYSIS_ROOT / ".mplconfig"))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(PANEL_SCRIPTS.resolve()))
import run_medium_office_diagnostic_panel as panel

__main__.FeatureSpec = panel.FeatureSpec
__main__.PredictorBundle = panel.PredictorBundle

SEED = 20260729
N_BOOT = 5_000
TAIL_INDEX = np.array([0, 1, 5, 6])
TAIL_BIN_EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00])
MODEL_LABELS = [
    "Primary TSV model",
    "No-PMV TSV model",
    "PMV-only calibrated tail baseline",
]
METRICS = [
    "observed_tail_prevalence",
    "mean_predicted_p_tail",
    "calibration_gap_predicted_minus_observed",
    "tail_brier",
    "tail_ece_fixed_bins",
    "tail_auroc",
    "tail_average_precision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_exact_split(
    path: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    columns = [
        "thermal_sensation",
        "ta",
        "mean_radiant_temperature",
        "vel",
        "rh",
        "metabolic_rate",
        "clothing_insulation",
        "height_cm",
        "weight_kg",
        "bsa_m2",
        "outdoor_air_temp",
        "prevailing_outdoor_mean",
        "contributor",
    ]
    data = pd.read_csv(path, usecols=columns)
    data["contributor"] = data["contributor"].fillna("Unknown").astype(str)
    y = panel.round_tsv(data["thermal_sensation"])
    train_df, hold_df, y_train, y_hold = train_test_split(
        data,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    cal_df, test_df, y_cal, y_test = train_test_split(
        hold_df,
        y_hold,
        test_size=0.50,
        random_state=42,
        stratify=y_hold,
    )
    fit_df = pd.concat([train_df, cal_df], ignore_index=True)
    y_fit = np.concatenate([y_train, y_cal])
    return (
        fit_df.reset_index(drop=True),
        y_fit,
        test_df.reset_index(drop=True),
        y_test,
    )


def load_predictions(
    fit_df: pd.DataFrame,
    y_fit: np.ndarray,
    test_df: pd.DataFrame,
    model_dir: Path,
) -> dict[str, np.ndarray]:
    primary: panel.PredictorBundle = joblib.load(
        model_dir / "control_predictors.joblib"
    )
    no_pmv: panel.PredictorBundle = joblib.load(
        model_dir / "control_predictors_no_pmv.joblib"
    )
    predictions: dict[str, np.ndarray] = {}
    for label, bundle in [
        ("Primary TSV model", primary),
        ("No-PMV TSV model", no_pmv),
    ]:
        features = panel.build_features_from_raw(test_df, bundle.spec)
        probs = bundle.predict_ordinal(features)
        predictions[label] = probs[:, TAIL_INDEX].sum(axis=1)

    fit_features = panel.build_features_from_raw(fit_df, primary.spec)
    test_features = panel.build_features_from_raw(test_df, primary.spec)
    pmv_fit = pd.to_numeric(fit_features["pmv"], errors="coerce").fillna(0.0)
    pmv_test = pd.to_numeric(test_features["pmv"], errors="coerce").fillna(0.0)
    tail_fit = np.isin(y_fit, TAIL_INDEX).astype(float)
    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    isotonic.fit(np.abs(pmv_fit.to_numpy(float)), tail_fit)
    predictions["PMV-only calibrated tail baseline"] = np.clip(
        isotonic.predict(np.abs(pmv_test.to_numpy(float))),
        0.0,
        1.0,
    )
    return predictions


def fixed_bin_ece(
    y: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    weights = (
        np.ones(len(y), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    total = weights.sum()
    bin_ids = np.digitize(p, TAIL_BIN_EDGES[1:-1], right=False)
    error = 0.0
    for bin_index in range(len(TAIL_BIN_EDGES) - 1):
        mask = bin_ids == bin_index
        bin_weight = weights[mask].sum()
        if bin_weight <= 0:
            continue
        predicted = np.average(p[mask], weights=weights[mask])
        observed = np.average(y[mask], weights=weights[mask])
        error += bin_weight * abs(predicted - observed)
    return float(error / total)


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    observed = float(y.mean())
    predicted = float(p.mean())
    return {
        "observed_tail_prevalence": observed,
        "mean_predicted_p_tail": predicted,
        "calibration_gap_predicted_minus_observed": predicted - observed,
        "tail_brier": float(np.mean((p - y) ** 2)),
        "tail_ece_fixed_bins": fixed_bin_ece(y, p),
        "tail_auroc": float(roc_auc_score(y, p)),
        "tail_average_precision": float(average_precision_score(y, p)),
    }


def cluster_aggregate_matrix(
    cluster_index: np.ndarray,
    values: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    return np.bincount(
        cluster_index,
        weights=np.asarray(values, dtype=float),
        minlength=n_clusters,
    ).astype(float)


def weighted_rank_metrics(
    y: np.ndarray,
    p: np.ndarray,
    row_weights: np.ndarray,
) -> tuple[float, float]:
    """Return weighted AUROC and average precision with exact score-tie handling."""

    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    sorted_w = row_weights[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_p) != 0) + 1]
    positive_by_score = np.add.reduceat(sorted_w * sorted_y, boundaries)
    negative_by_score = np.add.reduceat(sorted_w * (1.0 - sorted_y), boundaries)
    total_positive = positive_by_score.sum()
    total_negative = negative_by_score.sum()
    if total_positive <= 0 or total_negative <= 0:
        return np.nan, np.nan

    cumulative_positive = np.cumsum(positive_by_score)
    cumulative_total = np.cumsum(positive_by_score + negative_by_score)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    average_precision = float(
        np.sum((positive_by_score / total_positive) * precision)
    )

    # With scores ordered high to low, negatives in later groups have lower
    # scores. Reverse cumulative negatives therefore count strict concordance.
    lower_score_negative = (
        np.cumsum(negative_by_score[::-1])[::-1] - negative_by_score
    )
    concordant = np.sum(
        positive_by_score
        * (lower_score_negative + 0.5 * negative_by_score)
    )
    auroc = float(concordant / (total_positive * total_negative))
    return auroc, average_precision


def validate_rank_implementation(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    weights = np.ones(len(y), dtype=float)
    for label, p in predictions.items():
        auroc, ap = weighted_rank_metrics(y, p, weights)
        expected_auc = roc_auc_score(y, p)
        expected_ap = average_precision_score(y, p)
        if not np.isclose(auroc, expected_auc, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"Custom weighted AUROC mismatch for {label}: "
                f"{auroc} versus {expected_auc}"
            )
        if not np.isclose(ap, expected_ap, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"Custom weighted AP mismatch for {label}: {ap} versus {expected_ap}"
            )


def bootstrap_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    clusters: pd.Series,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_codes, cluster_names = pd.factorize(clusters, sort=True)
    n_clusters = len(cluster_names)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(
        n_clusters,
        np.repeat(1.0 / n_clusters, n_clusters),
        size=n_boot,
    ).astype(float)

    cluster_n = cluster_aggregate_matrix(
        cluster_codes, np.ones(len(y)), n_clusters
    )
    cluster_positive = cluster_aggregate_matrix(cluster_codes, y, n_clusters)
    denominator = multiplicities @ cluster_n
    prevalence_draws = (multiplicities @ cluster_positive) / denominator

    draw_data: dict[str, Any] = {"draw": np.arange(n_boot, dtype=int)}
    summary_rows: list[dict[str, Any]] = []

    for label in MODEL_LABELS:
        p = predictions[label]
        cluster_p = cluster_aggregate_matrix(cluster_codes, p, n_clusters)
        cluster_brier = cluster_aggregate_matrix(
            cluster_codes, (p - y) ** 2, n_clusters
        )
        predicted_draws = (multiplicities @ cluster_p) / denominator
        brier_draws = (multiplicities @ cluster_brier) / denominator

        bin_ids = np.digitize(p, TAIL_BIN_EDGES[1:-1], right=False)
        n_bins = len(TAIL_BIN_EDGES) - 1
        bin_n = np.zeros((n_clusters, n_bins), dtype=float)
        bin_p = np.zeros((n_clusters, n_bins), dtype=float)
        bin_y = np.zeros((n_clusters, n_bins), dtype=float)
        for bin_index in range(n_bins):
            mask = bin_ids == bin_index
            bin_n[:, bin_index] = cluster_aggregate_matrix(
                cluster_codes[mask], np.ones(mask.sum()), n_clusters
            )
            bin_p[:, bin_index] = cluster_aggregate_matrix(
                cluster_codes[mask], p[mask], n_clusters
            )
            bin_y[:, bin_index] = cluster_aggregate_matrix(
                cluster_codes[mask], y[mask], n_clusters
            )
        boot_bin_n = multiplicities @ bin_n
        boot_bin_p = multiplicities @ bin_p
        boot_bin_y = multiplicities @ bin_y
        predicted_by_bin = np.divide(
            boot_bin_p,
            boot_bin_n,
            out=np.zeros_like(boot_bin_p),
            where=boot_bin_n > 0,
        )
        observed_by_bin = np.divide(
            boot_bin_y,
            boot_bin_n,
            out=np.zeros_like(boot_bin_y),
            where=boot_bin_n > 0,
        )
        ece_draws = (
            boot_bin_n * np.abs(predicted_by_bin - observed_by_bin)
        ).sum(axis=1) / denominator

        auroc_draws = np.empty(n_boot, dtype=float)
        ap_draws = np.empty(n_boot, dtype=float)
        for draw_index in range(n_boot):
            row_weights = multiplicities[draw_index, cluster_codes]
            auroc_draws[draw_index], ap_draws[draw_index] = weighted_rank_metrics(
                y,
                p,
                row_weights,
            )

        model_draws = {
            "observed_tail_prevalence": prevalence_draws,
            "mean_predicted_p_tail": predicted_draws,
            "calibration_gap_predicted_minus_observed": (
                predicted_draws - prevalence_draws
            ),
            "tail_brier": brier_draws,
            "tail_ece_fixed_bins": ece_draws,
            "tail_auroc": auroc_draws,
            "tail_average_precision": ap_draws,
        }
        point = point_metrics(y, p)
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        for metric, values in model_draws.items():
            draw_data[f"{safe_label}__{metric}"] = values
            finite = values[np.isfinite(values)]
            lower, upper = np.quantile(finite, [0.025, 0.975])
            summary_rows.append(
                {
                    "predictor": label,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower_2_5": float(lower),
                    "ci_upper_97_5": float(upper),
                    "bootstrap_mean": float(finite.mean()),
                    "bootstrap_sd": float(finite.std(ddof=1)),
                    "n_bootstrap_valid": int(len(finite)),
                    "resampling_unit": "contributor",
                    "n_resampling_units": int(n_clusters),
                }
            )

    draws = pd.DataFrame(draw_data)
    summary = pd.DataFrame(summary_rows)

    inventory = (
        pd.DataFrame(
            {
                "contributor": clusters.to_numpy(),
                "observed_tail": y,
            }
        )
        .groupby("contributor", sort=True)
        .agg(
            n_test=("observed_tail", "size"),
            observed_tail_count=("observed_tail", "sum"),
            observed_tail_prevalence=("observed_tail", "mean"),
        )
        .reset_index()
    )
    return summary, draws, inventory


def paired_contrasts(
    point_summary: pd.DataFrame,
    draws: pd.DataFrame,
) -> pd.DataFrame:
    point = point_summary.set_index(["predictor", "metric"])["estimate"]
    labels = {
        label: (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        for label in MODEL_LABELS
    }
    rows = []
    for comparator in MODEL_LABELS[1:]:
        for metric in METRICS:
            first_col = f"{labels[MODEL_LABELS[0]]}__{metric}"
            second_col = f"{labels[comparator]}__{metric}"
            values = draws[first_col].to_numpy(float) - draws[
                second_col
            ].to_numpy(float)
            finite = values[np.isfinite(values)]
            lower, upper = np.quantile(finite, [0.025, 0.975])
            rows.append(
                {
                    "contrast": f"{MODEL_LABELS[0]} minus {comparator}",
                    "metric": metric,
                    "estimate": float(
                        point.loc[(MODEL_LABELS[0], metric)]
                        - point.loc[(comparator, metric)]
                    ),
                    "ci_lower_2_5": float(lower),
                    "ci_upper_97_5": float(upper),
                    "bootstrap_mean": float(finite.mean()),
                    "bootstrap_sd": float(finite.std(ddof=1)),
                    "n_bootstrap_valid": int(len(finite)),
                    "direction_note": (
                        "negative favors first predictor"
                        if metric in {"tail_brier", "tail_ece_fixed_bins"}
                        else (
                            "positive favors first predictor"
                            if metric
                            in {"tail_auroc", "tail_average_precision"}
                            else "raw first-minus-second difference"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def fmt_interval(row: pd.Series) -> str:
    return (
        f"{row['estimate']:.3f} "
        f"[{row['ci_lower_2_5']:.3f}, {row['ci_upper_97_5']:.3f}]"
    )


def write_summary(
    out_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    n_boot: int,
    seed: int,
    n_test: int,
    n_clusters: int,
) -> None:
    lookup = summary.set_index(["predictor", "metric"])
    comp = contrasts.set_index(["contrast", "metric"])
    primary_vs_pmv = (
        "Primary TSV model minus PMV-only calibrated tail baseline"
    )
    primary_vs_no_pmv = "Primary TSV model minus No-PMV TSV model"
    lines = [
        "# Contributor-clustered held-out uncertainty",
        "",
        f"- Exact held-out records: {n_test:,}",
        f"- Contributor clusters: {n_clusters}",
        f"- Bootstrap draws: {n_boot:,}",
        f"- Seed: `{seed}`",
        "- Intervals: empirical 2.5th and 97.5th percentiles.",
        (
            "- Resampling: contributors are sampled with replacement; every "
            "held-out record belonging to a selected contributor receives the "
            "same multiplicity."
        ),
        "",
        "## Metrics",
        "",
        "| Predictor | Brier | ECE | AUROC | Average precision | Observed prevalence | Mean predicted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in MODEL_LABELS:
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                [
                    fmt_interval(lookup.loc[(label, "tail_brier")]),
                    fmt_interval(lookup.loc[(label, "tail_ece_fixed_bins")]),
                    fmt_interval(lookup.loc[(label, "tail_auroc")]),
                    fmt_interval(lookup.loc[(label, "tail_average_precision")]),
                    fmt_interval(
                        lookup.loc[(label, "observed_tail_prevalence")]
                    ),
                    fmt_interval(lookup.loc[(label, "mean_predicted_p_tail")]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired differences",
            "",
            (
                "- Primary minus PMV-only Brier (negative favors primary): "
                f"{fmt_interval(comp.loc[(primary_vs_pmv, 'tail_brier')])}."
            ),
            (
                "- Primary minus PMV-only AUROC (positive favors primary): "
                f"{fmt_interval(comp.loc[(primary_vs_pmv, 'tail_auroc')])}."
            ),
            (
                "- Primary minus PMV-only average precision (positive favors "
                "primary): "
                f"{fmt_interval(comp.loc[(primary_vs_pmv, 'tail_average_precision')])}."
            ),
            (
                "- Primary minus no-PMV Brier: "
                f"{fmt_interval(comp.loc[(primary_vs_no_pmv, 'tail_brier')])}."
            ),
            (
                "- Primary minus no-PMV AUROC: "
                f"{fmt_interval(comp.loc[(primary_vs_no_pmv, 'tail_auroc')])}."
            ),
            (
                "- Primary minus no-PMV average precision: "
                f"{fmt_interval(comp.loc[(primary_vs_no_pmv, 'tail_average_precision')])}."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "The original split is stratified at the record level, so the "
                "same contributors can occur in fitting and held-out partitions. "
                "Contributor-clustered intervals prevent repeated votes from being "
                "treated as independent for interval calculation, but they do not "
                "establish contributor-level transport. The separate grouped and "
                "cross-corpus validations address that harder question."
            ),
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_df, y_fit, test_df, y_test_class = load_exact_split(args.data)
    y_test = np.isin(y_test_class, TAIL_INDEX).astype(float)
    predictions = load_predictions(fit_df, y_fit, test_df, args.model_dir)
    validate_rank_implementation(y_test, predictions)

    summary, draws, inventory = bootstrap_metrics(
        y_test,
        predictions,
        test_df["contributor"],
        args.n_bootstrap,
        args.seed,
    )
    contrasts = paired_contrasts(summary, draws)

    # Reproducibility guard against the already-published exact-split outputs.
    expected = {
        "Primary TSV model": {
            "tail_brier": 0.13735693564675405,
            "tail_ece_fixed_bins": 0.010640069427200572,
            "tail_auroc": 0.747432209128543,
            "tail_average_precision": 0.4870558039038029,
        },
        "No-PMV TSV model": {
            "tail_brier": 0.1376973799887932,
            "tail_ece_fixed_bins": 0.006499845168288775,
            "tail_auroc": 0.7466148413476872,
            "tail_average_precision": 0.4826375541985294,
        },
        "PMV-only calibrated tail baseline": {
            "tail_brier": 0.16085568132567135,
            "tail_ece_fixed_bins": 0.0032210231062228356,
            "tail_auroc": 0.5706889723494039,
            "tail_average_precision": 0.25996345411018135,
        },
    }
    lookup = summary.set_index(["predictor", "metric"])["estimate"]
    for label, values in expected.items():
        for metric, value in values.items():
            actual = lookup.loc[(label, metric)]
            if not np.isclose(actual, value, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"Exact-split guard failed for {label} {metric}: "
                    f"{actual} versus {value}"
                )

    summary.to_csv(args.output_dir / "heldout_clustered_intervals.csv", index=False)
    contrasts.to_csv(
        args.output_dir / "heldout_paired_model_contrasts.csv", index=False
    )
    inventory.to_csv(
        args.output_dir / "heldout_contributor_inventory.csv", index=False
    )
    draws.to_csv(
        args.output_dir / "heldout_cluster_bootstrap_draws.csv.gz",
        index=False,
        compression="gzip",
    )
    prediction_frame = pd.DataFrame(
        {
            "test_row": np.arange(len(test_df), dtype=int),
            "contributor": test_df["contributor"],
            "observed_tail": y_test.astype(int),
            **{
                label: values
                for label, values in predictions.items()
            },
        }
    )
    prediction_frame.to_csv(
        args.output_dir / "heldout_tail_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    config = {
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "interval": "percentile_2.5_97.5",
        "data": str(args.data),
        "model_dir": str(args.model_dir),
        "split": {
            "first_stage_test_size": 0.30,
            "second_stage_test_size": 0.50,
            "random_state": 42,
            "stratified_by_rounded_tsv": True,
        },
        "resampling_unit": "contributor",
        "n_test": int(len(test_df)),
        "n_contributors": int(test_df["contributor"].nunique()),
        "tail_classes": [-3, -2, 2, 3],
        "ece_bin_edges": TAIL_BIN_EDGES.tolist(),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(
        args.output_dir,
        summary,
        contrasts,
        args.n_bootstrap,
        args.seed,
        len(test_df),
        test_df["contributor"].nunique(),
    )
    print(args.output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

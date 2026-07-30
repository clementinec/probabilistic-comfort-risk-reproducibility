#!/usr/bin/env python3
"""Contributor-clustered intervals for the endpoint-only TSV sensitivity.

The event is observed TSV in {-3,+3}. It is intentionally analyzed separately
from the primary broad outer-category event in {-3,-2,+2,+3}. This script reuses
the exact held-out split, saved predictors, cluster-bootstrap implementation, and
fixed seed from ``heldout_cluster_uncertainty.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import heldout_cluster_uncertainty as base


DEFAULT_OUT = base.REPO_ROOT / "outputs" / "uncertainty" / "heldout_endpoint"
ENDPOINT_INDEX = np.array([0, 6])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=base.DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=base.DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=base.N_BOOT)
    parser.add_argument("--seed", type=int, default=base.SEED)
    return parser.parse_args()


def endpoint_predictions(
    fit_df: pd.DataFrame,
    y_fit: np.ndarray,
    test_df: pd.DataFrame,
    model_dir: Path,
) -> dict[str, np.ndarray]:
    primary: base.panel.PredictorBundle = joblib.load(
        model_dir / "tsv_predictor_bundle.joblib"
    )
    no_pmv: base.panel.PredictorBundle = joblib.load(
        model_dir / "tsv_predictor_bundle_no_pmv.joblib"
    )
    predictions: dict[str, np.ndarray] = {}
    for label, bundle in [
        ("Primary TSV model", primary),
        ("No-PMV TSV model", no_pmv),
    ]:
        features = base.panel.build_features_from_raw(test_df, bundle.spec)
        probs = bundle.predict_ordinal(features)
        predictions[label] = probs[:, ENDPOINT_INDEX].sum(axis=1)

    fit_features = base.panel.build_features_from_raw(fit_df, primary.spec)
    test_features = base.panel.build_features_from_raw(test_df, primary.spec)
    pmv_fit = pd.to_numeric(fit_features["pmv"], errors="coerce").fillna(0.0)
    pmv_test = pd.to_numeric(test_features["pmv"], errors="coerce").fillna(0.0)
    endpoint_fit = np.isin(y_fit, ENDPOINT_INDEX).astype(float)
    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    isotonic.fit(np.abs(pmv_fit.to_numpy(float)), endpoint_fit)
    predictions["PMV-only calibrated tail baseline"] = np.clip(
        isotonic.predict(np.abs(pmv_test.to_numpy(float))),
        0.0,
        1.0,
    )
    return predictions


def fmt_interval(row: pd.Series) -> str:
    return (
        f"{row['estimate']:.3f} "
        f"[{row['ci_lower_2_5']:.3f}, {row['ci_upper_97_5']:.3f}]"
    )


def write_endpoint_summary(
    output_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    n_test: int,
    n_contributors: int,
    endpoint_support: int,
    cold_support: int,
    hot_support: int,
) -> None:
    lookup = summary.set_index(["predictor", "metric"])
    comparison = contrasts.set_index(["contrast", "metric"])
    primary_vs_pmv = (
        "Primary TSV model minus PMV-only calibrated tail baseline"
    )
    primary_vs_no_pmv = "Primary TSV model minus No-PMV TSV model"
    lines = [
        "# Contributor-clustered endpoint-only held-out uncertainty",
        "",
        "- Event: observed or predicted probability of `TSV in {-3,+3}`.",
        f"- Exact held-out records: {n_test:,}",
        (
            f"- Observed endpoint records: {endpoint_support:,} "
            f"(`TSV=-3`: {cold_support:,}; `TSV=+3`: {hot_support:,})"
        ),
        f"- Contributor clusters: {n_contributors}",
        f"- Bootstrap draws: {n_bootstrap:,}",
        f"- Seed: `{seed}`",
        (
            "- Interpretation: support-limited bounding sensitivity; not a "
            "replacement outcome and not an occupant-dissatisfaction rate."
        ),
        "",
        "## Metrics",
        "",
        "| Predictor | Brier | ECE | AUROC | Average precision | Observed prevalence | Mean predicted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in base.MODEL_LABELS:
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                [
                    fmt_interval(lookup.loc[(label, "tail_brier")]),
                    fmt_interval(lookup.loc[(label, "tail_ece_fixed_bins")]),
                    fmt_interval(lookup.loc[(label, "tail_auroc")]),
                    fmt_interval(
                        lookup.loc[(label, "tail_average_precision")]
                    ),
                    fmt_interval(
                        lookup.loc[(label, "observed_tail_prevalence")]
                    ),
                    fmt_interval(
                        lookup.loc[(label, "mean_predicted_p_tail")]
                    ),
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
                f"{fmt_interval(comparison.loc[(primary_vs_pmv, 'tail_brier')])}."
            ),
            (
                "- Primary minus PMV-only AUROC (positive favors primary): "
                f"{fmt_interval(comparison.loc[(primary_vs_pmv, 'tail_auroc')])}."
            ),
            (
                "- Primary minus PMV-only average precision (positive favors "
                "primary): "
                f"{fmt_interval(comparison.loc[(primary_vs_pmv, 'tail_average_precision')])}."
            ),
            (
                "- Primary minus no-PMV Brier: "
                f"{fmt_interval(comparison.loc[(primary_vs_no_pmv, 'tail_brier')])}."
            ),
            (
                "- Primary minus no-PMV AUROC: "
                f"{fmt_interval(comparison.loc[(primary_vs_no_pmv, 'tail_auroc')])}."
            ),
            (
                "- Primary minus no-PMV average precision: "
                f"{fmt_interval(comparison.loc[(primary_vs_no_pmv, 'tail_average_precision')])}."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "The endpoints have materially less support than the primary "
                "outer-category event. Contributor clustering addresses repeated "
                "records within the test set but does not propagate model-refit "
                "uncertainty or establish transport to a new corpus."
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
    fit_df, y_fit, test_df, y_test_class = base.load_exact_split(args.data)
    y_test = np.isin(y_test_class, ENDPOINT_INDEX).astype(float)
    endpoint_support = int(y_test.sum())
    cold_support = int(np.sum(y_test_class == 0))
    hot_support = int(np.sum(y_test_class == 6))
    if endpoint_support != 1_350:
        raise ValueError(
            f"Exact-split endpoint support guard failed: {endpoint_support} != 1350"
        )
    predictions = endpoint_predictions(
        fit_df,
        y_fit,
        test_df,
        args.model_dir,
    )
    base.validate_rank_implementation(y_test, predictions)
    summary, draws, inventory = base.bootstrap_metrics(
        y_test,
        predictions,
        test_df["contributor"],
        args.n_bootstrap,
        args.seed,
    )
    contrasts = base.paired_contrasts(summary, draws)

    summary.to_csv(
        args.output_dir / "endpoint_clustered_intervals.csv",
        index=False,
    )
    contrasts.to_csv(
        args.output_dir / "endpoint_paired_model_contrasts.csv",
        index=False,
    )
    inventory.to_csv(
        args.output_dir / "endpoint_contributor_inventory.csv",
        index=False,
    )
    draws.to_csv(
        args.output_dir / "endpoint_cluster_bootstrap_draws.csv.gz",
        index=False,
        compression="gzip",
    )
    prediction_frame = pd.DataFrame(
        {
            "test_row": np.arange(len(test_df), dtype=int),
            "contributor": test_df["contributor"],
            "observed_endpoint": y_test.astype(int),
            **predictions,
        }
    )
    prediction_frame.to_csv(
        args.output_dir / "endpoint_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "interval": "percentile_2.5_97.5",
                "data": str(args.data),
                "model_dir": str(args.model_dir),
                "resampling_unit": "contributor",
                "n_test": int(len(test_df)),
                "n_contributors": int(test_df["contributor"].nunique()),
                "event_classes": [-3, 3],
                "observed_endpoint_support": endpoint_support,
                "observed_cold_endpoint_support": cold_support,
                "observed_hot_endpoint_support": hot_support,
                "ece_bin_edges": base.TAIL_BIN_EDGES.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_endpoint_summary(
        args.output_dir,
        summary,
        contrasts,
        args.n_bootstrap,
        args.seed,
        len(test_df),
        test_df["contributor"].nunique(),
        endpoint_support,
        cold_support,
        hot_support,
    )
    print(args.output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

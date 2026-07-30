#!/usr/bin/env python3
"""Exact-spec contributor-grouped validation for Paper A revision 01.

This revision-only script matches the deployed 400-tree model configuration
and defines a positive diagnostic screen as p_tail >= 0.20.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
R01_ROOT = ANALYSIS_ROOT.parent
DRAFTS_ROOT = R01_ROOT.parents[1]
PANEL_SCRIPTS = R01_ROOT / "01_baseline_submission" / "scripts"
DEFAULT_DATA = DRAFTS_ROOT / "TCN" / "newin_with_bmr.csv"
DEFAULT_OUT = ANALYSIS_ROOT / "outputs" / "grouped_tsv_validation_exact"

os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ANALYSIS_ROOT / ".mplconfig"))
os.environ.setdefault("FC_CACHEDIR", str(ANALYSIS_ROOT / ".mplconfig"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(PANEL_SCRIPTS.resolve()))
import run_medium_office_diagnostic_panel as panel

TAIL_INDEX = np.array([0, 1, 5, 6])
TAIL_BIN_EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--group-column", default="contributor")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--calibration-size", type=float, default=0.1765)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def read_data(path: Path, group_column: str) -> pd.DataFrame:
    cols = [
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
        group_column,
    ]
    df = pd.read_csv(path, usecols=lambda col: col in cols)
    df[group_column] = df[group_column].fillna("Unknown").astype(str)
    return df.reset_index(drop=True)


def contributor_split(
    df: pd.DataFrame,
    y: np.ndarray,
    group_col: str,
    seed: int,
    test_size: float,
    calibration_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = df[group_col].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_cal_idx, test_idx = next(outer.split(df, y, groups))
    train_cal = df.iloc[train_cal_idx]
    y_train_cal = y[train_cal_idx]
    groups_train_cal = train_cal[group_col].to_numpy()
    inner = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=seed + 10_000)
    train_rel, cal_rel = next(inner.split(train_cal, y_train_cal, groups_train_cal))
    train_idx = train_cal_idx[train_rel]
    cal_idx = train_cal_idx[cal_rel]
    return train_idx, cal_idx, test_idx


def fit_ordinal(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    y_train: np.ndarray,
    y_cal: np.ndarray,
    feature_columns: list[str],
    n_estimators: int,
) -> tuple[panel.FeatureSpec, StandardScaler, list[CalibratedClassifierCV]]:
    spec = panel.fit_feature_spec(train_df)
    x_train_df = panel.build_features_from_raw(train_df, spec)
    x_cal_df = panel.build_features_from_raw(cal_df, spec)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df[feature_columns].to_numpy(float))
    x_cal = scaler.transform(x_cal_df[feature_columns].to_numpy(float))
    common = dict(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    models: list[CalibratedClassifierCV] = []
    for threshold in range(6):
        base = lgb.LGBMClassifier(objective="binary", **common)
        base.fit(x_train, (y_train > threshold).astype(int))
        cal = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv="prefit",
            ensemble=False,
        )
        cal.fit(x_cal, (y_cal > threshold).astype(int))
        models.append(cal)
    return spec, scaler, models


def predict_ordinal(
    test_df: pd.DataFrame,
    spec: panel.FeatureSpec,
    scaler: StandardScaler,
    models: list[CalibratedClassifierCV],
    feature_columns: list[str],
) -> np.ndarray:
    x_test_df = panel.build_features_from_raw(test_df, spec)
    x = scaler.transform(x_test_df[feature_columns].to_numpy(float))
    cumulative = []
    for model in models:
        cumulative.append(model.predict_proba(x)[:, 1])
    p_gt = np.column_stack(cumulative)
    p_gt = np.minimum.accumulate(np.clip(p_gt, 0.0, 1.0), axis=1)
    probs = np.empty((x.shape[0], 7), dtype=float)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    probs[:, 1:6] = p_gt[:, :-1] - p_gt[:, 1:]
    probs[:, 6] = p_gt[:, 5]
    return panel.normalize_probs(probs)


def ece_bins(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    tail_true = np.isin(y_true, TAIL_INDEX).astype(float)
    tail_prob = probs[:, TAIL_INDEX].sum(axis=1)
    bin_ids = np.digitize(tail_prob, TAIL_BIN_EDGES[1:-1], right=False)
    rows = []
    for i in range(len(TAIL_BIN_EDGES) - 1):
        mask = bin_ids == i
        if not mask.any():
            continue
        pred = float(tail_prob[mask].mean())
        obs = float(tail_true[mask].mean())
        rows.append(
            {
                "bin_left": float(TAIL_BIN_EDGES[i]),
                "bin_right": float(TAIL_BIN_EDGES[i + 1]),
                "n": int(mask.sum()),
                "predicted_p_tail_mean": pred,
                "observed_tail_frequency": obs,
                "absolute_calibration_error": abs(pred - obs),
            }
        )
    bins = pd.DataFrame(rows)
    weights = bins["n"].to_numpy(float) / float(len(y_true))
    abs_err = bins["absolute_calibration_error"].to_numpy(float)
    return float(np.sum(weights * abs_err)), float(abs_err.max()), bins


def summarize(
    split_id: int,
    feature_set: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    group_counts: dict[str, Any],
) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    tail_true = np.isin(y_true, TAIL_INDEX)
    tail_prob = probs[:, TAIL_INDEX].sum(axis=1)
    tail_pred = tail_prob >= 0.20
    ece, mce, _ = ece_bins(y_true, probs)
    row: dict[str, Any] = {
        "split": split_id,
        "feature_set": feature_set,
        "n_test": int(len(y_true)),
        "test_groups": int(group_counts["test"]),
        "exact_accuracy": float(accuracy_score(y_true, pred)),
        "within_one_class_accuracy": float(np.mean(np.abs(pred - y_true) <= 1)),
        "ordinal_mae_classes": float(np.mean(np.abs((pred - 3) - (y_true - 3)))),
        "tail_precision": float(precision_score(tail_true, tail_pred, zero_division=0)),
        "tail_recall": float(recall_score(tail_true, tail_pred, zero_division=0)),
        "tail_f1": float(f1_score(tail_true, tail_pred, zero_division=0)),
        "tail_f1_screen": 0.20,
        "tail_auroc": float(roc_auc_score(tail_true, tail_prob)),
        "tail_average_precision": float(average_precision_score(tail_true, tail_prob)),
        "log_loss": float(log_loss(y_true, probs, labels=np.arange(7))),
        "mean_predicted_p_tail": float(tail_prob.mean()),
        "observed_tail_frequency": float(tail_true.mean()),
        "tail_brier": float(np.mean((tail_prob - tail_true.astype(float)) ** 2)),
        "tail_ece_fixed_bins": ece,
        "tail_mce_fixed_bins": mce,
    }
    return row


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_test",
        "test_groups",
        "exact_accuracy",
        "within_one_class_accuracy",
        "ordinal_mae_classes",
        "tail_f1",
        "tail_auroc",
        "tail_average_precision",
        "log_loss",
        "mean_predicted_p_tail",
        "observed_tail_frequency",
        "tail_brier",
        "tail_ece_fixed_bins",
        "tail_mce_fixed_bins",
    ]
    rows = []
    for feature_set, sdf in summary.groupby("feature_set", sort=False):
        row: dict[str, Any] = {"feature_set": feature_set, "n_splits": int(len(sdf))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sdf[metric].mean())
            row[f"{metric}_sd"] = float(sdf[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex(agg: pd.DataFrame, out_dir: Path) -> None:
    table = agg.copy()
    cols = [
        ("exact_accuracy", "Exact (\\%)", True),
        ("within_one_class_accuracy", "Within $\\pm1$ (\\%)", True),
        ("ordinal_mae_classes", "Class MAE", False),
        ("tail_f1", "Tail F1@0.20 (\\%)", True),
        ("log_loss", "Log loss", False),
        ("tail_ece_fixed_bins", "Tail ECE", False),
    ]
    out = pd.DataFrame({"Feature set": table["feature_set"].map({"full": "Primary", "no_pmv": "No-PMV"})})
    for base, label, pct in cols:
        def fmt(row: pd.Series) -> str:
            mean = row[f"{base}_mean"]
            sd = row[f"{base}_sd"]
            if pct:
                return f"{100 * mean:.1f} ({100 * sd:.1f})"
            return f"{mean:.3f} ({sd:.3f})"

        out[label] = table.apply(fmt, axis=1)
    (out_dir / "grouped_validation_summary_table.tex").write_text(
        out.to_latex(index=False, escape=False, column_format="lrrrrrr"),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = read_data(args.data, args.group_column)
    y = panel.round_tsv(df["thermal_sensation"])
    summary_rows = []
    split_inventory = []
    for split_id in range(args.n_splits):
        train_idx, cal_idx, test_idx = contributor_split(
            df,
            y,
            args.group_column,
            seed=42 + split_id,
            test_size=args.test_size,
            calibration_size=args.calibration_size,
        )
        group_counts = {
            "train": df.iloc[train_idx][args.group_column].nunique(),
            "calibration": df.iloc[cal_idx][args.group_column].nunique(),
            "test": df.iloc[test_idx][args.group_column].nunique(),
        }
        split_inventory.append(
            {
                "split": split_id,
                "n_train": int(len(train_idx)),
                "n_calibration": int(len(cal_idx)),
                "n_test": int(len(test_idx)),
                "train_groups": int(group_counts["train"]),
                "calibration_groups": int(group_counts["calibration"]),
                "test_groups": int(group_counts["test"]),
            }
        )
        for feature_set, feature_columns in [
            ("full", panel.FEATURE_COLUMNS_FULL),
            ("no_pmv", panel.FEATURE_COLUMNS_NO_PMV),
        ]:
            print(f"[fit] split={split_id} feature_set={feature_set}")
            spec, scaler, models = fit_ordinal(
                df.iloc[train_idx].reset_index(drop=True),
                df.iloc[cal_idx].reset_index(drop=True),
                y[train_idx],
                y[cal_idx],
                feature_columns,
                args.n_estimators,
            )
            probs = predict_ordinal(
                df.iloc[test_idx].reset_index(drop=True),
                spec,
                scaler,
                models,
                feature_columns,
            )
            summary_rows.append(
                summarize(split_id, feature_set, y[test_idx], probs, group_counts)
            )
    summary = pd.DataFrame(summary_rows)
    agg = aggregate(summary)
    inv = pd.DataFrame(split_inventory)
    summary.to_csv(args.output_dir / "grouped_validation_split_summary.csv", index=False)
    agg.to_csv(args.output_dir / "grouped_validation_aggregate_summary.csv", index=False)
    inv.to_csv(args.output_dir / "grouped_validation_split_inventory.csv", index=False)
    (args.output_dir / "grouped_validation_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_latex(agg, args.output_dir)
    print(f"[write] {args.output_dir / 'grouped_validation_split_summary.csv'}")
    print(f"[write] {args.output_dir / 'grouped_validation_aggregate_summary.csv'}")
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

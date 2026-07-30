#!/usr/bin/env python3
"""Overlap-excluded reciprocal source-holdout validation for Paper A revision 01.

For each direction (ASHRAE -> China and China -> ASHRAE), the source corpus is
split by TSV-stratified sampling into 82.3529% model fitting and 17.6471%
isotonic calibration. The target corpus remains wholly held out: it is not
used for preprocessing, model fitting, calibration, or model selection.

Before either directional fit, six city-year strata with plausible study
overlap between the source-labelled corpora are removed from both corpora:
Changsha 2007, Yueyang 2007, Nanyang 2006, and Harbin 2001/2009/2011.

The LightGBM settings, six cumulative ordinal heads, fold-safe preprocessing,
and isotonic calibration match the deployed Paper A pipeline. Both the full
and no-PMV feature sets are evaluated, with nominal counterparts included as
a model-family sensitivity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
R01_ROOT = ANALYSIS_ROOT.parent
DRAFTS_ROOT = R01_ROOT.parents[1]
PANEL_SCRIPTS = R01_ROOT / "01_baseline_submission" / "scripts"
DEFAULT_DATA = DRAFTS_ROOT / "TCN" / "newin_with_bmr.csv"
DEFAULT_OUT = ANALYSIS_ROOT / "outputs" / "cross_corpus_validation_overlap_excluded"

os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ANALYSIS_ROOT / ".mplconfig"))
os.environ.setdefault("FC_CACHEDIR", str(ANALYSIS_ROOT / ".mplconfig"))

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(PANEL_SCRIPTS.resolve()))
import run_medium_office_diagnostic_panel as panel

SOURCE_LEVELS = ("ASHRAE", "China")
DIRECTIONS = (("ASHRAE", "China"), ("China", "ASHRAE"))
TAIL_INDEX = np.array([0, 1, 5, 6])
TAIL_SCREEN = 0.20
TAIL_BIN_EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00])
DEFAULT_CALIBRATION_FRACTION = 3.0 / 17.0
RAW_COLUMNS = [
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
    "source",
    "city",
    "country",
    "year",
]
OVERLAP_CITY_YEARS = (
    ("changsha", 2007),
    ("yueyang", 2007),
    ("nanyang", 2006),
    ("harbin", 2001),
    ("harbin", 2009),
    ("harbin", 2011),
)
MODEL_SETTINGS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": -1,
    "subsample": 0.9,
    "subsample_freq": 0,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help="Fraction of each source corpus reserved for source-only isotonic calibration.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap_stratum_mask(df: pd.DataFrame) -> np.ndarray:
    city = df["city"].astype("string").str.casefold().fillna("")
    year = pd.to_numeric(df["year"], errors="coerce")
    country = df["country"].astype("string").str.casefold().str.strip()
    china_record = (df["source"].eq("China") | country.eq("china")).fillna(False)
    mask = np.zeros(len(df), dtype=bool)
    for city_token, target_year in OVERLAP_CITY_YEARS:
        mask |= (
            china_record.to_numpy(dtype=bool)
            & city.str.contains(city_token, regex=False).fillna(False).to_numpy(dtype=bool)
            & year.eq(target_year).to_numpy(dtype=bool)
        )
    return mask


def read_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, usecols=RAW_COLUMNS)
    df["source"] = df["source"].astype(str).str.strip()
    unexpected = sorted(set(df["source"].dropna().unique()) - set(SOURCE_LEVELS))
    if unexpected:
        raise ValueError(f"Unexpected source values: {unexpected}")
    missing_source = int(df["source"].isna().sum())
    if missing_source:
        raise ValueError(f"{missing_source} records have missing source labels")
    if df["thermal_sensation"].isna().any():
        raise ValueError("thermal_sensation contains missing values")
    excluded_mask = overlap_stratum_mask(df)
    excluded = df.loc[excluded_mask, ["source", "city", "country", "year"]].copy()
    excluded["normalized_overlap_stratum"] = ""
    city = excluded["city"].astype("string").str.casefold().fillna("")
    year = pd.to_numeric(excluded["year"], errors="coerce")
    for city_token, target_year in OVERLAP_CITY_YEARS:
        mask = city.str.contains(city_token, regex=False) & year.eq(target_year)
        excluded.loc[mask, "normalized_overlap_stratum"] = (
            f"{city_token}:{target_year}"
        )
    exclusion_inventory = (
        excluded.groupby(
            ["normalized_overlap_stratum", "source"],
            dropna=False,
        )
        .size()
        .rename("n_excluded")
        .reset_index()
    )
    retained = df.loc[~excluded_mask].reset_index(drop=True)
    return retained, exclusion_inventory


def source_split(
    source_df: pd.DataFrame,
    calibration_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    y = panel.round_tsv(source_df["thermal_sensation"])
    train_df, cal_df, y_train, y_cal = train_test_split(
        source_df,
        y,
        test_size=calibration_fraction,
        random_state=random_state,
        stratify=y,
    )
    return (
        train_df.reset_index(drop=True),
        cal_df.reset_index(drop=True),
        np.asarray(y_train, dtype=int),
        np.asarray(y_cal, dtype=int),
    )


def fit_scaler(
    train_features: pd.DataFrame,
    calibration_features: pd.DataFrame,
    target_features: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features[feature_columns].to_numpy(float))
    x_cal = scaler.transform(calibration_features[feature_columns].to_numpy(float))
    x_target = scaler.transform(target_features[feature_columns].to_numpy(float))
    return scaler, x_train, x_cal, x_target


def common_model_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = dict(MODEL_SETTINGS)
    settings["n_estimators"] = int(args.n_estimators)
    settings["random_state"] = int(args.random_state)
    return settings


def fit_predict_ordinal(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_target: np.ndarray,
    settings: dict[str, Any],
) -> np.ndarray:
    cumulative = []
    for threshold in range(6):
        base = lgb.LGBMClassifier(objective="binary", **settings)
        base.fit(x_train, (y_train > threshold).astype(int))
        calibrated = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv="prefit",
            ensemble=False,
        )
        calibrated.fit(x_cal, (y_cal > threshold).astype(int))
        cumulative.append(calibrated.predict_proba(x_target)[:, 1])
        del calibrated, base
        gc.collect()
    p_gt = np.column_stack(cumulative)
    p_gt = np.minimum.accumulate(np.clip(p_gt, 0.0, 1.0), axis=1)
    probs = np.empty((x_target.shape[0], 7), dtype=float)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    probs[:, 1:6] = p_gt[:, :-1] - p_gt[:, 1:]
    probs[:, 6] = p_gt[:, 5]
    return panel.normalize_probs(probs)


def fit_predict_nominal(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_target: np.ndarray,
    settings: dict[str, Any],
) -> np.ndarray:
    base = lgb.LGBMClassifier(objective="multiclass", num_class=7, **settings)
    base.fit(x_train, y_train)
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv="prefit",
        ensemble=False,
    )
    calibrated.fit(x_cal, y_cal)
    raw = calibrated.predict_proba(x_target)
    probs = panel.normalize_probs(panel.align_nominal_classes(raw, calibrated.classes_))
    del calibrated, base
    gc.collect()
    return probs


def calibration_bins(
    direction: str,
    feature_set: str,
    model: str,
    y_true: np.ndarray,
    tail_prob: np.ndarray,
) -> pd.DataFrame:
    tail_true = np.isin(y_true, TAIL_INDEX).astype(float)
    bin_ids = np.digitize(tail_prob, TAIL_BIN_EDGES[1:-1], right=False)
    rows: list[dict[str, Any]] = []
    for i in range(len(TAIL_BIN_EDGES) - 1):
        mask = bin_ids == i
        if not mask.any():
            continue
        pred = float(tail_prob[mask].mean())
        obs = float(tail_true[mask].mean())
        n = int(mask.sum())
        rows.append(
            {
                "direction": direction,
                "feature_set": feature_set,
                "model": model,
                "bin_left": float(TAIL_BIN_EDGES[i]),
                "bin_right": float(TAIL_BIN_EDGES[i + 1]),
                "n": n,
                "predicted_p_tail_mean": pred,
                "observed_tail_frequency": obs,
                "absolute_calibration_error": abs(pred - obs),
                "binomial_se": float(np.sqrt(max(obs * (1.0 - obs), 0.0) / n)),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    direction: str,
    source_train: str,
    target_test: str,
    feature_set: str,
    model: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    bins: pd.DataFrame,
    elapsed_seconds: float,
) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    tail_true = np.isin(y_true, TAIL_INDEX)
    tail_prob = probs[:, TAIL_INDEX].sum(axis=1)
    tail_pred = tail_prob >= TAIL_SCREEN
    weights = bins["n"].to_numpy(float) / float(len(y_true))
    abs_error = bins["absolute_calibration_error"].to_numpy(float)
    return {
        "direction": direction,
        "source_train": source_train,
        "target_test": target_test,
        "feature_set": feature_set,
        "model": model,
        "n_target_test": int(len(y_true)),
        "exact_accuracy": float(accuracy_score(y_true, pred)),
        "within_one_class_accuracy": float(np.mean(np.abs(pred - y_true) <= 1)),
        "ordinal_mae_classes": float(np.mean(np.abs(pred - y_true))),
        "log_loss": float(log_loss(y_true, probs, labels=np.arange(7))),
        "observed_tail_prevalence": float(tail_true.mean()),
        "mean_predicted_p_tail": float(tail_prob.mean()),
        "predicted_high_tail_screen_rate": float(tail_pred.mean()),
        "tail_brier": float(np.mean((tail_prob - tail_true.astype(float)) ** 2)),
        "tail_ece_fixed_bins": float(np.sum(weights * abs_error)),
        "tail_mce_fixed_bins": float(abs_error.max()),
        "tail_auroc": float(roc_auc_score(tail_true, tail_prob)),
        "tail_average_precision": float(average_precision_score(tail_true, tail_prob)),
        "tail_precision_at_0_20": float(precision_score(tail_true, tail_pred, zero_division=0)),
        "tail_recall_at_0_20": float(recall_score(tail_true, tail_pred, zero_division=0)),
        "tail_f1_at_0_20": float(f1_score(tail_true, tail_pred, zero_division=0)),
        "elapsed_seconds": float(elapsed_seconds),
    }


def markdown_table(metrics: pd.DataFrame) -> str:
    header = (
        "| Direction | Features | Model | Exact | Within ±1 | MAE | Log loss | "
        "Tail prev. | Mean p_tail | Brier | ECE | AUROC | AP | F1@0.20 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for row in metrics.itertuples(index=False):
        rows.append(
            f"| {row.direction} | {row.feature_set} | {row.model} "
            f"| {100 * row.exact_accuracy:.1f}% "
            f"| {100 * row.within_one_class_accuracy:.1f}% "
            f"| {row.ordinal_mae_classes:.3f} "
            f"| {row.log_loss:.3f} "
            f"| {100 * row.observed_tail_prevalence:.1f}% "
            f"| {row.mean_predicted_p_tail:.3f} "
            f"| {row.tail_brier:.3f} "
            f"| {row.tail_ece_fixed_bins:.3f} "
            f"| {row.tail_auroc:.3f} "
            f"| {row.tail_average_precision:.3f} "
            f"| {100 * row.tail_f1_at_0_20:.1f}% |"
        )
    return "\n".join(rows)


def write_summary(
    metrics: pd.DataFrame,
    inventory: pd.DataFrame,
    exclusion_inventory: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Overlap-excluded reciprocal source-holdout validation",
        "",
        "Each retained target corpus was held out in full after the prespecified "
        "overlap exclusions. Preprocessing statistics, feature scaling, LightGBM "
        "fitting, and isotonic calibration used only the named source corpus. There "
        "was no target-source recalibration or fine-tuning.",
        "",
        markdown_table(metrics),
        "",
        "## Split inventory",
        "",
        "| Direction | Source total | Train | Calibration | Target test | "
        "Neutral exact | Neutral within ±1 | Neutral MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in inventory.itertuples(index=False):
        lines.append(
            f"| {row.direction} | {row.n_source_total:,} | {row.n_train:,} "
            f"| {row.n_calibration:,} | {row.n_target_test:,} "
            f"| {100 * row.target_neutral_exact_accuracy:.1f}% "
            f"| {100 * row.target_neutral_within_one_class_accuracy:.1f}% "
            f"| {row.target_neutral_mae_classes:.3f} |"
        )
    lines.extend(
        [
            "",
            "The neutral baseline predicts TSV = 0 for every target record. It is "
            "reported because Within ±1 can be high under a neutral-heavy class "
            "distribution without demonstrating useful transport.",
            "",
            "## Excluded overlap strata",
            "",
            "| Normalized city-year | Source label | Excluded records |",
            "|---|---|---:|",
        ]
    )
    for row in exclusion_inventory.itertuples(index=False):
        lines.append(
            f"| {row.normalized_overlap_stratum} | {row.source} "
            f"| {row.n_excluded:,} |"
        )
    lines.extend(
        [
            "",
            "## Specification",
            "",
            f"- Source split: TSV-stratified, random state {args.random_state}; "
            f"requested train/calibration fractions "
            f"{1.0 - args.calibration_fraction:.6f}/{args.calibration_fraction:.6f}.",
            f"- LightGBM trees per model/head: {args.n_estimators}; learning rate 0.05; "
            "unlimited depth; column fraction 0.9; no effective row bagging "
            "(`subsample=0.9`, `subsample_freq=0`); minimum child samples 20; "
            "L1/L2 regularization 0.",
            "- Ordinal model: six cumulative binary heads, source-only isotonic "
            "calibration, cumulative monotonic repair, and seven-class reconstruction.",
            "- Tail event: observed |TSV| >= 2; diagnostic screen: predicted "
            "p_tail >= 0.20.",
            "- Tail ECE uses fixed probability bins "
            "[0,.05,.10,.15,.20,.30,.40,.60,1.00], matching the Paper A validation.",
            "- The nominal rows are a model-family sensitivity; the ordinal/full row "
            "is the exact primary model specification.",
            "",
            "The six potentially overlapping city-year strata were removed from both "
            "source labels before splitting or evaluation.",
            "",
            "Cross-corpus calibration is intentionally evaluated without access to the "
            "target distribution. Accordingly, mean-probability/prevalence gaps and ECE "
            "measure transport under dataset shift, not an in-domain recalibration result.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex(metrics: pd.DataFrame, out_dir: Path) -> None:
    table = metrics[
        [
            "direction",
            "feature_set",
            "model",
            "exact_accuracy",
            "within_one_class_accuracy",
            "ordinal_mae_classes",
            "log_loss",
            "tail_brier",
            "tail_ece_fixed_bins",
            "tail_auroc",
            "tail_average_precision",
            "tail_f1_at_0_20",
        ]
    ].copy()
    for col in ["exact_accuracy", "within_one_class_accuracy", "tail_f1_at_0_20"]:
        table[col] = table[col].map(lambda value: f"{100 * value:.1f}")
    for col in [
        "ordinal_mae_classes",
        "log_loss",
        "tail_brier",
        "tail_ece_fixed_bins",
        "tail_auroc",
        "tail_average_precision",
    ]:
        table[col] = table[col].map(lambda value: f"{value:.3f}")
    table.columns = [
        "Direction",
        "Features",
        "Model",
        "Exact (\\%)",
        "Within $\\pm1$ (\\%)",
        "MAE",
        "Log loss",
        "Tail Brier",
        "Tail ECE",
        "Tail AUROC",
        "Tail AP",
        "Tail F1@0.20 (\\%)",
    ]
    (out_dir / "cross_corpus_summary_table.tex").write_text(
        table.to_latex(index=False, escape=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must lie strictly between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    print(f"[read] {args.data}")
    data, exclusion_inventory = read_data(args.data)
    data_hash = sha256(args.data)
    exclusion_inventory.to_csv(
        args.output_dir / "overlap_exclusion_inventory.csv",
        index=False,
    )
    settings = common_model_settings(args)
    metrics_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []

    for source_name, target_name in DIRECTIONS:
        direction = f"{source_name}->{target_name}"
        source_df = data.loc[data["source"].eq(source_name)].reset_index(drop=True)
        target_df = data.loc[data["source"].eq(target_name)].reset_index(drop=True)
        y_target = panel.round_tsv(target_df["thermal_sensation"])
        train_df, cal_df, y_train, y_cal = source_split(
            source_df,
            args.calibration_fraction,
            args.random_state,
        )
        inventory_rows.append(
            {
                "direction": direction,
                "source_train": source_name,
                "target_test": target_name,
                "n_source_total": int(len(source_df)),
                "n_train": int(len(train_df)),
                "n_calibration": int(len(cal_df)),
                "n_target_test": int(len(target_df)),
                "actual_train_fraction": float(len(train_df) / len(source_df)),
                "actual_calibration_fraction": float(len(cal_df) / len(source_df)),
                "source_train_tail_prevalence": float(np.isin(y_train, TAIL_INDEX).mean()),
                "source_calibration_tail_prevalence": float(np.isin(y_cal, TAIL_INDEX).mean()),
                "target_tail_prevalence": float(np.isin(y_target, TAIL_INDEX).mean()),
                "target_neutral_exact_accuracy": float(np.mean(y_target == 3)),
                "target_neutral_within_one_class_accuracy": float(
                    np.mean(np.abs(y_target - 3) <= 1)
                ),
                "target_neutral_mae_classes": float(np.mean(np.abs(y_target - 3))),
            }
        )
        print(
            f"[split] {direction}: train={len(train_df):,}, "
            f"calibration={len(cal_df):,}, target={len(target_df):,}"
        )

        spec = panel.fit_feature_spec(train_df)
        train_features = panel.build_features_from_raw(train_df, spec)
        cal_features = panel.build_features_from_raw(cal_df, spec)
        target_features = panel.build_features_from_raw(target_df, spec)

        for feature_set, feature_columns in [
            ("full", panel.FEATURE_COLUMNS_FULL),
            ("no_pmv", panel.FEATURE_COLUMNS_NO_PMV),
        ]:
            _, x_train, x_cal, x_target = fit_scaler(
                train_features,
                cal_features,
                target_features,
                feature_columns,
            )
            for model_name, fit_predict in [
                ("ordinal", fit_predict_ordinal),
                ("nominal", fit_predict_nominal),
            ]:
                print(f"[fit] {direction} / {feature_set} / {model_name}")
                model_started = time.time()
                probs = fit_predict(
                    x_train,
                    y_train,
                    x_cal,
                    y_cal,
                    x_target,
                    settings,
                )
                tail_prob = probs[:, TAIL_INDEX].sum(axis=1)
                bins = calibration_bins(
                    direction,
                    feature_set,
                    model_name,
                    y_target,
                    tail_prob,
                )
                calibration_frames.append(bins)
                metrics_rows.append(
                    summarize(
                        direction,
                        source_name,
                        target_name,
                        feature_set,
                        model_name,
                        y_target,
                        probs,
                        bins,
                        time.time() - model_started,
                    )
                )
                del probs, tail_prob, bins
                gc.collect()
            del x_train, x_cal, x_target
            gc.collect()
        del train_features, cal_features, target_features, source_df, target_df
        gc.collect()

    metrics = pd.DataFrame(metrics_rows)
    inventory = pd.DataFrame(inventory_rows)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "cross_corpus_metrics.csv", index=False)
    inventory.to_csv(args.output_dir / "cross_corpus_split_inventory.csv", index=False)
    calibration.to_csv(args.output_dir / "cross_corpus_calibration_bins.csv", index=False)
    write_summary(metrics, inventory, exclusion_inventory, args.output_dir, args)
    write_latex(metrics, args.output_dir)

    config = {
        "script": str(Path(__file__).resolve()),
        "data": str(args.data.resolve()),
        "data_sha256": data_hash,
        "output_dir": str(args.output_dir.resolve()),
        "directions": [list(direction) for direction in DIRECTIONS],
        "target_source_use": "wholly held out; no target fitting, calibration, or fine-tuning",
        "overlap_exclusion": {
            "city_year_strata": [list(item) for item in OVERLAP_CITY_YEARS],
            "applied_to_both_source_labels": True,
            "inventory_file": str(
                (args.output_dir / "overlap_exclusion_inventory.csv").resolve()
            ),
        },
        "requested_source_train_fraction": 1.0 - args.calibration_fraction,
        "requested_source_calibration_fraction": args.calibration_fraction,
        "random_state": args.random_state,
        "tail_indices_zero_based": TAIL_INDEX.tolist(),
        "tail_event": "abs(TSV) >= 2",
        "tail_screen": TAIL_SCREEN,
        "tail_bin_edges": TAIL_BIN_EDGES.tolist(),
        "feature_sets": {
            "full": panel.FEATURE_COLUMNS_FULL,
            "no_pmv": panel.FEATURE_COLUMNS_NO_PMV,
        },
        "models": ["ordinal", "nominal"],
        "model_settings": settings,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
        "elapsed_seconds": time.time() - run_started,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {args.output_dir / 'cross_corpus_metrics.csv'}")
    print(f"[write] {args.output_dir / 'summary.md'}")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

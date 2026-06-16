#!/usr/bin/env python3
"""Held-out validation for TSV predictors and scalar PMV baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import __main__
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
DEFAULT_DATA = WORKSPACE / "TCN" / "newin_with_bmr.csv"
DEFAULT_MODEL_DIR = ROOT / "runs" / "diagnostic_reference_zone_raw_full" / "models"
DEFAULT_OUT = ROOT / "diagnostics" / "tsv_predictor_validation"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))
os.environ.setdefault("FC_CACHEDIR", str(ROOT / ".mplconfig"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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

sys.path.insert(0, str((ROOT / "scripts").resolve()))
import run_medium_office_diagnostic_panel as panel

__main__.FeatureSpec = panel.FeatureSpec
__main__.PredictorBundle = panel.PredictorBundle

TSV_CLASSES = np.arange(-3, 4, dtype=int)
CLASS_LABELS = [f"TSV {k:+d}" for k in TSV_CLASSES]
TAIL_INDEX = np.array([0, 1, 5, 6])
TAIL_BIN_EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def heldout_split(data_path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    df = panel.read_training_data(data_path, sample_limit=None)
    y = panel.round_tsv(df["thermal_sensation"])
    train_df, hold_df, y_train, y_hold = train_test_split(
        df,
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
    return fit_df.reset_index(drop=True), y_fit, test_df.reset_index(drop=True), y_test


def load_bundle(path: Path) -> panel.PredictorBundle:
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def pmv_class_from_features(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pmv = pd.to_numeric(features["pmv"], errors="coerce").fillna(0.0).to_numpy(float)
    signed = np.clip(np.rint(pmv), -3, 3).astype(int)
    pred = signed + 3
    probs = np.zeros((len(pred), 7), dtype=float)
    probs[np.arange(len(pred)), pred] = 1.0
    return pred, probs


def pmv_abs_from_features(features: pd.DataFrame) -> np.ndarray:
    pmv = pd.to_numeric(features["pmv"], errors="coerce").fillna(0.0).to_numpy(float)
    return np.abs(pmv)


def pmv_only_tail_baseline(
    fit_features: pd.DataFrame,
    y_fit: np.ndarray,
    test_features: pd.DataFrame,
) -> np.ndarray:
    tail_fit = np.isin(y_fit, TAIL_INDEX).astype(float)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(pmv_abs_from_features(fit_features), tail_fit)
    return np.clip(model.predict(pmv_abs_from_features(test_features)), 0.0, 1.0)


def model_outputs(
    name: str,
    bundle: panel.PredictorBundle,
    test_df: pd.DataFrame,
) -> tuple[str, np.ndarray, np.ndarray]:
    features = panel.build_features_from_raw(test_df, bundle.spec)
    probs = bundle.predict_ordinal(features)
    return name, probs.argmax(axis=1), probs


def summarize_model(name: str, y_true: np.ndarray, pred: np.ndarray, probs: np.ndarray | None) -> dict:
    tail_true = np.isin(y_true, TAIL_INDEX)
    tail_pred = np.isin(pred, TAIL_INDEX)
    y_signed = y_true - 3
    pred_signed = pred - 3
    row = {
        "predictor": name,
        "exact_accuracy": accuracy_score(y_true, pred),
        "within_one_class_accuracy": np.mean(np.abs(pred - y_true) <= 1),
        "ordinal_mae_classes": np.mean(np.abs(pred_signed - y_signed)),
        "tail_precision": precision_score(tail_true, tail_pred, zero_division=0),
        "tail_recall": recall_score(tail_true, tail_pred, zero_division=0),
        "tail_f1": f1_score(tail_true, tail_pred, zero_division=0),
    }
    if probs is not None:
        tail_prob = probs[:, TAIL_INDEX].sum(axis=1)
        row.update(
            {
                "log_loss": log_loss(y_true, probs, labels=np.arange(7)),
                "expected_tsv_mae": np.mean(np.abs((probs @ panel.TSV_VALUES) - y_signed)),
                "tail_probability_mae": np.mean(np.abs(tail_prob - tail_true.astype(float))),
                "tail_brier": np.mean((tail_prob - tail_true.astype(float)) ** 2),
            }
        )
    else:
        row.update(
            {
                "log_loss": np.nan,
                "expected_tsv_mae": np.nan,
                "tail_probability_mae": np.mean(np.abs(tail_pred.astype(float) - tail_true.astype(float))),
                "tail_brier": np.mean((tail_pred.astype(float) - tail_true.astype(float)) ** 2),
            }
        )
    return row


def class_recall_rows(name: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    row = {"predictor": name}
    for idx, label in enumerate(CLASS_LABELS):
        mask = y_true == idx
        row[f"recall_{label}"] = np.mean(pred[mask] == idx) if mask.any() else np.nan
    return row


def class_distribution_rows(name: str, pred: np.ndarray) -> dict:
    row = {"predictor": name}
    for idx, label in enumerate(CLASS_LABELS):
        row[f"pred_share_{label}"] = np.mean(pred == idx)
    return row


def tail_calibration_bins(
    name: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    edges: np.ndarray = TAIL_BIN_EDGES,
) -> pd.DataFrame:
    tail_true = np.isin(y_true, TAIL_INDEX).astype(float)
    tail_prob = probs[:, TAIL_INDEX].sum(axis=1)
    records = []
    bin_ids = np.digitize(tail_prob, edges[1:-1], right=False)
    for i in range(len(edges) - 1):
        mask = bin_ids == i
        if not mask.any():
            continue
        pred_mean = float(tail_prob[mask].mean())
        obs_mean = float(tail_true[mask].mean())
        n = int(mask.sum())
        se = float(np.sqrt(max(obs_mean * (1.0 - obs_mean), 0.0) / n))
        records.append(
            {
                "predictor": name,
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "n": n,
                "predicted_p_tail_mean": pred_mean,
                "observed_tail_frequency": obs_mean,
                "absolute_calibration_error": abs(pred_mean - obs_mean),
                "binomial_se": se,
            }
        )
    return pd.DataFrame(records)


def tail_calibration_summary(name: str, calibration: pd.DataFrame, total_n: int, y_true: np.ndarray, probs: np.ndarray) -> dict:
    tail_true = np.isin(y_true, TAIL_INDEX).astype(float)
    tail_prob = probs[:, TAIL_INDEX].sum(axis=1)
    weights = calibration["n"].to_numpy(float) / float(total_n)
    abs_err = calibration["absolute_calibration_error"].to_numpy(float)
    return {
        "predictor": name,
        "n_test": int(total_n),
        "mean_predicted_p_tail": float(tail_prob.mean()),
        "observed_tail_frequency": float(tail_true.mean()),
        "tail_brier": float(np.mean((tail_prob - tail_true) ** 2)),
        "tail_ece_fixed_bins": float(np.sum(weights * abs_err)),
        "tail_mce_fixed_bins": float(abs_err.max()) if len(abs_err) else np.nan,
    }


def tail_calibration_bins_from_probability(
    name: str,
    y_true: np.ndarray,
    tail_prob: np.ndarray,
    edges: np.ndarray = TAIL_BIN_EDGES,
) -> pd.DataFrame:
    tail_true = np.isin(y_true, TAIL_INDEX).astype(float)
    records = []
    bin_ids = np.digitize(tail_prob, edges[1:-1], right=False)
    for i in range(len(edges) - 1):
        mask = bin_ids == i
        if not mask.any():
            continue
        pred_mean = float(tail_prob[mask].mean())
        obs_mean = float(tail_true[mask].mean())
        n = int(mask.sum())
        se = float(np.sqrt(max(obs_mean * (1.0 - obs_mean), 0.0) / n))
        records.append(
            {
                "predictor": name,
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "n": n,
                "predicted_p_tail_mean": pred_mean,
                "observed_tail_frequency": obs_mean,
                "absolute_calibration_error": abs(pred_mean - obs_mean),
                "binomial_se": se,
            }
        )
    return pd.DataFrame(records)


def tail_probability_summary(
    name: str,
    y_true: np.ndarray,
    tail_prob: np.ndarray,
    threshold: float = 0.20,
) -> dict:
    tail_true = np.isin(y_true, TAIL_INDEX).astype(float)
    tail_pred = tail_prob >= threshold
    bins = tail_calibration_bins_from_probability(name, y_true, tail_prob)
    weights = bins["n"].to_numpy(float) / float(len(y_true))
    abs_err = bins["absolute_calibration_error"].to_numpy(float)
    return {
        "predictor": name,
        "n_test": int(len(y_true)),
        "mean_predicted_p_tail": float(tail_prob.mean()),
        "observed_tail_frequency": float(tail_true.mean()),
        "tail_brier": float(np.mean((tail_prob - tail_true) ** 2)),
        "tail_ece_fixed_bins": float(np.sum(weights * abs_err)),
        "tail_mce_fixed_bins": float(abs_err.max()) if len(abs_err) else np.nan,
        "tail_precision_at_0_20": precision_score(tail_true, tail_pred, zero_division=0),
        "tail_recall_at_0_20": recall_score(tail_true, tail_pred, zero_division=0),
        "tail_f1_at_0_20": f1_score(tail_true, tail_pred, zero_division=0),
        "tail_auroc": roc_auc_score(tail_true, tail_prob),
        "tail_average_precision": average_precision_score(tail_true, tail_prob),
    }


def write_calibration_plot(calibration: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.4), dpi=180)
    styles = {
        "Primary TSV model": {"color": "#2b6cb0", "marker": "o"},
        "No-PMV TSV model": {"color": "#b45309", "marker": "s"},
    }
    for name, group in calibration.groupby("predictor", sort=False):
        style = styles.get(name, {"color": "#333333", "marker": "o"})
        x = group["predicted_p_tail_mean"].to_numpy(float)
        y = group["observed_tail_frequency"].to_numpy(float)
        se = group["binomial_se"].to_numpy(float)
        ax.errorbar(
            x,
            y,
            yerr=1.96 * se,
            label=name,
            lw=1.2,
            marker=style["marker"],
            ms=4.5,
            capsize=2.5,
            color=style["color"],
        )
    ax.plot([0, 1], [0, 1], color="#555555", lw=0.9, ls="--", label="Ideal calibration")
    ax.axvline(0.20, color="#7a2d42", lw=0.8, ls=":")
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel(r"Mean predicted $p_{\mathrm{tail}}$ in bin")
    ax.set_ylabel("Observed tail frequency")
    ax.grid(color="#dddddd", lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "p_tail_calibration_reliability.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "p_tail_calibration_reliability.png", bbox_inches="tight")
    plt.close(fig)


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{100.0 * x:.1f}"


def write_latex_tables(summary: pd.DataFrame, recall: pd.DataFrame, out_dir: Path) -> None:
    compact = summary.copy()
    compact["Exact acc. (\\%)"] = compact["exact_accuracy"].map(format_pct)
    compact["Within $\\pm1$ (\\%)"] = compact["within_one_class_accuracy"].map(format_pct)
    compact["Class MAE"] = compact["ordinal_mae_classes"].map(lambda x: f"{x:.2f}")
    compact["Tail recall (\\%)"] = compact["tail_recall"].map(format_pct)
    compact["Tail F1 (\\%)"] = compact["tail_f1"].map(format_pct)
    compact["Log loss"] = compact["log_loss"].map(lambda x: "--" if pd.isna(x) else f"{x:.3f}")
    compact["Tail prob. MAE"] = compact["tail_probability_mae"].map(lambda x: f"{x:.3f}")
    compact = compact[
        [
            "predictor",
            "Exact acc. (\\%)",
            "Within $\\pm1$ (\\%)",
            "Class MAE",
            "Tail recall (\\%)",
            "Tail F1 (\\%)",
            "Log loss",
            "Tail prob. MAE",
        ]
    ]
    compact = compact.rename(columns={"predictor": "Predictor"})
    (out_dir / "tsv_predictor_validation_table.tex").write_text(
        compact.to_latex(index=False, escape=False, column_format="lrrrrrrr"),
        encoding="utf-8",
    )

    recall_table = recall.copy()
    for col in recall_table.columns:
        if col != "predictor":
            recall_table[col] = recall_table[col].map(format_pct)
    recall_table = recall_table.rename(
        columns={
            "predictor": "Predictor",
            **{f"recall_{label}": label for label in CLASS_LABELS},
        }
    )
    (out_dir / "tsv_predictor_class_recall_table.tex").write_text(
        recall_table.to_latex(index=False, escape=False, column_format="lrrrrrrr"),
        encoding="utf-8",
    )


def write_calibration_latex_tables(cal_summary: pd.DataFrame, calibration: pd.DataFrame, out_dir: Path) -> None:
    compact = cal_summary.copy()
    for col in [
        "mean_predicted_p_tail",
        "observed_tail_frequency",
        "tail_brier",
        "tail_ece_fixed_bins",
        "tail_mce_fixed_bins",
    ]:
        compact[col] = compact[col].map(lambda x: f"{x:.3f}")
    compact = compact.rename(
        columns={
            "predictor": "Predictor",
            "n_test": "$n$",
            "mean_predicted_p_tail": "Mean predicted $p_{\\mathrm{tail}}$",
            "observed_tail_frequency": "Observed tail frequency",
            "tail_brier": "Tail Brier",
            "tail_ece_fixed_bins": "Tail ECE",
            "tail_mce_fixed_bins": "Tail MCE",
        }
    )
    (out_dir / "p_tail_calibration_summary_table.tex").write_text(
        compact.to_latex(index=False, escape=False, column_format="lrrrrrr"),
        encoding="utf-8",
    )

    bins = calibration.copy()
    bins = bins[bins["predictor"].eq("Primary TSV model")].copy()
    bins["Bin"] = bins.apply(
        lambda r: f"{r['bin_left']:.2f}--{r['bin_right']:.2f}", axis=1
    )
    bins["Mean predicted"] = bins["predicted_p_tail_mean"].map(lambda x: f"{x:.3f}")
    bins["Observed"] = bins["observed_tail_frequency"].map(lambda x: f"{x:.3f}")
    bins["Abs. error"] = bins["absolute_calibration_error"].map(lambda x: f"{x:.3f}")
    bins = bins[["Bin", "n", "Mean predicted", "Observed", "Abs. error"]].rename(columns={"n": "$n$"})
    (out_dir / "p_tail_primary_calibration_bins_table.tex").write_text(
        bins.to_latex(index=False, escape=False, column_format="lrrrr"),
        encoding="utf-8",
    )


def write_pmv_only_tail_latex(pmv_summary: pd.DataFrame, out_dir: Path) -> None:
    compact = pmv_summary.copy()
    for col in [
        "mean_predicted_p_tail",
        "observed_tail_frequency",
        "tail_brier",
        "tail_ece_fixed_bins",
        "tail_mce_fixed_bins",
        "tail_auroc",
        "tail_average_precision",
    ]:
        compact[col] = compact[col].map(lambda x: f"{x:.3f}")
    for col in ["tail_precision_at_0_20", "tail_recall_at_0_20", "tail_f1_at_0_20"]:
        compact[col] = compact[col].map(format_pct)
    compact = compact.rename(
        columns={
            "predictor": "Baseline",
            "n_test": "$n$",
            "mean_predicted_p_tail": "Mean predicted",
            "observed_tail_frequency": "Observed",
            "tail_brier": "Brier",
            "tail_ece_fixed_bins": "ECE",
            "tail_mce_fixed_bins": "MCE",
            "tail_precision_at_0_20": "Precision (\\%)",
            "tail_recall_at_0_20": "Recall (\\%)",
            "tail_f1_at_0_20": "F1 (\\%)",
            "tail_auroc": "AUROC",
            "tail_average_precision": "Avg. precision",
        }
    )
    compact = compact[
        [
            "Baseline",
            "$n$",
            "Mean predicted",
            "Observed",
            "Brier",
            "ECE",
            "MCE",
            "F1 (\\%)",
            "AUROC",
            "Avg. precision",
        ]
    ]
    (out_dir / "pmv_only_tail_baseline_table.tex").write_text(
        compact.to_latex(index=False, escape=False, column_format="lrrrrrrrrr"),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_df, y_fit, test_df, y_test = heldout_split(args.data)

    primary = load_bundle(args.model_dir / "control_predictors.joblib")
    no_pmv = load_bundle(args.model_dir / "control_predictors_no_pmv.joblib")

    outputs: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for name, bundle in [("Primary TSV model", primary), ("No-PMV TSV model", no_pmv)]:
        model_name, pred, probs = model_outputs(name, bundle, test_df)
        outputs.append((model_name, pred, probs))

    fit_features = panel.build_features_from_raw(fit_df, primary.spec)
    primary_features = panel.build_features_from_raw(test_df, primary.spec)
    pmv_pred, _ = pmv_class_from_features(primary_features)
    outputs.append(("Rounded PMV class", pmv_pred, None))
    pmv_only_tail_prob = pmv_only_tail_baseline(fit_features, y_fit, primary_features)

    summary = pd.DataFrame([summarize_model(name, y_test, pred, probs) for name, pred, probs in outputs])
    recall = pd.DataFrame([class_recall_rows(name, y_test, pred) for name, pred, _ in outputs])
    pred_share = pd.DataFrame([class_distribution_rows(name, pred) for name, pred, _ in outputs])

    support = {
        "n_test": int(len(y_test)),
        "class_support": {
            f"TSV {cls:+d}": int(np.sum(y_test == idx)) for idx, cls in enumerate(TSV_CLASSES)
        },
    }

    calibration_frames = []
    calibration_summary_rows = []
    for name, _, probs in outputs:
        if probs is None:
            continue
        bins = tail_calibration_bins(name, y_test, probs)
        calibration_frames.append(bins)
        calibration_summary_rows.append(
            tail_calibration_summary(name, bins, len(y_test), y_test, probs)
        )
    calibration = pd.concat(calibration_frames, ignore_index=True)
    calibration_summary = pd.DataFrame(calibration_summary_rows)

    summary.to_csv(args.output_dir / "tsv_predictor_validation_summary.csv", index=False)
    recall.to_csv(args.output_dir / "tsv_predictor_class_recall.csv", index=False)
    pred_share.to_csv(args.output_dir / "tsv_predictor_predicted_class_share.csv", index=False)
    calibration.to_csv(args.output_dir / "p_tail_calibration_bins.csv", index=False)
    calibration_summary.to_csv(args.output_dir / "p_tail_calibration_summary.csv", index=False)
    pmv_only_bins = tail_calibration_bins_from_probability("PMV-only calibrated tail baseline", y_test, pmv_only_tail_prob)
    pmv_only_summary = pd.DataFrame(
        [tail_probability_summary("PMV-only calibrated tail baseline", y_test, pmv_only_tail_prob)]
    )
    pmv_only_bins.to_csv(args.output_dir / "pmv_only_tail_calibration_bins.csv", index=False)
    pmv_only_summary.to_csv(args.output_dir / "pmv_only_tail_baseline_summary.csv", index=False)
    (args.output_dir / "tsv_predictor_validation_support.json").write_text(
        json.dumps(support, indent=2) + "\n",
        encoding="utf-8",
    )
    write_latex_tables(summary, recall, args.output_dir)
    write_calibration_latex_tables(calibration_summary, calibration, args.output_dir)
    write_pmv_only_tail_latex(pmv_only_summary, args.output_dir)
    write_calibration_plot(calibration, args.output_dir)

    print(f"[write] {args.output_dir / 'tsv_predictor_validation_summary.csv'}")
    print(f"[write] {args.output_dir / 'tsv_predictor_class_recall.csv'}")
    print(f"[write] {args.output_dir / 'p_tail_calibration_summary.csv'}")
    print(f"[write] {args.output_dir / 'pmv_only_tail_baseline_summary.csv'}")
    print(f"[write] {args.output_dir / 'p_tail_calibration_reliability.pdf'}")
    print(f"[write] {args.output_dir / 'tsv_predictor_validation_table.tex'}")
    print(summary.to_string(index=False))
    print(calibration_summary.to_string(index=False))
    print(pmv_only_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

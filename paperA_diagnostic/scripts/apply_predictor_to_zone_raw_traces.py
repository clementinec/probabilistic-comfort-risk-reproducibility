#!/usr/bin/env python3
"""Apply a saved TSV predictor bundle to fixed-reference zone-raw traces.

This is used for robustness checks such as a no-PMV ordinal predictor. It does
not rerun EnergyPlus. It preserves the simulated environmental trace and
recomputes only the probability-derived TSV fields.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import numpy as np
import pandas as pd

import analyze_metabolic_spread as mean_met
import run_medium_office_diagnostic_panel as runner


DEFAULT_TRACE_DIR = ROOT / "runs" / "diagnostic_reference_zone_raw_full" / "traces"
DEFAULT_MODEL = (
    ROOT
    / "runs"
    / "diagnostic_reference_zone_raw_full"
    / "models"
    / "control_predictors_no_pmv.joblib"
)
DEFAULT_OUT = ROOT / "runs" / "diagnostic_reference_zone_raw_no_pmv" / "traces"

BASE_COLS = {
    "strategy",
    "weather",
    "calendar_year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "current_time",
    "sim_time_days",
    "occupied",
    "outdoor_temp_c",
    "running_mean_outdoor_c",
    "comfort_low_c",
    "comfort_high_c",
    "mean_air_temp_c",
    "mean_mrt_c",
    "mean_operative_temp_c",
    "mean_rh_pct",
    "mean_pmv",
    "action_delta_c",
    "action_direction",
    "setpoint_shift_c",
    "grid_event",
    "grid_stress_score",
    "grid_oat_c",
    "grid_ghi_w_m2",
    "grid_requested_delta_c",
    "grid_served_delta_c",
    "grid_rejected",
    "heating_setpoint_c",
    "cooling_setpoint_c",
    "zone_heating_rate_w",
    "zone_cooling_rate_w",
    "hvac_on",
    "electricity_facility_j",
    "natural_gas_facility_j",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--predictor", choices=["ordinal", "nominal"], default="ordinal")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--combined", action="store_true")
    return parser.parse_args()


def load_bundle(model_path: Path):
    import __main__

    for name in ["FeatureSpec", "PredictorBundle"]:
        if not hasattr(__main__, name):
            setattr(__main__, name, getattr(runner, name))
    return joblib.load(model_path)


def zone_raw_cols() -> list[str]:
    cols: list[str] = []
    for slug in runner.ZONE_FIELD_NAMES:
        cols.extend(
            [
                f"zone_{slug}_ta_c",
                f"zone_{slug}_tr_c",
                f"zone_{slug}_rh_pct",
            ]
        )
    return cols


def drop_probability_cols(columns: list[str]) -> list[str]:
    suffixes = (
        "_expected_tsv",
        "_p_disc",
        "_warm_tail",
        "_cold_tail",
        "_d_tail",
    )
    drop = {
        "expected_tsv",
        "discomfort_probability",
        "warm_discomfort_probability",
        "cold_discomfort_probability",
    }
    return [
        col
        for col in columns
        if col not in drop
        and not (col.startswith("zone_") and col.endswith(suffixes))
    ]


def trace_paths(trace_dir: Path, max_cases: int | None) -> list[Path]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    paths = [path for path in paths if path.name != "medium_office_control_traces.csv"]
    if not paths:
        raise FileNotFoundError(f"No diagnostic-reference traces found in {trace_dir}")
    return paths[:max_cases] if max_cases is not None else paths


def predict_probabilities(bundle, df: pd.DataFrame, predictor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ta_cols = [f"zone_{slug}_ta_c" for slug in runner.ZONE_FIELD_NAMES]
    tr_cols = [f"zone_{slug}_tr_c" for slug in runner.ZONE_FIELD_NAMES]
    rh_cols = [f"zone_{slug}_rh_pct" for slug in runner.ZONE_FIELD_NAMES]
    ta = df[ta_cols].to_numpy(float)
    tr = df[tr_cols].to_numpy(float)
    rh = df[rh_cols].to_numpy(float)
    n, z = ta.shape
    rm = df["running_mean_outdoor_c"].where(
        df["running_mean_outdoor_c"].notna(), df["outdoor_temp_c"]
    ).to_numpy(float)
    features = runner.build_features_from_arrays(
        ta=ta.reshape(-1),
        tr=tr.reshape(-1),
        v=np.full(n * z, 0.10),
        rh=rh.reshape(-1),
        met=np.full(n * z, 1.10),
        clo=np.full(n * z, 0.65),
        bsa=np.full(n * z, 1.80),
        rm_out=np.repeat(rm, z),
        spec=bundle.spec,
    )
    if predictor == "ordinal":
        probs = bundle.predict_ordinal(features)
    else:
        probs = bundle.predict_nominal(features)
    probs = probs.reshape(n, z, -1)
    zone_mu = probs @ runner.TSV_VALUES
    zone_cold = probs[:, :, [0, 1]].sum(axis=2)
    zone_warm = probs[:, :, [5, 6]].sum(axis=2)
    return zone_mu, zone_cold, zone_warm


def rewrite_trace(path: Path, bundle, output_dir: Path, predictor: str) -> Path:
    raw_cols = set(zone_raw_cols())
    keep_base = BASE_COLS | raw_cols
    df = pd.read_csv(path, usecols=lambda col: col in keep_base or col.startswith("zone_"))
    missing = sorted(raw_cols - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing zone raw columns: {', '.join(missing[:6])}")
    out = df[drop_probability_cols(list(df.columns))].copy()
    out["strategy"] = "diagnostic_reference"

    occupied = mean_met.bool_series(out["occupied"])
    zone_mu = np.full((len(out), len(runner.ZONE_FIELD_NAMES)), np.nan, dtype=float)
    zone_cold = np.full_like(zone_mu, np.nan)
    zone_warm = np.full_like(zone_mu, np.nan)
    if occupied.any():
        mu_occ, cold_occ, warm_occ = predict_probabilities(bundle, out.loc[occupied], predictor)
        zone_mu[occupied.to_numpy(bool), :] = mu_occ
        zone_cold[occupied.to_numpy(bool), :] = cold_occ
        zone_warm[occupied.to_numpy(bool), :] = warm_occ

    zone_p = zone_cold + zone_warm
    zone_d = zone_warm - zone_cold
    out["expected_tsv"] = np.nan
    out["warm_discomfort_probability"] = np.nan
    out["cold_discomfort_probability"] = np.nan
    occ_mask = occupied.to_numpy(bool)
    out.loc[occ_mask, "expected_tsv"] = zone_mu[occ_mask].mean(axis=1)
    out.loc[occ_mask, "warm_discomfort_probability"] = zone_warm[occ_mask].mean(axis=1)
    out.loc[occ_mask, "cold_discomfort_probability"] = zone_cold[occ_mask].mean(axis=1)
    out["discomfort_probability"] = np.nan
    out.loc[occ_mask, "discomfort_probability"] = (
        out.loc[occ_mask, "warm_discomfort_probability"]
        + out.loc[occ_mask, "cold_discomfort_probability"]
    )
    for idx, slug in enumerate(runner.ZONE_FIELD_NAMES):
        out[f"zone_{slug}_expected_tsv"] = zone_mu[:, idx]
        out[f"zone_{slug}_p_disc"] = zone_p[:, idx]
        out[f"zone_{slug}_warm_tail"] = zone_warm[:, idx]
        out[f"zone_{slug}_cold_tail"] = zone_cold[:, idx]
        out[f"zone_{slug}_d_tail"] = zone_d[:, idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / path.name
    out.to_csv(target, index=False)
    return target


def write_combined(paths: list[Path], combined_path: Path) -> None:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    header: str | None = None
    with combined_path.open("w", encoding="utf-8", newline="") as out:
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as handle:
                current_header = handle.readline()
                if header is None:
                    header = current_header
                    out.write(current_header)
                elif current_header != header:
                    raise ValueError(f"Trace schema mismatch in {path}")
                for line in handle:
                    out.write(line)


def main() -> int:
    args = parse_args()
    bundle = load_bundle(args.model_path)
    print(f"[model] loaded {args.model_path}")
    print(f"[model] feature columns: {bundle.feature_columns}")
    written: list[Path] = []
    for i, path in enumerate(trace_paths(args.trace_dir, args.max_cases), start=1):
        target = rewrite_trace(path, bundle, args.output_dir, args.predictor)
        written.append(target)
        print(f"[write] {i}: {target}")
    if args.combined:
        combined = args.output_dir / "medium_office_control_traces.csv"
        write_combined(written, combined)
        print(f"[write] {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

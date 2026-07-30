#!/usr/bin/env python3
"""Endpoint-only and nominal-model robustness checks for Paper A.

This script does not rerun EnergyPlus. It applies the ordinal and nominal
predictors already stored in the Paper A bundle to the same occupied zone-level
environmental states. It writes compact case/group summaries instead of
duplicating the multi-gigabyte traces.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SCRIPT = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT.parents[1]
R01_DIR = SCRIPT.parents[2]
WORKSPACE = SCRIPT.parents[3]
REBUILD = WORKSPACE / "paperA_rebuild"
REBUILD_SCRIPTS = REBUILD / "scripts"
if str(REBUILD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REBUILD_SCRIPTS))

os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ANALYSIS_DIR / ".mplconfig"))
os.environ.setdefault("FC_CACHEDIR", str(ANALYSIS_DIR / ".mplconfig"))

import __main__

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import run_medium_office_diagnostic_panel as panel

__main__.FeatureSpec = panel.FeatureSpec
__main__.PredictorBundle = panel.PredictorBundle


DEFAULT_TRACE_DIR = REBUILD / "runs/diagnostic_reference_zone_raw_full/traces"
DEFAULT_MODEL = REBUILD / "runs/diagnostic_reference_zone_raw_full/models/control_predictors.joblib"
DEFAULT_NO_PMV_MODEL = (
    REBUILD
    / "runs/diagnostic_reference_zone_raw_full/models/control_predictors_no_pmv.joblib"
)
DEFAULT_DATA = WORKSPACE.parent / "TCN/newin_with_bmr.csv"
DEFAULT_PANEL_MANIFEST = REBUILD / "data/panel_manifest.csv"
DEFAULT_OUT = ANALYSIS_DIR / "outputs/robustness_endpoint_model"

TSV_VALUES = np.arange(-3, 4, dtype=float)
TAIL_INDEX = np.array([0, 1, 5, 6])
OUTERMOST_INDEX = np.array([0, 6])
MODEL_NAMES = ("ordinal", "nominal")
STORED_MODEL_NAME = "stored_ordinal"
AGGREGATORS = ("equal_zone_mean", "area_weighted_mean", "zone_p90", "any_zone")

ENDPOINT_SCREENS = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20])
BROAD_CURVE_SCREENS = np.round(np.arange(0.05, 0.401, 0.01), 3)
BROAD_KEY_SCREENS = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
CALIBRATION_EDGES = np.array([0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0])
HIST_BINS = 10_000
MATERIAL_RATE_DIFFERENCE_PP = 5.0
MATERIAL_MEAN_PROBABILITY_DIFFERENCE = 0.02
NEGLIGIBLE_CONTRAST_PP = 1.0

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

WEATHER_RE = re.compile(
    r"^(?P<city>ahmedabad|beijing|guangzhou|houston|kolkata|phoenix)_"
    r"(?P<scenario>ssp245|ssp585)_"
    r"(?P<time_slice>baseline_2020s|near_2030s|mid_2050s|late_2080s)_"
    r"(?P<severity>typical|hot|heatwave_extreme)_"
    r"(?P<weather_year>\d{4})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--no-pmv-model-path", type=Path, default=DEFAULT_NO_PMV_MODEL)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--panel-manifest", type=Path, default=DEFAULT_PANEL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(
    occupied: np.ndarray,
    running_mean: np.ndarray,
    ta: np.ndarray,
    tr: np.ndarray,
    rh: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"PaperA occupied environmental state v1\0")
    for arr in (occupied.astype(np.uint8), running_mean, ta, tr, rh):
        stable = np.ascontiguousarray(arr, dtype="<f8" if arr.dtype != np.uint8 else np.uint8)
        digest.update(str(stable.shape).encode("ascii"))
        digest.update(stable.tobytes(order="C"))
    return digest.hexdigest()


def bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"}).to_numpy(bool)


def trace_paths(trace_dir: Path, max_cases: int | None) -> list[Path]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    paths = [path for path in paths if path.name != "medium_office_control_traces.csv"]
    if not paths:
        raise FileNotFoundError(f"No case traces found in {trace_dir}")
    return paths[:max_cases] if max_cases is not None else paths


def metadata(weather: str) -> dict[str, object]:
    match = WEATHER_RE.match(weather)
    if not match:
        raise ValueError(f"Unrecognized weather label: {weather}")
    values = match.groupdict()
    return {
        "weather": weather,
        "city": values["city"].title(),
        "scenario": values["scenario"],
        "time_slice": values["time_slice"],
        "severity": values["severity"],
        "weather_year": int(values["weather_year"]),
    }


def load_bundle(path: Path) -> panel.PredictorBundle:
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def zone_columns(suffix: str) -> list[str]:
    return [f"zone_{slug}_{suffix}" for slug in panel.ZONE_FIELD_NAMES]


def input_usecols() -> set[str]:
    fixed = {
        "weather",
        "occupied",
        "month",
        "day",
        "hour",
        "current_time",
        "outdoor_temp_c",
        "running_mean_outdoor_c",
        "mean_pmv",
        "expected_tsv",
        "discomfort_probability",
    }
    return fixed | set(zone_columns("ta_c")) | set(zone_columns("tr_c")) | set(
        zone_columns("rh_pct")
    ) | set(zone_columns("p_disc")) | set(zone_columns("expected_tsv"))


def read_case(path: Path) -> dict[str, object]:
    usecols = input_usecols()
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    occupied_full = bool_array(df["occupied"])
    if not occupied_full.any():
        raise ValueError(f"No occupied records in {path}")
    occ = df.loc[occupied_full].reset_index(drop=True)
    source_row_index = np.flatnonzero(occupied_full).astype(np.int32)
    weather_values = occ["weather"].dropna().astype(str).unique()
    if len(weather_values) != 1:
        raise ValueError(f"Expected one weather label in {path}, found {weather_values}")
    rm = pd.to_numeric(occ["running_mean_outdoor_c"], errors="coerce")
    oat = pd.to_numeric(occ["outdoor_temp_c"], errors="coerce")
    rm = rm.where(rm.notna(), oat).to_numpy(float)
    ta = occ[zone_columns("ta_c")].to_numpy(float)
    tr = occ[zone_columns("tr_c")].to_numpy(float)
    rh = occ[zone_columns("rh_pct")].to_numpy(float)
    stored_p = occ[zone_columns("p_disc")].to_numpy(float)
    stored_mu = occ[zone_columns("expected_tsv")].to_numpy(float)
    return {
        "weather": str(weather_values[0]),
        "source_row_index": source_row_index,
        "month": pd.to_numeric(occ["month"], errors="coerce").to_numpy(np.int16),
        "day": pd.to_numeric(occ["day"], errors="coerce").to_numpy(np.int16),
        "hour": pd.to_numeric(occ["hour"], errors="coerce").to_numpy(np.int16),
        "current_time": pd.to_numeric(occ["current_time"], errors="coerce").to_numpy(float),
        "occupied_full": occupied_full,
        "running_mean": rm,
        "ta": ta,
        "tr": tr,
        "rh": rh,
        "stored_p": stored_p,
        "stored_mu": stored_mu,
        "stored_equal_p": pd.to_numeric(occ["discomfort_probability"], errors="coerce").to_numpy(float),
        "stored_equal_mu": pd.to_numeric(occ["expected_tsv"], errors="coerce").to_numpy(float),
        "stored_mean_pmv": pd.to_numeric(occ["mean_pmv"], errors="coerce").to_numpy(float),
    }


def write_corrected_zone_npz(
    output_dir: Path,
    state_hash: str,
    case: dict[str, object],
    ordinal_probs: np.ndarray,
    zone_pmv: np.ndarray,
    no_pmv_probs: np.ndarray,
) -> tuple[Path, str]:
    zone_dir = output_dir / "corrected_zone_npz"
    zone_dir.mkdir(parents=True, exist_ok=True)
    target = zone_dir / f"{state_hash}.npz"
    needs_write = not target.exists()
    if target.exists():
        try:
            with np.load(target) as existing:
                needs_write = (
                    str(existing["schema_version"]) != "paperA_corrected_same_state_v3"
                    or "zone_pmv" not in existing.files
                    or "zone_no_pmv_expected_tsv" not in existing.files
                )
        except Exception:
            needs_write = True
    if needs_write:
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            schema_version=np.array("paperA_corrected_same_state_v3"),
            source_row_index=np.asarray(case["source_row_index"], dtype=np.int32),
            month=np.asarray(case["month"], dtype=np.int16),
            day=np.asarray(case["day"], dtype=np.int16),
            hour=np.asarray(case["hour"], dtype=np.int16),
            current_time=np.asarray(case["current_time"], dtype=np.float32),
            zone_names=np.asarray(panel.ZONE_FIELD_NAMES),
            zone_pmv=np.asarray(zone_pmv, dtype=np.float32),
            zone_expected_tsv=(ordinal_probs @ TSV_VALUES).astype(np.float32),
            zone_cold_tail=ordinal_probs[:, :, [0, 1]].sum(axis=2).astype(np.float32),
            zone_warm_tail=ordinal_probs[:, :, [5, 6]].sum(axis=2).astype(np.float32),
            zone_cold_endpoint=ordinal_probs[:, :, 0].astype(np.float32),
            zone_warm_endpoint=ordinal_probs[:, :, 6].astype(np.float32),
            zone_no_pmv_expected_tsv=(no_pmv_probs @ TSV_VALUES).astype(np.float32),
            zone_no_pmv_cold_tail=no_pmv_probs[:, :, [0, 1]].sum(axis=2).astype(np.float32),
            zone_no_pmv_warm_tail=no_pmv_probs[:, :, [5, 6]].sum(axis=2).astype(np.float32),
        )
        os.replace(temporary, target)
    return target, file_sha256(target)


def write_corrected_zone_schema(output_dir: Path) -> None:
    text = """# Corrected same-state zone array schema

Each file in this directory is named by the SHA-256 of the analyzed occupied
environmental state. `../input_trace_manifest.csv` maps each of the 144 weather
roles to one of these files; duplicate hot/heatwave roles can intentionally map
to the same state file.

Arrays:

| Field | Shape | Meaning |
|---|---:|---|
| `source_row_index` | `(n,)` | Zero-based row index in the corresponding source case trace. |
| `month`, `day`, `hour`, `current_time` | `(n,)` | Occupied-timestep keys retained for verification. |
| `zone_names` | `(15,)` | Zone slug order for the second array dimension. |
| `zone_pmv` | `(n,15)` | PMV recomputed from the recorded end-of-step Ta/MRT/RH and fixed occupant inputs; this is the PMV feature used by both fresh predictors. |
| `zone_expected_tsv` | `(n,15)` | Corrected ordinal expected TSV for the recorded state. |
| `zone_cold_tail` | `(n,15)` | `P(TSV in {-3,-2})`. |
| `zone_warm_tail` | `(n,15)` | `P(TSV in {+2,+3})`. |
| `zone_cold_endpoint` | `(n,15)` | `P(TSV=-3)`. |
| `zone_warm_endpoint` | `(n,15)` | `P(TSV=+3)`. |
| `zone_no_pmv_expected_tsv` | `(n,15)` | Expected TSV from the separately fitted no-PMV ordinal predictor on the same state. |
| `zone_no_pmv_cold_tail`, `zone_no_pmv_warm_tail` | `(n,15)` | Broad-tail components from the no-PMV predictor, used to regenerate the PMV-feature robustness figure. |

All probability and expected-TSV arrays are stored as float32 to keep the
artifact compact; screening and headline summaries were computed from the
original float64 inference before this serialization. Derive
`p_tail = zone_cold_tail + zone_warm_tail` and
`p_outermost = zone_cold_endpoint + zone_warm_endpoint`.
"""
    zone_dir = output_dir / "corrected_zone_npz"
    zone_dir.mkdir(parents=True, exist_ok=True)
    (zone_dir / "README.md").write_text(text, encoding="utf-8")


def build_features(bundle: panel.PredictorBundle, case: dict[str, object]) -> pd.DataFrame:
    ta = np.asarray(case["ta"], dtype=float)
    tr = np.asarray(case["tr"], dtype=float)
    rh = np.asarray(case["rh"], dtype=float)
    rm = np.asarray(case["running_mean"], dtype=float)
    n, z = ta.shape
    return panel.build_features_from_arrays(
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


def infer_models(
    bundle: panel.PredictorBundle,
    no_pmv_bundle: panel.PredictorBundle,
    case: dict[str, object],
) -> dict[str, np.ndarray]:
    features = build_features(bundle, case)
    no_pmv_features = build_features(no_pmv_bundle, case)
    n, z = np.asarray(case["ta"]).shape
    return {
        "ordinal": bundle.predict_ordinal(features).reshape(n, z, 7),
        "nominal": bundle.predict_nominal(features).reshape(n, z, 7),
        "no_pmv_ordinal": no_pmv_bundle.predict_ordinal(no_pmv_features).reshape(
            n, z, 7
        ),
        "zone_pmv": features["pmv"].to_numpy(float).reshape(n, z),
    }


def aggregations(probability: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "equal_zone_mean": probability.mean(axis=1),
        "area_weighted_mean": probability @ weights,
        "zone_p90": np.quantile(probability, 0.90, axis=1),
        "any_zone": probability.max(axis=1),
    }


def event_arrays(probs: np.ndarray, event: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if event == "broad_tail":
        cold = probs[:, :, [0, 1]].sum(axis=2)
        warm = probs[:, :, [5, 6]].sum(axis=2)
    elif event == "outermost":
        cold = probs[:, :, 0]
        warm = probs[:, :, 6]
    else:
        raise ValueError(event)
    return cold + warm, cold, warm


@dataclass
class DistributionAccumulator:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    histogram: np.ndarray = field(default_factory=lambda: np.zeros(HIST_BINS, dtype=np.int64))

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            return
        self.n += int(len(arr))
        self.total += float(arr.sum())
        self.total_sq += float(np.square(arr).sum())
        self.minimum = min(self.minimum, float(arr.min()))
        self.maximum = max(self.maximum, float(arr.max()))
        self.histogram += np.histogram(arr, bins=HIST_BINS, range=(0.0, 1.0))[0]

    def quantile(self, q: float) -> float:
        if self.n == 0:
            return float("nan")
        target = max(1, int(math.ceil(q * self.n)))
        idx = int(np.searchsorted(np.cumsum(self.histogram), target, side="left"))
        return min(1.0, (idx + 0.5) / HIST_BINS)

    def record(self) -> dict[str, float | int]:
        mean = self.total / self.n if self.n else float("nan")
        variance = max(self.total_sq / self.n - mean * mean, 0.0) if self.n else float("nan")
        return {
            "n": self.n,
            "mean": mean,
            "sd": math.sqrt(variance) if self.n else float("nan"),
            "min": self.minimum if self.n else float("nan"),
            "p05_approx": self.quantile(0.05),
            "p50_approx": self.quantile(0.50),
            "p90_approx": self.quantile(0.90),
            "p95_approx": self.quantile(0.95),
            "p99_approx": self.quantile(0.99),
            "max": self.maximum if self.n else float("nan"),
            "histogram_resolution": 1.0 / HIST_BINS,
        }


@dataclass
class PairAccumulator:
    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0
    sae: float = 0.0
    sse: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if not len(x):
            return
        delta = y - x
        self.n += int(len(x))
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float(np.square(x).sum())
        self.syy += float(np.square(y).sum())
        self.sxy += float((x * y).sum())
        self.sae += float(np.abs(delta).sum())
        self.sse += float(np.square(delta).sum())

    def record(self) -> dict[str, float | int]:
        if not self.n:
            return {"n": 0}
        mx = self.sx / self.n
        my = self.sy / self.n
        vx = max(self.sxx / self.n - mx * mx, 0.0)
        vy = max(self.syy / self.n - my * my, 0.0)
        covariance = self.sxy / self.n - mx * my
        corr = covariance / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
        return {
            "n": self.n,
            "ordinal_mean": mx,
            "nominal_mean": my,
            "nominal_minus_ordinal_mean": my - mx,
            "mae": self.sae / self.n,
            "rmse": math.sqrt(self.sse / self.n),
            "pearson_r": corr,
        }


def exact_case_distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_sd": float(np.std(values)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(np.max(values)),
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def scalar_case_record(
    base: dict[str, object],
    corrected_zone_pmv: np.ndarray,
    stored_mean_pmv: np.ndarray,
    corrected_equal_p_tail: np.ndarray,
    stored_equal_p_tail: np.ndarray,
    corrected_equal_expected_tsv: np.ndarray,
    no_pmv_equal_p_tail: np.ndarray,
    no_pmv_equal_expected_tsv: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    corrected_zone_pmv = np.asarray(corrected_zone_pmv, dtype=float)
    corrected = corrected_zone_pmv.mean(axis=1)
    corrected_area = corrected_zone_pmv @ weights
    stored = np.asarray(stored_mean_pmv, dtype=float)
    difference = corrected - stored
    record: dict[str, object] = {
        **base,
        "n_steps": int(len(corrected)),
        "corrected_mean_pmv_mean": float(corrected.mean()),
        "corrected_mean_pmv_sd": float(corrected.std()),
        "corrected_mean_pmv_p05": float(np.quantile(corrected, 0.05)),
        "corrected_mean_pmv_p50": float(np.quantile(corrected, 0.50)),
        "corrected_mean_pmv_p95": float(np.quantile(corrected, 0.95)),
        "corrected_mean_pmv_min": float(corrected.min()),
        "corrected_mean_pmv_max": float(corrected.max()),
        "corrected_mean_abs_pmv": float(np.abs(corrected).mean()),
        "corrected_area_pmv_mean": float(corrected_area.mean()),
        "corrected_abs_pmv_ge_0p5_pct": float((np.abs(corrected) >= 0.5).mean() * 100.0),
        "corrected_abs_pmv_ge_1p0_pct": float((np.abs(corrected) >= 1.0).mean() * 100.0),
        "corrected_abs_pmv_ge_2p0_pct": float((np.abs(corrected) >= 2.0).mean() * 100.0),
        "stored_mean_pmv_mean": float(stored.mean()),
        "stored_mean_pmv_sd": float(stored.std()),
        "stored_mean_pmv_p05": float(np.quantile(stored, 0.05)),
        "stored_mean_pmv_p50": float(np.quantile(stored, 0.50)),
        "stored_mean_pmv_p95": float(np.quantile(stored, 0.95)),
        "stored_mean_pmv_min": float(stored.min()),
        "stored_mean_pmv_max": float(stored.max()),
        "stored_mean_abs_pmv": float(np.abs(stored).mean()),
        "stored_abs_pmv_ge_0p5_pct": float((np.abs(stored) >= 0.5).mean() * 100.0),
        "stored_abs_pmv_ge_1p0_pct": float((np.abs(stored) >= 1.0).mean() * 100.0),
        "stored_abs_pmv_ge_2p0_pct": float((np.abs(stored) >= 2.0).mean() * 100.0),
        "corrected_minus_stored_pmv_mean": float(difference.mean()),
        "corrected_vs_stored_pmv_mae": float(np.abs(difference).mean()),
        "corrected_vs_stored_pmv_mse": float(np.square(difference).mean()),
        "corrected_vs_stored_pmv_rmse": float(np.sqrt(np.square(difference).mean())),
        "corrected_vs_stored_pmv_max_abs": float(np.abs(difference).max()),
        "corrected_vs_stored_pmv_pearson_r": safe_corr(stored, corrected),
        "corrected_pmv_vs_expected_tsv_pearson_r": safe_corr(
            corrected, corrected_equal_expected_tsv
        ),
        "corrected_abs_pmv_vs_p_tail_pearson_r": safe_corr(
            np.abs(corrected), corrected_equal_p_tail
        ),
        "stored_abs_pmv_vs_stored_p_tail_pearson_r": safe_corr(
            np.abs(stored), stored_equal_p_tail
        ),
        "corrected_abs_pmv_vs_no_pmv_p_tail_pearson_r": safe_corr(
            np.abs(corrected), no_pmv_equal_p_tail
        ),
        "no_pmv_abs_expected_tsv_vs_p_tail_pearson_r": safe_corr(
            np.abs(no_pmv_equal_expected_tsv), no_pmv_equal_p_tail
        ),
        "corrected_pmv_central_but_tail_020_pct": float(
            ((np.abs(corrected) <= 0.5) & (corrected_equal_p_tail >= 0.20)).mean()
            * 100.0
        ),
        "stored_pmv_central_but_tail_020_pct": float(
            ((np.abs(stored) <= 0.5) & (stored_equal_p_tail >= 0.20)).mean()
            * 100.0
        ),
        "corrected_pmv_central_but_no_pmv_tail_020_pct": float(
            ((np.abs(corrected) <= 0.5) & (no_pmv_equal_p_tail >= 0.20)).mean()
            * 100.0
        ),
    }
    return record


def global_scalar_summary(
    case_scalar: pd.DataFrame,
    old_new_pair: PairAccumulator,
    pmv_expected_pair: PairAccumulator,
    abs_pmv_tail_pair: PairAccumulator,
) -> pd.DataFrame:
    weights = case_scalar["n_steps"].to_numpy(float)
    weighted_fields = [
        "corrected_mean_pmv_mean",
        "corrected_mean_abs_pmv",
        "corrected_area_pmv_mean",
        "corrected_abs_pmv_ge_0p5_pct",
        "corrected_abs_pmv_ge_1p0_pct",
        "corrected_abs_pmv_ge_2p0_pct",
        "stored_mean_pmv_mean",
        "stored_mean_abs_pmv",
        "stored_abs_pmv_ge_0p5_pct",
        "stored_abs_pmv_ge_1p0_pct",
        "stored_abs_pmv_ge_2p0_pct",
        "corrected_minus_stored_pmv_mean",
        "corrected_vs_stored_pmv_mae",
        "corrected_vs_stored_pmv_mse",
        "corrected_pmv_central_but_tail_020_pct",
        "stored_pmv_central_but_tail_020_pct",
        "corrected_pmv_central_but_no_pmv_tail_020_pct",
    ]
    row: dict[str, object] = {
        "scope": "all_occupied_role_weighted",
        "n_steps": int(weights.sum()),
        "n_cases": int(len(case_scalar)),
    }
    for field_name in weighted_fields:
        row[field_name] = float(np.average(case_scalar[field_name], weights=weights))
    row["corrected_vs_stored_pmv_rmse"] = math.sqrt(
        float(row["corrected_vs_stored_pmv_mse"])
    )
    row["corrected_vs_stored_pmv_max_abs"] = float(
        case_scalar["corrected_vs_stored_pmv_max_abs"].max()
    )
    old_new = old_new_pair.record()
    row["corrected_vs_stored_pmv_pearson_r"] = old_new.get("pearson_r", np.nan)
    pmv_mu = pmv_expected_pair.record()
    row["corrected_pmv_vs_expected_tsv_pearson_r"] = pmv_mu.get(
        "pearson_r", np.nan
    )
    pmv_tail = abs_pmv_tail_pair.record()
    row["corrected_abs_pmv_vs_p_tail_pearson_r"] = pmv_tail.get(
        "pearson_r", np.nan
    )
    return pd.DataFrame([row])


def threshold_record(
    base: dict[str, object],
    model: str,
    event: str,
    threshold: float,
    zone_probability: np.ndarray,
    aggregate: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, object]:
    decisions = {name: values >= threshold for name, values in aggregate.items()}
    n = len(aggregate["equal_zone_mean"])
    zone_high = zone_probability >= threshold
    area_zone_time_mass = float((zone_high.astype(float) @ weights).sum())
    record: dict[str, object] = {
        **base,
        "model": model,
        "event": event,
        "threshold": float(threshold),
        "n_steps": n,
        "n_zone_steps": int(zone_probability.size),
        "area_zone_time_mass": area_zone_time_mass,
        "unweighted_zone_high_count": int(zone_high.sum()),
    }
    for name, mask in decisions.items():
        record[f"{name}_high_count"] = int(mask.sum())
        record[f"{name}_high_pct"] = float(mask.mean() * 100.0)
    hidden_any = (~decisions["equal_zone_mean"]) & decisions["any_zone"]
    hidden_p90 = (~decisions["equal_zone_mean"]) & decisions["zone_p90"]
    record["hidden_any_zone_count"] = int(hidden_any.sum())
    record["hidden_any_zone_pct"] = float(hidden_any.mean() * 100.0)
    record["hidden_p90_count"] = int(hidden_p90.sum())
    record["hidden_p90_pct"] = float(hidden_p90.mean() * 100.0)
    record["area_weighted_zone_time_high_pct"] = area_zone_time_mass / n * 100.0
    record["unweighted_zone_time_high_pct"] = float(zone_high.mean() * 100.0)
    return record


def paired_threshold_record(
    base: dict[str, object],
    event: str,
    threshold: float,
    ordinal: dict[str, np.ndarray],
    nominal: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    records = []
    for aggregator in AGGREGATORS:
        left = ordinal[aggregator] >= threshold
        right = nominal[aggregator] >= threshold
        union = left | right
        intersection = left & right
        records.append(
            {
                **base,
                "event": event,
                "threshold": float(threshold),
                "aggregator": aggregator,
                "n_steps": int(len(left)),
                "ordinal_high_count": int(left.sum()),
                "nominal_high_count": int(right.sum()),
                "both_high_count": int(intersection.sum()),
                "union_high_count": int(union.sum()),
                "ordinal_only_count": int((left & ~right).sum()),
                "nominal_only_count": int((right & ~left).sum()),
                "disagreement_count": int((left != right).sum()),
                "ordinal_high_pct": float(left.mean() * 100.0),
                "nominal_high_pct": float(right.mean() * 100.0),
                "nominal_minus_ordinal_pp": float((right.mean() - left.mean()) * 100.0),
                "disagreement_pct": float((left != right).mean() * 100.0),
                "jaccard": float(intersection.sum() / union.sum()) if union.any() else 1.0,
            }
        )
    return records


def update_global_threshold(
    totals: dict[tuple[str, str, float], dict[str, float]],
    record: dict[str, object],
) -> None:
    key = (str(record["model"]), str(record["event"]), float(record["threshold"]))
    target = totals.setdefault(
        key,
        {
            "n_steps": 0.0,
            "n_zone_steps": 0.0,
            "area_zone_time_mass": 0.0,
            "unweighted_zone_high_count": 0.0,
            **{f"{name}_high_count": 0.0 for name in AGGREGATORS},
            "hidden_any_zone_count": 0.0,
            "hidden_p90_count": 0.0,
        },
    )
    for field_name in target:
        target[field_name] += float(record[field_name])


def update_global_paired_threshold(
    totals: dict[tuple[str, float, str], dict[str, float]],
    record: dict[str, object],
) -> None:
    key = (str(record["event"]), float(record["threshold"]), str(record["aggregator"]))
    fields = [
        "n_steps",
        "ordinal_high_count",
        "nominal_high_count",
        "both_high_count",
        "union_high_count",
        "ordinal_only_count",
        "nominal_only_count",
        "disagreement_count",
    ]
    target = totals.setdefault(key, {field_name: 0.0 for field_name in fields})
    for field_name in fields:
        target[field_name] += float(record[field_name])


def threshold_totals_frame(
    totals: dict[tuple[str, str, float], dict[str, float]]
) -> pd.DataFrame:
    records = []
    for (model, event, threshold), counts in sorted(totals.items()):
        n = counts["n_steps"]
        nz = counts["n_zone_steps"]
        record: dict[str, object] = {
            "group_scope": "global",
            "model": model,
            "event": event,
            "threshold": threshold,
            **{key: int(value) if key.endswith("count") or key.startswith("n_") else value for key, value in counts.items()},
        }
        for aggregator in AGGREGATORS:
            record[f"{aggregator}_high_pct"] = counts[f"{aggregator}_high_count"] / n * 100.0
        record["hidden_any_zone_pct"] = counts["hidden_any_zone_count"] / n * 100.0
        record["hidden_p90_pct"] = counts["hidden_p90_count"] / n * 100.0
        record["area_weighted_zone_time_high_pct"] = counts["area_zone_time_mass"] / n * 100.0
        record["unweighted_zone_time_high_pct"] = counts["unweighted_zone_high_count"] / nz * 100.0
        records.append(record)
    return pd.DataFrame(records)


def paired_threshold_totals_frame(
    totals: dict[tuple[str, float, str], dict[str, float]]
) -> pd.DataFrame:
    records = []
    for (event, threshold, aggregator), counts in sorted(totals.items()):
        n = counts["n_steps"]
        union = counts["union_high_count"]
        records.append(
            {
                "group_scope": "global",
                "event": event,
                "threshold": threshold,
                "aggregator": aggregator,
                **{key: int(value) for key, value in counts.items()},
                "ordinal_high_pct": counts["ordinal_high_count"] / n * 100.0,
                "nominal_high_pct": counts["nominal_high_count"] / n * 100.0,
                "nominal_minus_ordinal_pp": (
                    counts["nominal_high_count"] - counts["ordinal_high_count"]
                )
                / n
                * 100.0,
                "disagreement_pct": counts["disagreement_count"] / n * 100.0,
                "jaccard": counts["both_high_count"] / union if union else 1.0,
            }
        )
    return pd.DataFrame(records)


def grouped_threshold_summary(case_threshold: pd.DataFrame) -> pd.DataFrame:
    group_specs = [
        ("city", ["city"]),
        ("time_slice", ["time_slice"]),
        ("scenario_time_slice", ["scenario", "time_slice"]),
        ("scenario_time_slice_severity", ["scenario", "time_slice", "severity"]),
        ("city_scenario_time_slice", ["city", "scenario", "time_slice"]),
    ]
    additive = [
        "n_steps",
        "n_zone_steps",
        "area_zone_time_mass",
        "unweighted_zone_high_count",
        *[f"{name}_high_count" for name in AGGREGATORS],
        "hidden_any_zone_count",
        "hidden_p90_count",
    ]
    frames = []
    for scope, columns in group_specs:
        group_columns = ["model", "event", "threshold", *columns]
        grouped = case_threshold.groupby(group_columns, as_index=False, sort=True)[additive].sum()
        grouped.insert(0, "group_scope", scope)
        n = grouped["n_steps"]
        nz = grouped["n_zone_steps"]
        for aggregator in AGGREGATORS:
            grouped[f"{aggregator}_high_pct"] = grouped[f"{aggregator}_high_count"] / n * 100.0
        grouped["hidden_any_zone_pct"] = grouped["hidden_any_zone_count"] / n * 100.0
        grouped["hidden_p90_pct"] = grouped["hidden_p90_count"] / n * 100.0
        grouped["area_weighted_zone_time_high_pct"] = grouped["area_zone_time_mass"] / n * 100.0
        grouped["unweighted_zone_time_high_pct"] = (
            grouped["unweighted_zone_high_count"] / nz * 100.0
        )
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True, sort=False)


def grouped_continuous_summary(case_continuous: pd.DataFrame) -> pd.DataFrame:
    group_specs = [
        ("city", ["city"]),
        ("time_slice", ["time_slice"]),
        ("scenario_time_slice", ["scenario", "time_slice"]),
        ("scenario_time_slice_severity", ["scenario", "time_slice", "severity"]),
        ("city_scenario_time_slice", ["city", "scenario", "time_slice"]),
    ]
    mean_fields = [
        "equal_zone_mean_mean",
        "area_weighted_mean_mean",
        "zone_p90_mean",
        "any_zone_mean",
        "cold_equal_mean",
        "warm_equal_mean",
        "cold_area_mean",
        "warm_area_mean",
        "expected_tsv_equal_mean",
        "expected_tsv_area_mean",
    ]
    frames = []
    for scope, columns in group_specs:
        records = []
        for keys, group in case_continuous.groupby(["model", "event", *columns], sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = {"group_scope": scope}
            for col, value in zip(["model", "event", *columns], keys, strict=True):
                row[col] = value
            weights = group["n_steps"].to_numpy(float)
            row["n_steps"] = int(weights.sum())
            row["n_cases"] = int(len(group))
            for field_name in mean_fields:
                row[field_name] = float(np.average(group[field_name], weights=weights))
            records.append(row)
        frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True, sort=False)


def headline_case_summary(
    case_continuous: pd.DataFrame, case_threshold: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "weather",
        "city",
        "scenario",
        "time_slice",
        "severity",
        "weather_year",
    ]
    continuous_fields = [
        *keys,
        "n_steps",
        "state_hash",
        "equal_zone_mean_mean",
        "area_weighted_mean_mean",
        "zone_p90_mean",
        "any_zone_mean",
        "expected_tsv_equal_mean",
        "expected_tsv_area_mean",
    ]
    threshold_fields = [
        *keys,
        "equal_zone_mean_high_pct",
        "area_weighted_mean_high_pct",
        "zone_p90_high_pct",
        "any_zone_high_pct",
        "hidden_any_zone_pct",
        "hidden_p90_pct",
        "area_weighted_zone_time_high_pct",
        "unweighted_zone_time_high_pct",
    ]
    merged: pd.DataFrame | None = None
    for model, prefix in [
        ("ordinal", "corrected"),
        (STORED_MODEL_NAME, "stored"),
        ("nominal", "nominal"),
        ("no_pmv_ordinal", "no_pmv"),
    ]:
        continuous = case_continuous[
            case_continuous["model"].eq(model)
            & case_continuous["event"].eq("broad_tail")
        ][continuous_fields].copy()
        threshold = case_threshold[
            case_threshold["model"].eq(model)
            & case_threshold["event"].eq("broad_tail")
            & np.isclose(case_threshold["threshold"], 0.20)
        ][threshold_fields].copy()
        model_frame = continuous.merge(threshold, on=keys, how="inner", validate="one_to_one")
        protected = set(keys)
        model_frame = model_frame.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in model_frame.columns
                if column not in protected
            }
        )
        merged = (
            model_frame
            if merged is None
            else merged.merge(model_frame, on=keys, how="inner", validate="one_to_one")
        )
    if merged is None:
        return pd.DataFrame()
    for metric in [
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
    ]:
        merged[f"corrected_minus_stored_{metric}"] = (
            merged[f"corrected_{metric}"] - merged[f"stored_{metric}"]
        )
        merged[f"nominal_minus_corrected_{metric}"] = (
            merged[f"nominal_{metric}"] - merged[f"corrected_{metric}"]
        )
        merged[f"no_pmv_minus_corrected_{metric}"] = (
            merged[f"no_pmv_{metric}"] - merged[f"corrected_{metric}"]
        )
    return merged.sort_values(keys).reset_index(drop=True)


def endpoint_case_summary(
    case_continuous: pd.DataFrame, case_threshold: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "weather",
        "city",
        "scenario",
        "time_slice",
        "severity",
        "weather_year",
    ]
    base = case_continuous[
        case_continuous["model"].eq("ordinal")
        & case_continuous["event"].eq("outermost")
    ].copy()
    keep = [
        *keys,
        "n_steps",
        "state_hash",
        "equal_zone_mean_mean",
        "area_weighted_mean_mean",
        "zone_p90_mean",
        "any_zone_mean",
        "cold_equal_mean",
        "warm_equal_mean",
        "cold_area_mean",
        "warm_area_mean",
    ]
    out = base[keep].copy()
    threshold_fields = [
        "equal_zone_mean_high_pct",
        "area_weighted_mean_high_pct",
        "zone_p90_high_pct",
        "any_zone_high_pct",
        "hidden_any_zone_pct",
        "area_weighted_zone_time_high_pct",
    ]
    source = case_threshold[
        case_threshold["model"].eq("ordinal")
        & case_threshold["event"].eq("outermost")
    ]
    for threshold in ENDPOINT_SCREENS:
        screen = source[np.isclose(source["threshold"], threshold)][
            [*keys, *threshold_fields]
        ].copy()
        token = str(threshold).replace(".", "p")
        screen = screen.rename(
            columns={
                field_name: f"{field_name}_at_{token}"
                for field_name in threshold_fields
            }
        )
        out = out.merge(screen, on=keys, how="inner", validate="one_to_one")
    return out.sort_values(keys).reset_index(drop=True)


def future_contrasts(grouped_threshold: pd.DataFrame) -> pd.DataFrame:
    source = grouped_threshold[
        grouped_threshold["group_scope"].eq("scenario_time_slice_severity")
    ].copy()
    metrics = [
        *[f"{name}_high_pct" for name in AGGREGATORS],
        "hidden_any_zone_pct",
        "hidden_p90_pct",
        "area_weighted_zone_time_high_pct",
    ]
    records = []
    key_columns = ["model", "event", "threshold", "scenario", "severity"]
    for keys, group in source.groupby(key_columns, sort=True):
        baseline = group[group["time_slice"].eq("baseline_2020s")]
        late = group[group["time_slice"].eq("late_2080s")]
        if baseline.empty or late.empty:
            continue
        row = dict(zip(key_columns, keys, strict=True))
        row["baseline_n_steps"] = int(baseline["n_steps"].iloc[0])
        row["late_n_steps"] = int(late["n_steps"].iloc[0])
        for metric in metrics:
            b = float(baseline[metric].iloc[0])
            l = float(late[metric].iloc[0])
            row[f"baseline_{metric}"] = b
            row[f"late_{metric}"] = l
            row[f"late_minus_baseline_{metric}_pp"] = l - b
        records.append(row)

    # Add severity-pooled scenario contrasts directly from case-level-equivalent
    # scenario/time-slice totals already in the grouped table.
    source = grouped_threshold[grouped_threshold["group_scope"].eq("scenario_time_slice")].copy()
    for keys, group in source.groupby(["model", "event", "threshold", "scenario"], sort=True):
        baseline = group[group["time_slice"].eq("baseline_2020s")]
        late = group[group["time_slice"].eq("late_2080s")]
        if baseline.empty or late.empty:
            continue
        row = dict(zip(["model", "event", "threshold", "scenario"], keys, strict=True))
        row["severity"] = "All"
        row["baseline_n_steps"] = int(baseline["n_steps"].iloc[0])
        row["late_n_steps"] = int(late["n_steps"].iloc[0])
        for metric in metrics:
            b = float(baseline[metric].iloc[0])
            l = float(late[metric].iloc[0])
            row[f"baseline_{metric}"] = b
            row[f"late_{metric}"] = l
            row[f"late_minus_baseline_{metric}_pp"] = l - b
        records.append(row)
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(
        columns=[
            "model",
            "event",
            "threshold",
            "scenario",
            "severity",
            "baseline_n_steps",
            "late_n_steps",
        ]
    )


def calibration_bins(
    model: str,
    component: str,
    y_true: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    bin_id = np.digitize(probability, CALIBRATION_EDGES[1:-1], right=False)
    rows = []
    for idx in range(len(CALIBRATION_EDGES) - 1):
        mask = bin_id == idx
        if not mask.any():
            continue
        observed = float(y_true[mask].mean())
        predicted = float(probability[mask].mean())
        n = int(mask.sum())
        rows.append(
            {
                "model": model,
                "component": component,
                "bin_left": float(CALIBRATION_EDGES[idx]),
                "bin_right": float(CALIBRATION_EDGES[idx + 1]),
                "n": n,
                "mean_predicted_probability": predicted,
                "observed_frequency": observed,
                "absolute_calibration_error": abs(predicted - observed),
                "binomial_se": math.sqrt(max(observed * (1.0 - observed), 0.0) / n),
            }
        )
    return pd.DataFrame(rows)


def validation_component_summary(
    model: str,
    component: str,
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: pd.DataFrame,
) -> dict[str, object]:
    weights = bins["n"].to_numpy(float) / len(y_true)
    errors = bins["absolute_calibration_error"].to_numpy(float)
    return {
        "model": model,
        "component": component,
        "n_test": int(len(y_true)),
        "support": int(y_true.sum()),
        "observed_frequency": float(y_true.mean()),
        "mean_predicted_probability": float(probability.mean()),
        "brier": float(np.mean(np.square(probability - y_true))),
        "ece_fixed_bins": float((weights * errors).sum()),
        "mce_fixed_bins": float(errors.max()) if len(errors) else float("nan"),
        "auroc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
    }


def heldout_split(data_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = panel.read_training_data(data_path, sample_limit=None)
    y = panel.round_tsv(df["thermal_sensation"])
    _, hold_df, _, y_hold = train_test_split(
        df, y, test_size=0.30, random_state=42, stratify=y
    )
    _, test_df, _, y_test = train_test_split(
        hold_df, y_hold, test_size=0.50, random_state=42, stratify=y_hold
    )
    return test_df.reset_index(drop=True), y_test


def run_validation(
    bundle: panel.PredictorBundle, data_path: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[validation] reconstructing fixed 15% held-out split", flush=True)
    test_df, y_test = heldout_split(data_path)
    features = panel.build_features_from_raw(test_df, bundle.spec)
    probs_by_model = {
        "ordinal": bundle.predict_ordinal(features),
        "nominal": bundle.predict_nominal(features),
    }
    support = pd.DataFrame(
        [
            {
                "tsv": signed,
                "class_index": index,
                "support": int((y_test == index).sum()),
                "share_pct": float((y_test == index).mean() * 100.0),
            }
            for index, signed in enumerate(range(-3, 4))
        ]
    )
    summaries = []
    calibration_frames = []
    threshold_rows = []
    recall_rows = []
    for model, probs in probs_by_model.items():
        pred = probs.argmax(axis=1)
        for index, signed in enumerate(range(-3, 4)):
            mask = y_test == index
            recall_rows.append(
                {
                    "model": model,
                    "tsv": signed,
                    "support": int(mask.sum()),
                    "argmax_recall": float((pred[mask] == index).mean()) if mask.any() else float("nan"),
                    "predicted_argmax_share_pct": float((pred == index).mean() * 100.0),
                }
            )
        components = {
            "outermost": ((y_test == 0) | (y_test == 6), probs[:, [0, 6]].sum(axis=1)),
            "cold_endpoint": (y_test == 0, probs[:, 0]),
            "warm_endpoint": (y_test == 6, probs[:, 6]),
            "broad_tail": (np.isin(y_test, TAIL_INDEX), probs[:, TAIL_INDEX].sum(axis=1)),
        }
        for component, (truth, probability) in components.items():
            truth = np.asarray(truth, dtype=float)
            bins = calibration_bins(model, component, truth, probability)
            calibration_frames.append(bins)
            summaries.append(
                validation_component_summary(model, component, truth, probability, bins)
            )
            screens = ENDPOINT_SCREENS if component != "broad_tail" else BROAD_KEY_SCREENS
            for threshold in screens:
                decision = probability >= threshold
                threshold_rows.append(
                    {
                        "model": model,
                        "component": component,
                        "threshold": float(threshold),
                        "predicted_positive_pct": float(decision.mean() * 100.0),
                        "precision": float(precision_score(truth, decision, zero_division=0)),
                        "recall": float(recall_score(truth, decision, zero_division=0)),
                        "f1": float(f1_score(truth, decision, zero_division=0)),
                    }
                )
    support.to_csv(output_dir / "heldout_class_support.csv", index=False)
    recall = pd.DataFrame(recall_rows)
    recall.to_csv(output_dir / "heldout_class_recall.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "heldout_event_validation_summary.csv", index=False)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    calibration.to_csv(output_dir / "heldout_event_calibration_bins.csv", index=False)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(output_dir / "heldout_event_threshold_metrics.csv", index=False)
    return support, recall, summary, calibration


def global_continuous_frame(
    distributions: dict[tuple[str, str, str], DistributionAccumulator]
) -> pd.DataFrame:
    rows = []
    for (model, event, metric), accumulator in sorted(distributions.items()):
        rows.append({"model": model, "event": event, "metric": metric, **accumulator.record()})
    return pd.DataFrame(rows)


def global_pair_frame(
    pairs: dict[tuple[str, str], PairAccumulator]
) -> pd.DataFrame:
    rows = []
    for (event, aggregator), accumulator in sorted(pairs.items()):
        rows.append({"event": event, "aggregator": aggregator, **accumulator.record()})
    return pd.DataFrame(rows)


def relabel_old_new_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "ordinal_mean": "stored_mean",
            "nominal_mean": "corrected_mean",
            "nominal_minus_ordinal_mean": "corrected_minus_stored_mean",
        }
    )


def relabel_old_new_threshold_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "ordinal_high_count": "stored_high_count",
            "nominal_high_count": "corrected_high_count",
            "ordinal_only_count": "stored_only_count",
            "nominal_only_count": "corrected_only_count",
            "ordinal_high_pct": "stored_high_pct",
            "nominal_high_pct": "corrected_high_pct",
            "nominal_minus_ordinal_pp": "corrected_minus_stored_pp",
        }
    )


def decision_rules(
    global_threshold: pd.DataFrame,
    paired_threshold: pd.DataFrame,
    global_continuous: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    endpoint = global_threshold[
        global_threshold["event"].eq("outermost")
        & global_threshold["model"].eq("ordinal")
        & global_threshold["threshold"].isin(ENDPOINT_SCREENS)
    ]
    endpoint_persists = bool(
        len(endpoint) == len(ENDPOINT_SCREENS)
        and (endpoint["any_zone_high_pct"] > endpoint["equal_zone_mean_high_pct"]).all()
        and (endpoint["hidden_any_zone_pct"] > 0).all()
    )
    rows.append(
        {
            "rule": "endpoint_spatial_underaggregation_all_prespecified_screens",
            "passed": endpoint_persists,
            "material_claim_narrowing": False,
            "detail": (
                "Any-zone exceeds equal-zone-mean and hidden-any-zone is nonzero at every "
                "endpoint screen."
                if endpoint_persists
                else "Endpoint spatial pattern does not persist at every prespecified screen."
            ),
        }
    )

    broad = global_threshold[
        global_threshold["event"].eq("broad_tail")
        & global_threshold["threshold"].isin(BROAD_KEY_SCREENS)
        & global_threshold["model"].isin(MODEL_NAMES)
    ]
    spatial_by_model = broad.groupby("model").apply(
        lambda g: bool(
            len(g) == len(BROAD_KEY_SCREENS)
            and (g["any_zone_high_pct"] > g["equal_zone_mean_high_pct"]).all()
            and (g["hidden_any_zone_pct"] > 0).all()
        ),
        include_groups=False,
    )
    representation_spatial = bool(
        set(spatial_by_model.index) == set(MODEL_NAMES) and spatial_by_model.all()
    )
    rows.append(
        {
            "rule": "broad_tail_spatial_direction_both_models",
            "passed": representation_spatial,
            "material_claim_narrowing": not representation_spatial,
            "detail": json.dumps(spatial_by_model.to_dict(), sort_keys=True),
        }
    )

    at_020 = paired_threshold[
        paired_threshold["event"].eq("broad_tail")
        & np.isclose(paired_threshold["threshold"], 0.20)
    ]
    max_abs_rate_diff = float(at_020["nominal_minus_ordinal_pp"].abs().max())
    material_rate = max_abs_rate_diff >= MATERIAL_RATE_DIFFERENCE_PP
    rows.append(
        {
            "rule": "broad_tail_model_rate_difference_at_0_20",
            "passed": not material_rate,
            "material_claim_narrowing": False,
            "detail": (
                f"Maximum absolute nominal-minus-ordinal difference across aggregators: "
                f"{max_abs_rate_diff:.3f} percentage points; disclosure threshold "
                f"{MATERIAL_RATE_DIFFERENCE_PP:.1f} pp."
            ),
        }
    )

    old_new = global_threshold[
        global_threshold["event"].eq("broad_tail")
        & global_threshold["threshold"].isin(BROAD_KEY_SCREENS)
        & global_threshold["model"].isin([STORED_MODEL_NAME, "ordinal"])
    ]
    old_new_spatial = old_new.groupby("model").apply(
        lambda g: bool(
            len(g) == len(BROAD_KEY_SCREENS)
            and (g["any_zone_high_pct"] > g["equal_zone_mean_high_pct"]).all()
            and (g["hidden_any_zone_pct"] > 0).all()
        ),
        include_groups=False,
    )
    old_new_spatial_pass = bool(
        set(old_new_spatial.index) == {STORED_MODEL_NAME, "ordinal"}
        and old_new_spatial.all()
    )
    rows.append(
        {
            "rule": "stored_vs_corrected_spatial_direction",
            "passed": old_new_spatial_pass,
            "material_claim_narrowing": not old_new_spatial_pass,
            "detail": json.dumps(old_new_spatial.to_dict(), sort_keys=True),
        }
    )

    old_new_020 = global_threshold[
        global_threshold["event"].eq("broad_tail")
        & np.isclose(global_threshold["threshold"], 0.20)
        & global_threshold["model"].isin([STORED_MODEL_NAME, "ordinal"])
    ]
    old_new_pivot = old_new_020.set_index("model")
    old_new_metrics = [
        "equal_zone_mean_high_pct",
        "area_weighted_mean_high_pct",
        "zone_p90_high_pct",
        "any_zone_high_pct",
        "hidden_any_zone_pct",
    ]
    if set(old_new_pivot.index) == {STORED_MODEL_NAME, "ordinal"}:
        old_new_differences = (
            old_new_pivot.loc["ordinal", old_new_metrics]
            - old_new_pivot.loc[STORED_MODEL_NAME, old_new_metrics]
        ).astype(float)
        max_old_new_rate = float(old_new_differences.abs().max())
    else:
        max_old_new_rate = float("nan")
    rows.append(
        {
            "rule": "stored_vs_corrected_rate_difference_at_0_20",
            "passed": bool(
                np.isfinite(max_old_new_rate)
                and max_old_new_rate < MATERIAL_RATE_DIFFERENCE_PP
            ),
            "material_claim_narrowing": False,
            "detail": (
                f"Maximum corrected-minus-stored absolute difference across headline "
                f"aggregators: {max_old_new_rate:.3f} pp; disclosure threshold "
                f"{MATERIAL_RATE_DIFFERENCE_PP:.1f} pp."
            ),
        }
    )

    mean_rows = global_continuous[
        global_continuous["event"].eq("broad_tail")
        & global_continuous["metric"].isin(AGGREGATORS)
    ]
    pivot = mean_rows.pivot(index="metric", columns="model", values="mean")
    max_mean_diff = float((pivot["nominal"] - pivot["ordinal"]).abs().max())
    material_mean = max_mean_diff >= MATERIAL_MEAN_PROBABILITY_DIFFERENCE
    rows.append(
        {
            "rule": "broad_tail_mean_probability_model_difference",
            "passed": not material_mean,
            "material_claim_narrowing": False,
            "detail": (
                f"Maximum absolute mean-probability difference: {max_mean_diff:.5f}; "
                f"disclosure threshold {MATERIAL_MEAN_PROBABILITY_DIFFERENCE:.3f}."
            ),
        }
    )

    future = contrasts[
        contrasts["event"].eq("broad_tail")
        & contrasts["threshold"].isin(BROAD_KEY_SCREENS)
        & contrasts["severity"].eq("All")
        & contrasts["scenario"].eq("ssp585")
        & contrasts["model"].isin(MODEL_NAMES)
    ].copy()
    direction_records = []
    if not future.empty:
        threshold_values = pd.to_numeric(future["threshold"], errors="coerce").to_numpy(float)
        for threshold in BROAD_KEY_SCREENS:
            subset = future[np.isclose(threshold_values, threshold)]
            if set(subset["model"]) != set(MODEL_NAMES):
                continue
            for metric in [
                "equal_zone_mean_high_pct",
                "area_weighted_mean_high_pct",
                "zone_p90_high_pct",
                "any_zone_high_pct",
                "hidden_any_zone_pct",
            ]:
                field_name = f"late_minus_baseline_{metric}_pp"
                values = subset.set_index("model")[field_name]
                left = float(values["ordinal"])
                right = float(values["nominal"])
                same = (
                    abs(left) < NEGLIGIBLE_CONTRAST_PP
                    and abs(right) < NEGLIGIBLE_CONTRAST_PP
                ) or np.sign(left) == np.sign(right)
                direction_records.append(same)
    preserved_fraction = float(np.mean(direction_records)) if direction_records else float("nan")
    future_pass = bool(direction_records and preserved_fraction >= 0.80)
    rows.append(
        {
            "rule": "ssp585_late_vs_baseline_direction_preserved",
            "passed": future_pass,
            "material_claim_narrowing": bool(direction_records) and not future_pass,
            "detail": (
                f"Direction agreement across prespecified screens/aggregators: "
                f"{preserved_fraction * 100.0:.1f}% (negligible if both <"
                f"{NEGLIGIBLE_CONTRAST_PP:.1f} pp)."
                if direction_records
                else "Not evaluated because the selected run lacks both baseline and late-2080s groups."
            ),
        }
    )

    future_old_new = contrasts[
        contrasts["event"].eq("broad_tail")
        & contrasts["threshold"].isin(BROAD_KEY_SCREENS)
        & contrasts["severity"].eq("All")
        & contrasts["scenario"].eq("ssp585")
        & contrasts["model"].isin([STORED_MODEL_NAME, "ordinal"])
    ].copy()
    old_new_direction_records = []
    if not future_old_new.empty:
        threshold_values = pd.to_numeric(
            future_old_new["threshold"], errors="coerce"
        ).to_numpy(float)
        for threshold in BROAD_KEY_SCREENS:
            subset = future_old_new[np.isclose(threshold_values, threshold)]
            if set(subset["model"]) != {STORED_MODEL_NAME, "ordinal"}:
                continue
            for metric in [
                "equal_zone_mean_high_pct",
                "area_weighted_mean_high_pct",
                "zone_p90_high_pct",
                "any_zone_high_pct",
                "hidden_any_zone_pct",
            ]:
                field_name = f"late_minus_baseline_{metric}_pp"
                values = subset.set_index("model")[field_name]
                stored_value = float(values[STORED_MODEL_NAME])
                corrected_value = float(values["ordinal"])
                same = (
                    abs(stored_value) < NEGLIGIBLE_CONTRAST_PP
                    and abs(corrected_value) < NEGLIGIBLE_CONTRAST_PP
                ) or np.sign(stored_value) == np.sign(corrected_value)
                old_new_direction_records.append(same)
    old_new_preserved_fraction = (
        float(np.mean(old_new_direction_records))
        if old_new_direction_records
        else float("nan")
    )
    old_new_future_pass = bool(
        old_new_direction_records and old_new_preserved_fraction >= 0.80
    )
    rows.append(
        {
            "rule": "stored_vs_corrected_ssp585_future_direction",
            "passed": old_new_future_pass,
            "material_claim_narrowing": bool(old_new_direction_records)
            and not old_new_future_pass,
            "detail": (
                f"Direction agreement across prespecified screens/aggregators: "
                f"{old_new_preserved_fraction * 100.0:.1f}%."
                if old_new_direction_records
                else "Not evaluated because the selected run lacks both baseline and late-2080s groups."
            ),
        }
    )
    return pd.DataFrame(rows)


def make_plots(
    output_dir: Path,
    global_threshold: pd.DataFrame,
    case_continuous: pd.DataFrame,
    calibration: pd.DataFrame | None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    colors = {"ordinal": "#2b6cb0", "nominal": "#c55a11"}
    for model in MODEL_NAMES:
        subset = global_threshold[
            global_threshold["event"].eq("broad_tail")
            & global_threshold["model"].eq(model)
        ].sort_values("threshold")
        axes[0].plot(
            subset["threshold"],
            subset["equal_zone_mean_high_pct"],
            color=colors[model],
            lw=1.8,
            label=f"{model.title()} mean",
        )
        axes[0].plot(
            subset["threshold"],
            subset["any_zone_high_pct"],
            color=colors[model],
            lw=1.3,
            ls="--",
            label=f"{model.title()} any-zone",
        )
        axes[1].plot(
            subset["threshold"],
            subset["hidden_any_zone_pct"],
            color=colors[model],
            lw=1.8,
            label=model.title(),
        )
    endpoint = global_threshold[
        global_threshold["event"].eq("outermost")
        & global_threshold["model"].eq("ordinal")
    ].sort_values("threshold")
    for field_name, label, style in [
        ("equal_zone_mean_high_pct", "Equal-zone mean", "-"),
        (
            "area_weighted_mean_high_pct",
            "Area-weighted mean probability",
            "-.",
        ),
        ("zone_p90_high_pct", "Zone p90", ":"),
        ("any_zone_high_pct", "Any-zone", "--"),
    ]:
        axes[2].plot(
            endpoint["threshold"],
            endpoint[field_name],
            lw=1.7,
            ls=style,
            label=label,
        )
    axes[0].set_title("Broad tail: model comparison")
    axes[1].set_title("Broad tail: mean-hidden any-zone")
    axes[2].set_title("Endpoint-only ordinal sensitivity")
    for ax in axes:
        ax.set_xlabel("Probability screen")
        ax.set_ylabel("Occupied timesteps (%)")
        ax.grid(color="#dddddd", lw=0.5)
        ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(output_dir / "robustness_threshold_curves.png", dpi=220)
    fig.savefig(output_dir / "robustness_threshold_curves.pdf")
    plt.close(fig)

    broad_case = case_continuous[case_continuous["event"].eq("broad_tail")]
    pivot = broad_case.pivot(index="weather", columns="model", values="equal_zone_mean_mean")
    fig, ax = plt.subplots(figsize=(5.1, 4.6), constrained_layout=True)
    ax.scatter(pivot["ordinal"], pivot["nominal"], s=18, alpha=0.72, color="#4c78a8")
    lo = float(min(pivot.min()))
    hi = float(max(pivot.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color="#555555")
    ax.set_xlabel("Ordinal case-mean broad-tail probability")
    ax.set_ylabel("Nominal case-mean broad-tail probability")
    ax.grid(color="#dddddd", lw=0.5)
    fig.savefig(output_dir / "ordinal_nominal_case_mean_scatter.png", dpi=220)
    fig.savefig(output_dir / "ordinal_nominal_case_mean_scatter.pdf")
    plt.close(fig)

    if calibration is not None and not calibration.empty:
        endpoint_cal = calibration[calibration["component"].eq("outermost")]
        fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
        for model in MODEL_NAMES:
            group = endpoint_cal[endpoint_cal["model"].eq(model)]
            ax.errorbar(
                group["mean_predicted_probability"],
                group["observed_frequency"],
                yerr=1.96 * group["binomial_se"],
                marker="o",
                ms=4,
                lw=1.3,
                capsize=2,
                color=colors[model],
                label=model.title(),
            )
        ax.plot([0, 0.5], [0, 0.5], ls="--", lw=0.9, color="#555555")
        ax.set_xlim(0, 0.35)
        ax.set_ylim(0, 0.35)
        ax.set_xlabel("Mean predicted endpoint-only probability")
        ax.set_ylabel("Observed endpoint frequency")
        ax.grid(color="#dddddd", lw=0.5)
        ax.legend(frameon=False)
        fig.savefig(output_dir / "endpoint_calibration.png", dpi=220)
        fig.savefig(output_dir / "endpoint_calibration.pdf")
        plt.close(fig)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    view = df.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy())
    return "\n".join(lines)


def write_summary(
    output_dir: Path,
    paths: list[Path],
    global_continuous: pd.DataFrame,
    global_threshold: pd.DataFrame,
    paired_threshold: pd.DataFrame,
    old_new_threshold: pd.DataFrame,
    scalar_global: pd.DataFrame,
    validation: pd.DataFrame | None,
    decisions: pd.DataFrame,
    parity: pd.DataFrame,
) -> None:
    endpoint_cont = global_continuous[
        global_continuous["model"].eq("ordinal")
        & global_continuous["event"].eq("outermost")
        & global_continuous["metric"].isin(
            ["equal_zone_mean", "area_weighted_mean", "zone_p90", "any_zone", "cold_equal", "warm_equal"]
        )
    ][["metric", "n", "mean", "p50_approx", "p90_approx", "p95_approx", "p99_approx", "max"]]
    endpoint_screen = global_threshold[
        global_threshold["model"].eq("ordinal")
        & global_threshold["event"].eq("outermost")
        & global_threshold["threshold"].isin(ENDPOINT_SCREENS)
    ][
        [
            "threshold",
            "equal_zone_mean_high_pct",
            "area_weighted_mean_high_pct",
            "zone_p90_high_pct",
            "any_zone_high_pct",
            "hidden_any_zone_pct",
            "area_weighted_zone_time_high_pct",
        ]
    ]
    broad_020 = global_threshold[
        global_threshold["event"].eq("broad_tail")
        & np.isclose(global_threshold["threshold"], 0.20)
    ][
        [
            "model",
            "equal_zone_mean_high_pct",
            "area_weighted_mean_high_pct",
            "zone_p90_high_pct",
            "any_zone_high_pct",
            "hidden_any_zone_pct",
            "area_weighted_zone_time_high_pct",
        ]
    ]
    comparison_020 = paired_threshold[
        paired_threshold["event"].eq("broad_tail")
        & np.isclose(paired_threshold["threshold"], 0.20)
    ][
        [
            "aggregator",
            "ordinal_high_pct",
            "nominal_high_pct",
            "nominal_minus_ordinal_pp",
            "disagreement_pct",
            "jaccard",
        ]
    ]
    old_new_020 = old_new_threshold[
        old_new_threshold["event"].eq("broad_tail")
        & np.isclose(old_new_threshold["threshold"], 0.20)
    ][
        [
            "aggregator",
            "stored_high_pct",
            "corrected_high_pct",
            "corrected_minus_stored_pp",
            "disagreement_pct",
            "jaccard",
        ]
    ]
    lines = [
        "# Endpoint-only and nominal-model robustness",
        "",
        "## Scope",
        "",
        f"- {len(paths)} frozen EnergyPlus case traces; no building simulation rerun.",
        "- Both saved predictors were applied to identical occupied zone-level states.",
        "- `outermost` means `P(TSV=-3) + P(TSV=+3)`; it is an endpoint-only bounding sensitivity.",
        "- Quantiles in the global continuous table use a deterministic probability histogram with resolution 0.0001.",
        "",
        "## Endpoint-only continuous summary: primary ordinal model",
        "",
        markdown_table(endpoint_cont),
        "",
        "## Endpoint-only spatial screens: primary ordinal model",
        "",
        markdown_table(endpoint_screen),
        "",
        "## Broad-tail results at the manuscript's 0.20 screen",
        "",
        markdown_table(broad_020),
        "",
        "## Paired ordinal-versus-nominal comparison at 0.20",
        "",
        markdown_table(comparison_020),
        "",
        "## Stored-versus-corrected same-state ordinal comparison at 0.20",
        "",
        markdown_table(old_new_020),
        "",
        "## Prespecified decision rules",
        "",
        markdown_table(decisions),
        "",
        "## Synchronized PMV scalar audit",
        "",
        markdown_table(
            scalar_global[
                [
                    "n_steps",
                    "corrected_mean_pmv_mean",
                    "stored_mean_pmv_mean",
                    "corrected_vs_stored_pmv_mae",
                    "corrected_vs_stored_pmv_rmse",
                    "corrected_vs_stored_pmv_max_abs",
                    "corrected_vs_stored_pmv_pearson_r",
                    "corrected_abs_pmv_vs_p_tail_pearson_r",
                    "corrected_pmv_central_but_tail_020_pct",
                ]
            ]
        ),
        "",
        "## State-alignment audit",
        "",
        (
            f"- Maximum absolute broad-tail probability difference between corrected same-state "
            f"ordinal zone values and stored callback-timed values: "
            f"{parity['max_abs_zone_p_tail_error'].max():.3e}."
        ),
        (
            f"- Maximum absolute expected-TSV difference: "
            f"{parity['max_abs_zone_expected_tsv_error'].max():.3e}."
        ),
        "- The mismatch is preserved rather than reconciled away: the callback that generated "
        "the stored probability ran at the beginning of the zone timestep, whereas Ta/MRT/RH "
        "on the CSV row were recorded at its end. The corrected comparison predicts both "
        "models from the recorded state on that row.",
        "",
    ]
    if validation is not None:
        endpoint_validation = validation[validation["component"].isin(
            ["outermost", "cold_endpoint", "warm_endpoint"]
        )][
            [
                "model",
                "component",
                "n_test",
                "support",
                "observed_frequency",
                "mean_predicted_probability",
                "brier",
                "ece_fixed_bins",
                "auroc",
                "average_precision",
            ]
        ]
        lines.extend(
            [
                "## Held-out endpoint support and calibration",
                "",
                markdown_table(endpoint_validation),
                "",
                "The endpoint event is unambiguous but thin: its two labels have substantially "
                "less support than the middle classes. Endpoint probability results therefore "
                "bound the primary analysis and should not be presented as a more stable or "
                "better-validated replacement outcome.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation safeguards",
            "",
            "- Screens are diagnostic conventions, not dissatisfaction, acceptability, or compliance rates.",
            "- Model agreement in direction does not erase disclosed magnitude or classification disagreements.",
            "- A nonzero endpoint pattern supports only the claim that the spatial diagnostic is not created solely by including TSV ±2.",
            "- Full paths, versions, checksums, state hashes, thresholds, and decision rules are recorded in this directory.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_callback_timing_evidence(output_dir: Path) -> None:
    source = REBUILD_SCRIPTS / "run_medium_office_diagnostic_panel.py"
    lines = source.read_text(encoding="utf-8").splitlines()

    def first_line(fragment: str) -> int:
        for index, line in enumerate(lines, start=1):
            if fragment in line:
                return index
        return -1

    apply_registration = first_line(
        "callback_begin_zone_timestep_after_init_heat_balance(state, apply_control)"
    )
    record_registration = first_line(
        "callback_end_zone_timestep_after_zone_reporting(state, record)"
    )
    probability_call = first_line(
        'signal = probability_diagnostic_signal(ctl, values, oat, predictor="ordinal")'
    )
    environment_write = first_line("add_zone_environment_record_fields(rec, values)")
    text = f"""# Callback timing evidence

Source: `{source.resolve()}`

Source SHA-256: `{file_sha256(source)}`

- The diagnostic probability is calculated from `values` in `apply_control`
  (line {probability_call}).
- `apply_control` is registered at the **beginning** of the zone timestep after
  heat-balance initialization (line {apply_registration}).
- `record` is registered at the **end** of the zone timestep after zone
  reporting (line {record_registration}).
- The zone Ta/MRT/RH fields written to the trace row come from the later
  `record` callback (line {environment_write}).

Consequently, the stored probability and the environmental values appearing on
the same CSV row were sampled at different callback points. The robustness run
does not overwrite or reinterpret those stored values. It applies the ordinal
and nominal predictors to the identical recorded end-of-step state and labels
that result `corrected same-state inference`; the per-case differences from the
stored callback-timed probabilities are retained in `ordinal_trace_parity.csv`
and the paired threshold outputs.
"""
    (output_dir / "callback_timing_evidence.md").write_text(text, encoding="utf-8")


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "joblib",
        "pythermalcomfort",
        "matplotlib",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_output_checksums(output_dir: Path) -> None:
    records = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "output_checksums.csv":
            records.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    pd.DataFrame(records).to_csv(output_dir / "output_checksums.csv", index=False)


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = trace_paths(args.trace_dir, args.max_cases)
    bundle = load_bundle(args.model_path)
    no_pmv_bundle = load_bundle(args.no_pmv_model_path)
    weights = np.array([ZONE_AREAS_M2[slug] for slug in panel.ZONE_FIELD_NAMES], dtype=float)
    weights /= weights.sum()

    config = {
        "command": sys.argv,
        "started_unix": started,
        "trace_dir": str(args.trace_dir.resolve()),
        "model_path": str(args.model_path.resolve()),
        "no_pmv_model_path": str(args.no_pmv_model_path.resolve()),
        "data_path": str(args.data_path.resolve()),
        "panel_manifest": str(args.panel_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "case_count_requested": len(paths),
        "random_seed": 42,
        "tsv_values": TSV_VALUES.tolist(),
        "broad_tail_classes": [-3, -2, 2, 3],
        "outermost_classes": [-3, 3],
        "endpoint_screens": ENDPOINT_SCREENS.tolist(),
        "broad_curve_screens": BROAD_CURVE_SCREENS.tolist(),
        "broad_key_screens": BROAD_KEY_SCREENS.tolist(),
        "calibration_edges": CALIBRATION_EDGES.tolist(),
        "zone_areas_m2": ZONE_AREAS_M2,
        "zone_area_total_m2": float(sum(ZONE_AREAS_M2.values())),
        "quantile_histogram_bins": HIST_BINS,
        "material_rate_difference_pp": MATERIAL_RATE_DIFFERENCE_PP,
        "material_mean_probability_difference": MATERIAL_MEAN_PROBABILITY_DIFFERENCE,
        "negligible_contrast_pp": NEGLIGIBLE_CONTRAST_PP,
        "software_versions": software_versions(),
        "input_checksums": {
            "model_sha256": file_sha256(args.model_path),
            "no_pmv_model_sha256": file_sha256(args.no_pmv_model_path),
            "script_sha256": file_sha256(SCRIPT),
            "data_sha256": None if args.skip_validation else file_sha256(args.data_path),
            "panel_manifest_sha256": (
                file_sha256(args.panel_manifest) if args.panel_manifest.exists() else None
            ),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_corrected_zone_schema(args.output_dir)

    distributions: dict[tuple[str, str, str], DistributionAccumulator] = {}
    pairs: dict[tuple[str, str], PairAccumulator] = {}
    old_new_pairs: dict[tuple[str, str], PairAccumulator] = {}
    threshold_totals: dict[tuple[str, str, float], dict[str, float]] = {}
    paired_threshold_totals: dict[tuple[str, float, str], dict[str, float]] = {}
    old_new_threshold_totals: dict[tuple[str, float, str], dict[str, float]] = {}
    case_continuous_rows: list[dict[str, object]] = []
    case_threshold_rows: list[dict[str, object]] = []
    paired_case_rows: list[dict[str, object]] = []
    old_new_case_rows: list[dict[str, object]] = []
    scalar_case_rows: list[dict[str, object]] = []
    old_new_pmv_pair = PairAccumulator()
    pmv_expected_pair = PairAccumulator()
    abs_pmv_tail_pair = PairAccumulator()
    parity_rows: list[dict[str, object]] = []
    trace_manifest_rows: list[dict[str, object]] = []

    cached_hash: str | None = None
    cached_probabilities: dict[str, np.ndarray] | None = None
    zone_npz_checksums: dict[str, str] = {}
    for case_index, path in enumerate(paths, start=1):
        case_started = time.time()
        case = read_case(path)
        base = metadata(str(case["weather"]))
        state_hash = state_sha256(
            np.ones(len(np.asarray(case["running_mean"])), dtype=np.uint8),
            np.asarray(case["running_mean"]),
            np.asarray(case["ta"]),
            np.asarray(case["tr"]),
            np.asarray(case["rh"]),
        )
        reused = state_hash == cached_hash and cached_probabilities is not None
        if reused:
            probabilities = cached_probabilities
        else:
            probabilities = infer_models(bundle, no_pmv_bundle, case)
            cached_hash = state_hash
            cached_probabilities = probabilities
        n_steps = int(np.asarray(case["ta"]).shape[0])
        ordinal = probabilities["ordinal"]
        if state_hash in zone_npz_checksums:
            zone_npz_path = (
                args.output_dir / "corrected_zone_npz" / f"{state_hash}.npz"
            )
            zone_npz_sha256 = zone_npz_checksums[state_hash]
        else:
            zone_npz_path, zone_npz_sha256 = write_corrected_zone_npz(
                args.output_dir,
                state_hash,
                case,
                ordinal,
                np.asarray(probabilities["zone_pmv"]),
                np.asarray(probabilities["no_pmv_ordinal"]),
            )
            zone_npz_checksums[state_hash] = zone_npz_sha256
        trace_manifest_rows.append(
            {
                **base,
                "trace_path": str(path.resolve()),
                "file_bytes": path.stat().st_size,
                "file_mtime_ns": path.stat().st_mtime_ns,
                "analysis_state_sha256": state_hash,
                "occupied_steps": n_steps,
                "reused_previous_state_inference": reused,
                "corrected_zone_npz": str(zone_npz_path.resolve()),
                "corrected_zone_npz_sha256": zone_npz_sha256,
            }
        )

        ordinal_broad = ordinal[:, :, TAIL_INDEX].sum(axis=2)
        ordinal_mu = ordinal @ TSV_VALUES
        no_pmv_for_scalar = np.asarray(probabilities["no_pmv_ordinal"])
        no_pmv_broad_for_scalar = no_pmv_for_scalar[:, :, TAIL_INDEX].sum(axis=2)
        no_pmv_mu_for_scalar = no_pmv_for_scalar @ TSV_VALUES
        corrected_zone_pmv = np.asarray(probabilities["zone_pmv"], dtype=float)
        corrected_mean_pmv = corrected_zone_pmv.mean(axis=1)
        stored_mean_pmv = np.asarray(case["stored_mean_pmv"], dtype=float)
        corrected_equal_p_tail = ordinal_broad.mean(axis=1)
        corrected_equal_mu = ordinal_mu.mean(axis=1)
        scalar_case_rows.append(
            scalar_case_record(
                base,
                corrected_zone_pmv,
                stored_mean_pmv,
                corrected_equal_p_tail,
                np.asarray(case["stored_equal_p"]),
                corrected_equal_mu,
                no_pmv_broad_for_scalar.mean(axis=1),
                no_pmv_mu_for_scalar.mean(axis=1),
                weights,
            )
        )
        old_new_pmv_pair.update(stored_mean_pmv, corrected_mean_pmv)
        pmv_expected_pair.update(corrected_mean_pmv, corrected_equal_mu)
        abs_pmv_tail_pair.update(np.abs(corrected_mean_pmv), corrected_equal_p_tail)
        parity_rows.append(
            {
                **base,
                "max_abs_zone_p_tail_error": float(
                    np.nanmax(np.abs(ordinal_broad - np.asarray(case["stored_p"])))
                ),
                "mean_abs_zone_p_tail_error": float(
                    np.nanmean(np.abs(ordinal_broad - np.asarray(case["stored_p"])))
                ),
                "max_abs_zone_expected_tsv_error": float(
                    np.nanmax(np.abs(ordinal_mu - np.asarray(case["stored_mu"])))
                ),
                "mean_abs_zone_expected_tsv_error": float(
                    np.nanmean(np.abs(ordinal_mu - np.asarray(case["stored_mu"])))
                ),
                "max_abs_equal_p_tail_error": float(
                    np.nanmax(
                        np.abs(ordinal_broad.mean(axis=1) - np.asarray(case["stored_equal_p"]))
                    )
                ),
                "max_abs_equal_expected_tsv_error": float(
                    np.nanmax(
                        np.abs(ordinal_mu.mean(axis=1) - np.asarray(case["stored_equal_mu"]))
                    )
                ),
                "max_abs_mean_pmv_error": float(
                    np.nanmax(np.abs(corrected_mean_pmv - stored_mean_pmv))
                ),
                "mean_abs_mean_pmv_error": float(
                    np.nanmean(np.abs(corrected_mean_pmv - stored_mean_pmv))
                ),
            }
        )

        aggregate_by_model_event: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        zone_by_model_event: dict[tuple[str, str], np.ndarray] = {}
        for model in MODEL_NAMES:
            probs = probabilities[model]
            mu = probs @ TSV_VALUES
            mu_equal = mu.mean(axis=1)
            mu_area = mu @ weights
            for event in ("broad_tail", "outermost"):
                zone_probability, cold, warm = event_arrays(probs, event)
                aggregate = aggregations(zone_probability, weights)
                aggregate_by_model_event[(model, event)] = aggregate
                zone_by_model_event[(model, event)] = zone_probability
                continuous: dict[str, object] = {
                    **base,
                    "model": model,
                    "event": event,
                    "n_steps": n_steps,
                    "state_hash": state_hash,
                    "expected_tsv_equal_mean": float(mu_equal.mean()),
                    "expected_tsv_area_mean": float(mu_area.mean()),
                    "cold_equal_mean": float(cold.mean(axis=1).mean()),
                    "warm_equal_mean": float(warm.mean(axis=1).mean()),
                    "cold_area_mean": float((cold @ weights).mean()),
                    "warm_area_mean": float((warm @ weights).mean()),
                }
                for aggregator, values in aggregate.items():
                    continuous.update(exact_case_distribution(values, aggregator))
                    distributions.setdefault(
                        (model, event, aggregator), DistributionAccumulator()
                    ).update(values)
                for component_name, component_values in [
                    ("cold_equal", cold.mean(axis=1)),
                    ("warm_equal", warm.mean(axis=1)),
                    ("cold_area", cold @ weights),
                    ("warm_area", warm @ weights),
                ]:
                    distributions.setdefault(
                        (model, event, component_name), DistributionAccumulator()
                    ).update(component_values)
                case_continuous_rows.append(continuous)

                screens = ENDPOINT_SCREENS if event == "outermost" else BROAD_CURVE_SCREENS
                for threshold in screens:
                    record = threshold_record(
                        base,
                        model,
                        event,
                        float(threshold),
                        zone_probability,
                        aggregate,
                        weights,
                    )
                    case_threshold_rows.append(record)
                    update_global_threshold(threshold_totals, record)

        # Recompute the separately fitted no-PMV ordinal sensitivity on the
        # synchronized end-of-step state needed for the PMV-feature figure.
        no_pmv_probs = probabilities["no_pmv_ordinal"]
        no_pmv_mu = no_pmv_probs @ TSV_VALUES
        no_pmv_zone_probability, no_pmv_cold, no_pmv_warm = event_arrays(
            no_pmv_probs, "broad_tail"
        )
        no_pmv_aggregate = aggregations(no_pmv_zone_probability, weights)
        aggregate_by_model_event[("no_pmv_ordinal", "broad_tail")] = no_pmv_aggregate
        zone_by_model_event[("no_pmv_ordinal", "broad_tail")] = no_pmv_zone_probability
        no_pmv_continuous: dict[str, object] = {
            **base,
            "model": "no_pmv_ordinal",
            "event": "broad_tail",
            "n_steps": n_steps,
            "state_hash": state_hash,
            "expected_tsv_equal_mean": float(no_pmv_mu.mean(axis=1).mean()),
            "expected_tsv_area_mean": float((no_pmv_mu @ weights).mean()),
            "cold_equal_mean": float(no_pmv_cold.mean(axis=1).mean()),
            "warm_equal_mean": float(no_pmv_warm.mean(axis=1).mean()),
            "cold_area_mean": float((no_pmv_cold @ weights).mean()),
            "warm_area_mean": float((no_pmv_warm @ weights).mean()),
        }
        for aggregator, values in no_pmv_aggregate.items():
            no_pmv_continuous.update(exact_case_distribution(values, aggregator))
            distributions.setdefault(
                ("no_pmv_ordinal", "broad_tail", aggregator),
                DistributionAccumulator(),
            ).update(values)
        for component_name, component_values in [
            ("cold_equal", no_pmv_cold.mean(axis=1)),
            ("warm_equal", no_pmv_warm.mean(axis=1)),
            ("cold_area", no_pmv_cold @ weights),
            ("warm_area", no_pmv_warm @ weights),
        ]:
            distributions.setdefault(
                ("no_pmv_ordinal", "broad_tail", component_name),
                DistributionAccumulator(),
            ).update(component_values)
        case_continuous_rows.append(no_pmv_continuous)
        for threshold in BROAD_CURVE_SCREENS:
            record = threshold_record(
                base,
                "no_pmv_ordinal",
                "broad_tail",
                float(threshold),
                no_pmv_zone_probability,
                no_pmv_aggregate,
                weights,
            )
            case_threshold_rows.append(record)
            update_global_threshold(threshold_totals, record)

        # Preserve the legacy callback-timed broad-tail values as a third,
        # comparison-only model. Endpoint classes cannot be recovered from the
        # legacy trace because only the combined broad tail was stored.
        stored_zone_probability = np.asarray(case["stored_p"], dtype=float)
        stored_mu = np.asarray(case["stored_mu"], dtype=float)
        stored_aggregate = aggregations(stored_zone_probability, weights)
        aggregate_by_model_event[(STORED_MODEL_NAME, "broad_tail")] = stored_aggregate
        zone_by_model_event[(STORED_MODEL_NAME, "broad_tail")] = stored_zone_probability
        stored_continuous: dict[str, object] = {
            **base,
            "model": STORED_MODEL_NAME,
            "event": "broad_tail",
            "n_steps": n_steps,
            "state_hash": state_hash,
            "expected_tsv_equal_mean": float(stored_mu.mean(axis=1).mean()),
            "expected_tsv_area_mean": float((stored_mu @ weights).mean()),
            "cold_equal_mean": float("nan"),
            "warm_equal_mean": float("nan"),
            "cold_area_mean": float("nan"),
            "warm_area_mean": float("nan"),
        }
        for aggregator, values in stored_aggregate.items():
            stored_continuous.update(exact_case_distribution(values, aggregator))
            distributions.setdefault(
                (STORED_MODEL_NAME, "broad_tail", aggregator),
                DistributionAccumulator(),
            ).update(values)
        case_continuous_rows.append(stored_continuous)
        for threshold in BROAD_CURVE_SCREENS:
            record = threshold_record(
                base,
                STORED_MODEL_NAME,
                "broad_tail",
                float(threshold),
                stored_zone_probability,
                stored_aggregate,
                weights,
            )
            case_threshold_rows.append(record)
            update_global_threshold(threshold_totals, record)

        for event in ("broad_tail", "outermost"):
            ordinal_aggregate = aggregate_by_model_event[("ordinal", event)]
            nominal_aggregate = aggregate_by_model_event[("nominal", event)]
            for aggregator in AGGREGATORS:
                pairs.setdefault((event, aggregator), PairAccumulator()).update(
                    ordinal_aggregate[aggregator], nominal_aggregate[aggregator]
                )
            screens = ENDPOINT_SCREENS if event == "outermost" else BROAD_CURVE_SCREENS
            for threshold in screens:
                records = paired_threshold_record(
                    base,
                    event,
                    float(threshold),
                    ordinal_aggregate,
                    nominal_aggregate,
                )
                paired_case_rows.extend(records)
                for record in records:
                    update_global_paired_threshold(paired_threshold_totals, record)

        corrected_aggregate = aggregate_by_model_event[("ordinal", "broad_tail")]
        for aggregator in AGGREGATORS:
            old_new_pairs.setdefault(
                ("broad_tail", aggregator), PairAccumulator()
            ).update(stored_aggregate[aggregator], corrected_aggregate[aggregator])
        for threshold in BROAD_CURVE_SCREENS:
            records = paired_threshold_record(
                base,
                "broad_tail",
                float(threshold),
                stored_aggregate,
                corrected_aggregate,
            )
            old_new_case_rows.extend(records)
            for record in records:
                update_global_paired_threshold(old_new_threshold_totals, record)

        elapsed = time.time() - case_started
        print(
            f"[case {case_index:03d}/{len(paths):03d}] {base['weather']} "
            f"occupied={n_steps} reused={reused} elapsed={elapsed:.1f}s",
            flush=True,
        )

    trace_manifest = pd.DataFrame(trace_manifest_rows)
    trace_manifest.to_csv(args.output_dir / "input_trace_manifest.csv", index=False)
    trace_manifest[
        [
            "weather",
            "city",
            "scenario",
            "time_slice",
            "severity",
            "weather_year",
            "trace_path",
            "analysis_state_sha256",
            "occupied_steps",
            "corrected_zone_npz",
            "corrected_zone_npz_sha256",
        ]
    ].to_csv(args.output_dir / "corrected_figure_input_manifest.csv", index=False)
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(args.output_dir / "ordinal_trace_parity.csv", index=False)
    scalar_cases = pd.DataFrame(scalar_case_rows)
    scalar_cases.to_csv(args.output_dir / "scalar_case_summary.csv", index=False)
    scalar_global = global_scalar_summary(
        scalar_cases, old_new_pmv_pair, pmv_expected_pair, abs_pmv_tail_pair
    )
    scalar_global.to_csv(args.output_dir / "scalar_global_summary.csv", index=False)
    case_continuous = pd.DataFrame(case_continuous_rows)
    case_continuous.to_csv(args.output_dir / "case_continuous_summary.csv", index=False)
    case_threshold = pd.DataFrame(case_threshold_rows)
    case_threshold.to_csv(args.output_dir / "case_threshold_summary.csv", index=False)
    headline_cases = headline_case_summary(case_continuous, case_threshold)
    headline_cases.to_csv(
        args.output_dir / "corrected_headline_case_summary.csv", index=False
    )
    endpoint_cases = endpoint_case_summary(case_continuous, case_threshold)
    endpoint_cases.to_csv(
        args.output_dir / "endpoint_case_summary.csv", index=False
    )
    paired_case = pd.DataFrame(paired_case_rows)
    paired_case.to_csv(args.output_dir / "case_paired_model_thresholds.csv", index=False)
    old_new_case = relabel_old_new_threshold_frame(pd.DataFrame(old_new_case_rows))
    old_new_case.to_csv(
        args.output_dir / "case_stored_vs_corrected_thresholds.csv", index=False
    )

    global_continuous = global_continuous_frame(distributions)
    global_continuous.to_csv(args.output_dir / "global_continuous_summary.csv", index=False)
    global_pairs = global_pair_frame(pairs)
    global_pairs.to_csv(args.output_dir / "global_paired_model_continuous.csv", index=False)
    old_new_global_pairs = relabel_old_new_pair_frame(global_pair_frame(old_new_pairs))
    old_new_global_pairs.to_csv(
        args.output_dir / "global_stored_vs_corrected_continuous.csv", index=False
    )
    global_threshold = threshold_totals_frame(threshold_totals)
    global_threshold.to_csv(args.output_dir / "global_threshold_curves.csv", index=False)
    paired_threshold = paired_threshold_totals_frame(paired_threshold_totals)
    paired_threshold.to_csv(args.output_dir / "global_paired_model_thresholds.csv", index=False)
    old_new_threshold = relabel_old_new_threshold_frame(
        paired_threshold_totals_frame(old_new_threshold_totals)
    )
    old_new_threshold.to_csv(
        args.output_dir / "global_stored_vs_corrected_thresholds.csv", index=False
    )

    grouped_threshold = grouped_threshold_summary(case_threshold)
    grouped_threshold.to_csv(args.output_dir / "grouped_threshold_curves.csv", index=False)
    grouped_continuous = grouped_continuous_summary(case_continuous)
    grouped_continuous.to_csv(args.output_dir / "grouped_continuous_summary.csv", index=False)
    contrasts = future_contrasts(grouped_threshold)
    contrasts.to_csv(args.output_dir / "future_slice_contrasts.csv", index=False)

    validation_summary: pd.DataFrame | None = None
    calibration: pd.DataFrame | None = None
    if not args.skip_validation:
        _, _, validation_summary, calibration = run_validation(
            bundle, args.data_path, args.output_dir
        )

    decisions = decision_rules(
        global_threshold, paired_threshold, global_continuous, contrasts
    )
    decisions.to_csv(args.output_dir / "prespecified_decision_rules.csv", index=False)

    if not args.skip_plots:
        make_plots(args.output_dir, global_threshold, case_continuous, calibration)
    write_summary(
        args.output_dir,
        paths,
        global_continuous,
        global_threshold,
        paired_threshold,
        old_new_threshold,
        scalar_global,
        validation_summary,
        decisions,
        parity,
    )
    write_callback_timing_evidence(args.output_dir)

    config["finished_unix"] = time.time()
    config["elapsed_seconds"] = config["finished_unix"] - started
    config["case_count_completed"] = len(paths)
    config["unique_environmental_state_count"] = int(trace_manifest["analysis_state_sha256"].nunique())
    config["inference_reuse_count"] = int(trace_manifest["reused_previous_state_inference"].sum())
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_output_checksums(args.output_dir)

    alerts = decisions[decisions["material_claim_narrowing"] & ~decisions["passed"]]
    if len(alerts):
        print("[MATERIAL ALERT] one or more prespecified claim-narrowing rules failed:", flush=True)
        print(alerts.to_string(index=False), flush=True)
    else:
        print("[decision] no prespecified claim-narrowing rule failed", flush=True)
    print(f"[write] {args.output_dir}", flush=True)
    print(f"[elapsed] {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

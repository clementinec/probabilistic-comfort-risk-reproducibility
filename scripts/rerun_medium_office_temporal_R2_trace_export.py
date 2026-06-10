#!/usr/bin/env python
"""Rerun Medium Office temporal control traces for the Control_Probablities paper.

The HPH project is used only as a source of TSV training data and EPW files.
The simulated building is the DOE Medium Office prototype shipped with the
local EnergyPlus install.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/control_probabilities_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/control_probabilities_cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/control_probabilities_fontconfig")
for cache_dir in (
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    os.environ["FC_CACHEDIR"],
):
    os.makedirs(cache_dir, exist_ok=True)
warnings.filterwarnings("ignore", message="The py23 module has been deprecated")
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if not hasattr(np, "unicode_"):
    # LightGBM 4.6 still imports np.unicode_ on NumPy 2.x.
    np.unicode_ = np.str_

import lightgbm as lgb
from pythermalcomfort.models import pmv_ppd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "newin_with_bmr.csv"
DEFAULT_OUT = REPO_ROOT
DEFAULT_IDF = REPO_ROOT / "configs" / "medium_office_otc_control.idf"
DEFAULT_EPLUS = Path(os.environ.get("ENERGYPLUS_ROOT", "/Applications/EnergyPlus-25-1-0"))
DEFAULT_WEATHER = REPO_ROOT / "weather" / "phoenix_ssp585_late_2080s_heatwave_extreme_2085.epw"

FEATURE_COLUMNS = [
    "top",
    "v",
    "rh",
    "met_c",
    "clo_c",
    "met_x_clo",
    "BSA_m2",
    "width_diff",
    "s_mono",
    "e_neg",
    "pmv",
    "s",
]
TSV_VALUES = np.arange(-3, 4, dtype=float)
GRID_STRATEGIES = {"grid_naive", "grid_gated"}
GRID_FULL_SHED_DELTA_C = 1.5
GRID_MILD_SHED_DELTA_C = 0.75
GRID_WARM_RISK_SOFT = 0.20
GRID_WARM_RISK_BLOCK = 0.35
ZONE_NAMES = [
    "Core_bottom",
    "Core_mid",
    "Core_top",
    "Perimeter_top_ZN_3",
    "Perimeter_top_ZN_2",
    "Perimeter_top_ZN_1",
    "Perimeter_top_ZN_4",
    "Perimeter_bot_ZN_3",
    "Perimeter_bot_ZN_2",
    "Perimeter_bot_ZN_1",
    "Perimeter_bot_ZN_4",
    "Perimeter_mid_ZN_3",
    "Perimeter_mid_ZN_2",
    "Perimeter_mid_ZN_1",
    "Perimeter_mid_ZN_4",
]
ZONE_FIELD_NAMES = [
    re.sub(r"[^a-z0-9]+", "_", zone.lower()).strip("_") for zone in ZONE_NAMES
]


@dataclass
class FeatureSpec:
    medians: dict[str, float]
    winsor: dict[str, tuple[float, float]]
    met_mean: float
    clo_mean: float


@dataclass
class PredictorBundle:
    spec: FeatureSpec
    scaler: StandardScaler
    nominal: CalibratedClassifierCV
    ordinal: list[CalibratedClassifierCV]
    feature_columns: list[str]

    def predict_nominal(self, features: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(features[self.feature_columns].to_numpy(float))
        probs = self.nominal.predict_proba(x)
        return normalize_probs(align_nominal_classes(probs, self.nominal.classes_))

    def predict_ordinal(self, features: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(features[self.feature_columns].to_numpy(float))
        cumulative = []
        for model in self.ordinal:
            p_gt = model.predict_proba(x)[:, 1]
            cumulative.append(p_gt)
        p_gt = np.column_stack(cumulative)
        p_gt = np.minimum.accumulate(np.clip(p_gt, 0.0, 1.0), axis=1)
        probs = np.empty((x.shape[0], 7), dtype=float)
        probs[:, 0] = 1.0 - p_gt[:, 0]
        probs[:, 1:6] = p_gt[:, :-1] - p_gt[:, 1:]
        probs[:, 6] = p_gt[:, 5]
        return normalize_probs(probs)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    totals = probs.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return probs / totals


def align_nominal_classes(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probs.shape[0], 7), dtype=float)
    for col, cls in enumerate(classes):
        aligned[:, int(cls)] = probs[:, col]
    return aligned


def round_tsv(series: pd.Series) -> np.ndarray:
    return np.clip(np.rint(series.to_numpy(float)), -3, 3).astype(int) + 3


def read_training_data(path: Path, sample_limit: int | None = None) -> pd.DataFrame:
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
    ]
    df = pd.read_csv(path, usecols=cols)
    if sample_limit and sample_limit < len(df):
        y = round_tsv(df["thermal_sensation"])
        df, _ = train_test_split(
            df,
            train_size=sample_limit,
            random_state=42,
            stratify=y,
        )
    return df.reset_index(drop=True)


def fit_feature_spec(train_df: pd.DataFrame) -> FeatureSpec:
    med_cols = [
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
    ]
    medians = {}
    for col in med_cols:
        med = pd.to_numeric(train_df[col], errors="coerce").median()
        if not np.isfinite(med):
            med = 0.0
        medians[col] = float(med)
    medians["vel"] = max(medians.get("vel", 0.01), 0.01)
    medians["rh"] = float(np.clip(medians.get("rh", 50.0), 1.0, 100.0))
    if not (1.0 <= medians.get("bsa_m2", 0.0) <= 3.0):
        medians["bsa_m2"] = 1.8

    raw = coerce_raw_inputs(train_df, medians)
    winsor = {}
    for col in [
        "ta",
        "tr",
        "top",
        "v",
        "rh",
        "met",
        "clo",
        "BSA_m2",
        "rm_out",
    ]:
        lo, hi = np.nanquantile(raw[col], [0.01, 0.99])
        winsor[col] = (float(lo), float(hi))

    met_mean = float(np.nanmean(np.clip(raw["met"], *winsor["met"])))
    clo_mean = float(np.nanmean(np.clip(raw["clo"], *winsor["clo"])))
    return FeatureSpec(medians=medians, winsor=winsor, met_mean=met_mean, clo_mean=clo_mean)


def coerce_raw_inputs(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ta = numeric_or_median(df, "ta", medians)
    tr = numeric_or_median(df, "mean_radiant_temperature", medians)
    tr = tr.where(np.isfinite(tr), ta)
    tr = tr.fillna(ta)
    out["ta"] = ta
    out["tr"] = tr
    out["top"] = (ta + tr) / 2.0
    out["v"] = numeric_or_median(df, "vel", medians).clip(lower=0.01, upper=2.0)
    out["rh"] = numeric_or_median(df, "rh", medians).clip(lower=1.0, upper=100.0)
    out["met"] = numeric_or_median(df, "metabolic_rate", medians).clip(lower=0.5, upper=4.0)
    out["clo"] = numeric_or_median(df, "clothing_insulation", medians).clip(lower=0.0, upper=3.0)

    bsa = numeric_or_median(df, "bsa_m2", medians)
    height = numeric_or_median(df, "height_cm", medians)
    weight = numeric_or_median(df, "weight_kg", medians)
    dubois = 0.007184 * np.power(height.clip(lower=100, upper=230), 0.725) * np.power(
        weight.clip(lower=30, upper=200), 0.425
    )
    out["BSA_m2"] = bsa.where(bsa.between(1.0, 3.0), dubois).fillna(medians.get("bsa_m2", 1.8))

    rm = numeric_or_median(df, "prevailing_outdoor_mean", medians)
    oat = numeric_or_median(df, "outdoor_air_temp", medians)
    out["rm_out"] = rm.where(np.isfinite(rm), oat).fillna(oat)
    return out


def numeric_or_median(df: pd.DataFrame, col: str, medians: dict[str, float]) -> pd.Series:
    if col in df:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(np.nan, index=df.index)
    return s.fillna(medians.get(col, 0.0))


def build_features_from_raw(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    raw = coerce_raw_inputs(df, spec.medians)
    for col, (lo, hi) in spec.winsor.items():
        if col in raw:
            raw[col] = raw[col].clip(lo, hi)
    return build_features_from_arrays(
        ta=raw["ta"].to_numpy(float),
        tr=raw["tr"].to_numpy(float),
        v=raw["v"].to_numpy(float),
        rh=raw["rh"].to_numpy(float),
        met=raw["met"].to_numpy(float),
        clo=raw["clo"].to_numpy(float),
        bsa=raw["BSA_m2"].to_numpy(float),
        rm_out=raw["rm_out"].to_numpy(float),
        spec=spec,
    )


def build_features_from_arrays(
    *,
    ta: np.ndarray,
    tr: np.ndarray,
    v: np.ndarray,
    rh: np.ndarray,
    met: np.ndarray,
    clo: np.ndarray,
    bsa: np.ndarray,
    rm_out: np.ndarray,
    spec: FeatureSpec,
) -> pd.DataFrame:
    ta = np.asarray(ta, dtype=float)
    tr = np.asarray(tr, dtype=float)
    v = np.clip(np.asarray(v, dtype=float), 0.01, 2.0)
    rh = np.clip(np.asarray(rh, dtype=float), 1.0, 100.0)
    met = np.clip(np.asarray(met, dtype=float), 0.5, 4.0)
    clo = np.clip(np.asarray(clo, dtype=float), 0.0, 3.0)
    bsa = np.clip(np.asarray(bsa, dtype=float), 1.0, 3.0)
    rm_out = np.asarray(rm_out, dtype=float)
    top = (ta + tr) / 2.0

    pmv = compute_pmv_array(ta, tr, v, rh, met, clo)
    t_comf = 0.31 * rm_out + 17.8
    warm_lift = np.zeros_like(top)
    warm_mask = (v >= 0.3) & (top > 25.0)
    warm_lift[warm_mask] = 1.2
    warm_lift[(v >= 0.9) & (top > 25.0)] = 1.8
    warm_lift[(v >= 1.2) & (top > 25.0)] = 2.2
    width_cold = np.full_like(top, 3.5)
    width_hot = 3.5 + warm_lift
    width = np.where(top >= t_comf, width_hot, width_cold)
    s = (top - t_comf) / np.maximum(width, 0.1)
    met_c = met - spec.met_mean
    clo_c = clo - spec.clo_mean

    features = pd.DataFrame(
        {
            "top": top,
            "v": v,
            "rh": rh,
            "met_c": met_c,
            "clo_c": clo_c,
            "met_x_clo": met_c * clo_c,
            "BSA_m2": bsa,
            "width_diff": width_hot - width_cold,
            "s_mono": np.clip(s, -1.0, 1.0),
            "e_neg": np.maximum(-s - 1.0, 0.0),
            "pmv": pmv,
            "s": s,
        }
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_pmv_array(
    ta: np.ndarray,
    tr: np.ndarray,
    v: np.ndarray,
    rh: np.ndarray,
    met: np.ndarray,
    clo: np.ndarray,
) -> np.ndarray:
    try:
        result = pmv_ppd(
            tdb=ta,
            tr=tr,
            vr=v,
            rh=rh,
            met=met,
            clo=clo,
            standard="ISO",
            units="SI",
            limit_inputs=False,
        )
        return np.asarray(result["pmv"], dtype=float)
    except Exception:
        return np.zeros_like(np.asarray(ta, dtype=float))


def train_predictors(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
    n_estimators: int,
    sample_limit: int | None,
) -> PredictorBundle:
    print(f"[train] reading TSV source: {data_path}")
    df = read_training_data(data_path, sample_limit=sample_limit)
    y = round_tsv(df["thermal_sensation"])
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

    spec = fit_feature_spec(train_df)
    x_train_df = build_features_from_raw(train_df, spec)
    x_cal_df = build_features_from_raw(cal_df, spec)
    x_test_df = build_features_from_raw(test_df, spec)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df[FEATURE_COLUMNS].to_numpy(float))
    x_cal = scaler.transform(x_cal_df[FEATURE_COLUMNS].to_numpy(float))
    x_test = scaler.transform(x_test_df[FEATURE_COLUMNS].to_numpy(float))

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

    print(f"[train] fitting nominal LightGBM with {n_estimators} trees")
    nominal_base = lgb.LGBMClassifier(objective="multiclass", num_class=7, **common)
    nominal_base.fit(x_train, y_train)
    nominal = CalibratedClassifierCV(
        estimator=nominal_base,
        method="isotonic",
        cv="prefit",
        ensemble=False,
    )
    nominal.fit(x_cal, y_cal)

    print("[train] fitting six cumulative ordinal LightGBM heads")
    ordinal_models: list[CalibratedClassifierCV] = []
    for threshold in range(6):
        y_train_bin = (y_train > threshold).astype(int)
        y_cal_bin = (y_cal > threshold).astype(int)
        base = lgb.LGBMClassifier(objective="binary", **common)
        base.fit(x_train, y_train_bin)
        cal = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv="prefit",
            ensemble=False,
        )
        cal.fit(x_cal, y_cal_bin)
        ordinal_models.append(cal)

    bundle = PredictorBundle(
        spec=spec,
        scaler=scaler,
        nominal=nominal,
        ordinal=ordinal_models,
        feature_columns=FEATURE_COLUMNS,
    )
    metrics = evaluate_bundle(bundle, x_test_df, y_test)
    metrics.update(
        {
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_calibration": int(len(cal_df)),
            "n_test": int(len(test_df)),
            "n_estimators": int(n_estimators),
            "sample_limit": sample_limit,
        }
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[train] saved model: {model_path}")
    print(f"[train] saved metrics: {metrics_path}")
    return bundle


def evaluate_bundle(bundle: PredictorBundle, x_test_df: pd.DataFrame, y_test: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, probs in [
        ("nominal", bundle.predict_nominal(x_test_df)),
        ("ordinal", bundle.predict_ordinal(x_test_df)),
    ]:
        pred = probs.argmax(axis=1)
        tail_true = np.isin(y_test, [0, 1, 5, 6]).astype(float)
        tail_prob = probs[:, [0, 1, 5, 6]].sum(axis=1)
        mu = probs @ TSV_VALUES
        y_signed = y_test - 3
        out[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "log_loss": float(log_loss(y_test, probs, labels=np.arange(7))),
            "tail_probability_mae": float(np.mean(np.abs(tail_prob - tail_true))),
            "expected_tsv_mae": float(np.mean(np.abs(mu - y_signed))),
        }
    return out


def patch_idf_for_control(
    source_idf: Path,
    target_idf: Path,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
) -> None:
    text = source_idf.read_text(errors="ignore")
    patched_objects: list[str] = []
    inserted_runperiod = False
    for obj in text.split(";"):
        if not obj.strip():
            continue
        fields = parse_idf_fields(obj)
        class_name = fields[0].lower() if fields else ""
        if class_name == "runperiod":
            if not inserted_runperiod:
                patched_objects.append(
                    f"""
  RunPeriod,
    OTC_CONTROL_TRACE,       !- Name
    {begin_month},           !- Begin Month
    {begin_day},             !- Begin Day of Month
    ,                        !- Begin Year
    {end_month},             !- End Month
    {end_day},               !- End Day of Month
    ,                        !- End Year
    Monday,                  !- Day of Week for Start Day
    No,                      !- Use Weather File Holidays and Special Days
    No,                      !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes"""
                )
                inserted_runperiod = True
            continue
        if class_name == "thermostatsetpoint:dualsetpoint":
            if len(fields) >= 2:
                name = fields[1]
                patched_objects.append(
                    f"""
  ThermostatSetpoint:DualSetpoint,
    {name},                  !- Name
    OTC_HEATING_SETPOINT,    !- Heating Setpoint Temperature Schedule Name
    OTC_COOLING_SETPOINT"""
                )
                continue
        patched_objects.append(obj)

    patched_objects.append(
        """

!-   ===========  OTC CONTROL API SCHEDULES ===========

  Schedule:Constant,
    OTC_HEATING_SETPOINT,    !- Name
    Temperature,             !- Schedule Type Limits Name
    22.0;                    !- Hourly Value

  Schedule:Constant,
    OTC_COOLING_SETPOINT,    !- Name
    Temperature,             !- Schedule Type Limits Name
    24.0;                    !- Hourly Value

  Output:SQLite,
    SimpleAndTabular;

  Output:Meter,
    Electricity:Facility,
    Hourly;

  Output:Meter,
    NaturalGas:Facility,
    Hourly;
"""
    )
    target_idf.parent.mkdir(parents=True, exist_ok=True)
    target_idf.write_text(";\n".join(patched_objects) + "\n")


def parse_idf_fields(obj: str) -> list[str]:
    no_comments = []
    for line in obj.splitlines():
        no_comments.append(line.split("!", 1)[0])
    raw = "\n".join(no_comments).replace("\n", " ")
    return [f.strip() for f in raw.split(",") if f.strip()]


@dataclass
class ControlState:
    strategy: str
    bundle: PredictorBundle | None
    heat_sp: float = 22.0
    cool_sp: float = 24.0
    rm_out: float | None = None
    last_control_key: tuple[int, int, int, float] | None = None
    last_record_key: tuple[int, int, int, float] | None = None
    initialized: bool = False
    handles: dict[str, Any] | None = None
    records: list[dict[str, Any]] | None = None
    grid_signal: dict[tuple[int, int, int], dict[str, float | int]] | None = None
    current_grid_signal: dict[str, float | int] | None = None


def run_energyplus_strategy(
    *,
    strategy: str,
    bundle: PredictorBundle | None,
    idf_path: Path,
    weather_path: Path,
    eplus_root: Path,
    out_dir: Path,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
) -> Path:
    if str(eplus_root) not in sys.path:
        sys.path.insert(0, str(eplus_root))
    from pyenergyplus.api import EnergyPlusAPI

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    run_dir = out_dir / "energyplus" / weather_path.stem / strategy
    run_dir.mkdir(parents=True, exist_ok=True)

    for zone in ZONE_NAMES:
        for variable in [
            "Zone Mean Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Air Relative Humidity",
            "Zone Air System Sensible Heating Rate",
            "Zone Air System Sensible Cooling Rate",
        ]:
            api.exchange.request_variable(state, variable, zone)
    api.exchange.request_variable(state, "Site Outdoor Air Drybulb Temperature", "Environment")

    grid_signal = (
        build_microgrid_signal_schedule(weather_path, begin_month, begin_day, end_month, end_day)
        if strategy in GRID_STRATEGIES
        else None
    )
    ctl = ControlState(strategy=strategy, bundle=bundle, records=[], grid_signal=grid_signal)

    def initialize_handles(st: Any) -> None:
        if ctl.initialized or not api.exchange.api_data_fully_ready(st):
            return
        handles: dict[str, Any] = {
            "heat_act": api.exchange.get_actuator_handle(
                st, "Schedule:Constant", "Schedule Value", "OTC_HEATING_SETPOINT"
            ),
            "cool_act": api.exchange.get_actuator_handle(
                st, "Schedule:Constant", "Schedule Value", "OTC_COOLING_SETPOINT"
            ),
            "oat": api.exchange.get_variable_handle(
                st, "Site Outdoor Air Drybulb Temperature", "Environment"
            ),
            "electricity": api.exchange.get_meter_handle(st, "Electricity:Facility"),
            "gas": api.exchange.get_meter_handle(st, "NaturalGas:Facility"),
            "zones": {},
        }
        for zone in ZONE_NAMES:
            handles["zones"][zone] = {
                "ta": api.exchange.get_variable_handle(st, "Zone Mean Air Temperature", zone),
                "tr": api.exchange.get_variable_handle(st, "Zone Mean Radiant Temperature", zone),
                "rh": api.exchange.get_variable_handle(st, "Zone Air Relative Humidity", zone),
                "heat_rate": api.exchange.get_variable_handle(
                    st, "Zone Air System Sensible Heating Rate", zone
                ),
                "cool_rate": api.exchange.get_variable_handle(
                    st, "Zone Air System Sensible Cooling Rate", zone
                ),
            }
        if handles["heat_act"] < 0 or handles["cool_act"] < 0:
            raise RuntimeError("Could not acquire OTC thermostat schedule actuators.")
        ctl.handles = handles
        ctl.initialized = True

    def apply_control(st: Any) -> None:
        initialize_handles(st)
        if not ctl.initialized:
            return
        key = current_key(api, st)
        if ctl.last_control_key == key:
            return
        ctl.last_control_key = key

        if api.exchange.warmup_flag(st):
            heat, cool = 22.0, 24.0
            set_api_setpoints(api, st, ctl, heat, cool)
            return

        values = read_zone_values(api, st, ctl)
        oat = read_handle(api, st, ctl.handles["oat"], default=np.nan)
        if not np.isfinite(oat):
            oat = values["ta_mean"]
        ctl.rm_out = update_running_mean(ctl.rm_out, oat)
        ctl.current_grid_signal = lookup_microgrid_signal(api, st, ctl)
        occupied = is_occupied(api, st)

        if not occupied:
            heat, cool = 12.0, 30.0
            signal = default_control_signal(values["pmv_mean"], ctl.current_grid_signal)
        elif strategy == "reference":
            heat, cool = 22.0, 24.0
            signal = default_control_signal(values["pmv_mean"], ctl.current_grid_signal)
        else:
            if ctl.heat_sp <= 12.01 or ctl.cool_sp >= 29.99:
                ctl.heat_sp, ctl.cool_sp = 22.0, 24.0
            heat, cool, signal = controller_step(strategy, ctl, values, oat)
        set_api_setpoints(api, st, ctl, heat, cool)
        ctl.pending_signal = signal

    def record(st: Any) -> None:
        initialize_handles(st)
        if not ctl.initialized or api.exchange.warmup_flag(st):
            return
        if not in_requested_period(api, st, begin_month, begin_day, end_month, end_day):
            return
        key = current_key(api, st)
        if ctl.last_record_key == key:
            return
        ctl.last_record_key = key
        values = read_zone_values(api, st, ctl)
        oat = read_handle(api, st, ctl.handles["oat"], default=np.nan)
        signal = getattr(ctl, "pending_signal", {})
        t_comf = 0.31 * (ctl.rm_out if ctl.rm_out is not None else oat) + 17.8
        rec = {
            "strategy": strategy,
            "weather": weather_path.stem,
            "calendar_year": int(api.exchange.calendar_year(st)),
            "month": int(api.exchange.month(st)),
            "day": int(api.exchange.day_of_month(st)),
            "day_of_week": int(api.exchange.day_of_week(st)),
            "hour": int(api.exchange.hour(st)),
            "current_time": float(api.exchange.current_time(st)),
            "sim_time_days": float(api.exchange.current_sim_time(st)),
            "occupied": bool(is_occupied(api, st)),
            "outdoor_temp_c": float(oat),
            "running_mean_outdoor_c": float(ctl.rm_out if ctl.rm_out is not None else oat),
            "comfort_low_c": float(t_comf - 2.5),
            "comfort_high_c": float(t_comf + 2.5),
            "mean_air_temp_c": float(values["ta_mean"]),
            "mean_mrt_c": float(values["tr_mean"]),
            "mean_operative_temp_c": float(values["top_mean"]),
            "mean_rh_pct": float(values["rh_mean"]),
            "mean_pmv": float(signal.get("mean_pmv", values["pmv_mean"])),
            "expected_tsv": float(signal.get("expected_tsv", np.nan)),
            "discomfort_probability": float(signal.get("discomfort_probability", np.nan)),
            "warm_discomfort_probability": float(
                signal.get("warm_discomfort_probability", np.nan)
            ),
            "cold_discomfort_probability": float(
                signal.get("cold_discomfort_probability", np.nan)
            ),
            "action_delta_c": float(signal.get("action_delta", 0.0)),
            "action_direction": int(signal.get("action_direction", 0)),
            "setpoint_shift_c": float(signal.get("setpoint_shift", 0.0)),
            "grid_event": int(signal.get("grid_event", 0)),
            "grid_stress_score": float(signal.get("grid_stress_score", np.nan)),
            "grid_oat_c": float(signal.get("grid_oat_c", np.nan)),
            "grid_ghi_w_m2": float(signal.get("grid_ghi_w_m2", np.nan)),
            "grid_requested_delta_c": float(signal.get("grid_requested_delta", 0.0)),
            "grid_served_delta_c": float(signal.get("grid_served_delta", 0.0)),
            "grid_rejected": int(signal.get("grid_rejected", 0)),
            "heating_setpoint_c": float(ctl.heat_sp),
            "cooling_setpoint_c": float(ctl.cool_sp),
            "zone_heating_rate_w": float(values["heat_rate_sum"]),
            "zone_cooling_rate_w": float(values["cool_rate_sum"]),
            "hvac_on": bool(values["heat_rate_sum"] > 10.0 or values["cool_rate_sum"] > 10.0),
            "electricity_facility_j": float(
                read_handle(api, st, ctl.handles["electricity"], default=0.0, meter=True)
            ),
            "natural_gas_facility_j": float(
                read_handle(api, st, ctl.handles["gas"], default=0.0, meter=True)
            ),
        }
        add_zone_probability_record_fields(rec, signal)
        ctl.records.append(rec)

    def progress(_pct: int) -> None:
        return

    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, apply_control)
    api.runtime.callback_end_zone_timestep_after_zone_reporting(state, record)
    api.runtime.callback_progress(state, progress)

    print(f"[simulate] {strategy}: {weather_path.name}")
    exit_code = api.runtime.run_energyplus(
        state,
        ["-w", str(weather_path), "-d", str(run_dir), str(idf_path)],
    )
    api.state_manager.delete_state(state)
    if exit_code != 0:
        raise RuntimeError(f"EnergyPlus failed for {strategy} with exit code {exit_code}.")

    trace_df = pd.DataFrame(ctl.records)
    trace_df = attach_hourly_meters(trace_df, run_dir / "eplusout.mtr")
    trace_path = out_dir / "traces" / f"{weather_path.stem}_{strategy}.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_path, index=False)
    print(f"[simulate] wrote trace: {trace_path}")
    return trace_path


def default_control_signal(
    mean_pmv: float, grid_signal: dict[str, float | int] | None = None
) -> dict[str, float | int]:
    signal: dict[str, float | int] = {
        "action_delta": 0.0,
        "action_direction": 0,
        "setpoint_shift": 0.0,
        "mean_pmv": float(mean_pmv),
        "expected_tsv": np.nan,
        "discomfort_probability": np.nan,
        "warm_discomfort_probability": np.nan,
        "cold_discomfort_probability": np.nan,
    }
    signal.update(format_grid_signal(grid_signal, requested=0.0, served=0.0, rejected=0))
    return signal


def format_grid_signal(
    grid_signal: dict[str, float | int] | None,
    *,
    requested: float,
    served: float,
    rejected: int,
) -> dict[str, float | int]:
    grid_signal = grid_signal or {}
    return {
        "grid_event": int(grid_signal.get("grid_event", 0)),
        "grid_stress_score": float(grid_signal.get("grid_stress_score", np.nan)),
        "grid_oat_c": float(grid_signal.get("grid_oat_c", np.nan)),
        "grid_ghi_w_m2": float(grid_signal.get("grid_ghi_w_m2", np.nan)),
        "grid_requested_delta": float(requested),
        "grid_served_delta": float(served),
        "grid_rejected": int(rejected),
    }


def build_microgrid_signal_schedule(
    weather_path: Path,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
) -> dict[tuple[int, int, int], dict[str, float | int]]:
    weather = read_epw_microgrid_weather(weather_path)
    if weather.empty:
        return {}
    period = weather[
        weather.apply(
            lambda row: (begin_month, begin_day)
            <= (int(row["month"]), int(row["day"]))
            <= (end_month, end_day),
            axis=1,
        )
    ].copy()
    if period.empty:
        period = weather.copy()
    occupied = period[(period["hour"] >= 6) & (period["hour"] < 22)].copy()
    if occupied.empty:
        occupied = period.copy()

    oat_q10 = float(occupied["drybulb_c"].quantile(0.10))
    oat_q90 = float(occupied["drybulb_c"].quantile(0.90))
    ghi_max = max(float(occupied["ghi_w_m2"].quantile(0.95)), 1.0)
    oat_norm = ((period["drybulb_c"] - oat_q10) / max(oat_q90 - oat_q10, 1e-6)).clip(0, 1)
    pv_deficit = (1.0 - (period["ghi_w_m2"] / ghi_max).clip(0, 1)).clip(0, 1)
    evening = period["hour"].map(evening_ramp_weight).astype(float)
    period["grid_stress_score"] = 0.60 * oat_norm + 0.30 * pv_deficit + 0.10 * evening

    event_pool = period[(period["hour"] >= 6) & (period["hour"] < 22)].copy()
    if event_pool.empty:
        event_pool = period.copy()
    threshold = float(event_pool["grid_stress_score"].quantile(0.85))
    hot_floor = float(event_pool["drybulb_c"].quantile(0.60))
    period["grid_event"] = (
        (period["grid_stress_score"] >= threshold)
        & (period["drybulb_c"] >= hot_floor)
        & (period["hour"] >= 12)
        & (period["hour"] < 22)
    ).astype(int)

    schedule: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for row in period.itertuples(index=False):
        schedule[(int(row.month), int(row.day), int(row.hour))] = {
            "grid_event": int(row.grid_event),
            "grid_stress_score": float(row.grid_stress_score),
            "grid_oat_c": float(row.drybulb_c),
            "grid_ghi_w_m2": float(row.ghi_w_m2),
        }
    event_hours = sum(v["grid_event"] for v in schedule.values())
    print(
        f"[grid] {weather_path.stem}: selected {event_hours} event hours "
        f"from {len(schedule)} simulated weather hours"
    )
    return schedule


def read_epw_microgrid_weather(weather_path: Path) -> pd.DataFrame:
    rows = []
    with weather_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for i, raw in enumerate(handle):
            if i < 8:
                continue
            parts = raw.strip().split(",")
            if len(parts) < 16:
                continue
            try:
                rows.append(
                    {
                        "month": int(float(parts[1])),
                        "day": int(float(parts[2])),
                        "hour": int(float(parts[3])),
                        "drybulb_c": float(parts[6]),
                        "ghi_w_m2": max(float(parts[13]), 0.0),
                    }
                )
            except ValueError:
                continue
    return pd.DataFrame(rows)


def evening_ramp_weight(hour: int) -> float:
    if 15 <= hour < 20:
        return 1.0
    if 13 <= hour < 15 or 20 <= hour < 22:
        return 0.5
    return 0.0


def lookup_microgrid_signal(
    api: Any, st: Any, ctl: ControlState
) -> dict[str, float | int] | None:
    if not ctl.grid_signal:
        return None
    month = int(api.exchange.month(st))
    day = int(api.exchange.day_of_month(st))
    meter_hour = int(math.ceil(float(api.exchange.current_time(st))))
    meter_hour = max(1, min(24, meter_hour))
    return ctl.grid_signal.get((month, day, meter_hour))


def current_key(api: Any, st: Any) -> tuple[int, int, int, float]:
    return (
        int(api.exchange.month(st)),
        int(api.exchange.day_of_month(st)),
        int(api.exchange.hour(st)),
        round(float(api.exchange.current_time(st)), 6),
    )


def attach_hourly_meters(trace_df: pd.DataFrame, mtr_path: Path) -> pd.DataFrame:
    if trace_df.empty or not mtr_path.exists():
        return trace_df
    hourly = parse_mtr_hourly(mtr_path)
    if hourly.empty:
        return trace_df
    out = trace_df.copy()
    out["meter_hour"] = np.ceil(out["current_time"].astype(float)).astype(int).clip(1, 24)
    merged = out.merge(hourly, on=["month", "day", "meter_hour"], how="left")
    counts = merged.groupby(["month", "day", "meter_hour"])["strategy"].transform("size").clip(lower=1)
    for col in ["electricity_facility_j", "natural_gas_facility_j"]:
        hourly_col = f"{col}_hourly"
        if hourly_col in merged:
            step_values = merged[hourly_col] / counts
            merged[col] = np.where(step_values.notna(), step_values, merged[col])
    return merged.drop(columns=[c for c in ["meter_hour"] if c in merged])


def parse_mtr_hourly(mtr_path: Path) -> pd.DataFrame:
    meter_ids: dict[str, str] = {}
    rows: list[dict[str, float | int]] = []
    current: dict[str, int] | None = None
    in_dictionary = True
    for raw in mtr_path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_dictionary:
            if line == "End of Data Dictionary":
                in_dictionary = False
                continue
            match = re.match(r"^(\d+),\d+,(.+?) \[J\] !Hourly", line)
            if match:
                meter_id, name = match.groups()
                if name == "Electricity:Facility":
                    meter_ids[meter_id] = "electricity_facility_j_hourly"
                elif name == "NaturalGas:Facility":
                    meter_ids[meter_id] = "natural_gas_facility_j_hourly"
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        if parts[0] == "2" and len(parts) >= 8:
            if current is not None:
                rows.append(current)
            current = {
                "month": int(float(parts[2])),
                "day": int(float(parts[3])),
                "meter_hour": int(float(parts[5])),
            }
        elif current is not None and parts[0] in meter_ids and len(parts) >= 2:
            current[meter_ids[parts[0]]] = float(parts[1])
    if current is not None:
        rows.append(current)
    return pd.DataFrame(rows)


def in_requested_period(
    api: Any,
    st: Any,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
) -> bool:
    cur = (int(api.exchange.month(st)), int(api.exchange.day_of_month(st)))
    return (begin_month, begin_day) <= cur <= (end_month, end_day)


def read_handle(api: Any, st: Any, handle: int, default: float, meter: bool = False) -> float:
    if handle is None or handle < 0:
        return default
    try:
        if meter:
            val = api.exchange.get_meter_value(st, handle)
        else:
            val = api.exchange.get_variable_value(st, handle)
        return float(val)
    except Exception:
        return default


def read_zone_values(api: Any, st: Any, ctl: ControlState) -> dict[str, float | np.ndarray]:
    ta_vals = []
    tr_vals = []
    rh_vals = []
    heat_rates = []
    cool_rates = []
    for zone in ZONE_NAMES:
        handles = ctl.handles["zones"][zone]
        ta = read_handle(api, st, handles["ta"], default=np.nan)
        tr = read_handle(api, st, handles["tr"], default=ta)
        rh = read_handle(api, st, handles["rh"], default=50.0)
        ta_vals.append(ta)
        tr_vals.append(tr if np.isfinite(tr) else ta)
        rh_vals.append(rh if np.isfinite(rh) else 50.0)
        heat_rates.append(read_handle(api, st, handles["heat_rate"], default=0.0))
        cool_rates.append(read_handle(api, st, handles["cool_rate"], default=0.0))
    ta_arr = np.asarray(ta_vals, dtype=float)
    tr_arr = np.asarray(tr_vals, dtype=float)
    rh_arr = np.asarray(rh_vals, dtype=float)
    bad = ~np.isfinite(ta_arr)
    if bad.any():
        ta_arr[bad] = np.nanmean(ta_arr[~bad]) if (~bad).any() else 24.0
    bad = ~np.isfinite(tr_arr)
    if bad.any():
        tr_arr[bad] = ta_arr[bad]
    rh_arr = np.clip(np.nan_to_num(rh_arr, nan=50.0), 1.0, 100.0)
    top = (ta_arr + tr_arr) / 2.0
    pmv = compute_pmv_array(
        ta=ta_arr,
        tr=tr_arr,
        v=np.full_like(ta_arr, 0.10),
        rh=rh_arr,
        met=np.full_like(ta_arr, 1.10),
        clo=np.full_like(ta_arr, 0.65),
    )
    return {
        "ta": ta_arr,
        "tr": tr_arr,
        "rh": rh_arr,
        "top": top,
        "ta_mean": float(np.mean(ta_arr)),
        "tr_mean": float(np.mean(tr_arr)),
        "top_mean": float(np.mean(top)),
        "rh_mean": float(np.mean(rh_arr)),
        "pmv_mean": float(np.mean(pmv)),
        "heat_rate_sum": float(np.nansum(heat_rates)),
        "cool_rate_sum": float(np.nansum(cool_rates)),
    }


def update_running_mean(previous: float | None, oat: float) -> float:
    if previous is None or not np.isfinite(previous):
        return float(oat)
    alpha_step = 0.8 ** (1.0 / 96.0)
    return float(alpha_step * previous + (1.0 - alpha_step) * oat)


def is_occupied(api: Any, st: Any) -> bool:
    day = int(api.exchange.day_of_week(st))
    current_time = float(api.exchange.current_time(st))
    hour = int(math.floor(current_time))
    weekday = 2 <= day <= 6
    return bool(weekday and 6 <= hour < 22)


def controller_step(
    strategy: str,
    ctl: ControlState,
    values: dict[str, float | np.ndarray],
    oat: float,
) -> tuple[float, float, dict[str, float | int]]:
    direction = 0
    delta = 0.0
    setpoint_shift = 0.0
    signal: dict[str, float | int] = default_control_signal(
        float(values["pmv_mean"]), ctl.current_grid_signal
    )

    if strategy in GRID_STRATEGIES:
        return grid_controller_step(strategy, ctl, values, oat, signal)

    if strategy == "pmv":
        pmv_signal = float(values["pmv_mean"])
        if abs(pmv_signal) > 1.5:
            delta = 1.25
        elif abs(pmv_signal) > 0.5:
            delta = 0.5
        direction = int(np.sign(pmv_signal))
        signal["expected_tsv"] = np.nan
        signal["discomfort_probability"] = np.nan
    else:
        if ctl.bundle is None:
            raise RuntimeError(f"{strategy} requires trained predictors.")
        probs = predict_zone_probabilities(ctl, values, oat, predictor=strategy)
        zone_mu, zone_cold_tail, zone_warm_tail = zone_probability_signals(probs)
        mu = float(np.mean(zone_mu))
        cold_tail = float(np.mean(zone_cold_tail))
        warm_tail = float(np.mean(zone_warm_tail))
        p_disc = cold_tail + warm_tail
        if p_disc >= 0.35:
            delta = 1.25
        elif p_disc > 0.065:
            delta = 0.5
        direction = int(np.sign(mu)) if delta > 0 else 0
        signal["expected_tsv"] = mu
        signal["discomfort_probability"] = p_disc
        signal["warm_discomfort_probability"] = warm_tail
        signal["cold_discomfort_probability"] = cold_tail
        signal["zone_expected_tsv"] = zone_mu
        signal["zone_cold_tail"] = zone_cold_tail
        signal["zone_warm_tail"] = zone_warm_tail

    if direction == 0 or delta == 0.0:
        heat, cool = ctl.heat_sp, ctl.cool_sp
    else:
        # direction is the thermal sensation side; setpoints move oppositely.
        setpoint_shift = -delta if direction > 0 else delta
        heat, cool = apply_setpoint_shift(ctl.heat_sp, ctl.cool_sp, setpoint_shift)

    signal["action_delta"] = delta
    signal["action_direction"] = direction
    signal["setpoint_shift"] = setpoint_shift
    return heat, cool, signal


def grid_controller_step(
    strategy: str,
    ctl: ControlState,
    values: dict[str, float | np.ndarray],
    oat: float,
    signal: dict[str, float | int],
) -> tuple[float, float, dict[str, float | int]]:
    grid_signal = ctl.current_grid_signal or {}
    grid_event = int(grid_signal.get("grid_event", 0))
    requested = GRID_FULL_SHED_DELTA_C if grid_event else 0.0
    served = requested
    rejected = 0

    signal["expected_tsv"] = np.nan
    signal["discomfort_probability"] = np.nan
    signal["warm_discomfort_probability"] = np.nan
    signal["cold_discomfort_probability"] = np.nan

    if strategy == "grid_gated" and grid_event:
        if ctl.bundle is None:
            raise RuntimeError(f"{strategy} requires trained predictors.")
        probs = predict_zone_probabilities(ctl, values, oat, predictor="ordinal")
        zone_mu, zone_cold_tail, zone_warm_tail = zone_probability_signals(probs)
        mu = float(np.mean(zone_mu))
        cold_tail = float(np.mean(zone_cold_tail))
        warm_tail = float(np.mean(zone_warm_tail))
        p_disc = cold_tail + warm_tail
        if warm_tail >= GRID_WARM_RISK_BLOCK:
            served = 0.0
            rejected = 1
        elif warm_tail >= GRID_WARM_RISK_SOFT:
            served = GRID_MILD_SHED_DELTA_C
        signal["expected_tsv"] = mu
        signal["discomfort_probability"] = p_disc
        signal["warm_discomfort_probability"] = warm_tail
        signal["cold_discomfort_probability"] = cold_tail
        signal["zone_expected_tsv"] = zone_mu
        signal["zone_cold_tail"] = zone_cold_tail
        signal["zone_warm_tail"] = zone_warm_tail

    heat, cool = apply_setpoint_shift(22.0, 24.0, served) if served > 0 else (22.0, 24.0)
    signal["action_delta"] = abs(served)
    signal["action_direction"] = -1 if served > 0 else 0
    signal["setpoint_shift"] = served
    signal.update(
        format_grid_signal(grid_signal, requested=requested, served=served, rejected=rejected)
    )
    return heat, cool, signal


def predict_zone_probabilities(
    ctl: ControlState,
    values: dict[str, float | np.ndarray],
    oat: float,
    *,
    predictor: str,
) -> np.ndarray:
    if ctl.bundle is None:
        raise RuntimeError("Predictor bundle is required.")
    n = len(values["ta"])
    features = build_features_from_arrays(
        ta=np.asarray(values["ta"], dtype=float),
        tr=np.asarray(values["tr"], dtype=float),
        v=np.full(n, 0.10),
        rh=np.asarray(values["rh"], dtype=float),
        met=np.full(n, 1.10),
        clo=np.full(n, 0.65),
        bsa=np.full(n, 1.80),
        rm_out=np.full(n, ctl.rm_out if ctl.rm_out is not None else oat),
        spec=ctl.bundle.spec,
    )
    if predictor == "nominal":
        return ctl.bundle.predict_nominal(features)
    if predictor == "ordinal":
        return ctl.bundle.predict_ordinal(features)
    raise ValueError(f"Unknown predictor: {predictor}")


def zone_probability_signals(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zone_mu = probs @ TSV_VALUES
    zone_cold_tail = probs[:, [0, 1]].sum(axis=1)
    zone_warm_tail = probs[:, [5, 6]].sum(axis=1)
    return zone_mu, zone_cold_tail, zone_warm_tail


def add_zone_probability_record_fields(
    rec: dict[str, Any],
    signal: dict[str, Any],
) -> None:
    if not all(
        key in signal
        for key in ("zone_expected_tsv", "zone_cold_tail", "zone_warm_tail")
    ):
        return
    zone_mu = np.asarray(signal["zone_expected_tsv"], dtype=float)
    zone_cold_tail = np.asarray(signal["zone_cold_tail"], dtype=float)
    zone_warm_tail = np.asarray(signal["zone_warm_tail"], dtype=float)
    if not (
        len(zone_mu) == len(zone_cold_tail) == len(zone_warm_tail) == len(ZONE_FIELD_NAMES)
    ):
        return
    zone_p_disc = zone_cold_tail + zone_warm_tail
    zone_d_tail = zone_warm_tail - zone_cold_tail
    for idx, slug in enumerate(ZONE_FIELD_NAMES):
        rec[f"zone_{slug}_expected_tsv"] = float(zone_mu[idx])
        rec[f"zone_{slug}_p_disc"] = float(zone_p_disc[idx])
        rec[f"zone_{slug}_warm_tail"] = float(zone_warm_tail[idx])
        rec[f"zone_{slug}_cold_tail"] = float(zone_cold_tail[idx])
        rec[f"zone_{slug}_d_tail"] = float(zone_d_tail[idx])


def apply_setpoint_shift(heat: float, cool: float, shift: float) -> tuple[float, float]:
    heat += shift
    cool += shift
    heat = float(np.clip(heat, 12.0, 23.25))
    cool = float(np.clip(cool, 23.25, 30.0))
    if cool - heat < 2.0:
        if shift >= 0:
            cool = min(30.0, heat + 2.0)
            heat = min(heat, cool - 2.0)
        else:
            heat = max(12.0, cool - 2.0)
            cool = max(cool, heat + 2.0)
    return heat, cool


def set_api_setpoints(api: Any, st: Any, ctl: ControlState, heat: float, cool: float) -> None:
    ctl.heat_sp = float(heat)
    ctl.cool_sp = float(cool)
    api.exchange.set_actuator_value(st, ctl.handles["heat_act"], ctl.heat_sp)
    api.exchange.set_actuator_value(st, ctl.handles["cool_act"], ctl.cool_sp)


def run_simulations(
    *,
    bundle: PredictorBundle | None,
    output_dir: Path,
    source_idf: Path,
    weather_paths: list[Path],
    eplus_root: Path,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
    strategies: list[str],
) -> list[Path]:
    idf_path = output_dir / "model" / "medium_office_otc_control.idf"
    patch_idf_for_control(source_idf, idf_path, begin_month, begin_day, end_month, end_day)
    trace_paths = []
    for weather in weather_paths:
        for strategy in strategies:
            trace_paths.append(
                run_energyplus_strategy(
                    strategy=strategy,
                    bundle=bundle,
                    idf_path=idf_path,
                    weather_path=weather,
                    eplus_root=eplus_root,
                    out_dir=output_dir,
                    begin_month=begin_month,
                    begin_day=begin_day,
                    end_month=end_month,
                    end_day=end_day,
                )
            )
    combined = pd.concat([pd.read_csv(p) for p in trace_paths], ignore_index=True)
    combined_path = output_dir / "traces" / "medium_office_control_traces.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[simulate] wrote combined traces: {combined_path}")
    return trace_paths


def add_trace_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "calendar_year" in df and df["calendar_year"].notna().any():
        years = df["calendar_year"].replace(0, 2001).fillna(2001).astype(int)
    else:
        years = pd.Series(2001, index=df.index)
    hour_float = df["current_time"].fillna(df["hour"]).astype(float)
    hour = np.floor(hour_float).astype(int)
    minute = np.rint((hour_float - hour) * 60).astype(int)
    hour = np.clip(hour, 0, 23)
    minute = np.clip(minute, 0, 59)
    df["timestamp"] = pd.to_datetime(
        {
            "year": years,
            "month": df["month"].astype(int),
            "day": df["day"].astype(int),
            "hour": hour,
            "minute": minute,
        },
        errors="coerce",
    )
    fallback = pd.Timestamp(2001, 1, 1) + pd.to_timedelta(
        np.arange(len(df)) * 15, unit="min"
    )
    df["timestamp"] = df["timestamp"].fillna(pd.Series(fallback, index=df.index))
    return df


def select_plot_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    if df.empty:
        return df
    reference = df[df["strategy"] == "reference"].copy()
    if reference.empty:
        reference = df.copy()
    reference["date"] = reference["timestamp"].dt.floor("D")
    daily = reference.groupby("date")["outdoor_temp_c"].mean().sort_index()
    if len(daily) <= days:
        start = daily.index.min()
    else:
        rolling = daily.rolling(days, min_periods=days).mean().dropna()
        end = rolling.idxmax()
        start = end - pd.Timedelta(days=days - 1)
    stop = start + pd.Timedelta(days=days)
    return df[(df["timestamp"] >= start) & (df["timestamp"] < stop)].copy()


def make_temporal_plot(output_dir: Path, weather_stem: str) -> Path:
    combined_path = output_dir / "traces" / "medium_office_control_traces.csv"
    df = pd.read_csv(combined_path)
    df = df[df["weather"] == weather_stem].copy()
    df = add_trace_datetime(df)
    df = select_plot_window(df, days=7)

    strategies = ["reference", "pmv", "nominal", "ordinal"]
    titles = {
        "reference": "Reference",
        "pmv": "PMV",
        "nominal": "Nominal Prob.",
        "ordinal": "Ordinal Prob.",
    }
    fig, axes = plt.subplots(
        4,
        len(strategies),
        figsize=(17.5, 10.5),
        sharex=True,
    )
    fig.subplots_adjust(top=0.86, hspace=0.22, wspace=0.16)

    for col, strategy in enumerate(strategies):
        sdf = df[df["strategy"] == strategy].sort_values("timestamp")
        if sdf.empty:
            continue
        x = sdf["timestamp"]

        ax = axes[0, col]
        ax.fill_between(
            x,
            sdf["comfort_low_c"],
            sdf["comfort_high_c"],
            color="#d8ead2",
            alpha=0.55,
            linewidth=0,
            label="Adaptive 90% band",
        )
        ax.plot(x, sdf["outdoor_temp_c"], color="#8f3d2f", lw=1.0, alpha=0.85, label="Outdoor")
        ax.plot(x, sdf["mean_operative_temp_c"], color="#1f5a85", lw=1.4, label="Mean operative")
        ax.plot(x, sdf["heating_setpoint_c"], color="#bf7b30", lw=1.0, ls="--", label="Heating SP")
        ax.plot(x, sdf["cooling_setpoint_c"], color="#455aa0", lw=1.0, ls="--", label="Cooling SP")
        ax.set_title(titles[strategy], fontsize=11, fontweight="bold")
        ax.set_ylabel("Temp. (C)" if col == 0 else "")
        ax.grid(True, color="#d9d9d9", lw=0.5)

        ax = axes[1, col]
        ax.axhspan(-0.5, 0.5, color="#e5f0df", alpha=0.6, linewidth=0)
        ax.plot(x, sdf["mean_pmv"], color="#3f6d3a", lw=1.2, label="PMV")
        if strategy in {"nominal", "ordinal"}:
            ax2 = ax.twinx()
            ax2.plot(
                x,
                sdf["discomfort_probability"],
                color="#872f59",
                lw=1.1,
                alpha=0.9,
                label="P(|TSV|>=2)",
            )
            ax2.axhline(0.065, color="#872f59", lw=0.8, ls=":")
            ax2.axhline(0.35, color="#872f59", lw=0.8, ls="--")
            ax2.set_ylim(0, max(0.45, float(sdf["discomfort_probability"].max()) * 1.15))
            if col == len(strategies) - 1:
                ax2.set_ylabel("Tail prob.")
        ax.axhline(-0.5, color="#777777", lw=0.7, ls=":")
        ax.axhline(0.5, color="#777777", lw=0.7, ls=":")
        ax.set_ylabel("PMV" if col == 0 else "")
        ax.grid(True, color="#d9d9d9", lw=0.5)

        ax = axes[2, col]
        if strategy in {"nominal", "ordinal"}:
            ax.plot(x, sdf["expected_tsv"], color="#514c85", lw=1.2, label="E[TSV]")
            ax.axhline(0, color="#666666", lw=0.8)
            ax.set_ylim(-2.8, 2.8)
        else:
            demand = np.where(
                sdf["mean_operative_temp_c"] > sdf["cooling_setpoint_c"],
                1,
                np.where(sdf["mean_operative_temp_c"] < sdf["heating_setpoint_c"], -1, 0),
            )
            ax.step(x, demand, where="post", color="#514c85", lw=1.2, label="Thermostat demand")
            ax.set_ylim(-1.4, 1.4)
        ax.bar(
            x,
            np.where(sdf["hvac_on"], 0.18, 0.0),
            width=0.008,
            bottom=ax.get_ylim()[0],
            color="#2b6f6b",
            alpha=0.35,
            label="HVAC on",
        )
        ax.set_ylabel("Signal" if col == 0 else "")
        ax.grid(True, color="#d9d9d9", lw=0.5)

        ax = axes[3, col]
        step_hours = 0.25
        elec_kw = sdf["electricity_facility_j"] / (step_hours * 3600.0 * 1000.0)
        ax.plot(x, elec_kw, color="#303030", lw=1.0, label="Facility electric")
        ax.set_ylabel("kW" if col == 0 else "")
        ax.grid(True, color="#d9d9d9", lw=0.5)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=30)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"Medium Office 15-min supervisory control trace\n{weather_stem}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig_path = output_dir / "figs" / f"medium_office_temporal_grid_{weather_stem}.png"
    pdf_path = fig_path.with_suffix(".pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[plot] wrote figure: {fig_path}")
    print(f"[plot] wrote figure: {pdf_path}")
    return fig_path


def summarize_traces(output_dir: Path) -> Path:
    path = output_dir / "traces" / "medium_office_control_traces.csv"
    df = pd.read_csv(path)
    rows = []
    for (weather, strategy), sdf in df.groupby(["weather", "strategy"]):
        occ = sdf[sdf["occupied"]]
        rows.append(
            {
                "weather": weather,
                "strategy": strategy,
                "n_steps": int(len(sdf)),
                "occupied_steps": int(len(occ)),
                "mean_operative_temp_c_occ": float(occ["mean_operative_temp_c"].mean()),
                "pmv_violation_pct_occ": float((occ["mean_pmv"].abs() > 0.5).mean() * 100.0),
                "adaptive_violation_pct_occ": float(
                    (
                        (occ["mean_operative_temp_c"] < occ["comfort_low_c"])
                        | (occ["mean_operative_temp_c"] > occ["comfort_high_c"])
                    ).mean()
                    * 100.0
                ),
                "hvac_on_pct_all": float(sdf["hvac_on"].mean() * 100.0),
                "electricity_kwh": float(sdf["electricity_facility_j"].sum() / 3.6e6),
                "natural_gas_kwh": float(sdf["natural_gas_facility_j"].sum() / 3.6e6),
            }
        )
    out = output_dir / "summary" / "medium_office_trace_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["weather", "strategy"]).to_csv(out, index=False)
    print(f"[summary] wrote: {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--idf", type=Path, default=DEFAULT_IDF)
    parser.add_argument("--eplus-root", type=Path, default=DEFAULT_EPLUS)
    parser.add_argument("--weather", type=Path, nargs="+", default=[DEFAULT_WEATHER])
    parser.add_argument("--begin-month", type=int, default=7)
    parser.add_argument("--begin-day", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=7)
    parser.add_argument("--end-day", type=int, default=14)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["reference", "pmv", "nominal", "ordinal"],
        choices=["reference", "pmv", "nominal", "ordinal", "grid_naive", "grid_gated"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "models" / "control_predictors.joblib"
    metrics_path = args.output_dir / "models" / "control_predictor_metrics.json"

    bundle = None
    needs_predictors = any(s in {"nominal", "ordinal", "grid_gated"} for s in args.strategies)
    if needs_predictors:
        if args.skip_train and not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        if args.retrain or not model_path.exists():
            bundle = train_predictors(
                data_path=args.data,
                model_path=model_path,
                metrics_path=metrics_path,
                n_estimators=args.n_estimators,
                sample_limit=args.sample_limit,
            )
        else:
            print(f"[train] loading existing model: {model_path}")
            bundle = joblib.load(model_path)

    if not args.skip_sim:
        run_simulations(
            bundle=bundle,
            output_dir=args.output_dir,
            source_idf=args.idf,
            weather_paths=args.weather,
            eplus_root=args.eplus_root,
            begin_month=args.begin_month,
            begin_day=args.begin_day,
            end_month=args.end_month,
            end_day=args.end_day,
            strategies=args.strategies,
        )
        summarize_traces(args.output_dir)

    if not args.skip_plot:
        for weather in args.weather:
            make_temporal_plot(args.output_dir, weather.stem)

    print(f"[done] elapsed minutes: {(time.time() - start) / 60.0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

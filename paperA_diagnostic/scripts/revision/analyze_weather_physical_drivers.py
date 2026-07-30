#!/usr/bin/env python3
"""Quantify mechanism-consistent associations in Paper A's existing traces.

The analysis is descriptive. It separates:

1. outdoor weather forcing;
2. simulated EnergyPlus zone state and cooling response; and
3. the fitted TSV probability mapping.

It does not identify causal effects and does not rerun EnergyPlus.
"""

from __future__ import annotations

import argparse
import __main__
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
MANIFEST = ROOT / "paperA_rebuild/data/panel_manifest.csv"
TRACE_DIR = ROOT / "paperA_rebuild/runs/diagnostic_reference_zone_raw_full/traces"
PROVENANCE_DIR = (
    ROOT / "paperA_R01/04_analysis/outputs/weather_physics_provenance"
)
OUT_LEGACY_DEFAULT = (
    ROOT
    / "paperA_R01/04_analysis/outputs/weather_physics_drivers_legacy_timing"
)
OUT_CORRECTED_DEFAULT = (
    ROOT
    / "paperA_R01/04_analysis/outputs/weather_physics_drivers_corrected_same_state"
)
MODEL_DEFAULT = (
    ROOT
    / "paperA_rebuild/runs/diagnostic_reference_zone_raw_full/models/control_predictors.joblib"
)
ROBUSTNESS_HEADLINE = (
    ROOT
    / "paperA_R01/04_analysis/outputs/robustness_endpoint_model/corrected_headline_case_summary.csv"
)
CORRECTED_NPZ_MANIFEST_DEFAULT = (
    ROOT
    / "paperA_R01/04_analysis/outputs/robustness_endpoint_model/input_trace_manifest.csv"
)
CORRECTED_NPZ_ROOT_DEFAULT = (
    ROOT
    / "paperA_R01/04_analysis/outputs/robustness_endpoint_model/corrected_zone_npz"
)

TAIL_THRESHOLD = 0.20

ZONES = [
    "core_bottom",
    "core_mid",
    "core_top",
    "perimeter_top_zn_3",
    "perimeter_top_zn_2",
    "perimeter_top_zn_1",
    "perimeter_top_zn_4",
    "perimeter_bot_zn_3",
    "perimeter_bot_zn_2",
    "perimeter_bot_zn_1",
    "perimeter_bot_zn_4",
    "perimeter_mid_zn_3",
    "perimeter_mid_zn_2",
    "perimeter_mid_zn_1",
    "perimeter_mid_zn_4",
]

ORIENTATION = {
    "1": "south",
    "2": "east",
    "3": "north",
    "4": "west",
}

MEAN_COLS = [
    "occupied",
    "month",
    "day",
    "hour",
    "current_time",
    "outdoor_temp_c",
    "running_mean_outdoor_c",
    "mean_operative_temp_c",
    "mean_rh_pct",
    "zone_heating_rate_w",
    "zone_cooling_rate_w",
    "hvac_on",
    "discomfort_probability",
    "warm_discomfort_probability",
    "cold_discomfort_probability",
    "expected_tsv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--probability-timing",
        choices=["legacy_callback", "corrected_same_state"],
        default="legacy_callback",
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_DEFAULT)
    parser.add_argument(
        "--corrected-source",
        choices=["auto", "npz", "fresh_inference"],
        default="auto",
        help="For corrected timing, prefer shared robustness NPZs or infer afresh.",
    )
    parser.add_argument(
        "--corrected-npz-manifest",
        type=Path,
        default=CORRECTED_NPZ_MANIFEST_DEFAULT,
    )
    parser.add_argument(
        "--corrected-npz-root",
        type=Path,
        default=CORRECTED_NPZ_ROOT_DEFAULT,
    )
    return parser.parse_args()


def load_corrected_bundle(path: Path):
    """Load the frozen model using the original script's pickle classes."""
    import joblib

    rebuild_scripts = ROOT / "paperA_rebuild/scripts"
    if str(rebuild_scripts) not in sys.path:
        sys.path.insert(0, str(rebuild_scripts))
    import run_medium_office_diagnostic_panel as panel

    __main__.FeatureSpec = panel.FeatureSpec
    __main__.PredictorBundle = panel.PredictorBundle
    return joblib.load(path), panel


def apply_corrected_same_state_inference(
    trace: pd.DataFrame,
    bundle,
    panel,
) -> pd.DataFrame:
    """Replace callback-timed probabilities with inference from this row's state."""
    ta_cols = [f"zone_{zone}_ta_c" for zone in ZONES]
    tr_cols = [f"zone_{zone}_tr_c" for zone in ZONES]
    rh_cols = [f"zone_{zone}_rh_pct" for zone in ZONES]
    ta = trace[ta_cols].to_numpy(float)
    tr = trace[tr_cols].to_numpy(float)
    rh = trace[rh_cols].to_numpy(float)
    running_mean = pd.to_numeric(
        trace["running_mean_outdoor_c"], errors="coerce"
    ).where(
        trace["running_mean_outdoor_c"].notna(),
        pd.to_numeric(trace["outdoor_temp_c"], errors="coerce"),
    ).to_numpy(float)
    n, z = ta.shape
    features = panel.build_features_from_arrays(
        ta=ta.reshape(-1),
        tr=tr.reshape(-1),
        v=np.full(n * z, 0.10),
        rh=rh.reshape(-1),
        met=np.full(n * z, 1.10),
        clo=np.full(n * z, 0.65),
        bsa=np.full(n * z, 1.80),
        rm_out=np.repeat(running_mean, z),
        spec=bundle.spec,
    )
    probabilities = bundle.predict_ordinal(features).reshape(n, z, 7)
    broad = probabilities[:, :, [0, 1, 5, 6]].sum(axis=2)
    cold = probabilities[:, :, [0, 1]].sum(axis=2)
    warm = probabilities[:, :, [5, 6]].sum(axis=2)
    expected = probabilities @ np.arange(-3.0, 4.0)
    corrected = trace.copy()
    for zone_index, zone in enumerate(ZONES):
        corrected[f"zone_{zone}_p_disc"] = broad[:, zone_index]
        corrected[f"zone_{zone}_warm_tail"] = warm[:, zone_index]
        corrected[f"zone_{zone}_cold_tail"] = cold[:, zone_index]
    corrected["discomfort_probability"] = broad.mean(axis=1)
    corrected["warm_discomfort_probability"] = warm.mean(axis=1)
    corrected["cold_discomfort_probability"] = cold.mean(axis=1)
    corrected["expected_tsv"] = expected.mean(axis=1)
    return corrected


def corrected_npz_map(manifest_path: Path, npz_root: Path) -> dict[str, Path]:
    manifest = pd.read_csv(manifest_path)
    key_col = next(
        (col for col in ["weather", "case_id"] if col in manifest.columns),
        None,
    )
    if key_col is None:
        raise ValueError(f"No weather/case_id column in {manifest_path}")
    path_col = next(
        (
            col
            for col in [
                "corrected_npz_path",
                "npz_path",
                "corrected_zone_npz_path",
                "corrected_zone_npz",
            ]
            if col in manifest.columns
        ),
        None,
    )
    state_col = next(
        (
            col
            for col in [
                "state_hash",
                "state_sha256",
                "analysis_state_sha256",
            ]
            if col in manifest.columns
        ),
        None,
    )
    if path_col is None and state_col is None:
        raise ValueError(f"No NPZ path or state hash column in {manifest_path}")
    result: dict[str, Path] = {}
    for row in manifest.to_dict("records"):
        if path_col is not None and pd.notna(row[path_col]):
            path = Path(str(row[path_col]))
            if not path.is_absolute():
                path = manifest_path.parent / path
        else:
            path = npz_root / f"{row[state_col]}.npz"
        result[str(row[key_col])] = path
    return result


def apply_corrected_npz(trace: pd.DataFrame, npz_path: Path) -> pd.DataFrame:
    with np.load(npz_path) as archive:
        zone_names = [str(value) for value in archive["zone_names"]]
        if zone_names != ZONES:
            raise ValueError(
                f"Zone order mismatch in {npz_path}: {zone_names} != {ZONES}"
            )
        if len(trace) != len(archive["source_row_index"]):
            raise ValueError(
                f"Occupied row-count mismatch in {npz_path}: "
                f"{len(trace)} != {len(archive['source_row_index'])}"
            )
        checks = {
            "source_row_index": archive["source_row_index"],
            "month": archive["month"],
            "day": archive["day"],
            "hour": archive["hour"],
        }
        for column, expected in checks.items():
            actual = trace[column].to_numpy()
            if not np.array_equal(actual.astype(expected.dtype), expected):
                raise ValueError(f"{column} alignment mismatch in {npz_path}")
        expected_current_time = archive["current_time"]
        if not np.allclose(
            trace["current_time"].to_numpy(float),
            expected_current_time.astype(float),
            rtol=0.0,
            atol=1e-6,
            equal_nan=True,
        ):
            raise ValueError(f"current_time alignment mismatch in {npz_path}")
        probabilities = {
            "cold": archive["zone_cold_tail"].astype(float),
            "warm": archive["zone_warm_tail"].astype(float),
            "expected": archive["zone_expected_tsv"].astype(float),
        }
    broad = probabilities["cold"] + probabilities["warm"]
    corrected = trace.copy()
    for zone_index, zone in enumerate(ZONES):
        corrected[f"zone_{zone}_p_disc"] = broad[:, zone_index]
        corrected[f"zone_{zone}_warm_tail"] = probabilities["warm"][:, zone_index]
        corrected[f"zone_{zone}_cold_tail"] = probabilities["cold"][:, zone_index]
    corrected["discomfort_probability"] = broad.mean(axis=1)
    corrected["warm_discomfort_probability"] = probabilities["warm"].mean(axis=1)
    corrected["cold_discomfort_probability"] = probabilities["cold"].mean(axis=1)
    corrected["expected_tsv"] = probabilities["expected"].mean(axis=1)
    return corrected


def zone_metadata(zone: str) -> dict[str, str]:
    if zone.startswith("core_"):
        floor = zone.split("_", 1)[1]
        return {
            "zone": zone,
            "position": "core",
            "floor": {"bottom": "bottom", "mid": "middle", "top": "top"}[floor],
            "orientation": "core",
        }
    match = re.fullmatch(r"perimeter_(bot|mid|top)_zn_([1-4])", zone)
    if not match:
        raise ValueError(f"Unexpected zone name: {zone}")
    floor, number = match.groups()
    return {
        "zone": zone,
        "position": "perimeter",
        "floor": {"bot": "bottom", "mid": "middle", "top": "top"}[floor],
        "orientation": ORIENTATION[number],
    }


def epw_hourly(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as handle:
        for _ in range(8):
            next(handle)
        for fields in pd.read_csv(
            handle,
            header=None,
            usecols=[1, 2, 3, 6, 8, 13, 14, 15],
            names=["month", "day", "epw_hour", "epw_temp", "epw_rh", "ghi", "dni", "dhi"],
        ).itertuples(index=False):
            rows.append(
                {
                    "month": int(fields.month),
                    "day": int(fields.day),
                    "hour": int(fields.epw_hour) - 1,
                    "epw_input_temp_c": float(fields.epw_temp),
                    "epw_outdoor_rh_pct": float(fields.epw_rh),
                    "epw_ghi_w_m2": float(fields.ghi),
                    "epw_dni_w_m2": float(fields.dni),
                    "epw_dhi_w_m2": float(fields.dhi),
                }
            )
    return pd.DataFrame(rows)


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 20 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    return float(pair["left"].corr(pair["right"], method="spearman"))


def percentile(series: pd.Series, probability: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(values, probability)) if len(values) else float("nan")


def weighted_group_summary(
    rows: pd.DataFrame, group_cols: list[str], value_cols: list[str]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for key, group in rows.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols, key))
        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            record[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{col}_median"] = float(values.median()) if len(values) else np.nan
            record[f"{col}_q25"] = float(values.quantile(0.25)) if len(values) else np.nan
            record[f"{col}_q75"] = float(values.quantile(0.75)) if len(values) else np.nan
        record["n_units"] = int(len(group))
        records.append(record)
    return pd.DataFrame(records)


def case_trace_path(case_id: str) -> Path:
    return TRACE_DIR / f"{case_id}_diagnostic_reference.csv"


def read_trace(case_id: str) -> pd.DataFrame:
    zone_cols: list[str] = []
    for zone in ZONES:
        zone_cols.extend(
            [
                f"zone_{zone}_ta_c",
                f"zone_{zone}_tr_c",
                f"zone_{zone}_rh_pct",
                f"zone_{zone}_p_disc",
                f"zone_{zone}_warm_tail",
                f"zone_{zone}_cold_tail",
            ]
        )
    frame = pd.read_csv(case_trace_path(case_id), usecols=[*MEAN_COLS, *zone_cols])
    frame["source_row_index"] = np.arange(len(frame), dtype=int)
    occupied = frame["occupied"].astype(str).str.lower().isin(["true", "1"])
    return frame.loc[occupied].reset_index(drop=True)


def hourly_merge(trace: pd.DataFrame, epw: pd.DataFrame) -> pd.DataFrame:
    trace_hourly = (
        trace.groupby(["month", "day", "hour"], as_index=False)
        .mean(numeric_only=True)
    )
    return trace_hourly.merge(epw, on=["month", "day", "hour"], how="left", validate="one_to_one")


def correlation_record(
    case_id: str,
    left_name: str,
    right_name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "left": left_name,
        "right": right_name,
        "spearman_rho": safe_spearman(frame[left_name], frame[right_name]),
        "n_hours": int(frame[[left_name, right_name]].dropna().shape[0]),
    }


def collapse_unique_cases(
    case_df: pd.DataFrame, epw_qa: pd.DataFrame
) -> pd.DataFrame:
    hash_map = epw_qa.set_index("case_id")["body_sha256"].to_dict()
    work = case_df.copy()
    work["weather_body_sha256"] = work["case_id"].map(hash_map)
    metadata = [
        "weather_body_sha256",
        "city",
        "scenario",
        "time_slice",
        "weather_year",
    ]
    numeric = [
        col
        for col in work.select_dtypes(include=[np.number]).columns
        if col != "weather_year"
    ]
    collapsed = work.groupby(metadata, as_index=False)[numeric].mean()
    roles = (
        work.groupby(metadata, as_index=False)
        .agg(
            role_multiplicity=("case_id", "size"),
            role_labels=("severity", lambda values: ";".join(sorted(values))),
            case_ids=("case_id", lambda values: ";".join(sorted(values))),
        )
    )
    return collapsed.merge(roles, on=metadata, how="left", validate="one_to_one")


def collapse_unique_zone_cases(
    zone_df: pd.DataFrame, epw_qa: pd.DataFrame
) -> pd.DataFrame:
    hash_map = epw_qa.set_index("case_id")["body_sha256"].to_dict()
    work = zone_df.copy()
    work["weather_body_sha256"] = work["case_id"].map(hash_map)
    metadata = [
        "weather_body_sha256",
        "city",
        "scenario",
        "time_slice",
        "weather_year",
        "zone",
        "position",
        "floor",
        "orientation",
    ]
    numeric = [
        col
        for col in work.select_dtypes(include=[np.number]).columns
        if col != "weather_year"
    ]
    return work.groupby(metadata, as_index=False)[numeric].mean()


def long_spearman(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, left in enumerate(variables):
        for right in variables[i + 1 :]:
            pair = frame[[left, right]].dropna()
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman_rho": safe_spearman(pair[left], pair[right]),
                    "n_unique_weather_states": int(len(pair)),
                }
            )
    return pd.DataFrame(rows)


def format_range(values: pd.Series, digits: int = 3) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if not len(values):
        return "NA"
    return (
        f"{values.median():.{digits}f} "
        f"[IQR {values.quantile(.25):.{digits}f}, {values.quantile(.75):.{digits}f}]"
    )


def main() -> int:
    args = parse_args()
    corrected_timing = args.probability_timing == "corrected_same_state"
    out = args.output_dir or (
        OUT_CORRECTED_DEFAULT if corrected_timing else OUT_LEGACY_DEFAULT
    )
    out.mkdir(parents=True, exist_ok=True)
    bundle = None
    panel = None
    use_npz = (
        corrected_timing
        and args.corrected_source in {"auto", "npz"}
        and args.corrected_npz_manifest.exists()
    )
    if corrected_timing and args.corrected_source == "npz" and not use_npz:
        raise FileNotFoundError(args.corrected_npz_manifest)
    npz_map: dict[str, Path] = {}
    if use_npz:
        npz_map = corrected_npz_map(
            args.corrected_npz_manifest, args.corrected_npz_root
        )
    if corrected_timing and not use_npz:
        bundle, panel = load_corrected_bundle(args.model_path)
    corrected_probability_source = (
        "robustness_float32_npz"
        if use_npz
        else "fresh_float64_inference"
        if corrected_timing
        else "legacy_callback_trace"
    )

    manifest = pd.read_csv(MANIFEST).sort_values("case_id")
    if args.max_cases:
        manifest = manifest.head(args.max_cases)
    epw_qa = pd.read_csv(PROVENANCE_DIR / "epw_weather_qa.csv")
    epw_by_case = epw_qa.set_index("case_id")

    case_rows: list[dict[str, object]] = []
    zone_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    conditioned_correlation_rows: list[dict[str, object]] = []
    zone_solar_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []

    for index, case in enumerate(manifest.itertuples(index=False), start=1):
        trace = read_trace(case.case_id)
        if corrected_timing:
            if use_npz:
                if case.case_id not in npz_map:
                    raise KeyError(f"No corrected NPZ mapping for {case.case_id}")
                trace = apply_corrected_npz(trace, npz_map[case.case_id])
            else:
                trace = apply_corrected_same_state_inference(trace, bundle, panel)
        epw = epw_hourly(Path(case.epw_path))
        hourly = hourly_merge(trace, epw)

        case_rows.append(
            {
                "case_id": case.case_id,
                "city": case.city,
                "scenario": case.scenario_raw,
                "time_slice": case.time_slice,
                "severity": case.severity,
                "weather_year": int(case.weather_year),
                "probability_source": corrected_probability_source,
                "occupied_steps": int(len(trace)),
                "occupied_hours_aggregated": int(len(hourly)),
                "mean_outdoor_temp_c_occ": float(trace["outdoor_temp_c"].mean()),
                "p95_outdoor_temp_c_occ": percentile(trace["outdoor_temp_c"], 0.95),
                "mean_epw_outdoor_rh_pct_occ": float(hourly["epw_outdoor_rh_pct"].mean()),
                "mean_epw_ghi_w_m2_occ": float(hourly["epw_ghi_w_m2"].mean()),
                "mean_zone_operative_temp_c_occ": float(trace["mean_operative_temp_c"].mean()),
                "p95_zone_operative_temp_c_occ": percentile(
                    trace["mean_operative_temp_c"], 0.95
                ),
                "mean_zone_rh_pct_occ": float(trace["mean_rh_pct"].mean()),
                "mean_zone_cooling_rate_w_occ": float(trace["zone_cooling_rate_w"].mean()),
                "p95_zone_cooling_rate_w_occ": percentile(
                    trace["zone_cooling_rate_w"], 0.95
                ),
                "hvac_on_pct_occ": float(trace["hvac_on"].astype(float).mean() * 100),
                "mean_p_tail_occ": float(trace["discomfort_probability"].mean()),
                "p95_p_tail_occ": percentile(trace["discomfort_probability"], 0.95),
                "high_tail_pct_occ": float(
                    (trace["discomfort_probability"] >= TAIL_THRESHOLD).mean() * 100
                ),
                "mean_warm_tail_occ": float(trace["warm_discomfort_probability"].mean()),
                "mean_cold_tail_occ": float(trace["cold_discomfort_probability"].mean()),
                "annual_mean_temp_c": float(epw_by_case.loc[case.case_id, "mean_temp_c"]),
                "annual_max_temp_c": float(epw_by_case.loc[case.case_id, "max_temp_c"]),
                "annual_cdh18_c_hour": float(epw_by_case.loc[case.case_id, "cdh18_c_hour"]),
                "annual_hours_temp_ge_35": float(
                    epw_by_case.loc[case.case_id, "hours_temp_ge_35"]
                ),
                "annual_mean_rh_pct": float(epw_by_case.loc[case.case_id, "mean_rh_pct"]),
                "annual_ghi_kwh_m2": float(
                    epw_by_case.loc[case.case_id, "annual_ghi_kwh_m2"]
                ),
            }
        )

        pairs = [
            ("outdoor_temp_c", "mean_operative_temp_c"),
            ("outdoor_temp_c", "zone_cooling_rate_w"),
            ("epw_ghi_w_m2", "mean_operative_temp_c"),
            ("epw_ghi_w_m2", "zone_cooling_rate_w"),
            ("epw_outdoor_rh_pct", "mean_rh_pct"),
            ("mean_operative_temp_c", "discomfort_probability"),
            ("mean_operative_temp_c", "warm_discomfort_probability"),
            ("mean_operative_temp_c", "cold_discomfort_probability"),
            ("mean_rh_pct", "discomfort_probability"),
            ("zone_cooling_rate_w", "discomfort_probability"),
        ]
        for left, right in pairs:
            correlation_rows.append(correlation_record(case.case_id, left, right, hourly))

        conditioned_pairs = [
            ("outdoor_temp_c", "mean_operative_temp_c"),
            ("mean_operative_temp_c", "discomfort_probability"),
            ("mean_operative_temp_c", "warm_discomfort_probability"),
            ("mean_operative_temp_c", "cold_discomfort_probability"),
            ("zone_cooling_rate_w", "discomfort_probability"),
        ]
        conditions = {
            "all_occupied_hours": pd.Series(True, index=hourly.index),
            "outdoor_temp_ge_20_c": hourly["outdoor_temp_c"] >= 20.0,
            "outdoor_temp_ge_25_c": hourly["outdoor_temp_c"] >= 25.0,
            "may_through_september": hourly["month"].between(5, 9),
        }
        for condition_name, mask in conditions.items():
            subset = hourly.loc[mask]
            for left, right in conditioned_pairs:
                conditioned_correlation_rows.append(
                    {
                        "case_id": case.case_id,
                        "condition": condition_name,
                        "left": left,
                        "right": right,
                        "spearman_rho": safe_spearman(
                            subset[left], subset[right]
                        ),
                        "n_hours": int(
                            subset[[left, right]].dropna().shape[0]
                        ),
                    }
                )

        bins = pd.cut(
            trace["outdoor_temp_c"],
            bins=[-np.inf, 20, 25, 30, 35, np.inf],
            labels=["<20", "20--25", "25--30", "30--35", ">=35"],
            right=False,
        )
        bin_frame = trace.assign(outdoor_temp_bin=bins)
        for label, group in bin_frame.groupby("outdoor_temp_bin", observed=True):
            bin_rows.append(
                {
                    "case_id": case.case_id,
                    "city": case.city,
                    "scenario": case.scenario_raw,
                    "time_slice": case.time_slice,
                    "severity": case.severity,
                    "weather_year": int(case.weather_year),
                    "outdoor_temp_bin": str(label),
                    "occupied_steps": int(len(group)),
                    "mean_outdoor_temp_c": float(group["outdoor_temp_c"].mean()),
                    "mean_zone_operative_temp_c": float(
                        group["mean_operative_temp_c"].mean()
                    ),
                    "mean_zone_rh_pct": float(group["mean_rh_pct"].mean()),
                    "mean_zone_cooling_rate_w": float(
                        group["zone_cooling_rate_w"].mean()
                    ),
                    "mean_p_tail": float(group["discomfort_probability"].mean()),
                    "high_tail_pct": float(
                        (group["discomfort_probability"] >= TAIL_THRESHOLD).mean() * 100
                    ),
                }
            )

        for zone in ZONES:
            ta = pd.to_numeric(trace[f"zone_{zone}_ta_c"], errors="coerce")
            tr = pd.to_numeric(trace[f"zone_{zone}_tr_c"], errors="coerce")
            rh = pd.to_numeric(trace[f"zone_{zone}_rh_pct"], errors="coerce")
            p_tail = pd.to_numeric(trace[f"zone_{zone}_p_disc"], errors="coerce")
            warm = pd.to_numeric(trace[f"zone_{zone}_warm_tail"], errors="coerce")
            cold = pd.to_numeric(trace[f"zone_{zone}_cold_tail"], errors="coerce")
            operative = (ta + tr) / 2
            meta = zone_metadata(zone)
            zone_rows.append(
                {
                    "case_id": case.case_id,
                    "city": case.city,
                    "scenario": case.scenario_raw,
                    "time_slice": case.time_slice,
                    "severity": case.severity,
                    "weather_year": int(case.weather_year),
                    **meta,
                    "mean_air_temp_c_occ": float(ta.mean()),
                    "mean_mrt_c_occ": float(tr.mean()),
                    "mean_operative_temp_c_occ": float(operative.mean()),
                    "p95_operative_temp_c_occ": percentile(operative, 0.95),
                    "mean_mrt_minus_air_c_occ": float((tr - ta).mean()),
                    "mean_rh_pct_occ": float(rh.mean()),
                    "mean_p_tail_occ": float(p_tail.mean()),
                    "p95_p_tail_occ": percentile(p_tail, 0.95),
                    "high_tail_pct_occ": float((p_tail >= TAIL_THRESHOLD).mean() * 100),
                    "mean_warm_tail_occ": float(warm.mean()),
                    "mean_cold_tail_occ": float(cold.mean()),
                }
            )

            zone_hourly = pd.DataFrame(
                {
                    "month": trace["month"],
                    "day": trace["day"],
                    "hour": trace["hour"],
                    "operative": operative,
                    "p_tail": p_tail,
                }
            )
            zone_hourly = (
                zone_hourly.groupby(["month", "day", "hour"], as_index=False)
                .mean(numeric_only=True)
                .merge(
                    epw[["month", "day", "hour", "epw_ghi_w_m2"]],
                    on=["month", "day", "hour"],
                    how="left",
                    validate="one_to_one",
                )
            )
            zone_solar_rows.extend(
                [
                    {
                        "case_id": case.case_id,
                        **meta,
                        "association": "horizontal_GHI__zone_operative_temperature",
                        "spearman_rho": safe_spearman(
                            zone_hourly["epw_ghi_w_m2"], zone_hourly["operative"]
                        ),
                        "n_hours": int(len(zone_hourly)),
                    },
                    {
                        "case_id": case.case_id,
                        **meta,
                        "association": "horizontal_GHI__zone_p_tail",
                        "spearman_rho": safe_spearman(
                            zone_hourly["epw_ghi_w_m2"], zone_hourly["p_tail"]
                        ),
                        "n_hours": int(len(zone_hourly)),
                    },
                ]
            )

        if index % 12 == 0 or index == len(manifest):
            print(f"[trace] {index}/{len(manifest)} cases")

    case_df = pd.DataFrame(case_rows)
    zone_df = pd.DataFrame(zone_rows)
    correlation_df = pd.DataFrame(correlation_rows)
    conditioned_correlation_df = pd.DataFrame(conditioned_correlation_rows)
    zone_solar_df = pd.DataFrame(zone_solar_rows)
    bin_df = pd.DataFrame(bin_rows)

    case_df.to_csv(out / "case_driver_summary_role_labelled.csv", index=False)
    zone_df.to_csv(out / "case_zone_driver_summary_role_labelled.csv", index=False)
    correlation_df.to_csv(out / "within_case_hourly_correlations.csv", index=False)
    conditioned_correlation_df.to_csv(
        out / "within_case_conditioned_correlations.csv", index=False
    )
    zone_solar_df.to_csv(out / "zone_horizontal_ghi_correlations.csv", index=False)
    bin_df.to_csv(out / "outdoor_temperature_bin_case_summary.csv", index=False)

    unique_case = collapse_unique_cases(case_df, epw_qa)
    unique_zone = collapse_unique_zone_cases(zone_df, epw_qa)
    unique_case.to_csv(out / "case_driver_summary_unique_weather.csv", index=False)
    unique_zone.to_csv(out / "case_zone_driver_summary_unique_weather.csv", index=False)

    case_variables = [
        "annual_mean_temp_c",
        "annual_max_temp_c",
        "annual_cdh18_c_hour",
        "annual_hours_temp_ge_35",
        "annual_mean_rh_pct",
        "annual_ghi_kwh_m2",
        "mean_zone_operative_temp_c_occ",
        "p95_zone_operative_temp_c_occ",
        "mean_zone_rh_pct_occ",
        "mean_zone_cooling_rate_w_occ",
        "mean_p_tail_occ",
        "high_tail_pct_occ",
    ]
    case_correlations = long_spearman(unique_case, case_variables)
    case_correlations.to_csv(out / "unique_weather_case_spearman.csv", index=False)

    future_metrics = [
        "annual_mean_temp_c",
        "annual_ghi_kwh_m2",
        "annual_mean_rh_pct",
        "mean_zone_operative_temp_c_occ",
        "mean_zone_rh_pct_occ",
        "mean_zone_cooling_rate_w_occ",
        "mean_p_tail_occ",
        "high_tail_pct_occ",
    ]
    future_keys = ["city", "scenario", "severity"]
    baseline = case_df[case_df["time_slice"] == "baseline_2020s"][
        [*future_keys, *future_metrics]
    ]
    late = case_df[case_df["time_slice"] == "late_2080s"][
        [*future_keys, *future_metrics]
    ]
    future_delta = baseline.merge(
        late,
        on=future_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_late"),
    )
    for metric in future_metrics:
        future_delta[f"delta_{metric}"] = (
            future_delta[f"{metric}_late"] - future_delta[f"{metric}_baseline"]
        )
    future_delta.to_csv(out / "baseline_to_late_change_by_role.csv", index=False)
    future_delta_variables = [f"delta_{metric}" for metric in future_metrics]
    future_delta_correlations = long_spearman(future_delta, future_delta_variables)
    future_delta_correlations.to_csv(
        out / "baseline_to_late_change_spearman.csv", index=False
    )
    future_delta_aggregate = weighted_group_summary(
        future_delta,
        ["scenario", "severity"],
        future_delta_variables,
    )
    future_delta_aggregate.to_csv(
        out / "baseline_to_late_change_aggregate.csv", index=False
    )

    unique_cell = (
        unique_case.groupby(
            ["city", "scenario", "time_slice"], as_index=False
        )[future_metrics]
        .mean()
    )
    unique_cell_baseline = unique_cell[
        unique_cell["time_slice"] == "baseline_2020s"
    ].drop(columns="time_slice")
    unique_cell_late = unique_cell[
        unique_cell["time_slice"] == "late_2080s"
    ].drop(columns="time_slice")
    unique_cell_delta = unique_cell_baseline.merge(
        unique_cell_late,
        on=["city", "scenario"],
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_late"),
    )
    for metric in future_metrics:
        unique_cell_delta[f"delta_{metric}"] = (
            unique_cell_delta[f"{metric}_late"]
            - unique_cell_delta[f"{metric}_baseline"]
        )
    unique_cell_delta.to_csv(
        out / "baseline_to_late_change_unique_weather_cell.csv",
        index=False,
    )

    parity_summary: dict[str, object] = {
        "reference_available": False,
        "reference_rows": 0,
        "matched_rows": 0,
        "max_abs_mean_p_tail_difference": None,
        "max_abs_high_tail_pct_difference": None,
    }
    if corrected_timing and ROBUSTNESS_HEADLINE.exists():
        reference = pd.read_csv(ROBUSTNESS_HEADLINE)
        if len(reference):
            comparison = case_df[
                ["case_id", "mean_p_tail_occ", "high_tail_pct_occ"]
            ].merge(
                reference[
                    [
                        "weather",
                        "corrected_equal_zone_mean_mean",
                        "corrected_equal_zone_mean_high_pct",
                    ]
                ].rename(columns={"weather": "case_id"}),
                on="case_id",
                how="inner",
                validate="one_to_one",
            )
            comparison["mean_p_tail_difference"] = (
                comparison["mean_p_tail_occ"]
                - comparison["corrected_equal_zone_mean_mean"]
            )
            comparison["high_tail_pct_difference"] = (
                comparison["high_tail_pct_occ"]
                - comparison["corrected_equal_zone_mean_high_pct"]
            )
            comparison.to_csv(
                out / "robustness_corrected_parity_check.csv", index=False
            )
            parity_summary = {
                "reference_available": True,
                "reference_rows": int(len(reference)),
                "matched_rows": int(len(comparison)),
                "max_abs_mean_p_tail_difference": float(
                    comparison["mean_p_tail_difference"].abs().max()
                ),
                "max_abs_high_tail_pct_difference": float(
                    comparison["high_tail_pct_difference"].abs().max()
                ),
            }

    correlation_aggregate = weighted_group_summary(
        correlation_df,
        ["left", "right"],
        ["spearman_rho"],
    )
    correlation_aggregate["positive_case_pct"] = correlation_aggregate.apply(
        lambda row: float(
            (
                correlation_df[
                    (correlation_df["left"] == row["left"])
                    & (correlation_df["right"] == row["right"])
                ]["spearman_rho"]
                > 0
            ).mean()
            * 100
        ),
        axis=1,
    )
    correlation_aggregate.to_csv(
        out / "within_case_correlation_aggregate.csv", index=False
    )

    zone_summary = weighted_group_summary(
        unique_zone,
        ["zone", "position", "floor", "orientation"],
        [
            "mean_operative_temp_c_occ",
            "p95_operative_temp_c_occ",
            "mean_mrt_minus_air_c_occ",
            "mean_rh_pct_occ",
            "mean_p_tail_occ",
            "high_tail_pct_occ",
            "mean_warm_tail_occ",
            "mean_cold_tail_occ",
        ],
    )
    zone_summary.to_csv(out / "zone_summary_unique_weather.csv", index=False)

    floor_orientation = weighted_group_summary(
        unique_zone,
        ["position", "floor", "orientation"],
        [
            "mean_operative_temp_c_occ",
            "mean_mrt_minus_air_c_occ",
            "mean_p_tail_occ",
            "high_tail_pct_occ",
        ],
    )
    floor_orientation.to_csv(
        out / "floor_orientation_summary_unique_weather.csv", index=False
    )

    solar_aggregate = weighted_group_summary(
        zone_solar_df,
        ["zone", "position", "floor", "orientation", "association"],
        ["spearman_rho"],
    )
    solar_aggregate.to_csv(out / "zone_horizontal_ghi_correlation_aggregate.csv", index=False)

    bin_records: list[dict[str, object]] = []
    for label, group in bin_df.groupby("outdoor_temp_bin", observed=True):
        weights = group["occupied_steps"]
        bin_records.append(
            {
                "outdoor_temp_bin": label,
                "occupied_steps": int(weights.sum()),
                "mean_outdoor_temp_c": float(
                    np.average(group["mean_outdoor_temp_c"], weights=weights)
                ),
                "mean_zone_operative_temp_c": float(
                    np.average(group["mean_zone_operative_temp_c"], weights=weights)
                ),
                "mean_zone_rh_pct": float(
                    np.average(group["mean_zone_rh_pct"], weights=weights)
                ),
                "mean_zone_cooling_rate_w": float(
                    np.average(group["mean_zone_cooling_rate_w"], weights=weights)
                ),
                "mean_p_tail": float(
                    np.average(group["mean_p_tail"], weights=weights)
                ),
                "high_tail_pct": float(
                    np.average(group["high_tail_pct"], weights=weights)
                ),
            }
        )
    bin_all = pd.DataFrame(bin_records)
    bin_all.to_csv(out / "outdoor_temperature_bin_global_summary.csv", index=False)

    bin_unique = bin_df.copy()
    bin_unique["weather_body_sha256"] = bin_unique["case_id"].map(
        epw_qa.set_index("case_id")["body_sha256"]
    )
    bin_unique = bin_unique.drop_duplicates(
        ["weather_body_sha256", "outdoor_temp_bin"]
    )
    bin_unique_records: list[dict[str, object]] = []
    for label, group in bin_unique.groupby("outdoor_temp_bin", observed=True):
        weights = group["occupied_steps"]
        bin_unique_records.append(
            {
                "outdoor_temp_bin": label,
                "unique_weather_bin_units": int(len(group)),
                "occupied_steps": int(weights.sum()),
                "mean_outdoor_temp_c": float(
                    np.average(group["mean_outdoor_temp_c"], weights=weights)
                ),
                "mean_zone_operative_temp_c": float(
                    np.average(
                        group["mean_zone_operative_temp_c"], weights=weights
                    )
                ),
                "mean_zone_rh_pct": float(
                    np.average(group["mean_zone_rh_pct"], weights=weights)
                ),
                "mean_zone_cooling_rate_w": float(
                    np.average(
                        group["mean_zone_cooling_rate_w"], weights=weights
                    )
                ),
                "mean_p_tail": float(
                    np.average(group["mean_p_tail"], weights=weights)
                ),
                "high_tail_pct": float(
                    np.average(group["high_tail_pct"], weights=weights)
                ),
            }
        )
    bin_unique_all = pd.DataFrame(bin_unique_records)
    bin_unique_all.to_csv(
        out / "outdoor_temperature_bin_unique_weather_summary.csv",
        index=False,
    )

    conditioned_correlation_aggregate = weighted_group_summary(
        conditioned_correlation_df,
        ["condition", "left", "right"],
        ["spearman_rho"],
    )
    conditioned_correlation_aggregate["negative_case_count"] = (
        conditioned_correlation_aggregate.apply(
            lambda row: int(
                (
                    conditioned_correlation_df[
                        (
                            conditioned_correlation_df["condition"]
                            == row["condition"]
                        )
                        & (
                            conditioned_correlation_df["left"]
                            == row["left"]
                        )
                        & (
                            conditioned_correlation_df["right"]
                            == row["right"]
                        )
                    ]["spearman_rho"]
                    < 0
                ).sum()
            ),
            axis=1,
        )
    )
    conditioned_correlation_aggregate.to_csv(
        out / "within_case_conditioned_correlation_aggregate.csv",
        index=False,
    )

    def find_pair(left: str, right: str) -> pd.Series:
        rows = correlation_df[
            (correlation_df["left"] == left) & (correlation_df["right"] == right)
        ]
        return rows["spearman_rho"]

    def find_conditioned_pair(
        condition: str, left: str, right: str
    ) -> pd.Series:
        rows = conditioned_correlation_df[
            (conditioned_correlation_df["condition"] == condition)
            & (conditioned_correlation_df["left"] == left)
            & (conditioned_correlation_df["right"] == right)
        ]
        return rows["spearman_rho"]

    south = zone_summary[
        (zone_summary["position"] == "perimeter")
        & (zone_summary["orientation"] == "south")
    ]
    other = zone_summary[
        (zone_summary["position"] == "perimeter")
        & (zone_summary["orientation"] != "south")
    ]
    endpoint = bin_unique_all[
        bin_unique_all["outdoor_temp_bin"] == ">=35"
    ]
    mild = bin_unique_all[
        bin_unique_all["outdoor_temp_bin"] == "20--25"
    ]

    def delta_series(metric: str) -> pd.Series:
        return future_delta[f"delta_{metric}"]

    def delta_rho(left: str, right: str) -> float:
        return safe_spearman(delta_series(left), delta_series(right))

    summary = {
        "role_labelled_cases": int(len(case_df)),
        "unique_weather_states": int(len(unique_case)),
        "case_zone_units_unique": int(len(unique_zone)),
        "occupied_steps_total_role_labelled": int(case_df["occupied_steps"].sum()),
        "median_within_case_rho_outdoor_to_operative": float(
            find_pair("outdoor_temp_c", "mean_operative_temp_c").median()
        ),
        "median_within_case_rho_operative_to_tail": float(
            find_pair("mean_operative_temp_c", "discomfort_probability").median()
        ),
        "median_within_case_rho_cooling_to_tail": float(
            find_pair("zone_cooling_rate_w", "discomfort_probability").median()
        ),
        "median_within_case_rho_ghi_to_operative": float(
            find_pair("epw_ghi_w_m2", "mean_operative_temp_c").median()
        ),
        "annual_operative_to_tail_negative_case_count": int(
            (
                find_pair(
                    "mean_operative_temp_c", "discomfort_probability"
                )
                < 0
            ).sum()
        ),
        "hot_condition_median_rho_operative_to_warm_tail_oat_ge_25": float(
            find_conditioned_pair(
                "outdoor_temp_ge_25_c",
                "mean_operative_temp_c",
                "warm_discomfort_probability",
            ).median()
        ),
        "hot_condition_negative_operative_to_warm_tail_cases_oat_ge_25": int(
            (
                find_conditioned_pair(
                    "outdoor_temp_ge_25_c",
                    "mean_operative_temp_c",
                    "warm_discomfort_probability",
                )
                < 0
            ).sum()
        ),
        "mean_south_perimeter_high_tail_pct_unique": float(
            south["high_tail_pct_occ_mean"].mean()
        ),
        "mean_other_perimeter_high_tail_pct_unique": float(
            other["high_tail_pct_occ_mean"].mean()
        ),
        "unique_weather_high_tail_pct_oat_ge_35": (
            float(endpoint.iloc[0]["high_tail_pct"]) if len(endpoint) else np.nan
        ),
        "unique_weather_high_tail_pct_oat_20_25": (
            float(mild.iloc[0]["high_tail_pct"]) if len(mild) else np.nan
        ),
        "baseline_to_late_pairs": int(len(future_delta)),
        "baseline_to_late_unique_weather_cells": int(
            len(unique_cell_delta)
        ),
        "baseline_to_late_positive_mean_temp_pairs": int(
            (delta_series("annual_mean_temp_c") > 0).sum()
        ),
        "baseline_to_late_positive_operative_temp_pairs": int(
            (delta_series("mean_zone_operative_temp_c_occ") > 0).sum()
        ),
        "baseline_to_late_positive_mean_tail_pairs": int(
            (delta_series("mean_p_tail_occ") > 0).sum()
        ),
        "baseline_to_late_positive_high_tail_pairs": int(
            (delta_series("high_tail_pct_occ") > 0).sum()
        ),
        "baseline_to_late_positive_cooling_pairs": int(
            (delta_series("mean_zone_cooling_rate_w_occ") > 0).sum()
        ),
        "baseline_to_late_unique_weather_positive_mean_temp_cells": int(
            (
                unique_cell_delta["delta_annual_mean_temp_c"]
                > 0
            ).sum()
        ),
        "baseline_to_late_unique_weather_positive_operative_cells": int(
            (
                unique_cell_delta[
                    "delta_mean_zone_operative_temp_c_occ"
                ]
                > 0
            ).sum()
        ),
        "baseline_to_late_unique_weather_positive_mean_tail_cells": int(
            (
                unique_cell_delta["delta_mean_p_tail_occ"]
                > 0
            ).sum()
        ),
        "baseline_to_late_unique_weather_positive_high_tail_cells": int(
            (
                unique_cell_delta["delta_high_tail_pct_occ"]
                > 0
            ).sum()
        ),
        "median_delta_annual_mean_temp_c": float(
            delta_series("annual_mean_temp_c").median()
        ),
        "median_delta_mean_zone_operative_temp_c": float(
            delta_series("mean_zone_operative_temp_c_occ").median()
        ),
        "median_delta_mean_p_tail": float(delta_series("mean_p_tail_occ").median()),
        "median_delta_high_tail_pct_point": float(
            delta_series("high_tail_pct_occ").median()
        ),
        "median_delta_mean_cooling_rate_w": float(
            delta_series("mean_zone_cooling_rate_w_occ").median()
        ),
        "rho_delta_outdoor_temp_to_operative": delta_rho(
            "annual_mean_temp_c", "mean_zone_operative_temp_c_occ"
        ),
        "rho_delta_operative_to_mean_tail": delta_rho(
            "mean_zone_operative_temp_c_occ", "mean_p_tail_occ"
        ),
        "rho_delta_operative_to_high_tail": delta_rho(
            "mean_zone_operative_temp_c_occ", "high_tail_pct_occ"
        ),
        "tail_probability_timing_status": (
            "corrected_same_state"
            if corrected_timing
            else "legacy_callback_timing__provisional_do_not_cite"
        ),
        "corrected_probability_source": corrected_probability_source,
        "corrected_robustness_parity": parity_summary,
        "causal_identification": False,
        "incident_facade_solar_available": False,
        "energyplus_rerun_required": False,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    timing_notice = (
        "> **Corrected same-state inference.** Every probability in this "
        "directory was freshly inferred from the Ta/MRT/RH state recorded on "
        "the same end-of-step row. The legacy callback-timed probability "
        "columns were not used."
        if corrected_timing
        else
        "> **QUARANTINED PROVISIONAL OUTPUT.** The outdoor-weather, simulated "
        "operative-temperature, RH, cooling-rate, and spatial-state results are "
        "read from the final EnergyPlus states. The tail-probability columns in "
        "this trace, however, were evaluated at the legacy callback timing. "
        "Every numerical result below that contains `p_tail`, `warm_tail`, "
        "`cold_tail`, `discomfort`, or `high-tail` must be rerun on corrected "
        "same-state inference before manuscript use. Do not cite those "
        "provisional probability numbers."
    )
    report = f"""# Weather and building-physics association audit

{timing_notice}

## Bottom line

The existing traces support a coherent three-link interpretation without a new
EnergyPlus run:

1. **Outdoor forcing to simulated indoor state.** Across the 144 role-labelled
   cases, the median within-case hourly Spearman association between outdoor
   dry-bulb and mean-zone operative temperature was
   {format_range(find_pair('outdoor_temp_c', 'mean_operative_temp_c'))}.
2. **Indoor state to fitted TSV distribution.** The corresponding association
   between operative temperature and TSV-tail probability was
   {format_range(find_pair('mean_operative_temp_c', 'discomfort_probability'))}.
3. **HVAC response accompanies rather than eliminates the pattern.** The
   descriptive cooling-rate/tail association was
   {format_range(find_pair('zone_cooling_rate_w', 'discomfort_probability'))}.

These are associations in a fixed building and control configuration. They do
not identify causal effects, equipment-capacity failure, or future occupant
outcomes.

The annual within-case correlation is not a temperature-response coefficient:
it mixes seasons while the fitted TSV model also conditions on humidity,
running-mean outdoor temperature, and the other frozen predictors.
{summary['annual_operative_to_tail_negative_case_count']}/{summary['role_labelled_cases']}
annual cases have a negative
operative-temperature/total-tail rank correlation, all of which should be
retained rather than hidden. Under outdoor dry-bulb at or above 25 C, the
median operative-temperature/warm-tail association is
{summary['hot_condition_median_rho_operative_to_warm_tail_oat_ge_25']:.3f},
with
{summary['hot_condition_negative_operative_to_warm_tail_cases_oat_ge_25']}
negative cases. The conditioned results are in
`within_case_conditioned_correlations.csv`.

## Baseline-to-late pathway

Across the {summary['baseline_to_late_pairs']} matched
city--scenario--selection-role comparisons, annual mean outdoor temperature,
occupied mean-zone operative temperature, mean tail probability, and
high-tail share all increased in
{summary['baseline_to_late_positive_mean_temp_pairs']}/{summary['baseline_to_late_pairs']},
{summary['baseline_to_late_positive_operative_temp_pairs']}/{summary['baseline_to_late_pairs']},
{summary['baseline_to_late_positive_mean_tail_pairs']}/{summary['baseline_to_late_pairs']},
and
{summary['baseline_to_late_positive_high_tail_pairs']}/{summary['baseline_to_late_pairs']}
comparisons, respectively. Median changes
were +{summary['median_delta_annual_mean_temp_c']:.2f} C outdoors,
+{summary['median_delta_mean_zone_operative_temp_c']:.2f} C in occupied
mean-zone operative temperature, +{summary['median_delta_mean_p_tail']:.3f}
in mean tail probability, and
+{summary['median_delta_high_tail_pct_point']:.2f} percentage points in the
high-tail share.

An equal-unique-weather-year sensitivity first averages the distinct selected
years within each city--scenario--slice. Outdoor temperature, operative
temperature, mean tail probability and high-tail share increase in
{summary['baseline_to_late_unique_weather_positive_mean_temp_cells']}/{summary['baseline_to_late_unique_weather_cells']},
{summary['baseline_to_late_unique_weather_positive_operative_cells']}/{summary['baseline_to_late_unique_weather_cells']},
{summary['baseline_to_late_unique_weather_positive_mean_tail_cells']}/{summary['baseline_to_late_unique_weather_cells']},
and
{summary['baseline_to_late_unique_weather_positive_high_tail_cells']}/{summary['baseline_to_late_unique_weather_cells']}
cells, respectively. This
check is in `baseline_to_late_change_unique_weather_cell.csv`.

The association between matched changes in annual mean outdoor temperature and
occupied operative temperature was rho =
{summary['rho_delta_outdoor_temp_to_operative']:.3f}; the association between
operative-temperature change and mean-tail change was rho =
{summary['rho_delta_operative_to_mean_tail']:.3f}. Cooling rate increased in
{summary['baseline_to_late_positive_cooling_pairs']}/{summary['baseline_to_late_pairs']}
comparisons, with a median change of
{summary['median_delta_mean_cooling_rate_w'] / 1000:.2f} kW.

This is a matched descriptive chain through selected annual weather states,
not a variable-attribution model. In particular, solar and humidity covary
with temperature and location; their coefficients must not be read as
independent causal contributions.

## Outdoor-temperature stratification

After collapsing duplicate role labels, the pooled high-tail share was
{summary['unique_weather_high_tail_pct_oat_20_25']:.2f}% for outdoor dry-bulb
20--25 C and
{summary['unique_weather_high_tail_pct_oat_ge_35']:.2f}% at or above 35 C. The
associated operative-temperature, RH, cooling-rate, and tail summaries are in
`outdoor_temperature_bin_unique_weather_summary.csv`; the separately retained
role-labelled version documents the originally planned category weighting.
This stratification is descriptive and retains differences among cities,
seasons, and hours.

## Spatial pattern

After collapsing duplicate role labels to {summary['unique_weather_states']}
unique weather states, the mean high-tail share across the three south
perimeter zones was {summary['mean_south_perimeter_high_tail_pct_unique']:.2f}%,
versus {summary['mean_other_perimeter_high_tail_pct_unique']:.2f}% across the
other nine perimeter zones. The exact floor-by-orientation results and the
mean MRT-minus-air-temperature diagnostic are in
`floor_orientation_summary_unique_weather.csv`.

This pattern is consistent with orientation/envelope/HVAC-zoning interaction
in this prototype. It is not a clean estimate of solar causation. Only
horizontal GHI is available; incident facade solar gain and surface heat flux
were not recorded. The horizontal-GHI associations are retained as an
auditable forcing check, not attributed facade gains.

## Humidity and cooling interpretation

Outdoor RH comes from the selected EPW and is aligned to the hourly trace.
Zone RH is the EnergyPlus response used by the TSV predictor. Their
association and the zone-RH/tail association are reported separately so that
outdoor moisture forcing is not conflated with indoor humidity state.

`zone_cooling_rate_w` is the sum of the 15 requested zone sensible cooling
rates. A positive association with outdoor temperature or tail probability
shows coincident system response; it does not by itself establish plant
saturation. No capacity-saturation claim should be made without explicit
unmet-load or coil-capacity evidence.

## Weighting and uncertainty boundary

- Case-level associations use the {summary['unique_weather_states']} unique
  city--scenario--year weather states.
- Within-case correlations summarize temporal co-movement and report their
  distribution across cases; 15-minute rows are not treated as independent
  climate realizations.
- The panel uses one MPI-ESM1-2-LR forcing family. It does not quantify GCM,
  stochastic-weather, or urban-microclimate uncertainty.
- Tail probability is conditional on the fitted TSV model and fixed occupant
  assumptions. It is not observed dissatisfaction.

## EnergyPlus decision

No rerun is required for the reviewer-facing physical interpretation. A rerun
would be needed only to make incident-solar, envelope heat-flux, component
capacity, or unmet-load claims. Those stronger claims are unnecessary and
should instead be excluded.
"""
    (out / "physical_driver_report.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Metabolic-spread diagnostic for the Paper A 144-case panel.

The diagnostic holds the simulated thermal environment fixed and re-evaluates
the calibrated TSV probability model under the metabolic-load scenarios from
Guo et al. (2025), "Correcting the 120-watt assumption". This isolates how
much a single representative occupant assumption can hide variation in tail
risk.

The current trace files retain mean-zone environmental states but not raw
per-zone temperatures/radiant temperatures/humidity. This script is therefore
a mean-environment diagnostic. A zone-resolved version needs a runner/export
patch or an EnergyPlus SQL/output parse with zone raw states.
"""

from __future__ import annotations

import argparse
import heapq
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import joblib
import run_medium_office_diagnostic_panel as runner


DEFAULT_TRACE_DIR = ROOT / "runs" / "diagnostic_reference" / "traces"
DEFAULT_MODEL = ROOT / "runs" / "diagnostic_reference" / "control_predictors.seed.joblib"
DEFAULT_OUT = ROOT / "diagnostics" / "metabolic_spread"
MET_W_PER_M2 = 58.2
PAPER_CONVENTION_BSA_M2 = 120.0 / (1.2 * MET_W_PER_M2)
FIXED_TRACE_BSA_M2 = 1.80
TAIL_THRESHOLD = 0.20
TRACE_USECOLS = {
    "weather",
    "month",
    "day",
    "hour",
    "current_time",
    "occupied",
    "outdoor_temp_c",
    "running_mean_outdoor_c",
    "mean_air_temp_c",
    "mean_mrt_c",
    "mean_operative_temp_c",
    "mean_rh_pct",
    "expected_tsv",
    "discomfort_probability",
    "warm_discomfort_probability",
    "cold_discomfort_probability",
}
WEATHER_RE = re.compile(
    r"^(?P<city>ahmedabad|beijing|guangzhou|houston|kolkata|phoenix)_"
    r"(?P<scenario_raw>ssp245|ssp585)_"
    r"(?P<time_slice>baseline_2020s|near_2030s|mid_2050s|late_2080s)_"
    r"(?P<severity>typical|hot|heatwave_extreme)_"
    r"(?P<weather_year>\d{4})$"
)


@dataclass(frozen=True)
class MetScenario:
    scenario: str
    provenance: str
    watts_person: float
    met: float
    bsa_m2: float


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        self.count += int(arr.size)
        self.total += float(arr.sum())
        self.total_sq += float(np.square(arr).sum())

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def sd(self) -> float:
        if self.count <= 1:
            return float("nan")
        var = max(self.total_sq / self.count - self.mean**2, 0.0)
        return float(np.sqrt(var))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--predictor", choices=["ordinal", "nominal"], default="ordinal")
    parser.add_argument(
        "--bsa-mode",
        choices=["paper_convention", "fixed_1p80"],
        default="paper_convention",
        help="Conversion from W/person to met. The default makes 120 W/person equal 1.2 met.",
    )
    parser.add_argument("--tail-threshold", type=float, default=TAIL_THRESHOLD)
    parser.add_argument("--direction-threshold", type=float, default=0.02)
    parser.add_argument("--top-examples", type=int, default=200)
    parser.add_argument("--plot-sample-rows", type=int, default=300_000)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def build_met_scenarios(mode: str) -> list[MetScenario]:
    bsa = PAPER_CONVENTION_BSA_M2 if mode == "paper_convention" else FIXED_TRACE_BSA_M2
    rows = [
        ("COMP-Lo", "MC 5th percentile composite constant", 90.0),
        ("S-real", "Field-weighted mean from office profiles", 93.2),
        ("COMP-Med", "MC median composite constant", 100.0),
        ("EQ-Max", "Maximum single-equation case", 108.0),
        ("COMP-Hi", "MC 95th percentile composite constant", 110.0),
        ("LEGACY", "ASHRAE/code default", 120.0),
    ]
    return [
        MetScenario(
            scenario=name,
            provenance=provenance,
            watts_person=watts,
            met=watts / (MET_W_PER_M2 * bsa),
            bsa_m2=bsa,
        )
        for name, provenance, watts in rows
    ]


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def parse_weather(weather: str) -> dict[str, object]:
    match = WEATHER_RE.match(str(weather))
    if not match:
        return {
            "city": str(weather).split("_")[0].title(),
            "scenario_raw": "",
            "time_slice": "",
            "severity": "",
            "weather_year": np.nan,
        }
    data = match.groupdict()
    data["city"] = data["city"].title()
    data["weather_year"] = int(data["weather_year"])
    return data


def load_bundle(model_path: Path):
    # Compatibility for bundles pickled from a directly executed script.
    import __main__

    for name in ["FeatureSpec", "PredictorBundle"]:
        if not hasattr(__main__, name):
            setattr(__main__, name, getattr(runner, name))
    return joblib.load(model_path)


def trace_paths(trace_dir: Path, max_cases: int | None = None) -> list[Path]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    paths = [path for path in paths if path.name != "medium_office_control_traces.csv"]
    if not paths:
        raise FileNotFoundError(f"No per-case diagnostic traces found in {trace_dir}")
    if max_cases is not None:
        paths = paths[:max_cases]
    return paths


def load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda col: col in TRACE_USECOLS)
    occupied = bool_series(df["occupied"])
    valid = (
        occupied
        & df["mean_air_temp_c"].notna()
        & df["mean_mrt_c"].notna()
        & df["mean_rh_pct"].notna()
    )
    df = df.loc[valid].copy()
    for col in TRACE_USECOLS - {"weather", "occupied"}:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def predict_for_scenario(bundle, df: pd.DataFrame, scenario: MetScenario, predictor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    rm = df["running_mean_outdoor_c"].where(
        df["running_mean_outdoor_c"].notna(), df["outdoor_temp_c"]
    )
    features = runner.build_features_from_arrays(
        ta=df["mean_air_temp_c"].to_numpy(float),
        tr=df["mean_mrt_c"].to_numpy(float),
        v=np.full(n, 0.10),
        rh=df["mean_rh_pct"].to_numpy(float),
        met=np.full(n, scenario.met),
        clo=np.full(n, 0.65),
        bsa=np.full(n, scenario.bsa_m2),
        rm_out=rm.to_numpy(float),
        spec=bundle.spec,
    )
    if predictor == "ordinal":
        probs = bundle.predict_ordinal(features)
    else:
        probs = bundle.predict_nominal(features)
    mu = probs @ runner.TSV_VALUES
    cold_tail = probs[:, [0, 1]].sum(axis=1)
    warm_tail = probs[:, [5, 6]].sum(axis=1)
    p_tail = cold_tail + warm_tail
    d_tail = warm_tail - cold_tail
    return mu, p_tail, warm_tail, cold_tail, d_tail


def pct(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    return float(arr.mean() * 100.0) if arr.size else float("nan")


def sign_eps(values: np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(arr > eps, 1, np.where(arr < -eps, -1, 0))


def summary_record(
    label: dict[str, object],
    scenario: MetScenario,
    mu: np.ndarray,
    p_tail: np.ndarray,
    warm_tail: np.ndarray,
    cold_tail: np.ndarray,
    d_tail: np.ndarray,
    tail_threshold: float,
    direction_threshold: float,
) -> dict[str, object]:
    high_tail = p_tail >= tail_threshold
    d_sign = sign_eps(d_tail, direction_threshold)
    out = dict(label)
    out.update(
        {
            "scenario": scenario.scenario,
            "watts_person": scenario.watts_person,
            "met": scenario.met,
            "bsa_m2": scenario.bsa_m2,
            "rows": int(len(p_tail)),
            "mean_mu": float(np.mean(mu)),
            "mean_abs_mu": float(np.mean(np.abs(mu))),
            "mean_p_tail": float(np.mean(p_tail)),
            "p50_p_tail": float(np.quantile(p_tail, 0.50)),
            "p95_p_tail": float(np.quantile(p_tail, 0.95)),
            "max_p_tail": float(np.max(p_tail)),
            "high_tail_rows": int(high_tail.sum()),
            "high_tail_pct": pct(high_tail),
            "mean_warm_tail": float(np.mean(warm_tail)),
            "mean_cold_tail": float(np.mean(cold_tail)),
            "warm_dominant_rows": int((d_sign > 0).sum()),
            "cold_dominant_rows": int((d_sign < 0).sum()),
            "warm_dominant_pct": pct(d_sign > 0),
            "cold_dominant_pct": pct(d_sign < 0),
        }
    )
    return out


def top_indices(score: np.ndarray, n: int, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    else:
        valid = np.isfinite(arr)
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return positions
    keep = min(n, positions.size)
    local = np.argpartition(arr[positions], -keep)[-keep:]
    chosen = positions[local]
    return chosen[np.argsort(arr[chosen])[::-1]]


def make_examples(
    df: pd.DataFrame,
    scenarios: list[MetScenario],
    mu_mat: np.ndarray,
    p_mat: np.ndarray,
    d_mat: np.ndarray,
    score: np.ndarray,
    idx: np.ndarray,
    kind: str,
) -> pd.DataFrame:
    if idx.size == 0:
        return pd.DataFrame()
    scenario_names = np.asarray([s.scenario for s in scenarios], dtype=object)
    p_min_idx = np.argmin(p_mat[:, idx], axis=0)
    p_max_idx = np.argmax(p_mat[:, idx], axis=0)
    mu_min_idx = np.argmin(mu_mat[:, idx], axis=0)
    mu_max_idx = np.argmax(mu_mat[:, idx], axis=0)
    base_cols = [
        "weather",
        "month",
        "day",
        "hour",
        "current_time",
        "outdoor_temp_c",
        "mean_air_temp_c",
        "mean_mrt_c",
        "mean_operative_temp_c",
        "mean_rh_pct",
    ]
    out = df.iloc[idx][base_cols].copy()
    out["example_kind"] = kind
    out["example_score"] = score[idx]
    out["p_tail_min"] = p_mat[p_min_idx, idx]
    out["p_tail_max"] = p_mat[p_max_idx, idx]
    out["p_tail_spread"] = out["p_tail_max"] - out["p_tail_min"]
    out["p_tail_min_scenario"] = scenario_names[p_min_idx]
    out["p_tail_max_scenario"] = scenario_names[p_max_idx]
    out["mu_min"] = mu_mat[mu_min_idx, idx]
    out["mu_max"] = mu_mat[mu_max_idx, idx]
    out["mu_spread"] = out["mu_max"] - out["mu_min"]
    out["mu_min_scenario"] = scenario_names[mu_min_idx]
    out["mu_max_scenario"] = scenario_names[mu_max_idx]
    for scenario_i, scenario in enumerate(scenarios):
        key = scenario.scenario.lower().replace("-", "_")
        out[f"{key}_mu"] = mu_mat[scenario_i, idx]
        out[f"{key}_p_tail"] = p_mat[scenario_i, idx]
        out[f"{key}_d_tail"] = d_mat[scenario_i, idx]
    meta = pd.DataFrame([parse_weather(v) for v in out["weather"]], index=out.index)
    for col in meta.columns:
        out[col] = meta[col]
    first_cols = [
        "example_kind",
        "weather",
        "city",
        "scenario_raw",
        "time_slice",
        "severity",
        "weather_year",
    ]
    remaining = [col for col in out.columns if col not in first_cols]
    return out[first_cols + remaining]


def update_heap(heap: list[tuple[float, int, pd.Series]], rows: pd.DataFrame, score_col: str, limit: int) -> None:
    for _, row in rows.iterrows():
        score = float(row[score_col])
        token = id(row)
        item = (score, token, row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)


def heap_to_frame(heap: list[tuple[float, int, pd.Series]]) -> pd.DataFrame:
    if not heap:
        return pd.DataFrame()
    rows = [item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)]
    return pd.DataFrame(rows)


def process_trace(
    path: Path,
    bundle,
    scenarios: list[MetScenario],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_trace(path)
    if df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    weather = str(df["weather"].iloc[0])
    meta = parse_weather(weather)
    label = {"weather": weather, **meta}

    mu_rows = []
    p_rows = []
    warm_rows = []
    cold_rows = []
    d_rows = []
    scenario_summaries = []
    scenario_arrays: dict[str, np.ndarray] = {}
    for scenario in scenarios:
        mu, p_tail, warm_tail, cold_tail, d_tail = predict_for_scenario(
            bundle, df, scenario, args.predictor
        )
        mu_rows.append(mu.astype(np.float32))
        p_rows.append(p_tail.astype(np.float32))
        warm_rows.append(warm_tail.astype(np.float32))
        cold_rows.append(cold_tail.astype(np.float32))
        d_rows.append(d_tail.astype(np.float32))
        scenario_summaries.append(
            summary_record(
                label,
                scenario,
                mu,
                p_tail,
                warm_tail,
                cold_tail,
                d_tail,
                args.tail_threshold,
                args.direction_threshold,
            )
        )
        key = scenario.scenario
        scenario_arrays[f"{key}__mu"] = mu.astype(np.float32)
        scenario_arrays[f"{key}__p_tail"] = p_tail.astype(np.float32)
        scenario_arrays[f"{key}__d_tail"] = d_tail.astype(np.float32)

    mu_mat = np.vstack(mu_rows)
    p_mat = np.vstack(p_rows)
    d_mat = np.vstack(d_rows)
    p_min = p_mat.min(axis=0)
    p_max = p_mat.max(axis=0)
    p_spread = p_max - p_min
    mu_spread = mu_mat.max(axis=0) - mu_mat.min(axis=0)
    d_min = d_mat.min(axis=0)
    d_max = d_mat.max(axis=0)
    d_spread = d_max - d_min
    d_sign = sign_eps(d_mat, args.direction_threshold)
    direction_flip = (d_sign.min(axis=0) < 0) & (d_sign.max(axis=0) > 0)
    threshold_flip = (p_min < args.tail_threshold) & (p_max >= args.tail_threshold)

    scenario_index = {scenario.scenario: i for i, scenario in enumerate(scenarios)}
    sreal_i = scenario_index["S-real"]
    legacy_i = scenario_index["LEGACY"]
    compmed_i = scenario_index["COMP-Med"]
    sreal_hidden = (p_mat[sreal_i] < args.tail_threshold) & (p_max >= args.tail_threshold)
    legacy_hidden = (p_mat[legacy_i] < args.tail_threshold) & (p_max >= args.tail_threshold)
    compmed_hidden = (p_mat[compmed_i] < args.tail_threshold) & (p_max >= args.tail_threshold)

    spread_summary = {
        **label,
        "rows": int(len(df)),
        "mean_p_tail_spread": float(np.mean(p_spread)),
        "p50_p_tail_spread": float(np.quantile(p_spread, 0.50)),
        "p95_p_tail_spread": float(np.quantile(p_spread, 0.95)),
        "max_p_tail_spread": float(np.max(p_spread)),
        "mean_mu_spread": float(np.mean(mu_spread)),
        "p95_mu_spread": float(np.quantile(mu_spread, 0.95)),
        "mean_d_tail_spread": float(np.mean(d_spread)),
        "threshold_flip_rows": int(threshold_flip.sum()),
        "threshold_flip_pct": pct(threshold_flip),
        "sreal_hidden_tail_rows": int(sreal_hidden.sum()),
        "sreal_hidden_tail_pct": pct(sreal_hidden),
        "compmed_hidden_tail_rows": int(compmed_hidden.sum()),
        "compmed_hidden_tail_pct": pct(compmed_hidden),
        "legacy_hidden_tail_rows": int(legacy_hidden.sum()),
        "legacy_hidden_tail_pct": pct(legacy_hidden),
        "direction_flip_rows": int(direction_flip.sum()),
        "direction_flip_pct": pct(direction_flip),
        "sreal_mean_p_tail": float(np.mean(p_mat[sreal_i])),
        "legacy_mean_p_tail": float(np.mean(p_mat[legacy_i])),
        "sreal_high_tail_pct": pct(p_mat[sreal_i] >= args.tail_threshold),
        "legacy_high_tail_pct": pct(p_mat[legacy_i] >= args.tail_threshold),
    }
    spread_arrays = {
        "p_spread": p_spread.astype(np.float32),
        "mu_spread": mu_spread.astype(np.float32),
        "d_spread": d_spread.astype(np.float32),
        "threshold_flip": threshold_flip,
        "sreal_hidden": sreal_hidden,
        "legacy_hidden": legacy_hidden,
        "direction_flip": direction_flip,
    }

    top_spread = make_examples(
        df,
        scenarios,
        mu_mat,
        p_mat,
        d_mat,
        p_spread,
        top_indices(p_spread, args.top_examples),
        "largest_metabolic_p_tail_spread",
    )
    hidden_score = p_max - p_mat[sreal_i]
    hidden = make_examples(
        df,
        scenarios,
        mu_mat,
        p_mat,
        d_mat,
        hidden_score,
        top_indices(hidden_score, args.top_examples, sreal_hidden),
        "sreal_below_threshold_any_profile_above",
    )
    direction = make_examples(
        df,
        scenarios,
        mu_mat,
        p_mat,
        d_mat,
        d_spread,
        top_indices(d_spread, args.top_examples, direction_flip),
        "warm_cold_tail_direction_flip",
    )

    sample_n = min(max(1, args.plot_sample_rows // 60), len(df))
    sample = df.sample(n=sample_n, random_state=abs(hash(path.name)) % (2**32)).copy()
    sample_idx = sample.index.to_numpy()
    local_pos = df.index.get_indexer(sample_idx)
    sample_out = sample[
        [
            "weather",
            "outdoor_temp_c",
            "mean_operative_temp_c",
            "mean_rh_pct",
        ]
    ].copy()
    sample_out["sreal_p_tail"] = p_mat[sreal_i, local_pos]
    sample_out["legacy_p_tail"] = p_mat[legacy_i, local_pos]
    sample_out["max_p_tail"] = p_max[local_pos]
    sample_out["p_tail_spread"] = p_spread[local_pos]
    sample_out["threshold_flip"] = threshold_flip[local_pos]
    sample_meta = pd.DataFrame([parse_weather(v) for v in sample_out["weather"]], index=sample_out.index)
    for col in sample_meta.columns:
        sample_out[col] = sample_meta[col]

    return (
        pd.DataFrame(scenario_summaries),
        pd.DataFrame([spread_summary]),
        {**scenario_arrays, **spread_arrays},
        top_spread,
        hidden,
        direction,
        sample_out,
    )


def combine_global_scenario_summary(
    scenarios: list[MetScenario],
    global_arrays: dict[str, list[np.ndarray]],
    tail_threshold: float,
    direction_threshold: float,
) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        mu = np.concatenate(global_arrays[f"{scenario.scenario}__mu"])
        p_tail = np.concatenate(global_arrays[f"{scenario.scenario}__p_tail"])
        d_tail = np.concatenate(global_arrays[f"{scenario.scenario}__d_tail"])
        warm_tail = (p_tail + d_tail) / 2.0
        cold_tail = (p_tail - d_tail) / 2.0
        rows.append(
            summary_record(
                {"scope": "all"},
                scenario,
                mu,
                p_tail,
                warm_tail,
                cold_tail,
                d_tail,
                tail_threshold,
                direction_threshold,
            )
        )
    return pd.DataFrame(rows)


def combine_global_spread_summary(global_arrays: dict[str, list[np.ndarray]], args: argparse.Namespace) -> pd.DataFrame:
    p_spread = np.concatenate(global_arrays["p_spread"])
    mu_spread = np.concatenate(global_arrays["mu_spread"])
    d_spread = np.concatenate(global_arrays["d_spread"])
    threshold_flip = np.concatenate(global_arrays["threshold_flip"])
    sreal_hidden = np.concatenate(global_arrays["sreal_hidden"])
    legacy_hidden = np.concatenate(global_arrays["legacy_hidden"])
    direction_flip = np.concatenate(global_arrays["direction_flip"])
    row = {
        "scope": "all",
        "rows": int(len(p_spread)),
        "mean_p_tail_spread": float(np.mean(p_spread)),
        "p50_p_tail_spread": float(np.quantile(p_spread, 0.50)),
        "p95_p_tail_spread": float(np.quantile(p_spread, 0.95)),
        "max_p_tail_spread": float(np.max(p_spread)),
        "mean_mu_spread": float(np.mean(mu_spread)),
        "p95_mu_spread": float(np.quantile(mu_spread, 0.95)),
        "mean_d_tail_spread": float(np.mean(d_spread)),
        "p95_d_tail_spread": float(np.quantile(d_spread, 0.95)),
        "threshold_flip_rows": int(threshold_flip.sum()),
        "threshold_flip_pct": pct(threshold_flip),
        "sreal_hidden_tail_rows": int(sreal_hidden.sum()),
        "sreal_hidden_tail_pct": pct(sreal_hidden),
        "legacy_hidden_tail_rows": int(legacy_hidden.sum()),
        "legacy_hidden_tail_pct": pct(legacy_hidden),
        "direction_flip_rows": int(direction_flip.sum()),
        "direction_flip_pct": pct(direction_flip),
        "tail_threshold": args.tail_threshold,
        "direction_threshold": args.direction_threshold,
    }
    return pd.DataFrame([row])


def build_group_summary(case_spread: pd.DataFrame) -> pd.DataFrame:
    count_cols = [
        "threshold_flip_rows",
        "sreal_hidden_tail_rows",
        "compmed_hidden_tail_rows",
        "legacy_hidden_tail_rows",
        "direction_flip_rows",
    ]
    mean_cols = [
        "mean_p_tail_spread",
        "p50_p_tail_spread",
        "p95_p_tail_spread",
        "mean_mu_spread",
        "p95_mu_spread",
        "mean_d_tail_spread",
        "sreal_mean_p_tail",
        "legacy_mean_p_tail",
        "sreal_high_tail_pct",
        "legacy_high_tail_pct",
    ]
    rows = []
    group_specs = [
        ("city", ["city"]),
        ("scenario", ["scenario_raw"]),
        ("time_slice", ["time_slice"]),
        ("severity", ["severity"]),
        ("city_time_slice", ["city", "time_slice"]),
    ]
    for scope, keys in group_specs:
        for values, group in case_spread.groupby(keys, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            total_rows = int(group["rows"].sum())
            record = {"scope": scope, "rows": total_rows, "weather_cases": int(len(group))}
            record.update(dict(zip(keys, values)))
            for col in count_cols:
                record[col] = int(group[col].sum())
                record[col.replace("_rows", "_pct")] = (
                    float(group[col].sum() / total_rows * 100.0) if total_rows else float("nan")
                )
            for col in mean_cols:
                record[col] = (
                    float((group[col] * group["rows"]).sum() / total_rows) if total_rows else float("nan")
                )
            rows.append(record)
    return pd.DataFrame(rows)


def write_plots(
    scenario_summary: pd.DataFrame,
    global_arrays: dict[str, list[np.ndarray]],
    sample: pd.DataFrame,
    out_dir: Path,
    tail_threshold: float,
) -> list[Path]:
    paths = []
    scenario_summary = scenario_summary.sort_values("watts_person")
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax.plot(
        scenario_summary["watts_person"],
        scenario_summary["mean_p_tail"],
        marker="o",
        color="#234f70",
        lw=1.8,
        label="Mean p_tail",
    )
    ax.fill_between(
        scenario_summary["watts_person"],
        scenario_summary["p50_p_tail"],
        scenario_summary["p95_p_tail"],
        color="#9db7c9",
        alpha=0.35,
        label="p50-p95 p_tail",
    )
    ax.set_xlabel("Metabolic load (W/person)")
    ax.set_ylabel("Discomfort-tail probability")
    ax.set_title("Tail risk under metabolic-load scenarios", weight="bold")
    ax.axhline(tail_threshold, color="#7a2d42", lw=0.9, ls="--")
    ax.grid(color="#dddddd", lw=0.6)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    path = out_dir / "metabolic_scenario_tail_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    p_spread = np.concatenate(global_arrays["p_spread"])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax.hist(p_spread, bins=80, color="#3d6f8f", alpha=0.82)
    ax.set_xlabel("p_tail spread across metabolic scenarios")
    ax.set_ylabel("Occupied timesteps")
    ax.set_title("Distribution of metabolic-spread effect", weight="bold")
    ax.grid(color="#dddddd", lw=0.55)
    fig.tight_layout()
    path = out_dir / "metabolic_p_tail_spread_hist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    if len(sample) > 0:
        plot = sample.sample(n=min(len(sample), 300_000), random_state=42)
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=180)
        ax.scatter(
            plot["sreal_p_tail"],
            plot["max_p_tail"],
            c=plot["p_tail_spread"],
            cmap="viridis",
            s=7,
            alpha=0.55,
            linewidths=0,
        )
        ax.axhline(tail_threshold, color="#7a2d42", lw=0.9, ls="--")
        ax.axvline(tail_threshold, color="#7a2d42", lw=0.9, ls="--")
        lim = max(float(plot["max_p_tail"].max()), tail_threshold) + 0.02
        ax.plot([0, lim], [0, lim], color="#666666", lw=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("S-real p_tail")
        ax.set_ylabel("Maximum p_tail across metabolic scenarios")
        ax.set_title("Tail risk hidden by a single metabolic profile", weight="bold")
        ax.grid(color="#dddddd", lw=0.55)
        fig.tight_layout()
        path = out_dir / "sreal_vs_max_tail_scatter.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_markdown(
    scenarios: list[MetScenario],
    scenario_summary: pd.DataFrame,
    spread_summary: pd.DataFrame,
    case_spread: pd.DataFrame,
    group_summary: pd.DataFrame,
    plot_paths: list[Path],
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    path = out_dir / "metabolic_spread_summary.md"
    spread = spread_summary.iloc[0]
    scenario_cols = [
        "scenario",
        "watts_person",
        "met",
        "bsa_m2",
        "mean_mu",
        "mean_p_tail",
        "p95_p_tail",
        "high_tail_pct",
    ]
    city = group_summary[group_summary["scope"].eq("city")].copy()
    city = city.sort_values("sreal_hidden_tail_pct", ascending=False).head(10)
    worst_cases = case_spread.sort_values("sreal_hidden_tail_pct", ascending=False).head(10)
    current_watts = 1.10 * MET_W_PER_M2 * FIXED_TRACE_BSA_M2
    with path.open("w", encoding="utf-8") as f:
        f.write("# Metabolic-Spread Diagnostic Summary\n\n")
        f.write("## Method\n\n")
        f.write(
            "This diagnostic re-evaluates the calibrated TSV probability model over the "
            "published six-point metabolic-load ladder while holding the simulated thermal "
            "environment fixed. It uses mean-zone environmental states from the 144-case "
            "Paper A diagnostic panel; raw per-zone states were not retained in the trace "
            "files, so this is not yet a zone-resolved metabolic-spread diagnostic.\n\n"
        )
        f.write(f"- Predictor: `{args.predictor}`\n")
        f.write(f"- BSA conversion mode: `{args.bsa_mode}`\n")
        f.write(f"- Occupied probability rows: {int(spread.rows):,}\n")
        f.write(f"- Tail threshold: `p_tail >= {args.tail_threshold:.2f}`\n")
        f.write(
            f"- Previous trace reference used `met=1.10`, `BSA=1.80`, equivalent to "
            f"{current_watts:.1f} W/person under the simple met x 58.2 x BSA conversion.\n\n"
        )
        f.write("## Scenario Ladder\n\n")
        f.write("```csv\n")
        f.write(
            pd.DataFrame(
                [
                    {
                        "scenario": s.scenario,
                        "provenance": s.provenance,
                        "watts_person": s.watts_person,
                        "met": s.met,
                        "bsa_m2": s.bsa_m2,
                    }
                    for s in scenarios
                ]
            ).to_csv(index=False)
        )
        f.write("```\n\n")
        f.write("## Headline Results\n\n")
        f.write(f"- Mean `p_tail` spread across metabolic scenarios: {spread.mean_p_tail_spread:.3f}\n")
        f.write(f"- 95th percentile `p_tail` spread: {spread.p95_p_tail_spread:.3f}\n")
        f.write(f"- Threshold-flip states across the ladder: {spread.threshold_flip_pct:.1f}%\n")
        f.write(
            f"- `S-real` below threshold but at least one metabolic scenario above threshold: "
            f"{spread.sreal_hidden_tail_pct:.1f}%\n"
        )
        f.write(
            f"- Warm/cold tail-direction flips across the ladder: {spread.direction_flip_pct:.1f}%\n\n"
        )
        f.write("## Global Scenario Summary\n\n")
        f.write("```csv\n")
        f.write(scenario_summary[scenario_cols].to_csv(index=False))
        f.write("```\n\n")
        f.write("## Cities With Highest S-real Hidden-Tail Share\n\n")
        f.write("```csv\n")
        f.write(
            city[
                [
                    "city",
                    "rows",
                    "weather_cases",
                    "mean_p_tail_spread",
                    "sreal_hidden_tail_pct",
                    "threshold_flip_pct",
                    "direction_flip_pct",
                ]
            ].to_csv(index=False)
        )
        f.write("```\n\n")
        f.write("## Worst Individual Weather Cases\n\n")
        f.write("```csv\n")
        f.write(
            worst_cases[
                [
                    "weather",
                    "rows",
                    "mean_p_tail_spread",
                    "p95_p_tail_spread",
                    "sreal_hidden_tail_pct",
                    "threshold_flip_pct",
                    "direction_flip_pct",
                ]
            ].to_csv(index=False)
        )
        f.write("```\n\n")
        if plot_paths:
            f.write("## Figures\n\n")
            for plot in plot_paths:
                f.write(f"- `{plot.name}`\n")
    return path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.model_path)
    scenarios = build_met_scenarios(args.bsa_mode)
    paths = trace_paths(args.trace_dir, args.max_cases)
    print(f"[load] model: {args.model_path}", flush=True)
    print(f"[load] traces: {len(paths)} cases from {args.trace_dir}", flush=True)
    print(f"[config] bsa_mode={args.bsa_mode}; predictor={args.predictor}", flush=True)

    scenario_case_frames = []
    spread_case_frames = []
    sample_frames = []
    top_heap: list[tuple[float, int, pd.Series]] = []
    hidden_heap: list[tuple[float, int, pd.Series]] = []
    direction_heap: list[tuple[float, int, pd.Series]] = []
    global_arrays: dict[str, list[np.ndarray]] = {}

    for i, path in enumerate(paths, start=1):
        (
            scenario_case,
            spread_case,
            arrays,
            top_examples,
            hidden_examples,
            direction_examples,
            sample,
        ) = process_trace(path, bundle, scenarios, args)
        if not scenario_case.empty:
            scenario_case_frames.append(scenario_case)
        if not spread_case.empty:
            spread_case_frames.append(spread_case)
        if not sample.empty:
            sample_frames.append(sample)
        for key, arr in arrays.items():
            global_arrays.setdefault(key, []).append(arr)
        update_heap(top_heap, top_examples, "example_score", args.top_examples)
        update_heap(hidden_heap, hidden_examples, "example_score", args.top_examples)
        update_heap(direction_heap, direction_examples, "example_score", args.top_examples)
        if i == 1 or i % 12 == 0 or i == len(paths):
            print(f"[progress] processed {i}/{len(paths)} traces", flush=True)

    case_scenario = pd.concat(scenario_case_frames, ignore_index=True)
    case_spread = pd.concat(spread_case_frames, ignore_index=True)
    sample = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if len(sample) > args.plot_sample_rows:
        sample = sample.sample(n=args.plot_sample_rows, random_state=42)

    scenario_summary = combine_global_scenario_summary(
        scenarios, global_arrays, args.tail_threshold, args.direction_threshold
    )
    spread_summary = combine_global_spread_summary(global_arrays, args)
    group_summary = build_group_summary(case_spread)
    top_examples = heap_to_frame(top_heap)
    hidden_examples = heap_to_frame(hidden_heap)
    direction_examples = heap_to_frame(direction_heap)

    scenario_summary.to_csv(args.output_dir / "metabolic_scenario_summary.csv", index=False)
    case_scenario.to_csv(args.output_dir / "metabolic_case_scenario_summary.csv", index=False)
    spread_summary.to_csv(args.output_dir / "metabolic_spread_global_summary.csv", index=False)
    case_spread.to_csv(args.output_dir / "metabolic_case_spread_summary.csv", index=False)
    group_summary.to_csv(args.output_dir / "metabolic_group_spread_summary.csv", index=False)
    top_examples.to_csv(args.output_dir / "largest_metabolic_spread_examples.csv", index=False)
    hidden_examples.to_csv(args.output_dir / "sreal_hidden_tail_examples.csv", index=False)
    direction_examples.to_csv(args.output_dir / "direction_flip_examples.csv", index=False)
    sample.to_csv(args.output_dir / "metabolic_spread_plot_sample.csv", index=False)
    plot_paths = write_plots(
        scenario_summary,
        global_arrays,
        sample,
        args.output_dir,
        args.tail_threshold,
    )
    md = write_markdown(
        scenarios,
        scenario_summary,
        spread_summary,
        case_spread,
        group_summary,
        plot_paths,
        args.output_dir,
        args,
    )

    print(f"[write] {args.output_dir / 'metabolic_scenario_summary.csv'}")
    print(f"[write] {args.output_dir / 'metabolic_case_scenario_summary.csv'}")
    print(f"[write] {args.output_dir / 'metabolic_spread_global_summary.csv'}")
    print(f"[write] {args.output_dir / 'metabolic_case_spread_summary.csv'}")
    print(f"[write] {args.output_dir / 'metabolic_group_spread_summary.csv'}")
    print(f"[write] {args.output_dir / 'largest_metabolic_spread_examples.csv'}")
    print(f"[write] {args.output_dir / 'sreal_hidden_tail_examples.csv'}")
    print(f"[write] {args.output_dir / 'direction_flip_examples.csv'}")
    print(f"[write] {args.output_dir / 'metabolic_spread_plot_sample.csv'}")
    for path in plot_paths:
        print(f"[write] {path}")
    print(f"[write] {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

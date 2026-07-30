#!/usr/bin/env python3
"""Zone-resolved metabolic-spread diagnostic for the future-weather panel.

This is the full version of the metabolic-spread diagnostic. It requires
traces that include raw zone environmental fields written by
`legacy_control_pipeline.py`:

    zone_<slug>_ta_c, zone_<slug>_tr_c, zone_<slug>_rh_pct

For each occupied timestep, each zone is re-evaluated across the published
120-W correction metabolic ladder. The output reports both the mean-zone
aggregation used by the original probability interface and the max/any-zone
tail exposure that is more appropriate for spatial hidden-risk diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import os
import sys
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

import analyze_metabolic_spread as mean_met
import legacy_control_pipeline as runner


DEFAULT_TRACE_DIR = ROOT / "restricted_inputs" / "simulation_traces"
DEFAULT_MODEL = ROOT / "models" / "tsv_predictor_bundle.joblib"
DEFAULT_OUT = ROOT / "outputs" / "metabolic_spread"
BASE_USECOLS = {
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
}


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
    )
    parser.add_argument("--tail-threshold", type=float, default=0.20)
    parser.add_argument("--direction-threshold", type=float, default=0.02)
    parser.add_argument("--top-examples", type=int, default=200)
    parser.add_argument("--plot-sample-rows", type=int, default=300_000)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def zone_raw_columns() -> list[str]:
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


def trace_paths(trace_dir: Path, max_cases: int | None) -> list[Path]:
    paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
    paths = [path for path in paths if path.name not in runner.COMBINED_TRACE_NAMES]
    if not paths:
        raise FileNotFoundError(f"No zone-raw traces found in {trace_dir}")
    return paths[:max_cases] if max_cases is not None else paths


def load_trace(path: Path) -> pd.DataFrame:
    raw_cols = set(zone_raw_columns())
    usecols = BASE_USECOLS | raw_cols
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    missing = sorted(raw_cols - set(df.columns))
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"{path} is missing zone raw columns: {preview}")
    occupied = mean_met.bool_series(df["occupied"])
    valid = occupied.copy()
    for col in raw_cols:
        valid &= df[col].notna()
    df = df.loc[valid].copy()
    for col in usecols - {"weather", "occupied"}:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def flatten_zone_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ta_cols = [f"zone_{slug}_ta_c" for slug in runner.ZONE_FIELD_NAMES]
    tr_cols = [f"zone_{slug}_tr_c" for slug in runner.ZONE_FIELD_NAMES]
    rh_cols = [f"zone_{slug}_rh_pct" for slug in runner.ZONE_FIELD_NAMES]
    ta = df[ta_cols].to_numpy(float)
    tr = df[tr_cols].to_numpy(float)
    rh = df[rh_cols].to_numpy(float)
    return ta, tr, rh


def predict_zone_arrays(
    bundle,
    df: pd.DataFrame,
    scenario: mean_met.MetScenario,
    predictor: str,
    tail_threshold: float,
) -> dict[str, np.ndarray]:
    ta, tr, rh = flatten_zone_inputs(df)
    n, z = ta.shape
    rm = df["running_mean_outdoor_c"].where(
        df["running_mean_outdoor_c"].notna(), df["outdoor_temp_c"]
    ).to_numpy(float)
    features = runner.build_features_from_arrays(
        ta=ta.reshape(-1),
        tr=tr.reshape(-1),
        v=np.full(n * z, 0.10),
        rh=rh.reshape(-1),
        met=np.full(n * z, scenario.met),
        clo=np.full(n * z, 0.65),
        bsa=np.full(n * z, scenario.bsa_m2),
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
    zone_p = zone_cold + zone_warm
    zone_d = zone_warm - zone_cold
    return {
        "mean_mu": zone_mu.mean(axis=1).astype(np.float32),
        "mean_p_tail": zone_p.mean(axis=1).astype(np.float32),
        "mean_d_tail": zone_d.mean(axis=1).astype(np.float32),
        "max_zone_p_tail": zone_p.max(axis=1).astype(np.float32),
        "p90_zone_p_tail": np.quantile(zone_p, 0.90, axis=1).astype(np.float32),
        "max_abs_zone_d_tail": np.max(np.abs(zone_d), axis=1).astype(np.float32),
        "zones_high_tail": (zone_p >= tail_threshold).sum(axis=1).astype(np.int16),
    }


def stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def scenario_summary(
    label: dict[str, object],
    scenario: mean_met.MetScenario,
    arrays: dict[str, np.ndarray],
    tail: float,
    direction: float,
) -> dict[str, object]:
    mean_p = arrays["mean_p_tail"]
    max_p = arrays["max_zone_p_tail"]
    mean_d = arrays["mean_d_tail"]
    d_sign = mean_met.sign_eps(mean_d, direction)
    return {
        **label,
        "scenario": scenario.scenario,
        "watts_person": scenario.watts_person,
        "met": scenario.met,
        "bsa_m2": scenario.bsa_m2,
        "rows": int(len(mean_p)),
        "mean_mu": float(np.mean(arrays["mean_mu"])),
        "mean_p_tail_mean_agg": float(np.mean(mean_p)),
        "p95_p_tail_mean_agg": float(np.quantile(mean_p, 0.95)),
        "high_tail_mean_agg_pct": mean_met.pct(mean_p >= tail),
        "mean_max_zone_p_tail": float(np.mean(max_p)),
        "p95_max_zone_p_tail": float(np.quantile(max_p, 0.95)),
        "any_zone_high_tail_pct": mean_met.pct(max_p >= tail),
        "mean_p90_zone_p_tail": float(np.mean(arrays["p90_zone_p_tail"])),
        "mean_zones_high_tail": float(np.mean(arrays["zones_high_tail"])),
        "warm_dominant_pct": mean_met.pct(d_sign > 0),
        "cold_dominant_pct": mean_met.pct(d_sign < 0),
    }


def top_indices(score: np.ndarray, n: int, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    pos = np.flatnonzero(valid)
    if pos.size == 0:
        return pos
    keep = min(n, pos.size)
    local = np.argpartition(arr[pos], -keep)[-keep:]
    chosen = pos[local]
    return chosen[np.argsort(arr[chosen])[::-1]]


def make_examples(
    df: pd.DataFrame,
    scenarios: list[mean_met.MetScenario],
    mean_p_mat: np.ndarray,
    max_p_mat: np.ndarray,
    score: np.ndarray,
    idx: np.ndarray,
    kind: str,
) -> pd.DataFrame:
    if idx.size == 0:
        return pd.DataFrame()
    scenario_names = np.asarray([s.scenario for s in scenarios], dtype=object)
    mean_min_i = np.argmin(mean_p_mat[:, idx], axis=0)
    mean_max_i = np.argmax(mean_p_mat[:, idx], axis=0)
    zone_min_i = np.argmin(max_p_mat[:, idx], axis=0)
    zone_max_i = np.argmax(max_p_mat[:, idx], axis=0)
    cols = [
        "weather",
        "month",
        "day",
        "hour",
        "current_time",
        "outdoor_temp_c",
        "mean_operative_temp_c",
        "mean_rh_pct",
    ]
    out = df.iloc[idx][cols].copy()
    out["example_kind"] = kind
    out["example_score"] = score[idx]
    out["mean_p_tail_min"] = mean_p_mat[mean_min_i, idx]
    out["mean_p_tail_max"] = mean_p_mat[mean_max_i, idx]
    out["mean_p_tail_spread"] = out["mean_p_tail_max"] - out["mean_p_tail_min"]
    out["mean_min_scenario"] = scenario_names[mean_min_i]
    out["mean_max_scenario"] = scenario_names[mean_max_i]
    out["max_zone_p_tail_min"] = max_p_mat[zone_min_i, idx]
    out["max_zone_p_tail_max"] = max_p_mat[zone_max_i, idx]
    out["max_zone_p_tail_spread"] = out["max_zone_p_tail_max"] - out["max_zone_p_tail_min"]
    out["zone_min_scenario"] = scenario_names[zone_min_i]
    out["zone_max_scenario"] = scenario_names[zone_max_i]
    meta = pd.DataFrame([mean_met.parse_weather(v) for v in out["weather"]], index=out.index)
    for col in meta.columns:
        out[col] = meta[col]
    first = ["example_kind", "weather", "city", "scenario_raw", "time_slice", "severity", "weather_year"]
    return out[first + [c for c in out.columns if c not in first]]


def update_heap(heap: list[tuple[float, int, pd.Series]], rows: pd.DataFrame, score_col: str, limit: int) -> None:
    for _, row in rows.iterrows():
        score = float(row[score_col])
        item = (score, id(row), row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)


def heap_to_frame(heap: list[tuple[float, int, pd.Series]]) -> pd.DataFrame:
    return pd.DataFrame([x[2] for x in sorted(heap, key=lambda x: x[0], reverse=True)])


def process_trace(path: Path, bundle, scenarios: list[mean_met.MetScenario], args: argparse.Namespace):
    df = load_trace(path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()
    weather = str(df["weather"].iloc[0])
    label = {"weather": weather, **mean_met.parse_weather(weather)}
    case_scenarios = []
    scenario_arrays: dict[str, dict[str, np.ndarray]] = {}
    for scenario in scenarios:
        arrays = predict_zone_arrays(
            bundle, df, scenario, args.predictor, args.tail_threshold
        )
        scenario_arrays[scenario.scenario] = arrays
        case_scenarios.append(
            scenario_summary(label, scenario, arrays, args.tail_threshold, args.direction_threshold)
        )

    mean_p_mat = np.vstack([scenario_arrays[s.scenario]["mean_p_tail"] for s in scenarios])
    max_p_mat = np.vstack([scenario_arrays[s.scenario]["max_zone_p_tail"] for s in scenarios])
    mean_d_mat = np.vstack([scenario_arrays[s.scenario]["mean_d_tail"] for s in scenarios])
    scenario_index = {s.scenario: i for i, s in enumerate(scenarios)}
    sreal_i = scenario_index["S-real"]

    mean_spread = mean_p_mat.max(axis=0) - mean_p_mat.min(axis=0)
    max_spread = max_p_mat.max(axis=0) - max_p_mat.min(axis=0)
    mean_flip = (mean_p_mat.min(axis=0) < args.tail_threshold) & (
        mean_p_mat.max(axis=0) >= args.tail_threshold
    )
    zone_flip = (max_p_mat.min(axis=0) < args.tail_threshold) & (
        max_p_mat.max(axis=0) >= args.tail_threshold
    )
    sreal_mean_hidden = (mean_p_mat[sreal_i] < args.tail_threshold) & (
        mean_p_mat.max(axis=0) >= args.tail_threshold
    )
    sreal_zone_hidden = (max_p_mat[sreal_i] < args.tail_threshold) & (
        max_p_mat.max(axis=0) >= args.tail_threshold
    )
    d_sign = mean_met.sign_eps(mean_d_mat, args.direction_threshold)
    direction_flip = (d_sign.min(axis=0) < 0) & (d_sign.max(axis=0) > 0)
    spread = {
        **label,
        "rows": int(len(df)),
        "mean_agg_p_tail_spread_mean": float(np.mean(mean_spread)),
        "mean_agg_p_tail_spread_p95": float(np.quantile(mean_spread, 0.95)),
        "max_zone_p_tail_spread_mean": float(np.mean(max_spread)),
        "max_zone_p_tail_spread_p95": float(np.quantile(max_spread, 0.95)),
        "mean_agg_threshold_flip_rows": int(mean_flip.sum()),
        "mean_agg_threshold_flip_pct": mean_met.pct(mean_flip),
        "max_zone_threshold_flip_rows": int(zone_flip.sum()),
        "max_zone_threshold_flip_pct": mean_met.pct(zone_flip),
        "sreal_mean_hidden_rows": int(sreal_mean_hidden.sum()),
        "sreal_mean_hidden_pct": mean_met.pct(sreal_mean_hidden),
        "sreal_zone_hidden_rows": int(sreal_zone_hidden.sum()),
        "sreal_zone_hidden_pct": mean_met.pct(sreal_zone_hidden),
        "direction_flip_rows": int(direction_flip.sum()),
        "direction_flip_pct": mean_met.pct(direction_flip),
        "sreal_mean_high_tail_pct": mean_met.pct(mean_p_mat[sreal_i] >= args.tail_threshold),
        "sreal_any_zone_high_tail_pct": mean_met.pct(max_p_mat[sreal_i] >= args.tail_threshold),
    }

    top_zone = make_examples(
        df,
        scenarios,
        mean_p_mat,
        max_p_mat,
        max_spread,
        top_indices(max_spread, args.top_examples),
        "largest_zone_resolved_metabolic_spread",
    )
    hidden_score = max_p_mat.max(axis=0) - max_p_mat[sreal_i]
    hidden = make_examples(
        df,
        scenarios,
        mean_p_mat,
        max_p_mat,
        hidden_score,
        top_indices(hidden_score, args.top_examples, sreal_zone_hidden),
        "sreal_any_zone_hidden_tail",
    )
    global_arrays: dict[str, np.ndarray] = {
        "mean_spread": mean_spread.astype(np.float32),
        "max_spread": max_spread.astype(np.float32),
        "mean_flip": mean_flip,
        "zone_flip": zone_flip,
        "sreal_mean_hidden": sreal_mean_hidden,
        "sreal_zone_hidden": sreal_zone_hidden,
        "direction_flip": direction_flip,
    }
    for scenario in scenarios:
        name = scenario.scenario
        global_arrays[f"{name}__mean_p_tail"] = scenario_arrays[name]["mean_p_tail"]
        global_arrays[f"{name}__max_zone_p_tail"] = scenario_arrays[name]["max_zone_p_tail"]
        global_arrays[f"{name}__mean_mu"] = scenario_arrays[name]["mean_mu"]

    sample_n = min(max(1, args.plot_sample_rows // 60), len(df))
    sample = df.sample(n=sample_n, random_state=stable_seed(path.name)).copy()
    pos = df.index.get_indexer(sample.index)
    sample_out = sample[["weather", "outdoor_temp_c", "mean_operative_temp_c", "mean_rh_pct"]].copy()
    sample_out["sreal_mean_p_tail"] = mean_p_mat[sreal_i, pos]
    sample_out["sreal_max_zone_p_tail"] = max_p_mat[sreal_i, pos]
    sample_out["max_profile_max_zone_p_tail"] = max_p_mat.max(axis=0)[pos]
    sample_out["max_zone_p_tail_spread"] = max_spread[pos]
    meta = pd.DataFrame([mean_met.parse_weather(v) for v in sample_out["weather"]], index=sample_out.index)
    for col in meta.columns:
        sample_out[col] = meta[col]

    return pd.DataFrame(case_scenarios), pd.DataFrame([spread]), global_arrays, top_zone, hidden, sample_out


def global_scenario_summary(scenarios: list[mean_met.MetScenario], arrays: dict[str, list[np.ndarray]], tail: float) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        mean_p = np.concatenate(arrays[f"{scenario.scenario}__mean_p_tail"])
        max_p = np.concatenate(arrays[f"{scenario.scenario}__max_zone_p_tail"])
        mean_mu = np.concatenate(arrays[f"{scenario.scenario}__mean_mu"])
        rows.append(
            {
                "scope": "all",
                "scenario": scenario.scenario,
                "watts_person": scenario.watts_person,
                "met": scenario.met,
                "bsa_m2": scenario.bsa_m2,
                "rows": int(len(mean_p)),
                "mean_mu": float(np.mean(mean_mu)),
                "mean_p_tail_mean_agg": float(np.mean(mean_p)),
                "p95_p_tail_mean_agg": float(np.quantile(mean_p, 0.95)),
                "high_tail_mean_agg_pct": mean_met.pct(mean_p >= tail),
                "mean_max_zone_p_tail": float(np.mean(max_p)),
                "p95_max_zone_p_tail": float(np.quantile(max_p, 0.95)),
                "any_zone_high_tail_pct": mean_met.pct(max_p >= tail),
            }
        )
    return pd.DataFrame(rows)


def global_spread_summary(arrays: dict[str, list[np.ndarray]], args: argparse.Namespace) -> pd.DataFrame:
    mean_spread = np.concatenate(arrays["mean_spread"])
    max_spread = np.concatenate(arrays["max_spread"])
    mean_flip = np.concatenate(arrays["mean_flip"])
    zone_flip = np.concatenate(arrays["zone_flip"])
    sreal_mean_hidden = np.concatenate(arrays["sreal_mean_hidden"])
    sreal_zone_hidden = np.concatenate(arrays["sreal_zone_hidden"])
    direction_flip = np.concatenate(arrays["direction_flip"])
    return pd.DataFrame(
        [
            {
                "scope": "all",
                "rows": int(len(mean_spread)),
                "mean_agg_p_tail_spread_mean": float(np.mean(mean_spread)),
                "mean_agg_p_tail_spread_p95": float(np.quantile(mean_spread, 0.95)),
                "max_zone_p_tail_spread_mean": float(np.mean(max_spread)),
                "max_zone_p_tail_spread_p95": float(np.quantile(max_spread, 0.95)),
                "mean_agg_threshold_flip_rows": int(mean_flip.sum()),
                "mean_agg_threshold_flip_pct": mean_met.pct(mean_flip),
                "max_zone_threshold_flip_rows": int(zone_flip.sum()),
                "max_zone_threshold_flip_pct": mean_met.pct(zone_flip),
                "sreal_mean_hidden_rows": int(sreal_mean_hidden.sum()),
                "sreal_mean_hidden_pct": mean_met.pct(sreal_mean_hidden),
                "sreal_zone_hidden_rows": int(sreal_zone_hidden.sum()),
                "sreal_zone_hidden_pct": mean_met.pct(sreal_zone_hidden),
                "direction_flip_rows": int(direction_flip.sum()),
                "direction_flip_pct": mean_met.pct(direction_flip),
                "tail_threshold": args.tail_threshold,
                "direction_threshold": args.direction_threshold,
            }
        ]
    )


def group_spread_summary(case_spread: pd.DataFrame) -> pd.DataFrame:
    count_cols = [
        "mean_agg_threshold_flip_rows",
        "max_zone_threshold_flip_rows",
        "sreal_mean_hidden_rows",
        "sreal_zone_hidden_rows",
        "direction_flip_rows",
    ]
    mean_cols = [
        "mean_agg_p_tail_spread_mean",
        "mean_agg_p_tail_spread_p95",
        "max_zone_p_tail_spread_mean",
        "max_zone_p_tail_spread_p95",
        "sreal_mean_high_tail_pct",
        "sreal_any_zone_high_tail_pct",
    ]
    rows = []
    specs = [
        ("city", ["city"]),
        ("time_slice", ["time_slice"]),
        ("severity", ["severity"]),
        ("city_time_slice", ["city", "time_slice"]),
    ]
    for scope, keys in specs:
        for values, group in case_spread.groupby(keys, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            total = int(group["rows"].sum())
            rec = {"scope": scope, "rows": total, "weather_cases": int(len(group))}
            rec.update(dict(zip(keys, values)))
            for col in count_cols:
                rec[col] = int(group[col].sum())
                rec[col.replace("_rows", "_pct")] = (
                    float(group[col].sum() / total * 100.0) if total else float("nan")
                )
            for col in mean_cols:
                rec[col] = (
                    float((group[col] * group["rows"]).sum() / total) if total else float("nan")
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def write_plots(scenario_summary: pd.DataFrame, arrays: dict[str, list[np.ndarray]], sample: pd.DataFrame, out_dir: Path, tail: float) -> list[Path]:
    paths = []
    plot = scenario_summary.sort_values("watts_person")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    ax.plot(plot["watts_person"], plot["mean_p_tail_mean_agg"], marker="o", lw=1.8, label="Mean-zone aggregation", color="#234f70")
    ax.plot(plot["watts_person"], plot["mean_max_zone_p_tail"], marker="s", lw=1.8, label="Max-zone aggregation", color="#8f4b2f")
    ax.axhline(tail, color="#7a2d42", lw=0.9, ls="--")
    ax.set_xlabel("Metabolic load (W/person)")
    ax.set_ylabel("Discomfort-tail probability")
    ax.set_title("Zone-resolved tail risk across metabolic profiles", weight="bold")
    ax.grid(color="#dddddd", lw=0.55)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    path = out_dir / "zone_metabolic_scenario_tail_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    mean_spread = np.concatenate(arrays["mean_spread"])
    max_spread = np.concatenate(arrays["max_spread"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    bins = np.linspace(0, max(float(max_spread.max()), float(mean_spread.max())), 80)
    ax.hist(mean_spread, bins=bins, alpha=0.65, label="Mean-zone spread", color="#4f6f9f")
    ax.hist(max_spread, bins=bins, alpha=0.55, label="Max-zone spread", color="#b66d47")
    ax.set_xlabel("p_tail spread across metabolic scenarios")
    ax.set_ylabel("Occupied timesteps")
    ax.set_title("Metabolic-spread effect by aggregation rule", weight="bold")
    ax.grid(color="#dddddd", lw=0.55)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    path = out_dir / "zone_metabolic_spread_hist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    if len(sample) > 0:
        plot_sample = sample.sample(n=min(len(sample), 300_000), random_state=42)
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=180)
        ax.scatter(
            plot_sample["sreal_max_zone_p_tail"],
            plot_sample["max_profile_max_zone_p_tail"],
            c=plot_sample["max_zone_p_tail_spread"],
            cmap="viridis",
            s=7,
            alpha=0.55,
            linewidths=0,
        )
        ax.axhline(tail, color="#7a2d42", lw=0.9, ls="--")
        ax.axvline(tail, color="#7a2d42", lw=0.9, ls="--")
        lim = max(float(plot_sample["max_profile_max_zone_p_tail"].max()), tail) + 0.02
        ax.plot([0, lim], [0, lim], color="#666666", lw=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("S-real max-zone p_tail")
        ax.set_ylabel("Maximum max-zone p_tail across profiles")
        ax.set_title("Any-zone tail risk hidden by one metabolic profile", weight="bold")
        ax.grid(color="#dddddd", lw=0.55)
        fig.tight_layout()
        path = out_dir / "zone_sreal_vs_max_tail_scatter.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_markdown(
    scenarios: list[mean_met.MetScenario],
    scenario_summary: pd.DataFrame,
    spread_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    case_spread: pd.DataFrame,
    plot_paths: list[Path],
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    path = out_dir / "zone_metabolic_spread_summary.md"
    spread = spread_summary.iloc[0]
    city = group_summary[group_summary["scope"].eq("city")].sort_values(
        "sreal_zone_hidden_pct", ascending=False
    )
    worst = case_spread.sort_values("sreal_zone_hidden_pct", ascending=False).head(10)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Zone-Resolved Metabolic-Spread Diagnostic Summary\n\n")
        f.write("This diagnostic recomputes TSV probabilities for each zone and each metabolic profile, then reports both mean-zone and max-zone tail-risk aggregation.\n\n")
        f.write(f"- Predictor: `{args.predictor}`\n")
        f.write(f"- BSA conversion mode: `{args.bsa_mode}`\n")
        f.write(f"- Occupied timesteps: {int(spread.rows):,}\n")
        f.write(f"- Zones per timestep: {len(runner.ZONE_FIELD_NAMES)}\n")
        f.write(f"- Zone-profile probability evaluations: {int(spread.rows) * len(runner.ZONE_FIELD_NAMES) * len(scenarios):,}\n")
        f.write(f"- Tail threshold: `p_tail >= {args.tail_threshold:.2f}`\n\n")
        f.write("## Headline Results\n\n")
        f.write(f"- Mean-zone `p_tail` spread, mean: {spread.mean_agg_p_tail_spread_mean:.3f}; p95: {spread.mean_agg_p_tail_spread_p95:.3f}\n")
        f.write(f"- Max-zone `p_tail` spread, mean: {spread.max_zone_p_tail_spread_mean:.3f}; p95: {spread.max_zone_p_tail_spread_p95:.3f}\n")
        f.write(f"- Mean-zone threshold flips: {spread.mean_agg_threshold_flip_pct:.1f}%\n")
        f.write(f"- Max-zone threshold flips: {spread.max_zone_threshold_flip_pct:.1f}%\n")
        f.write(f"- `S-real` mean-zone hidden tail: {spread.sreal_mean_hidden_pct:.1f}%\n")
        f.write(f"- `S-real` any-zone hidden tail: {spread.sreal_zone_hidden_pct:.1f}%\n")
        f.write(f"- Warm/cold mean-tail direction flips: {spread.direction_flip_pct:.1f}%\n\n")
        f.write("## Scenario Summary\n\n")
        f.write("```csv\n")
        f.write(
            scenario_summary[
                [
                    "scenario",
                    "watts_person",
                    "met",
                    "mean_p_tail_mean_agg",
                    "high_tail_mean_agg_pct",
                    "mean_max_zone_p_tail",
                    "any_zone_high_tail_pct",
                ]
            ].to_csv(index=False)
        )
        f.write("```\n\n")
        f.write("## City Summary\n\n")
        f.write("```csv\n")
        f.write(
            city[
                [
                    "city",
                    "rows",
                    "weather_cases",
                    "mean_agg_p_tail_spread_mean",
                    "max_zone_p_tail_spread_mean",
                    "sreal_mean_hidden_pct",
                    "sreal_zone_hidden_pct",
                    "max_zone_threshold_flip_pct",
                ]
            ].to_csv(index=False)
        )
        f.write("```\n\n")
        f.write("## Worst Weather Cases\n\n")
        f.write("```csv\n")
        f.write(
            worst[
                [
                    "weather",
                    "rows",
                    "mean_agg_p_tail_spread_mean",
                    "max_zone_p_tail_spread_mean",
                    "sreal_mean_hidden_pct",
                    "sreal_zone_hidden_pct",
                    "max_zone_threshold_flip_pct",
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
    bundle = mean_met.load_bundle(args.model_path)
    scenarios = mean_met.build_met_scenarios(args.bsa_mode)
    paths = trace_paths(args.trace_dir, args.max_cases)
    print(f"[load] model: {args.model_path}", flush=True)
    print(f"[load] traces: {len(paths)} cases from {args.trace_dir}", flush=True)

    scenario_case_frames = []
    spread_case_frames = []
    sample_frames = []
    top_heap: list[tuple[float, int, pd.Series]] = []
    hidden_heap: list[tuple[float, int, pd.Series]] = []
    global_arrays: dict[str, list[np.ndarray]] = {}

    for i, path in enumerate(paths, start=1):
        case_scenario, case_spread, arrays, top_zone, hidden, sample = process_trace(
            path, bundle, scenarios, args
        )
        if not case_scenario.empty:
            scenario_case_frames.append(case_scenario)
        if not case_spread.empty:
            spread_case_frames.append(case_spread)
        if not sample.empty:
            sample_frames.append(sample)
        for key, arr in arrays.items():
            global_arrays.setdefault(key, []).append(arr)
        update_heap(top_heap, top_zone, "example_score", args.top_examples)
        update_heap(hidden_heap, hidden, "example_score", args.top_examples)
        if i == 1 or i % 12 == 0 or i == len(paths):
            print(f"[progress] processed {i}/{len(paths)} traces", flush=True)

    case_scenario = pd.concat(scenario_case_frames, ignore_index=True)
    case_spread = pd.concat(spread_case_frames, ignore_index=True)
    sample = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if len(sample) > args.plot_sample_rows:
        sample = sample.sample(n=args.plot_sample_rows, random_state=42)
    scenario_summary = global_scenario_summary(scenarios, global_arrays, args.tail_threshold)
    spread_summary = global_spread_summary(global_arrays, args)
    group_summary = group_spread_summary(case_spread)
    top_examples = heap_to_frame(top_heap)
    hidden_examples = heap_to_frame(hidden_heap)

    scenario_summary.to_csv(args.output_dir / "zone_metabolic_scenario_summary.csv", index=False)
    case_scenario.to_csv(args.output_dir / "zone_metabolic_case_scenario_summary.csv", index=False)
    spread_summary.to_csv(args.output_dir / "zone_metabolic_spread_global_summary.csv", index=False)
    case_spread.to_csv(args.output_dir / "zone_metabolic_case_spread_summary.csv", index=False)
    group_summary.to_csv(args.output_dir / "zone_metabolic_group_spread_summary.csv", index=False)
    top_examples.to_csv(args.output_dir / "largest_zone_metabolic_spread_examples.csv", index=False)
    hidden_examples.to_csv(args.output_dir / "sreal_any_zone_hidden_tail_examples.csv", index=False)
    sample.to_csv(args.output_dir / "zone_metabolic_spread_plot_sample.csv", index=False)
    plot_paths = write_plots(scenario_summary, global_arrays, sample, args.output_dir, args.tail_threshold)
    md = write_markdown(
        scenarios,
        scenario_summary,
        spread_summary,
        group_summary,
        case_spread,
        plot_paths,
        args.output_dir,
        args,
    )

    for out in [
        "zone_metabolic_scenario_summary.csv",
        "zone_metabolic_case_scenario_summary.csv",
        "zone_metabolic_spread_global_summary.csv",
        "zone_metabolic_case_spread_summary.csv",
        "zone_metabolic_group_spread_summary.csv",
        "largest_zone_metabolic_spread_examples.csv",
        "sreal_any_zone_hidden_tail_examples.csv",
        "zone_metabolic_spread_plot_sample.csv",
    ]:
        print(f"[write] {args.output_dir / out}")
    for plot in plot_paths:
        print(f"[write] {plot}")
    print(f"[write] {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

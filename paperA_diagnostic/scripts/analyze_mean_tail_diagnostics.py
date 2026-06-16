#!/usr/bin/env python
"""Paper A diagnostic for expected sensation versus discomfort-tail risk.

The diagnostic tests whether expected TSV is an insufficient summary of the
ordinal probability distribution under future-weather stress. It uses the
diagnostic-reference traces and does not evaluate a control policy.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MPL_CACHE = ROOT / ".mplconfig"
LOCAL_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TRACE = (
    ROOT
    / "runs"
    / "diagnostic_reference"
    / "traces"
    / "medium_office_control_traces.csv"
)
DEFAULT_TRACE_DIR = ROOT / "runs" / "diagnostic_reference" / "traces"
DEFAULT_MANIFEST = ROOT / "data" / "panel_manifest.csv"
DEFAULT_OUT = ROOT / "diagnostics" / "mean_tail"
DEFAULT_STRATEGY = "diagnostic_reference"

REQUIRED_COLUMNS = {
    "strategy",
    "weather",
    "month",
    "day",
    "current_time",
    "occupied",
    "expected_tsv",
    "discomfort_probability",
    "warm_discomfort_probability",
    "cold_discomfort_probability",
}

OPTIONAL_COLUMNS = {
    "hour",
    "mean_pmv",
    "mean_operative_temp_c",
    "outdoor_temp_c",
    "grid_event",
}

WEATHER_RE = re.compile(
    r"^(?P<city>ahmedabad|beijing|guangzhou|houston|kolkata|phoenix)_"
    r"(?P<scenario_raw>ssp245|ssp585)_"
    r"(?P<time_slice>baseline_2020s|near_2030s|mid_2050s|late_2080s)_"
    r"(?P<severity>typical|hot|heatwave_extreme)_"
    r"(?P<weather_year>\d{4})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strategies", nargs="+", default=[DEFAULT_STRATEGY])
    parser.add_argument("--eps", type=float, default=0.15, help="Neutral mean threshold.")
    parser.add_argument("--tail-threshold", type=float, default=0.20)
    parser.add_argument("--direction-threshold", type=float, default=0.10)
    parser.add_argument("--mu-window", type=float, default=0.03)
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument(
        "--skip-matched-pairs",
        action="store_true",
        help="Skip the exhaustive matched-mean pair count; useful for full 144-case runs.",
    )
    parser.add_argument(
        "--plot-sample-rows",
        type=int,
        default=300_000,
        help="Maximum occupied probability rows used for scatter-style plots.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_trace(path: Path, trace_dir: Path | None = None) -> pd.DataFrame:
    wanted = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
    if path.exists():
        df = pd.read_csv(path, usecols=lambda col: col in wanted)
    elif trace_dir is not None and trace_dir.exists():
        paths = sorted(trace_dir.glob("*_diagnostic_reference.csv"))
        if not paths:
            raise FileNotFoundError(f"No diagnostic traces found in {trace_dir}")
        frames = [pd.read_csv(p, usecols=lambda col: col in wanted) for p in paths]
        df = pd.concat(frames, ignore_index=True)
    else:
        raise FileNotFoundError(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in [
        "expected_tsv",
        "discomfort_probability",
        "warm_discomfort_probability",
        "cold_discomfort_probability",
        "mean_pmv",
        "mean_operative_temp_c",
        "outdoor_temp_c",
        "grid_event",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["occupied"].dtype != bool:
        df["occupied"] = df["occupied"].astype(str).str.lower().isin({"true", "1", "yes"})
    return df


def availability_report(df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    rows = []
    for strategy, sdf in df.groupby("strategy", sort=True):
        if strategy not in strategies:
            continue
        prob_mask = probability_mask(sdf)
        rows.append(
            {
                "strategy": strategy,
                "rows": int(len(sdf)),
                "occupied_rows": int(sdf["occupied"].sum()),
                "probability_rows": int(prob_mask.sum()),
                "occupied_probability_rows": int((prob_mask & sdf["occupied"]).sum()),
                "weather_cases": int(sdf["weather"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def probability_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["expected_tsv"].notna()
        & df["warm_discomfort_probability"].notna()
        & df["cold_discomfort_probability"].notna()
    )


def prepare_diagnostic_rows(df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    work = df[df["strategy"].isin(strategies)].copy()
    work = work[work["occupied"] & probability_mask(work)].copy()
    work["p_tail_from_sides"] = (
        work["warm_discomfort_probability"] + work["cold_discomfort_probability"]
    )
    work["p_tail"] = work["p_tail_from_sides"]
    work["d_tail"] = work["warm_discomfort_probability"] - work["cold_discomfort_probability"]
    work["abs_mu"] = work["expected_tsv"].abs()
    work["abs_d_tail"] = work["d_tail"].abs()
    work["tail_consistency_abs_error"] = (
        work["p_tail_from_sides"] - work["discomfort_probability"]
    ).abs()
    work["case_label"] = work["weather"].map(case_label)
    return work


def threshold_sign(values: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(
        np.where(values > threshold, 1, np.where(values < -threshold, -1, 0)),
        index=values.index,
    )


def summarize_group(
    name: tuple[str, str],
    sdf: pd.DataFrame,
    *,
    eps: float,
    tail_threshold: float,
    direction_threshold: float,
) -> dict[str, float | str | int]:
    strategy, weather = name
    near_mean = sdf["abs_mu"] < eps
    high_tail = sdf["p_tail"] > tail_threshold
    directional_tail = sdf["abs_d_tail"] > direction_threshold
    mu_sign = threshold_sign(sdf["expected_tsv"], eps)
    tail_sign = threshold_sign(sdf["d_tail"], direction_threshold)
    sign_comparable = (mu_sign != 0) & (tail_sign != 0)
    sign_conflict = sign_comparable & (mu_sign != tail_sign)
    n = len(sdf)
    comparable_n = int(sign_comparable.sum())

    return {
        "strategy": strategy,
        "weather": weather,
        "case_label": "All cases" if weather == "ALL" else case_label(weather),
        "rows": int(n),
        "near_mean_pct": pct(near_mean.sum(), n),
        "high_tail_pct": pct(high_tail.sum(), n),
        "near_mean_high_tail_pct": pct((near_mean & high_tail).sum(), n),
        "near_mean_directional_tail_pct": pct((near_mean & directional_tail).sum(), n),
        "sign_conflict_pct_of_comparable": pct(sign_conflict.sum(), comparable_n),
        "sign_comparable_rows": comparable_n,
        "mean_abs_mu": float(sdf["abs_mu"].mean()),
        "mean_p_tail": float(sdf["p_tail"].mean()),
        "p95_p_tail": float(sdf["p_tail"].quantile(0.95)),
        "mean_abs_d_tail": float(sdf["abs_d_tail"].mean()),
        "warm_tail_dominant_pct": pct((sdf["d_tail"] > direction_threshold).sum(), n),
        "cold_tail_dominant_pct": pct((sdf["d_tail"] < -direction_threshold).sum(), n),
        "tail_consistency_mae": float(sdf["tail_consistency_abs_error"].mean()),
    }


def pct(count: int | float, denom: int | float) -> float:
    if denom <= 0:
        return np.nan
    return float(count) / float(denom) * 100.0


def build_summary(
    rows: pd.DataFrame,
    *,
    eps: float,
    tail_threshold: float,
    direction_threshold: float,
) -> pd.DataFrame:
    summaries = []
    for strategy, sdf in rows.groupby("strategy", sort=True):
        summaries.append(
            summarize_group(
                (strategy, "ALL"),
                sdf,
                eps=eps,
                tail_threshold=tail_threshold,
                direction_threshold=direction_threshold,
            )
        )
    for (strategy, weather), sdf in rows.groupby(["strategy", "weather"], sort=True):
        summaries.append(
            summarize_group(
                (strategy, weather),
                sdf,
                eps=eps,
                tail_threshold=tail_threshold,
                direction_threshold=direction_threshold,
            )
        )
    return pd.DataFrame(summaries)


def build_threshold_grid(rows: pd.DataFrame, strategy: str = DEFAULT_STRATEGY) -> pd.DataFrame:
    sdf = rows[rows["strategy"] == strategy].copy()
    records = []
    for eps in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for tail_threshold in [0.12, 0.15, 0.18, 0.20, 0.25]:
            for direction_threshold in [0.04, 0.06, 0.08, 0.10]:
                near_mean = sdf["expected_tsv"].abs() < eps
                high_tail = sdf["p_tail"] > tail_threshold
                directional_tail = sdf["d_tail"].abs() > direction_threshold
                mu_sign = threshold_sign(sdf["expected_tsv"], eps)
                tail_sign = threshold_sign(sdf["d_tail"], direction_threshold)
                comparable = (mu_sign != 0) & (tail_sign != 0)
                conflict = comparable & (mu_sign != tail_sign)
                records.append(
                    {
                        "strategy": strategy,
                        "eps": eps,
                        "tail_threshold": tail_threshold,
                        "direction_threshold": direction_threshold,
                        "near_mean_high_tail_pct": pct((near_mean & high_tail).sum(), len(sdf)),
                        "near_mean_directional_tail_pct": pct(
                            (near_mean & directional_tail).sum(), len(sdf)
                        ),
                        "sign_conflict_pct_of_comparable": pct(conflict.sum(), comparable.sum()),
                        "sign_comparable_rows": int(comparable.sum()),
                    }
                )
    return pd.DataFrame(records)


def find_same_mean_tail_pairs(
    rows: pd.DataFrame,
    *,
    strategy: str = DEFAULT_STRATEGY,
    mu_window: float,
    top_examples: int,
) -> pd.DataFrame:
    sdf = rows[rows["strategy"] == strategy].copy().reset_index(drop=True)
    if sdf.empty or top_examples <= 0:
        return pd.DataFrame()
    bin_width = max(mu_window / 2.0, 1e-6)
    sdf["_mu_bin"] = np.floor(sdf["expected_tsv"] / bin_width).astype(int)
    pairs = []
    for _, group in sdf.groupby("_mu_bin", sort=False):
        if len(group) < 2:
            continue
        add_extreme_pairs(sdf, group, mu_window, pairs)

    # Adjacent bins can contain the best same-mean examples near bin boundaries.
    for bin_id in sorted(sdf["_mu_bin"].unique()):
        group = sdf[sdf["_mu_bin"].isin([bin_id, bin_id + 1])]
        if len(group) >= 2:
            add_extreme_pairs(sdf, group, mu_window, pairs)
    if not pairs:
        return pd.DataFrame()
    ranked = (
        pd.DataFrame(pairs)
        .drop_duplicates(subset=["i", "j"])
        .sort_values("score", ascending=False)
        .head(top_examples)
    )
    out_rows = []
    keep_cols = [
        "strategy",
        "weather",
        "case_label",
        "month",
        "day",
        "current_time",
        "expected_tsv",
        "p_tail",
        "d_tail",
        "warm_discomfort_probability",
        "cold_discomfort_probability",
        "mean_pmv",
        "mean_operative_temp_c",
        "outdoor_temp_c",
        "grid_event",
    ]
    keep_cols = [c for c in keep_cols if c in sdf.columns]
    for pair_id, pair in enumerate(ranked.itertuples(index=False), start=1):
        for member, idx in [("A", int(pair.i)), ("B", int(pair.j))]:
            rec = sdf.loc[idx, keep_cols].to_dict()
            rec.update(
                {
                    "pair_id": pair_id,
                    "member": member,
                    "pair_mu_gap": float(pair.mu_gap),
                    "pair_tail_gap": float(pair.tail_gap),
                    "pair_direction_gap": float(pair.direction_gap),
                    "pair_score": float(pair.score),
                }
            )
            out_rows.append(rec)
    return pd.DataFrame(out_rows)


def add_extreme_pairs(
    sdf: pd.DataFrame, group: pd.DataFrame, mu_window: float, pairs: list[dict[str, float]]
) -> None:
    candidate_indices = {
        int(group["p_tail"].idxmin()),
        int(group["p_tail"].idxmax()),
        int(group["d_tail"].idxmin()),
        int(group["d_tail"].idxmax()),
    }
    candidate_indices = sorted(candidate_indices)
    for pos, i in enumerate(candidate_indices):
        for j in candidate_indices[pos + 1 :]:
            mu_gap = abs(float(sdf.loc[j, "expected_tsv"] - sdf.loc[i, "expected_tsv"]))
            if mu_gap > mu_window:
                continue
            tail_gap = abs(float(sdf.loc[j, "p_tail"] - sdf.loc[i, "p_tail"]))
            direction_gap = abs(float(sdf.loc[j, "d_tail"] - sdf.loc[i, "d_tail"]))
            if tail_gap <= 0 and direction_gap <= 0:
                continue
            pairs.append(
                {
                    "i": min(i, j),
                    "j": max(i, j),
                    "mu_gap": mu_gap,
                    "tail_gap": tail_gap,
                    "direction_gap": direction_gap,
                    "score": tail_gap + 0.5 * direction_gap,
                }
            )


def parse_weather_metadata(stem: str) -> dict[str, object]:
    match = WEATHER_RE.match(str(stem))
    if not match:
        return {
            "city": np.nan,
            "city_label": np.nan,
            "scenario_raw": np.nan,
            "time_slice": np.nan,
            "severity": np.nan,
            "forecast_year": np.nan,
        }
    data = match.groupdict()
    city_label = data["city"].title()
    severity_label = data["severity"].replace("_", " ")
    return {
        "city": data["city"],
        "city_label": city_label,
        "scenario_raw": data["scenario_raw"],
        "time_slice": data["time_slice"],
        "severity": data["severity"],
        "forecast_year": int(data["weather_year"]),
        "weather_label": (
            f"{city_label} {data['scenario_raw'].upper()} "
            f"{data['time_slice'].replace('_', ' ')} {severity_label}"
        ),
    }


def add_city_year(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    meta = pd.DataFrame([parse_weather_metadata(stem) for stem in out["weather"]], index=out.index)
    for col in meta.columns:
        out[col] = meta[col]
    return out


def find_phoenix_matched_pair(rows: pd.DataFrame, mu_bin_width: float = 0.02) -> pd.DataFrame:
    sdf = add_city_year(rows[rows["strategy"] == DEFAULT_STRATEGY]).reset_index(drop=True)
    sdf = sdf[
        sdf["city"].eq("phoenix")
        & sdf["time_slice"].isin(["baseline_2020s", "near_2030s", "late_2080s"])
        & sdf["expected_tsv"].notna()
        & sdf["p_tail"].notna()
    ].copy()
    if sdf.empty:
        return pd.DataFrame()
    sdf["_mu_bin"] = (sdf["expected_tsv"] / mu_bin_width).round() * mu_bin_width
    candidates = []
    for _, group in sdf.groupby("_mu_bin", sort=False):
        present = group[group["time_slice"].isin(["baseline_2020s", "near_2030s"])]
        future = group[group["time_slice"].eq("late_2080s")]
        if present.empty or future.empty:
            continue
        present_low = present.loc[present["p_tail"].idxmin()]
        future_high = future.loc[future["p_tail"].idxmax()]
        tail_gap = float(future_high["p_tail"] - present_low["p_tail"])
        if tail_gap <= 0:
            continue
        candidates.append(
            {
                "present_idx": int(present_low.name),
                "future_idx": int(future_high.name),
                "tail_gap": tail_gap,
                "mu_gap": abs(
                    float(future_high["expected_tsv"] - present_low["expected_tsv"])
                ),
            }
        )
    if not candidates:
        return pd.DataFrame()
    best = sorted(candidates, key=lambda item: (item["tail_gap"], -item["mu_gap"]), reverse=True)[0]
    pair = sdf.loc[[best["present_idx"], best["future_idx"]]].copy()
    pair["pair_tail_gap"] = best["tail_gap"]
    pair["pair_mu_gap"] = best["mu_gap"]
    return pair.sort_values("forecast_year")


class FenwickTree:
    def __init__(self, size: int):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int, value: int) -> None:
        idx = index + 1
        while idx <= self.size:
            self.tree[idx] += value
            idx += idx & -idx

    def prefix_sum(self, count: int) -> int:
        idx = min(max(int(count), 0), self.size)
        total = 0
        while idx > 0:
            total += int(self.tree[idx])
            idx -= idx & -idx
        return total


def count_matched_pairs(
    sdf: pd.DataFrame,
    *,
    deltas: list[float],
    etas: list[float],
    tail_threshold: float,
) -> pd.DataFrame:
    data = sdf[["expected_tsv", "p_tail"]].dropna().sort_values("expected_tsv")
    if len(data) < 2:
        return empty_pair_count_frame(deltas, etas)

    mu = data["expected_tsv"].to_numpy(dtype=float)
    p_tail = data["p_tail"].to_numpy(dtype=float)
    coords = np.unique(p_tail)
    coord_index = np.searchsorted(coords, p_tail)
    records = []

    for delta in deltas:
        tree = FenwickTree(len(coords))
        left = 0
        matched_pairs = 0
        threshold_crossing_pairs = 0
        divergent_by_eta = {eta: 0 for eta in etas}

        for right, (mu_right, p_right) in enumerate(zip(mu, p_tail)):
            while left < right and mu_right - mu[left] >= delta:
                tree.add(int(coord_index[left]), -1)
                left += 1

            window_count = right - left
            matched_pairs += window_count
            if window_count > 0:
                threshold_left_count = tree.prefix_sum(np.searchsorted(coords, tail_threshold, side="left"))
                if p_right >= tail_threshold:
                    threshold_crossing_pairs += threshold_left_count
                else:
                    threshold_crossing_pairs += window_count - threshold_left_count

                for eta in etas:
                    low_count = tree.prefix_sum(np.searchsorted(coords, p_right - eta, side="left"))
                    le_high_count = tree.prefix_sum(
                        np.searchsorted(coords, p_right + eta, side="right")
                    )
                    divergent_by_eta[eta] += low_count + (window_count - le_high_count)

            tree.add(int(coord_index[right]), 1)

        for eta in etas:
            records.append(
                {
                    "delta": delta,
                    "eta": eta,
                    "matched_mean_pairs": int(matched_pairs),
                    "tail_divergent_pairs": int(divergent_by_eta[eta]),
                    "risk_threshold_crossing_pairs": int(threshold_crossing_pairs),
                }
            )
    return pd.DataFrame(records)


def empty_pair_count_frame(deltas: list[float], etas: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "delta": delta,
                "eta": eta,
                "matched_mean_pairs": 0,
                "tail_divergent_pairs": 0,
                "risk_threshold_crossing_pairs": 0,
            }
            for delta in deltas
            for eta in etas
        ]
    )


def sum_pair_counts(frames: list[pd.DataFrame], deltas: list[float], etas: list[float]) -> pd.DataFrame:
    if not frames:
        return empty_pair_count_frame(deltas, etas)
    count_cols = [
        "matched_mean_pairs",
        "tail_divergent_pairs",
        "risk_threshold_crossing_pairs",
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["delta", "eta"], as_index=False)[count_cols]
        .sum()
    )


def subtract_pair_counts(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    count_cols = [
        "matched_mean_pairs",
        "tail_divergent_pairs",
        "risk_threshold_crossing_pairs",
    ]
    merged = primary.merge(secondary, on=["delta", "eta"], suffixes=("", "_subtract"))
    out = merged[["delta", "eta"]].copy()
    for col in count_cols:
        out[col] = (merged[col] - merged[f"{col}_subtract"]).clip(lower=0).astype("int64")
    return out


def add_pair_percentages(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "scope", scope)
    denom = out["matched_mean_pairs"].replace(0, np.nan)
    out["tail_divergent_pct_of_matched"] = out["tail_divergent_pairs"] / denom * 100.0
    out["risk_threshold_crossing_pct_of_matched"] = (
        out["risk_threshold_crossing_pairs"] / denom * 100.0
    )
    return out


def build_matched_mean_pair_diagnostics(
    rows: pd.DataFrame,
    *,
    deltas: list[float] | None = None,
    etas: list[float] | None = None,
    tail_threshold: float = 0.20,
    strategy: str = DEFAULT_STRATEGY,
) -> pd.DataFrame:
    if deltas is None:
        deltas = [0.005, 0.01, 0.025]
    if etas is None:
        etas = [0.05, 0.10]

    sdf = add_city_year(rows[rows["strategy"] == strategy]).copy()
    all_counts = count_matched_pairs(
        sdf, deltas=deltas, etas=etas, tail_threshold=tail_threshold
    )
    same_weather_counts = sum_pair_counts(
        [
            count_matched_pairs(group, deltas=deltas, etas=etas, tail_threshold=tail_threshold)
            for _, group in sdf.groupby("weather")
        ],
        deltas,
        etas,
    )
    cross_weather_counts = subtract_pair_counts(all_counts, same_weather_counts)

    city_counts = sum_pair_counts(
        [
            count_matched_pairs(group, deltas=deltas, etas=etas, tail_threshold=tail_threshold)
            for _, group in sdf.dropna(subset=["city"]).groupby("city")
        ],
        deltas,
        etas,
    )
    same_city_year_counts = sum_pair_counts(
        [
            count_matched_pairs(group, deltas=deltas, etas=etas, tail_threshold=tail_threshold)
            for _, group in sdf.dropna(subset=["city", "forecast_year"]).groupby(
                ["city", "forecast_year"]
            )
        ],
        deltas,
        etas,
    )
    same_city_cross_year_counts = subtract_pair_counts(city_counts, same_city_year_counts)

    out = pd.concat(
        [
            add_pair_percentages(all_counts, "all_pairs"),
            add_pair_percentages(cross_weather_counts, "cross_weather_pairs"),
            add_pair_percentages(same_city_cross_year_counts, "same_city_cross_year_pairs"),
        ],
        ignore_index=True,
    )
    out["tail_threshold"] = tail_threshold
    return out


def build_mean_bin_tail_spread(
    rows: pd.DataFrame,
    *,
    bin_widths: list[float] | None = None,
    strategy: str = DEFAULT_STRATEGY,
    tail_threshold: float = 0.20,
) -> pd.DataFrame:
    if bin_widths is None:
        bin_widths = [0.05, 0.10]
    sdf = rows[rows["strategy"] == strategy].copy()
    records = []
    for bin_width in bin_widths:
        bin_id = np.floor(sdf["expected_tsv"] / bin_width)
        work = sdf.assign(mu_bin_left=bin_id * bin_width)
        for left, group in work.groupby("mu_bin_left", sort=True):
            if len(group) < 20:
                continue
            q = group["p_tail"].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
            records.append(
                {
                    "strategy": strategy,
                    "bin_width": bin_width,
                    "mu_bin_left": float(left),
                    "mu_bin_right": float(left + bin_width),
                    "mu_bin_center": float(left + bin_width / 2.0),
                    "rows": int(len(group)),
                    "p_tail_q05": float(q.loc[0.05]),
                    "p_tail_q25": float(q.loc[0.25]),
                    "p_tail_median": float(q.loc[0.50]),
                    "p_tail_q75": float(q.loc[0.75]),
                    "p_tail_q95": float(q.loc[0.95]),
                    "p_tail_iqr": float(q.loc[0.75] - q.loc[0.25]),
                    "p_tail_90pct_spread": float(q.loc[0.95] - q.loc[0.05]),
                    "high_tail_pct": pct((group["p_tail"] >= tail_threshold).sum(), len(group)),
                }
            )
    return pd.DataFrame(records)


def build_directional_tail_diagnostics(
    rows: pd.DataFrame,
    *,
    eps_values: list[float] | None = None,
    direction_thresholds: list[float] | None = None,
    tail_threshold: float = 0.20,
    strategy: str = DEFAULT_STRATEGY,
) -> pd.DataFrame:
    if eps_values is None:
        eps_values = [0.15, 0.25, 0.35]
    if direction_thresholds is None:
        direction_thresholds = [0.04, 0.06, 0.08, 0.10]

    sdf = add_city_year(rows[rows["strategy"] == strategy]).copy()
    groups: list[tuple[str, str, pd.DataFrame]] = [("all_cases", "All cases", sdf)]
    groups.extend(
        ("weather_case", case_label(str(weather)), group)
        for weather, group in sdf.groupby("weather", sort=True)
    )
    records = []
    for scope, label, group in groups:
        high_tail = group["p_tail"] >= tail_threshold
        high_tail_n = int(high_tail.sum())
        for eps in eps_values:
            mu_sign = threshold_sign(group["expected_tsv"], eps)
            modest_mu = group["expected_tsv"].abs() < eps
            for direction_threshold in direction_thresholds:
                directional = group["abs_d_tail"] > direction_threshold
                tail_sign = threshold_sign(group["d_tail"], direction_threshold)
                comparable = (mu_sign != 0) & (tail_sign != 0)
                conflict = comparable & (mu_sign != tail_sign)
                ambiguous_high_tail = high_tail & ~directional
                records.append(
                    {
                        "scope": scope,
                        "case_label": label,
                        "eps": eps,
                        "direction_threshold": direction_threshold,
                        "tail_threshold": tail_threshold,
                        "rows": int(len(group)),
                        "modest_mu_directional_tail_pct": pct(
                            (modest_mu & directional).sum(), len(group)
                        ),
                        "sign_conflict_pct_of_comparable": pct(
                            conflict.sum(), comparable.sum()
                        ),
                        "sign_comparable_rows": int(comparable.sum()),
                        "high_tail_rows": high_tail_n,
                        "high_tail_ambiguous_direction_pct_of_all": pct(
                            ambiguous_high_tail.sum(), len(group)
                        ),
                        "high_tail_ambiguous_direction_pct_of_high_tail": pct(
                            ambiguous_high_tail.sum(), high_tail_n
                        ),
                        "warm_directional_tail_pct": pct(
                            (group["d_tail"] > direction_threshold).sum(), len(group)
                        ),
                        "cold_directional_tail_pct": pct(
                            (group["d_tail"] < -direction_threshold).sum(), len(group)
                        ),
                        "mean_abs_d_tail": float(group["abs_d_tail"].mean()),
                        "p95_abs_d_tail": float(group["abs_d_tail"].quantile(0.95)),
                    }
                )
    return pd.DataFrame(records)


def build_boundary_region_summary(
    rows: pd.DataFrame,
    *,
    lower: float = 0.50,
    upper: float = 0.70,
    tail_threshold: float = 0.20,
    strategy: str = DEFAULT_STRATEGY,
) -> pd.DataFrame:
    sdf = add_city_year(rows[rows["strategy"] == strategy]).copy()
    groups: list[tuple[str, str, pd.DataFrame]] = [("all_cases", "All cases", sdf)]
    groups.extend(
        ("weather_case", case_label(str(weather)), group)
        for weather, group in sdf.groupby("weather", sort=True)
    )
    records = []
    for scope, label, group in groups:
        boundary = group["expected_tsv"].ge(lower) & group["expected_tsv"].lt(upper)
        high_tail = group["p_tail"].ge(tail_threshold)
        boundary_rows = group[boundary]
        records.append(
            {
                "scope": scope,
                "case_label": label,
                "mu_lower": lower,
                "mu_upper": upper,
                "tail_threshold": tail_threshold,
                "occupied_probability_rows": int(len(group)),
                "boundary_rows": int(boundary.sum()),
                "boundary_pct_of_occupied": pct(boundary.sum(), len(group)),
                "boundary_high_tail_rows": int((boundary & high_tail).sum()),
                "boundary_high_tail_pct_of_boundary": pct(
                    (boundary & high_tail).sum(), boundary.sum()
                ),
                "boundary_high_tail_pct_of_occupied": pct(
                    (boundary & high_tail).sum(), len(group)
                ),
                "all_high_tail_pct": pct(high_tail.sum(), len(group)),
                "boundary_mean_p_tail": float(boundary_rows["p_tail"].mean())
                if not boundary_rows.empty
                else np.nan,
                "boundary_median_p_tail": float(boundary_rows["p_tail"].median())
                if not boundary_rows.empty
                else np.nan,
                "boundary_mean_abs_d_tail": float(boundary_rows["abs_d_tail"].mean())
                if not boundary_rows.empty
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def best_abs_mu_threshold_for_tail(sdf: pd.DataFrame, tail_threshold: float) -> float:
    data = sdf[["abs_mu", "p_tail"]].dropna().sort_values("abs_mu")
    if data.empty:
        return np.nan
    values = data["abs_mu"].to_numpy(dtype=float)
    high_tail = data["p_tail"].ge(tail_threshold).to_numpy(dtype=bool)
    thresholds = np.unique(values)

    best_threshold = float(thresholds[0])
    best_errors = len(values) + 1
    for threshold in thresholds:
        left = np.searchsorted(values, threshold, side="left")
        false_negatives = int(high_tail[:left].sum())
        false_positives = int((~high_tail[left:]).sum())
        errors = false_negatives + false_positives
        if errors < best_errors:
            best_errors = errors
            best_threshold = float(threshold)

    no_action_errors = int(high_tail.sum())
    if no_action_errors < best_errors:
        return float(values.max() + np.finfo(float).eps)
    return best_threshold


def build_mean_threshold_decision_diagnostics(
    rows: pd.DataFrame,
    *,
    lower: float = 0.50,
    upper: float = 0.70,
    tail_threshold: float = 0.20,
    strategy: str = DEFAULT_STRATEGY,
) -> pd.DataFrame:
    sdf = add_city_year(rows[rows["strategy"] == strategy]).copy()
    threshold = best_abs_mu_threshold_for_tail(sdf, tail_threshold)
    groups: list[tuple[str, str, pd.DataFrame]] = [("all_cases", "All cases", sdf)]
    groups.extend(
        ("weather_case", case_label(str(weather)), group)
        for weather, group in sdf.groupby("weather", sort=True)
    )
    records = []
    for scope, label, group in groups:
        probability_action = group["p_tail"].ge(tail_threshold)
        mean_action = group["abs_mu"].ge(threshold)
        disagreement = probability_action != mean_action
        false_negative = probability_action & ~mean_action
        false_positive = mean_action & ~probability_action
        boundary = group["expected_tsv"].ge(lower) & group["expected_tsv"].lt(upper)
        boundary_n = int(boundary.sum())
        records.append(
            {
                "scope": scope,
                "case_label": label,
                "mu_lower": lower,
                "mu_upper": upper,
                "tail_threshold": tail_threshold,
                "best_abs_mu_threshold": threshold,
                "rows": int(len(group)),
                "probability_gate_action_pct": pct(probability_action.sum(), len(group)),
                "mean_threshold_action_pct": pct(mean_action.sum(), len(group)),
                "decision_disagreement_pct": pct(disagreement.sum(), len(group)),
                "false_negative_pct": pct(false_negative.sum(), len(group)),
                "false_positive_pct": pct(false_positive.sum(), len(group)),
                "boundary_rows": boundary_n,
                "boundary_disagreement_pct": pct((boundary & disagreement).sum(), boundary_n),
                "boundary_false_negative_pct": pct((boundary & false_negative).sum(), boundary_n),
                "boundary_false_positive_pct": pct((boundary & false_positive).sum(), boundary_n),
            }
        )
    return pd.DataFrame(records)


def write_manuscript_figure(rows: pd.DataFrame, output_dir: Path) -> list[Path]:
    diagnostic = add_city_year(rows[rows["strategy"] == DEFAULT_STRATEGY].copy())
    if diagnostic.empty:
        return []

    pair = find_phoenix_matched_pair(rows)
    tail_spread = build_mean_bin_tail_spread(rows, bin_widths=[0.05], tail_threshold=0.20)
    boundary = build_boundary_region_summary(rows, tail_threshold=0.20)
    boundary_all = boundary[boundary["scope"].eq("all_cases")]
    boundary_pct = (
        float(boundary_all["boundary_pct_of_occupied"].iloc[0]) if not boundary_all.empty else np.nan
    )
    boundary_high_pct = (
        float(boundary_all["boundary_high_tail_pct_of_boundary"].iloc[0])
        if not boundary_all.empty
        else np.nan
    )
    fig = plt.figure(figsize=(11.2, 7.0), dpi=300)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0])
    ax_scatter = fig.add_subplot(grid[:, 0])
    ax_bins = fig.add_subplot(grid[0, 1])
    ax_pair = fig.add_subplot(grid[1, 1])

    sc = ax_scatter.scatter(
        diagnostic["expected_tsv"],
        diagnostic["p_tail"],
        c=diagnostic["d_tail"],
        cmap="coolwarm",
        s=10,
        alpha=0.58,
        linewidths=0,
    )
    ax_scatter.axvline(0, color="#444444", lw=0.75)
    ax_scatter.axvspan(-0.15, 0.15, color="#d9d9d9", alpha=0.25, lw=0)
    ax_scatter.axvspan(0.50, 0.70, color="#f2c66d", alpha=0.18, lw=0)
    ax_scatter.axhline(0.20, color="#7a2d42", lw=0.9, ls="--")
    ax_scatter.set_xlabel(r"Expected sensation $\mu_{\mathrm{TSV}}$")
    ax_scatter.set_ylabel("Discomfort-tail probability Pr(|TSV| >= 2)")
    ax_scatter.set_title(
        "A. Expected sensation is informative but\nnot sufficient near the diagnostic boundary",
        loc="left",
        weight="bold",
        fontsize=9.2,
    )
    ax_scatter.grid(color="#dddddd", lw=0.55)
    cbar = fig.colorbar(sc, ax=ax_scatter, fraction=0.046, pad=0.035)
    cbar.set_label(r"Directional tail $d_{\mathrm{tail}}$")
    ax_scatter.text(
        0.60,
        0.545,
        "risk-boundary\nregion",
        ha="center",
        va="top",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#b28a2e", "alpha": 0.9, "pad": 2.5},
    )

    if not pair.empty and len(pair) == 2:
        present, future = list(pair.itertuples(index=False))
        x_mid = float(np.mean([present.expected_tsv, future.expected_tsv]))
        low_tail = float(present.p_tail)
        high_tail = float(future.p_tail)
        ratio = high_tail / low_tail if low_tail > 0 else np.nan
        ax_scatter.scatter(
            [present.expected_tsv, future.expected_tsv],
            [present.p_tail, future.p_tail],
            s=84,
            marker="o",
            facecolors=["#ffffff", "#111111"],
            edgecolors="#111111",
            linewidths=1.3,
            zorder=5,
        )
        ax_scatter.annotate(
            "",
            xy=(x_mid, high_tail),
            xytext=(x_mid, low_tail),
            arrowprops={"arrowstyle": "<->", "color": "#111111", "lw": 1.2},
            zorder=6,
        )
        ax_scatter.text(
            x_mid + 0.08,
            (low_tail + high_tail) / 2.0,
            rf"similar $\mu_{{\mathrm{{TSV}}}}$, {ratio:.1f}x tail risk",
            fontsize=8.6,
            va="center",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.92, "pad": 3},
        )

    if not tail_spread.empty:
        bins = tail_spread[tail_spread["bin_width"].eq(0.05)].copy().sort_values("mu_bin_center")
        bins = bins[bins["mu_bin_left"].ge(0.30) & bins["mu_bin_right"].le(0.75)]
        colors = np.where(
            bins["mu_bin_left"].ge(0.50) & bins["mu_bin_right"].le(0.70),
            "#b76b4b",
            "#8ea6cf",
        )
        ax_bins.bar(
            bins["mu_bin_center"],
            bins["high_tail_pct"],
            width=0.043,
            color=colors,
            edgecolor="#5d6d83",
            linewidth=0.35,
        )
        ax_bins.axvspan(0.50, 0.70, color="#f2c66d", alpha=0.18, lw=0)
        ax_bins.set_xlim(0.28, 0.73)
        ax_bins.set_ylim(0, 105)
        ax_bins.set_xlabel(r"$\mu_{\mathrm{TSV}}$ bin")
        ax_bins.set_ylabel(r"High-tail prevalence (%)")
        ax_bins.set_title("B. Threshold decisions vary within mean bins", loc="left", weight="bold")
        if np.isfinite(boundary_pct) and np.isfinite(boundary_high_pct):
            ax_bins.text(
                0.305,
                78,
                f"{boundary_pct:.1f}% of occupied states\n{boundary_high_pct:.1f}% high-tail",
                fontsize=7.7,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "#b28a2e", "alpha": 0.9, "pad": 2.5},
            )
        ax_bins.grid(axis="y", color="#dddddd", lw=0.55)
    else:
        ax_bins.axis("off")

    if not pair.empty and len(pair) == 2:
        labels = [
            f"{int(row.forecast_year)}\n{int(row.month)}/{int(row.day)} {row.current_time:.2f}"
            for row in pair.itertuples(index=False)
        ]
        cold = pair["cold_discomfort_probability"].to_numpy(dtype=float)
        warm = pair["warm_discomfort_probability"].to_numpy(dtype=float)
        center = 1.0 - cold - warm
        x = np.arange(len(pair))
        ax_pair.bar(x, cold, color="#4f6f9f", label="Pr(TSV <= -2)")
        ax_pair.bar(
            x,
            center,
            bottom=cold,
            color="#d9d9d9",
            label="Pr(-1 <= TSV <= +1)",
        )
        ax_pair.bar(
            x,
            warm,
            bottom=cold + center,
            color="#b76b4b",
            label="Pr(TSV >= +2)",
        )
        for xpos, row in zip(x, pair.itertuples(index=False)):
            ax_pair.text(
                xpos,
                1.025,
                rf"$\mu$={row.expected_tsv:.2f}" "\n" rf"$p_{{tail}}$={row.p_tail:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax_pair.set_ylim(0, 1.16)
        ax_pair.set_xticks(x)
        ax_pair.set_xticklabels(labels)
        ax_pair.set_ylabel("Probability mass")
        ax_pair.set_title("C. Matched-mean Phoenix example", loc="left", weight="bold")
        ax_pair.grid(axis="y", color="#dddddd", lw=0.55)
        ax_pair.legend(frameon=True, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        ax_pair.axis("off")

    fig.suptitle(
        "Expected thermal sensation is not a sufficient statistic for discomfort-tail risk",
        weight="bold",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    png_path = output_dir / "probability_necessity_manuscript_figure.png"
    pdf_path = output_dir / "probability_necessity_manuscript_figure.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_tail_spread_plot(tail_spread: pd.DataFrame, output_dir: Path) -> list[Path]:
    if tail_spread.empty:
        return []
    plot_data = tail_spread[tail_spread["bin_width"].eq(0.05)].copy()
    if plot_data.empty:
        plot_data = tail_spread[tail_spread["bin_width"].eq(tail_spread["bin_width"].min())].copy()
    plot_data = plot_data.sort_values("mu_bin_center")

    x = plot_data["mu_bin_center"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    ax.fill_between(
        x,
        plot_data["p_tail_q05"].to_numpy(dtype=float),
        plot_data["p_tail_q95"].to_numpy(dtype=float),
        color="#b8c7e6",
        alpha=0.38,
        label="5-95%",
    )
    ax.fill_between(
        x,
        plot_data["p_tail_q25"].to_numpy(dtype=float),
        plot_data["p_tail_q75"].to_numpy(dtype=float),
        color="#4f6f9f",
        alpha=0.34,
        label="IQR",
    )
    ax.plot(
        x,
        plot_data["p_tail_median"].to_numpy(dtype=float),
        color="#243f73",
        lw=1.8,
        label="Median",
    )
    ax.axhline(0.20, color="#7a2d42", lw=0.9, ls="--", label=r"$p_{\mathrm{tail}}=0.20$")
    ax.axvline(0, color="#444444", lw=0.75)
    ax.axvspan(-0.15, 0.15, color="#d9d9d9", alpha=0.25, lw=0)
    ax.set_xlabel(r"Expected sensation bin center $\mu_{\mathrm{TSV}}$")
    ax.set_ylabel(r"Discomfort-tail probability $p_{\mathrm{tail}}$")
    ax.set_title("Within-bin spread of discomfort-tail risk", weight="bold")
    ax.grid(color="#dddddd", lw=0.55)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    path = output_dir / "tail_spread_by_mean_bin.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def sample_plot_rows(rows: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return rows.sample(n=max_rows, random_state=42).sort_index()


def case_label(stem: str) -> str:
    meta = parse_weather_metadata(str(stem))
    if pd.isna(meta.get("city")):
        return str(stem).replace("_", " ")
    scenario = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}.get(
        str(meta["scenario_raw"]), str(meta["scenario_raw"]).upper()
    )
    time_slice = str(meta["time_slice"]).replace("_", " ")
    severity = str(meta["severity"]).replace("_", " ")
    return f"{meta['city_label']} {scenario} {time_slice} {severity} {int(meta['forecast_year'])}"


def write_plots(rows: pd.DataFrame, examples: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    diagnostic = rows[rows["strategy"] == DEFAULT_STRATEGY].copy()
    if not diagnostic.empty:
        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
        sc = ax.scatter(
            diagnostic["expected_tsv"],
            diagnostic["p_tail"],
            c=diagnostic["d_tail"],
            cmap="coolwarm",
            s=12,
            alpha=0.72,
            linewidths=0,
        )
        ax.axvline(0, color="#444444", lw=0.8)
        ax.axvspan(-0.15, 0.15, color="#d9d9d9", alpha=0.25, lw=0)
        ax.axhline(0.20, color="#7a2d42", lw=0.9, ls="--")
        pair = find_phoenix_matched_pair(rows)
        if not pair.empty and len(pair) == 2:
            present, future = list(pair.itertuples(index=False))
            x_mid = float(np.mean([present.expected_tsv, future.expected_tsv]))
            low_tail = float(present.p_tail)
            high_tail = float(future.p_tail)
            ratio = high_tail / low_tail if low_tail > 0 else np.nan
            ax.scatter(
                [present.expected_tsv, future.expected_tsv],
                [present.p_tail, future.p_tail],
                s=82,
                marker="o",
                facecolors=["#ffffff", "#111111"],
                edgecolors="#111111",
                linewidths=1.3,
                zorder=5,
            )
            ax.annotate(
                "",
                xy=(x_mid, high_tail),
                xytext=(x_mid, low_tail),
                arrowprops={"arrowstyle": "<->", "color": "#111111", "lw": 1.2},
                zorder=6,
            )
            ax.text(
                x_mid + 0.08,
                (low_tail + high_tail) / 2.0,
                rf"similar $\mu_{{\mathrm{{TSV}}}}$, {ratio:.1f}x tail risk",
                fontsize=8.2,
                va="center",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.92, "pad": 3},
            )
        ax.set_xlabel(r"Expected sensation $\mu_{\mathrm{TSV}}$")
        ax.set_ylabel(r"Discomfort-tail probability $P(|\mathrm{TSV}|\geq2)$")
        ax.set_title(
            "Expected thermal sensation is not a sufficient statistic\n"
            "for discomfort-tail risk",
            weight="bold",
            fontsize=11.2,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"Directional tail $P(\mathrm{TSV}\geq2)-P(\mathrm{TSV}\leq-2)$")
        ax.grid(color="#dddddd", lw=0.6)
        fig.tight_layout()
        path = output_dir / "mean_vs_tail_scatter.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if not examples.empty:
        first = examples[examples["pair_id"] == examples["pair_id"].min()].copy()
        labels = [
            f"{row.member}\n{row.case_label}\n{int(row.month)}/{int(row.day)} {row.current_time:.2f}"
            for row in first.itertuples(index=False)
        ]
        metrics = [
            ("expected_tsv", r"$\mu_{\mathrm{TSV}}$"),
            ("p_tail", r"$P(|\mathrm{TSV}|\geq2)$"),
            ("d_tail", r"$d_{\mathrm{tail}}$"),
        ]
        x = np.arange(len(metrics))
        width = 0.34
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=180)
        colors = ["#4f6f9f", "#b76b4b"]
        for idx, row in enumerate(first.itertuples(index=False)):
            vals = [float(getattr(row, col)) for col, _label in metrics]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, color=colors[idx], label=labels[idx])
        ax.axhline(0, color="#444444", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _col, label in metrics])
        ax.set_title("Real trace pair: similar mean, different tail risk", weight="bold")
        ax.legend(fontsize=7.2, frameon=True)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        fig.tight_layout()
        path = output_dir / "same_mean_different_tail_pair.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    args = parse_args()
    df = load_trace(args.trace, args.trace_dir)
    availability = availability_report(df, args.strategies)
    print("[dry-run] trace:", args.trace)
    print("[dry-run] rows:", len(df))
    print("[dry-run] weather_cases:", df["weather"].nunique())
    print("[dry-run] availability:")
    print(availability.to_string(index=False))

    if args.dry_run:
        if availability.empty or int(availability["occupied_probability_rows"].sum()) <= 0:
            raise RuntimeError("No occupied probability rows available for diagnostics.")
        print("[dry-run] OK: probability-necessity diagnostic can run from existing traces.")
        return 0

    rows = prepare_diagnostic_rows(df, args.strategies)
    if rows.empty:
        raise RuntimeError("No occupied probability rows available after filtering.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        rows,
        eps=args.eps,
        tail_threshold=args.tail_threshold,
        direction_threshold=args.direction_threshold,
    )
    threshold_grid = build_threshold_grid(rows, strategy=DEFAULT_STRATEGY)
    examples = find_same_mean_tail_pairs(
        rows,
        strategy=DEFAULT_STRATEGY,
        mu_window=args.mu_window,
        top_examples=args.top_examples,
    )
    matched_mean_pairs = (
        pd.DataFrame()
        if args.skip_matched_pairs
        else build_matched_mean_pair_diagnostics(
            rows,
            tail_threshold=args.tail_threshold,
            strategy=DEFAULT_STRATEGY,
        )
    )
    tail_spread = build_mean_bin_tail_spread(
        rows,
        tail_threshold=args.tail_threshold,
        strategy=DEFAULT_STRATEGY,
    )
    directional_diagnostics = build_directional_tail_diagnostics(
        rows,
        tail_threshold=args.tail_threshold,
        strategy=DEFAULT_STRATEGY,
    )
    boundary_region = build_boundary_region_summary(
        rows,
        tail_threshold=args.tail_threshold,
        strategy=DEFAULT_STRATEGY,
    )
    mean_threshold_decisions = build_mean_threshold_decision_diagnostics(
        rows,
        tail_threshold=args.tail_threshold,
        strategy=DEFAULT_STRATEGY,
    )

    rows_path = args.output_dir / "probability_diagnostic_rows.csv"
    summary_path = args.output_dir / "probability_necessity_summary.csv"
    threshold_path = args.output_dir / "probability_threshold_grid.csv"
    examples_path = args.output_dir / "same_mean_different_tail_examples.csv"
    matched_pairs_path = args.output_dir / "matched_mean_tail_divergence.csv"
    tail_spread_path = args.output_dir / "tail_spread_by_mean_bin.csv"
    directional_path = args.output_dir / "directional_tail_diagnostics.csv"
    boundary_region_path = args.output_dir / "boundary_region_summary.csv"
    mean_threshold_decisions_path = args.output_dir / "mean_threshold_decision_diagnostics.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    threshold_grid.to_csv(threshold_path, index=False)
    examples.to_csv(examples_path, index=False)
    matched_mean_pairs.to_csv(matched_pairs_path, index=False)
    tail_spread.to_csv(tail_spread_path, index=False)
    directional_diagnostics.to_csv(directional_path, index=False)
    boundary_region.to_csv(boundary_region_path, index=False)
    mean_threshold_decisions.to_csv(mean_threshold_decisions_path, index=False)
    plot_rows = sample_plot_rows(rows, args.plot_sample_rows)
    if len(plot_rows) < len(rows):
        print(f"[plot] sampled {len(plot_rows)} of {len(rows)} rows for scatter plots")
    plot_paths = write_plots(plot_rows, examples, args.output_dir)
    plot_paths.extend(write_manuscript_figure(plot_rows, args.output_dir))
    plot_paths.extend(write_tail_spread_plot(tail_spread, args.output_dir))

    print(f"[write] {rows_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {threshold_path}")
    print(f"[write] {examples_path}")
    print(f"[write] {matched_pairs_path}")
    print(f"[write] {tail_spread_path}")
    print(f"[write] {directional_path}")
    print(f"[write] {boundary_region_path}")
    print(f"[write] {mean_threshold_decisions_path}")
    for path in plot_paths:
        print(f"[write] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

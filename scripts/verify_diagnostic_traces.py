#!/usr/bin/env python3
"""Audit diagnostic-reference trace files without loading all cases at once."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = REPO_ROOT / "restricted_inputs" / "simulation_traces"
DEFAULT_OUT = REPO_ROOT / "outputs" / "simulation_qa" / "trace_schema_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def audit_trace(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    zone_p_cols = [c for c in df.columns if c.startswith("zone_") and c.endswith("_p_disc")]
    zone_mu_cols = [c for c in df.columns if c.startswith("zone_") and c.endswith("_expected_tsv")]
    occupied = bool_series(df["occupied"])
    probability = df["expected_tsv"].notna() & df["discomfort_probability"].notna()
    return {
        "trace_file": path.name,
        "weather": str(df["weather"].iloc[0]) if "weather" in df and len(df) else path.stem,
        "strategy": str(df["strategy"].iloc[0]) if "strategy" in df and len(df) else "",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "zone_p_disc_cols": int(len(zone_p_cols)),
        "zone_expected_tsv_cols": int(len(zone_mu_cols)),
        "occupied_rows": int(occupied.sum()),
        "probability_rows": int(probability.sum()),
        "occupied_probability_rows": int((occupied & probability).sum()),
        "occupied_zone_probability_missing_rows": int(df.loc[occupied, zone_p_cols].isna().any(axis=1).sum())
        if zone_p_cols
        else -1,
        "action_delta_max_abs": float(pd.to_numeric(df["action_delta_c"], errors="coerce").abs().max()),
        "setpoint_shift_max_abs": float(
            pd.to_numeric(df["setpoint_shift_c"], errors="coerce").abs().max()
        ),
        "heating_setpoints": ";".join(
            map(
                str,
                sorted(
                    pd.to_numeric(df["heating_setpoint_c"], errors="coerce")
                    .dropna()
                    .round(3)
                    .unique()
                ),
            )
        ),
        "cooling_setpoints": ";".join(
            map(
                str,
                sorted(
                    pd.to_numeric(df["cooling_setpoint_c"], errors="coerce")
                    .dropna()
                    .round(3)
                    .unique()
                ),
            )
        ),
        "p_tail_min": float(pd.to_numeric(df["discomfort_probability"], errors="coerce").min()),
        "p_tail_max": float(pd.to_numeric(df["discomfort_probability"], errors="coerce").max()),
    }


def main() -> int:
    args = parse_args()
    paths = sorted(args.trace_dir.glob("*_diagnostic_reference.csv"))
    if not paths:
        raise FileNotFoundError(f"No diagnostic traces found in {args.trace_dir}")
    rows = [audit_trace(path) for path in paths]
    out = pd.DataFrame(rows).sort_values("weather")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[audit] wrote {len(out)} trace rows: {args.output}")
    print(f"[audit] weather cases: {out['weather'].nunique()}")
    print(f"[audit] occupied probability rows: {int(out['occupied_probability_rows'].sum())}")
    print(f"[audit] max action delta: {out['action_delta_max_abs'].max():.3f}")
    print(f"[audit] max setpoint shift: {out['setpoint_shift_max_abs'].max():.3f}")
    print(
        "[audit] max occupied zone-probability missing rows: "
        f"{int(out['occupied_zone_probability_missing_rows'].max())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

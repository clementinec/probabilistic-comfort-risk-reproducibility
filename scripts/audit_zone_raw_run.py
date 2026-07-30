#!/usr/bin/env python3
"""Audit a zone-raw simulation run before downstream diagnostic analyses."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "restricted_inputs" / "simulation_run"
DEFAULT_OUT = ROOT / "outputs" / "simulation_qa"
EXPECTED_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704
EXPECTED_CASES = 144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-cases", type=int, default=EXPECTED_CASES)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--expected-occupied-rows", type=int, default=EXPECTED_OCCUPIED_ROWS)
    return parser.parse_args()


def trace_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "traces").glob("*_diagnostic_reference.csv"))


def err_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "energyplus").glob("*/diagnostic_reference/eplusout.err"))


def end_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "energyplus").glob("*/diagnostic_reference/eplusout.end"))


def zone_raw_columns(columns: list[str]) -> list[str]:
    return [
        col
        for col in columns
        if col.startswith("zone_") and col.endswith(("_ta_c", "_tr_c", "_rh_pct"))
    ]


def audit_trace(path: Path, reference_header: list[str] | None, args: argparse.Namespace) -> dict[str, object]:
    header = list(pd.read_csv(path, nrows=0).columns)
    raw_cols = zone_raw_columns(header)
    cols = ["occupied", *raw_cols]
    rows = 0
    occupied_rows = 0
    missing_raw_occupied = 0
    nonfinite_raw_occupied = 0
    mins = np.full(len(raw_cols), np.inf)
    maxs = np.full(len(raw_cols), -np.inf)

    for chunk in pd.read_csv(path, usecols=cols, chunksize=50_000):
        rows += len(chunk)
        occupied = chunk["occupied"].astype(str).str.lower().isin({"1", "true", "yes"})
        occupied_rows += int(occupied.sum())
        if raw_cols and occupied.any():
            raw = chunk.loc[occupied, raw_cols].apply(pd.to_numeric, errors="coerce")
            missing_raw_occupied += int(raw.isna().sum().sum())
            values = raw.to_numpy(float)
            nonfinite_raw_occupied += int((~np.isfinite(values)).sum())
            if values.size:
                mins = np.minimum(mins, np.nanmin(values, axis=0))
                maxs = np.maximum(maxs, np.nanmax(values, axis=0))

    constant_raw_cols = 0
    if raw_cols:
        constant_raw_cols = int(np.sum(np.isfinite(mins) & np.isfinite(maxs) & np.isclose(mins, maxs)))

    return {
        "trace": path.name,
        "rows": rows,
        "occupied_rows": occupied_rows,
        "columns": len(header),
        "zone_raw_fields": len(raw_cols),
        "header_matches_reference": reference_header is None or header == reference_header,
        "expected_rows": rows == args.expected_rows,
        "expected_occupied_rows": occupied_rows == args.expected_occupied_rows,
        "missing_raw_occupied": missing_raw_occupied,
        "nonfinite_raw_occupied": nonfinite_raw_occupied,
        "constant_raw_columns": constant_raw_cols,
        "zero_byte": path.stat().st_size == 0,
    }


def audit_err(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    severe = len(re.findall(r"\*\*\s*Severe\s*\*\*", text))
    fatal = len(re.findall(r"\*\*\s*Fatal\s*\*\*", text))
    warmup_convergence = len(re.findall(r"CheckWarmupConvergence", text))
    summary = ""
    for line in reversed(text.splitlines()):
        if "EnergyPlus Completed" in line or "EnergyPlus Terminated" in line:
            summary = line.strip()
            break
    warning_match = re.search(r"Completed Successfully--\s*(\d+)\s*Warning;\s*(\d+)\s*Severe", summary)
    return {
        "case": path.parents[1].name,
        "err": str(path.relative_to(path.parents[4])),
        "severe_lines": severe,
        "fatal_lines": fatal,
        "warmup_convergence_lines": warmup_convergence,
        "summary_warnings": int(warning_match.group(1)) if warning_match else None,
        "summary_severe": int(warning_match.group(2)) if warning_match else None,
        "completed_successfully": "Completed Successfully" in summary,
        "summary": summary,
    }


def write_summary(
    path: Path,
    trace_df: pd.DataFrame,
    err_df: pd.DataFrame,
    end_count: int,
    args: argparse.Namespace,
) -> None:
    complete_traces = len(trace_df) == args.expected_cases
    complete_errs = len(err_df) == args.expected_cases
    complete_ends = end_count == args.expected_cases
    ok_schema = bool(
        len(trace_df)
        and trace_df["header_matches_reference"].all()
        and trace_df["zone_raw_fields"].eq(45).all()
        and trace_df["columns"].nunique() == 1
    )
    ok_rows = bool(
        len(trace_df)
        and trace_df["expected_rows"].all()
        and trace_df["expected_occupied_rows"].all()
    )
    ok_values = bool(
        len(trace_df)
        and trace_df["missing_raw_occupied"].sum() == 0
        and trace_df["nonfinite_raw_occupied"].sum() == 0
        and trace_df["constant_raw_columns"].sum() == 0
    )
    ok_energyplus = bool(
        len(err_df)
        and err_df["fatal_lines"].sum() == 0
        and err_df["completed_successfully"].all()
    )
    severe_mask = err_df["severe_lines"].gt(0) | err_df["summary_severe"].fillna(0).gt(0)
    severe_cases = err_df.loc[
        severe_mask, ["case", "severe_lines", "warmup_convergence_lines", "summary_severe", "summary"]
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Zone-Raw Diagnostic Run QA\n\n")
        f.write(f"- Run directory: `{args.run_dir}`\n")
        f.write(f"- Trace CSVs: {len(trace_df)} / {args.expected_cases}\n")
        f.write(f"- EnergyPlus `.err` files: {len(err_df)} / {args.expected_cases}\n")
        f.write(f"- EnergyPlus `.end` files: {end_count} / {args.expected_cases}\n")
        f.write(f"- Inventory complete: {complete_traces and complete_errs and complete_ends}\n")
        f.write(f"- Schema OK: {ok_schema}\n")
        f.write(f"- Annual row counts OK: {ok_rows}\n")
        f.write(f"- Occupied raw zone values OK: {ok_values}\n")
        f.write(f"- EnergyPlus completed without fatal errors: {ok_energyplus}\n")
        f.write(f"- Cases with Severe reports: {len(severe_cases)}\n\n")
        if not severe_cases.empty:
            f.write("## Severe Cases\n\n")
            f.write("```csv\n")
            f.write(severe_cases.to_csv(index=False))
            f.write("```\n\n")
        if len(trace_df):
            f.write("## Trace Audit Head\n\n")
            f.write("```csv\n")
            f.write(trace_df.head(20).to_csv(index=False))
            f.write("```\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = trace_paths(args.run_dir)
    reference_header = list(pd.read_csv(traces[0], nrows=0).columns) if traces else None
    trace_df = pd.DataFrame([audit_trace(path, reference_header, args) for path in traces])
    err_df = pd.DataFrame([audit_err(path) for path in err_paths(args.run_dir)])
    end_count = len(end_paths(args.run_dir))

    trace_df.to_csv(args.output_dir / "zone_raw_trace_audit.csv", index=False)
    err_df.to_csv(args.output_dir / "zone_raw_err_audit.csv", index=False)
    write_summary(args.output_dir / "zone_raw_run_qa_summary.md", trace_df, err_df, end_count, args)

    print(f"[write] {args.output_dir / 'zone_raw_trace_audit.csv'}")
    print(f"[write] {args.output_dir / 'zone_raw_err_audit.csv'}")
    print(f"[write] {args.output_dir / 'zone_raw_run_qa_summary.md'}")
    if len(trace_df) != args.expected_cases:
        print(f"[warn] trace inventory incomplete: {len(trace_df)}/{args.expected_cases}")
    if len(err_df) and int(err_df["fatal_lines"].sum()) > 0:
        print("[warn] fatal errors found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

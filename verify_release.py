#!/usr/bin/env python3
"""Verify the public reproducibility package using the standard library."""

from __future__ import annotations

import csv
import hashlib
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKSUMS = ROOT / "CHECKSUMS_SHA256.txt"
HEADLINE = ROOT / "outputs/core/corrected_headline_case_summary.csv"

EXPECTED_HEADLINES = {
    "mean-zone screened share": (
        "corrected_equal_zone_mean_high_pct",
        13.147300047893,
    ),
    "area-weighted mean-probability screen": (
        "corrected_area_weighted_mean_high_pct",
        12.130037050341,
    ),
    "any-zone screened share": (
        "corrected_any_zone_high_pct",
        36.643169300766,
    ),
    "hidden any-zone screened share": (
        "corrected_hidden_any_zone_pct",
        23.495869252874,
    ),
    "area-weighted zone-time screened share": (
        "corrected_area_weighted_zone_time_high_pct",
        12.872694464516,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums() -> int:
    checked = 0
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing checksummed file: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {relative}: {actual} != {expected}"
            )
        checked += 1
    return checked


def verify_headlines() -> None:
    with HEADLINE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 144:
        raise RuntimeError(f"Expected 144 role rows, found {len(rows)}")
    unique_states = {row["corrected_state_hash"] for row in rows}
    if len(unique_states) != 119:
        raise RuntimeError(
            f"Expected 119 unique synchronized states, found {len(unique_states)}"
        )

    for label, (column, expected) in EXPECTED_HEADLINES.items():
        actual = statistics.fmean(float(row[column]) for row in rows)
        if abs(actual - expected) > 1e-9:
            raise RuntimeError(
                f"{label} mismatch: {actual:.12f} != {expected:.12f}"
            )


def main() -> int:
    checked = verify_checksums()
    verify_headlines()
    print(
        f"PASS: {checked} checksums; 144 roles; "
        "119 synchronized states; headline values verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

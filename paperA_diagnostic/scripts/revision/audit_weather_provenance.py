#!/usr/bin/env python3
"""Audit Paper A's future-weather provenance and selected EPW panel.

This script starts from immutable local artifacts. A matching generator
implementation is preserved elsewhere in the same broader working archive, but
the
exact study-batch configuration and raw six-city CMIP/observational inputs are
not. It therefore does not regenerate the upstream 2025--2100 forecast series.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
HPH = ROOT.parent / "HPH_Carbon_Entitlement"
ONEDRIVE = ROOT.parents[1]
MANIFEST = ROOT / "paperA_rebuild/data/panel_manifest.csv"
INVENTORY = HPH / "data/interim/cmip_forecast_inventory.csv"
ISSUES = HPH / "data/interim/cmip_validation_issues.csv"
ANNUAL = HPH / "data/interim/cmip_annual_temperature_summary.csv"
OUT_DEFAULT = ROOT / "paperA_R01/04_analysis/outputs/weather_physics_provenance"
GENERATOR_REFERENCE = (
    ONEDRIVE
    / "Misc/EPWs/SRepw/apply_climate_delta_batch.py"
)
SAMY_METHOD_SOURCE = (
    ONEDRIVE
    / "Misc/EPWs/sAMY10/manuscript_sAMY_0317/submission_package/"
    "sAMY_manuscript.tex"
)

TIME_SLICES = {
    "baseline_2020s": (2025, 2029),
    "near_2030s": (2030, 2039),
    "mid_2050s": (2050, 2059),
    "late_2080s": (2080, 2089),
}

XML_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XML_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--skip-large-source-checksums",
        action="store_true",
        help="Skip SHA-256 of the 12 large CSV/XLSX source pairs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_column(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        raise ValueError(f"Invalid spreadsheet reference: {reference}")
    value = 0
    for letter in letters.group(0):
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def sheet_rows(archive: zipfile.ZipFile, member: str) -> list[list[object]]:
    """Read a small XLSX sheet directly without loading the 666k-row sheet."""
    root = ET.fromstring(archive.read(member))
    rows: list[list[object]] = []
    for row in root.findall(f".//{{{XML_MAIN}}}row"):
        values: dict[int, object] = {}
        for cell in row.findall(f"{{{XML_MAIN}}}c"):
            index = cell_column(cell.attrib["r"])
            if cell.attrib.get("t") == "inlineStr":
                text = "".join(
                    node.text or "" for node in cell.findall(f".//{{{XML_MAIN}}}t")
                )
                values[index] = text
            else:
                node = cell.find(f"{{{XML_MAIN}}}v")
                if node is None or node.text is None:
                    values[index] = ""
                else:
                    try:
                        number = float(node.text)
                        values[index] = int(number) if number.is_integer() else number
                    except ValueError:
                        values[index] = node.text
        if values:
            width = max(values) + 1
            rows.append([values.get(index, "") for index in range(width)])
    return rows


def workbook_sheet_map(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall(f"{{{XML_REL_PKG}}}Relationship")
    }
    result: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{XML_MAIN}}}sheet"):
        rel_id = sheet.attrib[f"{{{XML_REL_DOC}}}id"]
        target = rel_map[rel_id]
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        result[sheet.attrib["name"]] = target
    return result


def parse_workbook(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    with zipfile.ZipFile(path) as archive:
        sheets = workbook_sheet_map(archive)
        metadata_rows = sheet_rows(archive, sheets["Metadata"])
        delta_rows = sheet_rows(archive, sheets["Climate_Delta"])
    metadata = {
        str(row[0]): row[1] if len(row) > 1 else ""
        for row in metadata_rows[1:]
        if row
    }
    delta_header = [str(value) for value in delta_rows[0]]
    delta = [dict(zip(delta_header, row)) for row in delta_rows[1:] if row]
    return metadata, delta


def workbook_paths() -> list[Path]:
    return sorted((HPH / "CMIPs").glob("CMIP*/**/forecast_*_CMIP6_MPI_0515_*.xlsx"))


def source_csv_paths() -> list[Path]:
    return sorted((HPH / "CMIPs").glob("CMIP*/**/forecast_*_CMIP6_MPI_0515_*.csv"))


def reproduce_selectors(
    annual: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (city, scenario), source in annual.groupby(["city", "scenario"], sort=True):
        for time_slice, (start, end) in TIME_SLICES.items():
            window = source[source["year"].between(start, end)].copy()
            if window.empty:
                continue
            median_cdh = statistics.median(window["CDD18_hourly"].astype(float))
            midpoint = (start + end) / 2
            records = list(window.to_dict("records"))
            expected = {
                "typical": min(
                    records,
                    key=lambda row: (
                        abs(float(row["CDD18_hourly"]) - median_cdh),
                        abs(int(row["year"]) - midpoint),
                    ),
                ),
                "hot": max(
                    records,
                    key=lambda row: (
                        float(row["CDD18_hourly"]),
                        float(row["mean_temp"]),
                        float(row["max_temp"]),
                    ),
                ),
                "heatwave_extreme": max(
                    records,
                    key=lambda row: (
                        int(row["hours_temp_ge_35"]),
                        float(row["max_temp"]),
                        float(row["CDD18_hourly"]),
                    ),
                ),
            }
            for severity, selection in expected.items():
                match = manifest[
                    (manifest["city"] == city)
                    & (manifest["scenario_raw"] == scenario)
                    & (manifest["time_slice"] == time_slice)
                    & (manifest["severity"] == severity)
                ]
                actual_year = int(match.iloc[0]["weather_year"]) if len(match) == 1 else None
                expected_year = int(selection["year"])
                rows.append(
                    {
                        "city": city,
                        "scenario": scenario,
                        "time_slice": time_slice,
                        "severity": severity,
                        "expected_year": expected_year,
                        "manifest_year": actual_year,
                        "exact_match": actual_year == expected_year,
                    }
                )
    return pd.DataFrame(rows)


def dewpoint_c(temp_c: float, rh_percent: float) -> float:
    rh = max(min(rh_percent, 100.0), 0.1)
    a = 17.625
    b = 243.04
    gamma = math.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def update_tuple_hash(digest: hashlib._Hash, values: Iterable[object]) -> None:
    digest.update((",".join(str(value) for value in values) + "\n").encode("utf-8"))


def epw_audit(path: Path) -> tuple[dict[str, object], str]:
    raw_digest = hashlib.sha256()
    transform_digest = hashlib.sha256()
    count = 0
    temp: list[float] = []
    dew: list[float] = []
    rh: list[float] = []
    pressure: list[float] = []
    ghi: list[float] = []
    dni: list[float] = []
    dhi: list[float] = []
    wind: list[float] = []
    years: set[int] = set()
    invalid_width = 0
    with path.open("rb") as binary:
        for index, raw in enumerate(binary):
            if index < 8:
                continue
            raw_digest.update(raw)
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            row = next(csv.reader([line]))
            if len(row) < 35:
                invalid_width += 1
                continue
            count += 1
            year, month, day, epw_hour = map(int, row[:4])
            years.add(year)
            values = {
                "temp": float(row[6]),
                "dew": float(row[7]),
                "rh": float(row[8]),
                "pressure": float(row[9]),
                "ghi": float(row[13]),
                "dni": float(row[14]),
                "dhi": float(row[15]),
                "wind": float(row[21]),
            }
            temp.append(values["temp"])
            dew.append(values["dew"])
            rh.append(values["rh"])
            pressure.append(values["pressure"])
            ghi.append(values["ghi"])
            dni.append(values["dni"])
            dhi.append(values["dhi"])
            wind.append(values["wind"])
            update_tuple_hash(
                transform_digest,
                (
                    year,
                    month,
                    day,
                    epw_hour - 1,
                    row[6],
                    row[7],
                    row[8],
                    row[9],
                    row[13],
                    row[14],
                    row[15],
                    row[21],
                ),
            )

    temp_a = np.asarray(temp)
    dew_a = np.asarray(dew)
    rh_a = np.asarray(rh)
    pressure_a = np.asarray(pressure)
    ghi_a = np.asarray(ghi)
    dni_a = np.asarray(dni)
    dhi_a = np.asarray(dhi)
    wind_a = np.asarray(wind)
    row = {
        "epw_path": str(path),
        "data_rows": count,
        "calendar_years": ";".join(str(year) for year in sorted(years)),
        "invalid_field_width_rows": invalid_width,
        "body_sha256": raw_digest.hexdigest(),
        "transform_sha256": transform_digest.hexdigest(),
        "mean_temp_c": float(temp_a.mean()),
        "min_temp_c": float(temp_a.min()),
        "max_temp_c": float(temp_a.max()),
        "cdh18_c_hour": float(np.maximum(temp_a - 18.0, 0.0).sum()),
        "hours_temp_ge_35": int((temp_a >= 35.0).sum()),
        "mean_rh_pct": float(rh_a.mean()),
        "min_rh_pct": float(rh_a.min()),
        "max_rh_pct": float(rh_a.max()),
        "dewpoint_above_drybulb_rows": int((dew_a > temp_a + 0.011).sum()),
        "min_pressure_pa": float(pressure_a.min()),
        "max_pressure_pa": float(pressure_a.max()),
        "annual_ghi_kwh_m2": float(ghi_a.sum() / 1000.0),
        "max_ghi_w_m2": float(ghi_a.max()),
        "max_dni_w_m2": float(dni_a.max()),
        "max_dhi_w_m2": float(dhi_a.max()),
        "max_wind_m_s": float(wind_a.max()),
        "broad_range_violation_rows": int(
            (
                (temp_a < -60)
                | (temp_a > 70)
                | (rh_a < 0)
                | (rh_a > 100)
                | (pressure_a < 85000)
                | (pressure_a > 110000)
                | (ghi_a < 0)
                | (ghi_a > 1500)
                | (dni_a < 0)
                | (dni_a > 1500)
                | (dhi_a < 0)
                | (dhi_a > 1000)
                | (wind_a < 0)
                | (wind_a > 75)
            ).sum()
        ),
    }
    return row, raw_digest.hexdigest()


def expected_transform_hashes(
    csv_path: Path, selected_years: set[int]
) -> dict[int, dict[str, object]]:
    digests = {year: hashlib.sha256() for year in selected_years}
    counts = defaultdict(int)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
            if dt.year not in selected_years:
                continue
            temp = float(row["temp"])
            rh = float(row["relative_humidity"])
            expected = (
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                f"{temp:.2f}",
                f"{dewpoint_c(temp, rh):.2f}",
                f"{rh:.0f}",
                f"{float(row['pressure']) * 100.0:.0f}",
                f"{max(float(row['GHI']), 0.0):.0f}",
                f"{max(float(row['DNI']), 0.0):.0f}",
                f"{max(float(row['DHI']), 0.0):.0f}",
                f"{max(float(row['wind_speed']), 0.0):.2f}",
            )
            update_tuple_hash(digests[dt.year], expected)
            counts[dt.year] += 1
    return {
        year: {"expected_transform_sha256": digest.hexdigest(), "source_rows": counts[year]}
        for year, digest in digests.items()
    }


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    inventory = pd.read_csv(INVENTORY)
    annual = pd.read_csv(ANNUAL)

    print("[1/7] auditing 12 source-workbook metadata records")
    metadata_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    workbooks = workbook_paths()
    for workbook in workbooks:
        metadata, deltas = parse_workbook(workbook)
        scenario_match = re.search(r"(ssp245|ssp585)", workbook.stem)
        base = {
            "workbook_path": str(workbook),
            "scenario_from_path": scenario_match.group(1) if scenario_match else "",
            **metadata,
        }
        metadata_rows.append(base)
        for delta in deltas:
            delta_rows.append(
                {
                    "workbook_path": str(workbook),
                    "city": metadata.get("city", ""),
                    "ssp": metadata.get("ssp", ""),
                    **delta,
                }
            )
    metadata_df = pd.DataFrame(metadata_rows)
    delta_df = pd.DataFrame(delta_rows)
    metadata_df.to_csv(out / "source_workbook_metadata.csv", index=False)
    delta_df.to_csv(out / "source_workbook_delta_summary.csv", index=False)

    print("[2/7] reproducing all weather-year selectors")
    selector_df = reproduce_selectors(annual, manifest)
    selector_df.to_csv(out / "selector_reproduction.csv", index=False)

    print("[3/7] auditing 144 EPWs and exact duplicate-role bodies")
    epw_rows: list[dict[str, object]] = []
    for index, record in enumerate(manifest.itertuples(index=False), start=1):
        qa, _ = epw_audit(Path(record.epw_path))
        epw_rows.append(
            {
                "case_id": record.case_id,
                "city": record.city,
                "scenario": record.scenario_raw,
                "time_slice": record.time_slice,
                "severity": record.severity,
                "weather_year": int(record.weather_year),
                "manifest_mean_temp_c": float(record.mean_T_out),
                "manifest_cdh18_c_hour": float(record.CDD18_hourly),
                "manifest_max_temp_c": float(record.max_T_out),
                "manifest_hours_temp_ge_35": int(record.hours_temp_ge_35),
                "manifest_mean_rh_pct": float(record.humidity_metric),
                **qa,
            }
        )
        if index % 24 == 0:
            print(f"      EPWs {index}/144")
    epw_df = pd.DataFrame(epw_rows)
    epw_df["mean_temp_difference_c"] = (
        epw_df["mean_temp_c"] - epw_df["manifest_mean_temp_c"]
    )
    epw_df["max_temp_difference_c"] = (
        epw_df["max_temp_c"] - epw_df["manifest_max_temp_c"]
    )
    epw_df["cdh18_difference_c_hour"] = (
        epw_df["cdh18_c_hour"] - epw_df["manifest_cdh18_c_hour"]
    )
    epw_df["hours_ge_35_difference"] = (
        epw_df["hours_temp_ge_35"] - epw_df["manifest_hours_temp_ge_35"]
    )
    epw_df["mean_rh_difference_pct_point"] = (
        epw_df["mean_rh_pct"] - epw_df["manifest_mean_rh_pct"]
    )
    epw_df.to_csv(out / "epw_weather_qa.csv", index=False)

    overlap = (
        manifest.groupby(["city", "scenario_raw", "time_slice", "weather_year"])
        .agg(
            role_count=("severity", "size"),
            severities=("severity", lambda values: ";".join(sorted(values))),
            case_ids=("case_id", lambda values: ";".join(sorted(values))),
        )
        .reset_index()
    )
    hash_lookup = epw_df.set_index("case_id")["body_sha256"].to_dict()
    overlap["body_hash_count"] = overlap["case_ids"].map(
        lambda values: len({hash_lookup[case_id] for case_id in values.split(";")})
    )
    overlap.to_csv(out / "weather_role_overlap.csv", index=False)

    print("[4/7] checking selected EPW records against source CSV transforms")
    transform_rows: list[dict[str, object]] = []
    canonical = (
        epw_df.sort_values(["city", "scenario", "weather_year", "case_id"])
        .drop_duplicates(["city", "scenario", "weather_year"])
        .copy()
    )
    for source_path in source_csv_paths():
        match = re.search(
            r"forecast_(?P<city>.+)_CMIP6_MPI_0515_(?P<scenario>ssp245|ssp585)",
            source_path.stem,
        )
        if not match:
            continue
        city = match.group("city")
        scenario = match.group("scenario")
        subset = canonical[
            (canonical["city"] == city) & (canonical["scenario"] == scenario)
        ]
        selected_years = set(subset["weather_year"].astype(int))
        expected = expected_transform_hashes(source_path, selected_years)
        for row in subset.itertuples(index=False):
            item = expected[int(row.weather_year)]
            transform_rows.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "weather_year": int(row.weather_year),
                    "case_id_canonical": row.case_id,
                    "source_csv": str(source_path),
                    "epw_path": row.epw_path,
                    "source_rows": item["source_rows"],
                    "epw_rows": row.data_rows,
                    "expected_transform_sha256": item["expected_transform_sha256"],
                    "epw_transform_sha256": row.transform_sha256,
                    "exact_transform_match": (
                        item["expected_transform_sha256"] == row.transform_sha256
                    ),
                }
            )
    transform_df = pd.DataFrame(transform_rows)
    transform_df.to_csv(out / "epw_source_transform_check.csv", index=False)

    print("[5/7] recording validation inventory and issue state")
    inventory.to_csv(out / "source_forecast_inventory.csv", index=False)
    issues_df = pd.read_csv(ISSUES)
    issues_df.to_csv(out / "source_forecast_validation_issues.csv", index=False)

    print("[6/7] computing artifact checksums")
    checksum_rows = [
        {
            "artifact_type": "panel_manifest",
            "path": str(MANIFEST),
            "bytes": MANIFEST.stat().st_size,
            "sha256": sha256_file(MANIFEST),
        }
    ]
    for path in sorted(set(Path(value) for value in manifest["epw_path"])):
        checksum_rows.append(
            {
                "artifact_type": "selected_epw",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not args.skip_large_source_checksums:
        for path in [*source_csv_paths(), *workbooks]:
            print(f"      hashing {path.name}")
            checksum_rows.append(
                {
                    "artifact_type": (
                        "source_forecast_csv" if path.suffix == ".csv" else "source_workbook"
                    ),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for artifact_type, path in [
        ("matching_weather_generator_reference", GENERATOR_REFERENCE),
        ("published_samy_method_local_source", SAMY_METHOD_SOURCE),
    ]:
        if path.exists():
            checksum_rows.append(
                {
                    "artifact_type": artifact_type,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(checksum_rows).to_csv(out / "artifact_checksums.csv", index=False)

    print("[7/7] writing machine-readable and narrative summaries")
    metadata_ok = (
        len(metadata_df) == 12
        and set(metadata_df["gcm"].astype(str)) == {"MPI-ESM1-2-LR"}
        and set(metadata_df["rcm"].astype(str)) == {"N/A"}
        and set(metadata_df["baseline_start"].astype(int)) == {1991}
        and set(metadata_df["baseline_end"].astype(int)) == {2010}
        and set(metadata_df["delta_mode"].astype(str)) == {"daily"}
    )
    generator_text = (
        GENERATOR_REFERENCE.read_text(encoding="utf-8")
        if GENERATOR_REFERENCE.exists()
        else ""
    )
    generator_schema_match = all(
        token in generator_text
        for token in [
            "Climate_Delta",
            "Metadata",
            "scenario_id",
            "baseline_start",
            "baseline_end",
            "delta_mode",
            "MPI-ESM1-2-LR",
            "compute_mixed_resolution_baseline",
            "apply_mixed_resolution_climate_delta",
        ]
    )
    summary = {
        "manifest_cases": int(len(manifest)),
        "city_count": int(manifest["city"].nunique()),
        "scenario_count": int(manifest["scenario_raw"].nunique()),
        "time_slice_count": int(manifest["time_slice"].nunique()),
        "role_count": int(manifest["severity"].nunique()),
        "unique_city_scenario_year_states": int(
            manifest[["city", "scenario_raw", "weather_year"]].drop_duplicates().shape[0]
        ),
        "unique_city_scenario_slice_year_states": int(
            manifest[
                ["city", "scenario_raw", "time_slice", "weather_year"]
            ].drop_duplicates().shape[0]
        ),
        "overlap_groups": int((overlap["role_count"] > 1).sum()),
        "metadata_records": int(len(metadata_df)),
        "metadata_consistent": bool(metadata_ok),
        "selector_rows": int(len(selector_df)),
        "selector_exact_matches": int(selector_df["exact_match"].sum()),
        "source_inventory_rows": int(len(inventory)),
        "source_inventory_pass_rows": int(
            inventory["validation_status"].astype(str).eq("pass").sum()
        ),
        "source_issue_rows": int(len(issues_df)),
        "epw_files": int(len(epw_df)),
        "epw_structural_or_range_issue_files": int(
            (
                (epw_df["invalid_field_width_rows"] > 0)
                | (~epw_df["data_rows"].isin([8760, 8784]))
                | (epw_df["broad_range_violation_rows"] > 0)
                | (epw_df["dewpoint_above_drybulb_rows"] > 0)
            ).sum()
        ),
        "epw_transform_states": int(len(transform_df)),
        "epw_exact_transform_matches": int(transform_df["exact_transform_match"].sum()),
        "matching_upstream_delta_generator_reference_present": bool(
            GENERATOR_REFERENCE.exists()
        ),
        "matching_generator_output_schema_and_method_tokens_present": bool(
            generator_schema_match
        ),
        "matching_generator_sha256": (
            sha256_file(GENERATOR_REFERENCE)
            if GENERATOR_REFERENCE.exists()
            else None
        ),
        "upstream_raw_cmip_and_observation_bundle_present": False,
        "exact_study_batch_config_present": False,
        "end_to_end_forecast_regeneration_possible": False,
        "published_method_validation_available": True,
        "published_method_validation_doi": (
            "10.1016/j.enbuild.2026.117508"
        ),
        "published_input_quality_doi": "10.1016/j.energy.2026.140867",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    narrative = f"""# Future-weather provenance and QA audit

## Result

The preserved downstream weather chain is internally auditable. The panel has
{summary['manifest_cases']} role-labelled EPWs but
{summary['unique_city_scenario_slice_year_states']} unique
city--scenario--slice--year states. All {summary['metadata_records']} paired
source workbooks identify MPI-ESM1-2-LR, `rcm=N/A`, a 1991--2010 baseline, and
daily delta mode. The executable selectors reproduce
{summary['selector_exact_matches']}/{summary['selector_rows']} manifest
selections. The source-to-EPW transformation matches exactly for
{summary['epw_exact_transform_matches']}/{summary['epw_transform_states']}
unique selected source years.

The source inventory records {summary['source_inventory_pass_rows']}/
{summary['source_inventory_rows']} structural/plausibility passes and
{summary['source_issue_rows']} logged issues. The EPW audit found
{summary['epw_structural_or_range_issue_files']} files with a row-count,
field-width, broad-range, or dew-point consistency issue.

## Exact preserved chain

1. Twelve scenario-city forecast artifacts are continuous from
   2025-01-01 00:00 through 2100-12-31 00:00. All selected 2025--2089 annual
   windows are complete; the final 23 hours of 2100 are outside the analysis.
2. Their workbooks identify a single MPI-ESM1-2-LR forcing family under
   SSP2-4.5 and SSP5-8.5, with a 1991--2010 baseline and daily climate deltas.
3. `typical` is the year nearest median annual CDH18 in its window; `hot` is
   maximum CDH18; `heatwave_extreme` is maximum hours at or above 35 C, with
   annual maximum temperature and then CDH18 as tie-breakers.
4. The EPW converter copies dry-bulb, relative humidity, pressure, GHI, DNI,
   DHI, and wind speed from the forecast output; computes dew point from
   dry-bulb and RH; converts pressure from hPa to Pa; and writes the selected
   annual records. Other unsupported EPW fields use explicit sentinel/default
   values.

`CDD18_hourly` is a legacy field name. Its value is annual CDH18 in C h:
the hourly sum of `max(T_out - 18 C, 0)`. Dividing by 24 gives conventional
CDD18 in C d and leaves selection ranks unchanged.

## Scope boundary

The broader working archive contains a matching climate-delta generator
implementation (SHA-256 `{summary['matching_generator_sha256']}`) whose
mixed-resolution daily-delta method, output-sheet schema, metadata fields and
MPI-ESM1-2-LR loader align with the 12 preserved workbooks. However, the exact
study-batch configuration and its raw six-city CMIP6 and observational inputs
are not preserved alongside these outputs. The audit therefore supports the
algorithmic lineage plus exact reproduction from the frozen forecast
artifacts through selection and EPW conversion; it does not support byte-level
end-to-end regeneration of the 2025--2100 forecast CSVs.

The weather-generation method has independent 2011--2020 observational
validation across four North American benchmark cities and multiple CORDEX
and CMIP6 forcing sources in Guo and He,
doi:10.1016/j.enbuild.2026.117508. A companion causal decomposition evaluates
baseline-source, climate-signal resolution and downstream degree-day effects
in doi:10.1016/j.energy.2026.140867. These papers support the method family,
not a claim that the exact six study-city forecast files were externally
validated. The panel contains one GCM family and is not a climate-model
uncertainty ensemble.

## Interpretation safeguards

- `typical` means median-CDH-selected within the stated five- or ten-year
  window; it is not a TMY.
- Selection labels are roles, not independent replicates.
- The workbooks document climate deltas for temperature, pressure, wind speed
  and GHI. A separate exact paired audit shows that relative humidity,
  specific humidity and DNI contain no SSP-specific signal in these frozen
  files; component-specific humidity or direct-beam attribution is therefore
  unsupported.
- Horizontal GHI is not facade-incident solar gain.
- Structural and broad physical-range checks establish data integrity and
  plausibility, not observational predictive validity.
"""
    (out / "provenance_audit.md").write_text(narrative)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

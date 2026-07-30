#!/usr/bin/env python3
"""Render the existing metabolic sensitivity summaries with bounded screening labels.

This script changes presentation only. It reads the previously generated
case-zone and profile-zone summaries; it does not refit the TSV model, alter
probabilities, or rerun EnergyPlus.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "metabolic_profile"
DEFAULT_OUTPUT = ROOT / "figures"

os.environ.setdefault(
    "MPLCONFIGDIR", str(ROOT / ".mplconfig")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(ROOT / ".mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROFILE_ORDER = ["COMP-Lo", "S-real", "COMP-Med", "EQ-Max", "COMP-Hi", "LEGACY"]
ZONE_ORDER = [
    "core_bottom",
    "perimeter_bot_zn_1",
    "perimeter_bot_zn_2",
    "perimeter_bot_zn_3",
    "perimeter_bot_zn_4",
    "core_mid",
    "perimeter_mid_zn_1",
    "perimeter_mid_zn_2",
    "perimeter_mid_zn_3",
    "perimeter_mid_zn_4",
    "core_top",
    "perimeter_top_zn_1",
    "perimeter_top_zn_2",
    "perimeter_top_zn_3",
    "perimeter_top_zn_4",
]
ZONE_LABELS = {
    "core_bottom": "Core bottom",
    "core_mid": "Core middle",
    "core_top": "Core top",
    "perimeter_bot_zn_1": "Bottom P1",
    "perimeter_bot_zn_2": "Bottom P2",
    "perimeter_bot_zn_3": "Bottom P3",
    "perimeter_bot_zn_4": "Bottom P4",
    "perimeter_mid_zn_1": "Middle P1",
    "perimeter_mid_zn_2": "Middle P2",
    "perimeter_mid_zn_3": "Middle P3",
    "perimeter_mid_zn_4": "Middle P4",
    "perimeter_top_zn_1": "Top P1",
    "perimeter_top_zn_2": "Top P2",
    "perimeter_top_zn_3": "Top P3",
    "perimeter_top_zn_4": "Top P4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def render(input_dir: Path, output_dir: Path) -> None:
    zone_path = input_dir / "zone_metabolic_profile_zone_summary.csv"
    spread_path = input_dir / "zone_metabolic_profile_case_zone_spread.csv"
    scenario_zone = pd.read_csv(zone_path)
    spread_case_zone = pd.read_csv(spread_path)

    required_zone = {"scenario", "watts_person", "zone", "mean_high_tail_pct"}
    required_spread = {"zone", "high_tail_pct_profile_spread"}
    if not required_zone.issubset(scenario_zone.columns):
        raise ValueError(f"Missing fields in {zone_path}")
    if not required_spread.issubset(spread_case_zone.columns):
        raise ValueError(f"Missing fields in {spread_path}")

    profile_watts = (
        scenario_zone[["scenario", "watts_person"]]
        .drop_duplicates()
        .set_index("scenario")
        .loc[PROFILE_ORDER, "watts_person"]
    )
    profile_labels = [
        f"{profile}\n{profile_watts.loc[profile]:.0f} W" for profile in PROFILE_ORDER
    ]
    heat = (
        scenario_zone.pivot_table(
            index="scenario",
            columns="zone",
            values="mean_high_tail_pct",
            aggfunc="first",
            observed=True,
        )
        .loc[PROFILE_ORDER, ZONE_ORDER]
        .to_numpy(float)
    )

    spread_rows: list[dict[str, float | str]] = []
    for zone in ZONE_ORDER:
        values = spread_case_zone.loc[
            spread_case_zone["zone"].eq(zone), "high_tail_pct_profile_spread"
        ].to_numpy(float)
        spread_rows.append(
            {
                "zone": zone,
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "mean": float(np.mean(values)),
            }
        )
    spread = pd.DataFrame(spread_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 9.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.95, 1.35]},
    )
    vmax = max(40.0, float(np.ceil(heat.max() / 5.0) * 5.0))
    image = axes[0].imshow(
        heat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto"
    )
    axes[0].set_yticks(range(len(PROFILE_ORDER)))
    axes[0].set_yticklabels(profile_labels)
    axes[0].set_xticks(range(len(ZONE_ORDER)))
    axes[0].set_xticklabels(
        [ZONE_LABELS[zone] for zone in ZONE_ORDER], rotation=55, ha="right"
    )
    axes[0].set_title(
        "Mean high-tail screened share by metabolic profile and zone",
        fontweight="bold",
    )
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = float(heat[i, j])
            axes[0].text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > vmax * 0.55 else "#222222",
            )
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.025, pad=0.015)
    colorbar.set_label("High-tail screened share (%)")

    y = np.arange(len(ZONE_ORDER))
    axes[1].hlines(
        y,
        spread["q05"],
        spread["q95"],
        color="#bdbdbd",
        linewidth=1.2,
        label="5th-95th percentile",
    )
    axes[1].hlines(
        y,
        spread["q25"],
        spread["q75"],
        color="#525252",
        linewidth=5.0,
        alpha=0.88,
        label="Interquartile range",
    )
    axes[1].plot(
        spread["median"], y, "o", color="#111111", ms=4.2, label="Median"
    )
    axes[1].plot(
        spread["mean"],
        y,
        marker="D",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="#111111",
        markeredgewidth=1.0,
        ms=4.2,
        label="Mean",
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([ZONE_LABELS[zone] for zone in ZONE_ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, float(np.ceil(spread["q95"].max() / 5.0) * 5.0))
    axes[1].set_xlabel("Spread across metabolic profiles (percentage points)")
    axes[1].set_title(
        "Within-zone screened-share spread over weather cases", fontweight="bold"
    )
    axes[1].grid(axis="x", color="#d9d9d9", linewidth=0.6)
    axes[1].legend(frameon=False, ncol=4, loc="lower right", fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle(
        "Zone-resolved metabolic sensitivity of high-tail screening",
        fontweight="bold",
    )
    fig.savefig(output_dir / "zone_metabolic_profile_summary.png", dpi=260)
    fig.savefig(output_dir / "zone_metabolic_profile_summary.pdf")
    plt.close(fig)

    note = (
        "Presentation-only rerender from existing case-zone summaries. "
        "No probability inference, model fitting, or EnergyPlus simulation was rerun.\n"
    )
    (output_dir / "README.txt").write_text(note, encoding="utf-8")


def main() -> int:
    args = parse_args()
    render(args.input_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

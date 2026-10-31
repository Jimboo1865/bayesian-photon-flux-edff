#!/usr/bin/env python3
"""Rebuild BNN Figure 12 (notebook Part 12, cell 37) with a percent-error panel.

The main panel is the separated-uncertainty plot at b = 20 fm.  The shorter
lower panel is aligned to it with a shared rapidity axis and shows
100 * sigma / central for the epistemic, nuclear-parameter, and total
uncertainty components already calculated by cell 37.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    Path.home()
    / "Downloads"
    / "bnn_flux_outputs"
    / "uncertainty_decomposition_publication.csv"
)


def plot_band(axis, x, centre, sigma, label, color, alpha):
    lower = np.maximum(centre - sigma, np.finfo(float).tiny)
    upper = centre + sigma
    axis.fill_between(x, lower, upper, color=color, alpha=alpha, label=label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("fig_part12_cell37_with_percent_error"),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="CSV with y, edff_flux, finite_epa_flux columns to overlay on the main panel.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv).sort_values("y")
    required = {"y", "central", "sigma_epistemic", "sigma_nuclear", "sigma_total"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = frame["y"].to_numpy()
    centre = frame["central"].to_numpy()
    sigma_epistemic = frame["sigma_epistemic"].to_numpy()
    sigma_nuclear = frame["sigma_nuclear"].to_numpy()
    sigma_total = frame["sigma_total"].to_numpy()

    percent_epistemic = 100.0 * sigma_epistemic / centre
    percent_nuclear = 100.0 * sigma_nuclear / centre
    percent_total = 100.0 * sigma_total / centre

    fig, (axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(8, 6.7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1.0], "hspace": 0.06},
    )

    line_color = "#000F4D"
    epistemic_color = "#6baed6"
    total_color = "#fdae61"

    axis.plot(
        x,
        centre,
        color=line_color,
        linewidth=2.4,
        label="Surrogate mean",
        zorder=2,
    )
    if args.reference_csv is not None:
        ref = pd.read_csv(args.reference_csv).sort_values("y")
        axis.plot(
            ref["y"],
            ref["finite_epa_flux"],
            color="#FA8C19",
            linewidth=2.1,
            linestyle=(0, (7, 3)),
            label="Numerical finite-size EPA",
            zorder=3,
        )
        axis.plot(
            ref["y"],
            ref["edff_flux"],
            color="black",
            linestyle=(0, (3.5, 2)),
            linewidth=1.3,
            label="Analytical EDFF",
            zorder=3,
        )
    axis.set_yscale("log")
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_ylabel(r"$N(\omega,b=20\,\mathrm{fm})$ [fm$^{-2}$]")
    axis.set_title("Separated uncertainty sources")
    axis.legend(loc="lower left")

    error_axis.plot(x, percent_epistemic, color="#86269B", linewidth=1.6, label="Epistemic")
    error_axis.plot(
        x,
        percent_nuclear,
        color="#006B35",
        linestyle="--",
        linewidth=1.6,
        label="Nuclear parameters",
    )
    error_axis.plot(x, percent_total, color="#B0006B", linewidth=1.6, label="Total")
    error_axis.set_xlabel("Rapidity proxy y")
    error_axis.set_ylabel("Percent error (%)")
    error_axis.set_ylim(0, max(1.2, float(np.max(percent_total)) * 1.12))
    error_axis.grid(axis="y", linestyle=":", alpha=0.55)
    error_axis.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Percent-error range (total): {percent_total.min():.4f}–{percent_total.max():.4f}%")
    print(f"Percent-error range (epistemic): {percent_epistemic.min():.4f}–{percent_epistemic.max():.4f}%")
    print(f"Percent-error range (nuclear): {percent_nuclear.min():.4f}–{percent_nuclear.max():.4f}%")


if __name__ == "__main__":
    main()

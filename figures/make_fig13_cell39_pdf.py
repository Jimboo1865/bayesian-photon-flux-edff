#!/usr/bin/env python3
"""Rebuild BNN Figure 13 (Part 13, cell 39) as a vector PDF.

This follows the notebook's 2x2 Sobol sample-size layout: log-scale panels on
the top row and linear-scale panels on the bottom row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = (
    Path.home()
    / "Downloads"
    / "bnn_flux_outputs"
    / "sobol_sample_size_study_publication.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("fig_part13_cell39_n1.pdf"),
    )
    args = parser.parse_args()

    sample_size_table = pd.read_csv(args.input_csv).sort_values("n_train")
    plt.rcParams["legend.framealpha"] = 1.0
    required = {
        "n_train",
        "median_absolute_relative",
        "p95_absolute_relative",
        "median_epistemic_relative",
        "p90_epistemic_relative",
    }
    missing = required.difference(sample_size_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top row: log y-scale.
    axes[0, 0].plot(
        sample_size_table["n_train"],
        100.0 * sample_size_table["median_absolute_relative"],
        "o-",
        label="Median absolute residual",
    )
    axes[0, 0].plot(
        sample_size_table["n_train"],
        100.0 * sample_size_table["p95_absolute_relative"],
        "s--",
        label="95th-percentile residual",
    )
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Sobol training points")
    axes[0, 0].set_ylabel("Held-out residual [%] (log)")
    axes[0, 0].set_title("Accuracy vs. Sample Size (Log Scale)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, which="both", ls=":")

    axes[0, 1].plot(
        sample_size_table["n_train"],
        100.0 * sample_size_table["median_epistemic_relative"],
        "o-",
        label="Median epistemic uncertainty",
    )
    axes[0, 1].plot(
        sample_size_table["n_train"],
        100.0 * sample_size_table["p90_epistemic_relative"],
        "s--",
        label="90th-percentile epistemic uncertainty",
    )
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Sobol training points")
    axes[0, 1].set_ylabel("Calibrated epistemic uncertainty [%] (log)")
    axes[0, 1].set_title("Epistemic Uncertainty vs. Sample Size (Log Scale)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, which="both", ls=":")

    # Bottom row: linear y-scale.
    axes[1, 0].plot(
        sample_size_table["n_train"],
        100.0 * sample_size_table["median_absolute_relative"],
        "o-",
        label="Median absolute residual",
    )
    axes[1, 0].plot(
        sample_size_table["n_train"],
        100.0 * sample_size_table["p95_absolute_relative"],
        "s--",
        label="95th-percentile residual",
    )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("linear")
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].set_xlabel("Sobol training points")
    axes[1, 0].set_ylabel("Held-out residual [%] (linear)")
    axes[1, 0].set_title("Accuracy vs. Sample Size (Linear Scale)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, ls=":")

    axes[1, 1].plot(
        sample_size_table["n_train"],
        100.0 * sample_size_table["median_epistemic_relative"],
        "o-",
        label="Median epistemic uncertainty",
    )
    axes[1, 1].plot(
        sample_size_table["n_train"],
        100.0 * sample_size_table["p90_epistemic_relative"],
        "s--",
        label="90th-percentile epistemic uncertainty",
    )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("linear")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].set_xlabel("Sobol training points")
    axes[1, 1].set_ylabel("Calibrated epistemic uncertainty [%] (linear)")
    axes[1, 1].set_title("Epistemic Uncertainty vs. Sample Size (Linear Scale)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, ls=":")

    plt.tight_layout()
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf)
    plt.close(fig)
    print(f"Wrote {args.output_pdf}")
    print(f"Rows: {len(sample_size_table)}; n_train: {sample_size_table['n_train'].min()}–{sample_size_table['n_train'].max()}")


if __name__ == "__main__":
    main()

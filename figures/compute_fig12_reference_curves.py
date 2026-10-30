#!/usr/bin/env python3
"""Compute the two reference curves for BNN Figure 12 (Part 12, cell 37).

The curves are evaluated with the notebook's own physics code, extracted
directly from the executed .ipynb so nothing is re-derived by hand:

  * Analytical EDFF            -> analytical_edff_flux(omega, b)
  * Numerical finite-size EPA  -> finite_size_epa_flux(omega, b, delta_R, a)
    (the direct, pre-surrogate integrator, at the mean nuclear parameters
    DELTA_R_MEAN_FM / A_MEAN_FM, exactly as in notebook Part 5, cell 14)

Both are evaluated at b = 20 fm on the same 31 rapidity points as
uncertainty_decomposition_publication.csv.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import kv


def load_notebook_physics(notebook_path: Path):
    nb = json.loads(notebook_path.read_text())
    sources = ["".join(cell.get("source", [])) for cell in nb["cells"]]

    namespace: dict = {
        "dataclass": dataclass,
        "asdict": asdict,
        "json": json,
        "Path": Path,
        "MODE": "publication",
        "OUTPUT_DIR": Path("/tmp"),
        "np": np,
        "pd": pd,
        "leggauss": leggauss,
        "kv": kv,
        "display": lambda *a, **k: None,
    }
    import math

    namespace["math"] = math

    # Cell 3: RunConfig / CFG (publication). Cell 5: constants + helpers.
    # Cell 7: EDFF + finite-size EPA integrator.
    for idx in (3, 5, 7):
        exec(compile(sources[idx], f"notebook cell {idx}", "exec"), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("uncertainty_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    ns = load_notebook_physics(args.notebook)
    cfg = ns["CFG"]
    print(
        "Quadrature:",
        f"n_r={cfg.n_r}, n_phi={cfg.n_phi}, n_z={cfg.n_z}, "
        f"n_norm={cfg.n_norm}, tail_widths={cfg.tail_widths}",
    )

    frame = pd.read_csv(args.uncertainty_csv).sort_values("y")
    y = frame["y"].to_numpy()
    omega = np.asarray(ns["rapidity_to_omega"](y), dtype=float)

    edff = np.asarray(ns["analytical_edff_flux"](omega, 20.0), dtype=float)
    finite = np.array(
        [
            ns["finite_size_epa_flux"](
                float(o), 20.0, ns["DELTA_R_MEAN_FM"], ns["A_MEAN_FM"]
            )
            for o in omega
        ]
    )

    out = pd.DataFrame({"y": y, "edff_flux": edff, "finite_epa_flux": finite})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    ratio = finite / edff
    rel_vs_surrogate = finite / frame["central"].to_numpy() - 1.0
    print(f"Wrote {args.output_csv} ({len(out)} points)")
    print(f"finite/EDFF ratio: {ratio.min():.4f} to {ratio.max():.4f}")
    print(
        "finite vs surrogate central: "
        f"min {rel_vs_surrogate.min() * 100:+.3f}%, "
        f"max {rel_vs_surrogate.max() * 100:+.3f}%"
    )


if __name__ == "__main__":
    main()

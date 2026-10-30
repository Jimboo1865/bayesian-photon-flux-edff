#!/usr/bin/env python3
"""
Extract the figures from the executed BNN notebook into a Desktop folder.

The notebook run completed, but its cells end with plt.show(), which in
Jupyter closes each figure right after displaying it - so the runner
script's savefig hook found no open figures and saved nothing. The
figures themselves are safe: each plotting cell stored its rendered
figure inside the executed notebook as a PNG output.

This script pulls those embedded PNGs out (stdlib only, any python3):

    ~/Desktop/BNN Figures/            all 13 figures
    ~/Desktop/BNN Figures/Overleaf/   the 11 manuscript candidates
                                      (everything except the two
                                      diagnostic plots: the Part 5
                                      direct-EPA sanity curves and the
                                      Part 8 training-target histogram)

Run:  python3 ~/Documents/"BNN fix"/extract_bnn_figures.py
"""

import base64
import json
import re
from pathlib import Path

NB = (
    Path.home()
    / "Downloads"
    / "bayesian-photon-flux-resubmission-sobol-edff.ipynb"
)
OUT = Path.home() / "Desktop" / "BNN Figures"
OVERLEAF_DIR = OUT / "Overleaf"

# (part, cell index) pairs judged manuscript candidates, by content.
OVERLEAF = {
    (5, 14),    # flux at b=20 fm + finite-size correction
    (5, 15),    # numerical EPA vs analytical EDFF, 3 rapidities
    (9, 25),    # training history
    (10, 28),   # test-set parity + residual distribution
    (10, 29),   # residuals across the test domain
    (10, 30),   # held-out residual map
    (11, 33),   # pull distribution (calibration)
    (12, 36),   # physical-uncertainty validation
    (12, 37),   # separated uncertainty sources at b=20 fm
    (13, 39),   # Sobol sample-size convergence (7 sizes)
    (14, 43),   # integrated flux + analysis-level validation
}


def main():
    nb = json.loads(NB.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)

    part = 0
    total = 0
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "markdown":
            m = re.match(r"## (\d+)\.", "".join(cell.get("source", [])).strip())
            if m:
                part = int(m.group(1))
            continue
        if cell.get("cell_type") != "code":
            continue
        n = 0
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" not in data:
                continue
            n += 1
            name = f"fig_part{part:02d}_cell{idx:02d}_n{n}.png"
            blob = base64.b64decode(data["image/png"])
            (OUT / name).write_bytes(blob)
            total += 1
            tag = ""
            if (part, idx) in OVERLEAF:
                (OVERLEAF_DIR / name).write_bytes(blob)
                tag = "  -> Overleaf/"
            print(f"  {name}{tag}")

    print(f"\n{total} figures in {OUT}")
    print(f"{len(OVERLEAF)} of them also in {OVERLEAF_DIR}")


if __name__ == "__main__":
    main()

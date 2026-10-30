#!/usr/bin/env python3
"""
Run the BNN photon-flux notebook end-to-end at Part-13 trial counts and
save every figure it generates.

What this does:
  1. Patches the notebook in place (idempotent, safe to re-run):
       - publication n_train: 8192 -> 65536 (matches Part_13_file.py)
       - publication sample_size_study_sizes:
             (1024, 2048, 4096, 8192)
          -> (1024, 2048, 4096, 8192, 16384, 32768, 65536)
       - appends a figure-saving snippet to every plotting cell, so all
         figures land in bnn_flux_outputs/ as PNG + PDF
         (the notebook itself never calls savefig).
  2. Registers a 'bnn-run' Jupyter kernel pointing at the Python that
     launches this script (the notebook's stored 'python3' kernel is a
     stale kernelspec pointing at a deleted .venv).
  3. Executes the patched notebook headlessly with jupyter nbconvert
     (no per-cell timeout). All other outputs (CSVs, model package) go to
     bnn_flux_outputs/ next to the notebook, as usual.

Run it with the same Python that runs your Jupyter, e.g.:
    conda run -n base python run_bnn_full.py
or, if that Python is already active:
    python run_bnn_full.py

A backup of the original notebook was saved next to it as
bayesian-photon-flux-resubmission-sobol-edff.BACKUP-2026-10-20.ipynb
"""

import json
import re
import subprocess
import sys
from pathlib import Path

NB = (
    Path.home()
    / "Downloads"
    / "bayesian-photon-flux-resubmission-sobol-edff.ipynb"
)
MARKER = "# [auto] save all figures"

CONFIG_REPLACEMENTS = [
    (
        "        n_train=8192,",
        "        n_train=65536,",
    ),
    (
        "        sample_size_study_sizes=(1024, 2048, 4096, 8192),",
        "        sample_size_study_sizes="
        "(1024, 2048, 4096, 8192, 16384, 32768, 65536),",
    ),
    # Repair: the first copy of this script sent to the Mac dropped the
    # trailing comma on the line above, leaving a syntactically broken
    # notebook. Fix that exact broken line if present.
    (
        "        sample_size_study_sizes="
        "(1024, 2048, 4096, 8192, 16384, 32768, 65536)\n",
        "        sample_size_study_sizes="
        "(1024, 2048, 4096, 8192, 16384, 32768, 65536),\n",
    ),
]


def cell_source(cell):
    return "".join(cell.get("source", []))


def set_cell_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def patch_notebook(nb):
    cells = nb["cells"]

    # --- 1. Config: match Part 13 trial counts -------------------------
    changed = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        new = src
        for old, replacement in CONFIG_REPLACEMENTS:
            if old in new:
                new = new.replace(old, replacement)
        if new != src:
            set_cell_source(cell, new)
            changed += 1
    print(f"Config cells updated: {changed} (0 is fine if already patched)")

    # --- 2. Figure-saving snippets -------------------------------------
    part = 0
    patched = 0
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            m = re.match(r"## (\d+)\.", cell_source(cell).strip())
            if m:
                part = int(m.group(1))
            continue
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if "plt." not in src or MARKER in src:
            continue
        snippet = (
            f"\n\n{MARKER}\n"
            "for _fignum in plt.get_fignums():\n"
            "    _fig = plt.figure(_fignum)\n"
            f'    _fig.savefig(OUTPUT_DIR / "fig_part{part:02d}'
            f'_cell{idx:02d}_n{{_fignum}}.png", '
            'dpi=200, bbox_inches="tight")\n'
            f'    _fig.savefig(OUTPUT_DIR / "fig_part{part:02d}'
            f'_cell{idx:02d}_n{{_fignum}}.pdf", '
            'bbox_inches="tight")\n'
        )
        set_cell_source(cell, src.rstrip("\n") + snippet)
        patched += 1
    print(f"Plotting cells given figure-saving: {patched}")


def ensure_kernel_spec():
    """Register a kernelspec that points at THIS Python interpreter.

    The notebook's stored 'python3' kernel is a stale user kernelspec
    pointing at a deleted .venv, which is what killed the first run.
    A dedicated 'bnn-run' kernel that follows sys.executable avoids it:
    whichever environment you launch this script with becomes the kernel.
    """
    kernel_dir = (
        Path.home() / "Library" / "Jupyter" / "kernels" / "bnn-run"
    )
    kernel_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "BNN run (script python)",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(spec, indent=2))
    print(f"Kernel 'bnn-run' registered for: {sys.executable}")


def main():
    if not NB.exists():
        sys.exit(f"Notebook not found: {NB}")

    nb = json.loads(NB.read_text())
    patch_notebook(nb)
    NB.write_text(json.dumps(nb, indent=1))
    print(f"Notebook patched: {NB}")

    ensure_kernel_spec()

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=bnn-run",
        str(NB),
    ]
    print("\nExecuting notebook (this is the long part)...")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(NB.parent))
    if result.returncode != 0:
        sys.exit(
            "Headless execution failed. Open the notebook in Jupyter and "
            "Run All instead - the patch is already applied, so figures "
            "will still be saved into bnn_flux_outputs/."
        )

    out_dir = NB.parent / "bnn_flux_outputs"
    figures = sorted(out_dir.glob("fig_*.png"))
    print(f"\nDone. {len(figures)} figure PNGs in {out_dir}:")
    for fig in figures:
        print(" ", fig.name)


if __name__ == "__main__":
    main()

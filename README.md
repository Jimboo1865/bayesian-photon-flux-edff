# Bayesian photon-flux surrogate (BNN / EDFF)

Code for the Bayesian neural-network surrogate of the photon flux in
ultra-peripheral collisions: an MC-dropout network that learns the
finite-size EPA correction to the analytical EDFF photon flux, with
Sobol-sequence sampling, a separated uncertainty budget (epistemic vs.
nuclear parameters), and a sample-size convergence study.

## Contents

- `bayesian-photon-flux-resubmission-sobol-edff.ipynb` — the main
  notebook (17 parts): deterministic finite-size EPA integrator,
  analytical EDFF comparison, Sobol dataset generation, MC-dropout
  training, multidimensional validation, calibration, nuclear-parameter
  uncertainty decomposition, the Sobol sample-size study, and the
  reproducible model package.
- `run_bnn_full.py` — runs the notebook end-to-end, headlessly, at the
  publication settings (Part 13 study up to n_train = 65536), patches
  the publication configuration, and saves every figure the notebook
  produces into `bnn_flux_outputs/`.
- `extract_bnn_figures.py` — extracts the figures embedded in an
  executed notebook into PNG files (plus an `Overleaf/` subfolder with
  the manuscript candidates).
- `figures/` — code for the revised manuscript figures (see below).
- `bnn_flux_outputs/` — the publication CSVs the figure scripts read.

## Environment

Python 3.12 with numpy, scipy, pandas, matplotlib, tensorflow
(Keras) and tqdm (see `requirements.txt`). The reference run used a
conda environment (`bnn-tf`, conda-forge TensorFlow):

```
conda create -n bnn-tf python=3.12
conda activate bnn-tf
pip install -r requirements.txt
```

## Running

```
python run_bnn_full.py
```

or open the notebook and run all cells with `MODE = "publication"`.
A full publication run takes roughly an hour and writes its CSVs,
figures, and the trained model package to `bnn_flux_outputs/`.

## Manuscript figures

Figure 12 (separated uncertainty sources at b = 20 fm, with a
percent-error panel, the pre-surrogate numerical finite-size EPA, and
the analytical EDFF overlaid):

```
python figures/compute_fig12_reference_curves.py \
    bayesian-photon-flux-resubmission-sobol-edff.ipynb \
    bnn_flux_outputs/uncertainty_decomposition_publication.csv \
    bnn_flux_outputs/fig12_reference_curves.csv

python figures/make_fig12_cell37_percent_error.py \
    bnn_flux_outputs/uncertainty_decomposition_publication.csv \
    --reference-csv bnn_flux_outputs/fig12_reference_curves.csv \
    --output-prefix figures/fig_part12_cell37_with_percent_error
```

`compute_fig12_reference_curves.py` evaluates the reference curves
with the notebook's own physics code (extracted from the notebook at
runtime), at the publication quadrature settings.

Figure 13 (Sobol sample-size convergence) as a vector PDF:

```
python figures/make_fig13_cell39_pdf.py \
    bnn_flux_outputs/sobol_sample_size_study_publication.csv \
    --output-pdf figures/fig_part13_cell39_n1.pdf
```

## Notes

- The notebook's physics conventions (Woods–Saxon parameters,
  rapidity-to-photon-energy mapping) are documented in its Parts 2–5.
- Figure 12's surrogate line is drawn solid underneath the dashed EPA
  line, so both remain visible where they overlap, without relying on
  colour alone; all colours are high-contrast for print.

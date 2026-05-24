# Energy-Efficient Hybrid STAR-RIS

This repository contains the manuscript, simulation code, figures, and numerical
results for the study of energy-efficient hybrid active-passive STAR-RIS design
with sparse active amplification and gain control.

## Directory Structure

- `paper/`: LaTeX source and compiled PDF manuscript.
- `code/`: Reproducible Python simulation script.
- `figs/`: Manuscript figures in PNG and JPG formats.
- `results/`: Numerical results from the Monte Carlo experiments.
- `requirements.txt`: Python dependencies needed to run the simulation.

## Setup

Create and activate a Python environment, then install the required packages:

```bash
python -m pip install -r requirements.txt
```

The simulation uses NumPy and Matplotlib only.

## Reproducing the Results

Run the simulator from the project root:

```bash
python code/hybrid_star_ris_ee_se_repro.py
```

The default configuration uses 10000 Monte Carlo channel realizations with
integer seeds `1..10000` and uses the available CPU worker count.

The script writes:

- `results/hybrid_star_ris_results.json`
- `figs/fig_pareto_boundary.{png,jpg}`
- `figs/fig_design_rules.{png,jpg}`
- `figs/fig_noise_dynamic.{png,jpg}`
- `figs/fig_power_breakdown.{png,jpg}`

The system illustration `figs/systemEE3_clean.{png,jpg}` is a static manuscript
asset.

## Manuscript

The compiled manuscript is available at:

```text
paper/energy_efficient_hybrid_star_ris.pdf
```

To rebuild it, run `pdflatex` from the `paper/` directory on
`energy_efficient_hybrid_star_ris.tex`.

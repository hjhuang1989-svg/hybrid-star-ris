# Energy-Efficient Hybrid STAR-RIS

This package contains the revised manuscript, figures, and reproduction code.

## Contents

- `paper/energy_efficient_hybrid_star_ris.tex`: LaTeX source.
- `paper/energy_efficient_hybrid_star_ris.pdf`: compiled manuscript.
- `figs/`: figures used by the manuscript.
- `code/hybrid_star_ris_ee_se_repro.py`: reproduction script.
- `code/plot_hybrid_star_ris_eps.py`: Python script that redraws the manuscript EPS figures from JSON.
- `results/hybrid_star_ris_results.json`: generated numerical summary from the included script run.
- `requirements.txt`: Python dependencies.

## Latest edit

This revision reports the numerical figures from 10000 independent channel realizations per plotted MC point. Python generates the final JSON data and redraws the paper figures directly as EPS files from that JSON. Hardware sensitivity is shown as Fig. 3(a)/(b), power distribution is shown as Fig. 4(a)/(b), and the candidate-search trace is shown as Fig. 5. The code repository remains in the first-page author footnote.

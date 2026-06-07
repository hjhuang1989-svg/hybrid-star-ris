# Energy-Efficient Hybrid STAR-RIS

This package contains the revised manuscript, figures, and reproduction code.

## Contents

- `paper/energy_efficient_hybrid_star_ris.tex`: LaTeX source.
- `paper/energy_efficient_hybrid_star_ris.pdf`: compiled manuscript.
- `figs/`: figures used by the manuscript.
- `code/hybrid_star_ris_ee_se_repro.py`: reproduction script.
- `results/hybrid_star_ris_results.json`: generated numerical summary from the included script run.
- `requirements.txt`: Python dependencies.

## Latest edit

This revision keeps the two stronger scalar-model baselines--a reweighted-ell1 sparse active-RIS benchmark and a hybrid AO/SCA-refinement benchmark--and updates the numerical figures. Fig. 4 is sorted by descending total consumed power with total-power annotations on the bars. Two complexity-oriented figures are added: runtime versus the number of STAR-RIS elements and an EE-runtime tradeoff scatter plot at R_Sigma=12 bit/s/Hz. The code repository remains in the first-page author footnote. The separate IEEE-style system-model EPS with a blocked direct BS-user path remains included in `figs/fig1_ieee_blocked.eps`; the manuscript figure itself was not replaced.

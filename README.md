# Hybrid STAR-RIS submission package

Contents:
- `paper/hybrid_star_ris_ee_se_submission.tex`: revised LaTeX source
- `paper/hybrid_star_ris_ee_se_submission.pdf`: compiled PDF
- `code/hybrid_star_ris_ee_se_repro.py`: reproducible simulator with ES / TS / SS support
- `results/hybrid_star_ris_results.json`: updated numerical summary
- `figs/`: final figures used by the paper
- `CHANGELOG.md`: task-by-task modification record

To regenerate the numerical results and figures locally:

```bash
python code/hybrid_star_ris_ee_se_repro.py
```

To compile the paper:

```bash
cd paper
pdflatex -interaction=nonstopmode hybrid_star_ris_ee_se_submission.tex
pdflatex -interaction=nonstopmode hybrid_star_ris_ee_se_submission.tex
```

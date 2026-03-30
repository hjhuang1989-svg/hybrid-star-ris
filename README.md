# Hybrid STAR-RIS

Main files:
- `code/hybrid_star_ris_ee_se_repro.py`
- `paper/hybrid_star_ris_ee_se_submission0329v2.tex`
- `paper/hybrid_star_ris_ee_se_submission0329v2.pdf`
- `figs/`
- `results/`

Run the simulation:

```bash
python code/hybrid_star_ris_ee_se_repro.py
```

Compile the paper:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error hybrid_star_ris_ee_se_submission0329v2.tex
```

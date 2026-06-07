import json, importlib.util, sys
from pathlib import Path
base=Path('/mnt/data/work_current/energy_efficient_hybrid_star_ris')
code=base/'code'/'hybrid_star_ris_ee_se_repro.py'
spec=importlib.util.spec_from_file_location('sim', code)
sim=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=sim
spec.loader.exec_module(sim)
sim.style()
with open(base/'results'/'hybrid_star_ris_results.json', encoding='utf-8') as f:
    d=json.load(f)
fig_dir=base/'figs'
sim.plot_power_breakdown({12:d['power_breakdown_at_R12_es'],16:d['power_breakdown_at_R16_es']}, fig_dir)
# Keep timing lightweight and deterministic. It measures per-boundary-point time, not MC averaging time.
cfg=sim.SimConfig(num_mc=1, num_workers=1)
runtime_scaling=sim.measure_runtime_scaling(cfg, N_values=(12,24,36,48,60), Rtarget=12.0, repeats=3)
sim.plot_runtime_scaling(runtime_scaling, fig_dir)
sim.plot_ee_runtime_tradeoff(d['average_boundary_es'], runtime_scaling, fig_dir, N_ref=24, Rtarget=12.0)
d['runtime_scaling']=runtime_scaling
with open(base/'results'/'hybrid_star_ris_results.json','w',encoding='utf-8') as f:
    json.dump(d,f,indent=2)
print(json.dumps(runtime_scaling, indent=2))

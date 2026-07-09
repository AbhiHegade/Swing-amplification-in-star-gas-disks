#!/usr/bin/env python3
"""Recompute only the third row (red star) of Fig 4 and replot with updated styling."""
import json, os, sys, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

codedir = "../src/"
sys.path.append(os.path.abspath(codedir))
from paper_case_points import line_marker_size, POINTS_THIN_DISKS

# Import helpers from executed notebook context by loading notebook-defined functions
# via exec of relevant cells is fragile; import from notebook exports instead.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Load module by executing notebook utility cells minimally
import importlib.util
nb_path = Path(__file__).resolve().parent / "Maximum_Swing_Amplification_Factor.ipynb"
nb = json.loads(nb_path.read_text())
code = "\n".join("".join(c.get("source", [])) for c in nb["cells"][:8] if c["cell_type"] == "code")
ns = {"__name__": "__main__"}
exec(code, ns)

kcritvals = np.sort(1.0 / np.linspace(0.1, 10, 100))
tivals = np.linspace(-20, 18, 100)
cache = Path("max_amp_data_cache.pkl")

if cache.exists():
    max_amp_data = pickle.loads(cache.read_bytes())
    print("Loaded cached max_amp_data")
else:
    print("No cache found; computing all rows (slow)...")
    max_amp_data = ns["generate_max_swing_amplification_data"](
        kcritvals=kcritvals,
        tivals=tivals,
        R_values=(1.0, 0.7, 0.4),
    )
    cache.write_bytes(pickle.dumps(max_amp_data))

row_idx = 2  # third row / red star
rows = max_amp_data["rows"]
row = rows[row_idx]
Qs, Qg = POINTS_THIN_DISKS[1]
row["Qs"] = Qs
row["Qg"] = Qg
hg = row["hg"]; hs = row["hs"]
print(f"Recomputing row {row_idx}: Qs={Qs}, Qg={Qg}")

RvalQgQs = [[Rval, Qg, Qs] for Rval in max_amp_data["R_values"]]
    plotting_arr = ns["get_plotting_arr"](RvalQgQs, hg=hg, hs=hs, kcritvals=kcritvals, tivals=tivals)
    order = np.argsort(1.0 / kcritvals)
row["curves"] = []
for item in plotting_arr:
    params, maxvals = item
    Rval, Qg_check, Qs_check = params
    row["curves"].append({
        "R": Rval,
        "Qg": Qg_check,
        "Qs": Qs_check,
        "maxvals": np.asarray(maxvals)[order],
    })
rows[row_idx] = row
max_amp_data["rows"] = rows
cache.write_bytes(pickle.dumps(max_amp_data))

fig, axes = ns["plot_max_swing_amplification_from_data"](
    max_amp_data,
    R_colors=("red", "green", "blue"),
    figsize=(5.0, 8.8),
    save=True,
    xlim=(0.1, 10.1),
    filename="./Figures/max_swing_amplification.pdf",
)
plt.close(fig)
print("Wrote Figures/max_swing_amplification.pdf")

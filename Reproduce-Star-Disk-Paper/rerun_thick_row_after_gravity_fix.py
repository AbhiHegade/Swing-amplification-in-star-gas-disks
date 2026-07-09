#!/usr/bin/env python3
"""Recompute only the thick-disk row of Fig. 4 after the gravity convention fix."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
SRC = REPO.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO))

from paper_case_points import POINTS_THICK_DISKS
from plot_figures_round3 import plot_fig4_from_cache


def _load_max_amp_namespace() -> dict:
    nb_path = REPO / "Maximum_Swing_Amplification_Factor.ipynb"
    nb = json.loads(nb_path.read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"][:8]
        if cell.get("cell_type") == "code"
    )
    ns = {"__name__": "__main__", "np": np, "plt": plt}
    exec(code, ns)
    return ns


def main() -> None:
    ns = _load_max_amp_namespace()
    cache = REPO / "max_amp_data_cache.pkl"
    if not cache.exists():
        raise FileNotFoundError(f"Missing {cache}")

    max_amp_data = pickle.loads(cache.read_bytes())
    kcritvals = np.sort(1.0 / np.linspace(0.1, 10, 100))
    tivals = np.linspace(-20, 18, 100)

    row_idx = 3  # fourth row / thick-disk blue star
    rows = max_amp_data["rows"]
    row = rows[row_idx]
    Qs, Qg = POINTS_THICK_DISKS[0]
    row["Qs"] = Qs
    row["Qg"] = Qg
    hg = row["hg"]
    hs = row["hs"]

    print(f"Recomputing Fig. 4 row {row_idx}: Qs={Qs}, Qg={Qg}, hs={hs}, hg={hg}", flush=True)
    RvalQgQs = [[Rval, Qg, Qs] for Rval in max_amp_data["R_values"]]
    plotting_arr = ns["get_plotting_arr"](
        RvalQgQs, hg=hg, hs=hs, kcritvals=kcritvals, tivals=tivals
    )

    order = np.argsort(1.0 / kcritvals)
    row["curves"] = []
    for params, maxvals in plotting_arr:
        Rval, Qg_check, Qs_check = params
        row["curves"].append(
            {
                "R": Rval,
                "Qg": Qg_check,
                "Qs": Qs_check,
                "maxvals": np.asarray(maxvals)[order],
            }
        )

    rows[row_idx] = row
    max_amp_data["rows"] = rows
    cache.write_bytes(pickle.dumps(max_amp_data))
    print(f"Updated {cache}", flush=True)
    plot_fig4_from_cache()


if __name__ == "__main__":
    main()

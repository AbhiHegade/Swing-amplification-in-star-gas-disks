#!/usr/bin/env python3
"""Diagnostic: max swing amplification vs lambda/lambda_crit at r=1 for several Q values."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import smplotlib  # noqa: F401

REPO = Path(__file__).resolve().parent
FIG_DIR = REPO / "Figures"
MS_FIG_DIR = Path(
    "/Users/chamilton/Documents/Science_Projects/Hegade/"
    "swing_amplification_gas_stellar_shearing_sheet (2)/Figures"
)
CACHE = REPO / "max_amp_Q_sensitivity_r1_cache.pkl"

Q_VALUES = [2.05, 2.1, 2.2, 2.35, 2.5, 2.8]
R_VAL = 1.0
KCRIVALS = np.sort(1.0 / np.linspace(0.1, 10, 100))
TIVALS = np.linspace(-20, 18, 100)


def _load_notebook_ns() -> dict:
    nb = json.loads((REPO / "Maximum_Swing_Amplification_Factor.ipynb").read_text())
    code = "\n".join(
        "".join(c.get("source", []))
        for c in nb["cells"][:8]
        if c.get("cell_type") == "code"
    )
    ns: dict = {"__name__": "__main__", "np": np, "plt": plt}
    exec(code, ns)
    return ns


def compute_curves(ns: dict, *, recompute: bool = False) -> dict:
    if CACHE.exists() and not recompute:
        print(f"Loading cache {CACHE}")
        return pickle.loads(CACHE.read_bytes())

    RvalQgQs = [[R_VAL, Q, Q] for Q in Q_VALUES]
    print(f"Computing r={R_VAL} for Q = {Q_VALUES} ...")
    plotting_arr = ns["get_plotting_arr"](
        RvalQgQs, hg=0.0, hs=0.0, kcritvals=KCRIVALS, tivals=TIVALS,
    )
    lambda_over_lcrit = 1.0 / KCRIVALS
    order = np.argsort(lambda_over_lcrit)
    curves = []
    for (params, maxvals), Q in zip(plotting_arr, Q_VALUES):
        Rval, Qg, Qs = params
        curves.append({
            "Q": Q, "R": Rval, "Qs": Qs, "Qg": Qg,
            "maxvals": np.asarray(maxvals)[order],
        })
    data = {
        "lambda_over_lcrit": lambda_over_lcrit[order],
        "curves": curves,
        "R": R_VAL,
        "Q_values": Q_VALUES,
    }
    CACHE.write_bytes(pickle.dumps(data))
    print(f"Saved cache -> {CACHE}")
    return data


def plot_curves(data: dict) -> Path:
    lambda_over_lcrit = data["lambda_over_lcrit"]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(data["curves"])))

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for color, curve in zip(cmap, data["curves"]):
        Q = curve["Q"]
        ax.plot(
            lambda_over_lcrit, curve["maxvals"],
            color=color, lw=1.2, label=rf"$Q_s=Q_g={Q:.2g}$",
        )
        peak_idx = int(np.argmax(curve["maxvals"]))
        peak_lam = lambda_over_lcrit[peak_idx]
        print(f"Q={Q:.2g}: peak at lambda/lambda_crit = {peak_lam:.3f}, amp = {curve['maxvals'][peak_idx]:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 10.1)
    ax.set_xlabel(r"$\lambda/\lambda_{\rm crit}$", fontsize=14)
    ax.set_ylabel("max. amplification", fontsize=14)
    ax.set_title(rf"Maximum swing amplification at $r={data['R']}$ (thin disk)", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=12, prop={"weight": "normal"})
    ax.tick_params(labelsize=12)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=6)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "max_amp_Q_sensitivity_r1.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    MS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(out, MS_FIG_DIR / out.name)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    recompute = "--recompute" in sys.argv
    ns = _load_notebook_ns()
    data = compute_curves(ns, recompute=recompute)
    plot_curves(data)

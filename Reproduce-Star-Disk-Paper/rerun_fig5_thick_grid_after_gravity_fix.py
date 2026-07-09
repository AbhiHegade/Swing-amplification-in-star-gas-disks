#!/usr/bin/env python3
"""Recompute the thick-disk Fig. 5 panel after the gravity convention fix.

This is intentionally resumable: partial results are saved after each completed
grid point so a long local run can be restarted without losing progress.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import smplotlib  # noqa: F401
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent
SRC = REPO.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO))

from axisymmetric_stability import get_stability_boundary_thick_disk, test_stability_thick_disks
from maximum_amplification_grid import get_max_amplification

R = 1.0
HG = 0.87
HS = 0.4
N_GRID = 100
N_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))

OUT_DIR = REPO / "max_amplification_outputs"
FIG_DIR = REPO / "Figures"
MS_FIG_DIR = Path(
    "/Users/chamilton/Documents/Science_Projects/Hegade/"
    "swing_amplification_gas_stellar_shearing_sheet (2)/Figures"
)

PARTIAL = OUT_DIR / f"{R}_{HG}_{HS}.partial.npz"
FINAL = OUT_DIR / f"{R}_{HG}_{HS}.npz"
FIG = FIG_DIR / f"max_amplification_grid_{R}_{HG}_{HS}.pdf"


def _grid_setup():
    inv_q_axis = np.linspace(0.01, 2.0, N_GRID)
    qg_vec = 1.0 / inv_q_axis
    qs_vec = 1.0 / inv_q_axis
    kcritvals = np.sort(1.0 / np.linspace(0.5, 5, 100))
    tivals = np.linspace(-20, 18, 100)
    return inv_q_axis, qg_vec, qs_vec, kcritvals, tivals


def _load_or_initialize():
    inv_q_axis, qg_vec, qs_vec, kcritvals, tivals = _grid_setup()
    if PARTIAL.exists():
        data = np.load(PARTIAL)
        arr = data["max_amplification_arr"]
        print(f"Resuming from {PARTIAL}: {np.isfinite(arr).sum()} completed", flush=True)
    else:
        arr = np.full((N_GRID, N_GRID), np.nan)
        print(f"Starting new {N_GRID}x{N_GRID} thick-grid run", flush=True)
    return inv_q_axis, qg_vec, qs_vec, kcritvals, tivals, arr


def _save(path: Path, inv_q_axis, qg_vec, qs_vec, arr):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        inv_q_axis=inv_q_axis,
        Qg_vec=qg_vec,
        Qs_vec=qs_vec,
        max_amplification_arr=arr,
        hg=HG,
        hs=HS,
        R=R,
    )


def _task(args):
    i, j, qg, qs, kcritvals, tivals = args
    stable = test_stability_thick_disks(R, qg, qs, HG, HS)
    if not stable:
        return i, j, np.nan, False
    val = get_max_amplification(R, qg, qs, HG, HS, kcritvals, tivals)
    return i, j, val, True


def _auto_log_levels(values, n_levels=12, min_level=1.0):
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    vmin = max(min_level, np.nanmin(finite))
    vmax = np.nanmax(finite)
    decades = np.logspace(np.floor(np.log10(vmin)), np.ceil(np.log10(vmax)), n_levels)
    return np.unique(np.asarray([x for x in decades if x >= vmin and x <= vmax] + [vmax]))


def _plot(inv_q_axis, qg_vec, qs_vec, arr):
    inv_qg_sorted = 1.0 / qg_vec
    inv_qs_sorted = 1.0 / qs_vec
    inv_qg_fine = np.linspace(inv_qg_sorted.min(), inv_qg_sorted.max(), 300)
    inv_qs_fine = np.linspace(inv_qs_sorted.min(), inv_qs_sorted.max(), 300)

    inv_qg_mesh, inv_qs_mesh = np.meshgrid(inv_qg_sorted, inv_qs_sorted, indexing="ij")
    mask = np.isfinite(arr)
    points = np.column_stack([inv_qg_mesh[mask], inv_qs_mesh[mask]])
    values = arr[mask]
    if len(values) == 0:
        raise RuntimeError("No finite values to plot")

    inv_qg_fine_mesh, inv_qs_fine_mesh = np.meshgrid(inv_qg_fine, inv_qs_fine, indexing="ij")
    fine = griddata(points, values, (inv_qg_fine_mesh, inv_qs_fine_mesh), method="linear")
    fine = np.ma.masked_invalid(fine)
    fine = np.ma.masked_where(fine <= 0.0, fine)
    levels = _auto_log_levels(fine.compressed())

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    contf = ax.contourf(
        inv_qs_fine_mesh,
        inv_qg_fine_mesh,
        fine,
        levels=levels,
        cmap="viridis",
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        extend="both",
    )
    ax.contour(
        inv_qs_fine_mesh,
        inv_qg_fine_mesh,
        fine,
        levels=levels,
        colors="black",
        linewidths=0.45,
        alpha=0.35,
    )

    z_stability = get_stability_boundary_thick_disk(R, qg_vec, qs_vec, HG, HS)
    ax.contour(
        1.0 / qs_vec,
        1.0 / qg_vec,
        gaussian_filter(z_stability, sigma=1.0),
        levels=[0.5],
        colors=["black"],
        linewidths=1.2,
        linestyles="dashed",
        zorder=20,
    )

    cbar = fig.colorbar(contf, ax=ax, ticks=levels[::2])
    cbar.set_label(r"max. amplification")
    cbar.ax.set_yticklabels([rf"${tick:g}$" for tick in levels[::2]])

    ax.set_xlabel(r"$1/Q_{\rm s}$")
    ax.set_ylabel(r"$1/Q_{\rm g}$")
    ax.set_title(rf"$r={R}$, $h_s={HS}$, $h_g={HG}$", fontsize=14)
    ax.text(0.12, 0.90, "unstable", transform=ax.transAxes, ha="center", va="center", fontsize=14)
    ax.legend(
        handles=[Line2D([0], [0], color="black", lw=1.2, ls="dashed", label="stability boundary")],
        loc="upper right",
        frameon=False,
        fontsize=14,
    )
    ax.tick_params(which="major", direction="in", top=True, right=True, length=6)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(FIG, MS_FIG_DIR / FIG.name)
    print(f"Wrote {FIG}", flush=True)


def main():
    inv_q_axis, qg_vec, qs_vec, kcritvals, tivals, arr = _load_or_initialize()
    tasks = []
    for i, qg in enumerate(qg_vec):
        for j, qs in enumerate(qs_vec):
            if not np.isfinite(arr[i, j]):
                tasks.append((i, j, qg, qs, kcritvals, tivals))

    print(f"Pending grid points: {len(tasks)}; workers: {N_WORKERS}", flush=True)
    if tasks:
        start = time.perf_counter()
        completed = int(np.isfinite(arr).sum())
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(_task, task) for task in tasks]
            for n, fut in enumerate(as_completed(futures), start=1):
                i, j, value, stable = fut.result()
                arr[i, j] = value
                completed += 1
                if n == 1 or n % 10 == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"completed {completed}/{N_GRID*N_GRID} "
                        f"(new {n}/{len(tasks)}), latest=({i},{j}), stable={stable}, "
                        f"elapsed={elapsed/60:.1f} min",
                        flush=True,
                    )
                    _save(PARTIAL, inv_q_axis, qg_vec, qs_vec, arr)
        _save(PARTIAL, inv_q_axis, qg_vec, qs_vec, arr)

    _save(FINAL, inv_q_axis, qg_vec, qs_vec, arr)
    _plot(inv_q_axis, qg_vec, qs_vec, arr)
    print("Done", flush=True)


if __name__ == "__main__":
    main()

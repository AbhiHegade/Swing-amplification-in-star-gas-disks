#!/usr/bin/env python3
"""Round-3 figure tweaks: Fig2 layout/units, Fig3 long-time panels, Fig4 plot-only."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import smplotlib  # noqa: F401 — activates Hershey/smplot style for all figures
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parent
SRC = REPO.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO))

from paper_case_points import POINTS_THIN_DISKS, POINTS_THICK_DISKS

FIG_DIR = REPO / "Figures"
MS_FIG_DIR = Path(
    "/Users/chamilton/Documents/Science_Projects/Hegade/"
    "swing_amplification_gas_stellar_shearing_sheet (2)/Figures"
)
CACHE = REPO / "max_amp_data_cache.pkl"
WAKE_CACHE = REPO / "point_mass_wake_cache.pkl"
FONT_SIZE = 14


def _exec_notebook_cells(nb_path: Path, max_cell: int) -> dict:
    nb = json.loads(nb_path.read_text())
    ns: dict = {"__name__": "__main__", "np": np, "plt": plt}
    for cell in nb["cells"][: max_cell + 1]:
        if cell.get("cell_type") == "code":
            exec("".join(cell.get("source", [])), ns)
    return ns


def plot_point_mass_wake_summary_v2(
    wake_stars, wake_gas, *, x_grid, y_grid, lambda_crit, cloud_mass, delta,
    x_cuts=(0.4, 0.2, 0.0), cut_colors=("red", "green", "blue"),
    y_window=(-1.0, 1.0), x_window=(-0.5, 0.5), color_percentile=99.0,
    n_filled_levels=41, cmap="RdBu_r", filename=None, save=False,
):
    unit_area = np.pi * delta**2
    # Unit string inside a single mathtext expression (like \lambda_{\rm crit} elsewhere).
    mass_unit = r"M/(\pi\Delta^2)"
    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)
    X_over_lcrit = x_grid / lambda_crit
    Y_over_lcrit = y_grid / lambda_crit
    Z_stars = np.asarray(wake_stars) * unit_area / cloud_mass
    Z_gas = np.asarray(wake_gas) * unit_area / cloud_mass
    mask_x = (X_over_lcrit >= x_window[0]) & (X_over_lcrit <= x_window[1])
    mask_y = (Y_over_lcrit >= y_window[0]) & (Y_over_lcrit <= y_window[1])
    Xc = X_over_lcrit[mask_x]
    Yc = Y_over_lcrit[mask_y]
    Zs_c = Z_stars[np.ix_(mask_x, mask_y)]
    Zg_c = Z_gas[np.ix_(mask_x, mask_y)]
    vmin, vmax = -3.0, 3.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    levels = np.linspace(vmin, vmax, n_filled_levels)

    fig = plt.figure(figsize=(5.6, 7.0))
    gs = GridSpec(3, 2, width_ratios=[1.0, 0.045], height_ratios=[1.0, 1.0, 1.65],
                  hspace=0.08, wspace=0.10, figure=fig)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[1, 0], sharex=ax_s, sharey=ax_s)
    ax_cut = fig.add_subplot(gs[2, 0], sharex=ax_s)
    cax_s = fig.add_subplot(gs[0, 1])
    cax_g = fig.add_subplot(gs[1, 1])

    cf_s = ax_s.contourf(Yc, Xc, Zs_c, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax_s.set_xlim(y_window[1], y_window[0])
    ax_s.set_ylim(x_window[0], x_window[1])
    ax_s.set_aspect("equal", adjustable="box")
    ax_s.set_ylabel(r"$x/\lambda_{\rm crit}$")
    ax_s.text(0.03, 0.97, r"(a)", transform=ax_s.transAxes, ha="left", va="top", fontsize=14)

    cf_g = ax_g.contourf(Yc, Xc, Zg_c, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax_g.set_xlim(y_window[1], y_window[0])
    ax_g.set_ylim(x_window[0], x_window[1])
    ax_g.set_aspect("equal", adjustable="box")
    ax_g.set_ylabel(r"$x/\lambda_{\rm crit}$")
    ax_g.text(0.03, 0.97, r"(b)", transform=ax_g.transAxes, ha="left", va="top", fontsize=14)

    plt.setp(ax_s.get_xticklabels(), visible=False)
    plt.setp(ax_g.get_xticklabels(), visible=False)

    cbar_fmt = FormatStrFormatter("%.1f")
    for cax, mappable, comp in [
        (cax_s, cf_s, r"\rm s"),
        (cax_g, cf_g, r"\rm g"),
    ]:
        cbar = fig.colorbar(mappable, cax=cax, format=cbar_fmt)
        cbar.set_label(rf"$\delta\Sigma_{{{comp}}}/[{mass_unit}]$")
        cbar.ax.tick_params(direction="in")

    for x_cut, color in zip(x_cuts, cut_colors):
        ix = np.argmin(np.abs(X_over_lcrit - x_cut))
        y_cut = Y_over_lcrit[mask_y]
        ax_cut.plot(y_cut, Z_stars[ix, mask_y], color=color, ls="solid", lw=1.4)
        ax_cut.plot(y_cut, Z_gas[ix, mask_y], color=color, ls="dashed", lw=1.4)
    ax_cut.axhline(0.0, color="black", lw=0.6)
    ax_cut.set_xlim(y_window[1], y_window[0])
    ax_cut.set_xlabel(r"$y/\lambda_{\rm crit}$")
    ax_cut.set_ylabel(rf"$\delta\Sigma_i/[{mass_unit}]$")
    ax_cut.text(0.03, 0.97, r"(c)", transform=ax_cut.transAxes, ha="left", va="top", fontsize=14)
    ax_cut.set_xticks([1.0, 0.5, 0.0, -0.5, -1.0])
    ax_cut.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_cut.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax_cut.tick_params(which="major", direction="in", top=True, right=True, length=6.0)
    ax_cut.tick_params(which="minor", direction="in", top=True, right=True, length=3.5)
    leg = ax_cut.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.6, ls="solid", label="stars"),
            Line2D([0], [0], color="black", lw=1.6, ls="dashed", label="gas"),
        ],
        loc="upper left", bbox_to_anchor=(1.02, 0.99), frameon=False,
    )
    fig.canvas.draw()
    leg_bbox = leg.get_window_extent().transformed(ax_cut.transAxes.inverted())
    y_text = leg_bbox.y0 - 0.02
    for x_cut, color in zip(x_cuts, cut_colors):
        ax_cut.text(
            leg_bbox.x0, y_text,
            rf"$x/\lambda_{{\rm crit}} = {x_cut:.1f}$",
            color=color, transform=ax_cut.transAxes, ha="left", va="top", clip_on=False,
        )
        y_text -= 0.07
    fig.subplots_adjust(left=0.16, right=0.90, bottom=0.08, top=0.98)
    fig.canvas.draw()
    pos_s = ax_s.get_position()
    pos_cut = ax_cut.get_position()
    ax_cut.set_position([pos_s.x0, pos_cut.y0, pos_s.width, pos_cut.height])
    if save and filename:
        fig.savefig(filename, bbox_inches="tight", dpi=600)
    return fig


def regenerate_fig2(*, use_cache: bool = True, save_cache: bool = True):
    if use_cache and WAKE_CACHE.exists():
        data = pickle.loads(WAKE_CACHE.read_bytes())
        ns = data
        print(f"Loaded wake cache from {WAKE_CACHE}")
    else:
        ns = _exec_notebook_cells(REPO / "Response_to_point_mass_thin_disk_filled_plots.ipynb", 22)
        if save_cache:
            WAKE_CACHE.write_bytes(pickle.dumps({
                "wake_stars": ns["wake_stars"],
                "wake_gas": ns["wake_gas"],
                "x_values": ns["x_values"],
                "y_values": ns["y_values"],
                "LAMBDA_CRIT": ns["LAMBDA_CRIT"],
                "CLOUD_MASS": ns["CLOUD_MASS"],
                "DELTA": ns["DELTA"],
            }))
            print(f"Saved wake cache -> {WAKE_CACHE}")
    out = FIG_DIR / "point_mass_wake_summary_thin_disk.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_point_mass_wake_summary_v2(
        ns["wake_stars"], ns["wake_gas"], x_grid=ns["x_values"], y_grid=ns["y_values"],
        lambda_crit=ns["LAMBDA_CRIT"], cloud_mass=ns["CLOUD_MASS"], delta=ns["DELTA"],
        filename=str(out), save=True,
    )
    plt.close()
    import shutil
    MS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, MS_FIG_DIR / out.name)
    print(f"Fig 2 -> {out}")


def regenerate_fig3():
    """Regenerate main Fig 3 (single-k swing amplification grid)."""
    import shutil
    ns = _exec_notebook_cells(REPO / "Swing_Amplification_Single_k.ipynb", 6)
    out = FIG_DIR / "swing-amplification-single-wave.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ns["fig"].savefig(out, bbox_inches="tight")
    plt.close(ns["fig"])
    MS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, MS_FIG_DIR / out.name)
    print(f"Fig 3 -> {out}")


def _plot_long_time_panel(
    ax, ns, *, rval, label, Qs, Qg, hs, hg, kyval, kxval, ti, tf, nt,
    models=(("star-gas", "solid"), ("two-fluid", "dotted")),
):
    """Plot selected model(s) on one long-time panel."""
    for model, ls in models:
        t_grid, Sigma_g, Sigma_s = ns["get_surface_densities"](
            R=rval, Qgas=Qg, Qstars=Qs, hg=hg, hs=hs,
            kxval=kxval, kyval=kyval, ti=ti, tf=tf, model=model, nt=nt,
        )
        x = t_grid / np.pi
        ax.plot(x, np.real(Sigma_s), color="red", lw=1.0, ls=ls)
        ax.plot(x, np.real(Sigma_g), color="blue", lw=1.0, ls=ls)
    ax.axhline(0.0, color="black", lw=0.4, alpha=0.4)
    if hs == 0.0 and hg == 0.0:
        title = rf"panel ({label}), $r={rval}$, $Q_s=Q_g={Qs:.2g}$"
    else:
        title = (
            rf"panel ({label}), $r={rval}$, $Q_s={Qs:.2g}$, $Q_g={Qg:.2g}$, "
            rf"$h_s={hs:.2g}$, $h_g={hg:.2g}$"
        )
    ax.set_title(title, fontsize=14)
    ax.text(0.03, 0.92, rf"({label})", transform=ax.transAxes, ha="left", va="top", fontsize=14)
    ax.tick_params(labelsize=14)


def regenerate_fig3_long_panels(*, twofluid_only: bool = True):
    """Long-time diagnostic for panels (g) Qs=Qg=2.2 and (j) thick-disk at r=1."""
    import shutil
    ns = _exec_notebook_cells(REPO / "Swing_Amplification_Single_k.ipynb", 5)
    kyval, kxval = 0.5, 0.0
    ti, tf, nt = -3.0 * np.pi / 2.0, 100.0 * np.pi, 10000
    qs_g, qg_g = POINTS_THIN_DISKS[1]
    qs_j, qg_j = POINTS_THICK_DISKS[0]
    panels = [
        {
            "label": "g", "rval": 1.0,
            "Qs": qs_g, "Qg": qg_g, "hs": 0.0, "hg": 0.0,
        },
        {
            "label": "j", "rval": 1.0,
            "Qs": qs_j, "Qg": qg_j, "hs": 0.4, "hg": 0.87,
        },
    ]
    models = (("two-fluid", "solid"),) if twofluid_only else (
        ("star-gas", "solid"), ("two-fluid", "dotted"),
    )
    model_tag = "twofluid" if twofluid_only else "both"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    for ax, panel in zip(axes, panels):
        print(
            f"Integrating panel ({panel['label']}), r={panel['rval']}, "
            f"Qs={panel['Qs']}, Qg={panel['Qg']} ({model_tag}) ...",
            flush=True,
        )
        _plot_long_time_panel(
            ax, ns, rval=panel["rval"], label=panel["label"],
            Qs=panel["Qs"], Qg=panel["Qg"], hs=panel["hs"], hg=panel["hg"],
            kyval=kyval, kxval=kxval, ti=ti, tf=tf, nt=nt, models=models,
        )
    axes[0].set_ylabel(r"$\delta \hat{\Sigma}_i/\delta\hat{\Sigma}_{\rm imp}$")
    axes[-1].set_xlabel(r"$\kappa t / \pi$")
    axes[-1].set_xlim(None, 100.0)

    axes[0].legend(
        handles=[
            Line2D([0], [0], color="red", lw=1.5, ls="solid", label="stars"),
            Line2D([0], [0], color="blue", lw=1.5, ls="solid", label="gas"),
        ],
        loc="upper right", frameon=False, fontsize=13, prop={"weight": "normal"},
    )
    if not twofluid_only:
        axes[0].legend(
            handles=[
                Line2D([0], [0], color="black", lw=1.5, ls="solid", label="star-gas"),
                Line2D([0], [0], color="black", lw=1.5, ls="dotted", label="two-fluid"),
            ],
            loc="upper left", frameon=False, fontsize=13, prop={"weight": "normal"},
        )

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.10, top=0.96, hspace=0.12)
    combined = FIG_DIR / f"swing-amplification-long-time-panels-g-j-{model_tag}.pdf"
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 3 long combined -> {combined}", flush=True)

    MS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(combined, MS_FIG_DIR / combined.name)
    # Also keep a stable alias for the two-fluid diagnostic.
    if twofluid_only:
        alias = FIG_DIR / "swing-amplification-long-time-panels-g-j.pdf"
        shutil.copy2(combined, alias)
        shutil.copy2(combined, MS_FIG_DIR / alias.name)
    for name in (
        "swing-amplification-long-time-panels-g-i.pdf",
        "swing-amplification-long-time-panel-i.pdf",
        "swing-amplification-long-time-panels-g-h.pdf",
        "swing-amplification-long-time-panel-h.pdf",
    ):
        for base in (FIG_DIR, MS_FIG_DIR):
            p = base / name
            if p.exists():
                p.unlink()
                print(f"Removed superseded {p.name}")


def plot_fig4_from_cache():
    if not CACHE.exists():
        raise FileNotFoundError(
            f"Missing cache {CACHE}. Run: python rerun_max_amp_row3.py (once) to bootstrap."
        )
    max_amp_data = pickle.loads(CACHE.read_bytes())
    lambda_over_lcrit = max_amp_data["lambda_over_lcrit"]
    rows = max_amp_data["rows"]
    R_values = max_amp_data["R_values"]
    R_color_map = dict(zip(R_values, ("red", "green", "blue")))
    fig, axes = plt.subplots(len(rows), 1, figsize=(5.0, 8.8), sharex=True)
    if len(rows) == 1:
        axes = np.array([axes])
    panel_labels = list("abcdefghijklmnopqrstuvwxyz")
    for i, row in enumerate(rows):
        ax = axes[i]
        Qs = row.get("Qs", np.nan)
        Qg = row.get("Qg", np.nan)
        hs, hg = row.get("hs", 0.0), row.get("hg", 0.0)
        for curve in row["curves"]:
            ax.plot(lambda_over_lcrit, curve["maxvals"], color=R_color_map[curve["R"]],
                    lw=1.0, label=rf"$r = {curve['R']}$")
        ax.set_yscale(row.get("yscale", "log"))
        ax.set_ylabel("max. amplification", fontsize=FONT_SIZE)
        ax.text(0.04, 0.92, rf"({panel_labels[i]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=FONT_SIZE)
        ms = 8 * (1.15 if row.get("marker") == "*" else 0.85 if row.get("marker") == "D" else 1.0)
        ax.plot(0.94, 0.9, marker=row["marker"], color=row["marker_color"],
                markerfacecolor=row["marker_color"], markeredgecolor=row["marker_color"],
                markersize=ms, linestyle="None", transform=ax.transAxes, clip_on=False)
        ax.set_title(rf"$Q_s={Qs:.3g},\ Q_g={Qg:.3g},\ h_s={hs:.2g},\ h_g={hg:.2g}$",
                     pad=4, fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        ax.tick_params(which="major", direction="in", top=True, right=True, length=6)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=3)
        if i == 0:
            ax.legend(loc="center right", frameon=False, fontsize=FONT_SIZE,
                      prop={"weight": "normal"})
    axes[-1].set_xlabel(r"$\lambda/\lambda_{\rm crit}$", fontsize=FONT_SIZE)
    axes[-1].set_xlim(0.1, 10.1)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.08, top=0.96, hspace=0.17)
    out = FIG_DIR / "max_swing_amplification.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    import shutil
    shutil.copy2(out, MS_FIG_DIR / out.name)
    print(f"Fig 4 (plot-only) -> {out}")


def update_main_tex_units():
    tex = MS_FIG_DIR.parent / "main.tex"
    t = tex.read_text()
    old_unit = r"$M/(0.1\lambda_{\rm crit})^2$"
    new_unit = r"$M/(\pi\Delta^2)$"
    old_caption = (
        "Panels (a) and (b) show the stellar and gaseous surface density perturbations respectively in units of\n"
        f"    {old_unit}, where $M$ is the cloud mass and $\\lambda_\\mathrm{{crit}}=2\\pi/k_\\mathrm{{crit}}$ (equation \\eqref{{eqn:kcrit}})."
    )
    new_caption = (
        "Panels (a) and (b) show the stellar and gaseous surface density perturbations respectively in units of\n"
        f"    {new_unit}, where $M$ is the cloud mass and $\\Delta=0.05\\,\\lambda_{{\\rm crit}}$ with $\\lambda_\\mathrm{{crit}}=2\\pi/k_\\mathrm{{crit}}$ (equation \\eqref{{eqn:kcrit}})."
    )
    old_body = (
        "In each case the units are $M/(0.1\\lambda_{\\rm crit})^2$."
    )
    new_body = (
        "In each case the units are $M/(\\pi\\Delta^2)$ with $\\Delta=0.05\\,\\lambda_{\\rm crit}$."
    )
    changed = False
    if old_caption in t:
        t = t.replace(old_caption, new_caption)
        changed = True
    if old_body in t:
        t = t.replace(old_body, new_body)
        changed = True
    if changed:
        tex.write_text(t)
        print("Updated Fig 2 unit strings in main.tex")
    else:
        print("Warning: main.tex unit strings not found (may already be updated)")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fig2"
    if cmd == "fig2":
        regenerate_fig2()
        update_main_tex_units()
    elif cmd == "fig3":
        regenerate_fig3()
    elif cmd == "fig4":
        plot_fig4_from_cache()
    elif cmd == "fig3long":
        regenerate_fig3_long_panels()
    else:
        raise SystemExit(f"Unknown command: {cmd}")

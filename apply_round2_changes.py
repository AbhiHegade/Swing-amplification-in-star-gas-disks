#!/usr/bin/env python3
"""Round-2 paper polish: Q=2.2, ky=0.5, Fig2/4 layout fixes."""
import json, re
from pathlib import Path

REPO = Path(__file__).resolve().parent
PAPER_NB = REPO / "Reproduce-Star-Disk-Paper"
SRC = REPO / "src"
TEX = Path("/Users/chamilton/Documents/Science_Projects/Hegade/swing_amplification_gas_stellar_shearing_sheet (2)/main.tex")

# --- paper_case_points.py ---
(SRC / "paper_case_points.py").write_text(
    (SRC / "paper_case_points.py").read_text().replace(
        "# index 1: red star    (Qs=2.0,  Qg=2.0)",
        "# index 1: red star    (Qs=2.2,  Qg=2.2)",
    ).replace(
        "[1.0 / 0.5, 1.0 / 0.5]",
        "[1.0 / 2.2, 1.0 / 2.2]",
    )
)

OLD_PT = "[1.0 / 0.5,  1.0 / 0.5]"
NEW_PT = "[1.0 / 2.2,  1.0 / 2.2]"


def patch_notebook(path: Path, transforms):
    nb = json.loads(path.read_text())
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = src
        for fn in transforms:
            new_src = fn(new_src)
        if new_src != src:
            cell["source"] = [new_src]
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1))
        print(f"Patched {path.name}")


def replace_points(s):
    return s.replace(OLD_PT, NEW_PT).replace("[1.0 / 0.5, 1.0 / 0.5]", "[1.0 / 2.2, 1.0 / 2.2]")


def patch_swing(s):
    s = replace_points(s)
    return s.replace("kyval = 1.0", "kyval = 0.5")


def patch_response(s):
    s = replace_points(s)
    if "def plot_point_mass_wake_summary" in s:
        s = s.replace("height_ratios=[1.0, 1.0, 1.05]", "height_ratios=[1.0, 1.0, 1.75]")
        s = s.replace("fontsize=12,\n            fontweight=\"bold\",", "fontsize=14,")
        s = s.replace(
            "    # Panel label, if needed\n    # ax_cut.text(\n    #     0.03,\n    #     0.90,\n    #     r\"(c)\",\n    #     transform=ax_cut.transAxes,\n    #     ha=\"left\",\n    #     va=\"top\",\n    # )\n",
            "    ax_cut.text(\n        0.03,\n        0.97,\n        r\"(c)\",\n        transform=ax_cut.transAxes,\n        ha=\"left\",\n        va=\"top\",\n        fontsize=14,\n    )\n",
        )
        if "ax_cut.set_aspect" not in s:
            s = s.replace(
                "    ax_cut.set_xlim(y_window[1], y_window[0])\n",
                "    ax_cut.set_xlim(y_window[1], y_window[0])\n    ax_cut.set_aspect(\"equal\", adjustable=\"box\")\n",
            )
    return s


def patch_max_amp(s):
    s = replace_points(s)
    if "def plot_max_swing_amplification_from_data" in s:
        s = s.replace("hspace=0.24,", "hspace=0.17,")
    return s


for nb in PAPER_NB.glob("*.ipynb"):
    if nb.name.startswith("."):
        continue
    transforms = [replace_points]
    if nb.name == "Swing_Amplification_Single_k.ipynb":
        transforms = [patch_swing]
    elif nb.name == "Response_to_point_mass_thin_disk_filled_plots.ipynb":
        transforms = [patch_response]
    elif nb.name == "Maximum_Swing_Amplification_Factor.ipynb":
        transforms = [patch_max_amp]
    patch_notebook(nb, transforms)

# main.tex
if TEX.exists():
    t = TEX.read_text()
    t = t.replace("Q_{\\rm s} = Q_{\\rm g}=2.0", "Q_{\\rm s} = Q_{\\rm g}=2.2")
    t = t.replace("Q_\\mathrm{s}=Q_\\mathrm{g}=2.0", "Q_\\mathrm{s}=Q_\\mathrm{g}=2.2")
    t = t.replace("$k_{y_{\\mathrm{c}}}/k_\\mathrm{crit}=1.0$", "$k_{y_{\\mathrm{c}}}/k_\\mathrm{crit}=0.5$")
    t = t.replace("$k_y/k_\\mathrm{crit} = 1.0$", "$k_y/k_\\mathrm{crit} = 0.5$")
    TEX.write_text(t)
    print("Patched main.tex")

print("Round-2 patches applied.")

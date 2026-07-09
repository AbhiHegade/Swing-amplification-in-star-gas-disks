#!/usr/bin/env python3
"""Patch reproduction notebooks for paper polish and optionally execute them."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent
PAPER_DIR = REPO / "Reproduce-Star-Disk-Paper"
SRC_DIR = REPO / "src"

OLD_STAR_POINT = "[1.0 / 0.4,  1.0 / 0.4]"
NEW_STAR_POINT = "[1.0 / 0.5,  1.0 / 0.5]"
OLD_STAR_POINT2 = "[1.0 / 0.4, 1.0 / 0.4]"


def patch_cell_source(src: str) -> str:
  src = src.replace(OLD_STAR_POINT, NEW_STAR_POINT)
  src = src.replace(OLD_STAR_POINT2, NEW_STAR_POINT)
  src = src.replace("kyval = 0.5", "kyval = 1.0")
  src = src.replace("TEXT_FONTSIZE = 12", "TEXT_FONTSIZE = 14")
  return src


def patch_rafikov_cell(src: str) -> str:
  src = patch_cell_source(src)
  if "from paper_case_points import" not in src:
    src = (
      "import sys\n"
      "sys.path.insert(0, '../src')\n"
      "from paper_case_points import (\n"
      "    POINTS_THIN_DISKS as points_thin_disks,\n"
      "    POINTS_THICK_DISKS as points_thick_disks,\n"
      "    THIN_DISK_MARKERS as thin_disk_marker,\n"
      "    THICK_DISK_MARKERS as thick_disk_marker,\n"
      "    scatter_marker_size,\n"
      ")\n\n"
      + src
    )

  # Remove hard-coded point arrays if present after import
  src = re.sub(
    r"points_thick_disks = np\.array\(\[[\s\S]*?\]\)\n\nthick_disk_marker = \['\*'\]\n\npoints_thin_disks = np\.array\(\[[\s\S]*?\]\)\n\nthin_disk_marker = \['D', '\*', 'o'\]\n\n",
    "",
    src,
  )

  src = src.replace(
    "    plt.scatter(1.0 / Qs, 1.0 / Qg, marker=marker, s=50, color='blue')",
    "    plt.scatter(1.0 / Qs, 1.0 / Qg, marker=marker, s=scatter_marker_size(marker), color='blue')",
  )
  src = src.replace(
    "    plt.scatter(1.0 / Qs, 1.0 / Qg, marker=marker, s=50, color='red')",
    "    plt.scatter(1.0 / Qs, 1.0 / Qg, marker=marker, s=scatter_marker_size(marker), color='red')",
  )
  return src


def patch_swing_cell(src: str) -> str:
  src = patch_cell_source(src)
  if "from paper_case_points import line_marker_size" not in src:
    src = (
      "import sys\n"
      "sys.path.insert(0, '../src')\n"
      "from paper_case_points import line_marker_size\n\n"
      + src
    )

  src = src.replace("markersize=10,", "markersize=line_marker_size(row[\"marker\"], 10),")

  if "ax.margins(x=0)" not in src:
    src = src.replace(
      "        ax.tick_params(labelsize=TEXT_FONTSIZE)",
      "        ax.tick_params(labelsize=TEXT_FONTSIZE)\n        ax.margins(x=0)",
    )
  return src


def patch_max_amp_cell(src: str) -> str:
  src = patch_cell_source(src)
  if "FONT_SIZE = 11" not in src:
    src = "FONT_SIZE = 11\n\n" + src
  if "from paper_case_points import line_marker_size" not in src:
    src = (
      "import sys\n"
      "sys.path.insert(0, '../src')\n"
      "from paper_case_points import line_marker_size\n\n"
      + src
    )

  src = src.replace("fontsize = 12,", "fontsize=FONT_SIZE,")
  src = src.replace("fontsize = 12", "fontsize=FONT_SIZE")
  src = src.replace("markersize=8,", "markersize=line_marker_size(row[\"marker\"], 8),")

  if "ax.tick_params(labelsize=FONT_SIZE)" not in src:
    src = src.replace(
      "        ax.tick_params(\n            which=\"major\",",
      "        ax.tick_params(labelsize=FONT_SIZE)\n        ax.tick_params(\n            which=\"major\",",
    )

  if "axes[-1].set_xlabel" in src and "fontsize=FONT_SIZE" not in src.split("axes[-1].set_xlabel")[1][:80]:
    src = src.replace(
      "    axes[-1].set_xlabel(r\"$\\lambda/\\lambda_{\\rm crit}$\")",
      "    axes[-1].set_xlabel(r\"$\\lambda/\\lambda_{\\rm crit}$\", fontsize=FONT_SIZE)",
    )
  return src


def patch_response_panel_labels(src: str) -> str:
  old = """        # Panel label, if needed
        # ax.text(
        #     0.03,
        #     0.90,
        #     panel_label,
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top","""

  new = """        ax.text(
            0.03,
            0.97,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        """

  return src.replace(old, new)


def patch_notebook(path: pathlib.Path) -> None:
  nb = json.loads(path.read_text())
  changed = False
  for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
      continue
    src = "".join(cell.get("source", []))
    new_src = src
    name = path.name
    if name == "Rafikov-Disperion.ipynb" and "plt.savefig(\"./Figures/Rafikov_dispersion.pdf\")" in src:
      new_src = patch_rafikov_cell(src)
    elif name == "Swing_Amplification_Single_k.ipynb" and "plt.savefig(\"./Figures/swing-amplification-single-wave.pdf\")" in src:
      new_src = patch_swing_cell(src)
    elif name == "Maximum_Swing_Amplification_Factor.ipynb" and "def plot_max_swing_amplification_from_data" in src:
      new_src = patch_max_amp_cell(src)
    elif name == "Response_to_point_mass_thin_disk_filled_plots.ipynb":
      if "points_thin_disks = np.array" in src and "Qgas" not in src:
        new_src = patch_cell_source(src)
      if "def plot_point_mass_wake_summary" in src:
        new_src = patch_response_panel_labels(new_src if new_src != src else src)
    else:
      new_src = patch_cell_source(src)

    if new_src != src:
      cell["source"] = [new_src]
      changed = True
  if changed:
    path.write_text(json.dumps(nb, indent=1))
    print(f"Patched {path.name}")


def main() -> int:
  for nb_path in sorted(PAPER_DIR.glob("*.ipynb")):
    patch_notebook(nb_path)

  if "--execute" in sys.argv:
    python = sys.executable
    for nb_name in [
      "Rafikov-Disperion.ipynb",
      "Response_to_point_mass_thin_disk_filled_plots.ipynb",
      "Swing_Amplification_Single_k.ipynb",
      "Maximum_Swing_Amplification_Factor.ipynb",
    ]:
      nb = PAPER_DIR / nb_name
      print(f"Executing {nb_name}...")
      subprocess.run(
        [
          python,
          "-m",
          "jupyter",
          "nbconvert",
          "--to",
          "notebook",
          "--execute",
          "--inplace",
          str(nb),
        ],
        cwd=PAPER_DIR,
        check=True,
      )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

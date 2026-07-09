#!/usr/bin/env python3
"""Patch Fig. 3's left annotation strip using the same smplotlib style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import smplotlib  # noqa: F401
from matplotlib.patches import Rectangle


LEFT_PAD_BP = 0.0
WIDTH_BP = 895.067059 + LEFT_PAD_BP
HEIGHT_BP = 555.299983


def add_rotated(ax, x, y, text, fontsize=14):
    ax.text(
        x,
        y,
        text,
        rotation=90,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


fig = plt.figure(figsize=(WIDTH_BP / 72.0, HEIGHT_BP / 72.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WIDTH_BP)
ax.set_ylim(0, HEIGHT_BP)
ax.axis("off")

# Clear only the original far-left Q-label/symbol strip.  The base PDF keeps the
# original first-column y-axis labels and tick labels.
ax.add_patch(
    Rectangle((0, 0), 50, 548, facecolor="white", edgecolor="none")
)

rows = [
    (484, r"$Q_s=1.43,\ Q_g=5$", "o", "red"),
    (354, r"$Q_s=5,\ Q_g=1.43$", "D", "red"),
    (224, r"$Q_s=2.2,\ Q_g=2.2$", "*", "red"),
    (94, r"$Q_s=1.35,\ Q_g=1.35$", "*", "blue"),
]

for y, q_label, marker, color in rows:
    add_rotated(ax, 14, y, q_label, fontsize=14)
    markersize = 6.8 if marker == "D" else 8 if marker != "*" else 9
    ax.plot(
        34,
        y,
        marker=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=markersize,
        linestyle="None",
    )

fig.savefig(
    "swing-amplification-single-wave-smplot-overlay.pdf",
    transparent=True,
)
plt.close(fig)

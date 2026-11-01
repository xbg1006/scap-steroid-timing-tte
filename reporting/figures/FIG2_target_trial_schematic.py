from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figure_config import set_pub_style, save_figure

set_pub_style()

fig, ax = plt.subplots(figsize=(7.2, 5.0))
fig.subplots_adjust(top=0.88, bottom=0.09, left=0.05, right=0.98)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(0.25, 7.32, "Sequential eligibility assessments after respiratory time zero",
        fontsize=9.0, fontweight="bold", ha="left")

# Single vlines call prevents matplotlib's color cycle from coloring each tick.
x0, x1 = 0.55, 6.25
ticks = np.linspace(x0, x1, 10)
ax.plot([x0, x1], [6.72, 6.72], linewidth=1.1)
ax.vlines(ticks, 6.57, 6.87, linewidth=0.85)
for x, h in zip(ticks, range(0, 20, 2)):
    ax.text(x, 6.34, str(h), ha="center", va="top", fontsize=7.3)
ax.text((x0+x1)/2, 6.03, "Hours after respiratory time zero",
        ha="center", fontsize=7.7)

ax.add_patch(FancyArrowPatch(
    (3.40, 5.80), (3.40, 5.35),
    arrowstyle="-|>", mutation_scale=9, linewidth=0.8
))
ax.text(3.58, 5.58, "At each eligible trial baseline",
        fontsize=7.7, va="center")

ax.add_patch(FancyBboxPatch(
    (1.62, 4.55), 3.56, 0.72,
    boxstyle="round,pad=0.015,rounding_size=0.035",
    linewidth=0.85, fill=False
))
ax.text(3.40, 4.91, "Clone each eligible person-trial\nto both strategies",
        ha="center", va="center", fontsize=8.25, fontweight="bold")

# Strategy boxes
bw, bh = 3.55, 1.72
left_xy = (0.35, 2.30)
right_xy = (4.50, 2.30)
for xy in (left_xy, right_xy):
    ax.add_patch(FancyBboxPatch(
        xy, bw, bh,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=0.85, fill=False
    ))

ax.add_patch(FancyArrowPatch(
    (2.55, 4.55), (2.12, 4.05),
    arrowstyle="-|>", mutation_scale=9, linewidth=0.8
))
ax.add_patch(FancyArrowPatch(
    (4.22, 4.55), (6.25, 4.05),
    arrowstyle="-|>", mutation_scale=9, linewidth=0.8
))

ax.text(2.12, 3.70, "Strategy 1: initiate within 6 h",
        ha="center", fontsize=8.2, fontweight="bold")
ax.text(2.12, 3.28, "0-<3 h: early initiation", ha="center", fontsize=7.25)
ax.text(2.12, 2.95, "3-<6 h: recent initiation", ha="center", fontsize=7.25)
ax.text(2.12, 2.62, "No initiation by 6 h -> artificial censoring",
        ha="center", fontsize=6.95)

ax.text(6.28, 3.70, "Strategy 0: defer for >=6 h",
        ha="center", fontsize=8.2, fontweight="bold")
ax.text(6.28, 3.28, "Remain untreated through the 6-h grace period",
        ha="center", fontsize=6.95)
ax.text(6.28, 2.95, "Initiation during grace period -> artificial censoring",
        ha="center", fontsize=6.8)
ax.text(6.28, 2.62, "Hourly time-updated censoring probabilities",
        ha="center", fontsize=7.0)

# Both strategies converge to a common outcome box.
ax.add_patch(FancyArrowPatch(
    (2.12, 2.29), (4.42, 1.66),
    arrowstyle="-|>", mutation_scale=8.5, linewidth=0.75
))
ax.add_patch(FancyArrowPatch(
    (6.28, 2.29), (5.60, 1.66),
    arrowstyle="-|>", mutation_scale=8.5, linewidth=0.75
))

ax.add_patch(FancyBboxPatch(
    (2.95, 0.72), 4.10, 0.92,
    boxstyle="round,pad=0.015,rounding_size=0.035",
    linewidth=0.85, fill=False
))
ax.text(5.00, 1.37, "Follow-up from trial baseline",
        ha="center", fontsize=8.0, fontweight="bold")
ax.text(5.00, 1.04,
        "MIMIC-IV: 28-day all-cause mortality",
        ha="center", fontsize=7.15)
ax.text(5.00, 0.82,
        "eICU: 28-day in-hospital death",
        ha="center", fontsize=7.15)

ax.text(
    0.35, 0.18,
    "Overlap target q(X)=e(X){1-e(X)}; artificial censoring adjusted with time-updated inverse-probability weights.",
    ha="left", va="bottom", fontsize=6.75
)

fig.suptitle("Sequential target-trial emulation of corticosteroid initiation timing",
             y=0.965, fontsize=10.8, fontweight="bold")
save_figure(fig, "Figure_2_Target_Trial_Schematic")
plt.close(fig)

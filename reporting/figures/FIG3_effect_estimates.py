from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_config import read_csv_any, set_pub_style, clean_axes, save_figure

set_pub_style()

m = read_csv_any("final_effect_estimates.csv").iloc[0]
comp = read_csv_any("E32_05_INFERENCE_STABILITY_COMPARISON.csv")

e_primary = comp.loc[
    comp["analysis"] == "E3_primary_unpenalized_refit_bootstrap"
].iloc[0]
e_fixed = comp.loc[
    comp["analysis"] == "E32_fixed_nuisance_cluster_bootstrap"
].iloc[0]

study_labels = ["MIMIC-IV primary", "eICU external"]
y = np.array([1, 0], dtype=float)

rd_point = np.array([
    float(m["risk_difference"]) * 100,
    float(e_primary["risk_difference_point"]) * 100,
])
rd_lo = np.array([
    float(m["rd_ci_low"]) * 100,
    float(e_fixed["rd_ci_low"]) * 100,
])
rd_hi = np.array([
    float(m["rd_ci_high"]) * 100,
    float(e_fixed["rd_ci_high"]) * 100,
])

rr_point = np.array([
    float(m["risk_ratio"]),
    float(e_primary["risk_ratio_point"]),
])
rr_lo = np.array([
    float(m["rr_ci_low"]),
    float(e_fixed["rr_ci_low"]),
])
rr_hi = np.array([
    float(m["rr_ci_high"]),
    float(e_fixed["rr_ci_high"]),
])

# A slightly taller but still compact canvas.
fig = plt.figure(figsize=(8.2, 3.00))
gs = GridSpec(
    1, 5, figure=fig,
    width_ratios=[1.28, 2.20, 1.72, 1.92, 1.58],
    wspace=0.22
)

# IMPORTANT:
# no figure-level footnote inside this graphic.
# Statistical interval interpretation belongs in the manuscript figure legend.
fig.subplots_adjust(top=0.78, bottom=0.29, left=0.03, right=0.995)

# ---------------------------------------------------------------------
# Study column
# ---------------------------------------------------------------------
ax_study = fig.add_subplot(gs[0,0])
ax_study.set_xlim(0,1)
ax_study.set_ylim(-0.45,1.45)
ax_study.axis("off")

ax_study.text(
    0.98, 1.48, "Cohort",
    ha="right", va="bottom",
    fontsize=8.0, fontweight="bold"
)
for yi, lab in zip(y, study_labels):
    ax_study.text(
        0.98, yi, lab,
        ha="right", va="center",
        fontsize=8.0
    )

# ---------------------------------------------------------------------
# Absolute effect forest
# ---------------------------------------------------------------------
ax_rd = fig.add_subplot(gs[0,1])

for i, yi in enumerate(y):
    ls = "-" if i == 0 else ":"
    marker = "o" if i == 0 else "s"

    ax_rd.hlines(
        yi, rd_lo[i], rd_hi[i],
        linewidth=1.20, linestyles=ls
    )
    ax_rd.plot(
        [rd_lo[i], rd_hi[i]], [yi, yi],
        "|", markersize=6.5
    )
    ax_rd.plot(
        rd_point[i], yi,
        marker=marker,
        linestyle="None",
        markersize=5.5
    )

ax_rd.axvline(0, linestyle="--", linewidth=0.75)
ax_rd.set_xlim(-3.5, 13.5)
ax_rd.set_ylim(-0.45, 1.45)
ax_rd.set_yticks([])

# SINGLE LINE LABEL — avoids collision with bottom area.
ax_rd.set_xlabel("Risk difference (percentage points)", labelpad=7)
ax_rd.set_title("Absolute effect", pad=5)
clean_axes(ax_rd)

# ---------------------------------------------------------------------
# Absolute effect numeric column
# ---------------------------------------------------------------------
ax_rdt = fig.add_subplot(gs[0,2])
ax_rdt.set_xlim(0,1)
ax_rdt.set_ylim(-0.45,1.45)
ax_rdt.axis("off")

ax_rdt.text(
    0.02, 1.48, "Estimate (95% interval)",
    ha="left", va="bottom",
    fontsize=7.8, fontweight="bold"
)

rd_text = [
    f"{rd_point[0]:.2f} ({rd_lo[0]:.2f} to {rd_hi[0]:.2f})",
    f"{rd_point[1]:.2f} ({rd_lo[1]:.2f} to {rd_hi[1]:.2f})",
]
for yi, txt in zip(y, rd_text):
    ax_rdt.text(
        0.02, yi, txt,
        ha="left", va="center",
        fontsize=7.5
    )

# ---------------------------------------------------------------------
# Relative effect forest
# ---------------------------------------------------------------------
ax_rr = fig.add_subplot(gs[0,3])

for i, yi in enumerate(y):
    ls = "-" if i == 0 else ":"
    marker = "o" if i == 0 else "s"

    ax_rr.hlines(
        yi, rr_lo[i], rr_hi[i],
        linewidth=1.20, linestyles=ls
    )
    ax_rr.plot(
        [rr_lo[i], rr_hi[i]], [yi, yi],
        "|", markersize=6.5
    )
    ax_rr.plot(
        rr_point[i], yi,
        marker=marker,
        linestyle="None",
        markersize=5.5
    )

ax_rr.axvline(1, linestyle="--", linewidth=0.75)
ax_rr.set_xlim(0.86, 1.40)
ax_rr.set_ylim(-0.45,1.45)
ax_rr.set_yticks([])
ax_rr.set_xlabel("Risk ratio", labelpad=7)
ax_rr.set_title("Relative effect", pad=5)
clean_axes(ax_rr)

# ---------------------------------------------------------------------
# Relative effect numeric column
# ---------------------------------------------------------------------
ax_rrt = fig.add_subplot(gs[0,4])
ax_rrt.set_xlim(0,1)
ax_rrt.set_ylim(-0.45,1.45)
ax_rrt.axis("off")

ax_rrt.text(
    0.02, 1.48, "Estimate (95% interval)",
    ha="left", va="bottom",
    fontsize=7.8, fontweight="bold"
)

rr_text = [
    f"{rr_point[0]:.2f} ({rr_lo[0]:.2f} to {rr_hi[0]:.2f})",
    f"{rr_point[1]:.2f} ({rr_lo[1]:.2f} to {rr_hi[1]:.2f})",
]
for yi, txt in zip(y, rr_text):
    ax_rrt.text(
        0.02, yi, txt,
        ha="left", va="center",
        fontsize=7.5
    )

# NOTE:
# Do NOT add fig.text() footnotes here.
# Put statistical interval details in the manuscript figure legend.

save_figure(fig, "Figure_3_Effect_Estimates")
plt.close(fig)

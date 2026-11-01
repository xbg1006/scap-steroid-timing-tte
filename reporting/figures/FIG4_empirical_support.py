from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt

from figure_config import read_csv_any, set_pub_style, clean_axes, panel_label, save_figure

set_pub_style()

mtrial = read_csv_any("sequential_trial_counts.csv")
etrial = read_csv_any("E2_03_SEQUENTIAL_TRIAL_COUNTS.csv")
mdef = read_csv_any("16V9H_FINAL_6H_DEFER_HOURLY_SUMMARY.csv")
mdef = mdef[mdef["population"] == "broad_primary"].copy()
edef = read_csv_any("E2_06_DEFER_HOURLY_SUMMARY.csv")
mfeas = read_csv_any("16V9_FINAL_6H_FEASIBILITY_SUMMARY.csv")
mfeas = mfeas[mfeas["population"] == "broad_primary"].iloc[0]
erec = read_csv_any("E2_08_RECENT_INITIATOR_SUMMARY.csv").iloc[0]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45))
fig.subplots_adjust(top=0.94, bottom=0.10, left=0.11, right=0.98, hspace=0.43, wspace=0.30)

# ---------------------------------------------------------------------
# A. Treatment support over trial baselines
# ---------------------------------------------------------------------
ax = axes[0,0]
panel_label(ax, "A")
ax.plot(mtrial["trial_hour"], 100*mtrial["initiate_within6h_rate"], marker="o", label="MIMIC-IV")
ax.plot(etrial["trial_hour"], 100*etrial["initiate_within6h_rate"], marker="s", label="eICU")
ax.set_xticks(range(0,20,2))
ax.set_xlabel("Trial baseline, h")
ax.set_ylabel("Initiation within 6 h, %")
ax.set_title("Observed treatment support", pad=5)
ax.legend(frameon=False)
clean_axes(ax)

# ---------------------------------------------------------------------
# B. Deferral censoring balance
# ---------------------------------------------------------------------
ax = axes[0,1]
panel_label(ax, "B")
ax.plot(mdef["interval_index"] + 1, mdef["max_abs_smd"], marker="o", label="MIMIC-IV")
ax.plot(edef["interval_index"] + 1, edef["max_abs_smd"], marker="s", label="eICU")
obsmax = max(float(mdef["max_abs_smd"].max()), float(edef["max_abs_smd"].max()))
ax.set_ylim(0, max(0.009, obsmax*1.45))
ax.set_xticks(range(1,7))
ax.set_xlabel("Censoring interval")
ax.set_ylabel("Maximum |SMD|")
ax.set_title("Deferral-strategy balance", pad=5)
ax.legend(frameon=False, loc="upper left")
ax.text(
    0.98, 0.06, "criterion: <0.10",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7
)
clean_axes(ax)

# ---------------------------------------------------------------------
# C. Recent-initiation representation
# ---------------------------------------------------------------------
ax = axes[1,0]
panel_label(ax, "C")
names = ["MIMIC-IV", "eICU"]
ratios = np.array([
    float(mfeas["recent_representation_ratio"]),
    float(erec["representation_ratio"]),
])
ax.axhspan(0.90, 1.10, alpha=0.10)
ax.axhline(1.00, linestyle="--", linewidth=0.8)
ax.scatter([0,1], ratios, s=30)
ax.set_xticks([0,1])
ax.set_xticklabels(names)
ax.set_ylim(0.89, 1.11)
ax.set_ylabel("Representation ratio")
ax.set_title("Recent-initiation representation", pad=5)
for x, v in enumerate(ratios):
    ax.annotate(
        f"{v:.3f}",
        xy=(x,v), xytext=(0,7),
        textcoords="offset points",
        ha="center", fontsize=7.0
    )
clean_axes(ax)

# ---------------------------------------------------------------------
# D. Compact support-summary panel
# ---------------------------------------------------------------------
ax = axes[1,1]
panel_label(ax, "D")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Recent-initiation support", pad=5)

ess = [
    float(mfeas["recent_weight_ess"]),
    float(erec["recent_weight_ess"]),
]
smd = [
    float(mfeas["recent_max_abs_smd"]),
    float(erec["max_abs_smd"]),
]

# Column headers
ax.text(0.46, 0.86, "MIMIC-IV", ha="center", va="center", fontsize=7.8, fontweight="bold")
ax.text(0.74, 0.86, "eICU", ha="center", va="center", fontsize=7.8, fontweight="bold")
ax.text(0.95, 0.86, "Criterion", ha="right", va="center", fontsize=7.8, fontweight="bold")

# Row 1
ax.text(0.03, 0.62, "Effective sample size", ha="left", va="center", fontsize=7.6)
ax.text(0.46, 0.62, f"{ess[0]:.0f}", ha="center", va="center", fontsize=7.8)
ax.text(0.74, 0.62, f"{ess[1]:.0f}", ha="center", va="center", fontsize=7.8)
ax.text(0.95, 0.62, ">=50", ha="right", va="center", fontsize=7.4)

# Row 2
ax.text(0.03, 0.37, "Maximum |SMD|", ha="left", va="center", fontsize=7.6)
ax.text(0.46, 0.37, f"{smd[0]:.3f}", ha="center", va="center", fontsize=7.8)
ax.text(0.74, 0.37, f"{smd[1]:.3f}", ha="center", va="center", fontsize=7.8)
ax.text(0.95, 0.37, "<0.10", ha="right", va="center", fontsize=7.4)

# Light table rules
ax.plot([0.02,0.98],[0.76,0.76], linewidth=0.7)
ax.plot([0.02,0.98],[0.49,0.49], linewidth=0.6)
ax.plot([0.02,0.98],[0.24,0.24], linewidth=0.7)

ax.text(
    0.03, 0.10,
    "Both cohorts satisfied the prespecified recent-initiation support criteria.",
    ha="left", va="center", fontsize=6.7
)

save_figure(fig, "Figure_4_Empirical_Support")
plt.close(fig)

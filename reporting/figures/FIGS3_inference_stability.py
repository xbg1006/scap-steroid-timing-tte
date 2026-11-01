from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt

from figure_config import read_csv_any, set_pub_style, clean_axes, panel_label, save_figure

set_pub_style()

d = read_csv_any("E32_05_INFERENCE_STABILITY_COMPARISON.csv").copy()
display = {
    "E3_primary_unpenalized_refit_bootstrap": "Primary full-refit",
    "E32_fixed_nuisance_cluster_bootstrap": "Fixed-nuisance conditional",
    "E32_ridge_C1_full_sensitivity": "Ridge C=1 full-refit",
}
d["label"] = d["analysis"].map(display)
d = d[d["label"].notna()].copy().reset_index(drop=True)

y = np.arange(len(d))[::-1]
markers = ["o", "s", "^"]

fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
fig.subplots_adjust(top=0.88, bottom=0.10, left=0.22, right=0.98, hspace=0.42, wspace=0.28)

def draw_rd(ax, xlim, title, show_labels):
    for idx, (yi, r) in enumerate(zip(y, d.to_dict("records"))):
        p = 100*float(r["risk_difference_point"])
        lo = 100*float(r["rd_ci_low"])
        hi = 100*float(r["rd_ci_high"])
        left, right = max(lo, xlim[0]), min(hi, xlim[1])
        if right >= left:
            ax.hlines(yi, left, right, linewidth=1.15)
            if lo >= xlim[0]:
                ax.plot(lo, yi, "|", markersize=6)
            if hi <= xlim[1]:
                ax.plot(hi, yi, "|", markersize=6)
        if xlim[0] <= p <= xlim[1]:
            ax.plot(p, yi, marker=markers[idx], linestyle="None", markersize=5.4)
    ax.axvline(0, linestyle="--", linewidth=0.75)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.45, 2.45)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"] if show_labels else [])
    ax.set_xlabel("Risk difference, percentage points")
    ax.set_title(title, pad=5)
    clean_axes(ax)

def draw_rr(ax, xlim, title, show_labels, log=False):
    for idx, (yi, r) in enumerate(zip(y, d.to_dict("records"))):
        p = float(r["risk_ratio_point"])
        lo = float(r["rr_ci_low"])
        hi = float(r["rr_ci_high"])
        left, right = max(lo, xlim[0]), min(hi, xlim[1])
        if right >= left:
            ax.hlines(yi, left, right, linewidth=1.15)
            if lo >= xlim[0]:
                ax.plot(lo, yi, "|", markersize=6)
            if hi <= xlim[1]:
                ax.plot(hi, yi, "|", markersize=6)
        if xlim[0] <= p <= xlim[1]:
            ax.plot(p, yi, marker=markers[idx], linestyle="None", markersize=5.4)
    ax.axvline(1, linestyle="--", linewidth=0.75)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.45, 2.45)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"] if show_labels else [])
    ax.set_xlabel("Risk ratio" + (" (log scale)" if log else ""))
    ax.set_title(title, pad=5)
    clean_axes(ax)

panel_label(axes[0,0], "A")
draw_rd(axes[0,0], (-25, 82), "Risk difference — full range", True)

panel_label(axes[0,1], "B")
draw_rr(axes[0,1], (0.08, 50), "Risk ratio — full range", False, log=True)

panel_label(axes[1,0], "C")
draw_rd(axes[1,0], (-5, 10), "Risk difference — central range", True)

panel_label(axes[1,1], "D")
draw_rr(axes[1,1], (0.80, 1.45), "Risk ratio — central range", False, log=False)

fig.suptitle("eICU inference-stability analyses",
             y=0.965, fontsize=10.6, fontweight="bold")
save_figure(fig, "Figure_S3_eICU_Inference_Stability")
plt.close(fig)

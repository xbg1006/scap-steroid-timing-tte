from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt

from figure_config import read_csv_any, set_pub_style, clean_axes, panel_label, save_figure

set_pub_style()

m = read_csv_any("weight_support_diagnostics.csv").iloc[0]
e = read_csv_any("E3_03_WEIGHT_AND_SUPPORT_DIAGNOSTICS.csv").iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7))
fig.subplots_adjust(top=0.79, bottom=0.19, left=0.11, right=0.98, wspace=0.28)

ax = axes[0]
panel_label(ax, "A")
labels = ["p95", "p99", "maximum"]
mx = [float(m["recent_ipcw_p95"]), float(m["recent_ipcw_p99"]), float(m["recent_ipcw_max"])]
ex = [float(e["recent_ipcw_p95"]), float(e["recent_ipcw_p99"]), float(e["recent_ipcw_max"])]
x = np.arange(3)
ax.plot(x, mx, marker="o", label="MIMIC-IV")
ax.plot(x, ex, marker="s", label="eICU")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Inverse probability weight")
ax.set_title("Recent-initiation weight tails", pad=5)
ax.legend(frameon=False)
clean_axes(ax)

ax = axes[1]
panel_label(ax, "B")
labels = ["p99", "maximum"]
mx = [float(m["cumulative_defer_ipcw_p99"]), float(m["cumulative_defer_ipcw_max"])]
ex = [float(e["cumulative_defer_ipcw_p99"]), float(e["cumulative_defer_ipcw_max"])]
x = np.arange(2)
ax.plot(x, mx, marker="o", label="MIMIC-IV")
ax.plot(x, ex, marker="s", label="eICU")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Cumulative deferral IPCW")
ax.set_title("Deferral-weight tails", pad=5)
ax.legend(frameon=False)
clean_axes(ax)

fig.suptitle("Weight-distribution diagnostics",
             y=0.955, fontsize=10.5, fontweight="bold")
save_figure(fig, "Figure_S4_Weight_Diagnostics")
plt.close(fig)

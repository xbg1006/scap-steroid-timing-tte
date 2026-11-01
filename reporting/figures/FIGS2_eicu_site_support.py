from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt
from figure_config import read_csv_any, set_pub_style, clean_axes, save_figure

set_pub_style()
d = read_csv_any("E3_05_SITE_SUPPORT.csv").sort_values("recent_share", ascending=False).reset_index(drop=True)
topn = min(20, len(d))
top = d.iloc[:topn].copy()

fig, ax = plt.subplots(figsize=(6.0, 4.5))
fig.subplots_adjust(top=0.80, bottom=0.15, left=0.17, right=0.97)

y = np.arange(topn)
ax.barh(y, 100*top["recent_share"])
ax.set_yticks(y)
ax.set_yticklabels([f"Rank {i+1}" for i in range(topn)])
ax.invert_yaxis()
ax.set_xlabel("Share of recent-initiation person-trials, %")
ax.set_ylabel("Hospital rank")
ax.set_title("Recent corticosteroid initiation across eICU hospitals", pad=6)
clean_axes(ax)

fig.text(
    0.50, 0.885,
    f"{d['hospitalid'].nunique()} hospitals contributed recent initiation; "
    f"largest single-hospital share, {100*d['recent_share'].max():.1f}%",
    ha="center", va="center", fontsize=7.2
)

save_figure(fig, "Figure_S2_eICU_Site_Support")
plt.close(fig)

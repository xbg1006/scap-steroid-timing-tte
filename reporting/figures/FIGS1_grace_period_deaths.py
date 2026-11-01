from __future__ import annotations
import os

import matplotlib.pyplot as plt

from figure_config import read_csv_any, set_pub_style, clean_axes, save_figure

set_pub_style()
d = read_csv_any("grace_period_event_audit.csv")

fig, ax = plt.subplots(figsize=(5.2, 3.55))
fig.subplots_adjust(top=0.90, bottom=0.18, left=0.17, right=0.97)

ax.plot(d["grace_upper_h"], d["person_trial_pct"], marker="o")
ax.set_xlabel("Hours from trial baseline")
ax.set_ylabel("Person-trials with death by time, %")
ax.set_xticks(d["grace_upper_h"])
ax.set_ylim(0, max(d["person_trial_pct"])*1.16)
clean_axes(ax)

# Keep only the clinically useful endpoint annotation.
last = d.iloc[-1]
ax.annotate(
    f"6 h: {int(last['death_person_trials_n'])} person-trial deaths\n"
    f"{int(last['death_unique_patients_n'])} unique patients "
    f"({float(last['person_trial_pct']):.2f}%)",
    xy=(last["grace_upper_h"], last["person_trial_pct"]),
    xytext=(-112, -4),
    textcoords="offset points",
    ha="left", va="center",
    fontsize=7.2,
    arrowprops=dict(arrowstyle="-", linewidth=0.7)
)

save_figure(fig, "Figure_S1_MIMIC_Grace_Period_Deaths")
plt.close(fig)

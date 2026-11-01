from __future__ import annotations
import os

import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figure_config import read_csv_any, set_pub_style, save_figure

set_pub_style()

flow = read_csv_any("08_STAGE4_COHORT_FLOW.csv")
mimic = read_csv_any("final_effect_estimates.csv").iloc[0]
e2 = read_csv_any("E2_01_LOCKED_COHORT_COUNTS.csv")
e3 = read_csv_any("external_effect_estimates.csv").iloc[0]

def flow_n(step):
    hit = flow.loc[flow["step"].astype(str).str.contains(step, regex=False), "n"]
    return int(hit.iloc[0]) if len(hit) else None

def e2_n(step):
    hit = e2.loc[e2["step"].astype(str).str.contains(step, regex=False), "n"]
    return int(hit.iloc[0]) if len(hit) else None

mimic_boxes = [
    ("Strict respiratory-support time zero", flow_n("Strict respiratory t0 identified")),
    ("Respiratory time zero within 48 h of admission", flow_n("within 48 h")),
    ("Medication administration observable around time zero", flow_n("eMAR observable")),
    ("No systemic corticosteroid during the prior 24 h", flow_n("New-user washout")),
    ("Primary trial-eligible severe CAP cohort", flow_n("Primary trial-eligible cohort")),
    ("Frozen MIMIC-IV primary cohort", int(mimic["unique_patients_n"])),
]
eicu_boxes = [
    ("Explicit respiratory-time-zero CAP-like seed", e2_n("E1 explicit respiratory-t0")),
    ("Exclude systemic corticosteroid before time zero", e2_n("exclude pre-t0 systemic steroid")),
    ("Adult and still in ICU at time zero", e2_n("adult + still in ICU")),
    ("Hospital medication-interface gate", e2_n("locked hospital medication-interface gate")),
    ("Outcome-complete final external cohort", int(e3["locked_unit_stays_n"])),
]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.5))
fig.subplots_adjust(top=0.88, bottom=0.08, left=0.04, right=0.99, wspace=0.18)

def draw(ax, title, boxes, summary):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, title, ha="center", va="top",
            fontsize=9.5, fontweight="bold")

    top, bottom = 0.84, 0.23
    ys = [top - i*(top-bottom)/(len(boxes)-1) for i in range(len(boxes))]
    bh = 0.078

    for i, ((label, n), y) in enumerate(zip(boxes, ys)):
        patch = FancyBboxPatch(
            (0.055, y-bh/2), 0.89, bh,
            boxstyle="round,pad=0.006,rounding_size=0.006",
            linewidth=0.85, fill=False
        )
        ax.add_patch(patch)
        wrapped = "\n".join(textwrap.wrap(label, width=35))
        ax.text(0.5, y+0.008, wrapped, ha="center", va="center",
                fontsize=7.15, linespacing=1.05)
        if n is not None:
            ax.text(0.5, y-0.022, f"n = {n:,}",
                    ha="center", va="center", fontsize=7.05)

        if i < len(boxes)-1:
            ax.add_patch(FancyArrowPatch(
                (0.5, y-bh/2-0.002),
                (0.5, ys[i+1]+bh/2+0.002),
                arrowstyle="-|>", mutation_scale=8.5, linewidth=0.75
            ))

    # Separate analysis summary, so patient counts are not confused with person-trials.
    sy = 0.085
    ax.add_patch(FancyBboxPatch(
        (0.055, sy-0.047), 0.89, 0.094,
        boxstyle="round,pad=0.006,rounding_size=0.006",
        linewidth=0.85, fill=False
    ))
    ax.text(0.5, sy+0.028, "Analysis set",
            ha="center", va="center", fontsize=7.7, fontweight="bold")
    ax.text(0.5, sy-0.008, summary,
            ha="center", va="center", fontsize=6.95, linespacing=1.15)

draw(
    axes[0],
    "MIMIC-IV development and primary analysis",
    mimic_boxes,
    f"{int(mimic['person_trials_n']):,} sequential person-trials\n"
    f"{int(mimic['initiate_within6h_person_trials_n']):,} initiations within 6 h",
)
draw(
    axes[1],
    "eICU external validation",
    eicu_boxes,
    f"{int(e3['patient_clusters_n']):,} patient-level clusters\n"
    f"{int(e3['person_trials_n']):,} person-trials; "
    f"{int(e3['initiate_within6h_person_trials_n']):,} initiations within 6 h",
)

fig.suptitle("Study cohort construction and analysis flow",
             y=0.965, fontsize=10.8, fontweight="bold")
save_figure(fig, "Figure_1_Cohort_Flow")
plt.close(fig)

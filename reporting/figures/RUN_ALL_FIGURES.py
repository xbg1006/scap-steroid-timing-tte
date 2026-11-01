from __future__ import annotations
from pathlib import Path
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
scripts = [
    "FIG1_cohort_flow.py",
    "FIG2_target_trial_schematic.py",
    "FIG3_effect_estimates.py",
    "FIG4_empirical_support.py",
    "FIGS1_grace_period_deaths.py",
    "FIGS2_eicu_site_support.py",
    "FIGS3_inference_stability.py",
    "FIGS4_weight_diagnostics.py",
]

for s in scripts:
    print("\n" + "="*88)
    print("RUNNING:", s)
    print("="*88)
    subprocess.run([sys.executable, str(HERE / s)], check=True)

print("\nAll V3 figures completed.")

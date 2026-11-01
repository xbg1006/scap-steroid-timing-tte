from __future__ import annotations

from pathlib import Path
import argparse
import math
import pandas as pd

EXPECTED = {
    "mimic": {
        "unique_patients_n": 2014,
        "person_trials_n": 18487,
        "initiate_within6h_person_trials_n": 744,
        "bootstrap_target_n": 300,
        "bootstrap_success_n": 300,
        "risk_defer_6h": 0.36605012094926914,
        "risk_initiate_6h": 0.4233224740923963,
        "risk_difference": 0.057272353143127175,
        "risk_ratio": 1.1564604131111995,
        "rd_ci_low": -0.004101049898267947,
        "rd_ci_high": 0.1201589953952386,
        "rr_ci_low": 0.9893444765125403,
        "rr_ci_high": 1.335650979000664,
    },
    "eicu": {
        "locked_unit_stays_n": 5394,
        "patient_clusters_n": 4794,
        "person_trials_n": 47923,
        "initiate_within6h_person_trials_n": 1341,
        "bootstrap_target_n": 300,
        "bootstrap_success_n": 300,
        "risk_defer": 0.2146866457714955,
        "risk_initiate": 0.2383163316544062,
        "risk_difference": 0.0236296858829106,
        "risk_ratio": 1.1100659325967634,
        "rd_ci_low": -0.1929641085667108,
        "rd_ci_high": 0.7604160592068233,
        "rr_ci_low": 0.1082719580360272,
        "rr_ci_high": 13.45472187287526,
    },
}

MIMIC_TOL = {
    "risk_defer_6h": 2e-4,
    "risk_initiate_6h": 2e-4,
    "risk_difference": 2e-4,
    "risk_ratio": 5e-4,
    "rd_ci_low": 5e-4,
    "rd_ci_high": 5e-4,
    "rr_ci_low": 5e-4,
    "rr_ci_high": 5e-4,
}
EICU_TOL = {
    "risk_defer": 1e-10,
    "risk_initiate": 1e-10,
    "risk_difference": 1e-10,
    "risk_ratio": 1e-10,
    "rd_ci_low": 1e-10,
    "rd_ci_high": 1e-10,
    "rr_ci_low": 1e-10,
    "rr_ci_high": 1e-10,
}


def check_exact_counts(label, row, expected):
    failures = []
    for key, value in expected.items():
        if not isinstance(value, int):
            continue
        if key not in row.index:
            failures.append(f"missing column {key}")
            continue
        observed = int(row[key])
        if observed != value:
            failures.append(f"{key}: got {observed}, expected {value}")
    if failures:
        print(f"[FAIL] {label} discrete geometry")
        for failure in failures:
            print("  -", failure)
        return False
    print(f"[PASS] {label} discrete geometry")
    return True


def check_numeric(label, row, expected, tolerances):
    failures = []
    for key, atol in tolerances.items():
        if key not in row.index:
            failures.append(f"missing column {key}")
            continue
        observed = float(row[key])
        target = float(expected[key])
        if not math.isfinite(observed) or abs(observed - target) > atol:
            failures.append(
                f"{key}: got {observed}, expected {target}, "
                f"|difference|={abs(observed-target):.6g}, tolerance={atol}"
            )
    if failures:
        print(f"[FAIL] {label} effect")
        for failure in failures:
            print("  -", failure)
        return False
    print(f"[PASS] {label} effect")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-effect", required=True, type=Path)
    parser.add_argument("--eicu-effect", required=True, type=Path)
    args = parser.parse_args()

    mimic = pd.read_csv(args.mimic_effect).iloc[0]
    eicu = pd.read_csv(args.eicu_effect).iloc[0]

    ok = True
    ok &= check_exact_counts("MIMIC", mimic, EXPECTED["mimic"])
    ok &= check_numeric("MIMIC", mimic, EXPECTED["mimic"], MIMIC_TOL)
    ok &= check_exact_counts("eICU", eicu, EXPECTED["eicu"])
    ok &= check_numeric("eICU", eicu, EXPECTED["eicu"], EICU_TOL)

    if not ok:
        raise SystemExit(1)

    print("\n[PASS] MIMIC frozen effect: numerically concordant")
    print("[PASS] eICU frozen effect: reproduced")
    print("PUBLISHED-OUTPUT VERIFICATION PASS")


if __name__ == "__main__":
    main()

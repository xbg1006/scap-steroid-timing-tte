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
        "risk_defer": 0.36605012094926914,
        "risk_initiate": 0.4233224740923963,
        "risk_difference": 0.057272353143127175,
        "risk_ratio": 1.1564604131111995,
    },
    "eicu": {
        "locked_unit_stays_n": 5394,
        "patient_clusters_n": 4794,
        "person_trials_n": 47923,
        "initiate_within6h_person_trials_n": 1341,
        "risk_defer": 0.2147,
        "risk_initiate": 0.2383,
        "risk_difference": 0.0236,
        "risk_ratio": 1.11,
    },
}

def close(a, b, atol):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=atol)

def check_row(label, row, expected, atol):
    failures=[]
    for k,v in expected.items():
        if k not in row.index:
            failures.append(f"missing column {k}")
            continue
        if isinstance(v, int):
            if int(row[k]) != v:
                failures.append(f"{k}: got {row[k]}, expected {v}")
        else:
            if not close(row[k], v, atol):
                failures.append(f"{k}: got {row[k]}, expected approximately {v}")
    if failures:
        print(f"[FAIL] {label}")
        for f in failures:
            print("  -", f)
        return False
    print(f"[PASS] {label}")
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mimic-effect", required=True)
    ap.add_argument("--eicu-effect", required=True)
    args=ap.parse_args()

    m=pd.read_csv(args.mimic_effect).iloc[0]
    e=pd.read_csv(args.eicu_effect).iloc[0]

    ok1=check_row("MIMIC frozen effect", m, EXPECTED["mimic"], 1e-8)
    # eICU manuscript values are rounded; use a narrow publication-level tolerance.
    ok2=check_row("eICU frozen effect", e, EXPECTED["eicu"], 5e-4)

    if not (ok1 and ok2):
        raise SystemExit(1)
    print("\nPUBLISHED-OUTPUT VERIFICATION PASS")

if __name__=="__main__":
    main()

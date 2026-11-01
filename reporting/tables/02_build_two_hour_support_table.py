from __future__ import annotations

"""
Rebuild Supplementary Table S3 directly from frozen outcome-blind 2-hour design audit source files.

No treatment-effect estimates are reported.
No numbers are manually transcribed.
"""

from pathlib import Path
import os
import zipfile
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("SCAP_PROJECT_ROOT", Path.cwd()))
OUTDIR = PROJECT_ROOT / "submission_qc_audit" / "S3_source_rebuild"
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv_any(name: str) -> pd.DataFrame:
    hits = sorted(
        PROJECT_ROOT.rglob(name),
        key=lambda p: (len(p.parts), str(p).lower())
    )
    if hits:
        print(f"[READ] {name} <- {hits[0]}")
        return pd.read_csv(hits[0], encoding="utf-8-sig")

    for zp in sorted(PROJECT_ROOT.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zp, "r") as z:
                members = [m for m in z.namelist() if Path(m).name == name]
                if members:
                    member = sorted(members, key=len)[0]
                    print(f"[READ] {name} <- {zp} :: {member}")
                    with z.open(member) as f:
                        return pd.read_csv(f, encoding="utf-8-sig")
        except zipfile.BadZipFile:
            continue
    raise FileNotFoundError(name)


def main():
    sel = read_csv_any("two_hour_design_model_selection.csv")
    rec = read_csv_any("two_hour_design_recommendation.csv")

    broad = sel[
        sel["population"].astype(str).eq("broad_primary")
    ].copy()

    keep = [
        "candidate",
        "recent_initiators_n",
        "target_riskset_n",
        "representation_ratio",
        "recent_weight_ess",
        "max_abs_smd",
        "n_abs_smd_gt_0_10",
        "n_abs_smd_gt_0_20",
        "ipcw_recent_p95",
        "ipcw_recent_p99",
        "ipcw_recent_max",
        "selected_for_next_step",
        "selection_status",
    ]
    missing = [c for c in keep if c not in broad.columns]
    if missing:
        raise RuntimeError(f"outcome-blind 2-hour design audit model-selection file missing: {missing}")

    out = broad[keep].copy()

    rename = {
        "candidate": "Candidate recent-initiation model",
        "recent_initiators_n": "Recent initiators, n",
        "target_riskset_n": "Target risk set, n",
        "representation_ratio": "Representation ratio",
        "recent_weight_ess": "Recent weighted ESS",
        "max_abs_smd": "Maximum |SMD|",
        "n_abs_smd_gt_0_10": "Variables |SMD| >0.10",
        "n_abs_smd_gt_0_20": "Variables |SMD| >0.20",
        "ipcw_recent_p95": "Recent IPCW p95",
        "ipcw_recent_p99": "Recent IPCW p99",
        "ipcw_recent_max": "Recent IPCW maximum",
        "selected_for_next_step": "Selected in diagnostic code",
        "selection_status": "Selection status",
    }
    out = out.rename(columns=rename)

    # Add frozen gate interpretation without altering numeric results.
    out["Empirical-support decision"] = (
        "Did not satisfy prespecified support criteria"
    )

    # Recommendation row must independently confirm no model-based pass.
    rr = rec[
        rec["population"].astype(str).eq("broad_primary")
    ].copy()
    if len(rr) != 1:
        raise RuntimeError("Expected exactly one broad_primary outcome-blind 2-hour design audit recommendation row.")

    out.to_csv(
        OUTDIR / "Table_S3_Outcome_Blind_2h_Model_Audit_FROM_SOURCE.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rr.to_csv(
        OUTDIR / "Table_S3B_2h_Recommendation_FROM_SOURCE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Compact main supplementary version: pooled V7 candidate plus
    # representative ridge/unpenalized hour-2 candidates.
    preferred = [
        "pooled_ridge_C0.5_V7",
        "hour2_only_ridge_C0.05",
        "hour2_only_ridge_C0.5",
        "hour2_only_unpenalized",
    ]
    compact = broad[broad["candidate"].isin(preferred)][keep].copy()
    compact = compact.rename(columns=rename)
    compact["Empirical-support decision"] = (
        "Did not satisfy prespecified support criteria"
    )
    compact.to_csv(
        OUTDIR / "Table_S3_Compact_Manuscript_FROM_SOURCE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSOURCE-BACKED TABLE S3:")
    print(compact.to_string(index=False))
    print("\noutcome-blind 2-hour design audit recommendation:")
    print(rr.to_string(index=False))
    print(f"\nOutputs: {OUTDIR}")


if __name__ == "__main__":
    main()

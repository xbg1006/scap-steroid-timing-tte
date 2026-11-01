from __future__ import annotations

"""
E3.4 — Post-hoc eICU endpoint chronology sensitivity.

Purpose
-------
Submission-QC audit found a small set of final E3 stays with:
    hospitalDischargeOffset < unitDischargeOffset

This script does NOT tune or change:
    - 6-hour strategy
    - sequential trial starts
    - recent [3,6) h window
    - ATO/overlap target
    - baseline model specification
    - defer model specification
    - recent-initiation model specification
    - endpoint estimator

It produces point-estimate data-quality sensitivities only.

Two sensitivity estimators are reported:
1) fixed_original_nuisance:
   Exclude chronology-inconsistent stays from contribution to the estimator
   while retaining nuisance predictions fit in the original frozen E3 sample.
   This isolates the effect of removing the questionable source-data rows.

2) refit_same_locked_specification:
   Exclude the same stays, then refit exactly the same frozen nuisance model
   specifications. This is NOT model selection or tuning.

No bootstrap CI is generated here.
"""

from pathlib import Path
import os
import importlib.util
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
OUTDIR = PROJECT_ROOT / "submission_qc_audit" / "E34_chronology_sensitivity"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ORIGINAL_STAYS = 5394
EXPECTED_ORIGINAL_TRIALS = 47923
EXPECTED_BAD_STAYS = 92


def find_e31_script() -> Path:
    names = ["03_final_external_effect.py"]
    candidates = []
    for nm in names:
        candidates.extend(PROJECT_ROOT.rglob(nm))

    if not candidates:
        raise FileNotFoundError(
            "Could not find 03_final_external_effect.py "
            f"below {PROJECT_ROOT}"
        )

    # Prefer the copy whose sibling outputs_eicu_E3/cache exists.
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            0 if (p.parent / "outputs_eicu_E3" / "cache" / "e3_trials.parquet").exists() else 1,
            len(p.parts),
            str(p).lower(),
        ),
    )
    return candidates[0]


def load_module(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def to_record(label, est, stays_n, clusters_n, trials_n, bad_stays_removed):
    return {
        "analysis": label,
        "stays_n": int(stays_n),
        "patient_clusters_n": int(clusters_n),
        "person_trials_n": int(trials_n),
        "chronology_inconsistent_stays_removed_n": int(bad_stays_removed),
        "risk_defer": float(est["risk_defer"]),
        "risk_initiate": float(est["risk_initiate"]),
        "risk_difference": float(est["risk_difference"]),
        "risk_difference_percentage_points": float(est["risk_difference"] * 100),
        "risk_ratio": float(est["risk_ratio"]),
    }


def main():
    e31_path = find_e31_script()
    print(f"[E31 SOURCE] {e31_path}")

    e31 = load_module(e31_path, "e31_for_e34")
    e2 = e31.import_e2()

    trials_path = e31.CACHE_TRIALS
    panel_path = e31.CACHE_PANEL
    effect_path = e31.OUT_EFFECT

    if not trials_path.exists() or not panel_path.exists():
        raise FileNotFoundError(
            "Frozen E3 cache not found next to the selected E3.1 script.\n"
            f"Expected:\n  {trials_path}\n  {panel_path}"
        )

    trials = pd.read_parquet(trials_path)
    panel = pd.read_parquet(panel_path)

    if len(trials) != EXPECTED_ORIGINAL_TRIALS:
        raise RuntimeError(
            f"Original E3 trials={len(trials)}, expected {EXPECTED_ORIGINAL_TRIALS}. STOP."
        )
    if trials["patientunitstayid"].nunique() != EXPECTED_ORIGINAL_STAYS:
        raise RuntimeError(
            f"Original E3 stays={trials['patientunitstayid'].nunique()}, "
            f"expected {EXPECTED_ORIGINAL_STAYS}. STOP."
        )

    needed = [
        "patientunitstayid",
        "cluster_id",
        "trial_hour",
        "respiratory_t0_candidate_min",
        "hospital_discharge_offset_min",
        "unit_discharge_offset_min",
        "hospital_death",
    ]
    miss = [c for c in needed if c not in trials.columns]
    if miss:
        raise RuntimeError(f"Frozen E3 trials missing columns: {miss}")

    # ------------------------------------------------------------------
    # Define the objective chronology inconsistency directly from final E3.
    # ------------------------------------------------------------------
    bad_mask_row = (
        trials["hospital_discharge_offset_min"].notna()
        & trials["unit_discharge_offset_min"].notna()
        & (
            trials["hospital_discharge_offset_min"]
            < trials["unit_discharge_offset_min"]
        )
    )
    bad_ids = set(
        trials.loc[bad_mask_row, "patientunitstayid"]
        .dropna().astype(int).unique().tolist()
    )

    if len(bad_ids) != EXPECTED_BAD_STAYS:
        raise RuntimeError(
            f"Chronology-inconsistent final E3 stays={len(bad_ids)}, "
            f"expected {EXPECTED_BAD_STAYS}. STOP."
        )

    stay_level = (
        trials.sort_values(["patientunitstayid", "trial_hour"])
        .drop_duplicates("patientunitstayid")
        .copy()
    )
    bad_stay = stay_level[
        stay_level["patientunitstayid"].astype(int).isin(bad_ids)
    ].copy()

    # Actual trial-baseline violation.
    baseline_min = (
        trials["respiratory_t0_candidate_min"]
        + trials["trial_hour"] * 60
    )
    invalid_trial = (
        trials["hospital_discharge_offset_min"].notna()
        & (trials["hospital_discharge_offset_min"] <= baseline_min)
    )

    invalid_ids = set(
        trials.loc[invalid_trial, "patientunitstayid"]
        .dropna().astype(int).unique().tolist()
    )

    # ------------------------------------------------------------------
    # Recompute original frozen point estimate as a hard validation.
    # ------------------------------------------------------------------
    print("[FIT] Recomputing original frozen E3 point estimate...")
    prep_full = e31.prepare_models(e2, trials, panel)
    row_full, e6_full, q_full, p_hour_full, p_recent_full = e31.fit_nuisance(
        prep_full, None
    )
    est_original = e31.estimate_effect(
        prep_full,
        q_full,
        p_hour_full,
        p_recent_full,
        row_full,
    )

    if effect_path.exists():
        published = pd.read_csv(effect_path).iloc[0]
        checks = {
            "risk_defer": float(published["risk_defer"]),
            "risk_initiate": float(published["risk_initiate"]),
            "risk_difference": float(published["risk_difference"]),
            "risk_ratio": float(published["risk_ratio"]),
        }
        for k, v in checks.items():
            if not np.isclose(float(est_original[k]), v, rtol=1e-7, atol=1e-9):
                raise RuntimeError(
                    f"Frozen E3 recomputation mismatch for {k}: "
                    f"recomputed={est_original[k]}, published={v}. STOP."
                )
        print("[PASS] Original frozen E3 point estimate reproduced exactly.")

    # ------------------------------------------------------------------
    # Sensitivity A: fixed original nuisance predictions.
    # Remove all 92 stays only from estimator contribution.
    # ------------------------------------------------------------------
    print("[SENSITIVITY A] Excluding all 92 inconsistent stays, fixed nuisance...")
    row_fixed = np.asarray(row_full, float).copy()
    exclude_trial = (
        trials["patientunitstayid"].astype(int).isin(bad_ids).to_numpy()
    )
    row_fixed[exclude_trial] = 0.0

    est_fixed = e31.estimate_effect(
        prep_full,
        q_full,
        p_hour_full,
        p_recent_full,
        row_fixed,
    )

    # ------------------------------------------------------------------
    # Sensitivity B: same locked specification, nuisance models refit.
    # ------------------------------------------------------------------
    print("[SENSITIVITY B] Excluding all 92 inconsistent stays, frozen-spec refit...")
    trials_strict = trials.loc[~exclude_trial].copy()
    panel_strict = panel.loc[
        ~panel["patientunitstayid"].astype(int).isin(bad_ids)
    ].copy()

    prep_strict = e31.prepare_models(e2, trials_strict, panel_strict)
    row_s, e6_s, q_s, p_hour_s, p_recent_s = e31.fit_nuisance(
        prep_strict, None
    )
    est_refit = e31.estimate_effect(
        prep_strict,
        q_s,
        p_hour_s,
        p_recent_s,
        row_s,
    )

    # ------------------------------------------------------------------
    # Minimal diagnostic: remove only stays that generate >=1 trial
    # baseline at/after recorded hospital discharge.
    # This is diagnostic only, not the preferred manuscript sensitivity.
    # ------------------------------------------------------------------
    exclude_minimal = (
        trials["patientunitstayid"].astype(int).isin(invalid_ids).to_numpy()
    )
    row_min = np.asarray(row_full, float).copy()
    row_min[exclude_minimal] = 0.0
    est_minimal = e31.estimate_effect(
        prep_full,
        q_full,
        p_hour_full,
        p_recent_full,
        row_min,
    )

    # ------------------------------------------------------------------
    # Support check after strict exclusion, using same frozen gate rules.
    # ------------------------------------------------------------------
    tr0 = trials_strict.reset_index(drop=True).copy()
    tr0["trial_idx"] = np.arange(len(tr0), dtype=int)

    prep0 = e2.make_preprocessor(tr0, False)
    X0 = prep0.fit_transform(tr0)
    y0 = tr0["actual_initiate_next6h"].astype(int).to_numpy()
    m0 = e31.fit_unpenalized_weighted(X0, y0)
    e6_check = np.clip(
        m0.predict_proba(X0)[:, 1],
        e31.PS_CLIP,
        1 - e31.PS_CLIP,
    )
    q_check = e6_check * (1 - e6_check)

    long_check = e2.build_defer_long(tr0, panel_strict)
    _, defer_sum = e2.defer_audit(tr0, long_check, q_check)
    _, recent_sum, _ = e2.recent_audit(tr0, panel_strict, q_check)

    rr = recent_sum.iloc[0]
    support_pass = (
        (defer_sum["max_abs_smd"] < 0.10).all()
        and (defer_sum["n_abs_smd_gt_0_10"] == 0).all()
        and 0.90 <= float(rr["representation_ratio"]) <= 1.10
        and float(rr["recent_weight_ess"]) >= 50
        and float(rr["max_abs_smd"]) < 0.10
        and int(rr["n_abs_smd_gt_0_10"]) == 0
    )

    support = pd.DataFrame([{
        "analysis": "strict chronology exclusion",
        "stays_n": int(trials_strict["patientunitstayid"].nunique()),
        "person_trials_n": int(len(trials_strict)),
        "recent_representation_ratio": float(rr["representation_ratio"]),
        "recent_weight_ess": float(rr["recent_weight_ess"]),
        "recent_max_abs_smd": float(rr["max_abs_smd"]),
        "recent_n_abs_smd_gt_0_10": int(rr["n_abs_smd_gt_0_10"]),
        "defer_max_abs_smd_over_intervals": float(defer_sum["max_abs_smd"].max()),
        "defer_any_variable_gt_0_10": int(
            (defer_sum["n_abs_smd_gt_0_10"] > 0).any()
        ),
        "frozen_support_gate_pass": int(support_pass),
    }])

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    records = [
        to_record(
            "original_frozen_E3_recomputed",
            est_original,
            trials["patientunitstayid"].nunique(),
            trials["cluster_id"].nunique(),
            len(trials),
            0,
        ),
        to_record(
            "strict_exclusion_92_fixed_original_nuisance",
            est_fixed,
            trials_strict["patientunitstayid"].nunique(),
            trials_strict["cluster_id"].nunique(),
            len(trials_strict),
            len(bad_ids),
        ),
        to_record(
            "strict_exclusion_92_refit_same_locked_specification",
            est_refit,
            trials_strict["patientunitstayid"].nunique(),
            trials_strict["cluster_id"].nunique(),
            len(trials_strict),
            len(bad_ids),
        ),
        to_record(
            "minimal_exclusion_stays_with_trial_after_recorded_hospital_discharge_fixed_nuisance",
            est_minimal,
            trials.loc[~exclude_minimal, "patientunitstayid"].nunique(),
            trials.loc[~exclude_minimal, "cluster_id"].nunique(),
            int((~exclude_minimal).sum()),
            len(invalid_ids),
        ),
    ]
    result = pd.DataFrame(records)

    orig_rd = result.loc[
        result["analysis"] == "original_frozen_E3_recomputed",
        "risk_difference"
    ].iloc[0]
    orig_rr = result.loc[
        result["analysis"] == "original_frozen_E3_recomputed",
        "risk_ratio"
    ].iloc[0]

    result["delta_RD_vs_original_percentage_points"] = (
        (result["risk_difference"] - orig_rd) * 100
    )
    result["delta_RR_vs_original"] = result["risk_ratio"] - orig_rr

    # Stay-level descriptive audit.
    bad_stay["unit_minus_hospital_min"] = (
        bad_stay["unit_discharge_offset_min"]
        - bad_stay["hospital_discharge_offset_min"]
    )

    status = pd.DataFrame([{
        "final_E3_stays_n": int(trials["patientunitstayid"].nunique()),
        "chronology_inconsistent_stays_n": int(len(bad_ids)),
        "chronology_inconsistent_stays_pct": float(
            len(bad_ids) / trials["patientunitstayid"].nunique()
        ),
        "hospital_death_n_among_inconsistent": int(
            (bad_stay["hospital_death"].astype(int) == 1).sum()
        ),
        "hospital_alive_n_among_inconsistent": int(
            (bad_stay["hospital_death"].astype(int) == 0).sum()
        ),
        "hospitals_with_inconsistent_stays_n": int(
            bad_stay["hospitalid"].nunique()
        ) if "hospitalid" in bad_stay.columns else np.nan,
        "median_unit_minus_hospital_min": float(
            bad_stay["unit_minus_hospital_min"].median()
        ),
        "p95_unit_minus_hospital_min": float(
            bad_stay["unit_minus_hospital_min"].quantile(0.95)
        ),
        "max_unit_minus_hospital_min": float(
            bad_stay["unit_minus_hospital_min"].max()
        ),
        "person_trials_at_or_after_recorded_hospital_discharge_n": int(
            invalid_trial.sum()
        ),
        "unique_stays_with_trial_at_or_after_recorded_hospital_discharge_n": int(
            len(invalid_ids)
        ),
    }])

    result.to_csv(
        OUTDIR / "E34_01_CHRONOLOGY_SENSITIVITY_EFFECTS.csv",
        index=False,
        encoding="utf-8-sig",
    )
    support.to_csv(
        OUTDIR / "E34_02_STRICT_EXCLUSION_SUPPORT_CHECK.csv",
        index=False,
        encoding="utf-8-sig",
    )
    status.to_csv(
        OUTDIR / "E34_03_CHRONOLOGY_AUDIT_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bad_stay.to_csv(
        OUTDIR / "E34_04_STRICT_EXCLUDED_STAYS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "analysis_type": "post-hoc source-data-quality sensitivity triggered during submission QC",
        "primary_external_effect_replaced": False,
        "primary_external_effect_status": "remains frozen E3",
        "strict_exclusion_rule": "hospitalDischargeOffset < unitDischargeOffset",
        "strict_exclusion_stays_n": len(bad_ids),
        "model_tuning_performed": False,
        "nuisance_specifications_changed": False,
        "bootstrap_performed": False,
        "preferred_manuscript_sensitivity": "strict_exclusion_92_refit_same_locked_specification",
        "fixed_nuisance_role": "isolates the effect of deleting chronology-inconsistent rows",
        "minimal_exclusion_role": "diagnostic only",
    }
    with open(
        OUTDIR / "E34_PROTOCOL_AND_INTERPRETATION.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 110)
    print("E3.4 CHRONOLOGY SENSITIVITY COMPLETE")
    print("=" * 110)
    print(status.to_string(index=False))
    print("\nEffect sensitivity:")
    print(result.to_string(index=False))
    print("\nSupport after strict exclusion:")
    print(support.to_string(index=False))
    print(f"\nOutputs: {OUTDIR}")


if __name__ == "__main__":
    main()

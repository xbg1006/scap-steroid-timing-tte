from __future__ import annotations

"""
E3.1 FINAL — eICU locked external effect with pre-effect data-quality preflight

This is the FIRST stage that opens the eICU steroid-outcome effect.

Everything treatment-related was locked outcome-blind in E1/E2.

Frozen external strategy:
    initiate systemic corticosteroid within 6h
    vs
    defer at least 6h

Sequential trial starts:
    0,2,...,18h after explicit respiratory-support t0

Recent window:
    [3h,6h)

Frozen target:
    overlap / clinical-equipoise population

Frozen nuisance models:
    baseline e6: unpenalized logistic
    DEFER censoring: pooled 1h ridge logistic C=0.5
    recent initiation: unpenalized logistic at +3h

External endpoint declared in E2:
    documented in-hospital death within 28 days from each trial baseline

IMPORTANT:
    Alive hospital discharge is a competing event, not ordinary censoring.
    E3 therefore estimates weighted Aalen-Johansen cumulative incidence.

Inference:
    patient-level cluster bootstrap (uniquePID where available),
    checkpoint every 10 successful replicates.

No window/model/cohort selection is performed in E3.
"""

from pathlib import Path
import os
import importlib.util
import json
import traceback
import warnings

import numpy as np
import pandas as pd
import config

from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(config.PROJECT_ROOT) if hasattr(config, "PROJECT_ROOT") else Path.cwd()
HERE = Path(__file__).resolve().parent

DATA_DIR = Path(getattr(config, "EICU_DATA_DIR", os.environ.get("EICU_DATA_DIR", "PATH_TO_EICU_DATA")))
E1_DIR = HERE / "outputs_eicu_E1"
E2_DIR = HERE / "outputs_eicu_E2"
OUTPUT_DIR = HERE / "outputs_eicu_E3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

E2_SCRIPT_CANDIDATES = [
    HERE / "EICU_E21_locked_mapping_positivity.py",
    HERE / "02_locked_mapping_empirical_support.py",
]

E2_RECOMMEND = E2_DIR / "E2_11_RECOMMENDATION.csv"
E2_LOCK = E2_DIR / "E2_12_EXTERNAL_PROTOCOL_LOCK.json"

OUT_GATE = OUTPUT_DIR / "E3_00_PRE_EFFECT_MAPPING_GATE.csv"
OUT_MAPPING_DETAIL = OUTPUT_DIR / "E3_00B_PRE_EFFECT_MAPPING_PROBLEMS.csv"
OUT_MAPPING_SITE = OUTPUT_DIR / "E3_00C_PRE_EFFECT_MAPPING_BY_HOSPITAL.csv"
OUT_GRACE_PREFLIGHT = OUTPUT_DIR / "E3_00D_PRE_EFFECT_GRACE_OBSERVATION_AUDIT.csv"
OUT_EFFECT = OUTPUT_DIR / "external_effect_estimates.csv"
OUT_BOOT = OUTPUT_DIR / "E3_02_BOOTSTRAP_SUMMARY.csv"
OUT_DIAG = OUTPUT_DIR / "E3_03_WEIGHT_AND_SUPPORT_DIAGNOSTICS.csv"
OUT_TRIAL = OUTPUT_DIR / "E3_04_TRIAL_HOUR_COUNTS.csv"
OUT_SITE = OUTPUT_DIR / "E3_05_SITE_SUPPORT.csv"
OUT_ENDPOINT = OUTPUT_DIR / "E3_06_ENDPOINT_MAPPING_AUDIT.csv"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = CHECKPOINT_DIR / "eicu_e3_broad_6h_pure_ccw_cluster_bootstrap.csv"

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_COHORT = CACHE_DIR / "e3_locked_cohort.parquet"
CACHE_PANEL = CACHE_DIR / "e3_dynamic_panel.parquet"
CACHE_TRIALS = CACHE_DIR / "e3_trials.parquet"

SEED = 20261028
BOOTSTRAP_TARGET = 300
CHECKPOINT_EVERY = 10

FOLLOWUP_MIN = 28 * 24 * 60
WINDOW_H = 6
RECENT_H0 = 3
RECENT_H1 = 6
PS_CLIP = 1e-5

EXPECTED_E2_DECISION = "PROCEED_TO_E3_ONCE_LOCKED_EXTERNAL_EFFECT"
EXPECTED_EXTERNAL_OUTCOME = "documented in-hospital death within 28 days from trial baseline"

# Pure data-quality gates declared before effect estimation.
MIN_COMPLETE_OUTCOME_MAPPING = 0.95
MAX_TRIALS_WITH_ICU_DISCHARGE_INSIDE_GRACE = 0.05

# PRE-EFFECT data-quality rule added after the original mapping gate exposed
# timestamp anomalies, but BEFORE any external treatment effect was opened.
#
# A stay may be excluded as timestamp-inconsistent only when:
#   1) hospital discharge is at/before respiratory t0, AND
#   2) the hospital discharge timestamp is internally inconsistent with the
#      ICU unit discharge timestamp (hospital discharge < unit discharge), AND
#   3) such rows are <=1% of the locked E2 cohort.
#
# If the prevalence is >1%, E3.1 aborts rather than silently repairing the data.
MAX_TIMESTAMP_INCONSISTENT_STAYS_PCT = 0.01


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def import_e2():
    for p in E2_SCRIPT_CANDIDATES:
        if p.exists():
            return load_module(p, "eicu_e2_for_e3")
    raise FileNotFoundError(
        "Cannot find 02_locked_mapping_empirical_support.py next to the external-effect script."
    )


def enforce_e2_lock():
    if not E2_RECOMMEND.exists() or not E2_LOCK.exists():
        raise FileNotFoundError(
            "E2 outputs are required next to E3:\n"
            "  outputs_eicu_E2/E2_11_RECOMMENDATION.csv\n"
            "  outputs_eicu_E2/E2_12_EXTERNAL_PROTOCOL_LOCK.json"
        )

    rec = pd.read_csv(E2_RECOMMEND)
    if len(rec) != 1:
        raise RuntimeError("E2 recommendation must contain exactly one row.")
    row = rec.iloc[0]

    if str(row["decision"]) != EXPECTED_E2_DECISION:
        raise RuntimeError(
            f"E3 ABORTED. E2 decision was: {row['decision']}"
        )
    if int(row["effect_opened_in_E2"]) != 0:
        raise RuntimeError("E3 ABORTED. E2 unexpectedly reports effect already opened.")
    if str(row["external_outcome_predeclared"]) != EXPECTED_EXTERNAL_OUTCOME:
        raise RuntimeError(
            "E3 ABORTED. External endpoint does not match the E2 protocol lock."
        )

    with open(E2_LOCK, "r", encoding="utf-8") as f:
        lock = json.load(f)

    st = lock["strategy"]
    if int(st["initiate_window_h"]) != 6:
        raise RuntimeError("E3 ABORTED: locked treatment window is not 6h.")
    if list(st["recent_window_h"]) != [3, 6]:
        raise RuntimeError("E3 ABORTED: locked recent window is not [3,6].")

    print("[LOCK] E2 support gate verified. External effect can now be opened.")


def fit_unpenalized_weighted(X, y, sample_weight=None):
    attempts = [
        dict(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-9),
        dict(penalty="none", solver="lbfgs", max_iter=5000, tol=1e-9),
    ]
    last = None
    for kwargs in attempts:
        try:
            m = LogisticRegression(
                fit_intercept=True,
                random_state=SEED,
                **kwargs,
            )
            m.fit(X, y, sample_weight=sample_weight)
            return m
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unpenalized logistic failed: {last}")


def ess(w):
    x = np.asarray(w, float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    return (x.sum() ** 2) / np.sum(x ** 2)


def load_outcome_map(e2, eligible_ids):
    """
    eICU hospitalDischargeOffset is relative to ICU admission.

    status 'Expired' => death competing-event type 1
    status alive/non-expired => alive discharge competing-event type 2
    """
    p = e2.find_table("patient")
    if p is None:
        raise FileNotFoundError("patient table missing.")

    cols = e2.read_header(p)
    wanted = [
        "patientUnitStayID",
        "uniquePID",
        "patientHealthSystemStayID",
        "hospitalDischargeStatus",
        "hospitalDischargeOffset",
    ]
    actual = [e2.resolve_col(cols, c) for c in wanted]
    actual = [c for c in actual if c is not None]

    df = pd.read_csv(p, usecols=actual, low_memory=False)
    idc = e2.resolve_col(df.columns, "patientUnitStayID")
    df["patientunitstayid"] = e2.to_num(df[idc]).astype("Int64")
    df = df[df["patientunitstayid"].isin(eligible_ids)].copy()

    up = e2.resolve_col(df.columns, "uniquePID")
    hs = e2.resolve_col(df.columns, "patientHealthSystemStayID")
    st = e2.resolve_col(df.columns, "hospitalDischargeStatus")
    off = e2.resolve_col(df.columns, "hospitalDischargeOffset")

    out = pd.DataFrame({
        "patientunitstayid": df["patientunitstayid"],
        "unique_pid": (
            df[up].astype("string")
            if up else pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        "healthsystemstayid": (
            e2.to_num(df[hs]).astype("Int64")
            if hs else pd.Series(pd.NA, index=df.index, dtype="Int64")
        ),
        "hospital_discharge_status": (
            df[st].astype("string")
            if st else pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        "hospital_discharge_offset_min": (
            e2.to_num(df[off])
            if off else pd.Series(np.nan, index=df.index)
        ),
    })

    status = out["hospital_discharge_status"].astype("string").str.lower()
    out["hospital_death"] = status.str.contains(
        r"expired|dead|death",
        regex=True,
        na=False,
    ).astype("int8")

    # Cluster at patient level where possible.
    cluster = out["unique_pid"].astype("string")
    missing = cluster.isna() | (cluster.str.len() == 0) | (cluster == "<NA>")

    hs_text = out["healthsystemstayid"].astype("string")
    cluster = cluster.mask(missing, "HS_" + hs_text)

    missing2 = cluster.isna() | (cluster.str.len() == 0) | cluster.str.contains("<NA>", na=True)
    cluster = cluster.mask(
        missing2,
        "STAY_" + out["patientunitstayid"].astype("string"),
    )
    out["cluster_id"] = cluster.astype("string")

    return out.drop_duplicates("patientunitstayid")


def rebuild_locked_cohort(e2):
    seed = e2.load_e1_seed()
    seed_ids = e2.safe_int_set(seed["patientunitstayid"])
    patient = e2.load_patient_mapping(seed_ids)
    hospital = e2.load_hospital()

    comorb = e2.comorbidity_flags(seed_ids)
    patient = patient.merge(comorb, on="patientunitstayid", how="left")
    for c in e2.COMORBIDITY_REGEX:
        patient[c] = patient[c].fillna(0).astype("int8")

    gate, passed_hospitals = e2.medication_interface_gate(
        seed.merge(
            patient[["patientunitstayid", "hospitalid"]],
            on="patientunitstayid",
            how="left",
        )
    )

    cohort_all = e2.build_locked_cohort(
        seed,
        patient,
        hospital,
        passed_hospitals,
    )
    cohort = cohort_all[cohort_all["e2_locked_eligible"] == 1].copy()

    outcome = load_outcome_map(
        e2,
        e2.safe_int_set(cohort["patientunitstayid"]),
    )
    cohort = cohort.merge(
        outcome,
        on="patientunitstayid",
        how="left",
        validate="one_to_one",
    )
    return cohort, gate


def pre_effect_mapping_gate(cohort):
    """
    Outcome-blind endpoint mapping preflight.

    E3.0 required zero hospital-discharge timestamps at/before respiratory t0.
    E3.1 keeps that scientific intent but distinguishes a small number of
    internally impossible source-data timestamps from a systematic mapping
    failure.

    No treatment-effect information is used here.
    """
    x = cohort.copy()

    complete_status = x["hospital_discharge_status"].notna()
    complete_offset = x["hospital_discharge_offset_min"].notna()
    complete = complete_status & complete_offset
    completeness = float(complete.mean())

    invalid_before_t0 = (
        complete_offset
        & (
            x["hospital_discharge_offset_min"]
            <= x["respiratory_t0_candidate_min"]
        )
    )

    # Internal chronology check:
    # hospital discharge should not occur before ICU-unit discharge.
    have_unit = x["unit_discharge_offset_min"].notna()
    hospital_before_unit = (
        complete_offset
        & have_unit
        & (
            x["hospital_discharge_offset_min"]
            < x["unit_discharge_offset_min"]
        )
    )

    timestamp_inconsistent = invalid_before_t0 & hospital_before_unit

    invalid_n = int(invalid_before_t0.sum())
    inconsistent_n = int(timestamp_inconsistent.sum())
    inconsistent_pct = inconsistent_n / len(x) if len(x) else np.nan

    # All invalid-before-t0 rows must be explainable by the narrow internal
    # chronology inconsistency rule; otherwise this is a mapping problem.
    all_invalid_explainable = bool(
        invalid_n == inconsistent_n
    )

    allowed_timestamp_exclusion = bool(
        inconsistent_n > 0
        and all_invalid_explainable
        and inconsistent_pct <= MAX_TIMESTAMP_INCONSISTENT_STAYS_PCT
    )

    # Mapping passes when:
    #  - endpoint completeness >=95%; and
    #  - either no invalid timestamps, or only the predeclared tiny set of
    #    internally impossible timestamps.
    mapping_pass = bool(
        completeness >= MIN_COMPLETE_OUTCOME_MAPPING
        and (
            invalid_n == 0
            or allowed_timestamp_exclusion
        )
    )

    gate = pd.DataFrame([{
        "locked_e2_stays_n": len(x),
        "hospital_discharge_status_nonmissing_n": int(complete_status.sum()),
        "hospital_discharge_offset_nonmissing_n": int(complete_offset.sum()),
        "outcome_mapping_complete_n": int(complete.sum()),
        "outcome_mapping_complete_pct": completeness,
        "hospital_discharge_at_or_before_t0_n": invalid_n,
        "hospital_discharge_before_unit_discharge_n": int(hospital_before_unit.sum()),
        "timestamp_inconsistent_excludable_n": inconsistent_n,
        "timestamp_inconsistent_pct": inconsistent_pct,
        "all_invalid_before_t0_explainable_by_internal_chronology": int(
            all_invalid_explainable
        ),
        "timestamp_exclusion_rule_max_pct": MAX_TIMESTAMP_INCONSISTENT_STAYS_PCT,
        "timestamp_exclusion_rule_triggered": int(allowed_timestamp_exclusion),
        "complete_outcome_gate_threshold": MIN_COMPLETE_OUTCOME_MAPPING,
        "mapping_gate_pass": int(mapping_pass),
        "external_endpoint": EXPECTED_EXTERNAL_OUTCOME,
        "effect_opened_before_this_gate": 0,
    }])

    # Detailed problem rows for transparent audit.
    problem = x.loc[
        (~complete)
        | invalid_before_t0
        | hospital_before_unit,
        [
            "patientunitstayid",
            "cluster_id",
            "hospitalid",
            "respiratory_t0_candidate_min",
            "unit_discharge_offset_min",
            "hospital_discharge_offset_min",
            "hospital_discharge_status",
            "steroid_relative_to_t0_min",
        ],
    ].copy()

    problem["missing_discharge_status"] = (
        problem["hospital_discharge_status"].isna().astype("int8")
    )
    problem["missing_discharge_offset"] = (
        problem["hospital_discharge_offset_min"].isna().astype("int8")
    )
    problem["hospital_discharge_at_or_before_t0"] = (
        problem["hospital_discharge_offset_min"].notna()
        & (
            problem["hospital_discharge_offset_min"]
            <= problem["respiratory_t0_candidate_min"]
        )
    ).astype("int8")
    problem["hospital_discharge_before_unit_discharge"] = (
        problem["hospital_discharge_offset_min"].notna()
        & problem["unit_discharge_offset_min"].notna()
        & (
            problem["hospital_discharge_offset_min"]
            < problem["unit_discharge_offset_min"]
        )
    ).astype("int8")

    problem.to_csv(
        OUT_MAPPING_DETAIL,
        index=False,
        encoding="utf-8-sig",
    )

    # Hospital-level map quality.
    site_rows = []
    for hid, g in x.groupby("hospitalid", dropna=False):
        gc = (
            g["hospital_discharge_status"].notna()
            & g["hospital_discharge_offset_min"].notna()
        )
        gi = (
            g["hospital_discharge_offset_min"].notna()
            & (
                g["hospital_discharge_offset_min"]
                <= g["respiratory_t0_candidate_min"]
            )
        )
        gu = (
            g["hospital_discharge_offset_min"].notna()
            & g["unit_discharge_offset_min"].notna()
            & (
                g["hospital_discharge_offset_min"]
                < g["unit_discharge_offset_min"]
            )
        )
        site_rows.append({
            "hospitalid": hid,
            "locked_stays_n": len(g),
            "outcome_complete_n": int(gc.sum()),
            "outcome_complete_pct": float(gc.mean()) if len(g) else np.nan,
            "discharge_at_or_before_t0_n": int(gi.sum()),
            "hospital_before_unit_discharge_n": int(gu.sum()),
        })

    pd.DataFrame(site_rows).sort_values(
        ["discharge_at_or_before_t0_n", "locked_stays_n"],
        ascending=[False, False],
    ).to_csv(
        OUT_MAPPING_SITE,
        index=False,
        encoding="utf-8-sig",
    )

    gate.to_csv(OUT_GATE, index=False, encoding="utf-8-sig")

    print("\nPRE-EFFECT OUTCOME MAPPING GATE:")
    print(gate.to_string(index=False))

    if not mapping_pass:
        raise RuntimeError(
            "E3.1 ABORTED before effect estimation: detailed outcome mapping "
            "preflight failed. Inspect E3_00/00B/00C CSVs."
        )

    # Complete-case endpoint mapping was predeclared in E3.0 when completeness
    # is >=95%.
    keep = complete.copy()

    # Narrow timestamp data-quality exclusion, only if the fixed rule passed.
    if allowed_timestamp_exclusion:
        keep &= ~timestamp_inconsistent
        print(
            f"[PRE-EFFECT DATA QUALITY] Excluding {inconsistent_n} "
            f"({100*inconsistent_pct:.3f}%) internally inconsistent "
            "hospital-discharge timestamp rows before opening the effect."
        )

    cohort2 = x.loc[keep].copy()
    return cohort2, gate


def grace_observation_preflight_from_cohort(cohort):
    """
    Outcome-blind audit of whether eICU treatment observation remains available
    through the full 6h strategy window.

    This is computed BEFORE dynamic covariate extraction and BEFORE the external
    treatment effect is opened.
    """
    rows = []

    for h in range(0, 20, 2):
        start_abs = (
            cohort["respiratory_t0_candidate_min"]
            + h * 60
        )

        steroid = cohort["steroid_relative_to_t0_min"]
        untreated = steroid.isna() | (steroid >= h * 60)

        still_icu_at_start = (
            cohort["unit_discharge_offset_min"].isna()
            | (
                cohort["unit_discharge_offset_min"]
                > start_abs
            )
        )

        elig = untreated & still_icu_at_start
        end_abs = start_abs + WINDOW_H * 60

        early_exit = (
            elig
            & cohort["unit_discharge_offset_min"].notna()
            & (
                cohort["unit_discharge_offset_min"]
                < end_abs
            )
        )

        n = int(elig.sum())
        ex = int(early_exit.sum())

        rows.append({
            "trial_hour": h,
            "eligible_stays_n": n,
            "unit_discharge_before_grace_end_n": ex,
            "unit_discharge_before_grace_end_pct": (
                ex / n if n else np.nan
            ),
        })

    df = pd.DataFrame(rows)

    total_eligible = int(df["eligible_stays_n"].sum())
    total_exit = int(df["unit_discharge_before_grace_end_n"].sum())
    overall_pct = (
        total_exit / total_eligible
        if total_eligible
        else np.nan
    )

    df["overall_sequential_person_trials_n"] = total_eligible
    df["overall_unit_discharge_before_grace_end_n"] = total_exit
    df["overall_unit_discharge_before_grace_end_pct"] = overall_pct
    df["predeclared_max_pct"] = MAX_TRIALS_WITH_ICU_DISCHARGE_INSIDE_GRACE
    df["grace_observation_gate_pass"] = int(
        np.isfinite(overall_pct)
        and overall_pct <= MAX_TRIALS_WITH_ICU_DISCHARGE_INSIDE_GRACE
    )

    df.to_csv(
        OUT_GRACE_PREFLIGHT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nPRE-EFFECT 6H TREATMENT-OBSERVATION GATE:")
    print(df.to_string(index=False))

    if int(df.iloc[0]["grace_observation_gate_pass"]) != 1:
        raise RuntimeError(
            "E3.1 ABORTED before effect estimation: >5% of eligible "
            "sequential trials lose ICU treatment observation before the "
            "6h grace window ends. Inspect E3_00D_PRE_EFFECT_GRACE_OBSERVATION_AUDIT.csv."
        )

    return df


def load_or_build_analysis_data(e2, cohort):
    if CACHE_PANEL.exists() and CACHE_TRIALS.exists():
        panel = pd.read_parquet(CACHE_PANEL)
        trials = pd.read_parquet(CACHE_TRIALS)

        expected = set(cohort["patientunitstayid"].astype(int))
        cached = set(trials["patientunitstayid"].dropna().astype(int))
        if cached.issubset(expected):
            print("[CACHE] Reusing E3 dynamic panel / trial cache.")
            return panel, trials

    print("Building eICU dynamic panel with frozen E2 mapping...")
    panel = e2.build_dynamic_panel(cohort)
    trials = e2.build_trials(cohort, panel)

    panel.to_parquet(CACHE_PANEL, index=False)
    trials.to_parquet(CACHE_TRIALS, index=False)
    return panel, trials


def grace_observation_gate(trials):
    """
    eICU treatment data stop at ICU-unit discharge.
    Audit whether unit discharge occurs before the 6h strategy window ends.

    No patient is selected/excluded based on this audit; it is a pre-effect
    identifiability gate. If >5% of sequential trials lose treatment observation
    inside the grace period, E3 aborts rather than silently misclassifying them.
    """
    end_min = (
        trials["respiratory_t0_candidate_min"]
        + (trials["trial_hour"] + WINDOW_H) * 60
    )
    early_unit_exit = (
        trials["unit_discharge_offset_min"].notna()
        & (trials["unit_discharge_offset_min"] < end_min)
    )

    pct = float(early_unit_exit.mean())
    return pd.DataFrame([{
        "sequential_person_trials_n": len(trials),
        "unit_discharge_before_grace_end_n": int(early_unit_exit.sum()),
        "unit_discharge_before_grace_end_pct": pct,
        "predeclared_max_pct": MAX_TRIALS_WITH_ICU_DISCHARGE_INSIDE_GRACE,
        "grace_observation_gate_pass": int(
            pct <= MAX_TRIALS_WITH_ICU_DISCHARGE_INSIDE_GRACE
        ),
    }])


def prepare_models(e2, trials, panel):
    trials = trials.reset_index(drop=True).copy()
    trials["trial_idx"] = np.arange(len(trials), dtype=int)

    # Baseline frozen model.
    prep_base = e2.make_preprocessor(trials, False)
    X_base = prep_base.fit_transform(trials)
    y6 = trials["actual_initiate_next6h"].astype(int).to_numpy()

    # Defer hourly model.
    long = e2.build_defer_long(trials, panel)
    prep_defer = e2.make_preprocessor(long, True)
    X_defer = prep_defer.fit_transform(long)
    y_hour = long["start_next1h"].astype(int).to_numpy()

    # Recent model.
    recent = trials.copy()
    recent["decision_hour"] = recent["trial_hour"] + 3
    recent["decision_hour2"] = recent["decision_hour"] ** 2

    steroid = recent["steroid_relative_to_t0_min"]
    untreated = steroid.isna() | (
        steroid >= recent["decision_hour"] * 60
    )
    still_icu = (
        recent["unit_discharge_offset_min"].isna()
        | (
            recent["unit_discharge_offset_min"]
            > recent["respiratory_t0_candidate_min"]
            + recent["decision_hour"] * 60
        )
    )
    recent = recent[untreated & still_icu].copy()

    steroid = recent["steroid_relative_to_t0_min"]
    recent["recent_initiate_3_6h"] = (
        steroid.notna()
        & (steroid >= recent["decision_hour"] * 60)
        & (steroid < (recent["trial_hour"] + 6) * 60)
    ).astype("int8")

    recent = recent.drop(
        columns=e2.existing(e2.DYNAMIC_NUM + e2.DYNAMIC_BIN, recent),
        errors="ignore",
    )
    recent = recent.merge(
        panel,
        left_on=["patientunitstayid", "decision_hour"],
        right_on=["patientunitstayid", "hour"],
        how="left",
    ).drop(columns=["hour"], errors="ignore")

    prep_recent = e2.make_preprocessor(recent, True)
    X_recent = prep_recent.fit_transform(recent)
    y_recent = recent["recent_initiate_3_6h"].astype(int).to_numpy()

    # Cluster IDs at patient level.
    cluster_ids = trials["cluster_id"].astype("string").to_numpy()
    unique_clusters, inverse_cluster = np.unique(
        cluster_ids,
        return_inverse=True,
    )

    # Map long/recent rows to cluster index through trial_idx.
    trial_cluster = inverse_cluster
    long_cluster = trial_cluster[long["trial_idx"].to_numpy(int)]
    recent_cluster = trial_cluster[recent["trial_idx"].to_numpy(int)]

    return {
        "trials": trials,
        "long": long,
        "recent": recent,
        "X_base": X_base,
        "y6": y6,
        "X_defer": X_defer,
        "y_hour": y_hour,
        "X_recent": X_recent,
        "y_recent": y_recent,
        "unique_clusters": unique_clusters,
        "inverse_cluster": inverse_cluster,
        "long_cluster": long_cluster,
        "recent_cluster": recent_cluster,
    }


def fit_nuisance(prep, cluster_mult=None):
    n = len(prep["trials"])

    if cluster_mult is None:
        row_mult = np.ones(n, float)
    else:
        row_mult = cluster_mult[prep["inverse_cluster"]]

    mask_base = row_mult > 0
    m_base = fit_unpenalized_weighted(
        prep["X_base"][mask_base],
        prep["y6"][mask_base],
        sample_weight=row_mult[mask_base],
    )
    e6 = np.clip(
        m_base.predict_proba(prep["X_base"])[:, 1],
        PS_CLIP,
        1 - PS_CLIP,
    )
    q = e6 * (1 - e6)

    long_mult = row_mult[prep["long_cluster"]]
    mask_long = long_mult > 0
    m_defer = LogisticRegression(
        C=0.5,
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    m_defer.fit(
        prep["X_defer"][mask_long],
        prep["y_hour"][mask_long],
        sample_weight=long_mult[mask_long],
    )
    p_hour = np.clip(
        m_defer.predict_proba(prep["X_defer"])[:, 1],
        PS_CLIP,
        1 - PS_CLIP,
    )

    recent_mult = row_mult[prep["recent_cluster"]]
    mask_recent = recent_mult > 0
    m_recent = fit_unpenalized_weighted(
        prep["X_recent"][mask_recent],
        prep["y_recent"][mask_recent],
        sample_weight=recent_mult[mask_recent],
    )
    p_recent = np.clip(
        m_recent.predict_proba(prep["X_recent"])[:, 1],
        PS_CLIP,
        1 - PS_CLIP,
    )

    return row_mult, e6, q, p_hour, p_recent


def map_hourly_probs(prep, p_hour):
    n = len(prep["trials"])
    pmat = np.full((n, 6), np.nan, float)
    ti = prep["long"]["trial_idx"].to_numpy(int)
    jj = prep["long"]["interval_index"].to_numpy(int)
    pmat[ti, jj] = p_hour
    return pmat


def map_recent_probs(prep, p_recent):
    n = len(prep["trials"])
    out = np.full(n, np.nan, float)
    ti = prep["recent"]["trial_idx"].to_numpy(int)
    out[ti] = p_recent
    return out


def build_competing_geometry(prep):
    tr = prep["trials"]
    n = len(tr)

    baseline_min = (
        tr["respiratory_t0_candidate_min"].to_numpy(float)
        + tr["trial_hour"].to_numpy(float) * 60
    )

    hosp_end_abs = tr["hospital_discharge_offset_min"].to_numpy(float)
    hosp_rel_h = (hosp_end_abs - baseline_min) / 60.0

    death = tr["hospital_death"].astype(int).to_numpy() == 1

    # Cause 1 = death; cause 2 = alive hospital discharge.
    cause = np.where(death, 1, 2)

    # Only events/discharges after trial baseline and <=28 days are observed
    # as competing events inside the analysis horizon.
    valid_event = (
        np.isfinite(hosp_rel_h)
        & (hosp_rel_h > 0)
        & (hosp_rel_h <= 28 * 24)
    )
    event_time = np.where(valid_event, hosp_rel_h, np.inf)
    event_cause = np.where(valid_event, cause, 0)

    init_rel_h = (
        tr["steroid_relative_to_t0_min"].to_numpy(float)
        / 60.0
        - tr["trial_hour"].to_numpy(float)
    )
    y6 = prep["y6"]

    horizon = np.full(n, 28 * 24.0)

    # Defer clone: artificial censor at steroid initiation inside 6h.
    art0 = np.where(y6 == 1, init_rel_h, np.inf)

    # Initiate clone: non-initiator artificial censor at +6h.
    art1 = np.where(y6 == 1, np.inf, 6.0)

    def arm_geom(art):
        end = np.minimum.reduce([horizon, event_time, art])

        # Clinical event wins ties against artificial censor.
        clinical = (
            (event_cause > 0)
            & (event_time <= horizon)
            & (event_time <= art)
        )
        etype = np.where(clinical, event_cause, 0)
        end = np.where(clinical, event_time, end)
        return np.maximum(end, 1e-9), etype

    end0, type0 = arm_geom(art0)
    end1, type1 = arm_geom(art1)

    return {
        "end0": end0,
        "type0": type0,
        "end1": end1,
        "type1": type1,
        "init_rel_h": init_rel_h,
    }


def aj_phase(end, etype, weight, lower, upper, surv, cif_death):
    """
    Weighted Aalen-Johansen update over one interval.

    cause 1 = hospital death
    cause 2 = alive hospital discharge

    Artificial censoring contributes to the risk set until censor time,
    but not to either event count.
    """
    risk_phase = end > lower
    if not risk_phase.any():
        return surv, cif_death

    d = pd.DataFrame({
        "end": end[risk_phase],
        "risk_w": weight[risk_phase],
        "death_w": weight[risk_phase] * (etype[risk_phase] == 1),
        "discharge_w": weight[risk_phase] * (etype[risk_phase] == 2),
    })

    g = d.groupby("end", as_index=False, sort=True).sum(numeric_only=True)
    g = g.sort_values("end", ascending=False).reset_index(drop=True)
    g["risk_cum"] = g["risk_w"].cumsum()

    ev = g[
        (g["end"] > lower)
        & (g["end"] <= upper)
        & ((g["death_w"] > 0) | (g["discharge_w"] > 0))
    ].copy()

    if ev.empty:
        return surv, cif_death

    ev = ev.sort_values("end", ascending=True)

    for r in ev.itertuples(index=False):
        if r.risk_cum <= 0:
            continue
        h_death = max(float(r.death_w) / float(r.risk_cum), 0.0)
        h_dis = max(float(r.discharge_w) / float(r.risk_cum), 0.0)
        h_all = min(h_death + h_dis, 1.0)

        cif_death += surv * h_death
        surv *= 1.0 - h_all

    return surv, cif_death


def estimate_effect(prep, q, p_hour, p_recent, row_mult):
    tr = prep["trials"]
    geom = build_competing_geometry(prep)
    n = len(tr)

    freq = np.asarray(row_mult, float)
    base_q = freq * q

    pmat = map_hourly_probs(prep, p_hour)
    pmat = np.where(np.isfinite(pmat), pmat, 0.5)
    pmat = np.clip(pmat, PS_CLIP, 1 - PS_CLIP)
    inv_no = 1.0 / (1.0 - pmat)
    cum = np.cumprod(inv_no, axis=1)

    p_recent_trial = map_recent_probs(prep, p_recent)
    p_recent_trial = np.where(
        np.isfinite(p_recent_trial),
        p_recent_trial,
        0.5,
    )
    p_recent_trial = np.clip(
        p_recent_trial,
        PS_CLIP,
        1 - PS_CLIP,
    )

    y6 = prep["y6"]
    init_rel_h = geom["init_rel_h"]

    early = (
        (y6 == 1)
        & np.isfinite(init_rel_h)
        & (init_rel_h >= 0)
        & (init_rel_h < 3)
    )
    recent = (
        (y6 == 1)
        & np.isfinite(init_rel_h)
        & (init_rel_h >= 3)
        & (init_rel_h < 6)
    )

    # DEFER Aalen-Johansen with hourly IPCW.
    surv0 = 1.0
    cif0 = 0.0

    phase_weights0 = [
        base_q,
        base_q * cum[:, 0],
        base_q * cum[:, 1],
        base_q * cum[:, 2],
        base_q * cum[:, 3],
        base_q * cum[:, 4],
        base_q * cum[:, 5],
    ]
    bounds = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 28 * 24),
    ]

    for w, (lo, hi) in zip(phase_weights0, bounds):
        surv0, cif0 = aj_phase(
            geom["end0"],
            geom["type0"],
            w,
            lo,
            hi,
            surv0,
            cif0,
        )

    # INITIATE arm: q during grace; q for early, q/p_recent for recent after grace.
    surv1 = 1.0
    cif1 = 0.0
    for lo, hi in bounds[:6]:
        surv1, cif1 = aj_phase(
            geom["end1"],
            geom["type1"],
            base_q,
            lo,
            hi,
            surv1,
            cif1,
        )

    post1 = np.zeros(n, float)
    post1[early] = base_q[early]
    post1[recent] = base_q[recent] / p_recent_trial[recent]

    surv1, cif1 = aj_phase(
        geom["end1"],
        geom["type1"],
        post1,
        6,
        28 * 24,
        surv1,
        cif1,
    )

    return {
        "risk_defer": cif0,
        "risk_initiate": cif1,
        "risk_difference": cif1 - cif0,
        "risk_ratio": cif1 / cif0 if cif0 > 0 else np.nan,
        "early": early,
        "recent": recent,
        "p_recent_trial": p_recent_trial,
        "defer_post_weight": base_q * cum[:, 5],
        "initiate_post_weight": post1,
        "cumulative_defer_ipcw": cum[:, 5],
    }


def checkpoint_load():
    cols = [
        "rep",
        "risk_defer",
        "risk_initiate",
        "risk_difference",
        "risk_ratio",
    ]
    if not CHECKPOINT.exists():
        return pd.DataFrame(columns=cols)
    try:
        x = pd.read_csv(CHECKPOINT)
        if not set(cols).issubset(x.columns):
            return pd.DataFrame(columns=cols)
        return x[cols].copy()
    except Exception:
        return pd.DataFrame(columns=cols)


def checkpoint_save(df):
    tmp = CHECKPOINT.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(CHECKPOINT)


def cluster_bootstrap(prep, target, seed):
    old = checkpoint_load()
    completed = len(old)

    if completed >= target:
        print(f"[RESUME] checkpoint already has {completed}; target={target}")
        return old.iloc[:target].copy()

    if completed:
        print(f"[RESUME] loaded {completed}/{target} from {CHECKPOINT.name}")

    rng = np.random.default_rng(seed)
    nclusters = len(prep["unique_clusters"])

    # Replay draws to preserve deterministic resume.
    for _ in range(completed):
        rng.integers(0, nclusters, size=nclusters)

    records = old.to_dict("records")
    success = completed
    attempts = completed

    while success < target:
        attempts += 1
        draw = rng.integers(0, nclusters, size=nclusters)
        counts = np.bincount(draw, minlength=nclusters).astype(float)

        row_mult = counts[prep["inverse_cluster"]]
        if row_mult.sum() <= 0:
            continue

        # Treatment-process support safeguard only.
        if prep["y6"][row_mult > 0].sum() < 30:
            continue

        recent_mult = counts[prep["recent_cluster"]]
        if prep["y_recent"][recent_mult > 0].sum() < 20:
            continue

        try:
            row_mult, e6, q, p_hour, p_recent = fit_nuisance(
                prep,
                cluster_mult=counts,
            )
            est = estimate_effect(
                prep,
                q,
                p_hour,
                p_recent,
                row_mult,
            )

            success += 1
            records.append({
                "rep": success,
                "risk_defer": est["risk_defer"],
                "risk_initiate": est["risk_initiate"],
                "risk_difference": est["risk_difference"],
                "risk_ratio": est["risk_ratio"],
            })

        except Exception as exc:
            print(
                f"[WARN] bootstrap attempt {attempts} failed "
                f"({type(exc).__name__}); continuing"
            )
            continue

        if success % CHECKPOINT_EVERY == 0 or success == target:
            cur = pd.DataFrame(records)
            checkpoint_save(cur)
            print(f"checkpoint saved: {success}/{target}")

    return pd.DataFrame(records).iloc[:target].copy()


def ci95(x):
    z = pd.to_numeric(
        pd.Series(x),
        errors="coerce",
    ).dropna().to_numpy(float)
    if len(z) == 0:
        return np.nan, np.nan
    return tuple(np.quantile(z, [0.025, 0.975]))


def endpoint_audit(trials):
    baseline_min = (
        trials["respiratory_t0_candidate_min"]
        + trials["trial_hour"] * 60
    )
    rel = trials["hospital_discharge_offset_min"] - baseline_min

    death = trials["hospital_death"].astype(int) == 1
    alive = ~death

    return pd.DataFrame([{
        "person_trials_n": len(trials),
        "unique_stays_n": trials["patientunitstayid"].nunique(),
        "unique_patient_clusters_n": trials["cluster_id"].nunique(),
        "hospital_death_within28d_person_trials_n": int(
            (death & rel.gt(0) & rel.le(FOLLOWUP_MIN)).sum()
        ),
        "alive_discharge_within28d_person_trials_n": int(
            (alive & rel.gt(0) & rel.le(FOLLOWUP_MIN)).sum()
        ),
        "hospital_discharge_after28d_or_no_event_by28d_n": int(
            (rel.gt(FOLLOWUP_MIN)).sum()
        ),
        "invalid_nonpositive_discharge_time_from_trial_n": int(
            rel.le(0).sum()
        ),
        "endpoint_estimator": "weighted Aalen-Johansen cumulative incidence",
        "competing_event": "alive hospital discharge",
    }])


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 116)
    print("E3.1 FINAL — eICU locked external effect with pre-effect data-quality preflight")
    print("=" * 116)

    enforce_e2_lock()
    e2 = import_e2()

    cohort, site_gate = rebuild_locked_cohort(e2)
    cohort, mapping_gate = pre_effect_mapping_gate(cohort)

    # Check treatment-observation identifiability BEFORE building the dynamic
    # panel and BEFORE opening the external effect.
    grace_preflight = grace_observation_preflight_from_cohort(cohort)

    print(
        f"Outcome-complete locked cohort: {len(cohort)} stays; "
        f"{cohort.cluster_id.nunique()} patient clusters"
    )

    panel, trials = load_or_build_analysis_data(e2, cohort)

    # Ensure outcome / cluster columns are present on cached/new trial table.
    needed = [
        "hospital_discharge_status",
        "hospital_discharge_offset_min",
        "hospital_death",
        "cluster_id",
    ]
    if not set(needed).issubset(trials.columns):
        trials = trials.drop(columns=[c for c in needed if c in trials.columns], errors="ignore")
        trials = trials.merge(
            cohort[["patientunitstayid"] + needed],
            on="patientunitstayid",
            how="left",
            validate="many_to_one",
        )
        trials.to_parquet(CACHE_TRIALS, index=False)

    grace_gate = grace_observation_gate(trials)

    gate = mapping_gate.copy()
    for c in grace_gate.columns:
        gate[c] = grace_gate.iloc[0][c]

    gate["effect_opened_after_all_pre_effect_gates"] = int(
        int(gate.iloc[0]["mapping_gate_pass"]) == 1
        and int(gate.iloc[0]["grace_observation_gate_pass"]) == 1
    )
    gate.to_csv(OUT_GATE, index=False, encoding="utf-8-sig")

    if int(gate.iloc[0]["effect_opened_after_all_pre_effect_gates"]) != 1:
        raise RuntimeError(
            "E3 ABORTED before treatment-effect estimation: "
            "pre-effect mapping/observation gate failed."
        )

    # Re-run frozen E2 support audit on the outcome-complete cohort before effect.
    prep0 = e2.make_preprocessor(trials, False)
    X0 = prep0.fit_transform(trials)
    y0 = trials["actual_initiate_next6h"].astype(int).to_numpy()
    m0 = fit_unpenalized_weighted(X0, y0)
    e6_check = np.clip(m0.predict_proba(X0)[:, 1], PS_CLIP, 1 - PS_CLIP)
    q_check = e6_check * (1 - e6_check)

    long_check = e2.build_defer_long(trials.reset_index(drop=True).assign(
        trial_idx=np.arange(len(trials), dtype=int)
    ), panel)
    defer_bal, defer_sum = e2.defer_audit(
        trials.reset_index(drop=True).assign(
            trial_idx=np.arange(len(trials), dtype=int)
        ),
        long_check,
        q_check,
    )
    recent_bal, recent_sum, recent_site = e2.recent_audit(
        trials.reset_index(drop=True).assign(
            trial_idx=np.arange(len(trials), dtype=int)
        ),
        panel,
        q_check,
    )

    rr = recent_sum.iloc[0]
    support_pass = (
        (defer_sum["max_abs_smd"] < 0.10).all()
        and (defer_sum["n_abs_smd_gt_0_10"] == 0).all()
        and 0.90 <= rr["representation_ratio"] <= 1.10
        and rr["recent_weight_ess"] >= 50
        and rr["max_abs_smd"] < 0.10
        and int(rr["n_abs_smd_gt_0_10"]) == 0
    )

    if not support_pass:
        raise RuntimeError(
            "E3 ABORTED before effect estimation: frozen support gate "
            "failed after restricting to outcome-complete stays."
        )

    # NOW the external effect is opened.
    prep = prepare_models(e2, trials, panel)
    row_mult, e6, q, p_hour, p_recent = fit_nuisance(prep, None)

    point = estimate_effect(
        prep,
        q,
        p_hour,
        p_recent,
        row_mult,
    )

    print("\nLOCKED EXTERNAL POINT ESTIMATE OPENED:")
    print(f"  risk defer >=6h    = {point['risk_defer']:.6f}")
    print(f"  risk initiate <=6h = {point['risk_initiate']:.6f}")
    print(f"  RD                  = {point['risk_difference']:.6f}")
    print(f"  RR                  = {point['risk_ratio']:.6f}")

    boot = cluster_bootstrap(
        prep,
        BOOTSTRAP_TARGET,
        SEED,
    )

    ci0 = ci95(boot["risk_defer"])
    ci1 = ci95(boot["risk_initiate"])
    cird = ci95(boot["risk_difference"])
    cirr = ci95(boot["risk_ratio"])

    effect = pd.DataFrame([{
        "population": "eICU_locked_external",
        "analysis_status": "FINAL_FROZEN_EXTERNAL_EFFECT",
        "estimand": "ATO_overlap_population",
        "strategy_1": "initiate systemic corticosteroid within next 6h",
        "strategy_0": "defer systemic corticosteroid for at least 6h",
        "recent_window": "[3h,6h)",
        "external_outcome": EXPECTED_EXTERNAL_OUTCOME,
        "endpoint_replication": "partial_vs_MIMIC_28d_all_cause",
        "locked_unit_stays_n": prep["trials"]["patientunitstayid"].nunique(),
        "patient_clusters_n": len(prep["unique_clusters"]),
        "person_trials_n": len(prep["trials"]),
        "initiate_within6h_person_trials_n": int(prep["y6"].sum()),
        "recent_3_6h_person_trials_n": int(prep["y_recent"].sum()),
        "bootstrap_target_n": BOOTSTRAP_TARGET,
        "bootstrap_success_n": len(boot),
        "risk_defer": point["risk_defer"],
        "risk_defer_ci_low": ci0[0],
        "risk_defer_ci_high": ci0[1],
        "risk_initiate": point["risk_initiate"],
        "risk_initiate_ci_low": ci1[0],
        "risk_initiate_ci_high": ci1[1],
        "risk_difference": point["risk_difference"],
        "rd_ci_low": cird[0],
        "rd_ci_high": cird[1],
        "risk_ratio": point["risk_ratio"],
        "rr_ci_low": cirr[0],
        "rr_ci_high": cirr[1],
    }])
    effect.to_csv(OUT_EFFECT, index=False, encoding="utf-8-sig")

    bsum = pd.DataFrame([{
        "bootstrap_success_n": len(boot),
        "rd_bootstrap_median": float(boot["risk_difference"].median()),
        "rd_bootstrap_p025": cird[0],
        "rd_bootstrap_p975": cird[1],
        "rr_bootstrap_median": float(boot["risk_ratio"].median()),
        "rr_bootstrap_p025": cirr[0],
        "rr_bootstrap_p975": cirr[1],
        "checkpoint_file": str(CHECKPOINT),
    }])
    bsum.to_csv(OUT_BOOT, index=False, encoding="utf-8-sig")

    recent_mask = point["recent"]
    diag = pd.DataFrame([{
        "baseline_e6_model": "unpenalized logistic",
        "defer_censor_model": "pooled 1h ridge logistic C=0.5",
        "recent_model": "unpenalized logistic at +3h for initiation [3h,6h)",
        "baseline_q_sum": float(q.sum()),
        "baseline_q_ess": ess(q),
        "recent_person_trials_n": int(recent_mask.sum()),
        "recent_ipcw_p95": float(
            np.quantile(
                1.0 / point["p_recent_trial"][recent_mask],
                0.95,
            )
        ),
        "recent_ipcw_p99": float(
            np.quantile(
                1.0 / point["p_recent_trial"][recent_mask],
                0.99,
            )
        ),
        "recent_ipcw_max": float(
            np.max(1.0 / point["p_recent_trial"][recent_mask])
        ),
        "defer_postgrace_weight_ess": ess(
            point["defer_post_weight"][prep["y6"] == 0]
        ),
        "initiate_postgrace_weight_ess": ess(
            point["initiate_post_weight"][prep["y6"] == 1]
        ),
        "cumulative_defer_ipcw_p99": float(
            np.quantile(point["cumulative_defer_ipcw"], 0.99)
        ),
        "cumulative_defer_ipcw_max": float(
            np.max(point["cumulative_defer_ipcw"])
        ),
        "support_recheck_recent_representation_ratio": float(
            rr["representation_ratio"]
        ),
        "support_recheck_recent_weight_ess": float(
            rr["recent_weight_ess"]
        ),
        "support_recheck_recent_max_abs_smd": float(
            rr["max_abs_smd"]
        ),
        "support_recheck_defer_max_abs_smd": float(
            defer_sum["max_abs_smd"].max()
        ),
    }])
    diag.to_csv(OUT_DIAG, index=False, encoding="utf-8-sig")

    trial_counts = (
        prep["trials"]
        .groupby("trial_hour", as_index=False)
        .agg(
            eligible_person_trials_n=("trial_idx", "size"),
            initiate_within6h_n=("actual_initiate_next6h", "sum"),
            unique_stays_n=("patientunitstayid", "nunique"),
        )
    )
    trial_counts["initiate_within6h_rate"] = (
        trial_counts["initiate_within6h_n"]
        / trial_counts["eligible_person_trials_n"]
    )
    trial_counts.to_csv(OUT_TRIAL, index=False, encoding="utf-8-sig")

    recent_site.to_csv(OUT_SITE, index=False, encoding="utf-8-sig")
    endpoint_audit(prep["trials"]).to_csv(
        OUT_ENDPOINT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 116)
    print("E3 FINAL EXTERNAL EFFECT COMPLETE")
    print("=" * 116)
    print("\nUpload these seven files:")
    for p in [
        OUT_GATE,
        OUT_EFFECT,
        OUT_BOOT,
        OUT_DIAG,
        OUT_TRIAL,
        OUT_SITE,
        OUT_ENDPOINT,
    ]:
        print(" ", p)

    print(
        "\nThe eICU effect is now opened and frozen. "
        "Do not change window/cohort/nuisance models based on this result."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise

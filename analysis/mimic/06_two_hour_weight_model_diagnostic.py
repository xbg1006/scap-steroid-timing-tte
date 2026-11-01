from __future__ import annotations

"""
final causal-analysis stage — Sequential nested target trials (2-hour clinical decision trials)

CLINICAL QUESTION
-----------------
Among adults with severe CAP who are alive and have not yet received a systemic
corticosteroid at a given decision time:

    Strategy A: initiate systemic corticosteroid during the next 2 hours
    Strategy B: defer initiation for at least the next 2 hours

After the 2-hour decision window, treatment is unrestricted.

WHY THIS DESIGN
---------------
A patient can be eligible repeatedly while untreated. Instead of asking only
"did the patient initiate by 24h?", we emulate a sequence of trials at:

    t = 0, 2, 4, ..., 22 hours after respiratory t0.

A non-initiator in an early trial may later enter another trial and become an
initiator. This uses the data more efficiently and reduces reliance on the very
small number of late-window recent initiators that limited Stage 9.

Each 2-hour mini-trial uses clone-censor logic:
- initiate arm: censor at +2h only if no initiation occurred and the patient
  survived event-free to the end of the window
- defer arm: censor at actual steroid initiation if it occurs inside the 2h window
- deaths before artificial censoring count as events

At the end of the 2-hour window, adherent clones receive stabilized inverse
probability weights based ONLY on covariates measured at the trial start.

PRIMARY POPULATION
------------------
Stage 7 broad primary cohort (major immunocompromise excluded), n~2014.

HIGH-SPECIFICITY REPLICATION
----------------------------
CAPE-COD-like cohort from Stage 8B, n~707.

OUTCOME
-------
28-day all-cause mortality measured from EACH sequential trial baseline.

PRIMARY ESTIMAND
----------------
Average 28-day risk difference / risk ratio across eligible 2-hour decision trials:
"initiate during next 2h" versus "defer at least 2h".

This is a different estimand from Stage 9's "initiate by 6/12/24h" effects.

FILES REQUIRED
--------------
outputs/stage7_hourly_panel.parquet
outputs/stage5_5_analysis_cohort.parquet
14_run_high_specificity_ccw.py  (for CAPE-COD-like patient IDs)

GENERATED AUDIT OUTPUTS
-----------------
outputs/16_FINAL_SEQUENTIAL_PRIMARY.csv
outputs/16B_FINAL_TRIAL_HOUR_EFFECTS.csv
outputs/16C_FINAL_PROPENSITY_DIAGNOSTICS.csv
outputs/16D_FINAL_BALANCE.csv
outputs/16E_FINAL_CAPECOD_REPLICATION.csv
outputs/16F_FINAL_MODEL_DIAGNOSTICS.csv
"""

from pathlib import Path
import importlib.util
import math
import traceback
import warnings

import numpy as np
import pandas as pd

import config
from src.utils import connect_duckdb, find_table, source_sql

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    raise RuntimeError(
        "final causal-analysis stage needs scikit-learn.\n"
        "Run in the PyCharm terminal:\n"
        "    pip install scikit-learn\n"
        f"Original error: {e}"
    )

PROJECT_ROOT = Path(config.PROJECT_ROOT)
DATA_ROOT = Path(config.DATA_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

PANEL_PATH = OUTPUT_DIR / "stage7_hourly_panel.parquet"
COHORT_PATH = OUTPUT_DIR / "stage5_5_analysis_cohort.parquet"
STAGE8B_SCRIPT = PROJECT_ROOT / "14_run_high_specificity_ccw.py"

OUT_EFFECT = OUTPUT_DIR / "16V3_FINAL_WEIGHT_SCHEME_EFFECTS.csv"
OUT_BAL = OUTPUT_DIR / "16V3_FINAL_BALANCE.csv"
OUT_PS = OUTPUT_DIR / "16V3_FINAL_PROPENSITY_DIAGNOSTICS.csv"
OUT_HOUR = OUTPUT_DIR / "16V3_FINAL_TRIAL_HOUR_DIAGNOSTICS.csv"
OUT_MODEL = OUTPUT_DIR / "16V3_FINAL_MODEL_DIAGNOSTICS.csv"

WINDOW_H = 2
TRIAL_STARTS = list(range(0, 24, 2))
FOLLOWUP_H = 28 * 24

SEED = 20261027

# V3 is intentionally diagnostic-only. No bootstrap is run here.
# We first require acceptable balance/overlap before spending hours on inference.
FIXED_WEIGHT_CAP = 10.0
OLD_TRUNC_LOW = 0.01
OLD_TRUNC_HIGH = 0.99

BASELINE_NUM = [
    "age",
    "hospital_to_respiratory_t0_h",
    "weight_kg",
]
BASELINE_BIN = [
    "shock_active_at_t0",
    "copd_history",
    "asthma_history",
    "heart_failure",
    "coronary_disease",
    "diabetes",
    "chronic_kidney_disease",
    "chronic_liver_disease",
    "malignancy",
    "dementia",
    "pneumonia_top3_dx",
]
BASELINE_CAT = [
    "gender",
    "race_group",
    "respiratory_t0_type",
    "careunit_group",
    "anchor_year_group",
]

# Current physiology at each sequential trial baseline.
CURRENT_NUM = [
    "dyn_hr_max6",
    "dyn_rr_max6",
    "dyn_map_min6",
    "dyn_spo2_last6",
    "dyn_fio2_last6",
    "dyn_sf_ratio",
    "dyn_temp_max6",
    "dyn_gcs_motor_last6",
    "dyn_lactate_max12",
    "dyn_ph_min12",
    "dyn_po2_last12",
    "dyn_pco2_last12",
    "dyn_creatinine_last24",
    "dyn_bicarbonate_last24",
    "dyn_glucose_last24",
    "dyn_wbc_last24",
    "support_level",
    "vasopressor_agents_active",
]
CURRENT_BIN = [
    "vasopressor_active",
    "support_imv",
    "support_niv",
    "support_hfnc",
]


def import_stage8b():
    if not STAGE8B_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing {STAGE8B_SCRIPT}\n"
            "Keep 14_run_high_specificity_ccw.py in the project root."
        )
    spec = importlib.util.spec_from_file_location("stage8b_for_stage10", STAGE8B_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def safe_dt(x):
    return pd.to_datetime(x, errors="coerce")


def attach_full_death_hours() -> pd.DataFrame:
    """
    Reconstruct death time without truncating at 28 days from the original
    respiratory t0, because each sequential trial needs 28 days of follow-up
    from its own later trial baseline.
    """
    if not COHORT_PATH.exists():
        raise FileNotFoundError(f"Missing {COHORT_PATH}")

    c = pd.read_parquet(COHORT_PATH)[
        ["subject_id", "hadm_id", "stay_id", "respiratory_t0"]
    ].copy()
    c["respiratory_t0"] = safe_dt(c["respiratory_t0"])

    adm_path = find_table(DATA_ROOT, "hosp", "admissions")
    pat_path = find_table(DATA_ROOT, "hosp", "patients")

    con = connect_duckdb(PROJECT_ROOT, getattr(config, "DUCKDB_THREADS", 4))
    try:
        adm = source_sql(adm_path)
        pat = source_sql(pat_path)

        a = con.execute(f"""
            SELECT hadm_id, deathtime
            FROM {adm}
        """).fetchdf()

        p = con.execute(f"""
            SELECT subject_id, dod
            FROM {pat}
        """).fetchdf()
    finally:
        con.close()

    x = c.merge(a, on="hadm_id", how="left").merge(p, on="subject_id", how="left")
    x["deathtime"] = safe_dt(x["deathtime"])
    x["dod"] = safe_dt(x["dod"])

    death_dt = x["deathtime"].copy()

    # DOD is date-level. Place it at end-of-day to avoid impossible "death
    # before current trial baseline" solely from date coarsening.
    dod_fallback = (
        x["dod"].dt.normalize()
        + pd.Timedelta(hours=23, minutes=59, seconds=59)
    )
    dod_fallback = dod_fallback.where(dod_fallback >= x["respiratory_t0"])
    death_dt = death_dt.where(death_dt.notna(), dod_fallback)

    x["death_hours_full"] = (
        (death_dt - x["respiratory_t0"]).dt.total_seconds() / 3600.0
    )
    x.loc[x["death_hours_full"] < 0, "death_hours_full"] = np.nan

    return x[["stay_id", "death_hours_full"]]


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PANEL_PATH}\nRun Stage 7 first."
        )
    panel = pd.read_parquet(PANEL_PATH)
    death = attach_full_death_hours()

    panel = panel.drop(columns=["death_hours"], errors="ignore").merge(
        death, on="stay_id", how="left"
    )
    return panel


def build_sequential_trials(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One row = one eligible patient at one sequential trial baseline.
    """
    parts = []

    for h in TRIAL_STARTS:
        x = panel[panel["hour"] == h].copy()
        if x.empty:
            continue

        init = pd.to_numeric(x["any_init_h"], errors="coerce")
        death = pd.to_numeric(x["death_hours_full"], errors="coerce")

        # Eligible at the trial start only if alive and not previously treated.
        eligible = (
            (death.isna() | (death > h))
            & (init.isna() | (init >= h))
        )
        x = x[eligible].copy()
        init = pd.to_numeric(x["any_init_h"], errors="coerce")
        death = pd.to_numeric(x["death_hours_full"], errors="coerce")

        window_end = h + WINDOW_H

        # Treatment initiation must occur before death and inside [h, h+2).
        x["actual_initiate_next2h"] = (
            init.notna()
            & (init >= h)
            & (init < window_end)
            & (death.isna() | (init <= death))
        ).astype(int)

        x["trial_hour"] = h
        x["trial_hour2"] = h * h
        x["trial_window_end"] = window_end

        # 28-day event from THIS trial baseline.
        x["death_rel_h"] = death - h
        x["death_28d_from_trial"] = (
            death.notna()
            & (death > h)
            & (death <= h + FOLLOWUP_H)
        ).astype(int)

        keep = [
            "stay_id",
            "trial_hour",
            "trial_hour2",
            "trial_window_end",
            "actual_initiate_next2h",
            "any_init_h",
            "death_hours_full",
            "death_rel_h",
            "death_28d_from_trial",
        ]
        for c in BASELINE_NUM + BASELINE_BIN + BASELINE_CAT + CURRENT_NUM + CURRENT_BIN:
            if c in x.columns and c not in keep:
                keep.append(c)

        parts.append(x[keep].copy())

    if not parts:
        raise RuntimeError("No eligible sequential trial rows were created.")

    trials = pd.concat(parts, ignore_index=True)
    trials["trial_uid"] = np.arange(1, len(trials) + 1, dtype=np.int64)
    return trials


def make_preprocessor(df: pd.DataFrame, numerator=False):
    if numerator:
        return ColumnTransformer(
            [
                (
                    "time",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    ["trial_hour", "trial_hour2"],
                )
            ],
            remainder="drop",
        )

    num = [
        c
        for c in BASELINE_NUM + BASELINE_BIN + CURRENT_NUM + CURRENT_BIN
        if c in df.columns
    ]
    cat = [c for c in BASELINE_CAT if c in df.columns]

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                num + ["trial_hour", "trial_hour2"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )



def fit_propensity_v3(trials: pd.DataFrame):
    """
    Fit the same denominator PS model as final causal-analysis stage V2, but construct the primary
    ATE stabilized weight with a MARGINAL numerator:

        treated:   P(A=1) / PS
        untreated: P(A=0) / (1-PS)

    This is the standard marginal stabilized IPTW for a binary treatment.
    It allows the weighted sample to balance trial hour as well as clinical
    severity. V2 instead used P(A|trial hour) in the numerator; because the
    final estimator was a pooled marginal weighted KM without an explicit
    trial-hour outcome adjustment, that could intentionally preserve treatment-
    time imbalance.

    V3 also keeps:
      - fixed cap 10 sensitivity (not percentile truncation)
      - old time-stabilized raw weights for comparison only
      - old global 1st/99th percentile truncation for forensic comparison only
    """
    y = trials["actual_initiate_next2h"].astype(int)
    if y.sum() < 30:
        raise RuntimeError(
            f"Only {int(y.sum())} initiation trials; too few for final causal-analysis stage."
        )

    den = Pipeline(
        [
            ("prep", make_preprocessor(trials, numerator=False)),
            (
                "logit",
                LogisticRegression(
                    C=0.35,
                    solver="liblinear",
                    max_iter=1500,
                    random_state=SEED,
                ),
            ),
        ]
    )

    # Old V2 numerator, retained only to understand the previous result.
    old_num = Pipeline(
        [
            ("prep", make_preprocessor(trials, numerator=True)),
            (
                "logit",
                LogisticRegression(
                    C=10.0,
                    solver="liblinear",
                    max_iter=1500,
                    random_state=SEED,
                ),
            ),
        ]
    )

    den.fit(trials, y)
    old_num.fit(trials, y)

    x = trials.copy()
    x["ps_den"] = den.predict_proba(trials)[:, 1].clip(1e-5, 1 - 1e-5)
    x["ps_old_num_time"] = old_num.predict_proba(trials)[:, 1].clip(1e-5, 1 - 1e-5)

    p_treat = float(y.mean())
    p_control = 1.0 - p_treat

    # PRIMARY: marginal stabilized ATE weights.
    x["w_primary"] = np.where(
        y == 1,
        p_treat / x["ps_den"],
        p_control / (1.0 - x["ps_den"]),
    )

    # Sensitivity: only cap truly large weights at an absolute 10.
    # If all primary weights are <10 this is identical to the primary.
    x["w_cap10"] = x["w_primary"].clip(upper=FIXED_WEIGHT_CAP)

    # Old V2 untruncated time-stabilized weights.
    x["w_old_time_raw"] = np.where(
        y == 1,
        x["ps_old_num_time"] / x["ps_den"],
        (1.0 - x["ps_old_num_time"]) / (1.0 - x["ps_den"]),
    )

    # Old V2 global percentile truncation, retained for comparison only.
    old_raw = x["w_old_time_raw"].to_numpy(float)
    finite = old_raw[np.isfinite(old_raw)]
    lo, hi = np.quantile(finite, [OLD_TRUNC_LOW, OLD_TRUNC_HIGH])
    x["w_old_global_p99"] = x["w_old_time_raw"].clip(lo, hi)

    return x, den, old_num, p_treat, float(lo), float(hi)


def make_clones_v3(weighted_trials: pd.DataFrame, weight_col: str):
    rows = []

    for r in weighted_trials.itertuples(index=False):
        h = float(r.trial_hour)
        grace_end = h + WINDOW_H
        horizon = h + FOLLOWUP_H

        init_h = r.any_init_h
        death_h = r.death_hours_full
        actual_treat = int(r.actual_initiate_next2h)
        obs_weight = float(getattr(r, weight_col))

        for arm in (0, 1):
            death_end = float(death_h) if pd.notna(death_h) else np.inf

            if arm == 0:
                artificial = (
                    float(init_h)
                    if actual_treat == 1 and pd.notna(init_h)
                    else np.inf
                )
                adherent_after_grace = int(actual_treat == 0)
            else:
                artificial = np.inf if actual_treat == 1 else grace_end
                adherent_after_grace = int(actual_treat == 1)

            end_abs = min(horizon, death_end, artificial)
            event = int(
                pd.notna(death_h)
                and float(death_h) <= horizon
                and float(death_h) <= artificial
            )
            if event:
                end_abs = float(death_h)

            rows.append(
                {
                    "trial_uid": r.trial_uid,
                    "stay_id": r.stay_id,
                    "trial_hour": int(r.trial_hour),
                    "arm": arm,
                    "end_rel_h": max(float(end_abs - h), 1e-6),
                    "event": event,
                    "adherent_after_grace": adherent_after_grace,
                    "post_grace_weight": obs_weight if adherent_after_grace else np.nan,
                }
            )

    return pd.DataFrame(rows)


def clone_weight_v3(row, t_rel_h):
    if t_rel_h <= WINDOW_H:
        return 1.0
    if int(row.adherent_after_grace) != 1:
        return 1.0
    return float(row.post_grace_weight) if pd.notna(row.post_grace_weight) else 1.0


def weighted_km_v3(clones: pd.DataFrame, arm: int):
    d = clones[clones["arm"] == arm].copy()
    times = np.sort(d.loc[d["event"] == 1, "end_rel_h"].unique())

    surv = 1.0
    for t in times:
        risk = d[d["end_rel_h"] >= t]
        ev = d[(d["event"] == 1) & np.isclose(d["end_rel_h"], t)]
        if len(risk) == 0:
            continue

        yw = sum(clone_weight_v3(r, t) for r in risk.itertuples(index=False))
        dw = sum(clone_weight_v3(r, t) for r in ev.itertuples(index=False))

        if yw > 0:
            frac = min(max(dw / yw, 0.0), 1.0)
            surv *= 1.0 - frac

    return 1.0 - surv


def estimate_effect_v3(weighted_trials: pd.DataFrame, weight_col: str):
    clones = make_clones_v3(weighted_trials, weight_col)
    r0 = weighted_km_v3(clones, arm=0)
    r1 = weighted_km_v3(clones, arm=1)
    return {
        "risk_defer_2h": r0,
        "risk_initiate_next2h": r1,
        "risk_difference": r1 - r0,
        "risk_ratio": r1 / r0 if r0 > 0 else np.nan,
    }


def ess(w):
    x = np.asarray(w, float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    return (x.sum() ** 2) / np.sum(x ** 2)


def weighted_mean_var(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return np.nan, np.nan
    x, w = x[m], w[m]
    mu = np.average(x, weights=w)
    var = np.average((x - mu) ** 2, weights=w)
    return mu, var


def weighted_smd(a, wa, b, wb):
    ma, va = weighted_mean_var(a, wa)
    mb, vb = weighted_mean_var(b, wb)
    if not np.isfinite(ma) or not np.isfinite(mb):
        return np.nan
    den = math.sqrt((va + vb) / 2.0)
    return (ma - mb) / den if den > 0 else 0.0


def expanded_balance_features(df: pd.DataFrame):
    """
    Numeric/binary variables plus one-hot indicators for categorical covariates.
    This makes the balance audit more complete than V2.
    """
    out = {}

    numeric_vars = [
        c for c in CURRENT_NUM + CURRENT_BIN + BASELINE_NUM + BASELINE_BIN
        if c in df.columns
    ] + ["trial_hour"]

    for c in numeric_vars:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    for c in [x for x in BASELINE_CAT if x in df.columns]:
        s = df[c].astype("string").fillna("Missing")
        for level in sorted(s.dropna().unique().tolist()):
            out[f"{c}=={level}"] = (s == level).astype(float)

    return out


def balance_table_v3(weighted_trials: pd.DataFrame, population: str):
    treat = weighted_trials["actual_initiate_next2h"] == 1
    control = ~treat

    schemes = {
        "unweighted": np.ones(len(weighted_trials), dtype=float),
        "primary_marginal_stabilized_raw": weighted_trials["w_primary"].to_numpy(float),
        "sensitivity_fixed_cap10": weighted_trials["w_cap10"].to_numpy(float),
        "old_time_stabilized_raw": weighted_trials["w_old_time_raw"].to_numpy(float),
        "old_global_p99_truncation": weighted_trials["w_old_global_p99"].to_numpy(float),
    }

    features = expanded_balance_features(weighted_trials)
    rows = []

    for feature, series in features.items():
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        xa = values[treat.to_numpy()]
        xc = values[control.to_numpy()]

        for scheme, all_w in schemes.items():
            wa = all_w[treat.to_numpy()]
            wc = all_w[control.to_numpy()]
            smd = weighted_smd(xa, wa, xc, wc)
            rows.append(
                {
                    "population": population,
                    "weight_scheme": scheme,
                    "variable": feature,
                    "smd": smd,
                    "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
                    "balance_target_lt_0_10": int(pd.notna(smd) and abs(smd) < 0.10),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["population", "weight_scheme", "abs_smd"],
        ascending=[True, True, False],
        na_position="last",
    )


def propensity_diagnostics_v3(weighted_trials: pd.DataFrame, population: str):
    rows = []

    for group, sub in [
        ("all_person_trials", weighted_trials),
        ("actual_initiators", weighted_trials[weighted_trials.actual_initiate_next2h == 1]),
        ("actual_defer", weighted_trials[weighted_trials.actual_initiate_next2h == 0]),
    ]:
        p = sub["ps_den"].to_numpy(float)

        for scheme, col in [
            ("primary_marginal_stabilized_raw", "w_primary"),
            ("sensitivity_fixed_cap10", "w_cap10"),
            ("old_time_stabilized_raw", "w_old_time_raw"),
            ("old_global_p99_truncation", "w_old_global_p99"),
        ]:
            w = sub[col].to_numpy(float)
            rows.append(
                {
                    "population": population,
                    "group": group,
                    "weight_scheme": scheme,
                    "n_person_trials": len(sub),
                    "ps_min": np.min(p) if len(p) else np.nan,
                    "ps_p01": np.quantile(p, .01) if len(p) else np.nan,
                    "ps_p50": np.quantile(p, .50) if len(p) else np.nan,
                    "ps_p99": np.quantile(p, .99) if len(p) else np.nan,
                    "ps_max": np.max(p) if len(p) else np.nan,
                    "weight_mean": np.mean(w) if len(w) else np.nan,
                    "weight_p95": np.quantile(w, .95) if len(w) else np.nan,
                    "weight_p99": np.quantile(w, .99) if len(w) else np.nan,
                    "weight_max": np.max(w) if len(w) else np.nan,
                    "ess": ess(w),
                    "ps_lt_0_005_n": int((p < .005).sum()),
                    "ps_lt_0_01_n": int((p < .01).sum()),
                    "ps_gt_0_50_n": int((p > .50).sum()),
                }
            )

    return pd.DataFrame(rows)


def trial_hour_diagnostics_v3(weighted_trials: pd.DataFrame, population: str):
    rows = []
    for h in TRIAL_STARTS:
        sub = weighted_trials[weighted_trials["trial_hour"] == h].copy()
        if sub.empty:
            continue
        t = sub[sub.actual_initiate_next2h == 1]
        c = sub[sub.actual_initiate_next2h == 0]

        rows.append(
            {
                "population": population,
                "trial_hour": h,
                "eligible_person_trials_n": len(sub),
                "initiate_next2h_n": len(t),
                "defer_next2h_n": len(c),
                "initiation_rate": len(t) / len(sub),
                "treated_ps_min": t["ps_den"].min() if len(t) else np.nan,
                "treated_ps_median": t["ps_den"].median() if len(t) else np.nan,
                "treated_primary_weight_max": t["w_primary"].max() if len(t) else np.nan,
                "control_primary_weight_max": c["w_primary"].max() if len(c) else np.nan,
                "treated_ess_primary": ess(t["w_primary"]) if len(t) else np.nan,
                "control_ess_primary": ess(c["w_primary"]) if len(c) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def effect_scheme_table(weighted_trials: pd.DataFrame, population: str):
    rows = []
    schemes = [
        (
            "primary_marginal_stabilized_raw",
            "w_primary",
            "PRIMARY: marginal stabilized ATE weight; no percentile truncation",
        ),
        (
            "sensitivity_fixed_cap10",
            "w_cap10",
            "Sensitivity: primary weight capped only if >10",
        ),
        (
            "old_time_stabilized_raw",
            "w_old_time_raw",
            "Forensic comparison with V2 numerator P(A|trial hour), untruncated",
        ),
        (
            "old_global_p99_truncation",
            "w_old_global_p99",
            "Forensic comparison: V2-style global 1/99 percentile truncation",
        ),
    ]

    for scheme, col, note in schemes:
        e = estimate_effect_v3(weighted_trials, col)
        rows.append(
            {
                "population": population,
                "weight_scheme": scheme,
                "risk_defer_at_least_2h": e["risk_defer_2h"],
                "risk_initiate_next2h": e["risk_initiate_next2h"],
                "risk_difference": e["risk_difference"],
                "risk_ratio": e["risk_ratio"],
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def model_diagnostics_v3(
    weighted_trials: pd.DataFrame,
    balance: pd.DataFrame,
    population: str,
    p_treat: float,
    old_lo: float,
    old_hi: float,
):
    primary_bal = balance[
        balance["weight_scheme"] == "primary_marginal_stabilized_raw"
    ].copy()
    old_bal = balance[
        balance["weight_scheme"] == "old_global_p99_truncation"
    ].copy()

    w = weighted_trials["w_primary"].to_numpy(float)
    return pd.DataFrame(
        [
            {
                "population": population,
                "unique_patients_n": weighted_trials.stay_id.nunique(),
                "person_trials_n": len(weighted_trials),
                "initiation_person_trials_n": int(weighted_trials.actual_initiate_next2h.sum()),
                "marginal_treatment_probability": p_treat,
                "primary_weight_mean": np.mean(w),
                "primary_weight_p99": np.quantile(w, .99),
                "primary_weight_max": np.max(w),
                "primary_weight_ess": ess(w),
                "primary_max_abs_smd": primary_bal["abs_smd"].max(),
                "primary_covariates_abs_smd_gt_0_10": int(
                    (primary_bal["abs_smd"] > .10).sum()
                ),
                "primary_covariates_abs_smd_gt_0_20": int(
                    (primary_bal["abs_smd"] > .20).sum()
                ),
                "old_global_p99_lower": old_lo,
                "old_global_p99_upper": old_hi,
                "old_global_p99_max_abs_smd": old_bal["abs_smd"].max(),
                "decision_rule": (
                    "Proceed to final bootstrap only if primary balance is clinically acceptable; "
                    "inspect all key severity variables, not only the maximum SMD."
                ),
            }
        ]
    )


def run_population_v3(trials: pd.DataFrame, population: str):
    print(f"\n=== final causal-analysis stage V3 weight audit: {population} ===")
    print(
        f"patients={trials.stay_id.nunique()} | "
        f"person-trials={len(trials)} | "
        f"2h initiations={int(trials.actual_initiate_next2h.sum())}"
    )

    weighted, _, _, p_treat, old_lo, old_hi = fit_propensity_v3(trials)

    effects = effect_scheme_table(weighted, population)
    balance = balance_table_v3(weighted, population)
    psdiag = propensity_diagnostics_v3(weighted, population)
    hourdiag = trial_hour_diagnostics_v3(weighted, population)
    modeldiag = model_diagnostics_v3(
        weighted, balance, population, p_treat, old_lo, old_hi
    )

    primary = effects[
        effects["weight_scheme"] == "primary_marginal_stabilized_raw"
    ].iloc[0]
    md = modeldiag.iloc[0]

    print(
        f"PRIMARY point estimate: RD={primary['risk_difference']:.4f}, "
        f"RR={primary['risk_ratio']:.4f}"
    )
    print(
        f"PRIMARY weight max={md['primary_weight_max']:.3f}, "
        f"ESS={md['primary_weight_ess']:.1f}"
    )
    print(
        f"PRIMARY balance: max |SMD|={md['primary_max_abs_smd']:.3f}; "
        f"n(|SMD|>0.10)={int(md['primary_covariates_abs_smd_gt_0_10'])}"
    )

    print("\nTop 12 residual primary imbalances:")
    top = (
        balance[balance.weight_scheme == "primary_marginal_stabilized_raw"]
        .sort_values("abs_smd", ascending=False)
        .head(12)
    )
    print(top[["variable", "smd", "abs_smd"]].to_string(index=False))

    return effects, balance, psdiag, hourdiag, modeldiag


def main():
    print("=" * 100)
    print("final causal-analysis stage V3 — Weight fix diagnostic (NO BOOTSTRAP)")
    print("=" * 100)
    print(
        "Primary weight = marginal stabilized ATE IPTW, no percentile truncation.\n"
        "This run is intentionally fast: diagnose balance/overlap first, then bootstrap."
    )

    try:
        panel = load_panel()
        broad_trials = build_sequential_trials(panel)

        stage8b = import_stage8b()
        flags = stage8b.build_clinical_flags()
        cape_ids = set(flags.loc[flags["cape_cod_like"] == 1, "stay_id"])
        cape_panel = panel[panel["stay_id"].isin(cape_ids)].copy()
        cape_trials = build_sequential_trials(cape_panel)

        b = run_population_v3(broad_trials, "broad_primary")
        c = run_population_v3(cape_trials, "CAPE_COD_like")

        pd.concat([b[0], c[0]], ignore_index=True).to_csv(
            OUT_EFFECT, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[1], c[1]], ignore_index=True).to_csv(
            OUT_BAL, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[2], c[2]], ignore_index=True).to_csv(
            OUT_PS, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[3], c[3]], ignore_index=True).to_csv(
            OUT_HOUR, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[4], c[4]], ignore_index=True).to_csv(
            OUT_MODEL, index=False, encoding="utf-8-sig"
        )

        print("\n" + "=" * 100)
        print("STAGE 10 V3 DIAGNOSTIC COMPLETE")
        print("=" * 100)
        print("\nGenerated five audit CSVs:")
        for p in [OUT_EFFECT, OUT_BAL, OUT_PS, OUT_HOUR, OUT_MODEL]:
            print(" ", p)

        print(
            "\nIMPORTANT: Do NOT run another long bootstrap yet. "
            "First inspect V3 balance and overlap."
        )

    except Exception:
        print("\n[FAILED] Copy the complete traceback to ChatGPT:\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

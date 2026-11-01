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
        "Run in a terminal:\n"
        "    pip install scikit-learn\n"
        f"Original error: {e}"
    )

PROJECT_ROOT = Path(config.PROJECT_ROOT)
DATA_ROOT = Path(config.DATA_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

PANEL_PATH = OUTPUT_DIR / "stage7_hourly_panel.parquet"
COHORT_PATH = OUTPUT_DIR / "stage5_5_analysis_cohort.parquet"
STAGE8B_SCRIPT = PROJECT_ROOT / "14_run_high_specificity_ccw.py"

OUT_EFFECT = OUTPUT_DIR / "16V4_FINAL_OVERLAP_EFFECT.csv"
OUT_BAL = OUTPUT_DIR / "16V4_FINAL_OVERLAP_BALANCE.csv"
OUT_WEIGHT = OUTPUT_DIR / "16V4_FINAL_OVERLAP_WEIGHT_DIAGNOSTICS.csv"
OUT_TARGET = OUTPUT_DIR / "16V4_FINAL_OVERLAP_TARGET_PROFILE.csv"
OUT_HOUR = OUTPUT_DIR / "16V4_FINAL_OVERLAP_TRIAL_HOUR_DIAGNOSTICS.csv"
OUT_MODEL = OUTPUT_DIR / "16V4_FINAL_OVERLAP_MODEL_DIAGNOSTICS.csv"

WINDOW_H = 2
TRIAL_STARTS = list(range(0, 24, 2))
FOLLOWUP_H = 28 * 24

SEED = 20261027

# V4 is diagnostic-only: no bootstrap.
# Overlap weights are bounded by construction and are NOT truncated.

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




def _fit_unpenalized_logistic(X, y):
    """
    Fit an unpenalized logistic PS model.

    Exact finite-sample mean balance from overlap weights relies on the
    logistic score equations, so V4 does NOT use L1/L2 regularization.

    sklearn compatibility:
      - current versions: penalty=None
      - some older versions: penalty="none"
    """
    attempts = [
        dict(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-9),
        dict(penalty="none", solver="lbfgs", max_iter=5000, tol=1e-9),
    ]
    last = None
    for kwargs in attempts:
        try:
            model = LogisticRegression(
                fit_intercept=True,
                random_state=SEED,
                **kwargs,
            )
            model.fit(X, y)
            return model
        except Exception as exc:
            last = exc
    raise RuntimeError(
        "Could not fit an unpenalized logistic regression required for "
        f"overlap weighting. Last error: {last}"
    )


def fit_overlap_propensity(trials: pd.DataFrame):
    """
    Estimate PS by unpenalized logistic regression and construct overlap weights:

        treated: 1 - PS
        control: PS

    Estimand: ATO, the average treatment effect in the overlap population.
    """
    y = trials["actual_initiate_next2h"].astype(int).to_numpy()
    if y.sum() < 30:
        raise RuntimeError(
            f"Only {int(y.sum())} initiation person-trials; too few for V4."
        )

    prep = make_preprocessor(trials, numerator=False)
    X = prep.fit_transform(trials)

    model = _fit_unpenalized_logistic(X, y)
    ps = model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-8, 1 - 1e-8)

    x = trials.copy()
    x["ps_ow"] = ps
    x["overlap_weight"] = np.where(y == 1, 1.0 - ps, ps)

    return x, prep, model, X


def make_clones_ow(weighted_trials: pd.DataFrame):
    rows = []

    for r in weighted_trials.itertuples(index=False):
        h = float(r.trial_hour)
        grace_end = h + WINDOW_H
        horizon = h + FOLLOWUP_H

        init_h = r.any_init_h
        death_h = r.death_hours_full
        actual_treat = int(r.actual_initiate_next2h)
        ow = float(r.overlap_weight)

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
                    "post_grace_weight": ow if adherent_after_grace else np.nan,
                }
            )

    return pd.DataFrame(rows)


def clone_weight_ow(row, t_rel_h):
    # Before the 2-hour grace ends, both clones are still protocol-compatible.
    if t_rel_h <= WINDOW_H:
        return 1.0
    if int(row.adherent_after_grace) != 1:
        return 1.0
    return float(row.post_grace_weight) if pd.notna(row.post_grace_weight) else 1.0


def weighted_km_ow(clones: pd.DataFrame, arm: int):
    d = clones[clones["arm"] == arm].copy()
    times = np.sort(d.loc[d["event"] == 1, "end_rel_h"].unique())

    surv = 1.0
    for t in times:
        risk = d[d["end_rel_h"] >= t]
        ev = d[(d["event"] == 1) & np.isclose(d["end_rel_h"], t)]
        if len(risk) == 0:
            continue

        yw = sum(clone_weight_ow(r, t) for r in risk.itertuples(index=False))
        dw = sum(clone_weight_ow(r, t) for r in ev.itertuples(index=False))

        if yw > 0:
            frac = min(max(dw / yw, 0.0), 1.0)
            surv *= 1.0 - frac

    return 1.0 - surv


def estimate_overlap_effect(weighted_trials: pd.DataFrame):
    clones = make_clones_ow(weighted_trials)
    r0 = weighted_km_ow(clones, arm=0)
    r1 = weighted_km_ow(clones, arm=1)

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


def interpretable_balance_features(df: pd.DataFrame):
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


def interpretable_balance_table(weighted_trials: pd.DataFrame, population: str):
    treat = weighted_trials["actual_initiate_next2h"].astype(int).to_numpy() == 1
    control = ~treat
    w = weighted_trials["overlap_weight"].to_numpy(float)

    rows = []
    for feature, series in interpretable_balance_features(weighted_trials).items():
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)

        smd_unw = weighted_smd(
            values[treat], np.ones(treat.sum()),
            values[control], np.ones(control.sum()),
        )
        smd_ow = weighted_smd(
            values[treat], w[treat],
            values[control], w[control],
        )

        rows.append(
            {
                "population": population,
                "balance_scale": "interpretable_original_variable",
                "variable": feature,
                "smd_unweighted": smd_unw,
                "abs_smd_unweighted": abs(smd_unw) if pd.notna(smd_unw) else np.nan,
                "smd_overlap_weighted": smd_ow,
                "abs_smd_overlap_weighted": abs(smd_ow) if pd.notna(smd_ow) else np.nan,
                "target_lt_0_10": int(pd.notna(smd_ow) and abs(smd_ow) < .10),
            }
        )

    return pd.DataFrame(rows)


def transformed_feature_balance(X, y, w, population: str):
    """
    Audit the exact variables actually entering the PS regression.

    For a correctly converged unpenalized logistic PS, OW should make weighted
    treated/control means of these design-matrix columns nearly identical.
    """
    y = np.asarray(y, int)
    treat = y == 1
    control = y == 0

    # Sparse/dense compatible weighted means.
    def col_weighted_mean(Xsub, ww):
        denom = ww.sum()
        if denom <= 0:
            return np.full(Xsub.shape[1], np.nan)
        val = np.asarray(Xsub.T @ ww).reshape(-1) / denom
        return val

    mt = col_weighted_mean(X[treat], w[treat])
    mc = col_weighted_mean(X[control], w[control])
    diff = mt - mc

    # Exact balance is about means; record absolute mean differences on the
    # transformed/scaled design matrix.
    return pd.DataFrame(
        {
            "population": population,
            "balance_scale": "ps_design_matrix",
            "variable": [f"X_col_{i}" for i in range(X.shape[1])],
            "weighted_mean_treated": mt,
            "weighted_mean_control": mc,
            "absolute_mean_difference": np.abs(diff),
        }
    )


def weight_diagnostics(weighted_trials: pd.DataFrame, population: str):
    y = weighted_trials["actual_initiate_next2h"].astype(int).to_numpy()
    ps = weighted_trials["ps_ow"].to_numpy(float)
    w = weighted_trials["overlap_weight"].to_numpy(float)

    rows = []
    for label, mask in [
        ("all_person_trials", np.ones(len(y), dtype=bool)),
        ("actual_initiators", y == 1),
        ("actual_defer", y == 0),
    ]:
        pp = ps[mask]
        ww = w[mask]
        rows.append(
            {
                "population": population,
                "group": label,
                "n_person_trials": int(mask.sum()),
                "ps_min": np.min(pp) if len(pp) else np.nan,
                "ps_p01": np.quantile(pp, .01) if len(pp) else np.nan,
                "ps_p50": np.quantile(pp, .50) if len(pp) else np.nan,
                "ps_p99": np.quantile(pp, .99) if len(pp) else np.nan,
                "ps_max": np.max(pp) if len(pp) else np.nan,
                "ow_mean": np.mean(ww) if len(ww) else np.nan,
                "ow_p50": np.quantile(ww, .50) if len(ww) else np.nan,
                "ow_p95": np.quantile(ww, .95) if len(ww) else np.nan,
                "ow_p99": np.quantile(ww, .99) if len(ww) else np.nan,
                "ow_max": np.max(ww) if len(ww) else np.nan,
                "ow_sum": np.sum(ww) if len(ww) else np.nan,
                "ow_ess": ess(ww),
                "ps_lt_0_005_n": int((pp < .005).sum()),
                "ps_lt_0_01_n": int((pp < .01).sum()),
                "ps_gt_0_10_n": int((pp > .10).sum()),
            }
        )
    return pd.DataFrame(rows)


def trial_hour_diagnostics(weighted_trials: pd.DataFrame, population: str):
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
                "treated_ps_min": t["ps_ow"].min() if len(t) else np.nan,
                "treated_ps_median": t["ps_ow"].median() if len(t) else np.nan,
                "treated_ow_sum": t["overlap_weight"].sum() if len(t) else 0.0,
                "control_ow_sum": c["overlap_weight"].sum() if len(c) else 0.0,
                "treated_ow_ess": ess(t["overlap_weight"]) if len(t) else np.nan,
                "control_ow_ess": ess(c["overlap_weight"]) if len(c) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def overlap_target_profile(weighted_trials: pd.DataFrame, population: str):
    """
    Describe the ATO target population using overlap-weighted means.
    Because OW balances groups, the two weighted means should be very similar.
    """
    y = weighted_trials["actual_initiate_next2h"].astype(int).to_numpy()
    w = weighted_trials["overlap_weight"].to_numpy(float)
    features = interpretable_balance_features(weighted_trials)

    rows = []
    for feature, series in features.items():
        x = pd.to_numeric(series, errors="coerce").to_numpy(float)

        for label, mask in [
            ("initiator_weighted", y == 1),
            ("defer_weighted", y == 0),
        ]:
            m = mask & np.isfinite(x) & np.isfinite(w) & (w > 0)
            value = np.average(x[m], weights=w[m]) if m.sum() else np.nan
            rows.append(
                {
                    "population": population,
                    "variable": feature,
                    "group": label,
                    "overlap_weighted_mean": value,
                }
            )

    return pd.DataFrame(rows)


def run_population_v4(trials: pd.DataFrame, population: str):
    print(f"\n=== final causal-analysis stage V4 overlap weighting: {population} ===")
    print(
        f"patients={trials.stay_id.nunique()} | "
        f"person-trials={len(trials)} | "
        f"2h initiations={int(trials.actual_initiate_next2h.sum())}"
    )

    weighted, prep, model, X = fit_overlap_propensity(trials)
    y = weighted["actual_initiate_next2h"].astype(int).to_numpy()
    w = weighted["overlap_weight"].to_numpy(float)

    effect = estimate_overlap_effect(weighted)
    ibal = interpretable_balance_table(weighted, population)
    xbal = transformed_feature_balance(X, y, w, population)
    wdiag = weight_diagnostics(weighted, population)
    hour = trial_hour_diagnostics(weighted, population)
    profile = overlap_target_profile(weighted, population)

    max_interp = ibal["abs_smd_overlap_weighted"].max()
    n_gt_01 = int((ibal["abs_smd_overlap_weighted"] > .10).sum())
    max_design_diff = xbal["absolute_mean_difference"].max()

    treated_sum = float(w[y == 1].sum())
    control_sum = float(w[y == 0].sum())

    effect_df = pd.DataFrame(
        [
            {
                "population": population,
                "estimand": "ATO_overlap_population",
                "strategy_1": "initiate systemic corticosteroid during next 2h",
                "strategy_0": "defer systemic corticosteroid for at least 2h",
                "risk_defer_at_least_2h": effect["risk_defer_2h"],
                "risk_initiate_next2h": effect["risk_initiate_next2h"],
                "risk_difference": effect["risk_difference"],
                "risk_ratio": effect["risk_ratio"],
                "note": "diagnostic point estimate only; no CI until balance/overlap accepted",
            }
        ]
    )

    modeldiag = pd.DataFrame(
        [
            {
                "population": population,
                "unique_patients_n": trials.stay_id.nunique(),
                "person_trials_n": len(trials),
                "initiation_person_trials_n": int(y.sum()),
                "ps_model": "unpenalized_logistic_regression",
                "ps_design_columns_n": int(X.shape[1]),
                "overlap_weight_min": float(np.min(w)),
                "overlap_weight_max": float(np.max(w)),
                "treated_weight_sum": treated_sum,
                "control_weight_sum": control_sum,
                "absolute_difference_group_weight_sums": abs(treated_sum - control_sum),
                "treated_ess": ess(w[y == 1]),
                "control_ess": ess(w[y == 0]),
                "interpretable_max_abs_smd_after_ow": max_interp,
                "interpretable_covariates_abs_smd_gt_0_10": n_gt_01,
                "ps_design_max_absolute_mean_difference": max_design_diff,
                "decision_rule": (
                    "Proceed to final cluster bootstrap only if PS-design balance is near exact "
                    "and clinically interpretable key severity variables are acceptably balanced."
                ),
            }
        ]
    )

    print(
        f"ATO point estimate: RD={effect['risk_difference']:.4f}, "
        f"RR={effect['risk_ratio']:.4f}"
    )
    print(
        f"OW sums treated/control={treated_sum:.4f}/{control_sum:.4f}; "
        f"ESS treated/control={ess(w[y==1]):.1f}/{ess(w[y==0]):.1f}"
    )
    print(
        f"Interpretable balance max |SMD|={max_interp:.4f}; "
        f"n(|SMD|>0.10)={n_gt_01}"
    )
    print(
        f"PS design matrix max absolute weighted-mean difference="
        f"{max_design_diff:.3e}"
    )

    print("\nTop 12 interpretable residual imbalances:")
    print(
        ibal.sort_values("abs_smd_overlap_weighted", ascending=False)
        .head(12)[
            ["variable", "smd_unweighted", "smd_overlap_weighted", "abs_smd_overlap_weighted"]
        ]
        .to_string(index=False)
    )

    return effect_df, ibal, xbal, wdiag, profile, hour, modeldiag


def main():
    print("=" * 100)
    print("final causal-analysis stage V4 — Overlap-weighted sequential target trial (ATO diagnostic)")
    print("=" * 100)
    print(
        "No bootstrap in V4. First verify exact PS-design balance and clinical balance.\n"
        "Overlap weights: treated=1-PS; control=PS; no trimming/truncation."
    )

    try:
        panel = load_panel()
        broad_trials = build_sequential_trials(panel)

        stage8b = import_stage8b()
        flags = stage8b.build_clinical_flags()
        cape_ids = set(flags.loc[flags["cape_cod_like"] == 1, "stay_id"])
        cape_panel = panel[panel["stay_id"].isin(cape_ids)].copy()
        cape_trials = build_sequential_trials(cape_panel)

        b = run_population_v4(broad_trials, "broad_primary")
        c = run_population_v4(cape_trials, "CAPE_COD_like")

        pd.concat([b[0], c[0]], ignore_index=True).to_csv(
            OUT_EFFECT, index=False, encoding="utf-8-sig"
        )

        # Interpretable balance only in the main balance CSV.
        pd.concat([b[1], c[1]], ignore_index=True).to_csv(
            OUT_BAL, index=False, encoding="utf-8-sig"
        )

        pd.concat([b[3], c[3]], ignore_index=True).to_csv(
            OUT_WEIGHT, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[4], c[4]], ignore_index=True).to_csv(
            OUT_TARGET, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[5], c[5]], ignore_index=True).to_csv(
            OUT_HOUR, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[6], c[6]], ignore_index=True).to_csv(
            OUT_MODEL, index=False, encoding="utf-8-sig"
        )

        # Also save exact PS design balance for audit.
        out_xbal = OUTPUT_DIR / "16V4_FINAL_OVERLAP_PS_DESIGN_BALANCE.csv"
        pd.concat([b[2], c[2]], ignore_index=True).to_csv(
            out_xbal, index=False, encoding="utf-8-sig"
        )

        print("\n" + "=" * 100)
        print("STAGE 10 V4 DIAGNOSTIC COMPLETE")
        print("=" * 100)
        print("\nGenerated seven audit CSVs:")
        for p in [
            OUT_EFFECT,
            OUT_BAL,
            OUT_WEIGHT,
            OUT_TARGET,
            OUT_HOUR,
            OUT_MODEL,
            out_xbal,
        ]:
            print(" ", p)

        print(
            "\nDo NOT start another long bootstrap yet. "
            "First verify V4 balance and effective sample size."
        )

    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

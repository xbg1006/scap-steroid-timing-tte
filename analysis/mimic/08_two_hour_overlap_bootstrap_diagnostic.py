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

OUTPUTS TO UPLOAD
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

OUT_EFFECT = OUTPUT_DIR / "16V5_FINAL_FINAL_ATO_EFFECT.csv"
OUT_COMPARE = OUTPUT_DIR / "16V5B_FINAL_TARGET_CONTINUITY_COMPARISON.csv"
OUT_DIAG = OUTPUT_DIR / "16V5C_FINAL_FINAL_DIAGNOSTICS.csv"
OUT_BOOT = OUTPUT_DIR / "16V5D_FINAL_BOOTSTRAP_SUMMARY.csv"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_H = 2
TRIAL_STARTS = list(range(0, 24, 2))
FOLLOWUP_H = 28 * 24

SEED = 20261028

BROAD_BOOTSTRAP_TARGET = 200
CAPE_BOOTSTRAP_TARGET = 100
CHECKPOINT_EVERY = 10

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





def _fit_unpenalized_logistic(X, y, sample_weight=None):
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
            model.fit(X, y, sample_weight=sample_weight)
            return model
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unpenalized logistic regression failed: {last}")


def build_clone_geometry(trials: pd.DataFrame):
    h0 = trials["trial_hour"].to_numpy(float)
    init = pd.to_numeric(trials["any_init_h"], errors="coerce").to_numpy(float)
    death = pd.to_numeric(trials["death_hours_full"], errors="coerce").to_numpy(float)
    a = trials["actual_initiate_next2h"].astype(int).to_numpy()

    death_rel = death - h0
    death_end = np.where(np.isfinite(death_rel), death_rel, np.inf)
    horizon = np.full(len(trials), FOLLOWUP_H, dtype=float)
    init_rel = init - h0

    art0 = np.where(a == 1, init_rel, np.inf)
    end0 = np.minimum.reduce([horizon, death_end, art0])
    event0 = (
        np.isfinite(death_rel)
        & (death_rel > 0)
        & (death_rel <= FOLLOWUP_H)
        & (death_rel <= art0)
    ).astype(int)
    end0 = np.where(event0 == 1, death_rel, end0)

    art1 = np.where(a == 1, np.inf, float(WINDOW_H))
    end1 = np.minimum.reduce([horizon, death_end, art1])
    event1 = (
        np.isfinite(death_rel)
        & (death_rel > 0)
        & (death_rel <= FOLLOWUP_H)
        & (death_rel <= art1)
    ).astype(int)
    end1 = np.where(event1 == 1, death_rel, end1)

    return {
        "end0": np.maximum(end0, 1e-9),
        "event0": event0,
        "end1": np.maximum(end1, 1e-9),
        "event1": event1,
    }


def prepare_population(trials: pd.DataFrame):
    trials = trials.reset_index(drop=True)
    y = trials["actual_initiate_next2h"].astype(int).to_numpy()

    prep = make_preprocessor(trials, numerator=False)
    X = prep.fit_transform(trials)

    unique_ids, inverse_patient = np.unique(
        trials["stay_id"].to_numpy(), return_inverse=True
    )

    return {
        "trials": trials,
        "y": y,
        "X": X,
        "unique_ids": unique_ids,
        "inverse_patient": inverse_patient,
        "geom": build_clone_geometry(trials),
    }


def _phase_survival(end, event, weight, pre_grace: bool):
    if pre_grace:
        event_mask = (event == 1) & (end <= WINDOW_H)
        risk_mask = np.ones(len(end), dtype=bool)
    else:
        event_mask = (event == 1) & (end > WINDOW_H)
        risk_mask = end > WINDOW_H

    if not event_mask.any():
        return 1.0

    d = pd.DataFrame({
        "end": end[risk_mask],
        "risk_w": weight[risk_mask],
        "event_w": weight[risk_mask] * event[risk_mask],
    })
    g = d.groupby("end", sort=True, as_index=False).sum(numeric_only=True)
    g = g.sort_values("end", ascending=False).reset_index(drop=True)
    g["risk_cum"] = g["risk_w"].cumsum()

    if pre_grace:
        eg = g[(g["event_w"] > 0) & (g["end"] <= WINDOW_H)]
    else:
        eg = g[(g["event_w"] > 0) & (g["end"] > WINDOW_H)]

    if eg.empty:
        return 1.0

    frac = (eg["event_w"] / eg["risk_cum"]).clip(0, 1).to_numpy(float)
    return float(np.prod(1.0 - frac))


def estimate_ato(prep_obj, ps, row_mult=None, mixed_v4=False):
    y = prep_obj["y"]
    geom = prep_obj["geom"]
    ps = np.clip(np.asarray(ps, float), 1e-8, 1 - 1e-8)

    if row_mult is None:
        freq = np.ones(len(y), dtype=float)
    else:
        freq = np.asarray(row_mult, float)

    h = ps * (1 - ps)
    pre_w = freq if mixed_v4 else freq * h
    w0 = freq * ps
    w1 = freq * (1 - ps)

    s0 = _phase_survival(geom["end0"], geom["event0"], pre_w, True)
    s0 *= _phase_survival(geom["end0"], geom["event0"], w0, False)

    s1 = _phase_survival(geom["end1"], geom["event1"], pre_w, True)
    s1 *= _phase_survival(geom["end1"], geom["event1"], w1, False)

    r0 = 1 - s0
    r1 = 1 - s1
    return {
        "risk_defer_2h": r0,
        "risk_initiate_next2h": r1,
        "risk_difference": r1 - r0,
        "risk_ratio": r1 / r0 if r0 > 0 else np.nan,
    }


def fit_point(prep_obj):
    model = _fit_unpenalized_logistic(prep_obj["X"], prep_obj["y"])
    ps = model.predict_proba(prep_obj["X"])[:, 1]
    return model, np.clip(ps, 1e-8, 1 - 1e-8)


def checkpoint_path(population: str):
    safe = population.replace(" ", "_").replace("/", "_")
    return CHECKPOINT_DIR / f"stage10_v5_{safe}_ato_cluster_bootstrap.csv"


def load_checkpoint(path: Path):
    cols = [
        "rep", "risk_defer_2h", "risk_initiate_next2h",
        "risk_difference", "risk_ratio"
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        x = pd.read_csv(path)
        if not set(cols).issubset(x.columns):
            return pd.DataFrame(columns=cols)
        return x[cols].copy()
    except Exception:
        return pd.DataFrame(columns=cols)


def save_checkpoint(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def cluster_bootstrap_fast(prep_obj, population, target_reps, seed):
    ckpt = checkpoint_path(population)
    old = load_checkpoint(ckpt)
    completed = len(old)

    if completed >= target_reps:
        print(f"      [RESUME] checkpoint already has {completed}; target={target_reps}")
        return old.iloc[:target_reps].copy()

    if completed > 0:
        print(f"      [RESUME] loaded {completed}/{target_reps} from {ckpt.name}")

    rng = np.random.default_rng(seed)
    n_patients = len(prep_obj["unique_ids"])
    inverse = prep_obj["inverse_patient"]

    for _ in range(completed):
        rng.integers(0, n_patients, size=n_patients)

    records = old.to_dict("records")
    success = completed
    attempts = completed

    while success < target_reps:
        attempts += 1
        draw = rng.integers(0, n_patients, size=n_patients)
        patient_counts = np.bincount(draw, minlength=n_patients).astype(float)
        row_mult = patient_counts[inverse]
        mask = row_mult > 0

        yb = prep_obj["y"][mask]
        if yb.sum() < 10:
            continue

        try:
            model = _fit_unpenalized_logistic(
                prep_obj["X"][mask],
                yb,
                sample_weight=row_mult[mask],
            )

            ps = np.full(len(prep_obj["y"]), 0.5, dtype=float)
            ps[mask] = model.predict_proba(prep_obj["X"][mask])[:, 1]
            ps = np.clip(ps, 1e-8, 1 - 1e-8)

            e = estimate_ato(prep_obj, ps, row_mult=row_mult, mixed_v4=False)

            success += 1
            records.append({
                "rep": success,
                "risk_defer_2h": e["risk_defer_2h"],
                "risk_initiate_next2h": e["risk_initiate_next2h"],
                "risk_difference": e["risk_difference"],
                "risk_ratio": e["risk_ratio"],
            })

        except Exception as exc:
            print(
                f"      [WARN] bootstrap attempt {attempts} failed "
                f"({type(exc).__name__}); continuing"
            )
            continue

        if success % CHECKPOINT_EVERY == 0 or success == target_reps:
            cur = pd.DataFrame(records)
            save_checkpoint(ckpt, cur)
            print(f"      checkpoint saved: {success}/{target_reps} -> {ckpt.name}")

    return pd.DataFrame(records).iloc[:target_reps].copy()


def ci95(series):
    x = pd.to_numeric(pd.Series(series), errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return (np.nan, np.nan)
    return tuple(np.quantile(x, [0.025, 0.975]))


def run_population(trials, population, target_reps, seed):
    prep_obj = prepare_population(trials)
    _, ps = fit_point(prep_obj)

    corrected = estimate_ato(prep_obj, ps, mixed_v4=False)
    v4_mixed = estimate_ato(prep_obj, ps, mixed_v4=True)

    print(f"\n=== {population} ===")
    print(
        f"patients={trials.stay_id.nunique()} | "
        f"person-trials={len(trials)} | "
        f"initiations={int(prep_obj['y'].sum())}"
    )
    print(
        f"V5 corrected point: RD={corrected['risk_difference']:.4f}, "
        f"RR={corrected['risk_ratio']:.4f}"
    )
    print(
        f"V4 mixed-target comparison: RD={v4_mixed['risk_difference']:.4f}, "
        f"RR={v4_mixed['risk_ratio']:.4f}"
    )

    boot = cluster_bootstrap_fast(
        prep_obj, population, target_reps, seed
    )

    ci_r0 = ci95(boot["risk_defer_2h"])
    ci_r1 = ci95(boot["risk_initiate_next2h"])
    ci_rd = ci95(boot["risk_difference"])
    ci_rr = ci95(boot["risk_ratio"])

    effect = pd.DataFrame([{
        "population": population,
        "estimand": "ATO_overlap_population",
        "target_population_continuity": (
            "h(X)=e(1-e) from trial time-zero through 2h grace; "
            "arm-specific overlap weights after grace"
        ),
        "unique_patients_n": trials.stay_id.nunique(),
        "person_trials_n": len(trials),
        "initiation_person_trials_n": int(prep_obj["y"].sum()),
        "bootstrap_target_n": target_reps,
        "bootstrap_success_n": len(boot),
        "risk_defer_at_least_2h": corrected["risk_defer_2h"],
        "risk_defer_ci_low": ci_r0[0],
        "risk_defer_ci_high": ci_r0[1],
        "risk_initiate_next2h": corrected["risk_initiate_next2h"],
        "risk_initiate_ci_low": ci_r1[0],
        "risk_initiate_ci_high": ci_r1[1],
        "risk_difference": corrected["risk_difference"],
        "rd_ci_low": ci_rd[0],
        "rd_ci_high": ci_rd[1],
        "risk_ratio": corrected["risk_ratio"],
        "rr_ci_low": ci_rr[0],
        "rr_ci_high": ci_rr[1],
    }])

    compare = pd.DataFrame([
        {
            "population": population,
            "method": "V5_continuous_ATO_target",
            "risk_defer": corrected["risk_defer_2h"],
            "risk_initiate": corrected["risk_initiate_next2h"],
            "risk_difference": corrected["risk_difference"],
            "risk_ratio": corrected["risk_ratio"],
        },
        {
            "population": population,
            "method": "V4_mixed_target_pregrace_unweighted",
            "risk_defer": v4_mixed["risk_defer_2h"],
            "risk_initiate": v4_mixed["risk_initiate_next2h"],
            "risk_difference": v4_mixed["risk_difference"],
            "risk_ratio": v4_mixed["risk_ratio"],
        },
    ])

    y = prep_obj["y"]
    h = ps * (1 - ps)
    ow = np.where(y == 1, 1 - ps, ps)

    diag = pd.DataFrame([{
        "population": population,
        "ps_min": float(ps.min()),
        "ps_p01": float(np.quantile(ps, .01)),
        "ps_median": float(np.quantile(ps, .50)),
        "ps_p99": float(np.quantile(ps, .99)),
        "ps_max": float(ps.max()),
        "baseline_target_h_sum": float(h.sum()),
        "postgrace_treated_ow_sum": float(ow[y == 1].sum()),
        "postgrace_control_ow_sum": float(ow[y == 0].sum()),
        "absolute_difference_postgrace_weight_sums": float(
            abs(ow[y == 1].sum() - ow[y == 0].sum())
        ),
        "checkpoint_file": str(checkpoint_path(population)),
        "note": (
            "CAPE-COD-like remains exploratory because later 2h trials "
            "have sparse initiators."
        ),
    }])

    bsum = pd.DataFrame([{
        "population": population,
        "bootstrap_success_n": len(boot),
        "rd_bootstrap_median": float(boot["risk_difference"].median()),
        "rd_bootstrap_p025": ci_rd[0],
        "rd_bootstrap_p975": ci_rd[1],
        "rr_bootstrap_median": float(boot["risk_ratio"].median()),
        "rr_bootstrap_p025": ci_rr[0],
        "rr_bootstrap_p975": ci_rr[1],
        "checkpoint_file": str(checkpoint_path(population)),
    }])

    return effect, compare, diag, bsum


def main():
    print("=" * 100)
    print("final causal-analysis stage V5 FINAL — Continuous-target ATO + fast cluster bootstrap")
    print("=" * 100)

    try:
        panel = load_panel()
        broad_trials = build_sequential_trials(panel)

        stage8b = import_stage8b()
        flags = stage8b.build_clinical_flags()
        cape_ids = set(flags.loc[flags["cape_cod_like"] == 1, "stay_id"])
        cape_trials = build_sequential_trials(
            panel[panel["stay_id"].isin(cape_ids)].copy()
        )

        b = run_population(
            broad_trials, "broad_primary", BROAD_BOOTSTRAP_TARGET, SEED
        )
        c = run_population(
            cape_trials, "CAPE_COD_like", CAPE_BOOTSTRAP_TARGET, SEED + 1000
        )

        pd.concat([b[0], c[0]], ignore_index=True).to_csv(
            OUT_EFFECT, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[1], c[1]], ignore_index=True).to_csv(
            OUT_COMPARE, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[2], c[2]], ignore_index=True).to_csv(
            OUT_DIAG, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[3], c[3]], ignore_index=True).to_csv(
            OUT_BOOT, index=False, encoding="utf-8-sig"
        )

        print("\n" + "=" * 100)
        print("STAGE 10 V5 COMPLETE")
        print("=" * 100)
        print("\nUpload these four CSVs:")
        for p in [OUT_EFFECT, OUT_COMPARE, OUT_DIAG, OUT_BOOT]:
            print(" ", p)

        print(
            "\nCheckpoint files are target-independent. "
            "If you later increase 200->500 or 100->300, rerun and it continues."
        )

    except Exception:
        print("\n[FAILED] Copy the complete traceback to ChatGPT:\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

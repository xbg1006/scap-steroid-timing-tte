from __future__ import annotations

"""
final causal-analysis stage frozen 6-hour feasibility analysis — 6-hour initiation-window outcome-blind feasibility/positivity audit

NO mortality effect estimation.
NO bootstrap.

Clinical question to audit:
    initiate systemic corticosteroid within the next 6 hours
    versus
    defer systemic corticosteroid for at least 6 hours

Sequential trial starts:
    0, 2, 4, ..., 18 hours after respiratory t0

Why stop at 18h:
    the Stage 7 panel covers the first 24h; a 6h grace window starting at 18h
    still has complete hourly history through hour 23.

Recent-initiation window:
    [3h, 6h) after each sequential-trial baseline.

The audit asks, without using mortality outcomes:
1) Is there substantially more treatment support than the failed 2h strategy?
2) Can the DEFER arm's hourly artificial censoring be balanced across all 6 hours?
3) Can recent initiators in the final 3h represent the alive/untreated risk set at +3h?
4) Are the required IPCW weights and effective sample sizes acceptable?

Only if Broad passes the pre-declared support rules should we build a final
6h pure-CCW outcome analysis.
"""

from pathlib import Path
import importlib.util
import traceback
import warnings
import math

import numpy as np
import pandas as pd
import config

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
except Exception as exc:
    raise RuntimeError(
        "final causal-analysis stage V9 requires scikit-learn.\n"
        "Run: pip install scikit-learn\n"
        f"Original error: {exc}"
    )

PROJECT_ROOT = Path(config.PROJECT_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

PANEL_PATH = OUTPUT_DIR / "stage7_hourly_panel.parquet"

TRIAL_STARTS = list(range(0, 20, 2))  # 0,2,...,18
WINDOW_H = 6.0
RECENT_START_H = 3.0
RECENT_END_H = 6.0

SEED = 20261028
PS_CLIP = 1e-5
RIDGE_GRID = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 10.0]

# Same broad support thresholds as the prior 2h audit where possible.
RECENT_ESS_FLOOR = {
    "broad_primary": 50,
    "CAPE_COD_like": 15,
}

OUT_SUMMARY = OUTPUT_DIR / "16V9_FINAL_6H_FEASIBILITY_SUMMARY.csv"
OUT_RECENT_SELECT = OUTPUT_DIR / "16V9B_FINAL_6H_RECENT_MODEL_SELECTION.csv"
OUT_DEFER_BAL = OUTPUT_DIR / "16V9C_FINAL_6H_DEFER_CENSORING_BALANCE.csv"
OUT_RECENT_BAL = OUTPUT_DIR / "16V9D_FINAL_6H_RECENT_BALANCE.csv"
OUT_RECOMMEND = OUTPUT_DIR / "16V9E_FINAL_6H_RECOMMENDATION.csv"
OUT_HOUR = OUTPUT_DIR / "16V9F_FINAL_6H_TRIAL_HOUR_COUNTS.csv"
OUT_OVERLAP = OUTPUT_DIR / "16V9G_FINAL_6H_BASELINE_OVERLAP.csv"


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


DEATH_TIME_CANDIDATES = [
    "death_hours_full",
    "death_hours_from_t0",
    "death_hours_from_respiratory_t0",
    "death_rel_h",
    "death_hours",
    "hours_to_death",
    "death_time_hours",
    "death_h",
]


def _normalize_death_hours(df: pd.DataFrame, source_label: str):
    """Return stay_id + standardized death_hours_full if a compatible column exists."""
    if "stay_id" not in df.columns:
        return None

    # Exact/preferred names first.
    for c in DEATH_TIME_CANDIDATES:
        if c in df.columns:
            out = df[["stay_id", c]].copy()
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.rename(columns={c: "death_hours_full"})
            out = out.drop_duplicates("stay_id", keep="first")
            print(f"[frozen 6-hour feasibility analysis] death time source: {source_label} -> column '{c}'")
            return out

    # Common merge suffixes from earlier stages.
    suffix_hits = [
        c for c in df.columns
        if c.startswith("death_hours_full_")
        or c.startswith("death_hours_from_t0_")
    ]
    if suffix_hits:
        c = suffix_hits[0]
        out = df[["stay_id", c]].copy()
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.rename(columns={c: "death_hours_full"})
        out = out.drop_duplicates("stay_id", keep="first")
        print(f"[frozen 6-hour feasibility analysis] death time source: {source_label} -> column '{c}'")
        return out

    return None


def _candidate_cache_files():
    """Known/safe cohort caches first, then small recursive fallback inside OUTPUT_DIR."""
    preferred = [
        OUTPUT_DIR / "stage5_5_analysis_cohort.parquet",
        OUTPUT_DIR / "stage5_analysis_cohort.parquet",
        OUTPUT_DIR / "stage4_analysis_cohort.parquet",
        OUTPUT_DIR / "stage4_eligible_cohort.parquet",
        OUTPUT_DIR / "stage5_5_analysis_cohort.csv",
        OUTPUT_DIR / "stage5_analysis_cohort.csv",
        OUTPUT_DIR / "stage4_analysis_cohort.csv",
        OUTPUT_DIR / "stage4_eligible_cohort.csv",
    ]

    seen = set()
    out = []
    for p in preferred:
        if p.exists() and p not in seen:
            out.append(p)
            seen.add(p)

    # Fallback: inspect existing analysis/cohort caches only; never raw MIMIC tables.
    try:
        for p in OUTPUT_DIR.rglob("*"):
            if not p.is_file() or p in seen:
                continue
            name = p.name.lower()
            if p.suffix.lower() not in {".parquet", ".csv"}:
                continue
            if not any(k in name for k in ["cohort", "analysis", "stage5", "stage4"]):
                continue
            out.append(p)
            seen.add(p)
    except Exception:
        pass

    return out


def _load_death_time_from_cache():
    tried = []
    for p in _candidate_cache_files():
        tried.append(str(p))
        try:
            if p.suffix.lower() == ".parquet":
                # Read full cache because parquet column selection cannot safely request
                # an unknown candidate column. These cohort caches are small compared with
                # raw MIMIC tables.
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, low_memory=False)

            death = _normalize_death_hours(df, str(p))
            if death is not None:
                return death
        except Exception as exc:
            print(f"[frozen 6-hour feasibility analysis WARN] could not inspect {p.name}: {type(exc).__name__}: {exc}")

    msg = (
        "Could not recover a death-time-in-hours column from existing analysis caches.\\n"
        "V9 needs death timing only to keep sequential-trial risk sets alive/untreated; "
        "it does NOT use mortality as an outcome in this feasibility audit.\\n"
        "Expected first choice: outputs/stage5_5_analysis_cohort.parquet with one of: "
        + ", ".join(DEATH_TIME_CANDIDATES)
    )
    if tried:
        msg += "\\nCaches inspected:\\n  " + "\\n  ".join(tried[:20])
    raise RuntimeError(msg)


def load_panel():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {PANEL_PATH}\\n"
            "V9 reuses Stage 7 and should not rescan raw MIMIC tables."
        )

    panel = pd.read_parquet(PANEL_PATH)

    base_needed = {"stay_id", "hour", "any_init_h"}
    missing = base_needed - set(panel.columns)
    if missing:
        raise RuntimeError(
            f"Stage 7 panel missing essential non-outcome columns: {sorted(missing)}"
        )

    # Normalize if the panel already contains a compatible death-time column.
    death_from_panel = _normalize_death_hours(panel, str(PANEL_PATH))
    if death_from_panel is not None:
        if "death_hours_full" not in panel.columns:
            # Drop any alternate candidate columns only from the merge input; keep panel intact.
            panel = panel.merge(
                death_from_panel,
                on="stay_id",
                how="left",
                validate="many_to_one",
            )
        else:
            panel["death_hours_full"] = pd.to_numeric(
                panel["death_hours_full"], errors="coerce"
            )
    else:
        print(
            "[frozen 6-hour feasibility analysis] stage7_hourly_panel.parquet has no death_hours_full; "
            "recovering death timing from an existing cohort cache..."
        )
        death = _load_death_time_from_cache()
        panel = panel.merge(
            death,
            on="stay_id",
            how="left",
            validate="many_to_one",
        )

    # Missing death time means no recorded death time in the source cache and is valid;
    # it is interpreted as surviving beyond the observed time window for eligibility checks.
    panel["death_hours_full"] = pd.to_numeric(
        panel["death_hours_full"], errors="coerce"
    )

    n_stays = panel["stay_id"].nunique()
    n_with_death_time = panel.loc[
        panel["death_hours_full"].notna(), "stay_id"
    ].nunique()
    print(
        f"[frozen 6-hour feasibility analysis] panel ready: {n_stays} stays; "
        f"death timing available for {n_with_death_time} stays."
    )

    return panel


def locate_stage8b():
    candidates = [
        PROJECT_ROOT / "14_run_high_specificity_ccw.py",
        PROJECT_ROOT / "scripts" / "14_run_high_specificity_ccw.py",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = list(PROJECT_ROOT.rglob("14_run_high_specificity_ccw.py"))
    if hits:
        return hits[0]
    return None


def load_cape_ids():
    p = locate_stage8b()
    if p is None:
        print(
            "[WARN] Cannot find 14_run_high_specificity_ccw.py; "
            "CAPE-COD-like audit will be skipped."
        )
        return set()

    spec = importlib.util.spec_from_file_location("stage8b_for_v9", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    if not hasattr(mod, "build_clinical_flags"):
        raise RuntimeError(
            f"{p} does not expose build_clinical_flags()."
        )

    flags = mod.build_clinical_flags()
    if "cape_cod_like" not in flags.columns or "stay_id" not in flags.columns:
        raise RuntimeError(
            "Stage 8B clinical flags lack cape_cod_like/stay_id."
        )
    return set(flags.loc[flags["cape_cod_like"] == 1, "stay_id"])


def existing(cols, df):
    return [c for c in cols if c in df.columns]


def build_trials(panel: pd.DataFrame):
    parts = []

    base_and_current = []
    for c in BASELINE_NUM + BASELINE_BIN + BASELINE_CAT + CURRENT_NUM + CURRENT_BIN:
        if c in panel.columns and c not in base_and_current:
            base_and_current.append(c)

    for h in TRIAL_STARTS:
        x = panel[panel["hour"] == h].copy()
        if x.empty:
            continue

        init = pd.to_numeric(x["any_init_h"], errors="coerce")
        death = pd.to_numeric(x["death_hours_full"], errors="coerce")

        eligible = (
            (death.isna() | (death > h))
            & (init.isna() | (init >= h))
        )
        x = x[eligible].copy()

        init = pd.to_numeric(x["any_init_h"], errors="coerce")
        death = pd.to_numeric(x["death_hours_full"], errors="coerce")

        x["trial_hour"] = float(h)
        x["trial_hour2"] = float(h * h)
        x["trial_window_end"] = float(h + WINDOW_H)

        x["actual_initiate_next6h"] = (
            init.notna()
            & (init >= h)
            & (init < h + WINDOW_H)
            & (death.isna() | (init <= death))
        ).astype(int)

        rel = init - h
        x["early_initiate_0_3h"] = (
            (x["actual_initiate_next6h"] == 1)
            & (rel >= 0)
            & (rel < RECENT_START_H)
        ).astype(int)
        x["recent_initiate_3_6h"] = (
            (x["actual_initiate_next6h"] == 1)
            & (rel >= RECENT_START_H)
            & (rel < RECENT_END_H)
        ).astype(int)

        keep = [
            "stay_id",
            "trial_hour",
            "trial_hour2",
            "trial_window_end",
            "actual_initiate_next6h",
            "early_initiate_0_3h",
            "recent_initiate_3_6h",
            "any_init_h",
            "death_hours_full",
        ] + [c for c in base_and_current if c not in [
            "stay_id", "any_init_h", "death_hours_full"
        ]]

        parts.append(x[keep].copy())

    if not parts:
        raise RuntimeError("No 6h sequential trials were created.")

    trials = pd.concat(parts, ignore_index=True)
    trials["trial_uid"] = np.arange(1, len(trials) + 1, dtype=np.int64)
    trials["trial_idx"] = np.arange(len(trials), dtype=np.int64)
    return trials


def make_preprocessor(df: pd.DataFrame, include_decision_time=False):
    num = existing(
        BASELINE_NUM + BASELINE_BIN + CURRENT_NUM + CURRENT_BIN,
        df,
    )
    cat = existing(BASELINE_CAT, df)

    time_cols = ["trial_hour", "trial_hour2"]
    if include_decision_time:
        for c in ["interval_index", "decision_hour", "decision_hour2"]:
            if c in df.columns:
                time_cols.append(c)

    all_num = []
    for c in num + time_cols:
        if c not in all_num:
            all_num.append(c)

    transformers = []

    if all_num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                all_num,
            )
        )

    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=10,
                            ),
                        ),
                    ]
                ),
                cat,
            )
        )

    return ColumnTransformer(transformers, remainder="drop")


def fit_unpenalized(X, y):
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
            m.fit(X, y)
            return m
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unpenalized logistic failed: {last}")


def fit_ridge(X, y, C):
    m = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    m.fit(X, y)
    return m


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
    x = x[m]
    w = w[m]
    mu = np.average(x, weights=w)
    var = np.average((x - mu) ** 2, weights=w)
    return mu, var


def weighted_smd(x1, w1, x0, w0):
    m1, v1 = weighted_mean_var(x1, w1)
    m0, v0 = weighted_mean_var(x0, w0)
    if not np.isfinite(m1) or not np.isfinite(m0):
        return np.nan
    den = math.sqrt((v1 + v0) / 2.0)
    return (m1 - m0) / den if den > 0 else 0.0


def balance_frame(df: pd.DataFrame):
    out = {}

    for c in (
        CURRENT_NUM
        + CURRENT_BIN
        + BASELINE_NUM
        + BASELINE_BIN
        + ["trial_hour", "decision_hour"]
    ):
        if c not in df.columns or c in out:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        med = s.median()
        out[c] = s.fillna(med if pd.notna(med) else 0.0).astype(float)
        if s.isna().any():
            out[c + "__missing"] = s.isna().astype(float)

    for c in BASELINE_CAT:
        if c not in df.columns:
            continue
        s = df[c].astype("string").fillna("Missing")
        for lev in sorted(s.unique().tolist()):
            out[f"{c}=={lev}"] = (s == lev).astype(float)

    return pd.DataFrame(out, index=df.index)


def fit_baseline_overlap(trials: pd.DataFrame, population: str):
    prep = make_preprocessor(trials, include_decision_time=False)
    X = prep.fit_transform(trials)
    y = trials["actual_initiate_next6h"].astype(int).to_numpy()

    model_label = "unpenalized_logistic"
    try:
        m = fit_unpenalized(X, y)
    except Exception as exc:
        print(
            f"[WARN] {population}: baseline unpenalized e6 model failed; "
            f"using ridge C=10. Error: {exc}"
        )
        m = fit_ridge(X, y, C=10.0)
        model_label = "ridge_C10_fallback"

    e6 = np.clip(m.predict_proba(X)[:, 1], PS_CLIP, 1 - PS_CLIP)
    q = e6 * (1 - e6)

    return e6, q, model_label


def baseline_overlap_diagnostics(trials, population, e6, q, model_label):
    y = trials["actual_initiate_next6h"].astype(int).to_numpy()

    rows = []
    for label, mask in [
        ("all_person_trials", np.ones(len(y), bool)),
        ("actual_initiators_6h", y == 1),
        ("actual_defer_6h", y == 0),
    ]:
        p = e6[mask]
        qq = q[mask]
        rows.append(
            {
                "population": population,
                "baseline_model": model_label,
                "group": label,
                "n": int(mask.sum()),
                "e6_min": float(np.min(p)) if len(p) else np.nan,
                "e6_p01": float(np.quantile(p, .01)) if len(p) else np.nan,
                "e6_median": float(np.median(p)) if len(p) else np.nan,
                "e6_p99": float(np.quantile(p, .99)) if len(p) else np.nan,
                "e6_max": float(np.max(p)) if len(p) else np.nan,
                "q_sum": float(np.sum(qq)) if len(qq) else np.nan,
                "q_ess": ess(qq),
                "e6_lt_0_01_n": int((p < .01).sum()) if len(p) else 0,
                "e6_gt_0_10_n": int((p > .10).sum()) if len(p) else 0,
            }
        )
    return pd.DataFrame(rows)


def build_hourly_long(trials: pd.DataFrame, panel: pd.DataFrame):
    """
    Person-hour rows for the 6h grace period.

    A row exists at j=0,...,5 only if the patient is alive and untreated
    at the beginning of that hour.
    """
    current_cols = existing(CURRENT_NUM + CURRENT_BIN, panel)

    base_cols = [
        c
        for c in (
            ["trial_idx", "trial_uid", "stay_id", "trial_hour", "trial_hour2",
             "any_init_h", "death_hours_full"]
            + BASELINE_NUM + BASELINE_BIN + BASELINE_CAT
        )
        if c in trials.columns
    ]

    dyn = panel[["stay_id", "hour"] + current_cols].copy()
    dyn = dyn.rename(columns={"hour": "decision_hour"})

    parts = []

    for j in range(6):
        z = trials[base_cols].copy()
        z["interval_index"] = int(j)
        z["decision_hour"] = z["trial_hour"].astype(float) + float(j)
        z["decision_hour2"] = z["decision_hour"] ** 2

        init = pd.to_numeric(z["any_init_h"], errors="coerce")
        death = pd.to_numeric(z["death_hours_full"], errors="coerce")

        at_risk = (
            (death.isna() | (death > z["decision_hour"]))
            & (init.isna() | (init >= z["decision_hour"]))
        )
        z = z[at_risk].copy()

        z = z.merge(
            dyn,
            on=["stay_id", "decision_hour"],
            how="left",
            validate="many_to_one",
        )

        init = pd.to_numeric(z["any_init_h"], errors="coerce")
        death = pd.to_numeric(z["death_hours_full"], errors="coerce")

        z["start_next1h"] = (
            init.notna()
            & (init >= z["decision_hour"])
            & (init < z["decision_hour"] + 1.0)
            & (death.isna() | (init <= death))
        ).astype(int)

        z["alive_at_interval_end"] = (
            death.isna() | (death > z["decision_hour"] + 1.0)
        ).astype(int)

        z["no_start_through_interval_end"] = (
            init.isna() | (init >= z["decision_hour"] + 1.0)
        ).astype(int)

        parts.append(z)

    return pd.concat(parts, ignore_index=True, sort=False)


def fit_defer_hourly_model(long):
    prep = make_preprocessor(long, include_decision_time=True)
    X = prep.fit_transform(long)
    y = long["start_next1h"].astype(int).to_numpy()

    m = fit_ridge(X, y, C=0.5)
    p = np.clip(m.predict_proba(X)[:, 1], PS_CLIP, 1 - PS_CLIP)
    return p


def defer_balance_audit(trials, long, q, pstart, population):
    """
    Audit hour-by-hour artificial-censoring balance for the DEFER strategy.

    At hour j:
      target = alive at interval end among those at risk at interval start,
               weighted by target q and prior censoring IPCW.
      survivor = target members who also do not start during this hour,
                 weighted additionally by 1/P(no start this hour).
    """
    n = len(trials)

    pmat = np.full((n, 6), np.nan, float)
    li = long["trial_idx"].to_numpy(int)
    jj = long["interval_index"].to_numpy(int)
    pmat[li, jj] = pstart

    prior = np.ones((n, 6), float)
    current = np.ones((n, 6), float)

    cum = np.ones(n, float)
    for j in range(6):
        prior[:, j] = cum
        pj = np.where(np.isfinite(pmat[:, j]), pmat[:, j], 0.5)
        fac = 1.0 / (1.0 - np.clip(pj, PS_CLIP, 1 - PS_CLIP))
        current[:, j] = cum * fac
        cum = current[:, j]

    bal_rows = []
    sum_rows = []

    for j in range(6):
        z = long[long["interval_index"] == j].copy().reset_index(drop=True)
        ti = z["trial_idx"].to_numpy(int)

        target = z["alive_at_interval_end"].astype(int).to_numpy() == 1
        surv = (
            target
            & (
                z["no_start_through_interval_end"]
                .astype(int)
                .to_numpy()
                == 1
            )
        )

        tw = q[ti] * prior[ti, j]
        sw = q[ti] * current[ti, j]

        Xbal = balance_frame(z)

        for c in Xbal.columns:
            x = Xbal[c].to_numpy(float)
            smd = weighted_smd(
                x[surv],
                sw[surv],
                x[target],
                tw[target],
            )
            bal_rows.append(
                {
                    "population": population,
                    "interval_index": j,
                    "interval_label": f"[{j},{j+1}h)",
                    "variable": c,
                    "smd_after_ipcw": smd,
                    "abs_smd_after_ipcw": (
                        abs(smd) if pd.notna(smd) else np.nan
                    ),
                }
            )

        b = pd.DataFrame(
            [r for r in bal_rows if r["interval_index"] == j]
        )

        fac_current = np.where(
            np.isfinite(pmat[ti, j]),
            1.0
            / (
                1.0
                - np.clip(
                    pmat[ti, j],
                    PS_CLIP,
                    1 - PS_CLIP,
                )
            ),
            np.nan,
        )

        sum_rows.append(
            {
                "population": population,
                "interval_index": j,
                "interval_label": f"[{j},{j+1}h)",
                "at_risk_person_hours_n": len(z),
                "starts_next1h_n": int(z["start_next1h"].sum()),
                "start_rate": float(z["start_next1h"].mean()),
                "alive_end_target_n": int(target.sum()),
                "alive_end_no_start_n": int(surv.sum()),
                "max_abs_smd": b["abs_smd_after_ipcw"].max(),
                "n_abs_smd_gt_0_10": int(
                    (b["abs_smd_after_ipcw"] > .10).sum()
                ),
                "n_abs_smd_gt_0_20": int(
                    (b["abs_smd_after_ipcw"] > .20).sum()
                ),
                "hourly_no_start_ipcw_p99": float(
                    np.nanquantile(fac_current, .99)
                ),
                "hourly_no_start_ipcw_max": float(
                    np.nanmax(fac_current)
                ),
                "cumulative_no_start_ipcw_p99": float(
                    np.quantile(current[ti[target], j], .99)
                )
                if target.any()
                else np.nan,
                "cumulative_no_start_ipcw_max": float(
                    np.max(current[ti[target], j])
                )
                if target.any()
                else np.nan,
            }
        )

    return pd.DataFrame(bal_rows), pd.DataFrame(sum_rows)


def build_recent_riskset(trials, panel):
    """
    Risk set at +3h among patients still alive and untreated.
    Outcome-blind treatment-process endpoint:
        initiate during [3h,6h) before death.
    """
    current_cols = existing(CURRENT_NUM + CURRENT_BIN, panel)

    base_cols = [
        c
        for c in (
            ["trial_idx", "trial_uid", "stay_id", "trial_hour", "trial_hour2",
             "any_init_h", "death_hours_full"]
            + BASELINE_NUM + BASELINE_BIN + BASELINE_CAT
        )
        if c in trials.columns
    ]

    z = trials[base_cols].copy()
    z["interval_index"] = 3
    z["decision_hour"] = z["trial_hour"].astype(float) + RECENT_START_H
    z["decision_hour2"] = z["decision_hour"] ** 2

    init = pd.to_numeric(z["any_init_h"], errors="coerce")
    death = pd.to_numeric(z["death_hours_full"], errors="coerce")

    at_risk = (
        (death.isna() | (death > z["decision_hour"]))
        & (init.isna() | (init >= z["decision_hour"]))
    )
    z = z[at_risk].copy()

    dyn = panel[["stay_id", "hour"] + current_cols].copy()
    dyn = dyn.rename(columns={"hour": "decision_hour"})

    z = z.merge(
        dyn,
        on=["stay_id", "decision_hour"],
        how="left",
        validate="many_to_one",
    )

    init = pd.to_numeric(z["any_init_h"], errors="coerce")
    death = pd.to_numeric(z["death_hours_full"], errors="coerce")

    z["recent_initiate_3_6h"] = (
        init.notna()
        & (init >= z["decision_hour"])
        & (init < z["trial_hour"] + WINDOW_H)
        & (death.isna() | (init <= death))
    ).astype(int)

    return z.reset_index(drop=True)


def recent_model_candidates(recent):
    prep = make_preprocessor(recent, include_decision_time=True)
    X = prep.fit_transform(recent)
    y = recent["recent_initiate_3_6h"].astype(int).to_numpy()

    candidates = {}

    for C in RIDGE_GRID:
        try:
            m = fit_ridge(X, y, C=C)
            candidates[f"recent3h_ridge_C{C:g}"] = np.clip(
                m.predict_proba(X)[:, 1],
                PS_CLIP,
                1 - PS_CLIP,
            )
        except Exception as exc:
            print(f"[WARN] recent3h ridge C={C} failed: {exc}")

    try:
        m = fit_unpenalized(X, y)
        candidates["recent3h_unpenalized"] = np.clip(
            m.predict_proba(X)[:, 1],
            PS_CLIP,
            1 - PS_CLIP,
        )
    except Exception as exc:
        print(f"[WARN] recent3h unpenalized failed: {exc}")

    return candidates


def recent_balance_for_candidate(
    recent,
    q,
    candidate_name,
    p_recent,
    population,
):
    trial_idx = recent["trial_idx"].to_numpy(int)
    target_w = q[trial_idx]

    y = recent["recent_initiate_3_6h"].astype(int).to_numpy() == 1
    p = np.clip(np.asarray(p_recent, float), PS_CLIP, 1 - PS_CLIP)

    recent_w = target_w * (1.0 / p)

    Xbal = balance_frame(recent)

    rows = []
    for c in Xbal.columns:
        x = Xbal[c].to_numpy(float)
        smd = weighted_smd(
            x[y],
            recent_w[y],
            x,
            target_w,
        )
        rows.append(
            {
                "population": population,
                "candidate": candidate_name,
                "variable": c,
                "smd_after_weighting": smd,
                "abs_smd_after_weighting": (
                    abs(smd) if pd.notna(smd) else np.nan
                ),
            }
        )

    bal = pd.DataFrame(rows)

    target_sum = float(target_w.sum())
    recent_sum = float(recent_w[y].sum())
    ratio = recent_sum / target_sum if target_sum > 0 else np.nan

    unique_recent_patients = recent.loc[
        y, "stay_id"
    ].nunique()

    summary = {
        "population": population,
        "candidate": candidate_name,
        "recent_initiator_person_trials_n": int(y.sum()),
        "recent_initiator_unique_patients_n": int(unique_recent_patients),
        "target_riskset_person_trials_n": len(recent),
        "target_weight_sum": target_sum,
        "recent_weighted_sum": recent_sum,
        "representation_ratio": ratio,
        "recent_weight_ess": ess(recent_w[y]),
        "max_abs_smd": bal["abs_smd_after_weighting"].max(),
        "mean_abs_smd": bal["abs_smd_after_weighting"].mean(),
        "n_abs_smd_gt_0_10": int(
            (bal["abs_smd_after_weighting"] > .10).sum()
        ),
        "n_abs_smd_gt_0_20": int(
            (bal["abs_smd_after_weighting"] > .20).sum()
        ),
        "p_recent_min_initiators": float(p[y].min()) if y.any() else np.nan,
        "p_recent_median_initiators": float(
            np.median(p[y])
        )
        if y.any()
        else np.nan,
        "recent_ipcw_p95": float(
            np.quantile(1.0 / p[y], .95)
        )
        if y.any()
        else np.nan,
        "recent_ipcw_p99": float(
            np.quantile(1.0 / p[y], .99)
        )
        if y.any()
        else np.nan,
        "recent_ipcw_max": float(
            np.max(1.0 / p[y])
        )
        if y.any()
        else np.nan,
    }

    return bal, summary


def trial_hour_counts(trials, population):
    rows = []

    for h in TRIAL_STARTS:
        x = trials[trials["trial_hour"] == h]
        if x.empty:
            continue

        y = x["actual_initiate_next6h"].astype(int)
        early = x["early_initiate_0_3h"].astype(int)
        recent = x["recent_initiate_3_6h"].astype(int)

        rows.append(
            {
                "population": population,
                "trial_hour": h,
                "eligible_person_trials_n": len(x),
                "initiate_within6h_n": int(y.sum()),
                "initiate_within6h_rate": float(y.mean()),
                "early_0_3h_n": int(early.sum()),
                "recent_3_6h_n": int(recent.sum()),
                "recent_unique_patients_n": int(
                    x.loc[recent == 1, "stay_id"].nunique()
                ),
            }
        )

    return pd.DataFrame(rows)


def run_population(panel, population):
    trials = build_trials(panel)

    print(f"\n=== V9 6h feasibility: {population} ===")
    print(
        f"patients={trials.stay_id.nunique()} | "
        f"person-trials={len(trials)} | "
        f"initiate<=6h={int(trials.actual_initiate_next6h.sum())}"
    )

    e6, q, baseline_model = fit_baseline_overlap(trials, population)
    overlap = baseline_overlap_diagnostics(
        trials, population, e6, q, baseline_model
    )

    long = build_hourly_long(trials, panel)
    pstart_hour = fit_defer_hourly_model(long)
    defer_bal, defer_sum = defer_balance_audit(
        trials, long, q, pstart_hour, population
    )

    recent = build_recent_riskset(trials, panel)
    candidates = recent_model_candidates(recent)

    all_recent_bal = []
    recent_summaries = []

    for name, p in candidates.items():
        bal, summ = recent_balance_for_candidate(
            recent,
            q,
            name,
            p,
            population,
        )
        all_recent_bal.append(bal)
        recent_summaries.append(summ)

    select = pd.DataFrame(recent_summaries)

    ess_floor = RECENT_ESS_FLOOR[population]

    acceptable = select[
        select["representation_ratio"].between(.90, 1.10)
        & (select["max_abs_smd"] < .10)
        & (select["n_abs_smd_gt_0_10"] == 0)
        & (select["recent_weight_ess"] >= ess_floor)
    ].copy()

    if len(acceptable):
        chosen = acceptable.sort_values(
            ["mean_abs_smd", "recent_weight_ess"],
            ascending=[True, False],
        ).iloc[0]["candidate"]
        recent_status = "RECENT_MODEL_PASS"
    else:
        # For transparent reporting only. This is NOT permission to proceed.
        pool = select[
            select["representation_ratio"].between(.85, 1.15)
            & (select["recent_weight_ess"] >= ess_floor)
        ].copy()
        if pool.empty:
            pool = select.copy()

        chosen = pool.sort_values(
            [
                "n_abs_smd_gt_0_10",
                "max_abs_smd",
                "mean_abs_smd",
            ]
        ).iloc[0]["candidate"]
        recent_status = "NO_RECENT_MODEL_PASS"

    select["selected_for_next_step"] = (
        select["candidate"] == chosen
    ).astype(int)
    select["selection_status"] = recent_status

    defer_all_pass = bool(
        (defer_sum["max_abs_smd"] < .10).all()
        and (defer_sum["n_abs_smd_gt_0_10"] == 0).all()
    )

    chosen_row = select[
        select["candidate"] == chosen
    ].iloc[0]

    broad_gate = (
        recent_status == "RECENT_MODEL_PASS"
        and defer_all_pass
    )

    if broad_gate:
        decision = "PROCEED_TO_6H_FINAL_PURE_CCW_OUTCOME_ANALYSIS"
    else:
        decision = "DO_NOT_PROCEED_6H_SUPPORT_OR_BALANCE_LIMITED"

    y6 = trials["actual_initiate_next6h"].astype(int)
    early = trials["early_initiate_0_3h"].astype(int)
    recent_y = trials["recent_initiate_3_6h"].astype(int)

    summary = pd.DataFrame(
        [
            {
                "population": population,
                "unique_patients_n": trials.stay_id.nunique(),
                "person_trials_n": len(trials),
                "initiate_within6h_person_trials_n": int(y6.sum()),
                "initiate_within6h_unique_patients_n": int(
                    trials.loc[y6 == 1, "stay_id"].nunique()
                ),
                "initiate_within6h_rate": float(y6.mean()),
                "early_0_3h_person_trials_n": int(early.sum()),
                "early_0_3h_unique_patients_n": int(
                    trials.loc[early == 1, "stay_id"].nunique()
                ),
                "recent_3_6h_person_trials_n": int(recent_y.sum()),
                "recent_3_6h_unique_patients_n": int(
                    trials.loc[recent_y == 1, "stay_id"].nunique()
                ),
                "recent_riskset_at_plus3h_n": len(recent),
                "selected_recent_model": chosen,
                "recent_representation_ratio": chosen_row[
                    "representation_ratio"
                ],
                "recent_weight_ess": chosen_row[
                    "recent_weight_ess"
                ],
                "recent_max_abs_smd": chosen_row[
                    "max_abs_smd"
                ],
                "recent_n_abs_smd_gt_0_10": chosen_row[
                    "n_abs_smd_gt_0_10"
                ],
                "defer_all_6_hours_balance_pass": int(
                    defer_all_pass
                ),
                "decision": decision,
            }
        ]
    )

    recommendation = pd.DataFrame(
        [
            {
                "population": population,
                "recent_selected_candidate": chosen,
                "recent_selection_status": recent_status,
                "defer_all_hours_pass": int(defer_all_pass),
                "recent_ess_floor": ess_floor,
                "recent_representation_ratio": chosen_row[
                    "representation_ratio"
                ],
                "recent_weight_ess": chosen_row[
                    "recent_weight_ess"
                ],
                "recent_max_abs_smd": chosen_row[
                    "max_abs_smd"
                ],
                "recent_n_abs_smd_gt_0_10": chosen_row[
                    "n_abs_smd_gt_0_10"
                ],
                "decision": decision,
                "note": (
                    "Outcome-blind support decision only. "
                    "No mortality effect was estimated in V9."
                ),
            }
        ]
    )

    print("\nRecent-model candidates:")
    print(
        select.sort_values(
            [
                "n_abs_smd_gt_0_10",
                "max_abs_smd",
                "mean_abs_smd",
            ]
        ).to_string(index=False)
    )

    print("\nDefer hourly summary:")
    print(defer_sum.to_string(index=False))

    print("\nRecommendation:")
    print(recommendation.to_string(index=False))

    recent_bal = (
        pd.concat(all_recent_bal, ignore_index=True)
        if all_recent_bal
        else pd.DataFrame()
    )

    hour = trial_hour_counts(trials, population)

    return (
        summary,
        select,
        defer_bal,
        defer_sum,
        recent_bal,
        recommendation,
        hour,
        overlap,
    )


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 112)
    print("final causal-analysis stage frozen 6-hour feasibility analysis — 6h initiation-window outcome-blind feasibility/positivity audit")
    print("=" * 112)
    print(
        "No mortality effect. No bootstrap. "
        "Only treatment support, censoring balance, overlap and positivity."
    )

    panel = load_panel()

    broad = run_population(panel, "broad_primary")

    results = [broad]

    cape_ids = load_cape_ids()
    if cape_ids:
        cape_panel = panel[panel["stay_id"].isin(cape_ids)].copy()
        if len(cape_panel):
            cape = run_population(
                cape_panel,
                "CAPE_COD_like",
            )
            results.append(cape)

    pd.concat(
        [r[0] for r in results],
        ignore_index=True,
    ).to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        [r[1] for r in results],
        ignore_index=True,
    ).to_csv(
        OUT_RECENT_SELECT,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        [r[2] for r in results],
        ignore_index=True,
    ).to_csv(
        OUT_DEFER_BAL,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        [r[4] for r in results],
        ignore_index=True,
    ).to_csv(
        OUT_RECENT_BAL,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        [r[5] for r in results],
        ignore_index=True,
    ).to_csv(
        OUT_RECOMMEND,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        [r[6] for r in results],
        ignore_index=True,
    ).to_csv(
        OUT_HOUR,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        [r[7] for r in results],
        ignore_index=True,
    ).to_csv(
        OUT_OVERLAP,
        index=False,
        encoding="utf-8-sig",
    )

    # Also save the compact defer-hour summaries separately for easy inspection.
    defer_summary_path = (
        OUTPUT_DIR
        / "16V9H_FINAL_6H_DEFER_HOURLY_SUMMARY.csv"
    )
    pd.concat(
        [r[3] for r in results],
        ignore_index=True,
    ).to_csv(
        defer_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 112)
    print("STAGE 10 V9 COMPLETE")
    print("=" * 112)

    print("\nGenerated eight audit CSVs:")
    for p in [
        OUT_SUMMARY,
        OUT_RECENT_SELECT,
        OUT_DEFER_BAL,
        OUT_RECENT_BAL,
        OUT_RECOMMEND,
        OUT_HOUR,
        OUT_OVERLAP,
        defer_summary_path,
    ]:
        print(" ", p)

    print(
        "\nDecision rule: only if Broad says "
        "PROCEED_TO_6H_FINAL_PURE_CCW_OUTCOME_ANALYSIS "
        "do we build the mortality/bootstrap stage."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise

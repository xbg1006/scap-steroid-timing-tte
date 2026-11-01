from __future__ import annotations

"""
final causal-analysis stage outcome-blind 2-hour design audit — Hour-2 recent-initiator IPCW model audit

NO OUTCOME MODEL TUNING. NO BOOTSTRAP.

Purpose
-------
V7 showed:
- DEFER artificial-censoring IPCW balances extremely well.
- INITIATE-by-2h recent-initiator weighting has residual imbalance.

outcome-blind 2-hour design audit changes only the SECOND-HOUR censoring model used for:
    P(start during [1h,2h) | alive + untreated at +1h, history at +1h)

It compares a fixed, pre-declared candidate set:
1. V7 pooled 1h ridge model (benchmark)
2. hour-2-only ridge logistic, several fixed C values
3. hour-2-only unpenalized logistic if it converges
4. optional entropy-calibrated version of the best hour-2 logistic weights
   using a PRE-SPECIFIED core covariate set (Broad only; sensitivity)

Candidate choice uses only:
- covariates
- treatment/censoring process
- weight ESS
- representation ratio

It NEVER uses mortality/outcomes to select a model.
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
    from sklearn.linear_model import LogisticRegression
except Exception as exc:
    raise RuntimeError("scikit-learn is required") from exc

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

PROJECT_ROOT = Path(config.PROJECT_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
HERE = Path(__file__).resolve().parent

V5_PATH = HERE / "08_two_hour_overlap_bootstrap_diagnostic.py"
V6_PATH = HERE / "09_two_hour_hourly_ipcw_diagnostic.py"
V7_PATH = HERE / "10_two_hour_pure_ccw_support_audit.py"

OUT_SELECT = OUTPUT_DIR / "two_hour_design_model_selection.csv"
OUT_BAL = OUTPUT_DIR / "16outcome-blind 2-hour design auditB_FINAL_HOUR2_MODEL_BALANCE.csv"
OUT_WEIGHT = OUTPUT_DIR / "16outcome-blind 2-hour design auditC_FINAL_HOUR2_WEIGHT_DIAGNOSTICS.csv"
OUT_EFFECT = OUTPUT_DIR / "16outcome-blind 2-hour design auditD_FINAL_PURE_CCW_EFFECTS_BY_MODEL.csv"
OUT_RECOMMEND = OUTPUT_DIR / "two_hour_design_recommendation.csv"

SEED = 20261028
PS_CLIP = 1e-5

RIDGE_GRID = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 10.0]

# Pre-specified variables for calibration sensitivity.
# This set is chosen before outcome-blind 2-hour design audit sees any mortality outcome.
CALIBRATION_CORE = [
    "trial_hour",
    "dyn_sf_ratio",
    "dyn_fio2_last6",
    "dyn_spo2_last6",
    "dyn_rr_max6",
    "dyn_map_min6",
    "dyn_lactate_max12",
    "dyn_ph_min12",
    "dyn_gcs_motor_last6",
    "support_level",
    "vasopressor_agents_active",
    "support_imv",
    "support_niv",
    "support_hfnc",
    "age",
    "chronic_kidney_disease",
    "malignancy",
    "shock_active_at_t0",
    "gender",
    "race_group",
    "anchor_year_group",
    "careunit_group",
]


def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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


def weighted_smd(x1, w1, x0, w0):
    m1, v1 = weighted_mean_var(x1, w1)
    m0, v0 = weighted_mean_var(x0, w0)
    if not np.isfinite(m1) or not np.isfinite(m0):
        return np.nan
    den = math.sqrt((v1 + v0) / 2)
    return (m1 - m0) / den if den > 0 else 0.0


def fit_hour2_ridge(X, y, C):
    model = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    model.fit(X, y)
    return model


def fit_hour2_unpenalized(v5, X, y):
    return v5._fit_unpenalized_logistic(X, y)


def build_balance_frame(v5, z1):
    """
    Interpretable original-scale balance frame, including missingness indicators.
    """
    out = {}
    for c in (
        v5.CURRENT_NUM + v5.CURRENT_BIN
        + v5.BASELINE_NUM + v5.BASELINE_BIN
        + ["trial_hour", "decision_hour"]
    ):
        if c in z1.columns and c not in out:
            s = pd.to_numeric(z1[c], errors="coerce")
            # median-fill for the displayed value + explicit missingness indicator
            med = s.median()
            out[c] = s.fillna(med if pd.notna(med) else 0.0).astype(float)
            if s.isna().any():
                out[c + "__missing"] = s.isna().astype(float)

    for c in [x for x in v5.BASELINE_CAT if x in z1.columns]:
        s = z1[c].astype("string").fillna("Missing")
        for lev in sorted(s.unique().tolist()):
            out[f"{c}=={lev}"] = (s == lev).astype(float)

    return pd.DataFrame(out, index=z1.index)


def candidate_balance(v5, prep, q, p_late, model_name, population):
    long = prep["long"]
    z1 = long[long["interval_index"] == 1].copy().reset_index(drop=True)
    trial_idx = z1["trial_idx"].to_numpy(int)

    p = np.clip(np.asarray(p_late, float), PS_CLIP, 1 - PS_CLIP)
    late = z1["start_next1h"].astype(int).to_numpy() == 1

    target_w = q[trial_idx]
    recent_w = target_w * (1.0 / p)

    Xbal = build_balance_frame(v5, z1)
    rows = []
    for c in Xbal.columns:
        x = Xbal[c].to_numpy(float)
        smd = weighted_smd(
            x[late], recent_w[late],
            x, target_w,
        )
        rows.append({
            "population": population,
            "candidate": model_name,
            "variable": c,
            "smd_after_weighting": smd,
            "abs_smd_after_weighting": abs(smd) if pd.notna(smd) else np.nan,
        })

    target_sum = float(target_w.sum())
    recent_sum = float(recent_w[late].sum())
    ratio = recent_sum / target_sum if target_sum > 0 else np.nan
    recent_ess = ess(recent_w[late])

    bal = pd.DataFrame(rows)
    summary = {
        "population": population,
        "candidate": model_name,
        "recent_initiators_n": int(late.sum()),
        "target_riskset_n": len(z1),
        "target_weight_sum": target_sum,
        "recent_weighted_sum": recent_sum,
        "representation_ratio": ratio,
        "recent_weight_ess": recent_ess,
        "max_abs_smd": bal["abs_smd_after_weighting"].max(),
        "mean_abs_smd": bal["abs_smd_after_weighting"].mean(),
        "n_abs_smd_gt_0_10": int((bal["abs_smd_after_weighting"] > .10).sum()),
        "n_abs_smd_gt_0_20": int((bal["abs_smd_after_weighting"] > .20).sum()),
        "pstart_min_recent": float(p[late].min()) if late.any() else np.nan,
        "pstart_median_recent": float(np.median(p[late])) if late.any() else np.nan,
        "ipcw_recent_p95": float(np.quantile(1/p[late], .95)) if late.any() else np.nan,
        "ipcw_recent_p99": float(np.quantile(1/p[late], .99)) if late.any() else np.nan,
        "ipcw_recent_max": float(np.max(1/p[late])) if late.any() else np.nan,
    }
    return bal, summary, recent_w, target_w, late, z1


def calibration_design(z1):
    """
    Small, pre-specified calibration matrix.

    Continuous variables:
      median impute + standardize + missingness flag.
    Binary variables:
      0/1.
    Selected categorical variables:
      one-hot, dropping one level to avoid exact linear dependence.
    """
    cols = []
    arrs = []

    continuous = [
        "trial_hour",
        "dyn_sf_ratio",
        "dyn_fio2_last6",
        "dyn_spo2_last6",
        "dyn_rr_max6",
        "dyn_map_min6",
        "dyn_lactate_max12",
        "dyn_ph_min12",
        "dyn_gcs_motor_last6",
        "support_level",
        "vasopressor_agents_active",
        "age",
    ]
    binary = [
        "support_imv", "support_niv", "support_hfnc",
        "chronic_kidney_disease", "malignancy", "shock_active_at_t0",
    ]
    cats = ["gender", "race_group", "anchor_year_group", "careunit_group"]

    for c in continuous:
        if c not in z1.columns:
            continue
        s = pd.to_numeric(z1[c], errors="coerce")
        miss = s.isna().astype(float).to_numpy()
        med = s.median()
        x = s.fillna(med if pd.notna(med) else 0.0).to_numpy(float)
        sd = np.std(x)
        if sd > 1e-8:
            x = (x - np.mean(x)) / sd
        else:
            x = x - np.mean(x)
        arrs.append(x)
        cols.append(c)
        if miss.sum() > 0:
            arrs.append(miss)
            cols.append(c + "__missing")

    for c in binary:
        if c not in z1.columns:
            continue
        x = pd.to_numeric(z1[c], errors="coerce").fillna(0).to_numpy(float)
        if np.std(x) > 1e-8:
            arrs.append(x)
            cols.append(c)

    for c in cats:
        if c not in z1.columns:
            continue
        s = z1[c].astype("string").fillna("Missing")
        levels = sorted(s.unique().tolist())
        # Drop the first level.
        for lev in levels[1:]:
            x = (s == lev).astype(float).to_numpy()
            if np.std(x) > 1e-8:
                arrs.append(x)
                cols.append(f"{c}=={lev}")

    if not arrs:
        return np.zeros((len(z1), 0)), []
    return np.column_stack(arrs), cols


def entropy_calibrate(base_recent_w, late, target_w, Z, ridge=1e-6):
    """
    Exponential tilting of initial recent-initiator weights.

    Calibrates total weight + selected covariate totals toward the target risk set.
    Returns None if optimization fails.

    The calibration never uses outcome information.
    """
    if minimize is None:
        return None, {"success": False, "message": "scipy_not_available"}

    late = np.asarray(late, bool)
    Zr = Z[late]
    d = np.asarray(base_recent_w[late], float)
    tw = np.asarray(target_w, float)

    # Add intercept for exact total-weight calibration.
    Zr2 = np.column_stack([np.ones(len(Zr)), Zr])
    Zt2 = np.column_stack([np.ones(len(Z)), Z])
    target_totals = Zt2.T @ tw

    # Scale columns by target total magnitude to improve conditioning.
    scale = np.maximum(np.abs(target_totals), 1.0)
    Zr_s = Zr2 / scale
    target_s = target_totals / scale

    def obj(lam):
        eta = np.clip(Zr_s @ lam, -30, 30)
        w = d * np.exp(eta)
        return float(np.sum(w) - target_s @ lam + 0.5 * ridge * np.dot(lam, lam))

    def grad(lam):
        eta = np.clip(Zr_s @ lam, -30, 30)
        w = d * np.exp(eta)
        return Zr_s.T @ w - target_s + ridge * lam

    res = minimize(
        obj,
        np.zeros(Zr_s.shape[1], float),
        jac=grad,
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-9},
    )

    if not res.success:
        return None, {"success": False, "message": str(res.message)}

    eta = np.clip(Zr_s @ res.x, -30, 30)
    w_recent = d * np.exp(eta)

    out = np.zeros(len(late), float)
    out[late] = w_recent

    achieved = Zr2.T @ w_recent
    rel = np.max(np.abs(achieved - target_totals) / np.maximum(np.abs(target_totals), 1.0))

    return out, {
        "success": True,
        "message": str(res.message),
        "max_relative_moment_error": float(rel),
        "n_constraints": int(Zr2.shape[1]),
    }


def calibrated_balance(v5, z1, target_w, late, calibrated_recent_w,
                       population, candidate_name):
    Xbal = build_balance_frame(v5, z1)
    rows = []
    for c in Xbal.columns:
        x = Xbal[c].to_numpy(float)
        smd = weighted_smd(
            x[late], calibrated_recent_w[late],
            x, target_w,
        )
        rows.append({
            "population": population,
            "candidate": candidate_name,
            "variable": c,
            "smd_after_weighting": smd,
            "abs_smd_after_weighting": abs(smd) if pd.notna(smd) else np.nan,
        })
    bal = pd.DataFrame(rows)
    ratio = calibrated_recent_w[late].sum() / target_w.sum()
    summary = {
        "population": population,
        "candidate": candidate_name,
        "recent_initiators_n": int(late.sum()),
        "target_riskset_n": len(z1),
        "target_weight_sum": float(target_w.sum()),
        "recent_weighted_sum": float(calibrated_recent_w[late].sum()),
        "representation_ratio": float(ratio),
        "recent_weight_ess": ess(calibrated_recent_w[late]),
        "max_abs_smd": bal["abs_smd_after_weighting"].max(),
        "mean_abs_smd": bal["abs_smd_after_weighting"].mean(),
        "n_abs_smd_gt_0_10": int((bal["abs_smd_after_weighting"] > .10).sum()),
        "n_abs_smd_gt_0_20": int((bal["abs_smd_after_weighting"] > .20).sum()),
        "pstart_min_recent": np.nan,
        "pstart_median_recent": np.nan,
        "ipcw_recent_p95": np.nan,
        "ipcw_recent_p99": np.nan,
        "ipcw_recent_max": np.nan,
    }
    return bal, summary


def effect_with_custom_recent(v6, prep, q, pstart_long, custom_recent_w, target):
    """
    Pure CCW effect where only the initiate post-grace recent-initiator weights
    are replaced by the audited candidate/calibrated weights.
    """
    trials = prep["trials"]
    geom = prep["geom"]
    y2 = prep["y2"]

    # Start with V7-style probability arrays.
    p = np.full((len(trials), 2), np.nan, float)
    idx = prep["long"]["trial_idx"].to_numpy(int)
    ii = prep["long"]["interval_index"].to_numpy(int)
    p[idx, ii] = pstart_long

    p0 = np.clip(p[:, 0], PS_CLIP, 1 - PS_CLIP)
    p1 = np.where(np.isfinite(p[:, 1]), p[:, 1], 0.5)
    p1 = np.clip(p1, PS_CLIP, 1 - PS_CLIP)

    trial_hour = trials["trial_hour"].to_numpy(float)
    init_abs = pd.to_numeric(trials["any_init_h"], errors="coerce").to_numpy(float)
    rel = init_abs - trial_hour
    early = (y2 == 1) & np.isfinite(rel) & (rel >= 0) & (rel < 1)
    late = (y2 == 1) & np.isfinite(rel) & (rel >= 1) & (rel < 2)
    defer = y2 == 0

    w_pre = q
    w_defer_1 = q / (1 - p0)
    w_defer_post = q / ((1 - p0) * (1 - p1))

    w_init_post = np.zeros(len(trials), float)
    w_init_post[early] = q[early]

    # custom_recent_w is in z1 row order. Map back to trial rows.
    z1 = prep["long"][prep["long"]["interval_index"] == 1].copy().reset_index(drop=True)
    t_idx = z1["trial_idx"].to_numpy(int)
    late_z1 = z1["start_next1h"].astype(int).to_numpy() == 1
    w_init_post[t_idx[late_z1]] = custom_recent_w[late_z1]

    s0 = v6.phase_survival(geom["end0"], geom["event0"], w_pre, 0.0, 1.0)
    s0 *= v6.phase_survival(geom["end0"], geom["event0"], w_defer_1, 1.0, 2.0)
    s0 *= v6.phase_survival(geom["end0"], geom["event0"], w_defer_post, 2.0, np.inf)

    s1 = v6.phase_survival(geom["end1"], geom["event1"], w_pre, 0.0, 1.0)
    s1 *= v6.phase_survival(geom["end1"], geom["event1"], w_pre, 1.0, 2.0)
    s1 *= v6.phase_survival(geom["end1"], geom["event1"], w_init_post, 2.0, np.inf)

    r0, r1 = 1-s0, 1-s1
    return r0, r1, r1-r0, (r1/r0 if r0 > 0 else np.nan)


def run_population(v5, v6, v7, trials, panel, population):
    print(f"\n=== outcome-blind 2-hour design audit hour-2 model audit: {population} ===")

    prep = v6.prepare_population(v5, trials, panel)

    # Baseline ATO target q.
    m_base = v5._fit_unpenalized_logistic(prep["X_base"], prep["y2"])
    e2 = np.clip(
        m_base.predict_proba(prep["X_base"])[:, 1],
        PS_CLIP, 1-PS_CLIP
    )
    q = e2 * (1-e2)

    # First-hour model stays V7 pooled ridge; it already passed balance.
    pooled = v6.fit_ridge_logistic(prep["X_den"], prep["y1"], C=0.5)
    p_pooled_all = np.clip(
        pooled.predict_proba(prep["X_den"])[:, 1],
        PS_CLIP, 1-PS_CLIP
    )

    long = prep["long"].copy()
    mask1 = long["interval_index"].to_numpy(int) == 1
    X1 = prep["X_den"][mask1]
    y1 = prep["y1"][mask1]

    candidates = {}

    # Benchmark: V7 pooled prediction at hour 2.
    candidates["pooled_ridge_C0.5_V7"] = p_pooled_all[mask1]

    # Fixed hour-2-only ridge grid.
    for C in RIDGE_GRID:
        try:
            m = fit_hour2_ridge(X1, y1, C)
            candidates[f"hour2_only_ridge_C{C:g}"] = np.clip(
                m.predict_proba(X1)[:, 1], PS_CLIP, 1-PS_CLIP
            )
        except Exception as exc:
            print(f"[WARN] hour2 ridge C={C} failed: {exc}")

    # Unpenalized candidate.
    try:
        m = fit_hour2_unpenalized(v5, X1, y1)
        candidates["hour2_only_unpenalized"] = np.clip(
            m.predict_proba(X1)[:, 1], PS_CLIP, 1-PS_CLIP
        )
    except Exception as exc:
        print(f"[WARN] hour2 unpenalized failed: {exc}")

    all_bal = []
    summaries = []
    recent_weights = {}
    z1_ref = None
    late_ref = None
    target_w_ref = None

    for name, p1 in candidates.items():
        bal, summ, recent_w, target_w, late, z1 = candidate_balance(
            v5, prep, q, p1, name, population
        )
        all_bal.append(bal)
        summaries.append(summ)
        recent_weights[name] = recent_w
        z1_ref, late_ref, target_w_ref = z1, late, target_w

    summary_df = pd.DataFrame(summaries)

    # Pre-declared acceptable candidate:
    # representation 0.90-1.10, max SMD <0.10, no >0.10, ESS >=50 broad / >=15 CAPE.
    ess_floor = 50 if population == "broad_primary" else 15
    acceptable = summary_df[
        summary_df["representation_ratio"].between(.90, 1.10)
        & (summary_df["max_abs_smd"] < .10)
        & (summary_df["n_abs_smd_gt_0_10"] == 0)
        & (summary_df["recent_weight_ess"] >= ess_floor)
    ].copy()

    if len(acceptable):
        selected = acceptable.sort_values(
            ["mean_abs_smd", "recent_weight_ess"],
            ascending=[True, False]
        ).iloc[0]["candidate"]
        selection_status = "MODEL_BASED_PASS"
    else:
        # Best non-outcome-based model to seed calibration.
        pool = summary_df[
            summary_df["representation_ratio"].between(.85, 1.15)
            & (summary_df["recent_weight_ess"] >= ess_floor)
        ].copy()
        if pool.empty:
            pool = summary_df.copy()
        selected = pool.sort_values(
            ["n_abs_smd_gt_0_10", "max_abs_smd", "mean_abs_smd"],
            ascending=True
        ).iloc[0]["candidate"]
        selection_status = "NO_MODEL_BASED_PASS"

    # Optional entropy-calibrated sensitivity, Broad only.
    calibration_info = None
    if population == "broad_primary" and z1_ref is not None:
        Z, znames = calibration_design(z1_ref)
        base_recent_w = recent_weights[selected]
        cal_w, info = entropy_calibrate(
            base_recent_w, late_ref, target_w_ref, Z, ridge=1e-6
        )
        calibration_info = info
        if cal_w is not None:
            cname = f"{selected}__entropy_calibrated_core"
            bal_c, summ_c = calibrated_balance(
                v5, z1_ref, target_w_ref, late_ref,
                cal_w, population, cname
            )
            summ_c["calibration_constraints_n"] = info["n_constraints"]
            summ_c["calibration_max_relative_moment_error"] = info["max_relative_moment_error"]
            all_bal.append(bal_c)
            summary_df = pd.concat(
                [summary_df, pd.DataFrame([summ_c])],
                ignore_index=True
            )
            recent_weights[cname] = cal_w

            cal_pass = (
                .90 <= summ_c["representation_ratio"] <= 1.10
                and summ_c["max_abs_smd"] < .10
                and summ_c["n_abs_smd_gt_0_10"] == 0
                and summ_c["recent_weight_ess"] >= ess_floor
            )
            if selection_status == "NO_MODEL_BASED_PASS" and cal_pass:
                selected = cname
                selection_status = "CALIBRATED_PASS"

    # For effect calculation we need pstart_long:
    # interval 0 remains pooled V7; interval 1 uses selected candidate probabilities
    # unless selected is calibrated, where post-grace recent weights are supplied directly.
    selected_base = selected.split("__entropy_calibrated_core")[0]
    p1_selected = candidates[selected_base]
    pstart_long = p_pooled_all.copy()
    pstart_long[mask1] = p1_selected

    effect_rows = []
    for name in [x for x in summary_df["candidate"].tolist()]:
        if name.endswith("__entropy_calibrated_core"):
            custom_w = recent_weights[name]
        else:
            custom_w = recent_weights[name]

        # use model's p1 for DEFER denominator only if candidate exists;
        # for calibrated row the underlying selected-base p1 is used.
        base_name = name.split("__entropy_calibrated_core")[0]
        p1_model = candidates.get(base_name, p1_selected)
        p_long = p_pooled_all.copy()
        p_long[mask1] = p1_model

        r0, r1, rd, rr = effect_with_custom_recent(
            v6, prep, q, p_long, custom_w, "ATO"
        )
        effect_rows.append({
            "population": population,
            "candidate": name,
            "risk_defer": r0,
            "risk_initiate": r1,
            "risk_difference": rd,
            "risk_ratio": rr,
        })

    # Attach selection flags.
    summary_df["selected_for_next_step"] = (
        summary_df["candidate"] == selected
    ).astype(int)
    summary_df["selection_status"] = selection_status

    print("\nCandidate summaries:")
    print(
        summary_df.sort_values(
            ["n_abs_smd_gt_0_10", "max_abs_smd", "mean_abs_smd"]
        ).to_string(index=False)
    )
    print(f"\nSelected: {selected} | status={selection_status}")

    recommendation = pd.DataFrame([{
        "population": population,
        "selected_candidate": selected,
        "selection_status": selection_status,
        "model_based_pass_exists": int(len(acceptable) > 0),
        "ess_floor": ess_floor,
        "calibration_available": int(calibration_info is not None and calibration_info.get("success", False)),
        "decision": (
            "PROCEED_TO_FINAL_BOOTSTRAP"
            if selection_status in ["MODEL_BASED_PASS", "CALIBRATED_PASS"]
            else "DO_NOT_BOOTSTRAP_REDEFINE_OR_ACKNOWLEDGE_POSITIVITY_LIMIT"
        ),
    }])

    return (
        summary_df,
        pd.concat(all_bal, ignore_index=True),
        pd.DataFrame(effect_rows),
        recommendation,
    )


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    v5 = load_module(V5_PATH, "v5")
    v6 = load_module(V6_PATH, "v6")
    v7 = load_module(V7_PATH, "v7")

    print("=" * 110)
    print("final causal-analysis stage outcome-blind 2-hour design audit — Hour-2 recent-initiator IPCW model audit")
    print("=" * 110)
    print(
        "Outcome-blind model selection only. NO bootstrap.\n"
        "The DEFER censoring model is frozen because V7 already showed excellent balance."
    )

    panel = v5.load_panel()
    broad_trials = v5.build_sequential_trials(panel)

    stage8b = v5.import_stage8b()
    flags = stage8b.build_clinical_flags()
    cape_ids = set(flags.loc[flags["cape_cod_like"] == 1, "stay_id"])
    cape_panel = panel[panel["stay_id"].isin(cape_ids)].copy()
    cape_trials = v5.build_sequential_trials(cape_panel)

    b = run_population(v5, v6, v7, broad_trials, panel, "broad_primary")
    c = run_population(v5, v6, v7, cape_trials, cape_panel, "CAPE_COD_like")

    select = pd.concat([b[0], c[0]], ignore_index=True)
    bal = pd.concat([b[1], c[1]], ignore_index=True)
    effects = pd.concat([b[2], c[2]], ignore_index=True)
    rec = pd.concat([b[3], c[3]], ignore_index=True)

    select.to_csv(OUT_SELECT, index=False, encoding="utf-8-sig")
    bal.to_csv(OUT_BAL, index=False, encoding="utf-8-sig")

    # Weight diagnostics are the selection table plus a focused subset.
    select[
        [
            "population", "candidate", "recent_initiators_n",
            "target_weight_sum", "recent_weighted_sum",
            "representation_ratio", "recent_weight_ess",
            "pstart_min_recent", "pstart_median_recent",
            "ipcw_recent_p95", "ipcw_recent_p99", "ipcw_recent_max",
            "selected_for_next_step", "selection_status",
        ]
    ].to_csv(OUT_WEIGHT, index=False, encoding="utf-8-sig")

    effects.to_csv(OUT_EFFECT, index=False, encoding="utf-8-sig")
    rec.to_csv(OUT_RECOMMEND, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("outcome-blind 2-hour design audit COMPLETE")
    print("=" * 110)
    print("\nUpload these five CSVs:")
    for p in [OUT_SELECT, OUT_BAL, OUT_WEIGHT, OUT_EFFECT, OUT_RECOMMEND]:
        print(" ", p)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise

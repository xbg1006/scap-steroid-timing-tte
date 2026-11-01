from __future__ import annotations

"""
E3.2 — eICU external inference-stability audit

PURPOSE
-------
E3's frozen external POINT estimate is NOT changed.

E3.2 addresses a post-effect inferential problem:
the original patient-cluster bootstrap refit unpenalized logistic nuisance
models in every replicate, and the resulting bootstrap distribution showed
severe nonregularity / near-separation:
    - some defer risks near zero
    - some initiate risks near one
    - extremely large RR values

E3.2 does NOT:
    - change the 6h strategy
    - change the 3–6h recent window
    - change the cohort
    - change the outcome
    - trim/cap weights to obtain a preferred result
    - choose among sensitivity methods based on the outcome direction

It reports ALL of the following side-by-side:

A) Frozen E3 primary point estimate
   - exact same unpenalized nuisance models
   - exact same weighted Aalen-Johansen estimator

B) Existing original E3 refitted-unpenalized bootstrap
   - not discarded
   - explicitly diagnosed as stable/unstable

C) Fixed-nuisance patient-cluster bootstrap
   - nuisance probabilities fixed at full-sample frozen E3 estimates
   - resamples patient clusters and recomputes weighted Aalen-Johansen
   - conditional/sampling sensitivity only
   - does NOT include nuisance-model estimation uncertainty

D) Ridge-stabilized nuisance sensitivity
   - full-sample sensitivity point + cluster bootstrap
   - baseline e6: L2 logistic C=1
   - recent initiation: L2 logistic C=1
   - defer censoring remains frozen ridge C=0.5
   - intended only as a separation-control sensitivity
   - never replaces the frozen E3 point estimate

All outputs must be reported. No "best" CI is selected.
"""

from pathlib import Path
import os
import importlib.util
import traceback
import warnings
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent

E31_PATH = HERE / "03_final_external_effect.py"
E21_PATH = HERE / "02_locked_mapping_empirical_support.py"

E3_DIR = HERE / "outputs_eicu_E3"
CACHE_DIR = E3_DIR / "cache"

TRIALS_PATH = CACHE_DIR / "e3_trials.parquet"
PANEL_PATH = CACHE_DIR / "e3_dynamic_panel.parquet"
PRIMARY_EFFECT_PATH = E3_DIR / "external_effect_estimates.csv"
PRIMARY_BOOT_PATH = (
    E3_DIR
    / "checkpoints"
    / "eicu_e3_broad_6h_pure_ccw_cluster_bootstrap.csv"
)

OUT_PRIMARY_AUDIT = E3_DIR / "E32_01_ORIGINAL_BOOTSTRAP_INSTABILITY_AUDIT.csv"
OUT_FIXED = E3_DIR / "E32_02_FIXED_NUISANCE_BOOTSTRAP.csv"
OUT_RIDGE = E3_DIR / "E32_03_RIDGE_SENSITIVITY_EFFECT.csv"
OUT_RIDGE_DIAG = E3_DIR / "E32_04_RIDGE_BOOTSTRAP_DIAGNOSTICS.csv"
OUT_COMPARE = E3_DIR / "E32_05_INFERENCE_STABILITY_COMPARISON.csv"
OUT_RECOMMEND = E3_DIR / "E32_06_INTERPRETATION_FLAG.csv"

CHECKPOINT_DIR = E3_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_CKPT = CHECKPOINT_DIR / "e32_fixed_nuisance_cluster_bootstrap.csv"
RIDGE_CKPT = CHECKPOINT_DIR / "e32_ridge_nuisance_cluster_bootstrap.csv"

SEED = 20261028
FIXED_BOOTSTRAP_TARGET = 500
RIDGE_BOOTSTRAP_TARGET = 200
CHECKPOINT_EVERY = 20

RIDGE_C = 1.0
PS_CLIP = 1e-5

# These are diagnostic thresholds only.
# They do not delete or select replicates.
RISK_LOW_FLAG = 0.02
RISK_HIGH_FLAG = 0.80
RR_HIGH_FLAG = 5.0
RECENT_IPCW_HIGH_FLAG = 1000.0


def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def ci95(x):
    z = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(z) == 0:
        return np.nan, np.nan
    return tuple(np.quantile(z, [0.025, 0.975]))


def ess(w):
    x = np.asarray(w, float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    return (x.sum() ** 2) / np.sum(x ** 2)


def load_checkpoint(path, required):
    if not path.exists():
        return pd.DataFrame(columns=required)
    try:
        x = pd.read_csv(path)
        if not set(required).issubset(x.columns):
            return pd.DataFrame(columns=required)
        return x.copy()
    except Exception:
        return pd.DataFrame(columns=required)


def save_checkpoint(path, df):
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def original_bootstrap_audit():
    if not PRIMARY_BOOT_PATH.exists():
        raise FileNotFoundError(PRIMARY_BOOT_PATH)

    b = pd.read_csv(PRIMARY_BOOT_PATH)
    required = {
        "risk_defer",
        "risk_initiate",
        "risk_difference",
        "risk_ratio",
    }
    if not required.issubset(b.columns):
        raise RuntimeError("Original E3 checkpoint lacks expected columns.")

    rows = []

    for name, s in [
        ("risk_defer", b["risk_defer"]),
        ("risk_initiate", b["risk_initiate"]),
        ("risk_difference", b["risk_difference"]),
        ("risk_ratio", b["risk_ratio"]),
    ]:
        lo, hi = ci95(s)
        rows.append({
            "metric": name,
            "n": len(s),
            "mean": float(s.mean()),
            "sd": float(s.std(ddof=1)),
            "p025": lo,
            "median": float(s.median()),
            "p975": hi,
            "min": float(s.min()),
            "max": float(s.max()),
        })

    flags = pd.DataFrame([{
        "metric": "replicate_instability_flags",
        "n": len(b),
        "mean": np.nan,
        "sd": np.nan,
        "p025": np.nan,
        "median": np.nan,
        "p975": np.nan,
        "min": np.nan,
        "max": np.nan,
        "defer_risk_below_0_02_n": int(
            (b["risk_defer"] < RISK_LOW_FLAG).sum()
        ),
        "initiate_risk_below_0_02_n": int(
            (b["risk_initiate"] < RISK_LOW_FLAG).sum()
        ),
        "initiate_risk_above_0_80_n": int(
            (b["risk_initiate"] > RISK_HIGH_FLAG).sum()
        ),
        "risk_ratio_above_5_n": int(
            (b["risk_ratio"] > RR_HIGH_FLAG).sum()
        ),
        "risk_ratio_above_10_n": int(
            (b["risk_ratio"] > 10).sum()
        ),
        "interpretation": (
            "Original percentile bootstrap is inferentially unstable if "
            "a material fraction of replicates approach boundary risks or "
            "very large risk ratios."
        ),
    }])

    out = pd.concat([pd.DataFrame(rows), flags], ignore_index=True)
    out.to_csv(
        OUT_PRIMARY_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )
    return b, out


def prepare(e31, e21):
    if not TRIALS_PATH.exists() or not PANEL_PATH.exists():
        raise FileNotFoundError(
            "E3 cache is missing. Run E3.1 first."
        )

    trials = pd.read_parquet(TRIALS_PATH)
    panel = pd.read_parquet(PANEL_PATH)

    # E3 cache should already contain outcome/cluster columns.
    need = {
        "cluster_id",
        "hospital_death",
        "hospital_discharge_offset_min",
        "actual_initiate_next6h",
    }
    missing = need - set(trials.columns)
    if missing:
        raise RuntimeError(
            f"E3 cached trials missing columns: {sorted(missing)}"
        )

    prep = e31.prepare_models(e21, trials, panel)
    return trials, panel, prep


def frozen_point(e31, prep):
    row_mult, e6, q, p_hour, p_recent = e31.fit_nuisance(
        prep,
        None,
    )
    est = e31.estimate_effect(
        prep,
        q,
        p_hour,
        p_recent,
        row_mult,
    )
    return row_mult, e6, q, p_hour, p_recent, est


def verify_primary_point(est):
    if not PRIMARY_EFFECT_PATH.exists():
        raise FileNotFoundError(PRIMARY_EFFECT_PATH)
    p = pd.read_csv(PRIMARY_EFFECT_PATH).iloc[0]

    diffs = {
        "risk_defer": abs(float(p["risk_defer"]) - est["risk_defer"]),
        "risk_initiate": abs(float(p["risk_initiate"]) - est["risk_initiate"]),
        "risk_difference": abs(
            float(p["risk_difference"]) - est["risk_difference"]
        ),
        "risk_ratio": abs(float(p["risk_ratio"]) - est["risk_ratio"]),
    }
    worst = max(diffs.values())
    if worst > 1e-8:
        raise RuntimeError(
            f"Frozen E3 point estimate did not reproduce; max diff={worst}"
        )
    print("[LOCK] Frozen E3 point estimate reproduced exactly.")


def fixed_nuisance_bootstrap(e31, prep, q, p_hour, p_recent):
    required = [
        "rep",
        "risk_defer",
        "risk_initiate",
        "risk_difference",
        "risk_ratio",
    ]
    old = load_checkpoint(FIXED_CKPT, required)
    completed = len(old)

    if completed >= FIXED_BOOTSTRAP_TARGET:
        return old.iloc[:FIXED_BOOTSTRAP_TARGET].copy()

    rng = np.random.default_rng(SEED + 3200)
    nclusters = len(prep["unique_clusters"])

    for _ in range(completed):
        rng.integers(0, nclusters, size=nclusters)

    rec = old.to_dict("records")
    success = completed

    while success < FIXED_BOOTSTRAP_TARGET:
        draw = rng.integers(0, nclusters, size=nclusters)
        counts = np.bincount(draw, minlength=nclusters).astype(float)
        row_mult = counts[prep["inverse_cluster"]]

        est = e31.estimate_effect(
            prep,
            q,
            p_hour,
            p_recent,
            row_mult,
        )

        success += 1
        rec.append({
            "rep": success,
            "risk_defer": est["risk_defer"],
            "risk_initiate": est["risk_initiate"],
            "risk_difference": est["risk_difference"],
            "risk_ratio": est["risk_ratio"],
        })

        if (
            success % CHECKPOINT_EVERY == 0
            or success == FIXED_BOOTSTRAP_TARGET
        ):
            save_checkpoint(FIXED_CKPT, pd.DataFrame(rec))
            print(
                f"fixed-nuisance checkpoint: "
                f"{success}/{FIXED_BOOTSTRAP_TARGET}"
            )

    return pd.DataFrame(rec).iloc[:FIXED_BOOTSTRAP_TARGET].copy()


def fit_ridge_nuisance(e31, prep, cluster_counts=None):
    n = len(prep["trials"])

    if cluster_counts is None:
        row_mult = np.ones(n, float)
    else:
        row_mult = cluster_counts[prep["inverse_cluster"]]

    mask_base = row_mult > 0

    m_base = LogisticRegression(
        C=RIDGE_C,
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    m_base.fit(
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
        penalty="l2",
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

    m_recent = LogisticRegression(
        C=RIDGE_C,
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    m_recent.fit(
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


def ridge_diagnostics(prep, q, est):
    recent = est["recent"]
    recent_ipcw = 1.0 / est["p_recent_trial"][recent]
    return {
        "baseline_q_ess": ess(q),
        "recent_ipcw_p95": float(np.quantile(recent_ipcw, 0.95)),
        "recent_ipcw_p99": float(np.quantile(recent_ipcw, 0.99)),
        "recent_ipcw_max": float(np.max(recent_ipcw)),
        "defer_postgrace_weight_ess": ess(
            est["defer_post_weight"][prep["y6"] == 0]
        ),
        "initiate_postgrace_weight_ess": ess(
            est["initiate_post_weight"][prep["y6"] == 1]
        ),
        "cumulative_defer_ipcw_p99": float(
            np.quantile(est["cumulative_defer_ipcw"], 0.99)
        ),
        "cumulative_defer_ipcw_max": float(
            np.max(est["cumulative_defer_ipcw"])
        ),
    }


def ridge_bootstrap(e31, prep):
    required = [
        "rep",
        "risk_defer",
        "risk_initiate",
        "risk_difference",
        "risk_ratio",
        "baseline_q_ess",
        "recent_ipcw_p99",
        "recent_ipcw_max",
        "defer_postgrace_weight_ess",
        "initiate_postgrace_weight_ess",
        "cumulative_defer_ipcw_max",
    ]
    old = load_checkpoint(RIDGE_CKPT, required)
    completed = len(old)

    if completed >= RIDGE_BOOTSTRAP_TARGET:
        return old.iloc[:RIDGE_BOOTSTRAP_TARGET].copy()

    rng = np.random.default_rng(SEED + 3300)
    nclusters = len(prep["unique_clusters"])

    for _ in range(completed):
        rng.integers(0, nclusters, size=nclusters)

    rec = old.to_dict("records")
    success = completed
    attempts = completed

    while success < RIDGE_BOOTSTRAP_TARGET:
        attempts += 1
        draw = rng.integers(0, nclusters, size=nclusters)
        counts = np.bincount(draw, minlength=nclusters).astype(float)

        row_mult = counts[prep["inverse_cluster"]]

        # Treatment-process support safeguard only.
        if prep["y6"][row_mult > 0].sum() < 30:
            continue
        recent_mult = counts[prep["recent_cluster"]]
        if prep["y_recent"][recent_mult > 0].sum() < 20:
            continue

        try:
            row_mult, e6, q, p_hour, p_recent = fit_ridge_nuisance(
                e31,
                prep,
                cluster_counts=counts,
            )
            est = e31.estimate_effect(
                prep,
                q,
                p_hour,
                p_recent,
                row_mult,
            )
            d = ridge_diagnostics(prep, q, est)

            success += 1
            rec.append({
                "rep": success,
                "risk_defer": est["risk_defer"],
                "risk_initiate": est["risk_initiate"],
                "risk_difference": est["risk_difference"],
                "risk_ratio": est["risk_ratio"],
                "baseline_q_ess": d["baseline_q_ess"],
                "recent_ipcw_p99": d["recent_ipcw_p99"],
                "recent_ipcw_max": d["recent_ipcw_max"],
                "defer_postgrace_weight_ess": d[
                    "defer_postgrace_weight_ess"
                ],
                "initiate_postgrace_weight_ess": d[
                    "initiate_postgrace_weight_ess"
                ],
                "cumulative_defer_ipcw_max": d[
                    "cumulative_defer_ipcw_max"
                ],
            })

        except Exception as exc:
            print(
                f"[WARN] ridge bootstrap attempt {attempts} failed "
                f"({type(exc).__name__}); continuing"
            )
            continue

        if (
            success % CHECKPOINT_EVERY == 0
            or success == RIDGE_BOOTSTRAP_TARGET
        ):
            save_checkpoint(RIDGE_CKPT, pd.DataFrame(rec))
            print(
                f"ridge checkpoint: "
                f"{success}/{RIDGE_BOOTSTRAP_TARGET}"
            )

    return pd.DataFrame(rec).iloc[:RIDGE_BOOTSTRAP_TARGET].copy()


def summarize_effect(label, point, boot, role):
    rd_ci = ci95(boot["risk_difference"])
    rr_ci = ci95(boot["risk_ratio"])
    r0_ci = ci95(boot["risk_defer"])
    r1_ci = ci95(boot["risk_initiate"])

    return {
        "analysis": label,
        "role": role,
        "risk_defer_point": point["risk_defer"],
        "risk_defer_ci_low": r0_ci[0],
        "risk_defer_ci_high": r0_ci[1],
        "risk_initiate_point": point["risk_initiate"],
        "risk_initiate_ci_low": r1_ci[0],
        "risk_initiate_ci_high": r1_ci[1],
        "risk_difference_point": point["risk_difference"],
        "rd_ci_low": rd_ci[0],
        "rd_ci_high": rd_ci[1],
        "risk_ratio_point": point["risk_ratio"],
        "rr_ci_low": rr_ci[0],
        "rr_ci_high": rr_ci[1],
        "bootstrap_n": len(boot),
    }


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 118)
    print("E3.2 — external inference-stability audit")
    print("=" * 118)
    print(
        "Frozen E3 point estimate is NOT changed. "
        "All sensitivity methods are reported side-by-side."
    )

    e31 = load_module(E31_PATH, "e31")
    e21 = load_module(E21_PATH, "e21")

    original_boot, original_audit = original_bootstrap_audit()

    trials, panel, prep = prepare(e31, e21)

    # Frozen primary point.
    row_mult, e6, q, p_hour, p_recent, frozen = frozen_point(
        e31,
        prep,
    )
    verify_primary_point(frozen)

    # A) Fixed-nuisance cluster bootstrap.
    fixed_boot = fixed_nuisance_bootstrap(
        e31,
        prep,
        q,
        p_hour,
        p_recent,
    )
    fixed_boot.to_csv(
        OUT_FIXED,
        index=False,
        encoding="utf-8-sig",
    )

    # B) Ridge sensitivity point + bootstrap.
    (
        ridge_row_mult,
        ridge_e6,
        ridge_q,
        ridge_p_hour,
        ridge_p_recent,
    ) = fit_ridge_nuisance(
        e31,
        prep,
        cluster_counts=None,
    )

    ridge_point = e31.estimate_effect(
        prep,
        ridge_q,
        ridge_p_hour,
        ridge_p_recent,
        ridge_row_mult,
    )

    ridge_point_diag = ridge_diagnostics(
        prep,
        ridge_q,
        ridge_point,
    )

    ridge_boot = ridge_bootstrap(
        e31,
        prep,
    )
    ridge_boot.to_csv(
        OUT_RIDGE_DIAG,
        index=False,
        encoding="utf-8-sig",
    )

    # Original percentile CI kept exactly as observed.
    original_point = {
        "risk_defer": frozen["risk_defer"],
        "risk_initiate": frozen["risk_initiate"],
        "risk_difference": frozen["risk_difference"],
        "risk_ratio": frozen["risk_ratio"],
    }

    comparison = pd.DataFrame([
        summarize_effect(
            "E3_primary_unpenalized_refit_bootstrap",
            original_point,
            original_boot,
            "frozen_primary_but_bootstrap_inferentially_unstable",
        ),
        summarize_effect(
            "E32_fixed_nuisance_cluster_bootstrap",
            original_point,
            fixed_boot,
            "conditional_sampling_sensitivity_not_primary",
        ),
        summarize_effect(
            "E32_ridge_C1_full_sensitivity",
            ridge_point,
            ridge_boot,
            "posthoc_separation_control_sensitivity_not_primary",
        ),
    ])
    comparison.to_csv(
        OUT_COMPARE,
        index=False,
        encoding="utf-8-sig",
    )

    rd_ci = ci95(ridge_boot["risk_difference"])
    rr_ci = ci95(ridge_boot["risk_ratio"])

    ridge_effect = pd.DataFrame([{
        "analysis": "E32_ridge_C1_full_sensitivity",
        "baseline_model": "L2 logistic C=1",
        "recent_model": "L2 logistic C=1",
        "defer_model": "L2 logistic C=0.5 frozen from E2",
        "risk_defer": ridge_point["risk_defer"],
        "risk_initiate": ridge_point["risk_initiate"],
        "risk_difference": ridge_point["risk_difference"],
        "rd_ci_low": rd_ci[0],
        "rd_ci_high": rd_ci[1],
        "risk_ratio": ridge_point["risk_ratio"],
        "rr_ci_low": rr_ci[0],
        "rr_ci_high": rr_ci[1],
        **ridge_point_diag,
    }])
    ridge_effect.to_csv(
        OUT_RIDGE,
        index=False,
        encoding="utf-8-sig",
    )

    # Pure inferential-stability diagnostics; no replicate is removed.
    original_boundary_pct = 100 * (
        (
            (original_boot["risk_defer"] < RISK_LOW_FLAG)
            | (original_boot["risk_initiate"] < RISK_LOW_FLAG)
            | (original_boot["risk_initiate"] > RISK_HIGH_FLAG)
            | (original_boot["risk_ratio"] > RR_HIGH_FLAG)
        ).mean()
    )

    ridge_boundary_pct = 100 * (
        (
            (ridge_boot["risk_defer"] < RISK_LOW_FLAG)
            | (ridge_boot["risk_initiate"] < RISK_LOW_FLAG)
            | (ridge_boot["risk_initiate"] > RISK_HIGH_FLAG)
            | (ridge_boot["risk_ratio"] > RR_HIGH_FLAG)
        ).mean()
    )

    ridge_extreme_ipcw_pct = 100 * (
        ridge_boot["recent_ipcw_max"] > RECENT_IPCW_HIGH_FLAG
    ).mean()

    flag = pd.DataFrame([{
        "frozen_primary_point_rd": frozen["risk_difference"],
        "frozen_primary_point_rr": frozen["risk_ratio"],
        "original_refit_bootstrap_boundary_or_rr_gt5_pct": original_boundary_pct,
        "ridge_bootstrap_boundary_or_rr_gt5_pct": ridge_boundary_pct,
        "ridge_bootstrap_recent_ipcw_max_gt1000_pct": ridge_extreme_ipcw_pct,
        "ridge_point_rd": ridge_point["risk_difference"],
        "ridge_point_rr": ridge_point["risk_ratio"],
        "ridge_vs_primary_abs_rd_difference": abs(
            ridge_point["risk_difference"] - frozen["risk_difference"]
        ),
        "ridge_vs_primary_abs_rr_difference": abs(
            ridge_point["risk_ratio"] - frozen["risk_ratio"]
        ),
        "interpretation_flag": (
            "PRIMARY_POINT_DIRECTIONALLY_INTERPRETABLE_BUT_ORIGINAL_BOOTSTRAP_UNSTABLE__"
            "REPORT_FIXED_NUISANCE_AND_RIDGE_AS_TRANSPARENT_SENSITIVITIES"
        ),
        "scientific_rule": (
            "Do not replace the frozen E3 point estimate or select the narrowest CI. "
            "Report all inference approaches and describe the original refitted "
            "bootstrap instability as a positivity/separation limitation."
        ),
    }])
    flag.to_csv(
        OUT_RECOMMEND,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nInference comparison:")
    print(comparison.to_string(index=False))

    print("\nInterpretation flag:")
    print(flag.to_string(index=False))

    print("\nUpload these six files:")
    for p in [
        OUT_PRIMARY_AUDIT,
        OUT_FIXED,
        OUT_RIDGE,
        OUT_RIDGE_DIAG,
        OUT_COMPARE,
        OUT_RECOMMEND,
    ]:
        print(" ", p)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise

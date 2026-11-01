from __future__ import annotations

"""
final causal-analysis stage V7 — PURE clone-censor-weight (CCW) diagnostic

IMPORTANT
---------
V7 is a METHOD-CORRECTION / DIAGNOSTIC run. No bootstrap yet.

Why:
- In a cloned design, every eligible patient is assigned to BOTH strategies at time zero.
- Baseline balance between strategy arms is therefore exact by construction.
- The inverse-probability correction needed after cloning is for ARTIFICIAL CENSORING.
- If we want an overlap/ATO target population, h(X)=e(X)[1-e(X)] is used as a
  TARGET-POPULATION sampling weight on BOTH clones, not converted into ordinary
  treatment-IPTW weights h/e and h/(1-e).

V7 therefore estimates:
1) ATO target: q(X)=e2(X)[1-e2(X)] multiplied by pure IPCW.
2) ATE/full eligible target: q(X)=1 multiplied by pure IPCW (sensitivity).

Grace-period censoring:
- DEFER >=2h:
    clones starting during hour 1 or hour 2 are artificially censored.
    Weight survivors by inverse probability of NOT starting each hour.
- INITIATE within 2h:
    early initiators [0,1h) keep IPCW=1.
    recent initiators [1,2h) receive 1/P(start in hour 2 | history)
    to represent clones censored at +2h for failing to initiate.

This matches the initiation-window CCW tutorial logic more directly than V6,
which used stabilized p_num/p_den recent-initiator factors with mean ~1.
"""

from pathlib import Path
import importlib.util
import traceback
import warnings
import math

import numpy as np
import pandas as pd
import config

PROJECT_ROOT = Path(config.PROJECT_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
HERE = Path(__file__).resolve().parent

V5_PATHS = [
    HERE / "08_two_hour_overlap_bootstrap_diagnostic.py",
    PROJECT_ROOT / "08_two_hour_overlap_bootstrap_diagnostic.py",
]
V6_PATHS = [
    HERE / "09_two_hour_hourly_ipcw_diagnostic.py",
    PROJECT_ROOT / "09_two_hour_hourly_ipcw_diagnostic.py",
]

OUT_EFFECT = OUTPUT_DIR / "16V7_FINAL_PURE_CCW_EFFECT_DIAGNOSTIC.csv"
OUT_WEIGHT = OUTPUT_DIR / "16V7B_FINAL_PURE_CCW_WEIGHT_DIAGNOSTICS.csv"
OUT_BAL = OUTPUT_DIR / "16V7C_FINAL_PURE_CCW_CENSORING_BALANCE.csv"
OUT_SUM = OUTPUT_DIR / "16V7D_FINAL_PURE_CCW_BALANCE_SUMMARY.csv"
OUT_COMPARE = OUTPUT_DIR / "16V7E_FINAL_V5_V6_V7_COMPARISON.csv"

PS_CLIP = 1e-5
SEED = 20261028


def import_module_from(paths, name):
    for p in paths:
        if p.exists():
            spec = importlib.util.spec_from_file_location(name, p)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Cannot find dependency for {name}: {paths}")


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
    den = math.sqrt((v1 + v0) / 2.0)
    if den <= 0:
        return 0.0
    return (m1 - m0) / den


def balance_features(v5, df):
    out = {}

    numeric = []
    for c in (
        v5.CURRENT_NUM
        + v5.CURRENT_BIN
        + v5.BASELINE_NUM
        + v5.BASELINE_BIN
        + ["trial_hour", "decision_hour"]
    ):
        if c in df.columns and c not in numeric:
            numeric.append(c)

    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce")
        out[c] = x
        if x.isna().any():
            out[c + "__missing"] = x.isna().astype(float)

    for c in [x for x in v5.BASELINE_CAT if x in df.columns]:
        s = df[c].astype("string").fillna("Missing")
        for level in sorted(s.unique().tolist()):
            out[f"{c}=={level}"] = (s == level).astype(float)

    return out


def fit_models(v5, v6, prep):
    # e2 is used ONLY to define optional ATO target q=e2(1-e2).
    m_base = v5._fit_unpenalized_logistic(prep["X_base"], prep["y2"])
    e2 = np.clip(
        m_base.predict_proba(prep["X_base"])[:, 1],
        PS_CLIP, 1 - PS_CLIP
    )

    # Pure censoring model: denominator probability of starting in each next hour.
    m_start = v6.fit_ridge_logistic(
        prep["X_den"], prep["y1"], sample_weight=None, C=0.5
    )
    pstart = np.clip(
        m_start.predict_proba(prep["X_den"])[:, 1],
        PS_CLIP, 1 - PS_CLIP
    )

    return e2, pstart


def probs_by_trial(prep, pstart_long):
    n = len(prep["trials"])
    p = np.full((n, 2), np.nan, float)

    idx = prep["long"]["trial_idx"].to_numpy(int)
    interval = prep["long"]["interval_index"].to_numpy(int)
    p[idx, interval] = pstart_long
    return p


def pure_ccw_estimate(v6, prep, e2, pstart_long, target="ATO"):
    """
    Pure CCW:
      baseline target q stays a target-sampling weight.
      only artificial censoring receives inverse-probability correction.
    """
    trials = prep["trials"]
    geom = prep["geom"]
    y2 = prep["y2"]
    p = probs_by_trial(prep, pstart_long)

    p0 = np.clip(p[:, 0], PS_CLIP, 1 - PS_CLIP)
    p1 = np.where(np.isfinite(p[:, 1]), p[:, 1], 0.5)
    p1 = np.clip(p1, PS_CLIP, 1 - PS_CLIP)

    if target == "ATO":
        q = e2 * (1 - e2)
    elif target == "ATE":
        q = np.ones(len(trials), float)
    else:
        raise ValueError(target)

    trial_hour = trials["trial_hour"].to_numpy(float)
    init_abs = pd.to_numeric(trials["any_init_h"], errors="coerce").to_numpy(float)
    init_rel = init_abs - trial_hour

    early = (
        (y2 == 1)
        & np.isfinite(init_rel)
        & (init_rel >= 0)
        & (init_rel < 1)
    )
    late = (
        (y2 == 1)
        & np.isfinite(init_rel)
        & (init_rel >= 1)
        & (init_rel < 2)
    )
    defer = y2 == 0

    # DEFER: inverse probability of remaining uncensored (not starting).
    ipcw_no0 = 1.0 / (1.0 - p0)
    ipcw_no1 = 1.0 / (1.0 - p1)

    # INITIATE-by-2h: early initiators stay at 1; recent initiators represent
    # end-of-window nonstarters using inverse P(start during the final hour).
    ipcw_recent = 1.0 / p1

    w_pre0 = q
    w_defer_1 = q * ipcw_no0
    w_defer_post = q * ipcw_no0 * ipcw_no1

    w_init_1 = q
    w_init_post = np.zeros(len(trials), float)
    w_init_post[early] = q[early]
    w_init_post[late] = q[late] * ipcw_recent[late]

    s0 = v6.phase_survival(
        geom["end0"], geom["event0"], w_pre0, 0.0, 1.0
    )
    s0 *= v6.phase_survival(
        geom["end0"], geom["event0"], w_defer_1, 1.0, 2.0
    )
    s0 *= v6.phase_survival(
        geom["end0"], geom["event0"], w_defer_post, 2.0, np.inf
    )

    s1 = v6.phase_survival(
        geom["end1"], geom["event1"], w_pre0, 0.0, 1.0
    )
    s1 *= v6.phase_survival(
        geom["end1"], geom["event1"], w_init_1, 1.0, 2.0
    )
    s1 *= v6.phase_survival(
        geom["end1"], geom["event1"], w_init_post, 2.0, np.inf
    )

    r0 = 1 - s0
    r1 = 1 - s1

    return {
        "target": target,
        "risk_defer": r0,
        "risk_initiate": r1,
        "risk_difference": r1 - r0,
        "risk_ratio": r1 / r0 if r0 > 0 else np.nan,
        "q": q,
        "p0": p0,
        "p1": p1,
        "ipcw_no0": ipcw_no0,
        "ipcw_no1": ipcw_no1,
        "ipcw_recent": ipcw_recent,
        "early": early,
        "late": late,
        "defer": defer,
        "w_defer_1": w_defer_1,
        "w_defer_post": w_defer_post,
        "w_init_post": w_init_post,
    }


def make_weight_diagnostics(population, est):
    q = est["q"]
    p0 = est["p0"]
    p1 = est["p1"]
    late = est["late"]
    defer = est["defer"]
    early = est["early"]

    groups = [
        ("target_q_all", q, np.ones(len(q), bool)),
        ("defer_ipcw_no_start_hour1", est["ipcw_no0"], np.ones(len(q), bool)),
        ("defer_ipcw_no_start_hour2", est["ipcw_no1"], defer),
        ("init_recent_unstabilized_ipcw_1_over_pstart", est["ipcw_recent"], late),
        ("defer_total_postgrace_weight", est["w_defer_post"], defer),
        ("init_total_postgrace_weight", est["w_init_post"], early | late),
    ]

    rows = []
    for label, arr, mask in groups:
        x = np.asarray(arr, float)[mask]
        x = x[np.isfinite(x) & (x > 0)]
        rows.append({
            "population": population,
            "target": est["target"],
            "diagnostic": label,
            "n": len(x),
            "mean": np.mean(x) if len(x) else np.nan,
            "p01": np.quantile(x, .01) if len(x) else np.nan,
            "p50": np.quantile(x, .50) if len(x) else np.nan,
            "p95": np.quantile(x, .95) if len(x) else np.nan,
            "p99": np.quantile(x, .99) if len(x) else np.nan,
            "max": np.max(x) if len(x) else np.nan,
            "sum": np.sum(x) if len(x) else np.nan,
            "ess": ess(x) if len(x) else np.nan,
        })

    # Key calibration / representation check for the initiation arm.
    rows.append({
        "population": population,
        "target": est["target"],
        "diagnostic": "recent_initiator_representation_check",
        "n": int(late.sum()),
        "mean": np.nan,
        "p01": np.nan,
        "p50": np.nan,
        "p95": np.nan,
        "p99": np.nan,
        "max": np.nan,
        "sum": float(np.sum(q[late] * est["ipcw_recent"][late])),
        "ess": ess(q[late] * est["ipcw_recent"][late]),
        "target_riskset_q_sum_at_plus1": np.nan,
    })

    return pd.DataFrame(rows)


def censor_balance(v5, prep, est, population):
    """
    Three balance audits:
    1) defer censoring in hour 1
    2) defer censoring in hour 2
    3) initiation arm recent-initiator representation at grace end
    """
    tr = prep["trials"].copy()
    long = prep["long"].copy()
    q = est["q"]

    init = pd.to_numeric(tr["any_init_h"], errors="coerce")
    death = pd.to_numeric(tr["death_hours_full"], errors="coerce")
    h0 = tr["trial_hour"].astype(float)

    alive1 = death.isna() | (death > h0 + 1)
    no_start1 = init.isna() | (init >= h0 + 1)

    alive2 = death.isna() | (death > h0 + 2)
    no_start2 = init.isna() | (init >= h0 + 2)

    # q is trial-level. Map interval-1 probabilities/weights.
    w1 = est["w_defer_1"]
    w2 = est["w_defer_post"]

    # interval-1 long rows map to trial indices.
    z1 = long[long["interval_index"] == 1].copy()
    ti1 = z1["trial_idx"].to_numpy(int)

    rows = []

    # A. defer hour 1: among those alive at +1, remaining untreated should
    # represent the whole +1 alive target risk set.
    target_mask = alive1.to_numpy()
    survivor_mask = (alive1 & no_start1).to_numpy()

    feats0 = balance_features(v5, tr)
    for name, series in feats0.items():
        x = pd.to_numeric(series, errors="coerce").to_numpy(float)
        smd = weighted_smd(
            x[survivor_mask], w1[survivor_mask],
            x[target_mask], q[target_mask],
        )
        rows.append({
            "population": population,
            "target": est["target"],
            "mechanism": "defer_hour1_start_censoring",
            "variable": name,
            "smd_after_ipcw": smd,
            "abs_smd_after_ipcw": abs(smd) if pd.notna(smd) else np.nan,
        })

    # B. defer hour 2: at +1, among alive+untreated, those remaining no-start
    # through +2 should represent the +1 risk set.
    mask_target_z1 = np.ones(len(z1), bool)
    trial_indices_z1 = z1["trial_idx"].to_numpy(int)
    remain2 = (
        alive2.iloc[trial_indices_z1].to_numpy()
        & no_start2.iloc[trial_indices_z1].to_numpy()
    )

    feats1 = balance_features(v5, z1)
    for name, series in feats1.items():
        x = pd.to_numeric(series, errors="coerce").to_numpy(float)
        target_w = w1[trial_indices_z1]
        survivor_w = w2[trial_indices_z1]
        smd = weighted_smd(
            x[remain2], survivor_w[remain2],
            x[mask_target_z1], target_w[mask_target_z1],
        )
        rows.append({
            "population": population,
            "target": est["target"],
            "mechanism": "defer_hour2_start_censoring",
            "variable": name,
            "smd_after_ipcw": smd,
            "abs_smd_after_ipcw": abs(smd) if pd.notna(smd) else np.nan,
        })

    # C. initiation grace-end censoring: recent initiators [1,2h) should
    # represent the +1 alive/untreated target risk set.
    late_by_trial = est["late"]
    late_z1 = late_by_trial[trial_indices_z1]
    target_w_init = q[trial_indices_z1]
    late_w_init = (
        q[trial_indices_z1] * est["ipcw_recent"][trial_indices_z1]
    )

    for name, series in feats1.items():
        x = pd.to_numeric(series, errors="coerce").to_numpy(float)
        smd = weighted_smd(
            x[late_z1], late_w_init[late_z1],
            x, target_w_init,
        )
        rows.append({
            "population": population,
            "target": est["target"],
            "mechanism": "initiate_grace_end_recent_initiator",
            "variable": name,
            "smd_after_ipcw": smd,
            "abs_smd_after_ipcw": abs(smd) if pd.notna(smd) else np.nan,
        })

    bal = pd.DataFrame(rows)

    # Add the exact target-vs-representation weight sums for interpretation.
    target_plus1_q_sum = float(np.sum(q[trial_indices_z1]))
    recent_weighted_sum = float(
        np.sum(q[trial_indices_z1][late_z1] *
               est["ipcw_recent"][trial_indices_z1][late_z1])
    )

    return bal, target_plus1_q_sum, recent_weighted_sum


def run_population(v5, v6, trials, panel, population):
    print(f"\n=== final causal-analysis stage V7 PURE CCW: {population} ===")

    prep = v6.prepare_population(v5, trials, panel)
    e2, pstart = fit_models(v5, v6, prep)

    v7_ato = pure_ccw_estimate(v6, prep, e2, pstart, target="ATO")
    v7_ate = pure_ccw_estimate(v6, prep, e2, pstart, target="ATE")

    # Recompute V5 and V6 only for transparent comparison.
    v5_ref = v5.estimate_ato(
        {
            "trials": prep["trials"],
            "y": prep["y2"],
            "X": prep["X_base"],
            "unique_ids": prep["unique_ids"],
            "inverse_patient": prep["inverse_patient"],
            "geom": prep["geom"],
        },
        e2,
        mixed_v4=False,
    )

    # V6 requires numerator model.
    m_num = v6.fit_ridge_logistic(prep["X_num"], prep["y1"], C=10.0)
    pnum = np.clip(
        m_num.predict_proba(prep["X_num"])[:, 1],
        PS_CLIP, 1 - PS_CLIP
    )
    v6_ref = v6.estimate_v6(
        prep, e2, pstart, pnum, factor_cap=None
    )

    effects = []
    for label, est, role in [
        ("V7_pure_CCW_ATO", v7_ato, "candidate_final_CCW_estimand"),
        ("V7_pure_CCW_ATE", v7_ate, "full_eligible_population_sensitivity"),
    ]:
        effects.append({
            "population": population,
            "analysis": label,
            "role": role,
            "risk_defer": est["risk_defer"],
            "risk_initiate": est["risk_initiate"],
            "risk_difference": est["risk_difference"],
            "risk_ratio": est["risk_ratio"],
        })

    compare = pd.DataFrame([
        {
            "population": population,
            "analysis": "V5_ATO_treatment_weight_style",
            "risk_defer": v5_ref["risk_defer_2h"],
            "risk_initiate": v5_ref["risk_initiate_next2h"],
            "risk_difference": v5_ref["risk_difference"],
            "risk_ratio": v5_ref["risk_ratio"],
        },
        {
            "population": population,
            "analysis": "V6_stabilized_1h_IPCW_plus_treatment_weight_style",
            "risk_defer": v6_ref["risk_defer_2h"],
            "risk_initiate": v6_ref["risk_initiate_next2h"],
            "risk_difference": v6_ref["risk_difference"],
            "risk_ratio": v6_ref["risk_ratio"],
        },
        *effects,
    ])

    all_weights = []
    all_balance = []
    summaries = []

    for est in [v7_ato, v7_ate]:
        wd = make_weight_diagnostics(population, est)
        bal, target_sum, recent_sum = censor_balance(
            v5, prep, est, population
        )

        # Attach representation target to its diagnostic row.
        m = wd["diagnostic"] == "recent_initiator_representation_check"
        wd.loc[m, "target_riskset_q_sum_at_plus1"] = target_sum
        wd.loc[m, "representation_ratio_recent_weighted_to_target"] = (
            recent_sum / target_sum if target_sum > 0 else np.nan
        )

        all_weights.append(wd)
        all_balance.append(bal)

        for mech in bal["mechanism"].unique():
            b = bal[bal["mechanism"] == mech]
            summaries.append({
                "population": population,
                "target": est["target"],
                "mechanism": mech,
                "variables_n": len(b),
                "max_abs_smd": b["abs_smd_after_ipcw"].max(),
                "n_abs_smd_gt_0_10": int(
                    (b["abs_smd_after_ipcw"] > .10).sum()
                ),
                "n_abs_smd_gt_0_20": int(
                    (b["abs_smd_after_ipcw"] > .20).sum()
                ),
                "target_plus1_weight_sum": target_sum
                    if mech == "initiate_grace_end_recent_initiator" else np.nan,
                "recent_initiator_weighted_sum": recent_sum
                    if mech == "initiate_grace_end_recent_initiator" else np.nan,
                "representation_ratio": (
                    recent_sum / target_sum
                    if mech == "initiate_grace_end_recent_initiator"
                    and target_sum > 0
                    else np.nan
                ),
            })

    print(compare.to_string(index=False))

    print("\nCensoring-balance summary:")
    print(pd.DataFrame(summaries).to_string(index=False))

    return (
        pd.DataFrame(effects),
        pd.concat(all_weights, ignore_index=True),
        pd.concat(all_balance, ignore_index=True),
        pd.DataFrame(summaries),
        compare,
    )


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    v5 = import_module_from(V5_PATHS, "stage10_v5_for_v7")
    v6 = import_module_from(V6_PATHS, "stage10_v6_for_v7")

    print("=" * 110)
    print("final causal-analysis stage V7 — PURE CCW + unstabilized artificial-censoring IPCW diagnostic")
    print("=" * 110)
    print(
        "NO BOOTSTRAP. First verify censoring-weight calibration and balance.\n"
        "V7 does not divide the baseline target q(X) by treatment propensity after cloning."
    )

    try:
        panel = v5.load_panel()
        broad_trials = v5.build_sequential_trials(panel)

        stage8b = v5.import_stage8b()
        flags = stage8b.build_clinical_flags()
        cape_ids = set(flags.loc[flags["cape_cod_like"] == 1, "stay_id"])
        cape_panel = panel[panel["stay_id"].isin(cape_ids)].copy()
        cape_trials = v5.build_sequential_trials(cape_panel)

        b = run_population(
            v5, v6, broad_trials, panel, "broad_primary"
        )
        c = run_population(
            v5, v6, cape_trials, cape_panel, "CAPE_COD_like"
        )

        pd.concat([b[0], c[0]], ignore_index=True).to_csv(
            OUT_EFFECT, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[1], c[1]], ignore_index=True).to_csv(
            OUT_WEIGHT, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[2], c[2]], ignore_index=True).to_csv(
            OUT_BAL, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[3], c[3]], ignore_index=True).to_csv(
            OUT_SUM, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[4], c[4]], ignore_index=True).to_csv(
            OUT_COMPARE, index=False, encoding="utf-8-sig"
        )

        print("\n" + "=" * 110)
        print("STAGE 10 V7 DIAGNOSTIC COMPLETE")
        print("=" * 110)
        print("\nUpload these five CSVs:")
        for p in [OUT_EFFECT, OUT_WEIGHT, OUT_BAL, OUT_SUM, OUT_COMPARE]:
            print(" ", p)

        print(
            "\nDecision rule:\n"
            "1) recent-initiator representation ratio should be reasonably near 1;\n"
            "2) censoring-balance |SMD| should be mostly <0.10;\n"
            "3) only then build the final cluster bootstrap."
        )

    except Exception:
        print("\n[FAILED] Copy the complete traceback to ChatGPT:\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

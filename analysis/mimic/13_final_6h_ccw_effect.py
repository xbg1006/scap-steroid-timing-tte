from __future__ import annotations

"""
final causal-analysis stage frozen final 6-hour analysis FINAL — Frozen 6-hour pure CCW outcome analysis

THIS SCRIPT FREEZES THE DESIGN SELECTED OUTCOME-BLIND IN V9/frozen 6-hour feasibility analysis.

Primary population:
    Broad severe CAP only.

Primary strategy contrast:
    initiate systemic corticosteroid within the next 6 hours
    versus
    defer systemic corticosteroid for at least 6 hours.

Sequential trial starts:
    0, 2, 4, ..., 18 hours after respiratory t0.

Primary target population:
    overlap / clinical-equipoise population
    q(X) = e6(X) * [1 - e6(X)]

Frozen nuisance models:
    baseline e6 model:
        unpenalized logistic regression

    DEFER artificial-censoring model:
        pooled 1-hour ridge logistic, C=0.5,
        updated hourly over the 6-hour grace period

    INITIATE grace-end model:
        unpenalized logistic among patients alive/untreated at +3h,
        predicting initiation during [3h,6h)

    early initiators [0h,3h):
        post-grace weight = q(X)

    recent initiators [3h,6h):
        post-grace weight = q(X) / P(recent initiation | history at +3h)

Outcome:
    28-day all-cause mortality from each sequential-trial baseline.

Inference:
    patient-level cluster bootstrap.
    The SAME patient can enter multiple sequential trials, so the bootstrap
    resamples patients, not person-trials.

Scientific guardrail:
    This script aborts unless the saved V9 Broad recommendation says:
        PROCEED_TO_6H_FINAL_PURE_CCW_OUTCOME_ANALYSIS

No treatment-window/model selection is performed in frozen final 6-hour analysis.
No CAPE outcome analysis is performed because CAPE failed the V9 support gate.
"""

from pathlib import Path
import importlib.util
import traceback
import warnings

import numpy as np
import pandas as pd
import config

PROJECT_ROOT = Path(config.PROJECT_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
HERE = Path(__file__).resolve().parent

V91_PATHS = [
    HERE / "12_six_hour_empirical_support.py",
    PROJECT_ROOT / "12_six_hour_empirical_support.py",
]

V9_RECOMMENDATION = OUTPUT_DIR / "16V9E_FINAL_6H_RECOMMENDATION.csv"

OUT_EFFECT = OUTPUT_DIR / "final_effect_estimates.csv"
OUT_BOOT = OUTPUT_DIR / "bootstrap_summary.csv"
OUT_DIAG = OUTPUT_DIR / "weight_support_diagnostics.csv"
OUT_GRACE = OUTPUT_DIR / "grace_period_event_audit.csv"
OUT_HOUR = OUTPUT_DIR / "sequential_trial_counts.csv"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = CHECKPOINT_DIR / "stage10_v10_broad_6h_pure_ccw_cluster_bootstrap.csv"

SEED = 20261028
BOOTSTRAP_TARGET = 300
CHECKPOINT_EVERY = 10

FOLLOWUP_H = 28 * 24.0
WINDOW_H = 6.0
RECENT_START_H = 3.0
RECENT_END_H = 6.0
PS_CLIP = 1e-5

EXPECTED_V9_DECISION = "PROCEED_TO_6H_FINAL_PURE_CCW_OUTCOME_ANALYSIS"


def import_v91():
    for p in V91_PATHS:
        if p.exists():
            spec = importlib.util.spec_from_file_location("stage10_v91_for_v10", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Cannot find 12_six_hour_empirical_support.py.\n"
        "Extract the complete frozen final 6-hour analysis ZIP into the project root."
    )


def enforce_v9_gate():
    if not V9_RECOMMENDATION.exists():
        raise FileNotFoundError(
            f"Missing frozen feasibility decision:\n{V9_RECOMMENDATION}\n"
            "Run frozen 6-hour feasibility analysis first. frozen final 6-hour analysis intentionally refuses to bypass the outcome-blind gate."
        )

    x = pd.read_csv(V9_RECOMMENDATION)
    b = x[x["population"] == "broad_primary"].copy()
    if len(b) != 1:
        raise RuntimeError(
            "V9 recommendation must contain exactly one broad_primary row."
        )

    decision = str(b.iloc[0]["decision"])
    selected = str(b.iloc[0]["recent_selected_candidate"])

    if decision != EXPECTED_V9_DECISION:
        raise RuntimeError(
            "frozen final 6-hour analysis ABORTED: Broad did not pass the frozen V9 support rule.\n"
            f"Observed decision: {decision}"
        )

    if selected != "recent3h_unpenalized":
        raise RuntimeError(
            "frozen final 6-hour analysis ABORTED: V9 selected model differs from frozen frozen final 6-hour analysis specification.\n"
            f"Observed: {selected}\n"
            "Expected: recent3h_unpenalized"
        )

    print(
        "[LOCK] V9 outcome-blind gate verified: "
        f"{decision}; selected recent model={selected}"
    )


def ess(w):
    x = np.asarray(w, float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    return (x.sum() ** 2) / np.sum(x ** 2)


def build_clone_geometry(trials: pd.DataFrame):
    h0 = trials["trial_hour"].to_numpy(float)
    init_abs = pd.to_numeric(
        trials["any_init_h"], errors="coerce"
    ).to_numpy(float)
    death_abs = pd.to_numeric(
        trials["death_hours_full"], errors="coerce"
    ).to_numpy(float)

    y6 = trials["actual_initiate_next6h"].astype(int).to_numpy()

    init_rel = init_abs - h0
    death_rel = death_abs - h0

    death_end = np.where(
        np.isfinite(death_rel),
        death_rel,
        np.inf,
    )

    horizon = np.full(len(trials), FOLLOWUP_H, float)

    # DEFER >=6h:
    # if observed initiation occurs inside the 6h grace period,
    # the defer clone is artificially censored at that exact initiation time.
    art0 = np.where(
        y6 == 1,
        init_rel,
        np.inf,
    )
    end0 = np.minimum.reduce([horizon, death_end, art0])

    event0 = (
        np.isfinite(death_rel)
        & (death_rel > 0)
        & (death_rel <= FOLLOWUP_H)
        & (death_rel <= art0)
    ).astype(int)
    end0 = np.where(event0 == 1, death_rel, end0)

    # INITIATE <=6h:
    # non-initiators are artificially censored at +6h.
    art1 = np.where(
        y6 == 1,
        np.inf,
        WINDOW_H,
    )
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
        "init_rel": init_rel,
        "death_rel": death_rel,
    }


def exact_weighted_km_piecewise(end, event, weight_for_time):
    """
    Exact event-time weighted KM.

    weight_for_time(t) returns a vector of weights for the full clone table
    at event time t. Artificially censored clones are automatically excluded
    because end < t.
    """
    times = np.sort(np.unique(end[event == 1]))
    surv = 1.0

    for t in times:
        risk = end >= t
        ev = (event == 1) & (end == t)

        w = np.asarray(weight_for_time(float(t)), float)

        yw = float(np.sum(w[risk]))
        dw = float(np.sum(w[ev]))

        if yw > 0:
            frac = min(max(dw / yw, 0.0), 1.0)
            surv *= 1.0 - frac

    return 1.0 - surv


def prepare_analysis(v91, panel, trials):
    """
    Freeze preprocessing maps and row mappings once.

    Bootstrap refits all nuisance models with patient multiplicity weights.
    The covariate transformation map is fixed from the original sample to avoid
    physically duplicating ~100k hourly rows in every bootstrap replicate.
    """
    trials = trials.reset_index(drop=True).copy()

    # Baseline e6 / ATO target model.
    base_prep = v91.make_preprocessor(
        trials,
        include_decision_time=False,
    )
    X_base = base_prep.fit_transform(trials)
    y6 = trials["actual_initiate_next6h"].astype(int).to_numpy()

    # Hourly DEFER artificial-censoring model.
    long = v91.build_hourly_long(trials, panel)
    defer_prep = v91.make_preprocessor(
        long,
        include_decision_time=True,
    )
    X_defer = defer_prep.fit_transform(long)
    y_hour = long["start_next1h"].astype(int).to_numpy()

    # Frozen recent [3h,6h) initiation model selected outcome-blind in V9.
    recent = v91.build_recent_riskset(trials, panel)
    recent_prep = v91.make_preprocessor(
        recent,
        include_decision_time=True,
    )
    X_recent = recent_prep.fit_transform(recent)
    y_recent = recent["recent_initiate_3_6h"].astype(int).to_numpy()

    unique_ids, inverse_patient = np.unique(
        trials["stay_id"].to_numpy(),
        return_inverse=True,
    )

    long_patient_idx = inverse_patient[
        long["trial_idx"].to_numpy(int)
    ]
    recent_patient_idx = inverse_patient[
        recent["trial_idx"].to_numpy(int)
    ]

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
        "unique_ids": unique_ids,
        "inverse_patient": inverse_patient,
        "long_patient_idx": long_patient_idx,
        "recent_patient_idx": recent_patient_idx,
        "geom": build_clone_geometry(trials),
    }


def fit_models(v91, prep, row_mult=None):
    n = len(prep["trials"])

    if row_mult is None:
        row_mult = np.ones(n, float)
    row_mult = np.asarray(row_mult, float)

    mask_base = row_mult > 0
    yb = prep["y6"][mask_base]

    # Frozen baseline model: unpenalized logistic.
    m_base = v91.fit_unpenalized(
        prep["X_base"][mask_base],
        yb,
    ) if np.all(row_mult[mask_base] == 1) else None

    if m_base is None:
        # sklearn fit_unpenalized helper in frozen 6-hour feasibility analysis does not accept sample_weight;
        # duplicate weights are handled by direct compatible model fitting below.
        attempts = [
            dict(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-9),
            dict(penalty="none", solver="lbfgs", max_iter=5000, tol=1e-9),
        ]
        last = None
        from sklearn.linear_model import LogisticRegression
        for kwargs in attempts:
            try:
                m_base = LogisticRegression(
                    fit_intercept=True,
                    random_state=SEED,
                    **kwargs,
                )
                m_base.fit(
                    prep["X_base"][mask_base],
                    yb,
                    sample_weight=row_mult[mask_base],
                )
                break
            except Exception as exc:
                last = exc
                m_base = None
        if m_base is None:
            raise RuntimeError(
                f"Bootstrap baseline unpenalized logistic failed: {last}"
            )

    e6 = np.clip(
        m_base.predict_proba(prep["X_base"])[:, 1],
        PS_CLIP,
        1 - PS_CLIP,
    )
    q = e6 * (1 - e6)

    # Frozen hourly defer model: pooled ridge C=0.5.
    long_mult = row_mult[prep["long_patient_idx"]]
    mask_long = long_mult > 0

    from sklearn.linear_model import LogisticRegression
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

    # Frozen recent model: unpenalized logistic.
    recent_mult = row_mult[prep["recent_patient_idx"]]
    mask_recent = recent_mult > 0

    yr = prep["y_recent"][mask_recent]
    attempts = [
        dict(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-9),
        dict(penalty="none", solver="lbfgs", max_iter=5000, tol=1e-9),
    ]

    m_recent = None
    last = None
    for kwargs in attempts:
        try:
            m_recent = LogisticRegression(
                fit_intercept=True,
                random_state=SEED,
                **kwargs,
            )
            m_recent.fit(
                prep["X_recent"][mask_recent],
                yr,
                sample_weight=recent_mult[mask_recent],
            )
            break
        except Exception as exc:
            last = exc
            m_recent = None

    if m_recent is None:
        raise RuntimeError(
            f"Recent unpenalized logistic failed: {last}"
        )

    p_recent = np.clip(
        m_recent.predict_proba(prep["X_recent"])[:, 1],
        PS_CLIP,
        1 - PS_CLIP,
    )

    return e6, q, p_hour, p_recent


def hourly_probability_matrix(prep, p_hour):
    n = len(prep["trials"])
    mat = np.full((n, 6), np.nan, float)

    ti = prep["long"]["trial_idx"].to_numpy(int)
    jj = prep["long"]["interval_index"].to_numpy(int)
    mat[ti, jj] = p_hour
    return mat


def recent_probability_by_trial(prep, p_recent):
    n = len(prep["trials"])
    out = np.full(n, np.nan, float)

    ti = prep["recent"]["trial_idx"].to_numpy(int)
    out[ti] = p_recent
    return out


def estimate_effect(prep, q, p_hour, p_recent, row_mult=None):
    """
    Frozen pure-CCW ATO estimator.

    DEFER:
      target q from time zero.
      At event time t in grace, use cumulative IPCW through the completed
      hourly censoring intervals before t.
      After +6h, use all six no-initiation IPCW factors.

    INITIATE:
      q throughout the grace period because both early and not-yet-initiated
      clones remain protocol-compatible until +6h.
      After +6h:
        early [0,3h): q
        recent [3,6h): q / P(recent initiation | history at +3h)
        no initiation by 6h: artificially censored at +6h

    Patient bootstrap multiplicity multiplies all clone weights.
    """
    trials = prep["trials"]
    geom = prep["geom"]
    y6 = prep["y6"]

    n = len(trials)
    if row_mult is None:
        freq = np.ones(n, float)
    else:
        freq = np.asarray(row_mult, float)

    pmat = hourly_probability_matrix(prep, p_hour)
    pmat = np.where(np.isfinite(pmat), pmat, 0.5)
    pmat = np.clip(pmat, PS_CLIP, 1 - PS_CLIP)

    inv_no = 1.0 / (1.0 - pmat)

    # cumulative no-start IPCW after each completed hour:
    # cumulative_after[:,j] = product factors for intervals 0..j.
    cumulative_after = np.cumprod(inv_no, axis=1)

    p_recent_trial = recent_probability_by_trial(prep, p_recent)
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

    init_rel = geom["init_rel"]
    early = (
        (y6 == 1)
        & np.isfinite(init_rel)
        & (init_rel >= 0)
        & (init_rel < RECENT_START_H)
    )
    recent = (
        (y6 == 1)
        & np.isfinite(init_rel)
        & (init_rel >= RECENT_START_H)
        & (init_rel < RECENT_END_H)
    )

    base_q = freq * q

    def defer_weight(t):
        # Completed hourly intervals strictly before time t.
        if t <= 1.0:
            return base_q
        if t <= 2.0:
            return base_q * cumulative_after[:, 0]
        if t <= 3.0:
            return base_q * cumulative_after[:, 1]
        if t <= 4.0:
            return base_q * cumulative_after[:, 2]
        if t <= 5.0:
            return base_q * cumulative_after[:, 3]
        if t <= 6.0:
            return base_q * cumulative_after[:, 4]
        return base_q * cumulative_after[:, 5]

    init_post = np.zeros(n, float)
    init_post[early] = base_q[early]
    init_post[recent] = (
        base_q[recent]
        / p_recent_trial[recent]
    )

    def initiate_weight(t):
        if t <= WINDOW_H:
            return base_q
        return init_post

    r0 = exact_weighted_km_piecewise(
        geom["end0"],
        geom["event0"],
        defer_weight,
    )
    r1 = exact_weighted_km_piecewise(
        geom["end1"],
        geom["event1"],
        initiate_weight,
    )

    return {
        "risk_defer_6h": r0,
        "risk_initiate_6h": r1,
        "risk_difference": r1 - r0,
        "risk_ratio": r1 / r0 if r0 > 0 else np.nan,
        "early": early,
        "recent": recent,
        "defer_post_weight": base_q * cumulative_after[:, 5],
        "init_post_weight": init_post,
        "p_recent_trial": p_recent_trial,
        "cumulative_no_start_ipcw": cumulative_after[:, 5],
    }


def checkpoint_load():
    cols = [
        "rep",
        "risk_defer_6h",
        "risk_initiate_6h",
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


def cluster_bootstrap(v91, prep, target, seed):
    old = checkpoint_load()
    completed = len(old)

    if completed >= target:
        print(
            f"[RESUME] checkpoint already has {completed}; "
            f"target={target}"
        )
        return old.iloc[:target].copy()

    if completed:
        print(
            f"[RESUME] loaded {completed}/{target} from "
            f"{CHECKPOINT.name}"
        )

    rng = np.random.default_rng(seed)
    n_patients = len(prep["unique_ids"])

    # Replay already-completed patient draws to preserve deterministic resume.
    for _ in range(completed):
        rng.integers(
            0,
            n_patients,
            size=n_patients,
        )

    records = old.to_dict("records")
    success = completed
    attempts = completed

    while success < target:
        attempts += 1

        draw = rng.integers(
            0,
            n_patients,
            size=n_patients,
        )
        patient_counts = np.bincount(
            draw,
            minlength=n_patients,
        ).astype(float)

        row_mult = patient_counts[
            prep["inverse_patient"]
        ]

        mask = row_mult > 0

        # Support safeguards only; no outcome-based replicate deletion.
        if (
            prep["y6"][mask].sum() < 25
            or prep["y_recent"][
                row_mult[prep["recent_patient_idx"]] > 0
            ].sum()
            < 15
        ):
            continue

        try:
            e6, q, p_hour, p_recent = fit_models(
                v91,
                prep,
                row_mult=row_mult,
            )

            est = estimate_effect(
                prep,
                q,
                p_hour,
                p_recent,
                row_mult=row_mult,
            )

            success += 1
            records.append(
                {
                    "rep": success,
                    "risk_defer_6h": est[
                        "risk_defer_6h"
                    ],
                    "risk_initiate_6h": est[
                        "risk_initiate_6h"
                    ],
                    "risk_difference": est[
                        "risk_difference"
                    ],
                    "risk_ratio": est[
                        "risk_ratio"
                    ],
                }
            )

        except Exception as exc:
            print(
                f"[WARN] bootstrap attempt {attempts} failed "
                f"({type(exc).__name__}); continuing"
            )
            continue

        if (
            success % CHECKPOINT_EVERY == 0
            or success == target
        ):
            cur = pd.DataFrame(records)
            checkpoint_save(cur)
            print(
                f"checkpoint saved: "
                f"{success}/{target} -> {CHECKPOINT.name}"
            )

    return pd.DataFrame(records).iloc[:target].copy()


def ci95(x):
    z = (
        pd.to_numeric(
            pd.Series(x),
            errors="coerce",
        )
        .dropna()
        .to_numpy(float)
    )
    if len(z) == 0:
        return np.nan, np.nan
    return tuple(
        np.quantile(
            z,
            [0.025, 0.975],
        )
    )


def grace_event_audit(prep):
    geom = prep["geom"]
    death_rel = geom["death_rel"]

    rows = []
    for upper in [1, 2, 3, 4, 5, 6]:
        m = (
            np.isfinite(death_rel)
            & (death_rel > 0)
            & (death_rel <= upper)
        )
        rows.append(
            {
                "grace_upper_h": upper,
                "death_person_trials_n": int(m.sum()),
                "death_unique_patients_n": int(
                    prep["trials"]
                    .loc[m, "stay_id"]
                    .nunique()
                ),
                "person_trial_pct": 100
                * float(m.sum())
                / len(prep["trials"]),
            }
        )

    return pd.DataFrame(rows)


def trial_hour_counts(prep):
    tr = prep["trials"]
    rows = []

    for h in sorted(tr["trial_hour"].unique()):
        x = tr[tr["trial_hour"] == h]
        rel = (
            pd.to_numeric(
                x["any_init_h"],
                errors="coerce",
            )
            - x["trial_hour"]
        )
        y = x["actual_initiate_next6h"].astype(int)

        rows.append(
            {
                "trial_hour": h,
                "eligible_person_trials_n": len(x),
                "initiate_within6h_n": int(y.sum()),
                "initiate_within6h_rate": float(
                    y.mean()
                ),
                "early_0_3h_n": int(
                    (
                        (y == 1)
                        & (rel >= 0)
                        & (rel < 3)
                    ).sum()
                ),
                "recent_3_6h_n": int(
                    (
                        (y == 1)
                        & (rel >= 3)
                        & (rel < 6)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 112)
    print(
        "final causal-analysis stage frozen final 6-hour analysis FINAL — Frozen 6h pure CCW "
        "+ patient-cluster bootstrap"
    )
    print("=" * 112)

    enforce_v9_gate()

    v91 = import_v91()

    panel = v91.load_panel()
    trials = v91.build_trials(panel)

    print(
        f"Broad frozen cohort: "
        f"patients={trials.stay_id.nunique()} | "
        f"person-trials={len(trials)} | "
        f"initiate<=6h="
        f"{int(trials.actual_initiate_next6h.sum())}"
    )

    prep = prepare_analysis(
        v91,
        panel,
        trials,
    )

    # Original-sample nuisance models and point estimate.
    e6, q, p_hour, p_recent = fit_models(
        v91,
        prep,
        row_mult=None,
    )

    point = estimate_effect(
        prep,
        q,
        p_hour,
        p_recent,
        row_mult=None,
    )

    print(
        "\nFrozen point estimate:\n"
        f"  risk defer >=6h     = "
        f"{point['risk_defer_6h']:.6f}\n"
        f"  risk initiate <=6h  = "
        f"{point['risk_initiate_6h']:.6f}\n"
        f"  RD                   = "
        f"{point['risk_difference']:.6f}\n"
        f"  RR                   = "
        f"{point['risk_ratio']:.6f}"
    )

    # Bootstrap.
    boot = cluster_bootstrap(
        v91,
        prep,
        BOOTSTRAP_TARGET,
        SEED,
    )

    ci0 = ci95(
        boot["risk_defer_6h"]
    )
    ci1 = ci95(
        boot["risk_initiate_6h"]
    )
    ci_rd = ci95(
        boot["risk_difference"]
    )
    ci_rr = ci95(
        boot["risk_ratio"]
    )

    effect = pd.DataFrame(
        [
            {
                "population": "broad_primary",
                "analysis_status": "FINAL_FROZEN_6H_CCW",
                "estimand": "ATO_overlap_population",
                "strategy_1": (
                    "initiate systemic corticosteroid within next 6h"
                ),
                "strategy_0": (
                    "defer systemic corticosteroid for at least 6h"
                ),
                "recent_window": "[3h,6h)",
                "outcome": "28-day all-cause mortality",
                "unique_patients_n": trials.stay_id.nunique(),
                "person_trials_n": len(trials),
                "initiate_within6h_person_trials_n": int(
                    prep["y6"].sum()
                ),
                "bootstrap_target_n": BOOTSTRAP_TARGET,
                "bootstrap_success_n": len(boot),
                "risk_defer_6h": point[
                    "risk_defer_6h"
                ],
                "risk_defer_ci_low": ci0[0],
                "risk_defer_ci_high": ci0[1],
                "risk_initiate_6h": point[
                    "risk_initiate_6h"
                ],
                "risk_initiate_ci_low": ci1[0],
                "risk_initiate_ci_high": ci1[1],
                "risk_difference": point[
                    "risk_difference"
                ],
                "rd_ci_low": ci_rd[0],
                "rd_ci_high": ci_rd[1],
                "risk_ratio": point[
                    "risk_ratio"
                ],
                "rr_ci_low": ci_rr[0],
                "rr_ci_high": ci_rr[1],
            }
        ]
    )

    bsum = pd.DataFrame(
        [
            {
                "population": "broad_primary",
                "bootstrap_success_n": len(boot),
                "rd_bootstrap_median": float(
                    boot["risk_difference"].median()
                ),
                "rd_bootstrap_p025": ci_rd[0],
                "rd_bootstrap_p975": ci_rd[1],
                "rr_bootstrap_median": float(
                    boot["risk_ratio"].median()
                ),
                "rr_bootstrap_p025": ci_rr[0],
                "rr_bootstrap_p975": ci_rr[1],
                "checkpoint_file": str(CHECKPOINT),
            }
        ]
    )

    recent_mask = point["recent"]
    early_mask = point["early"]

    diag = pd.DataFrame(
        [
            {
                "population": "broad_primary",
                "design_frozen_from": (
                    "frozen 6-hour feasibility analysis outcome-blind feasibility gate"
                ),
                "baseline_e6_model": (
                    "unpenalized logistic"
                ),
                "defer_censor_model": (
                    "pooled 1h ridge logistic C=0.5"
                ),
                "recent_model": (
                    "unpenalized logistic at +3h "
                    "for initiation during [3h,6h)"
                ),
                "baseline_q_sum": float(
                    np.sum(q)
                ),
                "baseline_q_ess": ess(q),
                "early_0_3h_person_trials_n": int(
                    early_mask.sum()
                ),
                "recent_3_6h_person_trials_n": int(
                    recent_mask.sum()
                ),
                "recent_ipcw_p95": float(
                    np.quantile(
                        1.0
                        / point["p_recent_trial"][
                            recent_mask
                        ],
                        .95,
                    )
                ),
                "recent_ipcw_p99": float(
                    np.quantile(
                        1.0
                        / point["p_recent_trial"][
                            recent_mask
                        ],
                        .99,
                    )
                ),
                "recent_ipcw_max": float(
                    np.max(
                        1.0
                        / point["p_recent_trial"][
                            recent_mask
                        ]
                    )
                ),
                "defer_postgrace_weight_ess": ess(
                    point["defer_post_weight"][
                        prep["y6"] == 0
                    ]
                ),
                "initiate_postgrace_weight_ess": ess(
                    point["init_post_weight"][
                        prep["y6"] == 1
                    ]
                ),
                "cumulative_defer_ipcw_p99": float(
                    np.quantile(
                        point[
                            "cumulative_no_start_ipcw"
                        ],
                        .99,
                    )
                ),
                "cumulative_defer_ipcw_max": float(
                    np.max(
                        point[
                            "cumulative_no_start_ipcw"
                        ]
                    )
                ),
                "note": (
                    "CAPE-COD-like outcome analysis intentionally omitted "
                    "because V9 support gate failed."
                ),
            }
        ]
    )

    grace = grace_event_audit(
        prep
    )
    hour = trial_hour_counts(
        prep
    )

    effect.to_csv(
        OUT_EFFECT,
        index=False,
        encoding="utf-8-sig",
    )
    bsum.to_csv(
        OUT_BOOT,
        index=False,
        encoding="utf-8-sig",
    )
    diag.to_csv(
        OUT_DIAG,
        index=False,
        encoding="utf-8-sig",
    )
    grace.to_csv(
        OUT_GRACE,
        index=False,
        encoding="utf-8-sig",
    )
    hour.to_csv(
        OUT_HOUR,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 112)
    print("STAGE 10 frozen final 6-hour analysis FINAL COMPLETE")
    print("=" * 112)
    print("\nUpload these five CSVs:")
    for p in [
        OUT_EFFECT,
        OUT_BOOT,
        OUT_DIAG,
        OUT_GRACE,
        OUT_HOUR,
    ]:
        print(" ", p)

    print(
        "\nThis is the frozen Broad 6h analysis. "
        "Do not change the treatment window or nuisance-model forms "
        "after seeing these mortality results."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise

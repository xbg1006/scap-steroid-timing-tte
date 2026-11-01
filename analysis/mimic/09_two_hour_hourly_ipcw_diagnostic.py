from __future__ import annotations

"""
final causal-analysis stage V6 — 1-hour interval-specific IPCW sensitivity analysis

PRIMARY final causal-analysis stage remains V5.
This script is a sensitivity analysis for the 2-hour grace period.

Strategy 1:
    initiate systemic corticosteroid during [trial t, trial t + 2h)

Strategy 0:
    defer systemic corticosteroid for at least 2h

WHY V6
------
V5 uses baseline ATO overlap weighting and exact clone-censor survival analysis.
V6 retains that target population but adds interval-specific inverse-probability
of artificial-censoring weights during the 2-hour grace period.

The 2-hour window is split into:
    interval 0: [0h, 1h)
    interval 1: [1h, 2h)

For the DEFER strategy:
    artificial censoring can happen in either hour when steroid starts.
    A pooled 1-hour logistic model estimates P(no initiation next hour | history).

For the INITIATE-WITHIN-2H strategy:
    early initiators in [0,1h) have IPCW factor 1.
    Among patients still alive/untreated at +1h, "recent initiators" in [1,2h)
    receive a stabilized inverse probability of initiating in the second hour
    so they can represent clones censored at +2h for failing to initiate.

This follows the logic recommended for initiation windows beginning at time zero:
early initiators are not used to stand in for patients censored at the end of the
window; recent initiators near the end of the grace period are.

The analysis is deliberately a SENSITIVITY analysis. It does not replace V5.
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
        "final causal-analysis stage V6 needs scikit-learn. Run:\n"
        "    pip install scikit-learn\n"
        f"Original error: {exc}"
    )

PROJECT_ROOT = Path(config.PROJECT_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

HERE = Path(__file__).resolve().parent
V5_CANDIDATES = [
    HERE / "08_two_hour_overlap_bootstrap_diagnostic.py",
    PROJECT_ROOT / "08_two_hour_overlap_bootstrap_diagnostic.py",
]

OUT_EFFECT = OUTPUT_DIR / "16V6_FINAL_1H_IPCW_EFFECT.csv"
OUT_WEIGHTS = OUTPUT_DIR / "16V6B_FINAL_1H_IPCW_WEIGHT_DIAGNOSTICS.csv"
OUT_COUNTS = OUTPUT_DIR / "16V6C_FINAL_1H_IPCW_INTERVAL_COUNTS.csv"
OUT_COMPARE = OUTPUT_DIR / "16V6D_FINAL_V5_V6_COMPARISON.csv"
OUT_BOOT = OUTPUT_DIR / "16V6E_FINAL_1H_IPCW_BOOTSTRAP_SUMMARY.csv"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261028
CHECKPOINT_EVERY = 10

# Sensitivity-analysis bootstrap. Increase later if V6 agrees with V5.
BROAD_BOOTSTRAP_TARGET = 120
CAPE_BOOTSTRAP_TARGET = 60

# Only used for a point-estimate robustness row, never as the primary V6 estimate.
FACTOR_CAP_SENSITIVITY = 10.0

PS_CLIP = 1e-5


def import_v5():
    for path in V5_CANDIDATES:
        if path.exists():
            spec = importlib.util.spec_from_file_location("stage10_v5_for_v6", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Cannot find 08_two_hour_overlap_bootstrap_diagnostic.py.\n"
        "Extract the full V6 ZIP into the project root; the V5 dependency is bundled."
    )


def make_ipcw_preprocessor(v5, df: pd.DataFrame, numerator: bool = False):
    time_cols = [
        "trial_hour",
        "trial_hour2",
        "interval_index",
        "decision_hour",
        "decision_hour2",
    ]

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
                    time_cols,
                )
            ],
            remainder="drop",
        )

    num = [
        c
        for c in (
            v5.BASELINE_NUM
            + v5.BASELINE_BIN
            + v5.CURRENT_NUM
            + v5.CURRENT_BIN
        )
        if c in df.columns
    ]
    cat = [c for c in v5.BASELINE_CAT if c in df.columns]

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
                num + time_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                        ),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )


def fit_ridge_logistic(X, y, sample_weight=None, C=0.5):
    model = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=2500,
        random_state=SEED,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def build_one_hour_person_period(v5, trials: pd.DataFrame, panel: pd.DataFrame):
    """
    One row per at-risk 1h grace-period interval.

    interval 0:
        always starts at the sequential-trial baseline.

    interval 1:
        included only if alive and untreated at +1h.

    Dynamic physiology for interval 1 comes from the Stage 7 hourly panel at h+1.
    """
    trials = trials.reset_index(drop=True).copy()
    trials["trial_idx"] = np.arange(len(trials), dtype=int)

    base_cols = [
        c
        for c in (
            ["trial_idx", "trial_uid", "stay_id", "trial_hour", "trial_hour2",
             "any_init_h", "death_hours_full"]
            + v5.BASELINE_NUM
            + v5.BASELINE_BIN
            + v5.BASELINE_CAT
        )
        if c in trials.columns
    ]

    current_cols = [
        c for c in (v5.CURRENT_NUM + v5.CURRENT_BIN) if c in panel.columns
    ]

    # Interval 0: physiology is already attached to the trial baseline.
    z0 = trials[base_cols + current_cols].copy()
    z0["interval_index"] = 0
    z0["decision_hour"] = z0["trial_hour"].astype(float)
    z0["decision_hour2"] = z0["decision_hour"] ** 2

    # Interval 1: alive and untreated at +1h.
    init = pd.to_numeric(trials["any_init_h"], errors="coerce")
    death = pd.to_numeric(trials["death_hours_full"], errors="coerce")
    at_risk1 = (
        (death.isna() | (death > trials["trial_hour"] + 1))
        & (init.isna() | (init >= trials["trial_hour"] + 1))
    )

    z1_base = trials.loc[at_risk1, base_cols].copy()
    z1_base["interval_index"] = 1
    z1_base["decision_hour"] = z1_base["trial_hour"].astype(float) + 1.0
    z1_base["decision_hour2"] = z1_base["decision_hour"] ** 2

    # Pull updated physiology from exact h+1 panel row.
    dyn = panel[["stay_id", "hour"] + current_cols].copy()
    dyn = dyn.rename(columns={"hour": "decision_hour"})
    z1 = z1_base.merge(
        dyn,
        on=["stay_id", "decision_hour"],
        how="left",
        validate="many_to_one",
    )

    long = pd.concat([z0, z1], ignore_index=True, sort=False)

    init_long = pd.to_numeric(long["any_init_h"], errors="coerce")
    death_long = pd.to_numeric(long["death_hours_full"], errors="coerce")

    long["start_next1h"] = (
        init_long.notna()
        & (init_long >= long["decision_hour"])
        & (init_long < long["decision_hour"] + 1.0)
        & (death_long.isna() | (init_long <= death_long))
    ).astype(int)

    return long.sort_values(
        ["trial_idx", "interval_index"]
    ).reset_index(drop=True)


def prepare_population(v5, trials: pd.DataFrame, panel: pd.DataFrame):
    trials = trials.reset_index(drop=True).copy()

    # Baseline 2h treatment PS for the ATO target.
    y2 = trials["actual_initiate_next2h"].astype(int).to_numpy()
    prep_base = v5.make_preprocessor(trials, numerator=False)
    X_base = prep_base.fit_transform(trials)

    # 1h artificial-censoring model.
    long = build_one_hour_person_period(v5, trials, panel)
    prep_den = make_ipcw_preprocessor(v5, long, numerator=False)
    prep_num = make_ipcw_preprocessor(v5, long, numerator=True)
    X_den = prep_den.fit_transform(long)
    X_num = prep_num.fit_transform(long)
    y1 = long["start_next1h"].astype(int).to_numpy()

    unique_ids, inverse_patient = np.unique(
        trials["stay_id"].to_numpy(), return_inverse=True
    )
    long_patient_index = inverse_patient[long["trial_idx"].to_numpy(int)]

    return {
        "trials": trials,
        "long": long,
        "y2": y2,
        "y1": y1,
        "X_base": X_base,
        "X_den": X_den,
        "X_num": X_num,
        "unique_ids": unique_ids,
        "inverse_patient": inverse_patient,
        "long_patient_index": long_patient_index,
        "geom": v5.build_clone_geometry(trials),
    }


def fit_point_models(v5, prep):
    m_base = v5._fit_unpenalized_logistic(prep["X_base"], prep["y2"])
    ps2 = m_base.predict_proba(prep["X_base"])[:, 1]
    ps2 = np.clip(ps2, PS_CLIP, 1 - PS_CLIP)

    m_den = fit_ridge_logistic(prep["X_den"], prep["y1"], C=0.5)
    m_num = fit_ridge_logistic(prep["X_num"], prep["y1"], C=10.0)

    pden = np.clip(
        m_den.predict_proba(prep["X_den"])[:, 1], PS_CLIP, 1 - PS_CLIP
    )
    pnum = np.clip(
        m_num.predict_proba(prep["X_num"])[:, 1], PS_CLIP, 1 - PS_CLIP
    )

    return ps2, pden, pnum


def probabilities_by_trial(prep, pden_long, pnum_long):
    n = len(prep["trials"])
    pden = np.full((n, 2), np.nan, dtype=float)
    pnum = np.full((n, 2), np.nan, dtype=float)

    ti = prep["long"]["trial_idx"].to_numpy(int)
    ii = prep["long"]["interval_index"].to_numpy(int)

    pden[ti, ii] = pden_long
    pnum[ti, ii] = pnum_long

    return pden, pnum


def phase_survival(end, event, weight, lower_exclusive, upper_inclusive):
    """
    Exact-time weighted KM contribution for one phase.

    Phases used:
        (0,1], (1,2], (2, follow-up]
    """
    event_mask = event == 1
    event_mask &= end > lower_exclusive
    if np.isfinite(upper_inclusive):
        event_mask &= end <= upper_inclusive

    if not event_mask.any():
        return 1.0

    # Clones ending before/equal the phase lower bound cannot enter its risk sets.
    risk_phase = end > lower_exclusive

    d = pd.DataFrame(
        {
            "end": end[risk_phase],
            "risk_w": weight[risk_phase],
            "event_w": weight[risk_phase] * event[risk_phase],
        }
    )
    g = d.groupby("end", sort=True, as_index=False).sum(numeric_only=True)
    g = g.sort_values("end", ascending=False).reset_index(drop=True)
    g["risk_cum"] = g["risk_w"].cumsum()

    eg = g[g["event_w"] > 0].copy()
    eg = eg[eg["end"] > lower_exclusive]
    if np.isfinite(upper_inclusive):
        eg = eg[eg["end"] <= upper_inclusive]

    if eg.empty:
        return 1.0

    frac = (eg["event_w"] / eg["risk_cum"]).clip(0, 1).to_numpy(float)
    return float(np.prod(1.0 - frac))


def estimate_v6(
    prep,
    ps2,
    pden_long,
    pnum_long,
    row_mult=None,
    factor_cap=None,
):
    """
    Baseline ATO target + 1h stabilized artificial-censoring weights.

    DEFER arm:
      phase 0 (0,1]      : h(X)
      phase 1 (1,2]      : h(X) * SW(no start in hour 1)
      post 2h            : e(X) * SW(no start h1) * SW(no start h2)

    INITIATE arm:
      through 2h         : h(X)
      post 2h early init : 1-e(X), IPCW factor 1
      post 2h late init  : (1-e(X)) * SW(start in second hour)

    This "recent initiator" weighting prevents early initiators from standing in
    for patients censored at +2h for failing to initiate.
    """
    n = len(prep["trials"])
    y2 = prep["y2"]
    geom = prep["geom"]

    ps2 = np.clip(np.asarray(ps2, float), PS_CLIP, 1 - PS_CLIP)
    pden, pnum = probabilities_by_trial(prep, pden_long, pnum_long)

    if row_mult is None:
        freq = np.ones(n, dtype=float)
    else:
        freq = np.asarray(row_mult, float)

    # interval 0 exists for every eligible trial.
    p0d = np.clip(pden[:, 0], PS_CLIP, 1 - PS_CLIP)
    p0n = np.clip(pnum[:, 0], PS_CLIP, 1 - PS_CLIP)
    sw_no0 = (1.0 - p0n) / (1.0 - p0d)

    # interval 1 exists only among alive/untreated at +1h.
    p1d = pden[:, 1].copy()
    p1n = pnum[:, 1].copy()

    # Fill unused rows harmlessly; such rows cannot contribute where factor is needed.
    p1d = np.where(np.isfinite(p1d), p1d, 0.5)
    p1n = np.where(np.isfinite(p1n), p1n, 0.5)
    p1d = np.clip(p1d, PS_CLIP, 1 - PS_CLIP)
    p1n = np.clip(p1n, PS_CLIP, 1 - PS_CLIP)

    sw_no1 = (1.0 - p1n) / (1.0 - p1d)
    sw_start1 = p1n / p1d

    if factor_cap is not None:
        sw_no0 = np.clip(sw_no0, 1.0 / factor_cap, factor_cap)
        sw_no1 = np.clip(sw_no1, 1.0 / factor_cap, factor_cap)
        sw_start1 = np.clip(sw_start1, 1.0 / factor_cap, factor_cap)

    h = ps2 * (1.0 - ps2)

    trial_hour = prep["trials"]["trial_hour"].to_numpy(float)
    init_abs = pd.to_numeric(
        prep["trials"]["any_init_h"], errors="coerce"
    ).to_numpy(float)
    init_rel = init_abs - trial_hour

    early = (y2 == 1) & np.isfinite(init_rel) & (init_rel >= 0) & (init_rel < 1)
    late = (y2 == 1) & np.isfinite(init_rel) & (init_rel >= 1) & (init_rel < 2)

    # Both arms: same ATO target before any interval-specific censoring correction.
    w_pre0 = freq * h

    # From 1h to 2h:
    w_defer_1 = freq * h * sw_no0
    w_init_1 = freq * h

    # After 2h:
    w_defer_post = freq * ps2 * sw_no0 * sw_no1
    init_ipcw = np.ones(n, dtype=float)
    init_ipcw[late] = sw_start1[late]
    w_init_post = freq * (1.0 - ps2) * init_ipcw

    # Exact event-time survival, with piecewise weights.
    s0 = phase_survival(
        geom["end0"], geom["event0"], w_pre0, 0.0, 1.0
    )
    s0 *= phase_survival(
        geom["end0"], geom["event0"], w_defer_1, 1.0, 2.0
    )
    s0 *= phase_survival(
        geom["end0"], geom["event0"], w_defer_post, 2.0, np.inf
    )

    s1 = phase_survival(
        geom["end1"], geom["event1"], w_pre0, 0.0, 1.0
    )
    s1 *= phase_survival(
        geom["end1"], geom["event1"], w_init_1, 1.0, 2.0
    )
    s1 *= phase_survival(
        geom["end1"], geom["event1"], w_init_post, 2.0, np.inf
    )

    r0 = 1.0 - s0
    r1 = 1.0 - s1

    return {
        "risk_defer_2h": r0,
        "risk_initiate_next2h": r1,
        "risk_difference": r1 - r0,
        "risk_ratio": r1 / r0 if r0 > 0 else np.nan,
        "sw_no0": sw_no0,
        "sw_no1": sw_no1,
        "sw_start1": sw_start1,
        "early": early,
        "late": late,
        "w_defer_post": w_defer_post,
        "w_init_post": w_init_post,
    }


def ess(w):
    x = np.asarray(w, float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    return (x.sum() ** 2) / np.sum(x ** 2)


def interval_counts(prep, population):
    long = prep["long"]
    rows = []

    for i in [0, 1]:
        x = long[long["interval_index"] == i]
        rows.append(
            {
                "population": population,
                "interval_index": i,
                "interval_label": "[0,1h)" if i == 0 else "[1,2h)",
                "at_risk_person_periods_n": len(x),
                "starts_next1h_n": int(x["start_next1h"].sum()),
                "start_rate": float(x["start_next1h"].mean()) if len(x) else np.nan,
                "unique_patients_n": x["stay_id"].nunique(),
            }
        )

    tr = prep["trials"]
    rel = pd.to_numeric(tr["any_init_h"], errors="coerce") - tr["trial_hour"]
    y = prep["y2"]

    rows.append(
        {
            "population": population,
            "interval_index": 99,
            "interval_label": "2h_strategy_summary",
            "at_risk_person_periods_n": len(tr),
            "starts_next1h_n": int(y.sum()),
            "start_rate": float(y.mean()),
            "unique_patients_n": tr["stay_id"].nunique(),
            "early_initiators_0_1h_n": int(((y == 1) & (rel >= 0) & (rel < 1)).sum()),
            "late_initiators_1_2h_n": int(((y == 1) & (rel >= 1) & (rel < 2)).sum()),
        }
    )
    return pd.DataFrame(rows)


def weight_diagnostics(prep, population, est):
    y = prep["y2"]
    early = est["early"]
    late = est["late"]
    no2 = y == 0

    groups = [
        ("defer_hour1_no_start_factor_all_trials", est["sw_no0"], np.ones(len(y), bool)),
        ("defer_hour2_no_start_factor_2h_defer", est["sw_no1"], no2),
        ("initiate_recent_start_factor_late_initiators", est["sw_start1"], late),
        ("postgrace_total_weight_defer", est["w_defer_post"], no2),
        ("postgrace_total_weight_initiate", est["w_init_post"], y == 1),
    ]

    rows = []
    for label, arr, mask in groups:
        x = np.asarray(arr, float)[mask]
        x = x[np.isfinite(x) & (x > 0)]
        rows.append(
            {
                "population": population,
                "diagnostic": label,
                "n": len(x),
                "mean": np.mean(x) if len(x) else np.nan,
                "p01": np.quantile(x, .01) if len(x) else np.nan,
                "p50": np.quantile(x, .50) if len(x) else np.nan,
                "p95": np.quantile(x, .95) if len(x) else np.nan,
                "p99": np.quantile(x, .99) if len(x) else np.nan,
                "max": np.max(x) if len(x) else np.nan,
                "ess": ess(x) if len(x) else np.nan,
            }
        )

    rows.append(
        {
            "population": population,
            "diagnostic": "initiator_timing_counts",
            "n": int(y.sum()),
            "mean": np.nan,
            "p01": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "ess": np.nan,
            "early_initiators_0_1h_n": int(early.sum()),
            "late_initiators_1_2h_n": int(late.sum()),
        }
    )
    return pd.DataFrame(rows)


def checkpoint_path(population):
    safe = population.replace(" ", "_").replace("/", "_")
    return CHECKPOINT_DIR / f"stage10_v6_{safe}_1h_ipcw_cluster_bootstrap.csv"


def load_checkpoint(path):
    cols = [
        "rep",
        "risk_defer_2h",
        "risk_initiate_next2h",
        "risk_difference",
        "risk_ratio",
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


def save_checkpoint(path, df):
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def cluster_bootstrap(v5, prep, population, target_reps, seed):
    ckpt = checkpoint_path(population)
    old = load_checkpoint(ckpt)
    completed = len(old)

    if completed >= target_reps:
        print(f"      [RESUME] checkpoint has {completed}; target={target_reps}")
        return old.iloc[:target_reps].copy()

    if completed:
        print(f"      [RESUME] loaded {completed}/{target_reps} from {ckpt.name}")

    rng = np.random.default_rng(seed)
    n_patients = len(prep["unique_ids"])

    # Reproduce the same draw stream after resume.
    for _ in range(completed):
        rng.integers(0, n_patients, size=n_patients)

    rec = old.to_dict("records")
    success = completed
    attempts = completed

    while success < target_reps:
        attempts += 1
        draw = rng.integers(0, n_patients, size=n_patients)
        patient_counts = np.bincount(draw, minlength=n_patients).astype(float)

        row_mult = patient_counts[prep["inverse_patient"]]
        long_mult = patient_counts[prep["long_patient_index"]]

        mask_base = row_mult > 0
        mask_long = long_mult > 0

        y2b = prep["y2"][mask_base]
        y1b = prep["y1"][mask_long]

        if y2b.sum() < 10 or y1b.sum() < 10:
            continue

        try:
            m_base = v5._fit_unpenalized_logistic(
                prep["X_base"][mask_base],
                y2b,
                sample_weight=row_mult[mask_base],
            )
            ps2 = m_base.predict_proba(prep["X_base"])[:, 1]
            ps2 = np.clip(ps2, PS_CLIP, 1 - PS_CLIP)

            m_den = fit_ridge_logistic(
                prep["X_den"][mask_long],
                y1b,
                sample_weight=long_mult[mask_long],
                C=0.5,
            )
            m_num = fit_ridge_logistic(
                prep["X_num"][mask_long],
                y1b,
                sample_weight=long_mult[mask_long],
                C=10.0,
            )

            pden = np.clip(
                m_den.predict_proba(prep["X_den"])[:, 1],
                PS_CLIP,
                1 - PS_CLIP,
            )
            pnum = np.clip(
                m_num.predict_proba(prep["X_num"])[:, 1],
                PS_CLIP,
                1 - PS_CLIP,
            )

            e = estimate_v6(
                prep,
                ps2,
                pden,
                pnum,
                row_mult=row_mult,
                factor_cap=None,
            )

            success += 1
            rec.append(
                {
                    "rep": success,
                    "risk_defer_2h": e["risk_defer_2h"],
                    "risk_initiate_next2h": e["risk_initiate_next2h"],
                    "risk_difference": e["risk_difference"],
                    "risk_ratio": e["risk_ratio"],
                }
            )

        except Exception as exc:
            print(
                f"      [WARN] bootstrap attempt {attempts} failed "
                f"({type(exc).__name__}); continuing"
            )
            continue

        if success % CHECKPOINT_EVERY == 0 or success == target_reps:
            cur = pd.DataFrame(rec)
            save_checkpoint(ckpt, cur)
            print(f"      checkpoint saved: {success}/{target_reps}")

    return pd.DataFrame(rec).iloc[:target_reps].copy()


def ci95(x):
    z = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(z) == 0:
        return np.nan, np.nan
    return tuple(np.quantile(z, [0.025, 0.975]))


def run_population(v5, trials, panel, population, boot_target, seed):
    print(f"\n=== final causal-analysis stage V6 1h IPCW: {population} ===")

    prep = prepare_population(v5, trials, panel)
    ps2, pden, pnum = fit_point_models(v5, prep)

    v5_ref = v5.estimate_ato(
        {
            "trials": prep["trials"],
            "y": prep["y2"],
            "X": prep["X_base"],
            "unique_ids": prep["unique_ids"],
            "inverse_patient": prep["inverse_patient"],
            "geom": prep["geom"],
        },
        ps2,
        mixed_v4=False,
    )

    raw = estimate_v6(prep, ps2, pden, pnum, factor_cap=None)
    cap = estimate_v6(
        prep, ps2, pden, pnum, factor_cap=FACTOR_CAP_SENSITIVITY
    )

    print(
        f"V5 reference: RD={v5_ref['risk_difference']:.4f}, "
        f"RR={v5_ref['risk_ratio']:.4f}"
    )
    print(
        f"V6 raw 1h IPCW: RD={raw['risk_difference']:.4f}, "
        f"RR={raw['risk_ratio']:.4f}"
    )
    print(
        f"V6 cap10 point sensitivity: RD={cap['risk_difference']:.4f}, "
        f"RR={cap['risk_ratio']:.4f}"
    )

    wdiag = weight_diagnostics(prep, population, raw)
    counts = interval_counts(prep, population)

    # Abort bootstrap only for truly pathological weights; still save point diagnostics.
    key = wdiag[
        wdiag["diagnostic"].isin(
            [
                "defer_hour1_no_start_factor_all_trials",
                "defer_hour2_no_start_factor_2h_defer",
                "initiate_recent_start_factor_late_initiators",
            ]
        )
    ]
    max_factor = float(key["max"].max()) if len(key) else np.nan

    if np.isfinite(max_factor) and max_factor > 50:
        print(
            f"[WARN] Raw IPCW factor max={max_factor:.2f} (>50). "
            "Bootstrap is still run for transparency, but treat raw inference cautiously."
        )

    boot = cluster_bootstrap(
        v5, prep, population, boot_target, seed
    )

    ci_r0 = ci95(boot["risk_defer_2h"])
    ci_r1 = ci95(boot["risk_initiate_next2h"])
    ci_rd = ci95(boot["risk_difference"])
    ci_rr = ci95(boot["risk_ratio"])

    effect = pd.DataFrame(
        [
            {
                "population": population,
                "analysis": "V6_1h_interval_stabilized_IPCW_raw",
                "role": "sensitivity_not_primary",
                "risk_defer_at_least_2h": raw["risk_defer_2h"],
                "risk_defer_ci_low": ci_r0[0],
                "risk_defer_ci_high": ci_r0[1],
                "risk_initiate_next2h": raw["risk_initiate_next2h"],
                "risk_initiate_ci_low": ci_r1[0],
                "risk_initiate_ci_high": ci_r1[1],
                "risk_difference": raw["risk_difference"],
                "rd_ci_low": ci_rd[0],
                "rd_ci_high": ci_rd[1],
                "risk_ratio": raw["risk_ratio"],
                "rr_ci_low": ci_rr[0],
                "rr_ci_high": ci_rr[1],
                "bootstrap_success_n": len(boot),
                "note": (
                    "Baseline ATO target retained; 1h time-updated stabilized IPCW "
                    "adjusts artificial censoring during the 2h grace period."
                ),
            },
            {
                "population": population,
                "analysis": "V6_1h_IPCW_factor_cap10_point_only",
                "role": "weight_robustness_point_sensitivity",
                "risk_defer_at_least_2h": cap["risk_defer_2h"],
                "risk_defer_ci_low": np.nan,
                "risk_defer_ci_high": np.nan,
                "risk_initiate_next2h": cap["risk_initiate_next2h"],
                "risk_initiate_ci_low": np.nan,
                "risk_initiate_ci_high": np.nan,
                "risk_difference": cap["risk_difference"],
                "rd_ci_low": np.nan,
                "rd_ci_high": np.nan,
                "risk_ratio": cap["risk_ratio"],
                "rr_ci_low": np.nan,
                "rr_ci_high": np.nan,
                "bootstrap_success_n": 0,
                "note": "Each interval-specific stabilized IPCW factor capped to [0.1,10].",
            },
        ]
    )

    compare = pd.DataFrame(
        [
            {
                "population": population,
                "analysis": "V5_primary_ATO_reference",
                "risk_defer": v5_ref["risk_defer_2h"],
                "risk_initiate": v5_ref["risk_initiate_next2h"],
                "risk_difference": v5_ref["risk_difference"],
                "risk_ratio": v5_ref["risk_ratio"],
            },
            {
                "population": population,
                "analysis": "V6_1h_interval_IPCW_raw",
                "risk_defer": raw["risk_defer_2h"],
                "risk_initiate": raw["risk_initiate_next2h"],
                "risk_difference": raw["risk_difference"],
                "risk_ratio": raw["risk_ratio"],
            },
            {
                "population": population,
                "analysis": "V6_1h_interval_IPCW_cap10",
                "risk_defer": cap["risk_defer_2h"],
                "risk_initiate": cap["risk_initiate_next2h"],
                "risk_difference": cap["risk_difference"],
                "risk_ratio": cap["risk_ratio"],
            },
        ]
    )

    bsum = pd.DataFrame(
        [
            {
                "population": population,
                "bootstrap_success_n": len(boot),
                "rd_median": float(boot["risk_difference"].median()),
                "rd_p025": ci_rd[0],
                "rd_p975": ci_rd[1],
                "rr_median": float(boot["risk_ratio"].median()),
                "rr_p025": ci_rr[0],
                "rr_p975": ci_rr[1],
                "checkpoint_file": str(checkpoint_path(population)),
            }
        ]
    )

    return effect, wdiag, counts, compare, bsum


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    v5 = import_v5()

    print("=" * 100)
    print("final causal-analysis stage V6 — 1-hour interval-specific IPCW sensitivity")
    print("=" * 100)
    print(
        "V5 remains PRIMARY. V6 tests whether time-updated grace-period "
        "artificial-censoring adjustment materially changes the result."
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
            v5,
            broad_trials,
            panel,
            "broad_primary",
            BROAD_BOOTSTRAP_TARGET,
            SEED,
        )

        c = run_population(
            v5,
            cape_trials,
            cape_panel,
            "CAPE_COD_like",
            CAPE_BOOTSTRAP_TARGET,
            SEED + 1000,
        )

        pd.concat([b[0], c[0]], ignore_index=True).to_csv(
            OUT_EFFECT, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[1], c[1]], ignore_index=True).to_csv(
            OUT_WEIGHTS, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[2], c[2]], ignore_index=True).to_csv(
            OUT_COUNTS, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[3], c[3]], ignore_index=True).to_csv(
            OUT_COMPARE, index=False, encoding="utf-8-sig"
        )
        pd.concat([b[4], c[4]], ignore_index=True).to_csv(
            OUT_BOOT, index=False, encoding="utf-8-sig"
        )

        print("\n" + "=" * 100)
        print("STAGE 10 V6 COMPLETE")
        print("=" * 100)
        print("\nGenerated five audit CSVs:")
        for p in [
            OUT_EFFECT,
            OUT_WEIGHTS,
            OUT_COUNTS,
            OUT_COMPARE,
            OUT_BOOT,
        ]:
            print(" ", p)

        print(
            "\nIf V6 Broad is directionally/quantitatively consistent with V5 "
            "and weights are not pathological, freeze final causal-analysis stage and move to Stage 11."
        )

    except Exception:
        print("\n[FAILED] Copy the complete traceback to ChatGPT:\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

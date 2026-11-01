from __future__ import annotations

"""
Stage 7 — Time-updated Clone-Censor-Weight target trial emulation

WHY STAGE 7
-----------
Stage 6 used baseline covariates only. In severe CAP, steroid initiation is often
triggered by evolving physiology during the grace period (worsening oxygenation,
vasopressors, intubation, lactate/pH changes). Those time-updated factors predict
both treatment initiation and mortality.

Stage 7 therefore rebuilds the artificial-censoring model hourly using ONLY
information observed before the start of each hour.

PRIMARY TARGET TRIALS
---------------------
Three separate grace-period trials:
    A) initiate any systemic corticosteroid within 6 h vs no initiation within 6 h
    B) initiate any systemic corticosteroid within 12 h vs no initiation within 12 h
    C) initiate any systemic corticosteroid within 24 h vs no initiation within 24 h

After each grace period, treatment is no longer constrained.

Primary population:
    Stage 5.5 cohort excluding major_immunocompromise_proxy.

Primary outcome:
    28-day all-cause mortality.

Primary estimands:
    28-day risk under each strategy, RD, RR.

Exploratory exposure sensitivities:
    - IV systemic corticosteroid initiation <=24 h vs no IV initiation <=24 h
    - hydrocortisone initiation <=24 h vs no hydrocortisone initiation <=24 h
These are exposure-definition sensitivities; other systemic steroids are permitted
as co-interventions and therefore these are NOT equivalent to hydrocortisone-vs-none.

Generated audit outputs:
    outputs/12_STAGE7_DYNAMIC_PRIMARY.csv
    outputs/12B_STAGE7_WEIGHT_DIAGNOSTICS.csv
    outputs/12C_STAGE7_DYNAMIC_BALANCE.csv
    outputs/12D_STAGE7_DRUG_SENSITIVITY.csv
    outputs/12E_STAGE7_MODEL_DIAGNOSTICS.csv
    outputs/12F_STAGE7_STAGE6_COMPARISON.csv

Local caches (keep restricted locally):
    cache/stage7_dynamic_vitals.parquet
    cache/stage7_dynamic_labs.parquet
    cache/stage7_support_events.parquet
    cache/stage7_vasopressor_events.parquet
    cache/stage7_steroid_events.parquet
    outputs/stage7_hourly_panel.parquet

The first run may be slow because chartevents/labevents are scanned once.
"""

from pathlib import Path
import math
import traceback
import warnings

import duckdb
import numpy as np
import pandas as pd

import config
from src.utils import connect_duckdb, find_table, source_sql, sql_quote_path

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    raise RuntimeError(
        "Stage 7 needs scikit-learn.\n"
        "Run in a terminal:\n"
        "    pip install scikit-learn\n"
        f"Original import error: {e}"
    )

PROJECT_ROOT = Path(config.PROJECT_ROOT)
DATA_ROOT = Path(config.DATA_ROOT)
CACHE_DIR = Path(config.CACHE_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

COHORT_PATH = OUTPUT_DIR / "stage5_5_analysis_cohort.parquet"
STAGE6_PATH = OUTPUT_DIR / "11_STAGE6_PRIMARY_EFFECT.csv"

VITAL_CACHE = CACHE_DIR / "stage7_dynamic_vitals.parquet"
LAB_CACHE = CACHE_DIR / "stage7_dynamic_labs.parquet"
SUPPORT_CACHE = CACHE_DIR / "stage7_support_events.parquet"
VASO_CACHE = CACHE_DIR / "stage7_vasopressor_events.parquet"
STEROID_CACHE = CACHE_DIR / "stage7_steroid_events.parquet"
PANEL_PATH = OUTPUT_DIR / "stage7_hourly_panel.parquet"

OUT_PRIMARY = OUTPUT_DIR / "12_STAGE7_DYNAMIC_PRIMARY.csv"
OUT_WEIGHT = OUTPUT_DIR / "12B_STAGE7_WEIGHT_DIAGNOSTICS.csv"
OUT_BAL = OUTPUT_DIR / "12C_STAGE7_DYNAMIC_BALANCE.csv"
OUT_DRUG = OUTPUT_DIR / "12D_STAGE7_DRUG_SENSITIVITY.csv"
OUT_MODEL = OUTPUT_DIR / "12E_STAGE7_MODEL_DIAGNOSTICS.csv"
OUT_COMPARE = OUTPUT_DIR / "12F_STAGE7_STAGE6_COMPARISON.csv"

MAX_GRACE = 24
FOLLOWUP_H = 28 * 24
FORCE_REBUILD_STAGE7_CACHE = False
RANDOM_SEED = 20261027

# Discovery-stage bootstrap counts. Increase to 500-1000 for the final paper.
BOOTSTRAP_REPS = {6: 80, 12: 80, 24: 120}
DRUG_BOOTSTRAP_REPS = 50

TRUNC_Q_LOW = 0.01
TRUNC_Q_HIGH = 0.99

# Exact MIMIC-IV ICU itemids used in Stage 5.5.
VITAL_ITEMS = {
    220045: "heart_rate",
    220052: "map",
    220181: "map",
    225312: "map",
    220210: "respiratory_rate",
    224690: "respiratory_rate",
    220277: "spo2",
    223762: "temp_c",
    223761: "temp_f",
    223835: "fio2",
    223901: "gcs_motor",
}

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

DYNAMIC_NUM = [
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
DYNAMIC_BIN = [
    "vasopressor_active",
    "support_imv",
    "support_niv",
    "support_hfnc",
]


def qpath(p: Path) -> str:
    return sql_quote_path(p)


def safe_dt(x):
    return pd.to_datetime(x, errors="coerce")


def scalar(con, sql, params=None):
    return int(con.execute(sql, params or []).fetchone()[0] or 0)


def table_exists(con, table_name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE lower(table_name)=lower(?)
            """,
            [table_name],
        ).fetchone()[0]
    )


def attach_best_death_time(df: pd.DataFrame) -> pd.DataFrame:
    admissions_path = find_table(DATA_ROOT, "hosp", "admissions")
    con = connect_duckdb(PROJECT_ROOT, getattr(config, "DUCKDB_THREADS", 4))
    try:
        adm = source_sql(admissions_path)
        death = con.execute(f"SELECT hadm_id, deathtime FROM {adm}").fetchdf()
    finally:
        con.close()

    x = df.merge(death, on="hadm_id", how="left")
    x["respiratory_t0"] = safe_dt(x["respiratory_t0"])
    x["deathtime"] = safe_dt(x["deathtime"])
    x["dod"] = safe_dt(x.get("dod"))

    death_dt = x["deathtime"].copy()
    fallback = x["dod"].dt.normalize() + pd.Timedelta(
        hours=23, minutes=59, seconds=59
    )
    fallback = fallback.where(fallback >= x["respiratory_t0"])
    death_dt = death_dt.where(death_dt.notna(), fallback)

    x["death_hours"] = (
        (death_dt - x["respiratory_t0"]).dt.total_seconds() / 3600.0
    )
    x.loc[
        (x["death_hours"] < 0) | (x["death_hours"] > FOLLOWUP_H),
        "death_hours",
    ] = np.nan

    missing = (x["death_28d"] == 1) & x["death_hours"].isna()
    x.loc[missing, "death_hours"] = FOLLOWUP_H - 1e-3
    x["death_time_fallback_flag"] = missing.astype(int)
    return x


def load_primary_cohort() -> pd.DataFrame:
    if not COHORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {COHORT_PATH}\nRun Stage 5.5 first."
        )
    df = pd.read_parquet(COHORT_PATH)
    if df["stay_id"].duplicated().any():
        raise RuntimeError("stage5_5_analysis_cohort contains duplicate stay_id.")
    df = attach_best_death_time(df)
    df = df[df["major_immunocompromise_proxy"] == 0].copy()
    return df.reset_index(drop=True)


def build_dynamic_caches(df: pd.DataFrame):
    """
    Scan raw event tables once and cache only selected patients, itemids and time windows.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = connect_duckdb(PROJECT_ROOT, getattr(config, "DUCKDB_THREADS", 4))
    try:
        # Make the exact primary cohort visible to DuckDB.
        temp_primary = OUTPUT_DIR / "_stage7_primary_ids.parquet"
        df[
            ["subject_id", "hadm_id", "stay_id", "respiratory_t0"]
        ].to_parquet(temp_primary, index=False)
        ids = f"read_parquet('{qpath(temp_primary)}')"

        # ---------------- ICU dynamic vitals ----------------
        if FORCE_REBUILD_STAGE7_CACHE and VITAL_CACHE.exists():
            VITAL_CACHE.unlink()
        if not VITAL_CACHE.exists():
            print("[CACHE 1/5] Scanning selected dynamic ICU vitals...")
            ce_path = find_table(DATA_ROOT, "icu", "chartevents")
            ce = source_sql(ce_path)
            itemids = ",".join(str(x) for x in VITAL_ITEMS)
            con.execute(
                f"""
                COPY (
                    SELECT
                        p.stay_id,
                        p.respiratory_t0,
                        c.charttime,
                        c.itemid,
                        c.value,
                        c.valuenum,
                        c.valueuom
                    FROM {ids} p
                    JOIN {ce} c
                      ON c.stay_id=p.stay_id
                     AND c.charttime BETWEEN p.respiratory_t0 - INTERVAL '12 hours'
                                         AND p.respiratory_t0 + INTERVAL '24 hours'
                    WHERE c.itemid IN ({itemids})
                      AND c.valuenum IS NOT NULL
                )
                TO '{qpath(VITAL_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        else:
            print(f"[CACHE 1/5] Reusing {VITAL_CACHE}")

        # ---------------- Dynamic labs ----------------
        if FORCE_REBUILD_STAGE7_CACHE and LAB_CACHE.exists():
            LAB_CACHE.unlink()
        if not LAB_CACHE.exists():
            print("[CACHE 2/5] Scanning selected dynamic labs...")
            labs_path = find_table(DATA_ROOT, "hosp", "labevents")
            dlab_path = find_table(DATA_ROOT, "hosp", "d_labitems")
            labs = source_sql(labs_path)
            dlab = source_sql(dlab_path)

            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE stage7_lab_map AS
                SELECT DISTINCT
                    itemid,
                    CASE
                      WHEN lower(trim(label))='lactate' THEN 'lactate'
                      WHEN lower(trim(label))='creatinine' THEN 'creatinine'
                      WHEN lower(trim(label))='white blood cells' THEN 'wbc'
                      WHEN lower(trim(label))='bicarbonate' THEN 'bicarbonate'
                      WHEN lower(trim(label))='glucose' THEN 'glucose'
                      WHEN lower(trim(label))='ph' THEN 'ph'
                      WHEN lower(trim(label))='po2' THEN 'po2'
                      WHEN lower(trim(label))='pco2' THEN 'pco2'
                      ELSE NULL
                    END AS target
                FROM {dlab}
                WHERE lower(coalesce(fluid,'')) IN ('blood','serum','plasma')
                  AND lower(trim(label)) IN (
                    'lactate','creatinine','white blood cells',
                    'bicarbonate','glucose','ph','po2','pco2'
                  )
                """
            )

            con.execute(
                f"""
                COPY (
                    SELECT
                        p.stay_id,
                        p.respiratory_t0,
                        l.charttime,
                        m.target,
                        l.valuenum,
                        l.valueuom,
                        l.hadm_id AS lab_hadm_id
                    FROM {ids} p
                    JOIN {labs} l
                      ON l.subject_id=p.subject_id
                     AND (l.hadm_id=p.hadm_id OR l.hadm_id IS NULL)
                     AND l.charttime BETWEEN p.respiratory_t0 - INTERVAL '24 hours'
                                         AND p.respiratory_t0 + INTERVAL '24 hours'
                    JOIN stage7_lab_map m USING (itemid)
                    WHERE m.target IS NOT NULL
                      AND l.valuenum IS NOT NULL
                )
                TO '{qpath(LAB_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        else:
            print(f"[CACHE 2/5] Reusing {LAB_CACHE}")

        # ---------------- Respiratory support events ----------------
        if not table_exists(con, "respiratory_support_events"):
            raise RuntimeError(
                "Persistent table respiratory_support_events is missing. "
                "Run Stage 1/4 once in the same project."
            )
        if FORCE_REBUILD_STAGE7_CACHE and SUPPORT_CACHE.exists():
            SUPPORT_CACHE.unlink()
        if not SUPPORT_CACHE.exists():
            print("[CACHE 3/5] Caching respiratory support events...")
            con.execute(
                f"""
                COPY (
                    SELECT
                        p.stay_id,
                        p.respiratory_t0,
                        r.charttime,
                        r.invasive_device,
                        r.ventilator_signal,
                        r.niv_device,
                        r.hfnc_remap_like,
                        r.resp_support_strict
                    FROM {ids} p
                    JOIN respiratory_support_events r
                      ON r.stay_id=p.stay_id
                     AND r.charttime BETWEEN p.respiratory_t0 - INTERVAL '6 hours'
                                         AND p.respiratory_t0 + INTERVAL '24 hours'
                )
                TO '{qpath(SUPPORT_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        else:
            print(f"[CACHE 3/5] Reusing {SUPPORT_CACHE}")

        # ---------------- Vasopressors ----------------
        if not table_exists(con, "vasopressor_events"):
            raise RuntimeError(
                "Persistent table vasopressor_events is missing. "
                "Run Stage 1/4 once in the same project."
            )
        if FORCE_REBUILD_STAGE7_CACHE and VASO_CACHE.exists():
            VASO_CACHE.unlink()
        if not VASO_CACHE.exists():
            print("[CACHE 4/5] Caching vasopressor intervals...")
            con.execute(
                f"""
                COPY (
                    SELECT
                        p.stay_id,
                        p.respiratory_t0,
                        v.target_name,
                        v.starttime,
                        v.endtime,
                        v.rate,
                        v.rateuom
                    FROM {ids} p
                    JOIN vasopressor_events v
                      ON v.stay_id=p.stay_id
                     AND v.starttime <= p.respiratory_t0 + INTERVAL '24 hours'
                     AND coalesce(v.endtime, v.starttime + INTERVAL '1 minute')
                         >= p.respiratory_t0 - INTERVAL '1 hour'
                )
                TO '{qpath(VASO_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        else:
            print(f"[CACHE 4/5] Reusing {VASO_CACHE}")

        # ---------------- Actual systemic steroid events ----------------
        if not table_exists(con, "stage4_steroid_events"):
            raise RuntimeError(
                "Persistent table stage4_steroid_events is missing. "
                "Run Stage 4 once in the same project."
            )
        if FORCE_REBUILD_STAGE7_CACHE and STEROID_CACHE.exists():
            STEROID_CACHE.unlink()
        if not STEROID_CACHE.exists():
            print("[CACHE 5/5] Caching eMAR-confirmed systemic steroid events...")
            con.execute(
                f"""
                COPY (
                    SELECT
                        p.stay_id,
                        p.respiratory_t0,
                        s.charttime,
                        s.steroid_name,
                        s.route_best,
                        s.route_class,
                        date_diff('minute', p.respiratory_t0, s.charttime)/60.0
                            AS hours_from_t0
                    FROM {ids} p
                    JOIN stage4_steroid_events s USING (stay_id)
                    WHERE lower(trim(coalesce(s.event_txt,'')))='administered'
                      AND s.route_class='systemic'
                      AND s.charttime >= p.respiratory_t0
                      AND s.charttime <= p.respiratory_t0 + INTERVAL '24 hours'
                )
                TO '{qpath(STEROID_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        else:
            print(f"[CACHE 5/5] Reusing {STEROID_CACHE}")

        try:
            temp_primary.unlink()
        except Exception:
            pass
    finally:
        con.close()


def normalize_vitals(v: pd.DataFrame) -> pd.DataFrame:
    x = v.copy()
    x["charttime"] = safe_dt(x["charttime"])
    x["respiratory_t0"] = safe_dt(x["respiratory_t0"])
    x["h"] = (
        (x["charttime"] - x["respiratory_t0"]).dt.total_seconds() / 3600.0
    )
    x["target"] = x["itemid"].map(VITAL_ITEMS)
    x["value_norm"] = pd.to_numeric(x["valuenum"], errors="coerce")

    # Temperature conversion.
    m = x["target"].eq("temp_f")
    x.loc[m, "value_norm"] = (x.loc[m, "value_norm"] - 32.0) * 5.0 / 9.0
    x.loc[m, "target"] = "temp_c"

    # FiO2 conversion to percent.
    m = x["target"].eq("fio2")
    raw = x.loc[m, "value_norm"]
    x.loc[m & x["value_norm"].between(0.20, 1.0), "value_norm"] = (
        x.loc[m & x["value_norm"].between(0.20, 1.0), "value_norm"] * 100.0
    )

    # Physiologic plausibility.
    valid = pd.Series(True, index=x.index)
    ranges = {
        "heart_rate": (20, 250),
        "respiratory_rate": (2, 70),
        "map": (20, 200),
        "spo2": (40, 100),
        "temp_c": (25, 45),
        "fio2": (20, 100),
        "gcs_motor": (1, 6),
    }
    for target, (lo, hi) in ranges.items():
        mm = x["target"].eq(target)
        valid.loc[mm] = x.loc[mm, "value_norm"].between(lo, hi)
    return x[valid & x["target"].notna()].copy()


def normalize_labs(l: pd.DataFrame) -> pd.DataFrame:
    x = l.copy()
    x["charttime"] = safe_dt(x["charttime"])
    x["respiratory_t0"] = safe_dt(x["respiratory_t0"])
    x["h"] = (
        (x["charttime"] - x["respiratory_t0"]).dt.total_seconds() / 3600.0
    )
    x["value_norm"] = pd.to_numeric(x["valuenum"], errors="coerce")
    ranges = {
        "lactate": (0.1, 30),
        "creatinine": (0.1, 20),
        "wbc": (0.1, 100),
        "bicarbonate": (3, 60),
        "glucose": (20, 1000),
        "ph": (6.5, 8.0),
        "po2": (10, 700),
        "pco2": (10, 200),
    }
    valid = pd.Series(True, index=x.index)
    for target, (lo, hi) in ranges.items():
        mm = x["target"].eq(target)
        valid.loc[mm] = x.loc[mm, "value_norm"].between(lo, hi)
    return x[valid & x["target"].notna()].copy()


def make_value_store(df: pd.DataFrame):
    """
    {stay_id: {target: (times_array, values_array)}}
    """
    store = {}
    if len(df) == 0:
        return store
    for (sid, target), g in df.sort_values("h").groupby(["stay_id", "target"]):
        store.setdefault(sid, {})[target] = (
            g["h"].to_numpy(float),
            g["value_norm"].to_numpy(float),
        )
    return store


def values_before(store, sid, target, cutoff, lookback):
    if sid not in store or target not in store[sid]:
        return np.array([], dtype=float)
    times, vals = store[sid][target]
    # Strictly before hour start to avoid same-hour treatment leakage.
    mask = (times < cutoff - 1e-9) & (times >= cutoff - lookback)
    return vals[mask]


def last_before(store, sid, target, cutoff, lookback):
    if sid not in store or target not in store[sid]:
        return np.nan
    times, vals = store[sid][target]
    mask = (times < cutoff - 1e-9) & (times >= cutoff - lookback)
    if not mask.any():
        return np.nan
    tt = times[mask]
    vv = vals[mask]
    return vv[np.argmax(tt)]


def baseline_support_level(row) -> int:
    typ = str(row.get("respiratory_t0_type", ""))
    if typ == "IMV_or_ventilator":
        return 3
    if typ == "NIV":
        return 2
    if typ == "qualified_HFNC":
        return 1
    return 0


def build_support_store(sup: pd.DataFrame):
    store = {}
    if len(sup) == 0:
        return store
    x = sup.copy()
    x["charttime"] = safe_dt(x["charttime"])
    x["respiratory_t0"] = safe_dt(x["respiratory_t0"])
    x["h"] = (
        (x["charttime"] - x["respiratory_t0"]).dt.total_seconds() / 3600.0
    )
    x["level"] = np.select(
        [
            (x["invasive_device"].fillna(0) == 1)
            | (x["ventilator_signal"].fillna(0) == 1),
            x["niv_device"].fillna(0) == 1,
            x["hfnc_remap_like"].fillna(0) == 1,
        ],
        [3, 2, 1],
        default=0,
    )
    for sid, g in x.sort_values("h").groupby("stay_id"):
        store[sid] = (g["h"].to_numpy(float), g["level"].to_numpy(int))
    return store


def support_at(store, sid, cutoff, baseline_level):
    if cutoff <= 0:
        return baseline_level
    if sid not in store:
        return baseline_level
    times, levels = store[sid]
    # Prefer the most severe signal in the preceding 2 h.
    mask2 = (times < cutoff - 1e-9) & (times >= cutoff - 2)
    if mask2.any():
        return int(np.max(levels[mask2]))
    # Otherwise latest documented support within 6 h.
    mask6 = (times < cutoff - 1e-9) & (times >= cutoff - 6)
    if mask6.any():
        tt = times[mask6]
        ll = levels[mask6]
        return int(ll[np.argmax(tt)])
    return baseline_level


def build_vaso_store(vaso: pd.DataFrame):
    store = {}
    if len(vaso) == 0:
        return store
    x = vaso.copy()
    x["starttime"] = safe_dt(x["starttime"])
    x["endtime"] = safe_dt(x["endtime"])
    x["respiratory_t0"] = safe_dt(x["respiratory_t0"])
    x["start_h"] = (
        (x["starttime"] - x["respiratory_t0"]).dt.total_seconds() / 3600.0
    )
    x["end_h"] = (
        (x["endtime"] - x["respiratory_t0"]).dt.total_seconds() / 3600.0
    )
    x["end_h"] = x["end_h"].where(
        x["end_h"].notna(), x["start_h"] + 1.0 / 60.0
    )
    for sid, g in x.groupby("stay_id"):
        store[sid] = g[
            ["start_h", "end_h", "target_name"]
        ].copy()
    return store


def vaso_at(store, sid, cutoff):
    if sid not in store:
        return 0, 0
    g = store[sid]
    active = (g["start_h"] <= cutoff) & (g["end_h"] >= cutoff)
    if not active.any():
        return 0, 0
    return 1, int(g.loc[active, "target_name"].nunique())


def build_steroid_times(steroid: pd.DataFrame) -> pd.DataFrame:
    if len(steroid) == 0:
        return pd.DataFrame(
            columns=["stay_id", "any_init_h", "iv_init_h", "hydrocortisone_init_h"]
        )
    x = steroid.copy()
    x["hours_from_t0"] = pd.to_numeric(x["hours_from_t0"], errors="coerce")
    x = x[x["hours_from_t0"].between(0, 24, inclusive="both")].copy()
    route = x["route_best"].fillna("").astype(str).str.upper()
    x["is_iv"] = route.str.startswith("IV")
    x["is_hydro"] = x["steroid_name"].eq("hydrocortisone")
    out = (
        x.groupby("stay_id")
        .apply(
            lambda g: pd.Series(
                {
                    "any_init_h": g["hours_from_t0"].min(),
                    "iv_init_h": g.loc[g["is_iv"], "hours_from_t0"].min()
                    if g["is_iv"].any()
                    else np.nan,
                    "hydrocortisone_init_h": g.loc[
                        g["is_hydro"], "hours_from_t0"
                    ].min()
                    if g["is_hydro"].any()
                    else np.nan,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return out


def build_hourly_panel(df: pd.DataFrame) -> pd.DataFrame:
    if PANEL_PATH.exists() and not FORCE_REBUILD_STAGE7_CACHE:
        print(f"[PANEL] Reusing {PANEL_PATH}")
        return pd.read_parquet(PANEL_PATH)

    print("[PANEL] Building 24-hour time-updated physiology panel...")
    v = normalize_vitals(pd.read_parquet(VITAL_CACHE))
    l = normalize_labs(pd.read_parquet(LAB_CACHE))
    sup = pd.read_parquet(SUPPORT_CACHE)
    vaso = pd.read_parquet(VASO_CACHE)
    steroid = pd.read_parquet(STEROID_CACHE)

    vital_store = make_value_store(v)
    lab_store = make_value_store(l)
    support_store = build_support_store(sup)
    vaso_store = build_vaso_store(vaso)
    steroid_times = build_steroid_times(steroid)

    base = df.merge(steroid_times, on="stay_id", how="left")
    rows = []

    for i, r in base.iterrows():
        sid = r["stay_id"]
        death_h = r["death_hours"]
        b_support = baseline_support_level(r)

        for h in range(MAX_GRACE):
            # No hazard row after death at/before hour start.
            if pd.notna(death_h) and death_h <= h:
                break

            cutoff = float(h)

            # At h=0, use the already-validated Stage 5.5 baseline values.
            if h == 0:
                hr = r.get("hr_max", np.nan)
                rr = r.get("rr_max", np.nan)
                map_min = r.get("map_min", np.nan)
                spo2 = r.get("spo2_min", np.nan)
                fio2 = r.get("fio2_max_pct", np.nan)
                temp = r.get("temp_max_c", np.nan)
                gcs_m = r.get("gcs_motor_last", np.nan)
                lact = r.get("lactate_max", np.nan)
                ph = r.get("ph_min", np.nan)
                po2 = r.get("po2_last", np.nan)
                pco2 = r.get("pco2_last", np.nan)
                creat = r.get("creatinine_last", np.nan)
                bicarb = r.get("bicarbonate_last", np.nan)
                glucose = r.get("glucose_last", np.nan)
                wbc = r.get("wbc_last", np.nan)
            else:
                vals = values_before(vital_store, sid, "heart_rate", cutoff, 6)
                hr = np.max(vals) if len(vals) else np.nan
                vals = values_before(vital_store, sid, "respiratory_rate", cutoff, 6)
                rr = np.max(vals) if len(vals) else np.nan
                vals = values_before(vital_store, sid, "map", cutoff, 6)
                map_min = np.min(vals) if len(vals) else np.nan
                spo2 = last_before(vital_store, sid, "spo2", cutoff, 6)
                fio2 = last_before(vital_store, sid, "fio2", cutoff, 6)
                vals = values_before(vital_store, sid, "temp_c", cutoff, 6)
                temp = np.max(vals) if len(vals) else np.nan
                gcs_m = last_before(vital_store, sid, "gcs_motor", cutoff, 6)

                vals = values_before(lab_store, sid, "lactate", cutoff, 12)
                lact = np.max(vals) if len(vals) else np.nan
                vals = values_before(lab_store, sid, "ph", cutoff, 12)
                ph = np.min(vals) if len(vals) else np.nan
                po2 = last_before(lab_store, sid, "po2", cutoff, 12)
                pco2 = last_before(lab_store, sid, "pco2", cutoff, 12)
                creat = last_before(lab_store, sid, "creatinine", cutoff, 24)
                bicarb = last_before(lab_store, sid, "bicarbonate", cutoff, 24)
                glucose = last_before(lab_store, sid, "glucose", cutoff, 24)
                wbc = last_before(lab_store, sid, "wbc", cutoff, 24)

                # If a dynamic value is missing, carry baseline information forward
                # rather than treating missingness as physiologic normalization.
                if pd.isna(hr):
                    hr = r.get("hr_max", np.nan)
                if pd.isna(rr):
                    rr = r.get("rr_max", np.nan)
                if pd.isna(map_min):
                    map_min = r.get("map_min", np.nan)
                if pd.isna(spo2):
                    spo2 = r.get("spo2_min", np.nan)
                if pd.isna(fio2):
                    fio2 = r.get("fio2_max_pct", np.nan)
                if pd.isna(temp):
                    temp = r.get("temp_max_c", np.nan)
                if pd.isna(gcs_m):
                    gcs_m = r.get("gcs_motor_last", np.nan)
                if pd.isna(lact):
                    lact = r.get("lactate_max", np.nan)
                if pd.isna(ph):
                    ph = r.get("ph_min", np.nan)
                if pd.isna(po2):
                    po2 = r.get("po2_last", np.nan)
                if pd.isna(pco2):
                    pco2 = r.get("pco2_last", np.nan)
                if pd.isna(creat):
                    creat = r.get("creatinine_last", np.nan)
                if pd.isna(bicarb):
                    bicarb = r.get("bicarbonate_last", np.nan)
                if pd.isna(glucose):
                    glucose = r.get("glucose_last", np.nan)
                if pd.isna(wbc):
                    wbc = r.get("wbc_last", np.nan)

            sf = (
                spo2 / (fio2 / 100.0)
                if pd.notna(spo2) and pd.notna(fio2) and 20 <= fio2 <= 100
                else np.nan
            )

            support = support_at(support_store, sid, cutoff, b_support)
            vp, vp_agents = vaso_at(vaso_store, sid, cutoff)

            row = {
                "stay_id": sid,
                "hour": h,
                "hour2": h * h,
                "death_hours": death_h,
                "any_init_h": r.get("any_init_h", np.nan),
                "iv_init_h": r.get("iv_init_h", np.nan),
                "hydrocortisone_init_h": r.get("hydrocortisone_init_h", np.nan),
                "dyn_hr_max6": hr,
                "dyn_rr_max6": rr,
                "dyn_map_min6": map_min,
                "dyn_spo2_last6": spo2,
                "dyn_fio2_last6": fio2,
                "dyn_sf_ratio": sf,
                "dyn_temp_max6": temp,
                "dyn_gcs_motor_last6": gcs_m,
                "dyn_lactate_max12": lact,
                "dyn_ph_min12": ph,
                "dyn_po2_last12": po2,
                "dyn_pco2_last12": pco2,
                "dyn_creatinine_last24": creat,
                "dyn_bicarbonate_last24": bicarb,
                "dyn_glucose_last24": glucose,
                "dyn_wbc_last24": wbc,
                "support_level": support,
                "support_imv": int(support == 3),
                "support_niv": int(support == 2),
                "support_hfnc": int(support == 1),
                "vasopressor_active": vp,
                "vasopressor_agents_active": vp_agents,
            }
            for c in BASELINE_NUM + BASELINE_BIN + BASELINE_CAT:
                row[c] = r.get(c, np.nan)
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel.to_parquet(PANEL_PATH, index=False)
    print(f"[PANEL] Saved {len(panel):,} patient-hours -> {PANEL_PATH}")
    return panel


def make_preprocessor(panel: pd.DataFrame, numerator=False):
    if numerator:
        num_cols = ["hour", "hour2"]
        prep = ColumnTransformer(
            [
                (
                    "time",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    num_cols,
                )
            ],
            remainder="drop",
        )
        return prep

    numeric = [
        c
        for c in BASELINE_NUM + DYNAMIC_NUM + DYNAMIC_BIN + BASELINE_BIN
        if c in panel.columns
    ]
    categorical = [c for c in BASELINE_CAT if c in panel.columns]

    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric + ["hour", "hour2"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return prep


def hazard_dataset(panel: pd.DataFrame, exposure_col: str, grace: int):
    x = panel[panel["hour"] < grace].copy()
    init = pd.to_numeric(x[exposure_col], errors="coerce")
    death = pd.to_numeric(x["death_hours"], errors="coerce")

    # At risk at the start of this hour only if alive and untreated.
    alive = death.isna() | (death > x["hour"])
    untreated_start = init.isna() | (init >= x["hour"])
    x = x[alive & untreated_start].copy()

    init = pd.to_numeric(x[exposure_col], errors="coerce")
    death = pd.to_numeric(x["death_hours"], errors="coerce")
    event_hour = np.floor(init).clip(upper=grace - 1)

    x["y_init"] = (
        init.notna()
        & (init <= grace)
        & (event_hour == x["hour"])
        & (death.isna() | (init <= death))
    ).astype(int)

    return x


def fit_hazard_models(haz: pd.DataFrame):
    if int(haz["y_init"].sum()) < 20:
        raise RuntimeError(
            f"Too few initiation events for a stable model: {int(haz['y_init'].sum())}"
        )

    den = Pipeline(
        [
            ("prep", make_preprocessor(haz, numerator=False)),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    num = Pipeline(
        [
            ("prep", make_preprocessor(haz, numerator=True)),
            (
                "logit",
                LogisticRegression(
                    C=10.0,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    y = haz["y_init"].astype(int)
    den.fit(haz, y)
    num.fit(haz, y)

    out = haz.copy()
    out["p_den"] = den.predict_proba(haz)[:, 1].clip(1e-5, 1 - 1e-5)
    out["p_num"] = num.predict_proba(haz)[:, 1].clip(1e-5, 1 - 1e-5)
    return out, den, num


def predict_all(panel: pd.DataFrame, grace: int, den, num):
    x = panel[panel["hour"] < grace].copy()
    x["p_den"] = den.predict_proba(x)[:, 1].clip(1e-5, 1 - 1e-5)
    x["p_num"] = num.predict_proba(x)[:, 1].clip(1e-5, 1 - 1e-5)
    return x


def construct_weight_objects(pred: pd.DataFrame, grace: int):
    no_w = {}
    tx_w = {}
    raw_values = []

    for sid, g in pred.groupby("stay_id", sort=False):
        g = g.sort_values("hour")
        # Guarantee one row/hour. Patients dying before grace have fewer rows;
        # weights only need to exist while they are alive.
        pden = g["p_den"].to_numpy(float)
        pnum = g["p_num"].to_numpy(float)
        surv_den = np.cumprod(1 - pden)
        surv_num = np.cumprod(1 - pnum)
        sw_no = surv_num / surv_den

        qden = 1 - surv_den[-1]
        qnum = 1 - surv_num[-1]
        sw_tx = qnum / max(qden, 1e-8)

        no_w[sid] = sw_no
        tx_w[sid] = float(sw_tx)
        raw_values.extend(sw_no.tolist())
        raw_values.append(float(sw_tx))

    arr = np.asarray(raw_values, float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    lo, hi = np.quantile(arr, [TRUNC_Q_LOW, TRUNC_Q_HIGH])
    return no_w, tx_w, float(lo), float(hi), arr


def init_time_map(panel: pd.DataFrame, exposure_col: str):
    return (
        panel.groupby("stay_id")[exposure_col]
        .first()
        .to_dict()
    )


def death_time_map(panel: pd.DataFrame):
    return panel.groupby("stay_id")["death_hours"].first().to_dict()


def make_clones(panel: pd.DataFrame, exposure_col: str, grace: int):
    init_map = init_time_map(panel, exposure_col)
    death_map = death_time_map(panel)
    rows = []

    for sid in panel["stay_id"].drop_duplicates():
        init_h = init_map.get(sid, np.nan)
        death_h = death_map.get(sid, np.nan)
        if pd.notna(init_h) and init_h > grace:
            init_h_strategy = np.nan
        else:
            init_h_strategy = init_h

        for arm in (0, 1):
            death_end = death_h if pd.notna(death_h) else np.inf

            if arm == 0:
                artificial = (
                    init_h_strategy if pd.notna(init_h_strategy) else np.inf
                )
            else:
                artificial = (
                    np.inf if pd.notna(init_h_strategy) else float(grace)
                )

            end = min(FOLLOWUP_H, death_end, artificial)
            event = int(
                pd.notna(death_h)
                and death_h <= FOLLOWUP_H
                and death_h <= artificial
            )
            if event:
                end = death_h

            rows.append(
                {
                    "stay_id": sid,
                    "arm": arm,
                    "end_h": float(max(end, 1e-6)),
                    "event": event,
                    "grace": grace,
                }
            )
    return pd.DataFrame(rows)


def w_at(sid, arm, t, no_w, tx_w, lo, hi):
    if arm == 1:
        w = 1.0 if t <= 0 else tx_w.get(sid, 1.0)
        # For treatment arm we only apply the stabilized eventual-initiation
        # weight after the grace period.
        if t < 1:
            w = 1.0
    else:
        arr = no_w.get(sid)
        if arr is None or len(arr) == 0 or t <= 0:
            w = 1.0
        else:
            idx = min(max(int(math.floor(t)), 0), len(arr) - 1)
            w = arr[idx]
    return float(np.clip(w, lo, hi))


def weighted_km(clones, no_w, tx_w, lo, hi, arm, grace):
    d = clones[clones["arm"] == arm].copy()
    times = np.sort(d.loc[d["event"] == 1, "end_h"].unique())

    surv = 1.0
    for t in times:
        risk = d[d["end_h"] >= t]
        if len(risk) == 0:
            continue

        yw = sum(
            w_at(r.stay_id, arm, t, no_w, tx_w, lo, hi)
            if not (arm == 1 and t < grace)
            else 1.0
            for r in risk.itertuples(index=False)
        )
        ev = d[(d["event"] == 1) & np.isclose(d["end_h"], t)]
        dw = sum(
            w_at(r.stay_id, arm, t, no_w, tx_w, lo, hi)
            if not (arm == 1 and t < grace)
            else 1.0
            for r in ev.itertuples(index=False)
        )

        if yw > 0:
            surv *= 1 - min(max(dw / yw, 0.0), 1.0)

    return 1 - surv


def fit_trial(panel: pd.DataFrame, exposure_col: str, grace: int, details=False):
    haz = hazard_dataset(panel, exposure_col, grace)
    hazfit, den, num = fit_hazard_models(haz)
    pred = predict_all(panel, grace, den, num)
    no_w, tx_w, lo, hi, raw_w = construct_weight_objects(pred, grace)
    clones = make_clones(panel, exposure_col, grace)

    r0 = weighted_km(clones, no_w, tx_w, lo, hi, arm=0, grace=grace)
    r1 = weighted_km(clones, no_w, tx_w, lo, hi, arm=1, grace=grace)

    out = {
        "risk_no_initiation": r0,
        "risk_initiation": r1,
        "risk_difference": r1 - r0,
        "risk_ratio": r1 / r0 if r0 > 0 else np.nan,
        "initiation_events": int(haz["y_init"].sum()),
        "person_hours": len(haz),
        "weight_raw_max": float(np.max(raw_w)),
        "weight_raw_p99": float(np.quantile(raw_w, .99)),
        "trunc_lo": lo,
        "trunc_hi": hi,
        "pred_hazard_min": float(hazfit["p_den"].min()),
        "pred_hazard_max": float(hazfit["p_den"].max()),
        "pred_hazard_mean": float(hazfit["p_den"].mean()),
    }
    if details:
        out.update(
            {
                "hazfit": hazfit,
                "pred": pred,
                "no_w": no_w,
                "tx_w": tx_w,
                "clones": clones,
            }
        )
    return out


def duplicate_bootstrap_panel(panel: pd.DataFrame, rng):
    ids = panel["stay_id"].drop_duplicates().to_numpy()
    sample = rng.choice(ids, size=len(ids), replace=True)

    pieces = []
    for new_id, old_id in enumerate(sample, start=1):
        g = panel[panel["stay_id"] == old_id].copy()
        g["stay_id"] = new_id
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def bootstrap_trial(panel, exposure_col, grace, reps, seed):
    rng = np.random.default_rng(seed)
    est = []
    for b in range(reps):
        boot = duplicate_bootstrap_panel(panel, rng)
        try:
            x = fit_trial(boot, exposure_col, grace, details=False)
            est.append(
                [
                    x["risk_no_initiation"],
                    x["risk_initiation"],
                    x["risk_difference"],
                    x["risk_ratio"],
                ]
            )
        except Exception:
            continue
        if (b + 1) % 20 == 0:
            print(f"      bootstrap {b+1}/{reps}")
    return np.asarray(est, float)


def percentile_ci(arr, col):
    if arr.size == 0:
        return (np.nan, np.nan)
    x = arr[:, col]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    return tuple(np.quantile(x, [0.025, 0.975]))


def ess(w):
    x = np.asarray(w, float)
    x = x[np.isfinite(x) & (x > 0)]
    return (x.sum() ** 2) / np.sum(x ** 2) if len(x) else np.nan


def final_weights(fit, panel, grace):
    rows = []
    for sid in panel["stay_id"].drop_duplicates():
        nw = fit["no_w"].get(sid, np.array([1.0]))
        rows.append(
            {
                "stay_id": sid,
                "w_no": float(
                    np.clip(nw[-1], fit["trunc_lo"], fit["trunc_hi"])
                ),
                "w_tx": float(
                    np.clip(
                        fit["tx_w"].get(sid, 1.0),
                        fit["trunc_lo"],
                        fit["trunc_hi"],
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


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


def dynamic_balance(panel, fit, exposure_col, grace):
    """
    Diagnostic at the end of the grace period among patients alive at grace:
      strategy-1 adherers = actually initiated by grace
      strategy-0 adherers = not initiated by grace
    Uses final stabilized artificial-censoring weights.
    """
    hour = grace - 1
    x = panel[panel["hour"] == hour].copy()
    init = pd.to_numeric(x[exposure_col], errors="coerce")
    alive = x["death_hours"].isna() | (x["death_hours"] > grace)
    x = x[alive].copy()
    init = pd.to_numeric(x[exposure_col], errors="coerce")

    x["arm_obs"] = ((init.notna()) & (init <= grace)).astype(int)
    fw = final_weights(fit, panel, grace)
    x = x.merge(fw, on="stay_id", how="left")

    tx = x[x["arm_obs"] == 1]
    no = x[x["arm_obs"] == 0]

    rows = []
    for c in DYNAMIC_NUM + DYNAMIC_BIN:
        if c not in x.columns:
            continue
        s = weighted_smd(
            pd.to_numeric(tx[c], errors="coerce"),
            tx["w_tx"],
            pd.to_numeric(no[c], errors="coerce"),
            no["w_no"],
        )
        rows.append(
            {
                "grace_hours": grace,
                "variable": c,
                "smd_end_of_grace": s,
                "abs_smd_end_of_grace": abs(s) if pd.notna(s) else np.nan,
                "treated_adherers_alive_n": len(tx),
                "untreated_adherers_alive_n": len(no),
            }
        )
    return pd.DataFrame(rows)


def run_dynamic_primary(panel):
    primary_rows = []
    weight_rows = []
    balance_parts = []
    model_rows = []
    fits = {}

    for grace in (6, 12, 24):
        print(f"\n=== Dynamic CCW: grace period {grace} h ===")
        fit = fit_trial(panel, "any_init_h", grace, details=True)
        fits[grace] = fit

        reps = BOOTSTRAP_REPS[grace]
        print(f"Bootstrapping {reps} reps...")
        boot = bootstrap_trial(
            panel, "any_init_h", grace, reps, RANDOM_SEED + grace
        )
        c0 = percentile_ci(boot, 0)
        c1 = percentile_ci(boot, 1)
        crd = percentile_ci(boot, 2)
        crr = percentile_ci(boot, 3)

        actual_treated = (
            pd.to_numeric(panel.groupby("stay_id")["any_init_h"].first(), errors="coerce")
            .le(grace)
            .sum()
        )

        primary_rows.append(
            {
                "grace_hours": grace,
                "n": panel["stay_id"].nunique(),
                "actual_initiated_by_grace_n": int(actual_treated),
                "risk_no_initiation": fit["risk_no_initiation"],
                "risk_no_ci_low": c0[0],
                "risk_no_ci_high": c0[1],
                "risk_initiation": fit["risk_initiation"],
                "risk_init_ci_low": c1[0],
                "risk_init_ci_high": c1[1],
                "risk_difference": fit["risk_difference"],
                "rd_ci_low": crd[0],
                "rd_ci_high": crd[1],
                "risk_ratio": fit["risk_ratio"],
                "rr_ci_low": crr[0],
                "rr_ci_high": crr[1],
                "bootstrap_success_n": len(boot),
                "estimand": f"initiate any systemic corticosteroid within {grace}h vs no initiation within {grace}h; no treatment restriction after grace",
            }
        )

        fw = final_weights(fit, panel, grace)
        for name in ("w_no", "w_tx"):
            w = fw[name].to_numpy(float)
            weight_rows.append(
                {
                    "grace_hours": grace,
                    "arm_weight": name,
                    "mean": np.mean(w),
                    "sd": np.std(w, ddof=1),
                    "p01": np.quantile(w, .01),
                    "p50": np.quantile(w, .50),
                    "p99": np.quantile(w, .99),
                    "max": np.max(w),
                    "ess": ess(w),
                    "raw_weight_p99": fit["weight_raw_p99"],
                    "raw_weight_max": fit["weight_raw_max"],
                    "truncation_low": fit["trunc_lo"],
                    "truncation_high": fit["trunc_hi"],
                }
            )

        balance_parts.append(dynamic_balance(panel, fit, "any_init_h", grace))
        model_rows.append(
            {
                "analysis": f"any_steroid_{grace}h",
                "grace_hours": grace,
                "person_hours": fit["person_hours"],
                "initiation_events": fit["initiation_events"],
                "mean_predicted_hourly_hazard": fit["pred_hazard_mean"],
                "min_predicted_hourly_hazard": fit["pred_hazard_min"],
                "max_predicted_hourly_hazard": fit["pred_hazard_max"],
                "weight_raw_p99": fit["weight_raw_p99"],
                "weight_raw_max": fit["weight_raw_max"],
            }
        )

    pd.DataFrame(primary_rows).to_csv(
        OUT_PRIMARY, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(weight_rows).to_csv(
        OUT_WEIGHT, index=False, encoding="utf-8-sig"
    )
    pd.concat(balance_parts, ignore_index=True).sort_values(
        ["grace_hours", "abs_smd_end_of_grace"],
        ascending=[True, False],
    ).to_csv(OUT_BAL, index=False, encoding="utf-8-sig")

    return fits, model_rows


def run_drug_sensitivity(panel, model_rows):
    rows = []
    for exposure_col, label in [
        ("iv_init_h", "IV_systemic_steroid"),
        ("hydrocortisone_init_h", "hydrocortisone"),
    ]:
        print(f"\n=== Exploratory exposure sensitivity: {label} <=24h ===")
        try:
            fit = fit_trial(panel, exposure_col, 24, details=False)
            boot = bootstrap_trial(
                panel,
                exposure_col,
                24,
                DRUG_BOOTSTRAP_REPS,
                RANDOM_SEED + (100 if exposure_col == "iv_init_h" else 200),
            )
            crd = percentile_ci(boot, 2)
            crr = percentile_ci(boot, 3)

            first = pd.to_numeric(
                panel.groupby("stay_id")[exposure_col].first(), errors="coerce"
            )
            treated_n = int(first.le(24).sum())

            rows.append(
                {
                    "exposure_definition": label,
                    "grace_hours": 24,
                    "n": panel["stay_id"].nunique(),
                    "initiated_n": treated_n,
                    "risk_no_initiation": fit["risk_no_initiation"],
                    "risk_initiation": fit["risk_initiation"],
                    "risk_difference": fit["risk_difference"],
                    "rd_ci_low": crd[0],
                    "rd_ci_high": crd[1],
                    "risk_ratio": fit["risk_ratio"],
                    "rr_ci_low": crr[0],
                    "rr_ci_high": crr[1],
                    "bootstrap_success_n": len(boot),
                    "interpretation": "EXPLORATORY exposure-definition sensitivity; other systemic steroids are permitted as co-interventions",
                }
            )
            model_rows.append(
                {
                    "analysis": f"{label}_24h",
                    "grace_hours": 24,
                    "person_hours": fit["person_hours"],
                    "initiation_events": fit["initiation_events"],
                    "mean_predicted_hourly_hazard": fit["pred_hazard_mean"],
                    "min_predicted_hourly_hazard": fit["pred_hazard_min"],
                    "max_predicted_hourly_hazard": fit["pred_hazard_max"],
                    "weight_raw_p99": fit["weight_raw_p99"],
                    "weight_raw_max": fit["weight_raw_max"],
                }
            )
        except Exception as e:
            rows.append(
                {
                    "exposure_definition": label,
                    "grace_hours": 24,
                    "n": panel["stay_id"].nunique(),
                    "initiated_n": np.nan,
                    "risk_no_initiation": np.nan,
                    "risk_initiation": np.nan,
                    "risk_difference": np.nan,
                    "rd_ci_low": np.nan,
                    "rd_ci_high": np.nan,
                    "risk_ratio": np.nan,
                    "rr_ci_low": np.nan,
                    "rr_ci_high": np.nan,
                    "bootstrap_success_n": 0,
                    "interpretation": f"FAILED/TOO SPARSE: {repr(e)}",
                }
            )

    pd.DataFrame(rows).to_csv(OUT_DRUG, index=False, encoding="utf-8-sig")
    return model_rows


def compare_stage6():
    if not STAGE6_PATH.exists() or not OUT_PRIMARY.exists():
        pd.DataFrame(
            [
                {
                    "comparison": "Stage6 file not available",
                    "stage6_rd": np.nan,
                    "stage7_dynamic_24h_rd": np.nan,
                    "stage6_rr": np.nan,
                    "stage7_dynamic_24h_rr": np.nan,
                }
            ]
        ).to_csv(OUT_COMPARE, index=False, encoding="utf-8-sig")
        return

    s6 = pd.read_csv(STAGE6_PATH).iloc[0]
    s7 = pd.read_csv(OUT_PRIMARY)
    s7 = s7[s7.grace_hours == 24].iloc[0]

    pd.DataFrame(
        [
            {
                "comparison": "baseline-only CCW vs time-updated CCW, 24h grace",
                "stage6_rd": s6["risk_difference"],
                "stage7_dynamic_24h_rd": s7["risk_difference"],
                "rd_change_dynamic_minus_baseline": s7["risk_difference"]
                - s6["risk_difference"],
                "stage6_rr": s6["risk_ratio"],
                "stage7_dynamic_24h_rr": s7["risk_ratio"],
                "rr_change_dynamic_minus_baseline": s7["risk_ratio"]
                - s6["risk_ratio"],
                "interpretation": "If harmful estimate attenuates materially after time-updated adjustment, confounding by evolving severity is supported.",
            }
        ]
    ).to_csv(OUT_COMPARE, index=False, encoding="utf-8-sig")


def main():
    print("=" * 98)
    print("Stage 7 — Time-updated Clone-Censor-Weight target trial")
    print("=" * 98)

    try:
        df = load_primary_cohort()
        print(
            f"[COHORT] primary n={len(df)} | 28d deaths={int(df.death_28d.sum())}"
        )

        build_dynamic_caches(df)
        panel = build_hourly_panel(df)

        # Audit initiation-time completeness.
        ids = panel["stay_id"].nunique()
        any_treated = int(
            pd.to_numeric(panel.groupby("stay_id")["any_init_h"].first(), errors="coerce")
            .le(24)
            .sum()
        )
        print(
            f"[PANEL] patients={ids} | any systemic steroid initiated <=24h={any_treated}"
        )

        fits, model_rows = run_dynamic_primary(panel)
        model_rows = run_drug_sensitivity(panel, model_rows)

        pd.DataFrame(model_rows).to_csv(
            OUT_MODEL, index=False, encoding="utf-8-sig"
        )
        compare_stage6()

        print("\n" + "=" * 98)
        print("STAGE 7 COMPLETE")
        print("=" * 98)
        print("Generated output files:")
        for p in [
            OUT_PRIMARY,
            OUT_WEIGHT,
            OUT_BAL,
            OUT_DRUG,
            OUT_MODEL,
            OUT_COMPARE,
        ]:
            print(" ", p)

        print("\nInterpretation sequence:")
        print("1) Check 6/12/24h RD/RR trend.")
        print("2) Check weight p99/max and ESS.")
        print("3) Check dynamic end-of-grace SMDs.")
        print("4) Compare Stage 6 baseline-only 24h vs Stage 7 dynamic 24h.")
        print("5) Treat IV/hydrocortisone analyses as exploratory only.")
        print("\nDo not turn the 24h result into a clinical conclusion until all 4 checks pass.")

    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Stage 5.5 — Baseline data repair/enrichment before causal modeling

Why this patch exists
---------------------
Stage 5 audit revealed:
1) SpO2 mapping accidentally included alarm-threshold itemids.
2) GCS was 100% missing because MIMIC-IV stores GCS components, not a reliable
   single "GCS total" item for our dynamic lookup.
3) Common labs had unexpectedly high missingness because ED/pre-admission
   labevents can have hadm_id = NULL.
4) Many patients reach respiratory t0 soon after hospital/ICU arrival; MIMIC-IV-ED
   vitals can improve pretreatment coverage when available.

This script rebuilds ONLY baseline covariates. It does NOT estimate treatment effect.

Primary windows:
- ICU/ED vitals: respiratory t0 - 24 h through t0
- labs: respiratory t0 - 24 h through t0
- no post-t0 measurements are used

Outputs to upload:
    outputs/10_STAGE5_5_MISSINGNESS.csv
    outputs/10B_STAGE5_5_UNADJUSTED_SMD.csv
    outputs/10C_STAGE5_5_POSITIVITY.csv
    outputs/10D_STAGE5_5_COVARIATE_SUMMARY.csv
    outputs/10E_STAGE5_5_ITEM_MAP.csv
    outputs/10F_STAGE5_5_REPAIR_DIAGNOSTICS.csv

Keep local:
    outputs/stage5_5_analysis_cohort.parquet
"""

from pathlib import Path
import math
import traceback

import duckdb
import numpy as np
import pandas as pd

import config
from src.utils import connect_duckdb, find_table, try_find_table, source_sql, sql_quote_path

PROJECT_ROOT = Path(config.PROJECT_ROOT)
DATA_ROOT = Path(config.DATA_ROOT)
CACHE_DIR = Path(config.CACHE_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

STAGE4 = OUTPUT_DIR / "stage4_trial_eligible_cohort.parquet"

VITAL_CACHE = CACHE_DIR / "stage5_5_icu_vitals_gcs.parquet"
LAB_CACHE = CACHE_DIR / "stage5_5_labs.parquet"
ED_VITAL_CACHE = CACHE_DIR / "stage5_5_ed_vitals.parquet"

OUT_MISSING = OUTPUT_DIR / "10_STAGE5_5_MISSINGNESS.csv"
OUT_SMD = OUTPUT_DIR / "10B_STAGE5_5_UNADJUSTED_SMD.csv"
OUT_POS = OUTPUT_DIR / "10C_STAGE5_5_POSITIVITY.csv"
OUT_SUMMARY = OUTPUT_DIR / "10D_STAGE5_5_COVARIATE_SUMMARY.csv"
OUT_ITEMS = OUTPUT_DIR / "10E_STAGE5_5_ITEM_MAP.csv"
OUT_DIAG = OUTPUT_DIR / "10F_STAGE5_5_REPAIR_DIAGNOSTICS.csv"
OUT_COHORT = OUTPUT_DIR / "stage5_5_analysis_cohort.parquet"

LOOKBACK_H = 24
FORCE_REBUILD = False

# Exact official MIMIC-IV vital/GCS itemids used here.
ITEMS = {
    220045: "heart_rate",
    220052: "map_art",
    220181: "map_nibp",
    225312: "map_art_alt",
    220210: "respiratory_rate",
    224690: "respiratory_rate_total",
    220277: "spo2",
    223762: "temp_c",
    223761: "temp_f",
    223835: "fio2",
    226512: "weight_kg",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}


def qpath(p: Path) -> str:
    return sql_quote_path(p)


def scalar(con, sql, params=None):
    x = con.execute(sql, params or []).fetchone()[0]
    return int(x or 0)


def smd_cont(x1, x0):
    a = pd.to_numeric(x1, errors="coerce").dropna()
    b = pd.to_numeric(x0, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    den = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if not np.isfinite(den) or den == 0:
        return 0.0 if a.mean() == b.mean() else np.nan
    return (a.mean() - b.mean()) / den


def smd_bin(x1, x0):
    a = pd.to_numeric(x1, errors="coerce").dropna()
    b = pd.to_numeric(x0, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    p1, p0 = a.mean(), b.mean()
    den = math.sqrt((p1*(1-p1)+p0*(1-p0))/2)
    if den == 0:
        return 0.0 if p1 == p0 else np.nan
    return (p1-p0)/den


def fmt_num(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    return f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]"


def fmt_bin(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    n = int((x == 1).sum())
    return f"{n} ({100*n/len(x):.1f}%)"


def find_ed_file(name: str):
    """
    Locate MIMIC-IV-ED whether it is:
      1) inside DATA_ROOT, e.g.
         .../mimic-iv-3.1/mimic-iv-ed-2.2/
      2) a sibling of DATA_ROOT, e.g.
         .../mimic-iv-3.1/
         .../mimic-iv-ed-2.2/

    Recursively looks for name.csv.gz / name.csv / name.parquet.
    """
    roots = [DATA_ROOT, DATA_ROOT.parent]
    candidates = []

    for root in roots:
        if not root.exists():
            continue

        # First, directories explicitly named mimic-iv-ed*
        dirs = []
        if root.is_dir() and "mimic-iv-ed" in root.name.lower():
            dirs.append(root)

        try:
            dirs.extend(
                d for d in root.rglob("*")
                if d.is_dir() and "mimic-iv-ed" in d.name.lower()
            )
        except OSError:
            pass

        # De-duplicate directories.
        seen = set()
        unique_dirs = []
        for d in dirs:
            key = str(d.resolve()).lower()
            if key not in seen:
                seen.add(key)
                unique_dirs.append(d)

        for d in unique_dirs:
            for ext in (".csv.gz", ".csv", ".parquet"):
                candidates.extend(d.rglob(name + ext))

    if not candidates:
        return None

    # Prefer exact basename match and deterministic path.
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            0 if p.name.lower() == (name + ".csv.gz").lower() else
            1 if p.name.lower() == (name + ".parquet").lower() else 2,
            str(p).lower(),
        )
    )
    return candidates[0]


def main():
    print("=" * 90)
    print("Stage 5.5 V3 — Repair/enrich pretreatment covariates (ED path fix)")
    print("=" * 90)

    if not STAGE4.exists():
        raise FileNotFoundError(f"Missing {STAGE4}; run Stage 4 first.")

    chartevents_path = find_table(DATA_ROOT, "icu", "chartevents")
    labevents_path = find_table(DATA_ROOT, "hosp", "labevents")
    d_labitems_path = find_table(DATA_ROOT, "hosp", "d_labitems")
    diagnoses_path = find_table(DATA_ROOT, "hosp", "diagnoses_icd")
    d_dx_path = try_find_table(DATA_ROOT, "hosp", "d_icd_diagnoses")
    admissions_path = find_table(DATA_ROOT, "hosp", "admissions")
    patients_path = find_table(DATA_ROOT, "hosp", "patients")
    icustays_path = find_table(DATA_ROOT, "icu", "icustays")

    edstays_path = find_ed_file("edstays")
    edvitals_path = find_ed_file("vitalsign")

    print(f"[OK] chartevents: {chartevents_path}")
    print(f"[OK] labevents  : {labevents_path}")
    print(f"[ED] edstays    : {edstays_path if edstays_path else 'not found; ED enrichment skipped'}")
    print(f"[ED] vitalsign  : {edvitals_path if edvitals_path else 'not found; ED enrichment skipped'}")

    con = connect_duckdb(PROJECT_ROOT, getattr(config, "DUCKDB_THREADS", 4))

    try:
        s4 = f"read_parquet('{qpath(STAGE4)}')"
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE elig AS
            SELECT *
            FROM {s4}
            WHERE eligible_primary=1
        """)

        n = scalar(con, "SELECT count(*) FROM elig")
        nt = scalar(con, "SELECT count(*) FROM elig WHERE early_steroid_24h=1")
        print(f"[OK] eligible={n}, treated={nt}, untreated={n-nt}")

        # ------------------------------------------------------------------
        # Metadata and robust chronic comorbidity flags.
        # ------------------------------------------------------------------
        admissions = source_sql(admissions_path)
        patients = source_sql(patients_path)
        icustays = source_sql(icustays_path)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE meta AS
            SELECT
                e.stay_id,
                a.admission_type,
                a.admission_location,
                i.first_careunit,
                p.anchor_year_group
            FROM elig e
            LEFT JOIN {admissions} a USING (subject_id, hadm_id)
            LEFT JOIN {patients} p USING (subject_id)
            LEFT JOIN {icustays} i USING (stay_id)
        """)

        dx = source_sql(diagnoses_path)
        if d_dx_path is not None:
            ddx = source_sql(d_dx_path)
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE comorb AS
                SELECT
                    e.hadm_id,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%heart failure%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%cardiomyopathy%'
                             THEN 1 ELSE 0 END) AS heart_failure,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%coronary atherosclerosis%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%ischemic heart disease%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%coronary artery disease%'
                             THEN 1 ELSE 0 END) AS coronary_disease,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%diabetes mellitus%'
                             THEN 1 ELSE 0 END) AS diabetes,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%chronic kidney%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%end stage renal%'
                             THEN 1 ELSE 0 END) AS chronic_kidney_disease,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%cirrhosis%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%chronic hepatitis%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%chronic liver%'
                             THEN 1 ELSE 0 END) AS chronic_liver_disease,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%malignant neoplasm%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%leukemia%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%lymphoma%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%multiple myeloma%'
                             THEN 1 ELSE 0 END) AS malignancy,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%transplant%'
                             THEN 1 ELSE 0 END) AS transplant_history,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%immunodeficiency%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%human immunodeficiency%'
                                  OR lower(coalesce(x.long_title,'')) LIKE '%acquired immunodeficiency%'
                             THEN 1 ELSE 0 END) AS immunodeficiency,
                    max(CASE WHEN lower(coalesce(x.long_title,'')) LIKE '%dementia%'
                             THEN 1 ELSE 0 END) AS dementia
                FROM elig e
                LEFT JOIN {dx} d USING (hadm_id)
                LEFT JOIN {ddx} x
                  ON d.icd_code=x.icd_code AND d.icd_version=x.icd_version
                GROUP BY e.hadm_id
            """)

        else:
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE comorb AS
                SELECT
                    e.hadm_id,
                    0::INTEGER AS heart_failure,
                    0::INTEGER AS coronary_disease,
                    0::INTEGER AS diabetes,
                    0::INTEGER AS chronic_kidney_disease,
                    0::INTEGER AS chronic_liver_disease,
                    0::INTEGER AS malignancy,
                    0::INTEGER AS transplant_history,
                    0::INTEGER AS immunodeficiency,
                    0::INTEGER AS dementia
                FROM elig e
            """)

        # ------------------------------------------------------------------
        # ICU vitals + GCS exact itemids.
        # ------------------------------------------------------------------
        print("[1/7] Building corrected ICU vital/GCS cache...")
        if FORCE_REBUILD and VITAL_CACHE.exists():
            VITAL_CACHE.unlink()

        itemid_list = ",".join(str(x) for x in ITEMS.keys())

        if not VITAL_CACHE.exists():
            ce = source_sql(chartevents_path)
            print("      First run scans selected chartevents itemids only (MIMIC-IV 3.1 compatible; no c.error filter).")
            con.execute(f"""
                COPY (
                    SELECT
                        e.stay_id,
                        c.charttime,
                        c.itemid,
                        c.value,
                        c.valuenum,
                        c.valueuom
                    FROM elig e
                    JOIN {ce} c
                      ON c.stay_id=e.stay_id
                     AND c.charttime BETWEEN e.respiratory_t0 - INTERVAL '{LOOKBACK_H} hours'
                                         AND e.respiratory_t0
                    WHERE c.itemid IN ({itemid_list})
                )
                TO '{qpath(VITAL_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        else:
            print(f"      Reusing {VITAL_CACHE}")

        vit = f"read_parquet('{qpath(VITAL_CACHE)}')"

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE icu_vital AS
            SELECT
                stay_id,
                max(CASE WHEN itemid=220045 AND valuenum BETWEEN 20 AND 250 THEN valuenum END) AS hr_max,
                max(CASE WHEN itemid IN (220210,224690) AND valuenum BETWEEN 2 AND 70 THEN valuenum END) AS rr_max,
                min(CASE WHEN itemid IN (220052,220181,225312) AND valuenum BETWEEN 20 AND 200 THEN valuenum END) AS map_min,
                min(CASE WHEN itemid=220277 AND valuenum BETWEEN 40 AND 100 THEN valuenum END) AS spo2_min,
                max(CASE
                    WHEN itemid=223762 AND valuenum BETWEEN 25 AND 45 THEN valuenum
                    WHEN itemid=223761 AND valuenum BETWEEN 70 AND 115 THEN (valuenum-32)*5/9
                END) AS temp_max_c,
                arg_max(CASE WHEN itemid=226512 AND valuenum BETWEEN 25 AND 350 THEN valuenum END,
                        CASE WHEN itemid=226512 THEN charttime END) AS weight_kg,
                max(CASE
                    WHEN itemid=223835 AND valuenum BETWEEN 0.20 AND 1.0 THEN valuenum*100
                    WHEN itemid=223835 AND valuenum BETWEEN 20 AND 100 THEN valuenum
                END) AS fio2_max_pct,
                arg_max(CASE
                    WHEN itemid=223835 AND valuenum BETWEEN 0.20 AND 1.0 THEN valuenum*100
                    WHEN itemid=223835 AND valuenum BETWEEN 20 AND 100 THEN valuenum
                END, CASE WHEN itemid=223835 THEN charttime END) AS fio2_last_pct,
                arg_max(CASE WHEN itemid=220277 AND valuenum BETWEEN 40 AND 100 THEN valuenum END,
                        CASE WHEN itemid=220277 THEN charttime END) AS spo2_last,
                arg_max(CASE WHEN itemid=220739 AND valuenum BETWEEN 1 AND 4 THEN valuenum END,
                        CASE WHEN itemid=220739 THEN charttime END) AS gcs_eye_last,
                arg_max(CASE
                    WHEN itemid=223900 AND value='No Response-ETT' THEN 0
                    WHEN itemid=223900 AND valuenum BETWEEN 1 AND 5 THEN valuenum
                END, CASE WHEN itemid=223900 THEN charttime END) AS gcs_verbal_last,
                arg_max(CASE WHEN itemid=223901 AND valuenum BETWEEN 1 AND 6 THEN valuenum END,
                        CASE WHEN itemid=223901 THEN charttime END) AS gcs_motor_last
            FROM {vit}
            GROUP BY stay_id
        """)

        # Approximate total only when all components are present and verbal isn't ETT=0.
        con.execute("""
            CREATE OR REPLACE TEMP TABLE icu_vital2 AS
            SELECT *,
                CASE
                    WHEN gcs_eye_last IS NOT NULL
                     AND gcs_motor_last IS NOT NULL
                     AND gcs_verbal_last BETWEEN 1 AND 5
                    THEN gcs_eye_last + gcs_motor_last + gcs_verbal_last
                    ELSE NULL
                END AS gcs_total_components_last,
                CASE
                    WHEN spo2_last IS NOT NULL AND fio2_last_pct BETWEEN 21 AND 100
                    THEN spo2_last/(fio2_last_pct/100.0)
                    ELSE NULL
                END AS sf_ratio_last
            FROM icu_vital
        """)

        # ------------------------------------------------------------------
        # MIMIC-IV-ED vitals enrichment.
        # ------------------------------------------------------------------
        print("[2/7] Adding ED pretreatment vitals when available...")
        ed_available = bool(edstays_path and edvitals_path)
        if ed_available:
            if FORCE_REBUILD and ED_VITAL_CACHE.exists():
                ED_VITAL_CACHE.unlink()
            if not ED_VITAL_CACHE.exists():
                eds = source_sql(edstays_path)
                edv = source_sql(edvitals_path)
                con.execute(f"""
                    COPY (
                        SELECT
                            e.stay_id AS icu_stay_id,
                            v.charttime,
                            v.temperature,
                            v.heartrate,
                            v.resprate,
                            v.o2sat,
                            v.sbp,
                            v.dbp
                        FROM elig e
                        JOIN {eds} s
                          ON s.subject_id=e.subject_id
                         AND s.hadm_id=e.hadm_id
                        JOIN {edv} v
                          ON v.subject_id=s.subject_id
                         AND v.stay_id=s.stay_id
                         AND v.charttime BETWEEN e.respiratory_t0 - INTERVAL '{LOOKBACK_H} hours'
                                             AND e.respiratory_t0
                    )
                    TO '{qpath(ED_VITAL_CACHE)}'
                    (FORMAT PARQUET, COMPRESSION ZSTD)
                """)
            edv_cache = f"read_parquet('{qpath(ED_VITAL_CACHE)}')"
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE ed_vital AS
                SELECT
                    icu_stay_id AS stay_id,
                    max(CASE WHEN heartrate BETWEEN 20 AND 250 THEN heartrate END) AS ed_hr_max,
                    max(CASE WHEN resprate BETWEEN 2 AND 70 THEN resprate END) AS ed_rr_max,
                    min(CASE WHEN o2sat BETWEEN 40 AND 100 THEN o2sat END) AS ed_spo2_min,
                    max(CASE
                        WHEN temperature BETWEEN 25 AND 45 THEN temperature
                        WHEN temperature BETWEEN 70 AND 115 THEN (temperature-32)*5/9
                    END) AS ed_temp_max_c,
                    min(CASE
                        WHEN sbp BETWEEN 30 AND 300
                         AND dbp BETWEEN 10 AND 200
                        THEN (sbp + 2*dbp)/3.0
                    END) AS ed_map_est
                FROM {edv_cache}
                GROUP BY icu_stay_id
            """)
        else:
            con.execute("""
                CREATE OR REPLACE TEMP TABLE ed_vital AS
                SELECT stay_id,
                       NULL::DOUBLE AS ed_hr_max,
                       NULL::DOUBLE AS ed_rr_max,
                       NULL::DOUBLE AS ed_spo2_min,
                       NULL::DOUBLE AS ed_temp_max_c,
                       NULL::DOUBLE AS ed_map_est
                FROM elig
            """)

        # ------------------------------------------------------------------
        # Labs: include same-admission rows PLUS hadm_id IS NULL rows from ED,
        # matched by subject and pre-t0 time window.
        # ------------------------------------------------------------------
        print("[3/7] Building repaired lab cache (captures ED/null-hadm labs)...")
        dli = source_sql(d_labitems_path)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE lab_item_map AS
            SELECT DISTINCT
                itemid, label, fluid,
                CASE
                    WHEN lower(trim(label))='lactate' THEN 'lactate'
                    WHEN lower(trim(label))='creatinine' THEN 'creatinine'
                    WHEN lower(trim(label))='urea nitrogen' THEN 'bun'
                    WHEN lower(trim(label))='white blood cells' THEN 'wbc'
                    WHEN lower(trim(label))='hemoglobin' THEN 'hemoglobin'
                    WHEN lower(trim(label))='platelet count' THEN 'platelet'
                    WHEN lower(trim(label))='sodium' THEN 'sodium'
                    WHEN lower(trim(label))='potassium' THEN 'potassium'
                    WHEN lower(trim(label))='bicarbonate' THEN 'bicarbonate'
                    WHEN lower(trim(label))='glucose' THEN 'glucose'
                    WHEN lower(trim(label)) IN ('bilirubin, total','bilirubin total') THEN 'bilirubin_total'
                    WHEN lower(trim(label))='albumin' THEN 'albumin'
                    WHEN lower(label) LIKE '%c-reactive protein%' THEN 'crp'
                    WHEN lower(trim(label))='ph' THEN 'ph'
                    WHEN lower(trim(label))='po2' THEN 'po2'
                    WHEN lower(trim(label))='pco2' THEN 'pco2'
                    WHEN lower(label) LIKE '%absolute neutrophil%'
                      OR lower(label) LIKE '%neutrophils, absolute%' THEN 'neutrophils_abs'
                    WHEN lower(label) LIKE '%absolute lymphocyte%'
                      OR lower(label) LIKE '%lymphocytes, absolute%' THEN 'lymphocytes_abs'
                    ELSE NULL
                END AS target
            FROM {dli}
            WHERE lower(coalesce(fluid,'')) IN ('blood','serum','plasma')
              AND (
                    lower(trim(label)) IN (
                        'lactate','creatinine','urea nitrogen','white blood cells',
                        'hemoglobin','platelet count','sodium','potassium',
                        'bicarbonate','glucose','bilirubin, total','bilirubin total',
                        'albumin','ph','po2','pco2'
                    )
                 OR lower(label) LIKE '%c-reactive protein%'
                 OR lower(label) LIKE '%absolute neutrophil%'
                 OR lower(label) LIKE '%neutrophils, absolute%'
                 OR lower(label) LIKE '%absolute lymphocyte%'
                 OR lower(label) LIKE '%lymphocytes, absolute%'
              )
        """)

        if FORCE_REBUILD and LAB_CACHE.exists():
            LAB_CACHE.unlink()

        if not LAB_CACHE.exists():
            labs = source_sql(labevents_path)
            print("      First run scans selected labevents in the pre-t0 window.")
            con.execute(f"""
                COPY (
                    SELECT
                        e.stay_id,
                        l.hadm_id AS lab_hadm_id,
                        l.charttime,
                        m.target,
                        m.itemid,
                        m.label,
                        l.valuenum,
                        l.valueuom
                    FROM elig e
                    JOIN {labs} l
                      ON l.subject_id=e.subject_id
                     AND (l.hadm_id=e.hadm_id OR l.hadm_id IS NULL)
                     AND l.charttime BETWEEN e.respiratory_t0 - INTERVAL '{LOOKBACK_H} hours'
                                         AND e.respiratory_t0
                    JOIN lab_item_map m USING (itemid)
                    WHERE l.valuenum IS NOT NULL
                )
                TO '{qpath(LAB_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        else:
            print(f"      Reusing {LAB_CACHE}")

        lab = f"read_parquet('{qpath(LAB_CACHE)}')"

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE lab_clean AS
            SELECT *,
                CASE
                    WHEN target='lactate' AND valuenum BETWEEN 0.1 AND 30 THEN valuenum
                    WHEN target='creatinine' AND valuenum BETWEEN 0.1 AND 20 THEN valuenum
                    WHEN target='bun' AND valuenum BETWEEN 1 AND 300 THEN valuenum
                    WHEN target='wbc' AND valuenum BETWEEN 0.1 AND 100 THEN valuenum
                    WHEN target='hemoglobin' AND valuenum BETWEEN 2 AND 25 THEN valuenum
                    WHEN target='platelet' AND valuenum BETWEEN 1 AND 2000 THEN valuenum
                    WHEN target='sodium' AND valuenum BETWEEN 100 AND 180 THEN valuenum
                    WHEN target='potassium' AND valuenum BETWEEN 1 AND 10 THEN valuenum
                    WHEN target='bicarbonate' AND valuenum BETWEEN 3 AND 60 THEN valuenum
                    WHEN target='glucose' AND valuenum BETWEEN 20 AND 1000 THEN valuenum
                    WHEN target='bilirubin_total' AND valuenum BETWEEN 0.05 AND 80 THEN valuenum
                    WHEN target='albumin' AND valuenum BETWEEN 0.5 AND 6 THEN valuenum
                    WHEN target='crp' AND valuenum BETWEEN 0 AND 1000 THEN valuenum
                    WHEN target='ph' AND valuenum BETWEEN 6.5 AND 8.0 THEN valuenum
                    WHEN target='po2' AND valuenum BETWEEN 10 AND 700 THEN valuenum
                    WHEN target='pco2' AND valuenum BETWEEN 10 AND 200 THEN valuenum
                    WHEN target='neutrophils_abs' AND valuenum BETWEEN 0 AND 100 THEN valuenum
                    WHEN target='lymphocytes_abs' AND valuenum BETWEEN 0.01 AND 30 THEN valuenum
                    ELSE NULL
                END AS v
            FROM {lab}
        """)

        con.execute("""
            CREATE OR REPLACE TEMP TABLE lab_summary AS
            SELECT
                stay_id,
                max(CASE WHEN target='lactate' THEN v END) AS lactate_max,
                arg_max(CASE WHEN target='creatinine' THEN v END, CASE WHEN target='creatinine' THEN charttime END) AS creatinine_last,
                arg_max(CASE WHEN target='bun' THEN v END, CASE WHEN target='bun' THEN charttime END) AS bun_last,
                arg_max(CASE WHEN target='wbc' THEN v END, CASE WHEN target='wbc' THEN charttime END) AS wbc_last,
                arg_max(CASE WHEN target='hemoglobin' THEN v END, CASE WHEN target='hemoglobin' THEN charttime END) AS hemoglobin_last,
                arg_max(CASE WHEN target='platelet' THEN v END, CASE WHEN target='platelet' THEN charttime END) AS platelet_last,
                arg_max(CASE WHEN target='sodium' THEN v END, CASE WHEN target='sodium' THEN charttime END) AS sodium_last,
                arg_max(CASE WHEN target='potassium' THEN v END, CASE WHEN target='potassium' THEN charttime END) AS potassium_last,
                arg_max(CASE WHEN target='bicarbonate' THEN v END, CASE WHEN target='bicarbonate' THEN charttime END) AS bicarbonate_last,
                arg_max(CASE WHEN target='glucose' THEN v END, CASE WHEN target='glucose' THEN charttime END) AS glucose_last,
                arg_max(CASE WHEN target='bilirubin_total' THEN v END, CASE WHEN target='bilirubin_total' THEN charttime END) AS bilirubin_last,
                arg_max(CASE WHEN target='albumin' THEN v END, CASE WHEN target='albumin' THEN charttime END) AS albumin_last,
                max(CASE WHEN target='crp' THEN v END) AS crp_max,
                min(CASE WHEN target='ph' THEN v END) AS ph_min,
                arg_max(CASE WHEN target='po2' THEN v END, CASE WHEN target='po2' THEN charttime END) AS po2_last,
                arg_max(CASE WHEN target='pco2' THEN v END, CASE WHEN target='pco2' THEN charttime END) AS pco2_last,
                arg_max(CASE WHEN target='neutrophils_abs' THEN v END, CASE WHEN target='neutrophils_abs' THEN charttime END) AS neutrophils_abs_last,
                arg_max(CASE WHEN target='lymphocytes_abs' THEN v END, CASE WHEN target='lymphocytes_abs' THEN charttime END) AS lymphocytes_abs_last,
                max(CASE WHEN lab_hadm_id IS NULL THEN 1 ELSE 0 END) AS has_null_hadm_lab
            FROM lab_clean
            GROUP BY stay_id
        """)

        # ------------------------------------------------------------------
        # Vaso at t0
        # ------------------------------------------------------------------
        print("[4/7] Adding t0 treatment-severity proxies...")
        con.execute("""
            CREATE OR REPLACE TEMP TABLE vaso AS
            SELECT stay_id, shock_active_at_t0,
                   CASE WHEN shock_active_at_t0=1 THEN 1 ELSE 0 END AS vasopressor_any_at_t0
            FROM elig
        """)

        # ------------------------------------------------------------------
        # Assemble repaired cohort.
        # ------------------------------------------------------------------
        print("[5/7] Assembling repaired baseline cohort...")
        con.execute("""
            CREATE OR REPLACE TEMP TABLE repaired AS
            SELECT
                e.*,
                m.admission_type,
                m.admission_location,
                m.first_careunit,
                m.anchor_year_group,

                coalesce(c.heart_failure,0) AS heart_failure,
                coalesce(c.coronary_disease,0) AS coronary_disease,
                coalesce(c.diabetes,0) AS diabetes,
                coalesce(c.chronic_kidney_disease,0) AS chronic_kidney_disease,
                coalesce(c.chronic_liver_disease,0) AS chronic_liver_disease,
                coalesce(c.malignancy,0) AS malignancy,
                coalesce(c.transplant_history,0) AS transplant_history,
                coalesce(c.immunodeficiency,0) AS immunodeficiency,
                coalesce(c.dementia,0) AS dementia,

                greatest(iv.hr_max, ed.ed_hr_max) AS hr_max,
                greatest(iv.rr_max, ed.ed_rr_max) AS rr_max,
                least(iv.map_min, ed.ed_map_est) AS map_min,
                least(iv.spo2_min, ed.ed_spo2_min) AS spo2_min,
                greatest(iv.temp_max_c, ed.ed_temp_max_c) AS temp_max_c,

                iv.weight_kg,
                iv.fio2_max_pct,
                iv.fio2_last_pct,
                iv.spo2_last,
                iv.sf_ratio_last,

                iv.gcs_eye_last,
                iv.gcs_verbal_last,
                iv.gcs_motor_last,
                iv.gcs_total_components_last,

                l.lactate_max,
                l.creatinine_last,
                l.bun_last,
                l.wbc_last,
                l.hemoglobin_last,
                l.platelet_last,
                l.sodium_last,
                l.potassium_last,
                l.bicarbonate_last,
                l.glucose_last,
                l.bilirubin_last,
                l.albumin_last,
                l.crp_max,
                l.ph_min,
                l.po2_last,
                l.pco2_last,
                l.neutrophils_abs_last,
                l.lymphocytes_abs_last,
                CASE
                    WHEN l.neutrophils_abs_last IS NOT NULL
                     AND l.lymphocytes_abs_last > 0
                    THEN l.neutrophils_abs_last/l.lymphocytes_abs_last
                    ELSE NULL
                END AS nlr,
                coalesce(l.has_null_hadm_lab,0) AS has_null_hadm_lab,

                CASE
                    WHEN coalesce(c.transplant_history,0)=1
                      OR coalesce(c.immunodeficiency,0)=1
                    THEN 1 ELSE 0
                END AS major_immunocompromise_proxy,

                CASE
                    WHEN lower(coalesce(m.first_careunit,'')) LIKE '%medical%'
                      OR lower(coalesce(m.first_careunit,'')) IN ('medicine','intensive care unit (icu)')
                    THEN 'medical_mixed'
                    WHEN lower(coalesce(m.first_careunit,'')) LIKE '%coronary%'
                      OR lower(coalesce(m.first_careunit,'')) LIKE '%cardiac%'
                    THEN 'cardiac'
                    WHEN lower(coalesce(m.first_careunit,'')) LIKE '%surgical%'
                      OR lower(coalesce(m.first_careunit,'')) LIKE '%trauma%'
                      OR lower(coalesce(m.first_careunit,'')) LIKE '%surgery%'
                    THEN 'surgical_trauma'
                    WHEN lower(coalesce(m.first_careunit,'')) LIKE '%neuro%'
                    THEN 'neuro'
                    ELSE 'other'
                END AS careunit_group,

                CASE
                    WHEN upper(coalesce(e.race,'')) LIKE '%WHITE%' THEN 'White'
                    WHEN upper(coalesce(e.race,'')) LIKE '%BLACK%' THEN 'Black'
                    WHEN upper(coalesce(e.race,'')) LIKE '%ASIAN%' THEN 'Asian'
                    WHEN upper(coalesce(e.race,'')) LIKE '%HISPANIC%' OR upper(coalesce(e.race,'')) LIKE '%LATINO%' THEN 'Hispanic_Latino'
                    WHEN e.race IS NULL OR upper(e.race) IN ('UNKNOWN','UNABLE TO OBTAIN','PATIENT DECLINED TO ANSWER') THEN 'Unknown'
                    ELSE 'Other'
                END AS race_group
            FROM elig e
            LEFT JOIN meta m USING (stay_id)
            LEFT JOIN comorb c USING (hadm_id)
            LEFT JOIN icu_vital2 iv USING (stay_id)
            LEFT JOIN ed_vital ed USING (stay_id)
            LEFT JOIN lab_summary l USING (stay_id)
        """)

        con.execute(f"""
            COPY (SELECT * FROM repaired ORDER BY stay_id)
            TO '{qpath(OUT_COHORT)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        df = con.execute("SELECT * FROM repaired").fetchdf()
        exposure = "early_steroid_24h"
        t = df[df[exposure] == 1]
        u = df[df[exposure] == 0]

        # ------------------------------------------------------------------
        # Item map
        # ------------------------------------------------------------------
        item_rows = []
        for itemid, target in ITEMS.items():
            item_rows.append({
                "source": "chartevents",
                "target": target,
                "itemid": itemid,
                "note": "exact Stage 5.5 mapping"
            })
        lab_items = con.execute("""
            SELECT 'labevents' AS source, target, itemid,
                   label || ' | fluid=' || coalesce(fluid,'') AS note
            FROM lab_item_map
            WHERE target IS NOT NULL
        """).fetchdf()
        pd.concat([pd.DataFrame(item_rows), lab_items], ignore_index=True).to_csv(
            OUT_ITEMS, index=False, encoding="utf-8-sig"
        )

        numeric_covs = [
            "age","hospital_to_respiratory_t0_h",
            "hr_max","rr_max","map_min","spo2_min","temp_max_c","weight_kg",
            "fio2_max_pct","sf_ratio_last",
            "gcs_motor_last","gcs_total_components_last",
            "lactate_max","creatinine_last","bun_last","wbc_last","hemoglobin_last",
            "platelet_last","sodium_last","potassium_last","bicarbonate_last",
            "glucose_last","bilirubin_last","albumin_last","crp_max","ph_min",
            "po2_last","pco2_last","nlr",
        ]
        binary_covs = [
            "shock_active_at_t0",
            "t0_invasive_device","t0_ventilator_signal","t0_niv","t0_hfnc_qualified",
            "copd_history","asthma_history",
            "heart_failure","coronary_disease","diabetes",
            "chronic_kidney_disease","chronic_liver_disease","malignancy",
            "transplant_history","immunodeficiency","dementia",
            "major_immunocompromise_proxy","pneumonia_top3_dx",
        ]
        categorical_covs = [
            "gender","race_group","respiratory_t0_type",
            "admission_type","admission_location","careunit_group","anchor_year_group",
        ]

        # ------------------------------------------------------------------
        # Missingness
        # ------------------------------------------------------------------
        print("[6/7] Computing repaired missingness/SMD/positivity...")
        miss_rows = []
        for col in numeric_covs + binary_covs + categorical_covs:
            if col not in df.columns:
                continue
            for gname, sub in [("overall",df),("treated",t),("untreated",u)]:
                miss = int(sub[col].isna().sum())
                miss_rows.append({
                    "variable":col,
                    "group":gname,
                    "n":len(sub),
                    "missing_n":miss,
                    "missing_pct":round(100*miss/len(sub),1) if len(sub) else None
                })
        pd.DataFrame(miss_rows).to_csv(OUT_MISSING,index=False,encoding="utf-8-sig")

        # SMD
        smd_rows = []
        for col in numeric_covs:
            if col in df.columns:
                s = smd_cont(t[col],u[col])
                smd_rows.append({
                    "variable":col,"type":"continuous","level":"",
                    "treated_summary":fmt_num(t[col]),"untreated_summary":fmt_num(u[col]),
                    "smd":s,"abs_smd":abs(s) if pd.notna(s) else np.nan
                })
        for col in binary_covs:
            if col in df.columns:
                s = smd_bin(t[col],u[col])
                smd_rows.append({
                    "variable":col,"type":"binary","level":"1",
                    "treated_summary":fmt_bin(t[col]),"untreated_summary":fmt_bin(u[col]),
                    "smd":s,"abs_smd":abs(s) if pd.notna(s) else np.nan
                })
        for col in categorical_covs:
            if col not in df.columns:
                continue
            vals = df[col].astype(object).where(df[col].notna(),"<MISSING>").astype(str)
            for level in vals.value_counts().index:
                tx=(t[col].astype(object).where(t[col].notna(),"<MISSING>").astype(str)==level).astype(float)
                ux=(u[col].astype(object).where(u[col].notna(),"<MISSING>").astype(str)==level).astype(float)
                s=smd_bin(tx,ux)
                smd_rows.append({
                    "variable":col,"type":"categorical_level","level":level,
                    "treated_summary":fmt_bin(tx),"untreated_summary":fmt_bin(ux),
                    "smd":s,"abs_smd":abs(s) if pd.notna(s) else np.nan
                })
        smd_df=pd.DataFrame(smd_rows).sort_values("abs_smd",ascending=False,na_position="last")
        smd_df.to_csv(OUT_SMD,index=False,encoding="utf-8-sig")

        # Summary
        sum_rows=[]
        for col in numeric_covs:
            if col in df.columns:
                sum_rows.append({"variable":col,"type":"continuous","overall":fmt_num(df[col]),"treated":fmt_num(t[col]),"untreated":fmt_num(u[col])})
        for col in binary_covs:
            if col in df.columns:
                sum_rows.append({"variable":col,"type":"binary","overall":fmt_bin(df[col]),"treated":fmt_bin(t[col]),"untreated":fmt_bin(u[col])})
        pd.DataFrame(sum_rows).to_csv(OUT_SUMMARY,index=False,encoding="utf-8-sig")

        # Positivity coarse strata
        pos_rows=[]
        def add_pos(name, series):
            tmp=pd.DataFrame({"stratum":series.astype(object).where(pd.Series(series).notna(),"<MISSING>").astype(str),"A":df[exposure].values})
            g=tmp.groupby("stratum")["A"].agg(["count","sum","mean"]).reset_index()
            for _,r in g.iterrows():
                pos_rows.append({
                    "stratifier":name,"stratum":r["stratum"],"n":int(r["count"]),
                    "treated_n":int(r["sum"]),"treated_pct":round(100*float(r["mean"]),1),
                    "flag":"CAUTION_LOW_TREATED" if int(r["sum"])<20 else ("CAUTION_EXTREME_RATE" if float(r["mean"])<.05 or float(r["mean"])>.8 else "OK")
                })
        add_pos("shock_active_at_t0",df["shock_active_at_t0"].map({0:"no_shock",1:"shock"}))
        add_pos("respiratory_t0_type",df["respiratory_t0_type"])
        add_pos("careunit_group",df["careunit_group"])
        add_pos("major_immunocompromise_proxy",df["major_immunocompromise_proxy"].map({0:"no_major_immunocompromise",1:"major_immunocompromise"}))
        add_pos("age_group",pd.cut(df["age"],[17,49,64,74,200],labels=["18-49","50-64","65-74","75+"]))
        pd.DataFrame(pos_rows).to_csv(OUT_POS,index=False,encoding="utf-8-sig")

        # ------------------------------------------------------------------
        # Repair diagnostics
        # ------------------------------------------------------------------
        print("[7/7] Exporting repair diagnostics...")
        diag=[]
        def diagrow(metric,n_,den=n,notes=""):
            diag.append({"metric":metric,"n":int(n_),"denominator_n":int(den),"pct":round(100*n_/den,1) if den else None,"notes":notes})

        diagrow("ED_vitals_available_on_disk", 1 if ed_available else 0, 1)
        diagrow("patients_with_any_ED_vital", scalar(con,"SELECT count(*) FROM ed_vital WHERE ed_hr_max IS NOT NULL OR ed_rr_max IS NOT NULL OR ed_spo2_min IS NOT NULL OR ed_map_est IS NOT NULL"))
        diagrow("patients_with_null_hadm_pret0_lab", scalar(con,"SELECT count(*) FROM repaired WHERE has_null_hadm_lab=1"))
        diagrow("GCS_motor_available", int(df["gcs_motor_last"].notna().sum()))
        diagrow("GCS_total_components_available", int(df["gcs_total_components_last"].notna().sum()))
        diagrow("SpO2_available_corrected", int(df["spo2_min"].notna().sum()), notes="itemid 220277 only; alarm thresholds excluded")
        diagrow("major_immunocompromise_proxy", int(df["major_immunocompromise_proxy"].sum()))
        diagrow("major_immunocompromise_proxy_treated", int(t["major_immunocompromise_proxy"].sum()), len(t))
        diagrow("major_immunocompromise_proxy_untreated", int(u["major_immunocompromise_proxy"].sum()), len(u))
        pd.DataFrame(diag).to_csv(OUT_DIAG,index=False,encoding="utf-8-sig")

        print("\n=== Stage 5.5 complete ===")
        overall_miss = pd.read_csv(OUT_MISSING)
        overall_miss = overall_miss[overall_miss.group=="overall"].sort_values("missing_pct",ascending=False)
        print("\nTop missing variables:")
        print(overall_miss[["variable","missing_pct"]].head(15).to_string(index=False))

        print("\nTop baseline imbalances:")
        print(smd_df[["variable","level","abs_smd","treated_summary","untreated_summary"]].head(15).to_string(index=False))

        print("\nUpload these 6 small CSVs:")
        for p in [OUT_MISSING,OUT_SMD,OUT_POS,OUT_SUMMARY,OUT_ITEMS,OUT_DIAG]:
            print(" ",p)
        print("\nKeep local only:",OUT_COHORT)
        print("\nNo causal effect has been estimated yet.")

    except Exception:
        print("\n[FAILED] Copy the complete traceback to ChatGPT:\n")
        traceback.print_exc()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()

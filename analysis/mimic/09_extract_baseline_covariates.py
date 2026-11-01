from __future__ import annotations

"""
Stage 5 — Pretreatment covariate extraction + confounding/positivity audit

IMPORTANT
---------
This stage DOES NOT estimate the causal effect of steroids.
It extracts only variables available at/before respiratory time-zero
and audits whether the treated and untreated groups can be made comparable.

Required local file from Stage 4:
    outputs/stage4_trial_eligible_cohort.parquet

Outputs to upload:
    outputs/09_STAGE5_MISSINGNESS.csv
    outputs/09B_STAGE5_UNADJUSTED_SMD.csv
    outputs/09C_STAGE5_POSITIVITY.csv
    outputs/09D_STAGE5_COVARIATE_SUMMARY.csv
    outputs/09E_STAGE5_ITEM_MAP.csv

Keep local, do NOT upload:
    outputs/stage5_analysis_cohort.parquet

First run scans selected rows from:
    icu/chartevents
    hosp/labevents
and caches them to Parquet. Reruns are much faster.
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
VITAL_CACHE = CACHE_DIR / "stage5_pret0_vitals.parquet"
LAB_CACHE = CACHE_DIR / "stage5_pret0_labs.parquet"

OUT_MISSING = OUTPUT_DIR / "09_STAGE5_MISSINGNESS.csv"
OUT_SMD = OUTPUT_DIR / "09B_STAGE5_UNADJUSTED_SMD.csv"
OUT_POS = OUTPUT_DIR / "09C_STAGE5_POSITIVITY.csv"
OUT_SUMMARY = OUTPUT_DIR / "09D_STAGE5_COVARIATE_SUMMARY.csv"
OUT_ITEMS = OUTPUT_DIR / "09E_STAGE5_ITEM_MAP.csv"
OUT_COHORT = OUTPUT_DIR / "stage5_analysis_cohort.parquet"

FORCE_REBUILD_STAGE5_CACHE = False
VITAL_LOOKBACK_H = 6
LAB_LOOKBACK_H = 24


def qpath(p: Path) -> str:
    return sql_quote_path(p)


def scalar(con, sql, params=None):
    x = con.execute(sql, params or []).fetchone()[0]
    return int(x or 0)


def table_exists(con, table_name: str) -> bool:
    return bool(con.execute(
        """SELECT count(*) FROM information_schema.tables
           WHERE lower(table_name)=lower(?)""",
        [table_name],
    ).fetchone()[0])


def pooled_smd_cont(x1: pd.Series, x0: pd.Series):
    a = pd.to_numeric(x1, errors="coerce").dropna()
    b = pd.to_numeric(x0, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if not np.isfinite(sd) or sd == 0:
        return 0.0 if a.mean() == b.mean() else np.nan
    return (a.mean() - b.mean()) / sd


def pooled_smd_binary(x1: pd.Series, x0: pd.Series):
    a = pd.to_numeric(x1, errors="coerce").dropna()
    b = pd.to_numeric(x0, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    p1, p0 = a.mean(), b.mean()
    den = math.sqrt((p1 * (1-p1) + p0 * (1-p0)) / 2.0)
    if den == 0:
        return 0.0 if p1 == p0 else np.nan
    return (p1 - p0) / den


def fmt_num(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    return f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]"


def fmt_bin(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    n = int((x == 1).sum())
    return f"{n} ({100*n/len(x):.1f}%)"


def main():
    print("=" * 88)
    print("Stage 5 — Pretreatment covariates, SMD, missingness, positivity")
    print("=" * 88)

    if not STAGE4.exists():
        raise FileNotFoundError(
            f"Missing {STAGE4}\nRun Stage 4 first."
        )

    chartevents_path = find_table(DATA_ROOT, "icu", "chartevents")
    d_items_path = find_table(DATA_ROOT, "icu", "d_items")
    labevents_path = find_table(DATA_ROOT, "hosp", "labevents")
    d_labitems_path = find_table(DATA_ROOT, "hosp", "d_labitems")
    diagnoses_path = find_table(DATA_ROOT, "hosp", "diagnoses_icd")
    d_dx_path = try_find_table(DATA_ROOT, "hosp", "d_icd_diagnoses")
    admissions_path = find_table(DATA_ROOT, "hosp", "admissions")
    patients_path = find_table(DATA_ROOT, "hosp", "patients")
    icustays_path = find_table(DATA_ROOT, "icu", "icustays")

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
        nc = n - nt
        print(f"[OK] Primary eligible cohort: {n} (treated={nt}, untreated={nc})")

        # -------------------------------------------------------------
        # Metadata / admission covariates
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Comorbidity flags: chronic diagnoses only; no acute severity
        # diagnoses are used as confounders.
        # -------------------------------------------------------------
        print("[1/6] Building chronic comorbidity flags...")
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
            # Conservative code-based fallback.
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE comorb AS
                SELECT
                    e.hadm_id,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^I(09|11|13|25|42|43|50)'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(39891|402|404|425|428)'))
                        THEN 1 ELSE 0 END) AS heart_failure,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^I2[0-5]'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(410|411|412|413|414)'))
                        THEN 1 ELSE 0 END) AS coronary_disease,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^E1[0-4]'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^250'))
                        THEN 1 ELSE 0 END) AS diabetes,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^N1[89]'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(585|586)'))
                        THEN 1 ELSE 0 END) AS chronic_kidney_disease,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^K7[0-7]'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(571|572)'))
                        THEN 1 ELSE 0 END) AS chronic_liver_disease,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(C|D0[0-9]|D3[7-9]|D4[0-8])'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(14|15|16|17|18|19|20)'))
                        THEN 1 ELSE 0 END) AS malignancy,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^Z94'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^V42'))
                        THEN 1 ELSE 0 END) AS transplant_history,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(B2[0-4]|D8[0-4])'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(042|043|044|279)'))
                        THEN 1 ELSE 0 END) AS immunodeficiency,
                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^F0[0-3]'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(290|2941|3310)'))
                        THEN 1 ELSE 0 END) AS dementia
                FROM elig e
                LEFT JOIN {dx} d USING (hadm_id)
                GROUP BY e.hadm_id
            """)

        # -------------------------------------------------------------
        # Identify ICU vital itemids dynamically from d_items.
        # -------------------------------------------------------------
        print("[2/6] Mapping vital itemids...")
        di = source_sql(d_items_path)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE vital_item_map AS
            SELECT DISTINCT
                itemid,
                label,
                CASE
                    WHEN lower(trim(label))='heart rate' THEN 'heart_rate'
                    WHEN lower(trim(label)) LIKE 'respiratory rate%' THEN 'respiratory_rate'
                    WHEN lower(label) LIKE '%blood pressure mean%'
                      OR lower(label) LIKE '%arterial pressure mean%' THEN 'map'
                    WHEN lower(label) LIKE '%o2 saturation pulseoxymetry%'
                      OR lower(trim(label))='spo2' THEN 'spo2'
                    WHEN lower(label) LIKE '%temperature celsius%' THEN 'temp_c'
                    WHEN lower(label) LIKE '%temperature fahrenheit%' THEN 'temp_f'
                    WHEN lower(label) LIKE '%gcs%total%' THEN 'gcs_total'
                    WHEN lower(label) LIKE '%admission weight%kg%'
                      OR lower(trim(label))='weight (kg)' THEN 'weight_kg'
                    WHEN itemid=223835 OR lower(trim(label))='fio2' THEN 'fio2'
                    ELSE NULL
                END AS target
            FROM {di}
            WHERE
                lower(trim(label))='heart rate'
                OR lower(trim(label)) LIKE 'respiratory rate%'
                OR lower(label) LIKE '%blood pressure mean%'
                OR lower(label) LIKE '%arterial pressure mean%'
                OR lower(label) LIKE '%o2 saturation pulseoxymetry%'
                OR lower(trim(label))='spo2'
                OR lower(label) LIKE '%temperature celsius%'
                OR lower(label) LIKE '%temperature fahrenheit%'
                OR lower(label) LIKE '%gcs%total%'
                OR lower(label) LIKE '%admission weight%kg%'
                OR lower(trim(label))='weight (kg)'
                OR itemid=223835
                OR lower(trim(label))='fio2'
        """)

        # -------------------------------------------------------------
        # Cache selected pre-t0 vitals.
        # -------------------------------------------------------------
        print("[3/6] Building/reusing pre-t0 vital cache...")
        if FORCE_REBUILD_STAGE5_CACHE and VITAL_CACHE.exists():
            VITAL_CACHE.unlink()

        if not VITAL_CACHE.exists():
            ce = source_sql(chartevents_path)
            print("      First run: scanning selected chartevents; this may take a while.")
            con.execute(f"""
                COPY (
                    SELECT
                        e.stay_id,
                        c.charttime,
                        m.target,
                        m.itemid,
                        m.label,
                        c.valuenum,
                        c.valueuom,
                        CASE
                            WHEN m.target='temp_f' AND c.valuenum BETWEEN 70 AND 115
                                THEN (c.valuenum - 32.0) * 5.0/9.0
                            WHEN m.target='temp_c' AND c.valuenum BETWEEN 25 AND 45
                                THEN c.valuenum
                            ELSE NULL
                        END AS temp_c_norm,
                        CASE
                            WHEN m.target='fio2' AND c.valuenum BETWEEN 0.20 AND 1.0
                                THEN c.valuenum * 100.0
                            WHEN m.target='fio2' AND c.valuenum BETWEEN 20 AND 100
                                THEN c.valuenum
                            ELSE NULL
                        END AS fio2_percent_norm
                    FROM elig e
                    JOIN {ce} c
                      ON c.stay_id=e.stay_id
                     AND c.charttime BETWEEN e.respiratory_t0 - INTERVAL '{VITAL_LOOKBACK_H} hours'
                                         AND e.respiratory_t0
                    JOIN vital_item_map m USING (itemid)
                    WHERE c.valuenum IS NOT NULL
                )
                TO '{qpath(VITAL_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        else:
            print(f"      Reusing {VITAL_CACHE}")

        vit = f"read_parquet('{qpath(VITAL_CACHE)}')"

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE vital_summary AS
            SELECT
                stay_id,
                max(CASE WHEN target='heart_rate' AND valuenum BETWEEN 20 AND 250 THEN valuenum END) AS hr_max,
                arg_max(CASE WHEN target='heart_rate' AND valuenum BETWEEN 20 AND 250 THEN valuenum END,
                        CASE WHEN target='heart_rate' THEN charttime END) AS hr_last,
                max(CASE WHEN target='respiratory_rate' AND valuenum BETWEEN 2 AND 80 THEN valuenum END) AS rr_max,
                min(CASE WHEN target='map' AND valuenum BETWEEN 20 AND 200 THEN valuenum END) AS map_min,
                min(CASE WHEN target='spo2' AND valuenum BETWEEN 40 AND 100 THEN valuenum END) AS spo2_min,
                arg_max(CASE WHEN target='spo2' AND valuenum BETWEEN 40 AND 100 THEN valuenum END,
                        CASE WHEN target='spo2' THEN charttime END) AS spo2_last,
                max(CASE WHEN target IN ('temp_f','temp_c') THEN temp_c_norm END) AS temp_max_c,
                min(CASE WHEN target='gcs_total' AND valuenum BETWEEN 3 AND 15 THEN valuenum END) AS gcs_min,
                arg_max(CASE WHEN target='weight_kg' AND valuenum BETWEEN 25 AND 350 THEN valuenum END,
                        CASE WHEN target='weight_kg' THEN charttime END) AS weight_kg,
                max(CASE WHEN target='fio2' THEN fio2_percent_norm END) AS fio2_max_pct,
                arg_max(CASE WHEN target='fio2' THEN fio2_percent_norm END,
                        CASE WHEN target='fio2' THEN charttime END) AS fio2_last_pct
            FROM {vit}
            GROUP BY stay_id
        """)

        # -------------------------------------------------------------
        # Identify lab itemids dynamically.
        # -------------------------------------------------------------
        print("[4/6] Mapping laboratory itemids...")
        dli = source_sql(d_labitems_path)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE lab_item_map AS
            SELECT DISTINCT
                itemid,
                label,
                fluid,
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

        if FORCE_REBUILD_STAGE5_CACHE and LAB_CACHE.exists():
            LAB_CACHE.unlink()

        if not LAB_CACHE.exists():
            labs = source_sql(labevents_path)
            print("      First run: scanning selected labevents; this is another slow step.")
            con.execute(f"""
                COPY (
                    SELECT
                        e.stay_id,
                        l.charttime,
                        m.target,
                        m.itemid,
                        m.label,
                        l.valuenum,
                        l.valueuom
                    FROM elig e
                    JOIN {labs} l
                      ON l.subject_id=e.subject_id
                     AND l.hadm_id=e.hadm_id
                     AND l.charttime BETWEEN e.respiratory_t0 - INTERVAL '{LAB_LOOKBACK_H} hours'
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

        # Clean plausible values and summarize.
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
                arg_max(CASE WHEN target='lymphocytes_abs' THEN v END, CASE WHEN target='lymphocytes_abs' THEN charttime END) AS lymphocytes_abs_last
            FROM lab_clean
            GROUP BY stay_id
        """)

        # Active vasopressor burden at t0 if Stage-1 table is available.
        if table_exists(con, "vasopressor_events"):
            con.execute("""
                CREATE OR REPLACE TEMP TABLE vaso_baseline AS
                SELECT
                    e.stay_id,
                    count(DISTINCT v.target_name) AS vasopressor_agents_at_t0,
                    max(CASE
                        WHEN v.target_name='norepinephrine'
                         AND lower(coalesce(v.rateuom,'')) LIKE '%mcg/kg/min%'
                         AND v.rate BETWEEN 0 AND 5
                        THEN v.rate END) AS norepinephrine_rate_at_t0
                FROM elig e
                LEFT JOIN vasopressor_events v
                  ON v.stay_id=e.stay_id
                 AND v.starttime <= e.respiratory_t0
                 AND coalesce(v.endtime, v.starttime + INTERVAL '1 minute') >= e.respiratory_t0
                GROUP BY e.stay_id
            """)
        else:
            con.execute("""
                CREATE OR REPLACE TEMP TABLE vaso_baseline AS
                SELECT stay_id, 0 AS vasopressor_agents_at_t0,
                       cast(NULL AS DOUBLE) AS norepinephrine_rate_at_t0
                FROM elig
            """)

        # -------------------------------------------------------------
        # Assemble analysis-ready pretreatment cohort.
        # -------------------------------------------------------------
        print("[5/6] Assembling pretreatment analysis cohort...")
        con.execute("""
            CREATE OR REPLACE TEMP TABLE stage5 AS
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
                v.hr_max,
                v.hr_last,
                v.rr_max,
                v.map_min,
                v.spo2_min,
                v.spo2_last,
                v.temp_max_c,
                v.gcs_min,
                v.weight_kg,
                v.fio2_max_pct,
                v.fio2_last_pct,
                CASE
                    WHEN v.spo2_last IS NOT NULL AND v.fio2_last_pct BETWEEN 21 AND 100
                    THEN v.spo2_last / (v.fio2_last_pct/100.0)
                    ELSE NULL
                END AS sf_ratio_last,
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
                    THEN l.neutrophils_abs_last / l.lymphocytes_abs_last
                    ELSE NULL
                END AS nlr,
                coalesce(vb.vasopressor_agents_at_t0,0) AS vasopressor_agents_at_t0,
                vb.norepinephrine_rate_at_t0
            FROM elig e
            LEFT JOIN meta m USING (stay_id)
            LEFT JOIN comorb c USING (hadm_id)
            LEFT JOIN vital_summary v USING (stay_id)
            LEFT JOIN lab_summary l USING (stay_id)
            LEFT JOIN vaso_baseline vb USING (stay_id)
        """)

        con.execute(f"""
            COPY (SELECT * FROM stage5 ORDER BY stay_id)
            TO '{qpath(OUT_COHORT)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        df = con.execute("SELECT * FROM stage5").fetchdf()

        # -------------------------------------------------------------
        # Export item map so we can audit exactly what MIMIC labels were used.
        # -------------------------------------------------------------
        vi = con.execute("""
            SELECT 'chartevents' AS source, target, itemid, label,
                   cast(NULL AS VARCHAR) AS fluid
            FROM vital_item_map WHERE target IS NOT NULL
        """).fetchdf()
        li = con.execute("""
            SELECT 'labevents' AS source, target, itemid, label, fluid
            FROM lab_item_map WHERE target IS NOT NULL
        """).fetchdf()
        pd.concat([vi, li], ignore_index=True).sort_values(
            ["source","target","itemid"]
        ).to_csv(OUT_ITEMS, index=False, encoding="utf-8-sig")

        # -------------------------------------------------------------
        # Missingness audit.
        # -------------------------------------------------------------
        print("[6/6] Computing missingness, SMD and positivity diagnostics...")
        exposure = "early_steroid_24h"

        numeric_covs = [
            "age","hospital_to_respiratory_t0_h",
            "hr_max","rr_max","map_min","spo2_min","temp_max_c","gcs_min",
            "weight_kg","fio2_max_pct","sf_ratio_last",
            "lactate_max","creatinine_last","bun_last","wbc_last",
            "hemoglobin_last","platelet_last","sodium_last","potassium_last",
            "bicarbonate_last","glucose_last","bilirubin_last","albumin_last",
            "crp_max","ph_min","po2_last","pco2_last","nlr",
            "vasopressor_agents_at_t0","norepinephrine_rate_at_t0",
        ]
        binary_covs = [
            "shock_active_at_t0",
            "t0_invasive_device","t0_ventilator_signal","t0_niv","t0_hfnc_qualified",
            "copd_history","asthma_history",
            "heart_failure","coronary_disease","diabetes",
            "chronic_kidney_disease","chronic_liver_disease","malignancy",
            "transplant_history","immunodeficiency","dementia",
            "pneumonia_top3_dx",
        ]
        categorical_covs = [
            "gender","race","respiratory_t0_type",
            "admission_type","admission_location","first_careunit","anchor_year_group",
        ]

        missing_rows = []
        for col in numeric_covs + binary_covs + categorical_covs:
            if col not in df.columns:
                continue
            for gname, sub in [
                ("overall", df),
                ("treated", df[df[exposure] == 1]),
                ("untreated", df[df[exposure] == 0]),
            ]:
                miss = int(sub[col].isna().sum())
                missing_rows.append({
                    "variable": col,
                    "group": gname,
                    "n": len(sub),
                    "missing_n": miss,
                    "missing_pct": round(100*miss/len(sub),1) if len(sub) else None,
                })
        pd.DataFrame(missing_rows).to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")

        # Unadjusted SMD.
        smd_rows = []
        t = df[df[exposure] == 1]
        u = df[df[exposure] == 0]

        for col in numeric_covs:
            if col not in df.columns:
                continue
            smd = pooled_smd_cont(t[col], u[col])
            smd_rows.append({
                "variable": col,
                "type": "continuous",
                "level": "",
                "treated_summary": fmt_num(t[col]),
                "untreated_summary": fmt_num(u[col]),
                "smd": smd,
                "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
            })

        for col in binary_covs:
            if col not in df.columns:
                continue
            smd = pooled_smd_binary(t[col], u[col])
            smd_rows.append({
                "variable": col,
                "type": "binary",
                "level": "1",
                "treated_summary": fmt_bin(t[col]),
                "untreated_summary": fmt_bin(u[col]),
                "smd": smd,
                "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
            })

        # Categorical -> one indicator per level; missing is explicit.
        for col in categorical_covs:
            if col not in df.columns:
                continue
            vals = df[col].astype(object).where(df[col].notna(), "<MISSING>").astype(str)
            levels = vals.value_counts().index.tolist()
            for level in levels:
                tx = (t[col].astype(object).where(t[col].notna(), "<MISSING>").astype(str) == level).astype(float)
                ux = (u[col].astype(object).where(u[col].notna(), "<MISSING>").astype(str) == level).astype(float)
                smd = pooled_smd_binary(tx, ux)
                smd_rows.append({
                    "variable": col,
                    "type": "categorical_level",
                    "level": level,
                    "treated_summary": fmt_bin(tx),
                    "untreated_summary": fmt_bin(ux),
                    "smd": smd,
                    "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
                })

        smd_df = pd.DataFrame(smd_rows).sort_values(
            ["abs_smd","variable"], ascending=[False,True], na_position="last"
        )
        smd_df.to_csv(OUT_SMD, index=False, encoding="utf-8-sig")

        # Summary table.
        summary_rows = []
        for col in numeric_covs:
            if col in df.columns:
                summary_rows.append({
                    "variable": col,
                    "type": "continuous",
                    "overall": fmt_num(df[col]),
                    "treated": fmt_num(t[col]),
                    "untreated": fmt_num(u[col]),
                })
        for col in binary_covs:
            if col in df.columns:
                summary_rows.append({
                    "variable": col,
                    "type": "binary",
                    "overall": fmt_bin(df[col]),
                    "treated": fmt_bin(t[col]),
                    "untreated": fmt_bin(u[col]),
                })
        pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

        # Positivity precheck: empirical treatment probabilities in clinically
        # important coarse strata. Not a final propensity-score diagnostic.
        pos_rows = []

        def add_pos(var, series):
            tmp = pd.DataFrame({"stratum": series, "A": df[exposure]})
            tmp["stratum"] = tmp["stratum"].astype(object).where(tmp["stratum"].notna(), "<MISSING>").astype(str)
            g = tmp.groupby("stratum", dropna=False)["A"].agg(["count","sum","mean"]).reset_index()
            for _, r in g.iterrows():
                pos_rows.append({
                    "stratifier": var,
                    "stratum": r["stratum"],
                    "n": int(r["count"]),
                    "treated_n": int(r["sum"]),
                    "treated_pct": round(100*float(r["mean"]),1),
                    "flag": (
                        "CAUTION_LOW_TREATED" if int(r["sum"]) < 20 else
                        "CAUTION_EXTREME_RATE" if float(r["mean"]) < 0.05 or float(r["mean"]) > 0.80 else
                        "OK"
                    )
                })

        add_pos("shock_active_at_t0", df["shock_active_at_t0"].map({0:"no_shock",1:"shock"}))
        add_pos("respiratory_t0_type", df["respiratory_t0_type"])
        add_pos("copd_history", df["copd_history"].map({0:"no_COPD",1:"COPD"}))
        add_pos("asthma_history", df["asthma_history"].map({0:"no_asthma",1:"asthma"}))
        add_pos("age_group", pd.cut(
            df["age"], bins=[17,49,64,74,200],
            labels=["18-49","50-64","65-74","75+"]
        ))
        if "lactate_max" in df.columns:
            add_pos("lactate_category", pd.cut(
                df["lactate_max"], bins=[-np.inf,2,4,np.inf],
                labels=["<=2",">2-4",">4"]
            ))
        add_pos("first_careunit", df["first_careunit"])

        pd.DataFrame(pos_rows).to_csv(OUT_POS, index=False, encoding="utf-8-sig")

        # Console decision aids.
        miss_overall = pd.read_csv(OUT_MISSING)
        miss_overall = miss_overall[miss_overall.group == "overall"]
        high_missing = miss_overall[miss_overall.missing_pct >= 50].sort_values(
            "missing_pct", ascending=False
        )

        high_smd = smd_df[smd_df.abs_smd >= 0.20]

        print("\n=== Stage 5 key diagnostics ===")
        print(f"Eligible: {len(df)} | treated: {int(df[exposure].sum())} | untreated: {int((df[exposure]==0).sum())}")
        print(f"Variables/levels with |SMD| >= 0.20: {len(high_smd)}")
        if len(high_smd):
            print(high_smd[["variable","level","abs_smd","treated_summary","untreated_summary"]].head(15).to_string(index=False))

        print(f"\nVariables with >=50% missingness: {len(high_missing)}")
        if len(high_missing):
            print(high_missing[["variable","missing_pct"]].to_string(index=False))

        print("\nSaved small files to upload:")
        for p in [OUT_MISSING, OUT_SMD, OUT_POS, OUT_SUMMARY, OUT_ITEMS]:
            print(f"  {p}")
        print(f"\nLocal only (do NOT upload): {OUT_COHORT}")
        print("\nNext step after review: Stage 6 clone-censor-weight target trial.")
        print("Do NOT interpret crude treated-vs-untreated mortality as a treatment effect.")

    except Exception:
        print("\n[FAILED] Copy the full traceback to ChatGPT:\n")
        traceback.print_exc()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()

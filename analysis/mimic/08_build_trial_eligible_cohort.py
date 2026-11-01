from __future__ import annotations

"""
Stage 4 — Formal trial-eligible cohort builder
Severe CAP + early systemic corticosteroids

PRIMARY CLINICAL QUESTION
Among adults with presumed community-acquired pneumonia who newly develop
severe respiratory failure, does initiation of systemic corticosteroids
within 24 h improve outcomes, and is the effect modified by concurrent shock?

This stage DOES NOT estimate a causal treatment effect yet.
It locks:
  1) respiratory time-zero
  2) shock status at time-zero
  3) eMAR-confirmed administered systemic steroid exposure
  4) new-user washout
  5) obstructive-airway exacerbation exclusion
  6) primary and sensitivity trial-eligible cohorts

Put this file in the SAME project root as config.py.

Required prior files/tables:
  outputs/feasibility_cohort.parquet
  cache/resp_relevant_chartevents.parquet  (from Stage 1)
  cache/target_inputevents.parquet         (from Stage 1)
  cache/steroid_prescriptions.parquet      (from Stage 1)
  cache/feasibility.duckdb with persistent tables:
       respiratory_support_events
       vasopressor_events

Outputs (retain only aggregate CSV summaries for reporting, NOT the parquet):
  outputs/08_STAGE4_COHORT_FLOW.csv
  outputs/08B_STAGE4_EXPOSURE_SHOCK_COUNTS.csv
  outputs/08C_STAGE4_DRUG_TIMING.csv
  outputs/08D_STAGE4_ELIGIBILITY_DIAGNOSTICS.csv
  outputs/stage4_trial_eligible_cohort.parquet   <-- restricted local analysis file

Reusable caches created:
  cache/stage4_emar_window.parquet
  cache/stage4_target_steroid_emar_detail.parquet
"""

from pathlib import Path
import traceback
import duckdb
import pandas as pd

import config
from src.utils import (
    connect_duckdb,
    find_table,
    try_find_table,
    source_sql,
    sql_quote_path,
)

PROJECT_ROOT = Path(config.PROJECT_ROOT)
DATA_ROOT = Path(config.DATA_ROOT)
CACHE_DIR = Path(config.CACHE_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

FEASIBILITY = OUTPUT_DIR / "feasibility_cohort.parquet"
RESP_CACHE = CACHE_DIR / "resp_relevant_chartevents.parquet"
INPUT_CACHE = CACHE_DIR / "target_inputevents.parquet"
RX_STEROID_CACHE = CACHE_DIR / "steroid_prescriptions.parquet"

EMAR_WINDOW_CACHE = CACHE_DIR / "stage4_emar_window.parquet"
EMAR_DETAIL_CACHE = CACHE_DIR / "stage4_target_steroid_emar_detail.parquet"

OUT_FLOW = OUTPUT_DIR / "08_STAGE4_COHORT_FLOW.csv"
OUT_STRATA = OUTPUT_DIR / "08B_STAGE4_EXPOSURE_SHOCK_COUNTS.csv"
OUT_DRUG = OUTPUT_DIR / "08C_STAGE4_DRUG_TIMING.csv"
OUT_DIAG = OUTPUT_DIR / "08D_STAGE4_ELIGIBILITY_DIAGNOSTICS.csv"
OUT_COHORT = OUTPUT_DIR / "stage4_trial_eligible_cohort.parquet"

EARLY_H = 24
WASHOUT_H = 24
SHOCK_WINDOW_H = 6
PRIMARY_HOSP_T0_H = 48
SENS_HOSP_T0_H = 72

FORCE_REBUILD_STAGE4_CACHE = False


def scalar(con, sql: str, params=None) -> int:
    v = con.execute(sql, params or []).fetchone()[0]
    return int(v or 0)


def table_exists(con, table_name: str) -> bool:
    n = con.execute(
        """
        SELECT count(*)
        FROM information_schema.tables
        WHERE lower(table_name)=lower(?)
        """,
        [table_name],
    ).fetchone()[0]
    return int(n) > 0


def add_row(rows, step, n, previous=None, notes=""):
    n = int(n or 0)
    if previous in (None, 0):
        retained = None
    else:
        retained = round(100.0 * n / int(previous), 1)
    rows.append(
        {
            "step": step,
            "n": n,
            "retained_from_previous_pct": retained,
            "notes": notes,
        }
    )


def route_class_sql(route_expr: str) -> str:
    r = route_expr
    return f"""
    CASE
      WHEN {r} IS NULL OR trim({r})='' THEN 'unknown'
      WHEN upper({r}) IN (
        'IV','IVP','IVPB','IV PUSH','IV BOLUS',
        'PO','PO/NG','NG','NGT','OG','OGT',
        'GT','G-TUBE','PEG','IM'
      ) THEN 'systemic'
      WHEN upper({r}) LIKE 'IV%' THEN 'systemic'
      WHEN upper({r}) LIKE 'PO%' THEN 'systemic'
      WHEN upper({r}) LIKE '%NG%' THEN 'systemic'
      WHEN upper({r}) LIKE '%G TUBE%' THEN 'systemic'
      WHEN upper({r}) LIKE '%G-TUBE%' THEN 'systemic'
      WHEN upper({r}) LIKE '%PEG%' THEN 'systemic'
      ELSE 'non_systemic_or_other'
    END
    """


def main():
    print("=" * 84)
    print("Stage 4 — Formal trial-eligible severe-CAP steroid cohort")
    print("=" * 84)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not FEASIBILITY.exists():
        raise FileNotFoundError(
            f"Missing {FEASIBILITY}\nRun 01_run_feasibility.py first."
        )
    for p in [RESP_CACHE, INPUT_CACHE, RX_STEROID_CACHE]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing Stage-1 cache: {p}\nRun 01_run_feasibility.py first."
            )

    emar_path = find_table(DATA_ROOT, "hosp", "emar")
    emar_detail_path = find_table(DATA_ROOT, "hosp", "emar_detail")
    dx_path = find_table(DATA_ROOT, "hosp", "diagnoses_icd")
    d_dx_path = try_find_table(DATA_ROOT, "hosp", "d_icd_diagnoses")

    print(f"[OK] eMAR        : {emar_path}")
    print(f"[OK] eMAR detail : {emar_detail_path}")
    print(f"[OK] diagnoses   : {dx_path}")
    print(f"[OK] dx dictionary: {d_dx_path if d_dx_path else 'not found; code fallback only'}")

    con = connect_duckdb(PROJECT_ROOT, getattr(config, "DUCKDB_THREADS", 4))

    try:
        if not table_exists(con, "respiratory_support_events") or not table_exists(con, "vasopressor_events"):
            raise RuntimeError(
                "Stage-1 persistent support tables are missing from cache/feasibility.duckdb.\n"
                "Please run 01_run_feasibility.py once, then rerun Stage 4."
            )

        fc = f"read_parquet('{sql_quote_path(FEASIBILITY)}')"
        rx_cache = f"read_parquet('{sql_quote_path(RX_STEROID_CACHE)}')"

        # ------------------------------------------------------------------
        # 1. Respiratory time-zero, NOT broad organ-support t0.
        # ------------------------------------------------------------------
        print("\n[1/8] Locking respiratory time-zero...")
        con.execute(f"""
            CREATE OR REPLACE TABLE stage4_base AS
            SELECT
                f.*,
                f.first_resp_strict AS respiratory_t0,
                date_diff('hour', f.admittime, f.first_resp_strict) AS hospital_to_respiratory_t0_h,
                CASE
                  WHEN f.first_resp_strict IS NOT NULL
                   AND date_diff('hour', f.admittime, f.first_resp_strict)
                       BETWEEN 0 AND {PRIMARY_HOSP_T0_H}
                  THEN 1 ELSE 0
                END AS respiratory_t0_within_48h,
                CASE
                  WHEN f.first_resp_strict IS NOT NULL
                   AND date_diff('hour', f.admittime, f.first_resp_strict)
                       BETWEEN 0 AND {SENS_HOSP_T0_H}
                  THEN 1 ELSE 0
                END AS respiratory_t0_within_72h
            FROM {fc} f
            WHERE f.first_resp_strict IS NOT NULL
        """)

        # What exactly triggered respiratory t0?
        con.execute("""
            CREATE OR REPLACE TABLE stage4_resp_t0_type AS
            SELECT
                b.stay_id,
                max(coalesce(r.invasive_device,0)) AS t0_invasive_device,
                max(coalesce(r.ventilator_signal,0)) AS t0_ventilator_signal,
                max(coalesce(r.niv_device,0)) AS t0_niv,
                max(coalesce(r.hfnc_remap_like,0)) AS t0_hfnc_qualified,
                CASE
                  WHEN max(coalesce(r.invasive_device,0))=1
                    OR max(coalesce(r.ventilator_signal,0))=1
                    THEN 'IMV_or_ventilator'
                  WHEN max(coalesce(r.niv_device,0))=1
                    THEN 'NIV'
                  WHEN max(coalesce(r.hfnc_remap_like,0))=1
                    THEN 'qualified_HFNC'
                  ELSE 'other_strict_respiratory'
                END AS respiratory_t0_type
            FROM stage4_base b
            JOIN respiratory_support_events r
              ON r.stay_id=b.stay_id
             AND r.charttime=b.respiratory_t0
             AND r.resp_support_strict=1
            GROUP BY b.stay_id
        """)

        # ------------------------------------------------------------------
        # 2. Shock at respiratory t0 and within next 6 h.
        # ------------------------------------------------------------------
        print("[2/8] Defining shock status at respiratory t0...")
        con.execute(f"""
            CREATE OR REPLACE TABLE stage4_shock AS
            SELECT
                b.stay_id,
                max(CASE
                    WHEN v.starttime <= b.respiratory_t0
                     AND coalesce(v.endtime, v.starttime + INTERVAL '1 minute') >= b.respiratory_t0
                    THEN 1 ELSE 0 END) AS vasopressor_active_at_t0,
                max(CASE
                    WHEN v.starttime <= b.respiratory_t0 + INTERVAL '{SHOCK_WINDOW_H} hours'
                     AND coalesce(v.endtime, v.starttime + INTERVAL '1 minute') >= b.respiratory_t0
                    THEN 1 ELSE 0 END) AS vasopressor_overlap_t0_to_6h,
                min(CASE
                    WHEN v.starttime <= b.respiratory_t0 + INTERVAL '{SHOCK_WINDOW_H} hours'
                     AND coalesce(v.endtime, v.starttime + INTERVAL '1 minute') >= b.respiratory_t0
                    THEN v.starttime END) AS first_vasopressor_relevant
            FROM stage4_base b
            LEFT JOIN vasopressor_events v USING (stay_id)
            GROUP BY b.stay_id
        """)

        # ------------------------------------------------------------------
        # 3. Obstructive-airway exacerbation flags.
        #    Use diagnosis dictionary text when available, plus code fallback.
        # ------------------------------------------------------------------
        print("[3/8] Auditing COPD/asthma acute-exacerbation indications...")
        dx = source_sql(dx_path)
        if d_dx_path is not None:
            ddx = source_sql(d_dx_path)
            con.execute(f"""
                CREATE OR REPLACE TABLE stage4_obstructive_dx AS
                SELECT
                    d.hadm_id,
                    max(CASE
                        WHEN (
                            lower(coalesce(x.long_title,'')) LIKE '%asthma%exacerbation%'
                            OR lower(coalesce(x.long_title,'')) LIKE '%status asthmaticus%'
                            OR lower(coalesce(x.long_title,'')) LIKE '%obstructive%bronchitis%exacerbation%'
                            OR lower(coalesce(x.long_title,'')) LIKE '%chronic obstructive%exacerbation%'
                            OR (
                                d.icd_version=10
                                AND replace(upper(d.icd_code),'.','')='J441'
                            )
                        )
                        THEN 1 ELSE 0 END) AS obstructive_exacerbation_dx,
                    max(CASE
                        WHEN lower(coalesce(x.long_title,'')) LIKE '%chronic obstructive%'
                          OR lower(coalesce(x.long_title,'')) LIKE '%obstructive chronic bronchitis%'
                          OR (
                              d.icd_version=10
                              AND regexp_matches(replace(upper(d.icd_code),'.',''), '^J4[1-4]')
                          )
                          OR (
                              d.icd_version=9
                              AND regexp_matches(replace(upper(d.icd_code),'.',''), '^(490|491|492|496)')
                          )
                        THEN 1 ELSE 0 END) AS copd_history,
                    max(CASE
                        WHEN lower(coalesce(x.long_title,'')) LIKE '%asthma%'
                          OR (
                              d.icd_version=10
                              AND regexp_matches(replace(upper(d.icd_code),'.',''), '^J45')
                          )
                          OR (
                              d.icd_version=9
                              AND regexp_matches(replace(upper(d.icd_code),'.',''), '^493')
                          )
                        THEN 1 ELSE 0 END) AS asthma_history
                FROM {dx} d
                LEFT JOIN {ddx} x
                  ON d.icd_code=x.icd_code
                 AND d.icd_version=x.icd_version
                GROUP BY d.hadm_id
            """)
        else:
            con.execute(f"""
                CREATE OR REPLACE TABLE stage4_obstructive_dx AS
                SELECT
                    hadm_id,
                    max(CASE
                        WHEN (icd_version=10 AND replace(upper(icd_code),'.','')='J441')
                          OR (icd_version=9 AND regexp_matches(replace(upper(icd_code),'.',''), '^(49121|49122|493[0-9]*2|493[0-9]*1)'))
                        THEN 1 ELSE 0 END) AS obstructive_exacerbation_dx,
                    max(CASE
                        WHEN (icd_version=10 AND regexp_matches(replace(upper(icd_code),'.',''), '^J4[1-4]'))
                          OR (icd_version=9 AND regexp_matches(replace(upper(icd_code),'.',''), '^(490|491|492|496)'))
                        THEN 1 ELSE 0 END) AS copd_history,
                    max(CASE
                        WHEN (icd_version=10 AND regexp_matches(replace(upper(icd_code),'.',''), '^J45'))
                          OR (icd_version=9 AND regexp_matches(replace(upper(icd_code),'.',''), '^493'))
                        THEN 1 ELSE 0 END) AS asthma_history
                FROM {dx}
                GROUP BY hadm_id
            """)

        # ------------------------------------------------------------------
        # 4. Cache ALL eMAR events only within each respiratory t0 ±24h.
        #    This scans emar.csv.gz once and makes all future reruns fast.
        # ------------------------------------------------------------------
        print("[4/8] Building/reusing t0-centered eMAR window cache...")
        if FORCE_REBUILD_STAGE4_CACHE and EMAR_WINDOW_CACHE.exists():
            EMAR_WINDOW_CACHE.unlink()

        if not EMAR_WINDOW_CACHE.exists():
            emar = source_sql(emar_path)
            print("      First run: scanning emar.csv.gz once; this is the slow step.")
            con.execute(f"""
                COPY (
                    SELECT
                        b.stay_id,
                        b.subject_id,
                        b.hadm_id,
                        b.respiratory_t0,
                        e.emar_id,
                        e.emar_seq,
                        e.poe_id,
                        e.pharmacy_id,
                        e.charttime,
                        e.scheduletime,
                        e.storetime,
                        e.medication,
                        e.event_txt
                    FROM stage4_base b
                    JOIN {emar} e
                      ON e.subject_id=b.subject_id
                     AND e.hadm_id=b.hadm_id
                     AND e.charttime BETWEEN b.respiratory_t0 - INTERVAL '{WASHOUT_H} hours'
                                         AND b.respiratory_t0 + INTERVAL '{EARLY_H} hours'
                )
                TO '{sql_quote_path(EMAR_WINDOW_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        else:
            print(f"      Reusing {EMAR_WINDOW_CACHE}")

        emar_win = f"read_parquet('{sql_quote_path(EMAR_WINDOW_CACHE)}')"

        con.execute(f"""
            CREATE OR REPLACE TABLE stage4_emar_observable AS
            SELECT stay_id, count(*) AS emar_rows_near_t0
            FROM {emar_win}
            GROUP BY stay_id
        """)

        # Target steroid records from eMAR window.
        con.execute(f"""
            CREATE OR REPLACE TABLE stage4_steroid_emar_raw AS
            SELECT
                e.*,
                CASE
                    WHEN lower(e.medication) LIKE '%hydrocortisone%' THEN 'hydrocortisone'
                    WHEN lower(e.medication) LIKE '%methylpred%' THEN 'methylprednisolone'
                    WHEN lower(e.medication) LIKE '%dexameth%' THEN 'dexamethasone'
                    WHEN lower(e.medication) LIKE '%prednisolone%' THEN 'prednisolone'
                    WHEN lower(e.medication) LIKE '%prednisone%' THEN 'prednisone'
                    ELSE NULL
                END AS steroid_name,
                date_diff('minute', e.respiratory_t0, e.charttime)/60.0 AS hours_from_t0
            FROM {emar_win} e
            WHERE lower(trim(coalesce(e.event_txt,'')))='administered'
              AND (
                    lower(e.medication) LIKE '%hydrocortisone%'
                 OR lower(e.medication) LIKE '%methylpred%'
                 OR lower(e.medication) LIKE '%dexameth%'
                 OR lower(e.medication) LIKE '%prednisolone%'
                 OR lower(e.medication) LIKE '%prednisone%'
              )
              AND NOT (
                    lower(e.medication) LIKE '%ophth%'
                 OR lower(e.medication) LIKE '%eye%'
                 OR lower(e.medication) LIKE '%cream%'
                 OR lower(e.medication) LIKE '%ointment%'
                 OR lower(e.medication) LIKE '%lotion%'
                 OR lower(e.medication) LIKE '%suppos%'
                 OR lower(e.medication) LIKE '%rectal%'
                 OR lower(e.medication) LIKE '%enema%'
                 OR lower(e.medication) LIKE '%nasal%'
                 OR lower(e.medication) LIKE '%otic%'
                 OR lower(e.medication) LIKE '%inhal%'
                 OR lower(e.medication) LIKE '%nebul%'
              )
        """)

        # ------------------------------------------------------------------
        # 5. Route confirmation from emar_detail; cache only target emar_ids.
        # ------------------------------------------------------------------
        print("[5/8] Building/reusing route-confirmation cache...")
        if FORCE_REBUILD_STAGE4_CACHE and EMAR_DETAIL_CACHE.exists():
            EMAR_DETAIL_CACHE.unlink()

        if not EMAR_DETAIL_CACHE.exists():
            detail = source_sql(emar_detail_path)
            print("      First run: scanning emar_detail.csv.gz once; cached afterwards.")
            con.execute(f"""
                COPY (
                    SELECT
                        d.emar_id,
                        max(CASE
                            WHEN d.route IS NOT NULL AND trim(cast(d.route AS VARCHAR))<>''
                            THEN upper(trim(cast(d.route AS VARCHAR)))
                        END) AS detail_route,
                        string_agg(DISTINCT cast(d.administration_type AS VARCHAR), ' | ')
                          FILTER (WHERE d.administration_type IS NOT NULL) AS administration_type,
                        string_agg(DISTINCT cast(d.dose_given AS VARCHAR), ' | ')
                          FILTER (WHERE d.dose_given IS NOT NULL
                                  AND trim(cast(d.dose_given AS VARCHAR))<>'') AS dose_given_raw,
                        string_agg(DISTINCT cast(d.dose_given_unit AS VARCHAR), ' | ')
                          FILTER (WHERE d.dose_given_unit IS NOT NULL
                                  AND trim(cast(d.dose_given_unit AS VARCHAR))<>'') AS dose_given_unit_raw
                    FROM {detail} d
                    JOIN (SELECT DISTINCT emar_id FROM stage4_steroid_emar_raw) s
                      USING (emar_id)
                    GROUP BY d.emar_id
                )
                TO '{sql_quote_path(EMAR_DETAIL_CACHE)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        else:
            print(f"      Reusing {EMAR_DETAIL_CACHE}")

        detail_cache = f"read_parquet('{sql_quote_path(EMAR_DETAIL_CACHE)}')"

        # Fallback route from steroid prescription pharmacy_id.
        con.execute(f"""
            CREATE OR REPLACE TABLE stage4_rx_route AS
            SELECT
                pharmacy_id,
                max(upper(trim(cast(route AS VARCHAR))))
                  FILTER (WHERE route IS NOT NULL AND trim(cast(route AS VARCHAR))<>'') AS rx_route
            FROM {rx_cache}
            WHERE pharmacy_id IS NOT NULL
            GROUP BY pharmacy_id
        """)

        route_case = route_class_sql("coalesce(d.detail_route, r.rx_route)")

        con.execute(f"""
            CREATE OR REPLACE TABLE stage4_steroid_events AS
            SELECT
                e.*,
                d.detail_route,
                r.rx_route,
                coalesce(d.detail_route, r.rx_route) AS route_best,
                d.administration_type,
                d.dose_given_raw,
                d.dose_given_unit_raw,
                {route_case} AS route_class
            FROM stage4_steroid_emar_raw e
            LEFT JOIN {detail_cache} d USING (emar_id)
            LEFT JOIN stage4_rx_route r USING (pharmacy_id)
        """)

        # Strict administered systemic exposure summary.
        con.execute(f"""
            CREATE OR REPLACE TABLE stage4_steroid_summary AS
            SELECT
                b.stay_id,
                min(CASE
                    WHEN e.route_class='systemic'
                     AND e.hours_from_t0 >= 0
                     AND e.hours_from_t0 <= {EARLY_H}
                    THEN e.charttime END) AS first_early_steroid_time,
                min(CASE
                    WHEN e.route_class='systemic'
                     AND e.hours_from_t0 >= 0
                     AND e.hours_from_t0 <= {EARLY_H}
                    THEN e.hours_from_t0 END) AS first_early_steroid_h,
                max(CASE
                    WHEN e.route_class='systemic'
                     AND e.hours_from_t0 >= 0
                     AND e.hours_from_t0 <= {EARLY_H}
                    THEN 1 ELSE 0 END) AS early_steroid_24h,
                max(CASE
                    WHEN e.route_class IN ('systemic','unknown')
                     AND e.hours_from_t0 >= 0
                     AND e.hours_from_t0 <= {EARLY_H}
                    THEN 1 ELSE 0 END) AS early_steroid_24h_including_unknown_route,
                max(CASE
                    WHEN e.route_class='systemic'
                     AND e.hours_from_t0 < 0
                     AND e.hours_from_t0 >= -{WASHOUT_H}
                    THEN 1 ELSE 0 END) AS steroid_pre_t0_24h,
                max(CASE WHEN e.route_class='systemic'
                           AND e.hours_from_t0 BETWEEN 0 AND {EARLY_H}
                           AND e.steroid_name='hydrocortisone' THEN 1 ELSE 0 END) AS early_hydrocortisone,
                max(CASE WHEN e.route_class='systemic'
                           AND e.hours_from_t0 BETWEEN 0 AND {EARLY_H}
                           AND e.steroid_name='methylprednisolone' THEN 1 ELSE 0 END) AS early_methylprednisolone,
                max(CASE WHEN e.route_class='systemic'
                           AND e.hours_from_t0 BETWEEN 0 AND {EARLY_H}
                           AND e.steroid_name='dexamethasone' THEN 1 ELSE 0 END) AS early_dexamethasone,
                max(CASE WHEN e.route_class='systemic'
                           AND e.hours_from_t0 BETWEEN 0 AND {EARLY_H}
                           AND e.steroid_name='prednisone' THEN 1 ELSE 0 END) AS early_prednisone,
                max(CASE WHEN e.route_class='systemic'
                           AND e.hours_from_t0 BETWEEN 0 AND {EARLY_H}
                           AND e.steroid_name='prednisolone' THEN 1 ELSE 0 END) AS early_prednisolone
            FROM stage4_base b
            LEFT JOIN stage4_steroid_events e USING (stay_id)
            GROUP BY b.stay_id
        """)

        # ------------------------------------------------------------------
        # 6. Assemble trial-eligibility flags and outcomes.
        # ------------------------------------------------------------------
        print("[6/8] Building formal eligibility flags...")
        con.execute(f"""
            CREATE OR REPLACE TABLE stage4_cohort AS
            SELECT
                b.subject_id,
                b.hadm_id,
                b.stay_id,
                b.age,
                b.gender,
                b.race,
                b.admittime,
                b.intime,
                b.outtime,
                b.dischtime,
                b.dod,
                b.respiratory_t0,
                b.hospital_to_respiratory_t0_h,
                b.respiratory_t0_within_48h,
                b.respiratory_t0_within_72h,
                b.pneumonia_top3_dx,
                coalesce(rt.t0_invasive_device,0) AS t0_invasive_device,
                coalesce(rt.t0_ventilator_signal,0) AS t0_ventilator_signal,
                coalesce(rt.t0_niv,0) AS t0_niv,
                coalesce(rt.t0_hfnc_qualified,0) AS t0_hfnc_qualified,
                coalesce(rt.respiratory_t0_type,'unknown') AS respiratory_t0_type,
                coalesce(sh.vasopressor_active_at_t0,0) AS shock_active_at_t0,
                coalesce(sh.vasopressor_overlap_t0_to_6h,0) AS shock_t0_to_6h,
                sh.first_vasopressor_relevant,
                coalesce(dx.obstructive_exacerbation_dx,0) AS obstructive_exacerbation_dx,
                coalesce(dx.copd_history,0) AS copd_history,
                coalesce(dx.asthma_history,0) AS asthma_history,
                CASE WHEN eo.stay_id IS NOT NULL THEN 1 ELSE 0 END AS emar_observable,
                coalesce(ss.early_steroid_24h,0) AS early_steroid_24h,
                coalesce(ss.early_steroid_24h_including_unknown_route,0) AS early_steroid_24h_including_unknown_route,
                coalesce(ss.steroid_pre_t0_24h,0) AS steroid_pre_t0_24h,
                ss.first_early_steroid_time,
                ss.first_early_steroid_h,
                coalesce(ss.early_hydrocortisone,0) AS early_hydrocortisone,
                coalesce(ss.early_methylprednisolone,0) AS early_methylprednisolone,
                coalesce(ss.early_dexamethasone,0) AS early_dexamethasone,
                coalesce(ss.early_prednisone,0) AS early_prednisone,
                coalesce(ss.early_prednisolone,0) AS early_prednisolone,
                CASE WHEN b.hospital_expire_flag=1 THEN 1 ELSE 0 END AS hospital_death,
                CASE
                  WHEN b.dod IS NOT NULL
                   AND cast(b.dod AS date) >= cast(b.respiratory_t0 AS date)
                   AND cast(b.dod AS date) <= cast(b.respiratory_t0 + INTERVAL '28 days' AS date)
                  THEN 1 ELSE 0
                END AS death_28d,
                CASE
                  WHEN b.respiratory_t0_within_48h=1
                   AND eo.stay_id IS NOT NULL
                   AND coalesce(ss.steroid_pre_t0_24h,0)=0
                   AND coalesce(dx.obstructive_exacerbation_dx,0)=0
                  THEN 1 ELSE 0
                END AS eligible_primary,
                CASE
                  WHEN b.respiratory_t0_within_48h=1
                   AND eo.stay_id IS NOT NULL
                   AND coalesce(ss.steroid_pre_t0_24h,0)=0
                   AND coalesce(dx.obstructive_exacerbation_dx,0)=0
                   AND b.pneumonia_top3_dx=1
                  THEN 1 ELSE 0
                END AS eligible_strict_cap_sensitivity,
                CASE
                  WHEN b.respiratory_t0_within_72h=1
                   AND eo.stay_id IS NOT NULL
                   AND coalesce(ss.steroid_pre_t0_24h,0)=0
                   AND coalesce(dx.obstructive_exacerbation_dx,0)=0
                  THEN 1 ELSE 0
                END AS eligible_72h_sensitivity
            FROM stage4_base b
            LEFT JOIN stage4_resp_t0_type rt USING (stay_id)
            LEFT JOIN stage4_shock sh USING (stay_id)
            LEFT JOIN stage4_obstructive_dx dx USING (hadm_id)
            LEFT JOIN stage4_emar_observable eo USING (stay_id)
            LEFT JOIN stage4_steroid_summary ss USING (stay_id)
        """)

        # Save local patient-level/pseudonymized analysis cohort.
        con.execute(f"""
            COPY (
                SELECT * FROM stage4_cohort
                ORDER BY respiratory_t0, stay_id
            )
            TO '{sql_quote_path(OUT_COHORT)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # ------------------------------------------------------------------
        # 7. Output cohort flow.
        # ------------------------------------------------------------------
        print("[7/8] Exporting trial-eligibility summaries...")
        flow = []
        n0 = scalar(con, "SELECT count(*) FROM stage4_base")
        add_row(flow, "Strict respiratory t0 identified", n0, None,
                "First strict respiratory-support event; this is the new time-zero")

        n1 = scalar(con, f"""
            SELECT count(*) FROM stage4_cohort
            WHERE hospital_to_respiratory_t0_h BETWEEN 0 AND {PRIMARY_HOSP_T0_H}
        """)
        add_row(flow, "Respiratory t0 within 48 h of hospital admission", n1, n0,
                "Primary CAP timing restriction")

        n2 = scalar(con, f"""
            SELECT count(*) FROM stage4_cohort
            WHERE respiratory_t0_within_48h=1 AND emar_observable=1
        """)
        add_row(flow, "eMAR observable around respiratory t0", n2, n1,
                "Any eMAR event in t0-24 h to t0+24 h")

        n3 = scalar(con, """
            SELECT count(*) FROM stage4_cohort
            WHERE respiratory_t0_within_48h=1
              AND emar_observable=1
              AND steroid_pre_t0_24h=0
        """)
        add_row(flow, "New-user washout: no systemic steroid in prior 24 h", n3, n2)

        n4 = scalar(con, """
            SELECT count(*) FROM stage4_cohort
            WHERE eligible_primary=1
        """)
        add_row(flow, "Primary trial-eligible cohort: exclude COPD/asthma acute exacerbation", n4, n3)

        treated = scalar(con, """
            SELECT count(*) FROM stage4_cohort
            WHERE eligible_primary=1 AND early_steroid_24h=1
        """)
        untreated = scalar(con, """
            SELECT count(*) FROM stage4_cohort
            WHERE eligible_primary=1 AND early_steroid_24h=0
        """)
        add_row(flow, "Early systemic corticosteroid <=24 h (eMAR administered)", treated, n4)
        add_row(flow, "No early systemic corticosteroid <=24 h", untreated, n4)

        strict_cap = scalar(con, "SELECT count(*) FROM stage4_cohort WHERE eligible_strict_cap_sensitivity=1")
        sens72 = scalar(con, "SELECT count(*) FROM stage4_cohort WHERE eligible_72h_sensitivity=1")
        add_row(flow, "Sensitivity: pneumonia diagnosis in top 3 positions", strict_cap, n4,
                "Not part of primary cohort")
        add_row(flow, "Sensitivity: allow respiratory t0 within 72 h", sens72, n4,
                "Not part of primary cohort")

        flow_df = pd.DataFrame(flow)
        flow_df.to_csv(OUT_FLOW, index=False, encoding="utf-8-sig")

        # ------------------------------------------------------------------
        # Exposure x shock strata + outcome counts (descriptive only).
        # ------------------------------------------------------------------
        strata = con.execute("""
            SELECT
                'primary_shock_active_at_t0' AS shock_definition,
                shock_active_at_t0 AS shock,
                early_steroid_24h AS early_steroid,
                count(*) AS n,
                sum(death_28d) AS death_28d_n,
                round(100.0*avg(death_28d),1) AS death_28d_pct,
                sum(hospital_death) AS hospital_death_n,
                round(100.0*avg(hospital_death),1) AS hospital_death_pct
            FROM stage4_cohort
            WHERE eligible_primary=1
            GROUP BY shock_active_at_t0, early_steroid_24h

            UNION ALL

            SELECT
                'sensitivity_shock_overlap_t0_to_6h' AS shock_definition,
                shock_t0_to_6h AS shock,
                early_steroid_24h AS early_steroid,
                count(*) AS n,
                sum(death_28d) AS death_28d_n,
                round(100.0*avg(death_28d),1) AS death_28d_pct,
                sum(hospital_death) AS hospital_death_n,
                round(100.0*avg(hospital_death),1) AS hospital_death_pct
            FROM stage4_cohort
            WHERE eligible_primary=1
            GROUP BY shock_t0_to_6h, early_steroid_24h

            ORDER BY shock_definition, shock, early_steroid
        """).fetchdf()
        strata["warning"] = "DESCRIPTIVE ONLY — do not interpret mortality differences causally"
        strata.to_csv(OUT_STRATA, index=False, encoding="utf-8-sig")

        # Drug/timing counts in primary eligible treated patients.
        drug_rows = []
        treated_den = scalar(con, "SELECT count(*) FROM stage4_cohort WHERE eligible_primary=1 AND early_steroid_24h=1")
        for label, col in [
            ("hydrocortisone", "early_hydrocortisone"),
            ("methylprednisolone", "early_methylprednisolone"),
            ("dexamethasone", "early_dexamethasone"),
            ("prednisone", "early_prednisone"),
            ("prednisolone", "early_prednisolone"),
        ]:
            n = scalar(con, f"""
                SELECT count(*) FROM stage4_cohort
                WHERE eligible_primary=1 AND early_steroid_24h=1 AND {col}=1
            """)
            drug_rows.append(
                {
                    "section": "drug",
                    "item": label,
                    "n": n,
                    "denominator_n": treated_den,
                    "pct": None if treated_den == 0 else round(100*n/treated_den,1),
                }
            )

        for label, cond in [
            ("0_to_6h", "first_early_steroid_h>=0 AND first_early_steroid_h<=6"),
            (">6_to_12h", "first_early_steroid_h>6 AND first_early_steroid_h<=12"),
            (">12_to_24h", "first_early_steroid_h>12 AND first_early_steroid_h<=24"),
        ]:
            n = scalar(con, f"""
                SELECT count(*) FROM stage4_cohort
                WHERE eligible_primary=1 AND early_steroid_24h=1 AND {cond}
            """)
            drug_rows.append(
                {
                    "section": "timing",
                    "item": label,
                    "n": n,
                    "denominator_n": treated_den,
                    "pct": None if treated_den == 0 else round(100*n/treated_den,1),
                }
            )

        drug_df = pd.DataFrame(drug_rows)
        drug_df.to_csv(OUT_DRUG, index=False, encoding="utf-8-sig")

        # Diagnostics / potential residual-confounding flags.
        diag_queries = [
            ("primary_eligible", "eligible_primary=1"),
            ("treated", "eligible_primary=1 AND early_steroid_24h=1"),
            ("untreated", "eligible_primary=1 AND early_steroid_24h=0"),
            ("stable_COPD_history", "eligible_primary=1 AND copd_history=1"),
            ("stable_asthma_history", "eligible_primary=1 AND asthma_history=1"),
            ("shock_active_at_t0", "eligible_primary=1 AND shock_active_at_t0=1"),
            ("shock_t0_to_6h", "eligible_primary=1 AND shock_t0_to_6h=1"),
            ("IMV_or_ventilator_t0", "eligible_primary=1 AND respiratory_t0_type='IMV_or_ventilator'"),
            ("NIV_t0", "eligible_primary=1 AND respiratory_t0_type='NIV'"),
            ("qualified_HFNC_t0", "eligible_primary=1 AND respiratory_t0_type='qualified_HFNC'"),
            ("28d_deaths", "eligible_primary=1 AND death_28d=1"),
            ("hospital_deaths", "eligible_primary=1 AND hospital_death=1"),
            ("route_unknown_sensitivity_only",
             "eligible_primary=1 AND early_steroid_24h=0 AND early_steroid_24h_including_unknown_route=1"),
        ]
        primary_n = scalar(con, "SELECT count(*) FROM stage4_cohort WHERE eligible_primary=1")
        diag_rows = []
        for label, where in diag_queries:
            n = scalar(con, f"SELECT count(*) FROM stage4_cohort WHERE {where}")
            diag_rows.append(
                {
                    "metric": label,
                    "n": n,
                    "primary_eligible_n": primary_n,
                    "pct_of_primary": None if primary_n == 0 else round(100*n/primary_n,1),
                }
            )
        pd.DataFrame(diag_rows).to_csv(OUT_DIAG, index=False, encoding="utf-8-sig")

        # ------------------------------------------------------------------
        # 8. Console summary / go-no-go.
        # ------------------------------------------------------------------
        print("[8/8] Final Stage-4 summary\n")
        print(flow_df.to_string(index=False))
        print("\n=== Exposure × shock strata (DESCRIPTIVE ONLY) ===")
        print(strata.to_string(index=False))

        primary_n = scalar(con, "SELECT count(*) FROM stage4_cohort WHERE eligible_primary=1")
        primary_treated = scalar(con, "SELECT count(*) FROM stage4_cohort WHERE eligible_primary=1 AND early_steroid_24h=1")
        deaths28 = scalar(con, "SELECT count(*) FROM stage4_cohort WHERE eligible_primary=1 AND death_28d=1")
        treated_no_shock = scalar(con, """
            SELECT count(*) FROM stage4_cohort
            WHERE eligible_primary=1 AND early_steroid_24h=1 AND shock_active_at_t0=0
        """)
        treated_shock = scalar(con, """
            SELECT count(*) FROM stage4_cohort
            WHERE eligible_primary=1 AND early_steroid_24h=1 AND shock_active_at_t0=1
        """)

        # These are feasibility heuristics, NOT formal power calculations.
        criteria = {
            "primary eligible >= 1500": primary_n >= 1500,
            "treated >= 300": primary_treated >= 300,
            "28-day deaths >= 150": deaths28 >= 150,
            "treated/no-shock >= 200": treated_no_shock >= 200,
            "treated/shock >= 60": treated_shock >= 60,
        }
        print("\n=== Stage-4 heuristic GO/NO-GO ===")
        for k, v in criteria.items():
            print(f"  {'GO' if v else 'CAUTION'}  {k}")
        overall = "GO" if sum(criteria.values()) >= 4 else "CAUTION"
        print(f"\nOVERALL: {overall}")

        print("\nFiles created:")
        print(f"  {OUT_FLOW}")
        print(f"  {OUT_STRATA}")
        print(f"  {OUT_DRUG}")
        print(f"  {OUT_DIAG}")
        print(f"  {OUT_COHORT}  <-- restricted local analysis file")
        print("\nGenerated small audit CSV files:")
        print("  08_STAGE4_COHORT_FLOW.csv")
        print("  08B_STAGE4_EXPOSURE_SHOCK_COUNTS.csv")
        print("  08C_STAGE4_DRUG_TIMING.csv")
        print("  08D_STAGE4_ELIGIBILITY_DIAGNOSTICS.csv")
        print("\nImportant: mortality percentages shown here are descriptive only.")
        print("Causal estimation starts in Stage 5 after baseline covariate extraction.")

    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()

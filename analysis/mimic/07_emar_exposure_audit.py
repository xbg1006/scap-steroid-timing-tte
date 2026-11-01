from __future__ import annotations

"""
Stage 3: administration-level steroid exposure audit using MIMIC-IV eMAR.

WHY:
- prescriptions = medication orders, not guaranteed administrations
- eMAR = bedside medication administration record
- this script does NOT export patient-level data

Put this file in the SAME project root as:
    config.py
    01_run_feasibility.py
    outputs/feasibility_cohort.parquet

Run from a terminal.

Outputs:
    outputs/07_EMAR_EXPOSURE_AUDIT.csv
    outputs/07B_EMAR_STEROID_EVENT_AUDIT.csv
    outputs/07C_EMAR_RX_CONCORDANCE.csv

Primary strict exposure in this audit:
    event_txt = 'Administered'
    charttime from severe t0 through +24 h
    systemic route only when route information is available
"""

from pathlib import Path
import re
import duckdb
import pandas as pd

import config
from src.utils import connect_duckdb, find_table, try_find_table, source_sql, sql_quote_path

PROJECT_ROOT = Path(config.PROJECT_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_ROOT = Path(config.DATA_ROOT)
EARLY_HOURS = int(getattr(config, "EARLY_STEROID_WINDOW_HOURS", 24))
LOOKBACK_HOURS = int(getattr(config, "PRE_T0_STEROID_LOOKBACK_HOURS", 24))

COHORT = OUTPUT_DIR / "feasibility_cohort.parquet"
OUT_MAIN = OUTPUT_DIR / "07_EMAR_EXPOSURE_AUDIT.csv"
OUT_EVENTS = OUTPUT_DIR / "07B_EMAR_STEROID_EVENT_AUDIT.csv"
OUT_CONCORD = OUTPUT_DIR / "07C_EMAR_RX_CONCORDANCE.csv"


def scalar(con, sql, params=None):
    x = con.execute(sql, params or []).fetchone()[0]
    return int(x or 0)


def add(rows, section, item, n, denominator=None, notes=""):
    n = int(n or 0)
    if denominator is None:
        pct = None
    else:
        denominator = int(denominator or 0)
        pct = None if denominator == 0 else round(100.0 * n / denominator, 1)
    rows.append({
        "section": section,
        "item": item,
        "n": n,
        "denominator_n": denominator,
        "pct": pct,
        "notes": notes,
    })


def main():
    print("=" * 78)
    print("MIMIC severe-CAP steroid project: Stage 3 eMAR administration audit")
    print("=" * 78)

    if not COHORT.exists():
        raise FileNotFoundError(
            f"Missing {COHORT}\n"
            "Run 01_run_feasibility.py first."
        )

    emar_path = find_table(DATA_ROOT, "hosp", "emar")
    detail_path = try_find_table(DATA_ROOT, "hosp", "emar_detail")
    rx_path = try_find_table(DATA_ROOT, "hosp", "prescriptions")

    print(f"[OK] emar: {emar_path}")
    print(f"[OK] emar_detail: {detail_path if detail_path else 'NOT FOUND'}")
    print(f"[OK] prescriptions: {rx_path if rx_path else 'NOT FOUND'}")

    con = connect_duckdb(PROJECT_ROOT, threads=getattr(config, "DUCKDB_THREADS", 4))
    cohort_sql = f"read_parquet('{sql_quote_path(COHORT)}')"
    emar = source_sql(emar_path)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fc AS
        SELECT *
        FROM {cohort_sql}
        WHERE severe_cap_broad_proxy=1
          AND severe_t0_broad IS NOT NULL
    """)

    severe_n = scalar(con, "SELECT count(*) FROM fc")

    # ---------------------------------------------------------------
    # Is eMAR observable around t0?
    # Use ANY eMAR event in t0-24h to t0+24h as an observability marker.
    # ---------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE emar_observable AS
        SELECT DISTINCT f.stay_id
        FROM fc f
        JOIN {emar} e
          ON e.subject_id=f.subject_id
         AND e.hadm_id=f.hadm_id
         AND e.charttime BETWEEN f.severe_t0_broad - INTERVAL '{LOOKBACK_HOURS} hours'
                             AND f.severe_t0_broad + INTERVAL '{EARLY_HOURS} hours'
    """)

    observable_n = scalar(con, "SELECT count(*) FROM emar_observable")

    # ---------------------------------------------------------------
    # Build route lookup from emar_detail.
    # We deliberately do NOT sum doses yet; this is an exposure audit.
    # ---------------------------------------------------------------
    if detail_path is not None:
        detail = source_sql(detail_path)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE emar_detail_route AS
            SELECT
                emar_id,
                max(CASE WHEN route IS NOT NULL AND trim(cast(route AS VARCHAR))<>'' 
                         THEN upper(trim(cast(route AS VARCHAR))) END) AS detail_route,
                string_agg(DISTINCT cast(administration_type AS VARCHAR), ' | ')
                    FILTER (WHERE administration_type IS NOT NULL) AS administration_type,
                string_agg(DISTINCT cast(dose_given AS VARCHAR), ' | ')
                    FILTER (WHERE dose_given IS NOT NULL
                            AND trim(cast(dose_given AS VARCHAR))<>'') AS dose_given_raw,
                string_agg(DISTINCT cast(dose_given_unit AS VARCHAR), ' | ')
                    FILTER (WHERE dose_given_unit IS NOT NULL
                            AND trim(cast(dose_given_unit AS VARCHAR))<>'') AS dose_given_unit_raw
            FROM {detail}
            GROUP BY emar_id
        """)
    else:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE emar_detail_route AS
            SELECT
                cast(NULL AS VARCHAR) AS emar_id,
                cast(NULL AS VARCHAR) AS detail_route,
                cast(NULL AS VARCHAR) AS administration_type,
                cast(NULL AS VARCHAR) AS dose_given_raw,
                cast(NULL AS VARCHAR) AS dose_given_unit_raw
            WHERE FALSE
        """)

    # Fallback order route by pharmacy_id.
    if rx_path is not None:
        rx = source_sql(rx_path)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE rx_route AS
            SELECT
                pharmacy_id,
                max(upper(trim(cast(route AS VARCHAR))))
                    FILTER (WHERE route IS NOT NULL AND trim(cast(route AS VARCHAR))<>'') AS rx_route
            FROM {rx}
            WHERE pharmacy_id IS NOT NULL
            GROUP BY pharmacy_id
        """)
    else:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE rx_route AS
            SELECT cast(NULL AS BIGINT) AS pharmacy_id, cast(NULL AS VARCHAR) AS rx_route
            WHERE FALSE
        """)

    # ---------------------------------------------------------------
    # Candidate steroid eMAR events.
    # Map medication name and explicitly reject obvious local formulations.
    # ---------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE emar_steroid_raw AS
        SELECT
            f.stay_id,
            f.subject_id,
            f.hadm_id,
            f.severe_t0_broad,
            f.phenotype_broad,
            f.early_steroid_any AS rx_early_steroid_stage1,
            e.emar_id,
            e.pharmacy_id,
            e.charttime,
            e.scheduletime,
            e.medication,
            e.event_txt,
            d.detail_route,
            r.rx_route,
            coalesce(d.detail_route, r.rx_route) AS route_best,
            d.administration_type,
            d.dose_given_raw,
            d.dose_given_unit_raw,
            CASE
                WHEN lower(e.medication) LIKE '%hydrocortisone%' THEN 'hydrocortisone'
                WHEN lower(e.medication) LIKE '%methylpred%' THEN 'methylprednisolone'
                WHEN lower(e.medication) LIKE '%dexameth%' THEN 'dexamethasone'
                WHEN lower(e.medication) LIKE '%prednisolone%' THEN 'prednisolone'
                WHEN lower(e.medication) LIKE '%prednisone%' THEN 'prednisone'
                ELSE NULL
            END AS steroid_name
        FROM fc f
        JOIN {emar} e
          ON e.subject_id=f.subject_id
         AND e.hadm_id=f.hadm_id
         AND e.charttime BETWEEN f.severe_t0_broad - INTERVAL '{LOOKBACK_HOURS} hours'
                             AND f.severe_t0_broad + INTERVAL '{EARLY_HOURS} hours'
        LEFT JOIN emar_detail_route d USING (emar_id)
        LEFT JOIN rx_route r USING (pharmacy_id)
        WHERE (
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

    # Route classification.
    # Null route is retained as unknown for audit, but NOT counted in the strict
    # systemic-admin primary exposure.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE emar_steroid AS
        SELECT *,
            CASE
                WHEN route_best IS NULL OR trim(route_best)='' THEN 'unknown'
                WHEN upper(route_best) IN (
                    'IV','IVP','IVPB','IV PUSH','IV BOLUS',
                    'PO','PO/NG','NG','NGT','OG','OGT','GT','G-TUBE','PEG',
                    'IM'
                ) THEN 'systemic'
                WHEN upper(route_best) LIKE 'IV%' THEN 'systemic'
                WHEN upper(route_best) LIKE 'PO%' THEN 'systemic'
                WHEN upper(route_best) LIKE '%NG%' THEN 'systemic'
                WHEN upper(route_best) LIKE '%G TUBE%' THEN 'systemic'
                WHEN upper(route_best) LIKE '%G-TUBE%' THEN 'systemic'
                ELSE 'non_systemic_or_other'
            END AS route_class,
            date_diff('minute', severe_t0_broad, charttime)/60.0 AS hours_after_t0
        FROM emar_steroid_raw
    """)

    # Strict administered systemic event.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE strict_admin_events AS
        SELECT *
        FROM emar_steroid
        WHERE lower(trim(coalesce(event_txt,'')))='administered'
          AND route_class='systemic'
          AND charttime >= severe_t0_broad
          AND charttime <= severe_t0_broad + INTERVAL '{EARLY_HOURS} hours'
    """)

    # Sensitivity: administered + unknown route.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE administered_any_route AS
        SELECT *
        FROM emar_steroid
        WHERE lower(trim(coalesce(event_txt,'')))='administered'
          AND route_class IN ('systemic','unknown')
          AND charttime >= severe_t0_broad
          AND charttime <= severe_t0_broad + INTERVAL '{EARLY_HOURS} hours'
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE exposure_by_stay AS
        SELECT
            f.stay_id,
            max(CASE WHEN s.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS emar_strict_early_steroid,
            max(CASE WHEN a.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS emar_admin_anyroute_early_steroid
        FROM fc f
        LEFT JOIN (SELECT DISTINCT stay_id FROM strict_admin_events) s USING (stay_id)
        LEFT JOIN (SELECT DISTINCT stay_id FROM administered_any_route) a USING (stay_id)
        GROUP BY f.stay_id
    """)

    strict_n = scalar(con, "SELECT sum(emar_strict_early_steroid) FROM exposure_by_stay")
    anyroute_n = scalar(con, "SELECT sum(emar_admin_anyroute_early_steroid) FROM exposure_by_stay")

    rows = []
    add(rows, "coverage", "severe_CAP_total", severe_n, severe_n)
    add(rows, "coverage", "eMAR_observable_around_t0", observable_n, severe_n,
        "Any eMAR record from t0-24h to t0+24h")
    add(rows, "exposure", "early_steroid_eMAR_administered_systemic_strict", strict_n, observable_n,
        "Primary administration-level feasibility exposure")
    add(rows, "exposure", "early_steroid_eMAR_administered_systemic_or_unknown_route", anyroute_n, observable_n,
        "Sensitivity; includes administered events with missing route")

    # eMAR event_txt distribution among target steroid events.
    event_txt_df = con.execute("""
        SELECT
            coalesce(event_txt,'<NULL>') AS event_txt,
            route_class,
            count(*) AS event_rows,
            count(DISTINCT stay_id) AS patient_n
        FROM emar_steroid
        GROUP BY 1,2
        ORDER BY patient_n DESC, event_rows DESC
    """).fetchdf()
    event_txt_df.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")

    # Drug counts among strict administrations.
    drug_df = con.execute("""
        SELECT steroid_name, count(DISTINCT stay_id) AS n
        FROM strict_admin_events
        GROUP BY steroid_name
        ORDER BY n DESC
    """).fetchall()
    for drug, n in drug_df:
        add(rows, "drug_strict_emar", drug, n, strict_n,
            "Patient-level; drug categories can overlap")

    # First administration timing.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE first_strict AS
        SELECT stay_id, min(hours_after_t0) AS h
        FROM strict_admin_events
        GROUP BY stay_id
    """)
    for label, cond in [
        ("0_to_6h", "h>=0 AND h<=6"),
        (">6_to_12h", "h>6 AND h<=12"),
        (">12_to_24h", "h>12 AND h<=24"),
    ]:
        n = scalar(con, f"SELECT count(*) FROM first_strict WHERE {cond}")
        add(rows, "timing_first_strict_emar", label, n, strict_n)

    # Phenotype exposure prevalence in eMAR-observable patients.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE analysis_obs AS
        SELECT
            f.*,
            e.emar_strict_early_steroid,
            CASE WHEN o.stay_id IS NOT NULL THEN 1 ELSE 0 END AS emar_observable
        FROM fc f
        LEFT JOIN exposure_by_stay e USING (stay_id)
        LEFT JOIN emar_observable o USING (stay_id)
    """)
    for phenotype in ["respiratory_only", "shock_only", "combined"]:
        den = scalar(con, """
            SELECT count(*) FROM analysis_obs
            WHERE emar_observable=1 AND phenotype_broad=?
        """, [phenotype])
        n = scalar(con, """
            SELECT count(*) FROM analysis_obs
            WHERE emar_observable=1 AND phenotype_broad=?
              AND emar_strict_early_steroid=1
        """, [phenotype])
        add(rows, "phenotype_emar_observable", phenotype, n, den,
            "Strict eMAR administered systemic steroid")

    # Pre-t0 steroid administrations.
    pre_n = scalar(con, f"""
        SELECT count(DISTINCT stay_id)
        FROM emar_steroid
        WHERE lower(trim(coalesce(event_txt,'')))='administered'
          AND route_class='systemic'
          AND charttime < severe_t0_broad
          AND charttime >= severe_t0_broad - INTERVAL '{LOOKBACK_HOURS} hours'
    """)
    add(rows, "prevalent_exposure", "systemic_steroid_administered_t0_minus_24h_to_t0", pre_n, observable_n,
        "Important for new-user / washout sensitivity analysis")

    # Route availability.
    unknown_route_n = scalar(con, """
        SELECT count(DISTINCT stay_id)
        FROM emar_steroid
        WHERE lower(trim(coalesce(event_txt,'')))='administered'
          AND charttime >= severe_t0_broad
          AND charttime <= severe_t0_broad + INTERVAL '24 hours'
          AND route_class='unknown'
    """)
    add(rows, "data_quality", "administered_steroid_with_unknown_route", unknown_route_n, observable_n)

    # Stage-1 prescription vs eMAR concordance, restricted to eMAR-observable patients.
    concord = con.execute("""
        SELECT
            CASE WHEN f.early_steroid_any=1 THEN 1 ELSE 0 END AS rx_order_early,
            CASE WHEN coalesce(x.emar_strict_early_steroid,0)=1 THEN 1 ELSE 0 END AS emar_admin_early,
            count(*) AS n
        FROM fc f
        JOIN emar_observable o USING (stay_id)
        LEFT JOIN exposure_by_stay x USING (stay_id)
        GROUP BY 1,2
        ORDER BY 1 DESC,2 DESC
    """).fetchdf()
    concord["label"] = concord.apply(
        lambda r:
            "both_rx_and_emar" if r.rx_order_early==1 and r.emar_admin_early==1 else
            "rx_only" if r.rx_order_early==1 and r.emar_admin_early==0 else
            "emar_only" if r.rx_order_early==0 and r.emar_admin_early==1 else
            "neither",
        axis=1
    )
    concord.to_csv(OUT_CONCORD, index=False, encoding="utf-8-sig")

    # Add concordance rows to main summary.
    for _, r in concord.iterrows():
        add(rows, "rx_vs_emar_concordance", r["label"], int(r["n"]), observable_n)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")

    print("\n===== STAGE 3 SUMMARY =====")
    print(out.to_string(index=False))
    print("\nSaved:")
    print(f"  {OUT_MAIN}")
    print(f"  {OUT_EVENTS}")
    print(f"  {OUT_CONCORD}")
    print("\nGenerated exposure-audit summary files: 07_EMAR_EXPOSURE_AUDIT.csv and 07C_EMAR_RX_CONCORDANCE.csv.")
    print("07B is useful only if event_txt/route classification needs debugging.")

    con.close()


if __name__ == "__main__":
    main()

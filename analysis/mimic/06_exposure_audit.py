from __future__ import annotations

"""
Stage 2 exposure / indication audit for the MIMIC severe-CAP steroid project.

Put this file in the project root (same folder as config.py) and run it in PyCharm.
It DOES NOT export patient-level data.

Required from Stage 1:
    outputs/feasibility_cohort.parquet
    cache/feasibility.duckdb  (preferred, because it contains steroid_all_events)

Also uses MIMIC diagnoses_icd and, when available, the Stage-1 steroid prescription cache.

Outputs:
    outputs/06_EXPOSURE_AUDIT.csv
    outputs/06B_PRESCRIPTION_DOSE_ROUTE_AUDIT.csv   (if prescription cache exists)
"""

from pathlib import Path
import sys

import duckdb
import pandas as pd

import config
from src.utils import connect_duckdb, find_table, source_sql, sql_quote_path


PROJECT_ROOT = Path(config.PROJECT_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
CACHE_DIR = Path(config.CACHE_DIR)
DATA_ROOT = Path(config.DATA_ROOT)
EARLY_HOURS = int(getattr(config, "EARLY_STEROID_WINDOW_HOURS", 24))

COHORT_PARQUET = OUTPUT_DIR / "feasibility_cohort.parquet"
OUT_MAIN = OUTPUT_DIR / "06_EXPOSURE_AUDIT.csv"
OUT_DOSE = OUTPUT_DIR / "06B_PRESCRIPTION_DOSE_ROUTE_AUDIT.csv"


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT count(*) > 0
            FROM information_schema.tables
            WHERE lower(table_name) = lower(?)
            """,
            [name],
        ).fetchone()[0]
    )


def add_row(rows, section, item, n, denominator_n=None, notes=""):
    n = int(n or 0)
    if denominator_n is None:
        pct = None
    else:
        denominator_n = int(denominator_n or 0)
        pct = None if denominator_n == 0 else round(100.0 * n / denominator_n, 1)
    rows.append(
        {
            "section": section,
            "item": item,
            "n": n,
            "denominator_n": denominator_n,
            "pct": pct,
            "notes": notes,
        }
    )


def scalar(con, sql: str, params=None) -> int:
    params = params or []
    value = con.execute(sql, params).fetchone()[0]
    return int(value or 0)


def qcount(con, where="TRUE") -> int:
    return scalar(con, f"SELECT count(*) FROM fc WHERE {where}")


def main():
    print("=" * 72)
    print("MIMIC severe-CAP steroid project: Stage 2 exposure audit")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not COHORT_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing: {COHORT_PARQUET}\n"
            "Please run 01_run_feasibility.py first."
        )

    con = connect_duckdb(PROJECT_ROOT, threads=getattr(config, "DUCKDB_THREADS", 4))

    # Always read the exported cohort so this script is tied to the exact Stage-1 result.
    cohort_sql = f"read_parquet('{sql_quote_path(COHORT_PARQUET)}')"
    con.execute(f"CREATE OR REPLACE TEMP TABLE fc AS SELECT * FROM {cohort_sql}")

    severe_n = qcount(con, "severe_cap_broad_proxy=1")
    early_any_n = qcount(con, "severe_cap_broad_proxy=1 AND early_steroid_any=1")
    early_admin_n = qcount(con, "severe_cap_broad_proxy=1 AND early_steroid_admin=1")

    # ------------------------------------------------------------------
    # Diagnosis audit: COPD / asthma and acute-exacerbation proxies.
    # These are AUDIT FLAGS only; do not automatically exclude patients yet.
    # ------------------------------------------------------------------
    dx_path = find_table(DATA_ROOT, "hosp", "diagnoses_icd")
    dx = source_sql(dx_path)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE dx_audit AS
        WITH x AS (
            SELECT
                hadm_id,
                try_cast(icd_version AS INTEGER) AS ver,
                upper(replace(trim(cast(icd_code AS VARCHAR)), '.', '')) AS code
            FROM {dx}
        )
        SELECT
            hadm_id,
            max(CASE WHEN
                (ver=10 AND (code LIKE 'J41%' OR code LIKE 'J42%' OR code LIKE 'J43%' OR code LIKE 'J44%'))
                OR
                (ver=9 AND (code LIKE '490%' OR code LIKE '491%' OR code LIKE '492%' OR code LIKE '496%'))
                THEN 1 ELSE 0 END) AS copd_any,
            max(CASE WHEN
                (ver=10 AND code LIKE 'J441%')
                OR
                (ver=9 AND code LIKE '49121%')
                THEN 1 ELSE 0 END) AS copd_exacerbation_proxy,
            max(CASE WHEN
                (ver=10 AND code LIKE 'J45%')
                OR
                (ver=9 AND code LIKE '493%')
                THEN 1 ELSE 0 END) AS asthma_any,
            max(CASE WHEN
                (ver=10 AND code LIKE 'J45%' AND right(code,1)='1')
                OR
                (ver=9 AND code LIKE '493%' AND right(code,1)='2')
                THEN 1 ELSE 0 END) AS asthma_exacerbation_proxy
        FROM x
        GROUP BY hadm_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE fc2 AS
        SELECT
            f.*,
            coalesce(d.copd_any,0) AS copd_any,
            coalesce(d.copd_exacerbation_proxy,0) AS copd_exacerbation_proxy,
            coalesce(d.asthma_any,0) AS asthma_any,
            coalesce(d.asthma_exacerbation_proxy,0) AS asthma_exacerbation_proxy,
            CASE WHEN coalesce(d.copd_any,0)=1 OR coalesce(d.asthma_any,0)=1 THEN 1 ELSE 0 END AS copd_or_asthma,
            CASE WHEN coalesce(d.copd_exacerbation_proxy,0)=1 OR coalesce(d.asthma_exacerbation_proxy,0)=1 THEN 1 ELSE 0 END AS obstructive_exacerbation_proxy
        FROM fc f
        LEFT JOIN dx_audit d USING (hadm_id)
        """
    )
    con.execute("CREATE OR REPLACE TEMP TABLE fc AS SELECT * FROM fc2")

    # ------------------------------------------------------------------
    # Steroid event audit. Prefer persisted Stage-1 table.
    # If it does not exist, reconstruct from caches.
    # ------------------------------------------------------------------
    if table_exists(con, "steroid_all_events"):
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW steroid_events AS
            SELECT stay_id, event_time, steroid_name, raw_name, source
            FROM steroid_all_events
            """
        )
    else:
        input_cache = CACHE_DIR / "target_inputevents.parquet"
        rx_cache = CACHE_DIR / "steroid_prescriptions.parquet"
        if not input_cache.exists() or not rx_cache.exists():
            raise FileNotFoundError(
                "Stage-1 steroid event tables/caches were not found.\n"
                "Please run 01_run_feasibility.py once, then rerun this script."
            )

        inp = f"read_parquet('{sql_quote_path(input_cache)}')"
        rx = f"read_parquet('{sql_quote_path(rx_cache)}')"
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE steroid_events AS
            SELECT
                i.stay_id,
                i.starttime AS event_time,
                i.target_name AS steroid_name,
                i.label AS raw_name,
                'inputevents' AS source
            FROM {inp} i
            JOIN fc f USING (stay_id)
            WHERE i.target_class='steroid'
              AND f.severe_t0_broad IS NOT NULL
              AND i.starttime BETWEEN f.severe_t0_broad - INTERVAL '24 hours'
                                  AND f.severe_t0_broad + INTERVAL '{EARLY_HOURS} hours'
            UNION ALL
            SELECT
                f.stay_id,
                r.starttime AS event_time,
                r.steroid_name,
                r.drug AS raw_name,
                'prescriptions' AS source
            FROM {rx} r
            JOIN fc f USING (hadm_id, subject_id)
            WHERE f.severe_t0_broad IS NOT NULL
              AND r.starttime BETWEEN f.severe_t0_broad - INTERVAL '24 hours'
                                  AND f.severe_t0_broad + INTERVAL '{EARLY_HOURS} hours'
            """
        )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE early_events AS
        SELECT
            f.stay_id,
            f.hadm_id,
            f.severe_t0_broad,
            f.phenotype_broad,
            f.hospital_death,
            e.event_time,
            lower(coalesce(e.steroid_name,'unknown')) AS steroid_name,
            e.raw_name,
            e.source,
            date_diff('minute', f.severe_t0_broad, e.event_time) / 60.0 AS hours_after_t0
        FROM fc f
        JOIN steroid_events e USING (stay_id)
        WHERE f.severe_cap_broad_proxy=1
          AND e.event_time >= f.severe_t0_broad
          AND e.event_time <= f.severe_t0_broad + INTERVAL '{EARLY_HOURS} hours'
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE first_early_any AS
        SELECT stay_id, min(event_time) AS first_event_time
        FROM early_events
        GROUP BY stay_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE first_early_admin AS
        SELECT stay_id, min(event_time) AS first_event_time
        FROM early_events
        WHERE source='inputevents'
        GROUP BY stay_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE steroid_types_by_stay AS
        SELECT stay_id, count(DISTINCT steroid_name) AS n_steroid_types
        FROM early_events
        GROUP BY stay_id
        """
    )

    rows = []

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------
    add_row(rows, "overview", "severe_CAP_broad_proxy", severe_n, severe_n,
            "Denominator for this audit")
    add_row(rows, "overview", "early_systemic_steroid_any_source_0_24h", early_any_n, severe_n,
            "Stage-1 definition: inputevents OR prescription start time")
    add_row(rows, "overview", "early_steroid_inputevents_admin_like_0_24h", early_admin_n, severe_n,
            "More administration-like; may be smaller than any-source exposure")
    add_row(rows, "overview", "steroid_in_24h_before_t0", qcount(con, "severe_cap_broad_proxy=1 AND steroid_pre_t0_24h=1"), severe_n,
            "Important prevalent/other-indication signal")

    # ------------------------------------------------------------------
    # Drug-specific patient counts. Counts overlap if a patient got >1 steroid.
    # ------------------------------------------------------------------
    drug_names = ["hydrocortisone", "methylprednisolone", "dexamethasone", "prednisone", "prednisolone", "unknown"]
    for drug in drug_names:
        n_any = scalar(
            con,
            "SELECT count(DISTINCT stay_id) FROM early_events WHERE steroid_name=?",
            [drug],
        )
        if n_any > 0 or drug != "unknown":
            add_row(rows, "drug_any_source", drug, n_any, early_any_n,
                    "Patient-level count; drug categories can overlap")

        n_admin = scalar(
            con,
            "SELECT count(DISTINCT stay_id) FROM early_events WHERE steroid_name=? AND source='inputevents'",
            [drug],
        )
        if n_admin > 0 or drug != "unknown":
            add_row(rows, "drug_inputevents_admin_like", drug, n_admin, early_admin_n,
                    "Patient-level count; inputevents only")

    # Number of distinct steroid types used within 24 h.
    one_type = scalar(con, "SELECT count(*) FROM steroid_types_by_stay WHERE n_steroid_types=1")
    multi_type = scalar(con, "SELECT count(*) FROM steroid_types_by_stay WHERE n_steroid_types>=2")
    add_row(rows, "drug_mix", "exactly_1_steroid_type", one_type, early_any_n)
    add_row(rows, "drug_mix", "2_or_more_steroid_types", multi_type, early_any_n,
            "Useful for detecting switches/overlap")

    # ------------------------------------------------------------------
    # Timing of FIRST early steroid after t0.
    # ------------------------------------------------------------------
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE timing_any AS
        SELECT
            f.stay_id,
            date_diff('minute', f.severe_t0_broad, x.first_event_time)/60.0 AS h
        FROM fc f JOIN first_early_any x USING (stay_id)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE timing_admin AS
        SELECT
            f.stay_id,
            date_diff('minute', f.severe_t0_broad, x.first_event_time)/60.0 AS h
        FROM fc f JOIN first_early_admin x USING (stay_id)
        """
    )

    timing_bins = [
        ("0_to_6h", "h >= 0 AND h <= 6"),
        (">6_to_12h", "h > 6 AND h <= 12"),
        (">12_to_24h", "h > 12 AND h <= 24"),
    ]
    for label, where in timing_bins:
        add_row(rows, "timing_first_any_source", label,
                scalar(con, f"SELECT count(*) FROM timing_any WHERE {where}"), early_any_n)
        add_row(rows, "timing_first_admin_like", label,
                scalar(con, f"SELECT count(*) FROM timing_admin WHERE {where}"), early_admin_n)

    # ------------------------------------------------------------------
    # Physiologic phenotypes
    # ------------------------------------------------------------------
    phenotypes = ["respiratory_only", "shock_only", "combined", "other"]
    for ph in phenotypes:
        denom = qcount(con, f"severe_cap_broad_proxy=1 AND phenotype_broad='{ph}'")
        if denom == 0:
            continue
        trt = qcount(con, f"severe_cap_broad_proxy=1 AND phenotype_broad='{ph}' AND early_steroid_any=1")
        add_row(rows, "phenotype", ph, trt, denom,
                "n = early steroid patients; denominator = all patients in phenotype")

    # ------------------------------------------------------------------
    # COPD/asthma indication audit
    # ------------------------------------------------------------------
    dx_flags = [
        ("COPD_any", "copd_any=1"),
        ("COPD_exacerbation_proxy", "copd_exacerbation_proxy=1"),
        ("asthma_any", "asthma_any=1"),
        ("asthma_exacerbation_proxy", "asthma_exacerbation_proxy=1"),
        ("COPD_or_asthma", "copd_or_asthma=1"),
        ("obstructive_exacerbation_proxy", "obstructive_exacerbation_proxy=1"),
    ]
    for label, cond in dx_flags:
        denom = qcount(con, f"severe_cap_broad_proxy=1 AND {cond}")
        trt = qcount(con, f"severe_cap_broad_proxy=1 AND {cond} AND early_steroid_any=1")
        add_row(rows, "indication_audit", label + "__all_severe_CAP", denom, severe_n,
                "Prevalence in severe-CAP cohort")
        add_row(rows, "indication_audit", label + "__early_steroid", trt, early_any_n,
                "Prevalence among early-steroid patients")

    # ------------------------------------------------------------------
    # Organ support signals. These Stage-1 flags are descriptive and may occur
    # anywhere in the configured early support-search window, not necessarily at t0.
    # ------------------------------------------------------------------
    supports = [
        ("invasive_ventilation_signal", "any_invasive_device=1"),
        ("NIV_signal", "any_niv_device=1"),
        ("HFNC_any", "any_hfnc=1"),
        ("HFNC_flow_FiO2_qualified", "any_hfnc_remap_like=1"),
        ("vasopressor_support", "vasopressor_support=1"),
        ("strict_respiratory_support", "respiratory_support_strict=1"),
    ]
    for label, cond in supports:
        denom = qcount(con, f"severe_cap_broad_proxy=1 AND {cond}")
        trt = qcount(con, f"severe_cap_broad_proxy=1 AND {cond} AND early_steroid_any=1")
        add_row(rows, "organ_support", label, trt, denom,
                "n = treated; denominator = patients with this support signal")

    # ------------------------------------------------------------------
    # Simple descriptive mortality audit only (NOT causal effects).
    # ------------------------------------------------------------------
    for label, cond in [
        ("early_steroid_any", "early_steroid_any=1"),
        ("no_early_steroid", "early_steroid_any=0"),
    ]:
        denom = qcount(con, f"severe_cap_broad_proxy=1 AND {cond}")
        deaths = qcount(con, f"severe_cap_broad_proxy=1 AND {cond} AND hospital_death=1")
        add_row(rows, "descriptive_outcome_DO_NOT_INTERPRET_CAUSALLY", label + "__hospital_death", deaths, denom,
                "Unadjusted descriptive rate only; do NOT compare as treatment effect")

    # ------------------------------------------------------------------
    # Prescription route + raw dose audit (summary only; no patient-level export)
    # ------------------------------------------------------------------
    dose_df = None
    rx_cache = CACHE_DIR / "steroid_prescriptions.parquet"
    if rx_cache.exists():
        rx = f"read_parquet('{sql_quote_path(rx_cache)}')"
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE early_rx AS
            SELECT
                f.stay_id,
                lower(coalesce(r.steroid_name,'unknown')) AS steroid_name,
                r.drug,
                r.dose_val_rx,
                r.dose_unit_rx,
                r.doses_per_24_hrs,
                r.route,
                CASE
                    WHEN lower(coalesce(r.route,'')) LIKE '%iv%'
                      OR lower(coalesce(r.route,'')) LIKE '%intraven%'
                      OR lower(coalesce(r.route,'')) LIKE '%intravenous%'
                    THEN 'IV'
                    WHEN lower(coalesce(r.route,'')) LIKE '%po%'
                      OR lower(coalesce(r.route,'')) LIKE '%oral%'
                      OR lower(coalesce(r.route,'')) LIKE '%enteral%'
                    THEN 'PO_enteral'
                    WHEN trim(coalesce(r.route,''))='' THEN 'route_missing'
                    ELSE 'other_systemic_route'
                END AS route_group
            FROM {rx} r
            JOIN fc f USING (hadm_id, subject_id)
            WHERE f.severe_cap_broad_proxy=1
              AND f.severe_t0_broad IS NOT NULL
              AND r.starttime >= f.severe_t0_broad
              AND r.starttime <= f.severe_t0_broad + INTERVAL '{EARLY_HOURS} hours'
            """
        )

        route_df = con.execute(
            """
            SELECT steroid_name, route_group,
                   count(DISTINCT stay_id) AS patient_n
            FROM early_rx
            GROUP BY steroid_name, route_group
            ORDER BY steroid_name, patient_n DESC
            """
        ).fetchdf()
        for _, r in route_df.iterrows():
            drug_denom = scalar(
                con,
                "SELECT count(DISTINCT stay_id) FROM early_rx WHERE steroid_name=?",
                [r["steroid_name"]],
            )
            add_row(rows, "prescription_route", f"{r['steroid_name']}__{r['route_group']}",
                    int(r["patient_n"]), drug_denom,
                    "Prescription records only")

        # Raw dose/route combinations: summary only, mainly to inspect parseability and regimen patterns.
        dose_df = con.execute(
            """
            SELECT
                steroid_name,
                drug AS raw_drug,
                coalesce(cast(dose_val_rx AS VARCHAR),'') AS dose_val_rx,
                coalesce(cast(dose_unit_rx AS VARCHAR),'') AS dose_unit_rx,
                coalesce(cast(doses_per_24_hrs AS VARCHAR),'') AS doses_per_24_hrs,
                coalesce(cast(route AS VARCHAR),'') AS route,
                count(*) AS prescription_rows,
                count(DISTINCT stay_id) AS patient_n
            FROM early_rx
            GROUP BY ALL
            ORDER BY patient_n DESC, prescription_rows DESC
            LIMIT 250
            """
        ).fetchdf()

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")
    if dose_df is not None:
        dose_df.to_csv(OUT_DOSE, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Main audit: {OUT_MAIN}")
    if dose_df is not None:
        print(f"Dose/route audit: {OUT_DOSE}")
    else:
        print("Dose/route audit skipped: cache/steroid_prescriptions.parquet not found.")

    print("\n===== KEY RESULTS (copy/paste is okay) =====")
    key_sections = ["overview", "drug_any_source", "drug_mix", "timing_first_any_source", "phenotype", "indication_audit"]
    print(out[out["section"].isin(key_sections)].to_string(index=False))
    print("===== END =====\n")

    con.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR] Stage-2 audit failed:")
        print(exc)
        print("\nPlease copy the FULL error message back to ChatGPT; do not modify the data files.")
        raise

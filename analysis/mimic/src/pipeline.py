from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd

from .utils import find_table, try_find_table, source_sql, sql_quote_path, pct


RESP_ITEMIDS = [
    226732,  # O2 Delivery Device(s)
    223834,  # O2 flow
    227582,  # BiPAP O2 flow
    227287,  # additional O2 flow
    223835,  # FiO2
    223849,  # ventilator mode
    229314,  # ventilator mode (Hamilton)
    223848,  # ventilator type
]


@dataclass
class ProjectPaths:
    data_root: Path
    project_root: Path
    cache_dir: Path
    output_dir: Path


class FeasibilityPipeline:
    def __init__(self, con: duckdb.DuckDBPyConnection, cfg):
        self.con = con
        self.cfg = cfg
        self.paths = ProjectPaths(
            data_root=Path(cfg.DATA_ROOT),
            project_root=Path(cfg.PROJECT_ROOT),
            cache_dir=Path(cfg.CACHE_DIR),
            output_dir=Path(cfg.OUTPUT_DIR),
        )
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # File discovery / validation
    # ------------------------------------------------------------------
    def discover_tables(self) -> Dict[str, Path]:
        required = {
            "hosp.admissions": ("hosp", "admissions"),
            "hosp.patients": ("hosp", "patients"),
            "hosp.diagnoses_icd": ("hosp", "diagnoses_icd"),
            "hosp.prescriptions": ("hosp", "prescriptions"),
            "icu.icustays": ("icu", "icustays"),
            "icu.d_items": ("icu", "d_items"),
            "icu.inputevents": ("icu", "inputevents"),
            "icu.chartevents": ("icu", "chartevents"),
        }
        optional = {
            "hosp.d_icd_diagnoses": ("hosp", "d_icd_diagnoses"),
            "hosp.emar": ("hosp", "emar"),
            "hosp.labevents": ("hosp", "labevents"),
            "hosp.d_labitems": ("hosp", "d_labitems"),
        }
        found = {}
        for key, (module, table) in required.items():
            found[key] = find_table(self.paths.data_root, module, table)
        for key, (module, table) in optional.items():
            p = try_find_table(self.paths.data_root, module, table)
            if p is not None:
                found[key] = p
        self.tables = found
        return found

    def inventory_dataframe(self) -> pd.DataFrame:
        if not self.tables:
            self.discover_tables()
        rows = []
        for key, path in sorted(self.tables.items()):
            try:
                desc = self.con.execute(f"DESCRIBE SELECT * FROM {source_sql(path)} LIMIT 0").fetchdf()
                cols = ", ".join(desc["column_name"].astype(str).tolist())
                status = "OK"
            except Exception as e:  # pragma: no cover - diagnostic path
                cols = ""
                status = f"ERROR: {e}"
            rows.append({
                "table": key,
                "path": str(path),
                "size_gb": round(path.stat().st_size / (1024 ** 3), 3),
                "status": status,
                "columns": cols,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Reusable caches. These are deliberately narrow to keep reruns fast.
    # ------------------------------------------------------------------
    def build_resp_cache(self) -> Path:
        path = self.paths.cache_dir / "resp_relevant_chartevents.parquet"
        if path.exists() and not self.cfg.FORCE_REBUILD_CACHE:
            return path
        if path.exists():
            path.unlink()
        ce = source_sql(self.tables["icu.chartevents"])
        itemids = ",".join(str(x) for x in RESP_ITEMIDS)
        sql = f"""
        COPY (
            SELECT
                subject_id,
                hadm_id,
                stay_id,
                charttime,
                storetime,
                itemid,
                value,
                valuenum,
                valueuom
            FROM {ce}
            WHERE stay_id IS NOT NULL
              AND itemid IN ({itemids})
              AND value IS NOT NULL
        ) TO '{sql_quote_path(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        self.con.execute(sql)
        return path

    def build_input_cache(self) -> Path:
        path = self.paths.cache_dir / "target_inputevents.parquet"
        if path.exists() and not self.cfg.FORCE_REBUILD_CACHE:
            return path
        if path.exists():
            path.unlink()
        ie = source_sql(self.tables["icu.inputevents"])
        di = source_sql(self.tables["icu.d_items"])
        sql = f"""
        COPY (
            SELECT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.starttime,
                i.endtime,
                i.itemid,
                d.label,
                i.amount,
                i.amountuom,
                i.rate,
                i.rateuom,
                CASE
                    WHEN lower(d.label) LIKE '%norepinephrine%' THEN 'norepinephrine'
                    WHEN lower(d.label) LIKE '%epinephrine%' THEN 'epinephrine'
                    WHEN lower(d.label) LIKE '%phenylephrine%' THEN 'phenylephrine'
                    WHEN lower(d.label) LIKE '%vasopressin%' THEN 'vasopressin'
                    WHEN lower(d.label) LIKE '%dopamine%' THEN 'dopamine'
                    WHEN lower(d.label) LIKE '%angiotensin ii%' THEN 'angiotensin_ii'
                    WHEN lower(d.label) LIKE '%hydrocortisone%' THEN 'hydrocortisone'
                    WHEN lower(d.label) LIKE '%methylpred%' THEN 'methylprednisolone'
                    WHEN lower(d.label) LIKE '%dexameth%' THEN 'dexamethasone'
                    WHEN lower(d.label) LIKE '%prednisolone%' THEN 'prednisolone'
                    WHEN lower(d.label) LIKE '%prednisone%' THEN 'prednisone'
                    ELSE NULL
                END AS target_name,
                CASE
                    WHEN lower(d.label) LIKE '%norepinephrine%'
                      OR lower(d.label) LIKE '%epinephrine%'
                      OR lower(d.label) LIKE '%phenylephrine%'
                      OR lower(d.label) LIKE '%vasopressin%'
                      OR lower(d.label) LIKE '%dopamine%'
                      OR lower(d.label) LIKE '%angiotensin ii%'
                    THEN 'vasopressor'
                    WHEN lower(d.label) LIKE '%hydrocortisone%'
                      OR lower(d.label) LIKE '%methylpred%'
                      OR lower(d.label) LIKE '%dexameth%'
                      OR lower(d.label) LIKE '%prednisolone%'
                      OR lower(d.label) LIKE '%prednisone%'
                    THEN 'steroid'
                    ELSE NULL
                END AS target_class
            FROM {ie} i
            JOIN {di} d USING (itemid)
            WHERE
                lower(d.label) LIKE '%norepinephrine%'
                OR lower(d.label) LIKE '%epinephrine%'
                OR lower(d.label) LIKE '%phenylephrine%'
                OR lower(d.label) LIKE '%vasopressin%'
                OR lower(d.label) LIKE '%dopamine%'
                OR lower(d.label) LIKE '%angiotensin ii%'
                OR lower(d.label) LIKE '%hydrocortisone%'
                OR lower(d.label) LIKE '%methylpred%'
                OR lower(d.label) LIKE '%dexameth%'
                OR lower(d.label) LIKE '%prednisolone%'
                OR lower(d.label) LIKE '%prednisone%'
        ) TO '{sql_quote_path(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        self.con.execute(sql)
        return path

    def build_rx_steroid_cache(self) -> Path:
        path = self.paths.cache_dir / "steroid_prescriptions.parquet"
        if path.exists() and not self.cfg.FORCE_REBUILD_CACHE:
            return path
        if path.exists():
            path.unlink()
        rx = source_sql(self.tables["hosp.prescriptions"])
        sql = f"""
        COPY (
            SELECT
                subject_id,
                hadm_id,
                pharmacy_id,
                starttime,
                stoptime,
                drug,
                dose_val_rx,
                dose_unit_rx,
                doses_per_24_hrs,
                route,
                CASE
                    WHEN lower(drug) LIKE '%hydrocortisone%' THEN 'hydrocortisone'
                    WHEN lower(drug) LIKE '%methylpred%' THEN 'methylprednisolone'
                    WHEN lower(drug) LIKE '%dexameth%' THEN 'dexamethasone'
                    WHEN lower(drug) LIKE '%prednisolone%' THEN 'prednisolone'
                    WHEN lower(drug) LIKE '%prednisone%' THEN 'prednisone'
                    ELSE 'other_steroid'
                END AS steroid_name
            FROM {rx}
            WHERE
                (
                    lower(drug) LIKE '%hydrocortisone%'
                    OR lower(drug) LIKE '%methylpred%'
                    OR lower(drug) LIKE '%dexameth%'
                    OR lower(drug) LIKE '%prednisolone%'
                    OR lower(drug) LIKE '%prednisone%'
                )
                AND NOT (
                    lower(coalesce(route,'')) LIKE '%ophth%'
                    OR lower(coalesce(route,'')) LIKE '%top%'
                    OR lower(coalesce(route,'')) LIKE '%otic%'
                    OR lower(coalesce(route,'')) LIKE '%nasal%'
                    OR lower(coalesce(route,'')) LIKE '%inh%'
                    OR lower(coalesce(route,'')) LIKE '%neb%'
                )
        ) TO '{sql_quote_path(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        self.con.execute(sql)
        return path

    # ------------------------------------------------------------------
    # Cohort assembly
    # ------------------------------------------------------------------
    def build_base_cohort(self) -> None:
        admissions = source_sql(self.tables["hosp.admissions"])
        patients = source_sql(self.tables["hosp.patients"])
        icustays = source_sql(self.tables["icu.icustays"])
        dx = source_sql(self.tables["hosp.diagnoses_icd"])

        self.con.execute(f"""
        CREATE OR REPLACE TABLE base_icu AS
        WITH first_icu AS (
            SELECT *,
                   row_number() OVER (PARTITION BY hadm_id ORDER BY intime, stay_id) AS rn
            FROM {icustays}
        ),
        dx_flags AS (
            SELECT
                hadm_id,
                max(CASE WHEN
                    (icd_version = 9 AND regexp_matches(replace(upper(icd_code),'.',''), '^48[0-6]'))
                    OR
                    (icd_version = 10 AND regexp_matches(replace(upper(icd_code),'.',''), '^J1[2-8]'))
                    THEN 1 ELSE 0 END) AS pneumonia_dx,
                max(CASE WHEN seq_num <= 3 AND (
                    (icd_version = 9 AND regexp_matches(replace(upper(icd_code),'.',''), '^48[0-6]'))
                    OR
                    (icd_version = 10 AND regexp_matches(replace(upper(icd_code),'.',''), '^J1[2-8]'))
                    ) THEN 1 ELSE 0 END) AS pneumonia_top3_dx,
                max(CASE WHEN
                    (icd_version = 9 AND regexp_matches(replace(upper(icd_code),'.',''), '^(487|488)'))
                    OR
                    (icd_version = 10 AND regexp_matches(replace(upper(icd_code),'.',''), '^J0[9]|^J1[01]'))
                    THEN 1 ELSE 0 END) AS influenza_dx,
                max(CASE WHEN
                    icd_version = 10 AND replace(upper(icd_code),'.','') = 'U071'
                    THEN 1 ELSE 0 END) AS covid_dx,
                max(CASE WHEN
                    (icd_version = 9 AND regexp_matches(replace(upper(icd_code),'.',''), '^507'))
                    OR
                    (icd_version = 10 AND regexp_matches(replace(upper(icd_code),'.',''), '^J69'))
                    THEN 1 ELSE 0 END) AS aspiration_dx,
                max(CASE WHEN
                    (icd_version = 9 AND regexp_matches(replace(upper(icd_code),'.',''), '^(1363|1173)'))
                    OR
                    (icd_version = 10 AND regexp_matches(replace(upper(icd_code),'.',''), '^(B59|B44)'))
                    THEN 1 ELSE 0 END) AS opportunistic_pna_dx
            FROM {dx}
            GROUP BY hadm_id
        )
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.first_careunit,
            i.last_careunit,
            i.intime,
            i.outtime,
            i.los AS icu_los_days,
            a.admittime,
            a.dischtime,
            a.deathtime,
            a.admission_type,
            a.admission_location,
            a.discharge_location,
            a.insurance,
            a.race,
            a.hospital_expire_flag,
            p.gender,
            p.anchor_age,
            p.anchor_year,
            p.dod,
            (p.anchor_age + extract(year from a.admittime) - p.anchor_year) AS age,
            date_diff('hour', a.admittime, i.intime) AS hospital_to_icu_hours,
            coalesce(d.pneumonia_dx,0) AS pneumonia_dx,
            coalesce(d.pneumonia_top3_dx,0) AS pneumonia_top3_dx,
            coalesce(d.influenza_dx,0) AS influenza_dx,
            coalesce(d.covid_dx,0) AS covid_dx,
            coalesce(d.aspiration_dx,0) AS aspiration_dx,
            coalesce(d.opportunistic_pna_dx,0) AS opportunistic_pna_dx
        FROM first_icu i
        JOIN {admissions} a USING (hadm_id, subject_id)
        JOIN {patients} p USING (subject_id)
        LEFT JOIN dx_flags d USING (hadm_id)
        WHERE i.rn = 1
        """)

        self.con.execute(f"""
        CREATE OR REPLACE TABLE cap_cohort AS
        SELECT *,
            CASE WHEN age >= 18 THEN 1 ELSE 0 END AS adult,
            CASE WHEN hospital_to_icu_hours BETWEEN 0 AND {int(self.cfg.ICU_WITHIN_HOSPITAL_HOURS)} THEN 1 ELSE 0 END AS early_icu,
            CASE WHEN
                age >= 18
                AND hospital_to_icu_hours BETWEEN 0 AND {int(self.cfg.ICU_WITHIN_HOSPITAL_HOURS)}
                AND pneumonia_dx = 1
                THEN 1 ELSE 0 END AS cap_broad,
            CASE WHEN
                age >= 18
                AND hospital_to_icu_hours BETWEEN 0 AND {int(self.cfg.ICU_WITHIN_HOSPITAL_HOURS)}
                AND pneumonia_dx = 1
                AND influenza_dx = 0
                AND covid_dx = 0
                AND aspiration_dx = 0
                AND opportunistic_pna_dx = 0
                THEN 1 ELSE 0 END AS cap_proxy
        FROM base_icu
        """)

    def build_support_events(self, resp_cache: Path, input_cache: Path) -> None:
        resp = source_sql(resp_cache)
        inp = source_sql(input_cache)
        pre = int(self.cfg.SUPPORT_PRE_ICU_HOURS)
        post = int(self.cfg.SUPPORT_POST_ICU_HOURS)
        pair = int(self.cfg.HFNC_PAIR_WINDOW_HOURS)
        min_flow = float(self.cfg.HFNC_MIN_FLOW_LPM)
        min_fio2 = float(self.cfg.HFNC_MIN_FIO2_PERCENT)
        phenotype_h = int(self.cfg.PHENOTYPE_WINDOW_HOURS)

        # Pull only CAP candidate stays and only a clinically relevant time window.
        self.con.execute(f"""
        CREATE OR REPLACE TABLE resp_window AS
        SELECT r.*,
               c.intime,
               c.outtime
        FROM {resp} r
        JOIN cap_cohort c USING (stay_id)
        WHERE c.cap_proxy = 1
          AND r.charttime BETWEEN c.intime - INTERVAL '{pre} hours'
                              AND c.intime + INTERVAL '{post} hours'
        """)

        # Device / ventilator signals. We intentionally keep a broad signal and a stricter HFNC signal.
        self.con.execute("""
        CREATE OR REPLACE TABLE respiratory_device_events AS
        SELECT
            stay_id,
            charttime,
            max(CASE WHEN itemid = 226732 THEN value ELSE NULL END) AS device_value,
            max(CASE WHEN itemid IN (223849,229314) THEN value ELSE NULL END) AS vent_mode,
            max(CASE WHEN itemid = 223848 THEN value ELSE NULL END) AS vent_type,
            max(CASE WHEN itemid = 223835 THEN
                CASE
                    WHEN valuenum BETWEEN 0.20 AND 1.0 THEN valuenum * 100.0
                    WHEN valuenum >= 20 AND valuenum <= 100 THEN valuenum
                    ELSE NULL
                END ELSE NULL END) AS fio2_percent_same_time,
            max(CASE WHEN itemid IN (223834,227582,227287) THEN valuenum ELSE NULL END) AS o2_flow_same_time,
            max(CASE WHEN itemid = 226732 AND (
                    lower(value) LIKE '%endotracheal%'
                    OR lower(value) LIKE '%tracheostomy%'
                    OR lower(value) LIKE '%trach%tube%'
                ) THEN 1 ELSE 0 END) AS invasive_device,
            max(CASE WHEN itemid = 226732 AND (
                    lower(value) LIKE '%bipap%'
                    OR lower(value) LIKE '%cpap%'
                    OR lower(value) LIKE '%non-invasive%'
                    OR lower(value) LIKE '%noninvasive%'
                ) THEN 1 ELSE 0 END) AS niv_device,
            max(CASE WHEN itemid = 226732 AND lower(value) LIKE '%high flow%' THEN 1 ELSE 0 END) AS hfnc_device,
            max(CASE WHEN itemid IN (223849,229314,223848)
                       AND value IS NOT NULL
                       AND trim(value) <> ''
                       AND lower(trim(value)) NOT IN ('other','other/remarks','standby')
                THEN 1 ELSE 0 END) AS ventilator_signal
        FROM resp_window
        GROUP BY stay_id, charttime
        """)

        # HFNC often has flow/FiO2 charted on neighboring timestamps. Pair within +/- configured hours.
        self.con.execute(f"""
        CREATE OR REPLACE TABLE hfnc_qualified_events AS
        WITH h AS (
            SELECT stay_id, charttime
            FROM respiratory_device_events
            WHERE hfnc_device = 1
        ),
        fio2 AS (
            SELECT stay_id, charttime,
                   CASE
                     WHEN valuenum BETWEEN 0.20 AND 1.0 THEN valuenum * 100.0
                     WHEN valuenum >= 20 AND valuenum <= 100 THEN valuenum
                     ELSE NULL END AS fio2_percent
            FROM resp_window
            WHERE itemid = 223835
        ),
        flow AS (
            SELECT stay_id, charttime, valuenum AS flow_lpm
            FROM resp_window
            WHERE itemid IN (223834,227582,227287)
              AND valuenum IS NOT NULL
        )
        SELECT
            h.stay_id,
            h.charttime,
            max(fio2.fio2_percent) AS nearby_fio2_percent,
            max(flow.flow_lpm) AS nearby_flow_lpm,
            CASE WHEN max(fio2.fio2_percent) >= {min_fio2}
                       AND max(flow.flow_lpm) >= {min_flow}
                 THEN 1 ELSE 0 END AS hfnc_remap_like
        FROM h
        LEFT JOIN fio2
          ON h.stay_id = fio2.stay_id
         AND abs(date_diff('minute', h.charttime, fio2.charttime)) <= {pair * 60}
        LEFT JOIN flow
          ON h.stay_id = flow.stay_id
         AND abs(date_diff('minute', h.charttime, flow.charttime)) <= {pair * 60}
        GROUP BY h.stay_id, h.charttime
        """)

        self.con.execute("""
        CREATE OR REPLACE TABLE respiratory_support_events AS
        SELECT
            d.stay_id,
            d.charttime,
            d.invasive_device,
            d.niv_device,
            d.hfnc_device,
            d.ventilator_signal,
            coalesce(h.hfnc_remap_like,0) AS hfnc_remap_like,
            CASE WHEN
                d.invasive_device = 1
                OR d.niv_device = 1
                OR d.ventilator_signal = 1
                OR d.hfnc_device = 1
                THEN 1 ELSE 0 END AS resp_support_broad,
            CASE WHEN
                d.invasive_device = 1
                OR d.niv_device = 1
                OR d.ventilator_signal = 1
                OR coalesce(h.hfnc_remap_like,0) = 1
                THEN 1 ELSE 0 END AS resp_support_strict
        FROM respiratory_device_events d
        LEFT JOIN hfnc_qualified_events h USING (stay_id, charttime)
        WHERE
            d.invasive_device = 1
            OR d.niv_device = 1
            OR d.ventilator_signal = 1
            OR d.hfnc_device = 1
        """)

        # Vasopressor events from ICU inputevents.
        self.con.execute(f"""
        CREATE OR REPLACE TABLE vasopressor_events AS
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.starttime,
            i.endtime,
            i.target_name,
            i.label,
            i.rate,
            i.rateuom
        FROM {inp} i
        JOIN cap_cohort c USING (stay_id)
        WHERE c.cap_proxy = 1
          AND i.target_class = 'vasopressor'
          AND i.starttime BETWEEN c.intime - INTERVAL '{pre} hours'
                              AND c.intime + INTERVAL '{post} hours'
        """)

        # Summarize the first qualifying support times.
        self.con.execute(f"""
        CREATE OR REPLACE TABLE support_summary AS
        WITH r AS (
            SELECT
                stay_id,
                min(CASE WHEN resp_support_broad=1 THEN charttime END) AS first_resp_broad,
                min(CASE WHEN resp_support_strict=1 THEN charttime END) AS first_resp_strict,
                max(invasive_device) AS any_invasive_device,
                max(niv_device) AS any_niv_device,
                max(hfnc_device) AS any_hfnc,
                max(hfnc_remap_like) AS any_hfnc_remap_like
            FROM respiratory_support_events
            GROUP BY stay_id
        ),
        v AS (
            SELECT stay_id, min(starttime) AS first_vasopressor
            FROM vasopressor_events
            GROUP BY stay_id
        )
        SELECT
            c.stay_id,
            r.first_resp_broad,
            r.first_resp_strict,
            v.first_vasopressor,
            coalesce(r.any_invasive_device,0) AS any_invasive_device,
            coalesce(r.any_niv_device,0) AS any_niv_device,
            coalesce(r.any_hfnc,0) AS any_hfnc,
            coalesce(r.any_hfnc_remap_like,0) AS any_hfnc_remap_like,
            CASE WHEN r.first_resp_broad IS NOT NULL THEN 1 ELSE 0 END AS respiratory_support_broad,
            CASE WHEN r.first_resp_strict IS NOT NULL THEN 1 ELSE 0 END AS respiratory_support_strict,
            CASE WHEN v.first_vasopressor IS NOT NULL THEN 1 ELSE 0 END AS vasopressor_support,
            least(r.first_resp_broad, v.first_vasopressor) AS severe_t0_broad,
            least(r.first_resp_strict, v.first_vasopressor) AS severe_t0_strict,
            CASE WHEN r.first_resp_broad IS NOT NULL
                       AND r.first_resp_broad <= least(r.first_resp_broad, v.first_vasopressor) + INTERVAL '{phenotype_h} hours'
                 THEN 1 ELSE 0 END AS initial_resp_broad,
            CASE WHEN r.first_resp_strict IS NOT NULL
                       AND r.first_resp_strict <= least(r.first_resp_strict, v.first_vasopressor) + INTERVAL '{phenotype_h} hours'
                 THEN 1 ELSE 0 END AS initial_resp_strict,
            CASE WHEN v.first_vasopressor IS NOT NULL
                       AND v.first_vasopressor <= least(r.first_resp_broad, v.first_vasopressor) + INTERVAL '{phenotype_h} hours'
                 THEN 1 ELSE 0 END AS initial_vasopressor_broad,
            CASE WHEN v.first_vasopressor IS NOT NULL
                       AND v.first_vasopressor <= least(r.first_resp_strict, v.first_vasopressor) + INTERVAL '{phenotype_h} hours'
                 THEN 1 ELSE 0 END AS initial_vasopressor_strict
        FROM cap_cohort c
        LEFT JOIN r USING (stay_id)
        LEFT JOIN v USING (stay_id)
        WHERE c.cap_proxy=1
        """)

    def build_steroid_exposure(self, input_cache: Path, rx_cache: Path) -> None:
        inp = source_sql(input_cache)
        rx = source_sql(rx_cache)
        window = int(self.cfg.EARLY_STEROID_WINDOW_HOURS)
        lookback = int(self.cfg.PRE_T0_STEROID_LOOKBACK_HOURS)

        # Inputevents = more administration-like ICU exposure.
        self.con.execute(f"""
        CREATE OR REPLACE TABLE steroid_admin_events AS
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.starttime AS event_time,
            i.target_name AS steroid_name,
            i.label AS raw_name,
            'inputevents' AS source
        FROM {inp} i
        JOIN support_summary s USING (stay_id)
        WHERE i.target_class='steroid'
          AND i.starttime BETWEEN s.severe_t0_broad - INTERVAL '{lookback} hours'
                              AND s.severe_t0_broad + INTERVAL '{window} hours'
        """)

        # Prescriptions are orders, not guaranteed administrations; use as sensitivity / feasibility source.
        self.con.execute(f"""
        CREATE OR REPLACE TABLE steroid_order_events AS
        SELECT
            r.subject_id,
            r.hadm_id,
            c.stay_id,
            r.starttime AS event_time,
            r.steroid_name,
            r.drug AS raw_name,
            'prescriptions' AS source
        FROM {rx} r
        JOIN cap_cohort c USING (hadm_id, subject_id)
        JOIN support_summary s USING (stay_id)
        WHERE c.cap_proxy=1
          AND r.starttime BETWEEN s.severe_t0_broad - INTERVAL '{lookback} hours'
                              AND s.severe_t0_broad + INTERVAL '{window} hours'
        """)

        self.con.execute("""
        CREATE OR REPLACE TABLE steroid_all_events AS
        SELECT * FROM steroid_admin_events
        UNION ALL
        SELECT * FROM steroid_order_events
        """)

        self.con.execute(f"""
        CREATE OR REPLACE TABLE steroid_summary AS
        SELECT
            s.stay_id,
            min(CASE WHEN a.source='inputevents' THEN a.event_time END) AS first_steroid_admin,
            min(CASE WHEN a.source='prescriptions' THEN a.event_time END) AS first_steroid_order,
            min(CASE WHEN a.steroid_name='hydrocortisone' THEN a.event_time END) AS first_hydrocortisone_any,
            max(CASE WHEN a.event_time >= s.severe_t0_broad
                      AND a.event_time <= s.severe_t0_broad + INTERVAL '{window} hours'
                     THEN 1 ELSE 0 END) AS early_steroid_any,
            max(CASE WHEN a.source='inputevents'
                      AND a.event_time >= s.severe_t0_broad
                      AND a.event_time <= s.severe_t0_broad + INTERVAL '{window} hours'
                     THEN 1 ELSE 0 END) AS early_steroid_admin,
            max(CASE WHEN a.steroid_name='hydrocortisone'
                      AND a.event_time >= s.severe_t0_broad
                      AND a.event_time <= s.severe_t0_broad + INTERVAL '{window} hours'
                     THEN 1 ELSE 0 END) AS early_hydrocortisone_any,
            max(CASE WHEN a.source='inputevents'
                      AND a.steroid_name='hydrocortisone'
                      AND a.event_time >= s.severe_t0_broad
                      AND a.event_time <= s.severe_t0_broad + INTERVAL '{window} hours'
                     THEN 1 ELSE 0 END) AS early_hydrocortisone_admin,
            max(CASE WHEN a.event_time < s.severe_t0_broad
                      AND a.event_time >= s.severe_t0_broad - INTERVAL '{lookback} hours'
                     THEN 1 ELSE 0 END) AS steroid_pre_t0_24h
        FROM support_summary s
        LEFT JOIN steroid_all_events a USING (stay_id)
        GROUP BY s.stay_id
        """)

    def build_final_patient_table(self) -> None:
        self.con.execute("""
        CREATE OR REPLACE TABLE feasibility_cohort AS
        SELECT
            c.*,
            s.first_resp_broad,
            s.first_resp_strict,
            s.first_vasopressor,
            s.severe_t0_broad,
            s.severe_t0_strict,
            coalesce(s.respiratory_support_broad,0) AS respiratory_support_broad,
            coalesce(s.respiratory_support_strict,0) AS respiratory_support_strict,
            coalesce(s.vasopressor_support,0) AS vasopressor_support,
            coalesce(s.initial_resp_broad,0) AS initial_resp_broad,
            coalesce(s.initial_resp_strict,0) AS initial_resp_strict,
            coalesce(s.initial_vasopressor_broad,0) AS initial_vasopressor_broad,
            coalesce(s.initial_vasopressor_strict,0) AS initial_vasopressor_strict,
            coalesce(s.any_invasive_device,0) AS any_invasive_device,
            coalesce(s.any_niv_device,0) AS any_niv_device,
            coalesce(s.any_hfnc,0) AS any_hfnc,
            coalesce(s.any_hfnc_remap_like,0) AS any_hfnc_remap_like,
            CASE WHEN s.severe_t0_broad IS NOT NULL THEN 1 ELSE 0 END AS severe_cap_broad_proxy,
            CASE WHEN s.severe_t0_strict IS NOT NULL THEN 1 ELSE 0 END AS severe_cap_strict_proxy,
            CASE
                WHEN s.severe_t0_broad IS NULL THEN 'no_early_organ_support'
                WHEN s.initial_resp_broad=1 AND s.initial_vasopressor_broad=0 THEN 'respiratory_only'
                WHEN s.initial_resp_broad=0 AND s.initial_vasopressor_broad=1 THEN 'shock_only'
                WHEN s.initial_resp_broad=1 AND s.initial_vasopressor_broad=1 THEN 'combined'
                ELSE 'other'
            END AS phenotype_broad,
            CASE
                WHEN s.severe_t0_strict IS NULL THEN 'no_early_organ_support'
                WHEN s.initial_resp_strict=1 AND s.initial_vasopressor_strict=0 THEN 'respiratory_only'
                WHEN s.initial_resp_strict=0 AND s.initial_vasopressor_strict=1 THEN 'shock_only'
                WHEN s.initial_resp_strict=1 AND s.initial_vasopressor_strict=1 THEN 'combined'
                ELSE 'other'
            END AS phenotype_strict,
            coalesce(st.early_steroid_any,0) AS early_steroid_any,
            coalesce(st.early_steroid_admin,0) AS early_steroid_admin,
            coalesce(st.early_hydrocortisone_any,0) AS early_hydrocortisone_any,
            coalesce(st.early_hydrocortisone_admin,0) AS early_hydrocortisone_admin,
            coalesce(st.steroid_pre_t0_24h,0) AS steroid_pre_t0_24h,
            st.first_steroid_admin,
            st.first_steroid_order,
            st.first_hydrocortisone_any,
            CASE WHEN c.hospital_expire_flag=1 THEN 1 ELSE 0 END AS hospital_death,
            CASE WHEN c.dod IS NOT NULL AND s.severe_t0_broad IS NOT NULL
                       AND cast(c.dod as timestamp) >= s.severe_t0_broad
                       AND cast(c.dod as timestamp) < s.severe_t0_broad + INTERVAL '28 days'
                 THEN 1 ELSE 0 END AS death_28d_proxy
        FROM cap_cohort c
        LEFT JOIN support_summary s USING (stay_id)
        LEFT JOIN steroid_summary st USING (stay_id)
        WHERE c.cap_proxy=1
        """)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _count(self, where: str = "TRUE") -> int:
        return int(self.con.execute(f"SELECT count(*) FROM feasibility_cohort WHERE {where}").fetchone()[0])

    def make_flow_table(self) -> pd.DataFrame:
        # Counts from base stages preserve the full denominator.
        queries = [
            ("All first ICU stays per hospital admission", "SELECT count(*) FROM base_icu"),
            ("Adults (age >=18)", "SELECT count(*) FROM cap_cohort WHERE adult=1"),
            (f"ICU within {self.cfg.ICU_WITHIN_HOSPITAL_HOURS} h of hospital admission", f"SELECT count(*) FROM cap_cohort WHERE adult=1 AND early_icu=1"),
            ("Pneumonia ICD present", "SELECT count(*) FROM cap_cohort WHERE cap_broad=1"),
            ("Exclude influenza/COVID/aspiration/opportunistic pneumonia (CAP proxy)", "SELECT count(*) FROM cap_cohort WHERE cap_proxy=1"),
            ("Severe CAP broad proxy: respiratory support and/or vasopressor", "SELECT count(*) FROM feasibility_cohort WHERE severe_cap_broad_proxy=1"),
            ("Severe CAP strict proxy: invasive/NIV/qualified HFNC and/or vasopressor", "SELECT count(*) FROM feasibility_cohort WHERE severe_cap_strict_proxy=1"),
            ("Early systemic steroid <=24 h after broad severe t0 (any source)", "SELECT count(*) FROM feasibility_cohort WHERE severe_cap_broad_proxy=1 AND early_steroid_any=1"),
            ("Early steroid <=24 h (inputevents/admin-like only)", "SELECT count(*) FROM feasibility_cohort WHERE severe_cap_broad_proxy=1 AND early_steroid_admin=1"),
            ("Early hydrocortisone <=24 h (any source)", "SELECT count(*) FROM feasibility_cohort WHERE severe_cap_broad_proxy=1 AND early_hydrocortisone_any=1"),
            ("Early hydrocortisone <=24 h (inputevents/admin-like only)", "SELECT count(*) FROM feasibility_cohort WHERE severe_cap_broad_proxy=1 AND early_hydrocortisone_admin=1"),
        ]
        rows = []
        previous = None
        for label, sql in queries:
            n = int(self.con.execute(sql).fetchone()[0])
            rows.append({
                "step": label,
                "n": n,
                "retained_from_previous_pct": None if previous in (None, 0) else round(pct(n, previous), 1),
            })
            previous = n
        return pd.DataFrame(rows)

    def make_phenotype_table(self) -> pd.DataFrame:
        df = self.con.execute("""
            SELECT
                phenotype_broad AS phenotype,
                count(*) AS n,
                sum(early_steroid_any) AS early_steroid_any_n,
                sum(early_steroid_admin) AS early_steroid_admin_n,
                sum(early_hydrocortisone_any) AS early_hydrocortisone_any_n,
                sum(early_hydrocortisone_admin) AS early_hydrocortisone_admin_n,
                sum(hospital_death) AS hospital_death_n,
                round(100.0 * avg(hospital_death), 1) AS hospital_mortality_pct
            FROM feasibility_cohort
            WHERE severe_cap_broad_proxy=1
            GROUP BY phenotype_broad
            ORDER BY n DESC
        """).fetchdf()
        for col in ["early_steroid_any_n", "early_steroid_admin_n", "early_hydrocortisone_any_n", "early_hydrocortisone_admin_n"]:
            pct_col = col.replace("_n", "_pct")
            df[pct_col] = (100 * df[col] / df["n"]).round(1)
        return df

    def make_steroid_dictionary(self) -> pd.DataFrame:
        return self.con.execute("""
            SELECT source, steroid_name, raw_name, count(*) AS event_rows
            FROM steroid_all_events
            GROUP BY source, steroid_name, raw_name
            ORDER BY source, steroid_name, event_rows DESC
        """).fetchdf()

    def make_quality_table(self) -> pd.DataFrame:
        return self.con.execute("""
            SELECT metric, n FROM (
                SELECT 'CAP proxy total' metric, count(*) n FROM feasibility_cohort
                UNION ALL
                SELECT 'Pneumonia in top 3 diagnosis positions', sum(pneumonia_top3_dx) FROM feasibility_cohort
                UNION ALL
                SELECT 'Any broad respiratory support', sum(respiratory_support_broad) FROM feasibility_cohort
                UNION ALL
                SELECT 'Any strict respiratory support', sum(respiratory_support_strict) FROM feasibility_cohort
                UNION ALL
                SELECT 'Any vasopressor support', sum(vasopressor_support) FROM feasibility_cohort
                UNION ALL
                SELECT 'Any HFNC documented', sum(any_hfnc) FROM feasibility_cohort
                UNION ALL
                SELECT 'HFNC meeting flow+FiO2 proxy', sum(any_hfnc_remap_like) FROM feasibility_cohort
                UNION ALL
                SELECT 'Steroid in 24 h before t0 (possible prevalent/other indication)', sum(steroid_pre_t0_24h) FROM feasibility_cohort
            )
        """).fetchdf()

    def make_go_nogo(self) -> pd.DataFrame:
        values = {
            "cap_proxy": self._count(),
            "severe_cap_proxy": self._count("severe_cap_broad_proxy=1"),
            "early_systemic_steroid": self._count("severe_cap_broad_proxy=1 AND early_steroid_any=1"),
            "early_hydrocortisone": self._count("severe_cap_broad_proxy=1 AND early_hydrocortisone_any=1"),
            "hospital_deaths_severe_cap": self._count("severe_cap_broad_proxy=1 AND hospital_death=1"),
        }
        rows = []
        for key, threshold in self.cfg.GO_NOGO_THRESHOLDS.items():
            n = values[key]
            if n >= threshold:
                status = "GO"
            elif n >= threshold * 0.6:
                status = "CAUTION"
            else:
                status = "NO-GO"
            rows.append({"metric": key, "observed_n": n, "heuristic_threshold": threshold, "status": status})
        overall = "GO" if all(r["status"] == "GO" for r in rows) else ("CAUTION" if any(r["status"] == "CAUTION" for r in rows) else "NO-GO")
        rows.append({"metric": "OVERALL", "observed_n": None, "heuristic_threshold": None, "status": overall})
        return pd.DataFrame(rows)

    def export_outputs(self, inventory: pd.DataFrame) -> Dict[str, Path]:
        flow = self.make_flow_table()
        phen = self.make_phenotype_table()
        steroid_dict = self.make_steroid_dictionary()
        quality = self.make_quality_table()
        gonogo = self.make_go_nogo()
        cohort = self.con.execute("SELECT * FROM feasibility_cohort ORDER BY severe_t0_broad NULLS LAST").fetchdf()

        paths = {
            "flow_csv": self.paths.output_dir / "01_cohort_flow.csv",
            "phenotype_csv": self.paths.output_dir / "02_phenotype_steroid_counts.csv",
            "quality_csv": self.paths.output_dir / "03_data_quality.csv",
            "steroid_dict_csv": self.paths.output_dir / "04_steroid_dictionary.csv",
            "gonogo_csv": self.paths.output_dir / "05_GO_NOGO.csv",
            "cohort_parquet": self.paths.output_dir / "feasibility_cohort.parquet",
            "excel": self.paths.output_dir / "feasibility_report.xlsx",
        }
        flow.to_csv(paths["flow_csv"], index=False, encoding="utf-8-sig")
        phen.to_csv(paths["phenotype_csv"], index=False, encoding="utf-8-sig")
        quality.to_csv(paths["quality_csv"], index=False, encoding="utf-8-sig")
        steroid_dict.to_csv(paths["steroid_dict_csv"], index=False, encoding="utf-8-sig")
        gonogo.to_csv(paths["gonogo_csv"], index=False, encoding="utf-8-sig")
        cohort.to_parquet(paths["cohort_parquet"], index=False)

        preview_n = min(len(cohort), int(self.cfg.EXCEL_COHORT_PREVIEW_ROWS))
        with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
            inventory.to_excel(writer, sheet_name="data_inventory", index=False)
            flow.to_excel(writer, sheet_name="cohort_flow", index=False)
            phen.to_excel(writer, sheet_name="phenotypes", index=False)
            quality.to_excel(writer, sheet_name="data_quality", index=False)
            steroid_dict.to_excel(writer, sheet_name="steroid_dictionary", index=False)
            gonogo.to_excel(writer, sheet_name="GO_NOGO", index=False)
            cohort.head(preview_n).to_excel(writer, sheet_name="cohort_preview", index=False)

        return paths

    def run(self) -> Dict[str, Path]:
        self.discover_tables()
        inventory = self.inventory_dataframe()
        inventory.to_csv(self.paths.output_dir / "00_data_inventory.csv", index=False, encoding="utf-8-sig")

        resp_cache = self.build_resp_cache()
        input_cache = self.build_input_cache()
        rx_cache = self.build_rx_steroid_cache()

        self.build_base_cohort()
        self.build_support_events(resp_cache, input_cache)
        self.build_steroid_exposure(input_cache, rx_cache)
        self.build_final_patient_table()
        return self.export_outputs(inventory)

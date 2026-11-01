from __future__ import annotations

"""
Stage 8B — High-specificity clinical replication of the Stage 7 timing signal

This stage does NOT replace Stage 7.
It asks whether the 6h -> 12h -> 24h timing gradient persists when the cohort
is restricted to a clinically specific, CAPE-COD-like population.

Primary high-specificity cohort:
    - Stage 5.5 primary cohort (major immunocompromise already excluded)
    - pneumonia among top 3 diagnosis positions
    - exclude influenza, COVID-19, aspiration pneumonia/pneumonitis,
      Pneumocystis, tuberculosis, adrenal replacement indication
    - no vasopressor-dependent shock at respiratory t0

Expected from Stage 8A:
    n ~707
    any systemic steroid initiated:
        <=6h  ~47
        <=12h ~77
        <=24h ~100

Primary Stage 8B analyses:
    CAPE-COD-like cohort:
       any systemic steroid <=6h  vs no initiation <=6h
       any systemic steroid <=12h vs no initiation <=12h
       any systemic steroid <=24h vs no initiation <=24h

Sensitivity:
    strict clinical CAP including shock, 24h trial only

Exploratory:
    CAPE-COD-like IV systemic steroid <=24h
    (expected sparse; do not overinterpret)

This script reuses the Stage 7 hourly panel and Stage 7 modeling functions,
so it does NOT rescan chartevents/labevents.

Outputs to upload:
    outputs/14_STAGE8B_CAPECOD_DYNAMIC.csv
    outputs/14B_STAGE8B_STRICTCAP_SENSITIVITY.csv
    outputs/14C_STAGE8B_WEIGHT_DIAGNOSTICS.csv
    outputs/14D_STAGE8B_STAGE7_COMPARISON.csv
    outputs/14E_STAGE8B_IV_EXPLORATORY.csv
    outputs/14F_STAGE8B_MODEL_DIAGNOSTICS.csv
"""

from pathlib import Path
import importlib.util
import traceback

import duckdb
import numpy as np
import pandas as pd

import config
from src.utils import connect_duckdb, find_table, try_find_table, source_sql, sql_quote_path

PROJECT_ROOT = Path(config.PROJECT_ROOT)
DATA_ROOT = Path(config.DATA_ROOT)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

COHORT = OUTPUT_DIR / "stage5_5_analysis_cohort.parquet"
PANEL = OUTPUT_DIR / "stage7_hourly_panel.parquet"
STAGE7_SCRIPT = PROJECT_ROOT / "12_run_time_updated_ccw.py"
STAGE7_PRIMARY = OUTPUT_DIR / "12_STAGE7_DYNAMIC_PRIMARY.csv"

OUT_CAPE = OUTPUT_DIR / "14_STAGE8B_CAPECOD_DYNAMIC.csv"
OUT_STRICT = OUTPUT_DIR / "14B_STAGE8B_STRICTCAP_SENSITIVITY.csv"
OUT_WEIGHT = OUTPUT_DIR / "14C_STAGE8B_WEIGHT_DIAGNOSTICS.csv"
OUT_COMPARE = OUTPUT_DIR / "14D_STAGE8B_STAGE7_COMPARISON.csv"
OUT_IV = OUTPUT_DIR / "14E_STAGE8B_IV_EXPLORATORY.csv"
OUT_MODEL = OUTPUT_DIR / "14F_STAGE8B_MODEL_DIAGNOSTICS.csv"

# First publication-feasibility pass.
BOOT_REPS = {6: 100, 12: 100, 24: 150}
STRICT24_BOOT = 100
IV24_BOOT = 60
SEED = 20261027


def qpath(p: Path) -> str:
    return sql_quote_path(p)


def load_stage7_module():
    if not STAGE7_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing {STAGE7_SCRIPT}\n"
            "Put the Stage 7 script in the project root first."
        )
    spec = importlib.util.spec_from_file_location("stage7_ccw", STAGE7_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_clinical_flags():
    if not COHORT.exists():
        raise FileNotFoundError(f"Missing {COHORT}; run Stage 5.5 first.")

    dx_path = find_table(DATA_ROOT, "hosp", "diagnoses_icd")
    ddx_path = try_find_table(DATA_ROOT, "hosp", "d_icd_diagnoses")

    con = connect_duckdb(PROJECT_ROOT, getattr(config, "DUCKDB_THREADS", 4))
    try:
        cohort_sql = f"read_parquet('{qpath(COHORT)}')"
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE c0 AS
            SELECT *
            FROM {cohort_sql}
            WHERE major_immunocompromise_proxy=0
        """)
        dx = source_sql(dx_path)

        if ddx_path is not None:
            ddx = source_sql(ddx_path)
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE flags AS
                SELECT
                    c.stay_id,
                    c.hadm_id,
                    c.pneumonia_top3_dx,
                    c.shock_active_at_t0,

                    max(CASE
                        WHEN lower(coalesce(x.long_title,'')) LIKE '%influenza%'
                          OR (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^J0(9|10|11)'))
                        THEN 1 ELSE 0 END) AS influenza_flag,

                    max(CASE
                        WHEN replace(upper(d.icd_code),'.','')='U071'
                          OR lower(coalesce(x.long_title,'')) LIKE '%covid-19%'
                          OR lower(coalesce(x.long_title,'')) LIKE '%covid 19%'
                        THEN 1 ELSE 0 END) AS covid_flag,

                    max(CASE
                        WHEN lower(coalesce(x.long_title,'')) LIKE '%aspiration pneumonia%'
                          OR lower(coalesce(x.long_title,'')) LIKE '%pneumonitis due to inhalation%'
                          OR lower(coalesce(x.long_title,'')) LIKE '%aspiration pneumonitis%'
                          OR (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^J69'))
                          OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^507'))
                        THEN 1 ELSE 0 END) AS aspiration_flag,

                    max(CASE
                        WHEN replace(upper(d.icd_code),'.','') IN ('B59','1363')
                          OR lower(coalesce(x.long_title,'')) LIKE '%pneumocyst%'
                        THEN 1 ELSE 0 END) AS pneumocystis_flag,

                    max(CASE
                        WHEN lower(coalesce(x.long_title,'')) LIKE '%tuberculosis%'
                          OR (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^A1[5-9]'))
                          OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^01[0-8]'))
                        THEN 1 ELSE 0 END) AS tuberculosis_flag,

                    max(CASE
                        WHEN lower(coalesce(x.long_title,'')) LIKE '%adrenal insufficiency%'
                          OR lower(coalesce(x.long_title,'')) LIKE '%addison%'
                          OR lower(coalesce(x.long_title,'')) LIKE '%adrenocortical insufficiency%'
                          OR (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^E27'))
                        THEN 1 ELSE 0 END) AS adrenal_flag

                FROM c0 c
                LEFT JOIN {dx} d USING (hadm_id)
                LEFT JOIN {ddx} x
                  ON d.icd_code=x.icd_code
                 AND d.icd_version=x.icd_version
                GROUP BY
                    c.stay_id, c.hadm_id,
                    c.pneumonia_top3_dx, c.shock_active_at_t0
            """)
        else:
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE flags AS
                SELECT
                    c.stay_id,
                    c.hadm_id,
                    c.pneumonia_top3_dx,
                    c.shock_active_at_t0,

                    max(CASE WHEN
                        d.icd_version=10
                        AND regexp_matches(replace(upper(d.icd_code),'.',''), '^J0(9|10|11)')
                        THEN 1 ELSE 0 END) AS influenza_flag,

                    max(CASE WHEN replace(upper(d.icd_code),'.','')='U071'
                        THEN 1 ELSE 0 END) AS covid_flag,

                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^J69'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^507'))
                        THEN 1 ELSE 0 END) AS aspiration_flag,

                    max(CASE WHEN replace(upper(d.icd_code),'.','') IN ('B59','1363')
                        THEN 1 ELSE 0 END) AS pneumocystis_flag,

                    max(CASE WHEN
                        (d.icd_version=10 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^A1[5-9]'))
                        OR (d.icd_version=9 AND regexp_matches(replace(upper(d.icd_code),'.',''), '^01[0-8]'))
                        THEN 1 ELSE 0 END) AS tuberculosis_flag,

                    max(CASE WHEN
                        d.icd_version=10
                        AND regexp_matches(replace(upper(d.icd_code),'.',''), '^E27')
                        THEN 1 ELSE 0 END) AS adrenal_flag

                FROM c0 c
                LEFT JOIN {dx} d USING (hadm_id)
                GROUP BY
                    c.stay_id, c.hadm_id,
                    c.pneumonia_top3_dx, c.shock_active_at_t0
            """)

        out = con.execute("""
            SELECT
                stay_id,
                CASE WHEN
                    pneumonia_top3_dx=1
                    AND coalesce(influenza_flag,0)=0
                    AND coalesce(covid_flag,0)=0
                    AND coalesce(aspiration_flag,0)=0
                    AND coalesce(pneumocystis_flag,0)=0
                    AND coalesce(tuberculosis_flag,0)=0
                    AND coalesce(adrenal_flag,0)=0
                THEN 1 ELSE 0 END AS strict_cap,

                CASE WHEN
                    pneumonia_top3_dx=1
                    AND coalesce(influenza_flag,0)=0
                    AND coalesce(covid_flag,0)=0
                    AND coalesce(aspiration_flag,0)=0
                    AND coalesce(pneumocystis_flag,0)=0
                    AND coalesce(tuberculosis_flag,0)=0
                    AND coalesce(adrenal_flag,0)=0
                    AND shock_active_at_t0=0
                THEN 1 ELSE 0 END AS cape_cod_like
            FROM flags
        """).fetchdf()
        return out
    finally:
        con.close()


def subset_panel(panel, ids):
    ids = set(ids)
    x = panel[panel["stay_id"].isin(ids)].copy()
    return x.reset_index(drop=True)


def ci(mod, boot, col):
    return mod.percentile_ci(boot, col)


def weight_diag(mod, fit, panel, grace, label):
    fw = mod.final_weights(fit, panel, grace)
    rows = []
    for col in ["w_no","w_tx"]:
        w = fw[col].to_numpy(float)
        rows.append({
            "analysis": label,
            "grace_hours": grace,
            "weight": col,
            "n": len(w),
            "mean": np.mean(w),
            "sd": np.std(w, ddof=1),
            "p01": np.quantile(w,.01),
            "p50": np.quantile(w,.50),
            "p99": np.quantile(w,.99),
            "max": np.max(w),
            "ess": mod.ess(w),
            "raw_weight_p99": fit["weight_raw_p99"],
            "raw_weight_max": fit["weight_raw_max"],
            "truncation_low": fit["trunc_lo"],
            "truncation_high": fit["trunc_hi"],
        })
    return rows


def run_cape_dynamic(mod, panel):
    rows = []
    weights = []
    models = []

    for grace in [6,12,24]:
        print(f"\n=== CAPE-COD-like dynamic CCW: {grace}h ===")
        fit = mod.fit_trial(panel, "any_init_h", grace, details=True)
        reps = BOOT_REPS[grace]
        boot = mod.bootstrap_trial(
            panel, "any_init_h", grace, reps, SEED + 1000 + grace
        )
        c0 = ci(mod,boot,0)
        c1 = ci(mod,boot,1)
        crd = ci(mod,boot,2)
        crr = ci(mod,boot,3)

        first = pd.to_numeric(
            panel.groupby("stay_id")["any_init_h"].first(),
            errors="coerce"
        )
        initiated_n = int(first.le(grace).sum())

        rows.append({
            "population":"CAPE_COD_like_no_shock",
            "grace_hours":grace,
            "n":panel["stay_id"].nunique(),
            "initiated_n":initiated_n,
            "risk_no_initiation":fit["risk_no_initiation"],
            "risk_no_ci_low":c0[0],
            "risk_no_ci_high":c0[1],
            "risk_initiation":fit["risk_initiation"],
            "risk_init_ci_low":c1[0],
            "risk_init_ci_high":c1[1],
            "risk_difference":fit["risk_difference"],
            "rd_ci_low":crd[0],
            "rd_ci_high":crd[1],
            "risk_ratio":fit["risk_ratio"],
            "rr_ci_low":crr[0],
            "rr_ci_high":crr[1],
            "bootstrap_success_n":len(boot),
            "interpretation":"HIGH-SPECIFICITY replication; primary clinical sensitivity analysis"
        })
        weights.extend(weight_diag(mod,fit,panel,grace,f"CAPE_COD_like_{grace}h"))
        models.append({
            "analysis":f"CAPE_COD_like_any_steroid_{grace}h",
            "n":panel["stay_id"].nunique(),
            "grace_hours":grace,
            "person_hours":fit["person_hours"],
            "initiation_events":fit["initiation_events"],
            "pred_hazard_mean":fit["pred_hazard_mean"],
            "pred_hazard_min":fit["pred_hazard_min"],
            "pred_hazard_max":fit["pred_hazard_max"],
            "raw_weight_p99":fit["weight_raw_p99"],
            "raw_weight_max":fit["weight_raw_max"],
        })

    return pd.DataFrame(rows), weights, models


def run_strict24(mod, panel):
    print("\n=== Strict clinical CAP including shock: 24h sensitivity ===")
    fit = mod.fit_trial(panel,"any_init_h",24,details=False)
    boot = mod.bootstrap_trial(
        panel,"any_init_h",24,STRICT24_BOOT,SEED+2024
    )
    crd = ci(mod,boot,2)
    crr = ci(mod,boot,3)
    first = pd.to_numeric(
        panel.groupby("stay_id")["any_init_h"].first(),
        errors="coerce"
    )
    return pd.DataFrame([{
        "population":"strict_clinical_CAP_including_shock",
        "grace_hours":24,
        "n":panel["stay_id"].nunique(),
        "initiated_n":int(first.le(24).sum()),
        "risk_no_initiation":fit["risk_no_initiation"],
        "risk_initiation":fit["risk_initiation"],
        "risk_difference":fit["risk_difference"],
        "rd_ci_low":crd[0],
        "rd_ci_high":crd[1],
        "risk_ratio":fit["risk_ratio"],
        "rr_ci_low":crr[0],
        "rr_ci_high":crr[1],
        "bootstrap_success_n":len(boot),
        "interpretation":"Sensitivity cohort; includes shock-at-t0"
    }])


def run_iv24(mod,panel):
    print("\n=== CAPE-COD-like IV steroid <=24h exploratory analysis ===")
    try:
        fit=mod.fit_trial(panel,"iv_init_h",24,details=False)
        boot=mod.bootstrap_trial(
            panel,"iv_init_h",24,IV24_BOOT,SEED+3024
        )
        crd=ci(mod,boot,2)
        crr=ci(mod,boot,3)
        first=pd.to_numeric(
            panel.groupby("stay_id")["iv_init_h"].first(),
            errors="coerce"
        )
        return pd.DataFrame([{
            "population":"CAPE_COD_like_no_shock",
            "exposure":"IV_systemic_steroid",
            "grace_hours":24,
            "n":panel["stay_id"].nunique(),
            "initiated_n":int(first.le(24).sum()),
            "risk_no_initiation":fit["risk_no_initiation"],
            "risk_initiation":fit["risk_initiation"],
            "risk_difference":fit["risk_difference"],
            "rd_ci_low":crd[0],
            "rd_ci_high":crd[1],
            "risk_ratio":fit["risk_ratio"],
            "rr_ci_low":crr[0],
            "rr_ci_high":crr[1],
            "bootstrap_success_n":len(boot),
            "raw_weight_p99":fit["weight_raw_p99"],
            "raw_weight_max":fit["weight_raw_max"],
            "interpretation":"EXPLORATORY ONLY; expected sparse positivity"
        }])
    except Exception as e:
        return pd.DataFrame([{
            "population":"CAPE_COD_like_no_shock",
            "exposure":"IV_systemic_steroid",
            "grace_hours":24,
            "n":panel["stay_id"].nunique(),
            "initiated_n":np.nan,
            "risk_no_initiation":np.nan,
            "risk_initiation":np.nan,
            "risk_difference":np.nan,
            "rd_ci_low":np.nan,
            "rd_ci_high":np.nan,
            "risk_ratio":np.nan,
            "rr_ci_low":np.nan,
            "rr_ci_high":np.nan,
            "bootstrap_success_n":0,
            "raw_weight_p99":np.nan,
            "raw_weight_max":np.nan,
            "interpretation":f"FAILED/TOO SPARSE: {repr(e)}"
        }])


def compare_to_stage7(cape):
    rows=[]
    if not STAGE7_PRIMARY.exists():
        return pd.DataFrame([{
            "grace_hours":np.nan,
            "stage7_broad_rd":np.nan,
            "stage8b_cape_cod_rd":np.nan,
            "stage7_broad_rr":np.nan,
            "stage8b_cape_cod_rr":np.nan,
            "note":"Stage7 result file not found"
        }])

    s7=pd.read_csv(STAGE7_PRIMARY)
    for grace in [6,12,24]:
        a=s7[s7.grace_hours==grace].iloc[0]
        b=cape[cape.grace_hours==grace].iloc[0]
        rows.append({
            "grace_hours":grace,
            "stage7_broad_n":int(a["n"]),
            "stage8b_cape_cod_n":int(b["n"]),
            "stage7_broad_initiated_n":int(a["actual_initiated_by_grace_n"]),
            "stage8b_cape_cod_initiated_n":int(b["initiated_n"]),
            "stage7_broad_rd":a["risk_difference"],
            "stage8b_cape_cod_rd":b["risk_difference"],
            "rd_change_specific_minus_broad":b["risk_difference"]-a["risk_difference"],
            "stage7_broad_rr":a["risk_ratio"],
            "stage8b_cape_cod_rr":b["risk_ratio"],
            "rr_change_specific_minus_broad":b["risk_ratio"]-a["risk_ratio"],
            "interpretation":"If the timing gradient persists after clinical restriction, contamination by non-CAP steroid indications is less likely to fully explain Stage7."
        })
    return pd.DataFrame(rows)


def main():
    print("="*96)
    print("Stage 8B — High-specificity CAPE-COD-like dynamic CCW replication")
    print("="*96)

    try:
        if not PANEL.exists():
            raise FileNotFoundError(
                f"Missing {PANEL}; run Stage 7 first."
            )

        mod=load_stage7_module()
        flags=build_clinical_flags()
        panel=pd.read_parquet(PANEL)

        cape_ids=flags.loc[flags.cape_cod_like==1,"stay_id"]
        strict_ids=flags.loc[flags.strict_cap==1,"stay_id"]

        cape_panel=subset_panel(panel,cape_ids)
        strict_panel=subset_panel(panel,strict_ids)

        print(f"[COHORT] CAPE-COD-like n={cape_panel.stay_id.nunique()}")
        print(f"[COHORT] strict clinical CAP n={strict_panel.stay_id.nunique()}")

        cape,weight_rows,model_rows=run_cape_dynamic(mod,cape_panel)
        cape.to_csv(OUT_CAPE,index=False,encoding="utf-8-sig")
        pd.DataFrame(weight_rows).to_csv(
            OUT_WEIGHT,index=False,encoding="utf-8-sig"
        )

        strict=run_strict24(mod,strict_panel)
        strict.to_csv(OUT_STRICT,index=False,encoding="utf-8-sig")

        iv=run_iv24(mod,cape_panel)
        iv.to_csv(OUT_IV,index=False,encoding="utf-8-sig")

        compare=compare_to_stage7(cape)
        compare.to_csv(OUT_COMPARE,index=False,encoding="utf-8-sig")

        pd.DataFrame(model_rows).to_csv(
            OUT_MODEL,index=False,encoding="utf-8-sig"
        )

        print("\n=== CAPE-COD-like results ===")
        print(cape.to_string(index=False))
        print("\n=== Broad Stage7 vs high-specificity Stage8B ===")
        print(compare.to_string(index=False))
        print("\nUpload these files:")
        for p in [
            OUT_CAPE,OUT_STRICT,OUT_WEIGHT,OUT_COMPARE,OUT_IV,OUT_MODEL
        ]:
            print(" ",p)

        print("\nDecision rule:")
        print("- If 6/12/24 timing gradient persists in CAPE-COD-like cohort,")
        print("  the timing signal becomes substantially more clinically credible.")
        print("- If it disappears, Stage7 likely reflects exposure/indication heterogeneity.")
        print("- IV-only remains exploratory because only ~65 initiators are expected.")

    except Exception:
        print("\n[FAILED] Copy the entire traceback to ChatGPT:\n")
        traceback.print_exc()
        raise


if __name__=="__main__":
    main()

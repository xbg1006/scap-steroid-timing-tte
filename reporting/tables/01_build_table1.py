from __future__ import annotations

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("SCAP_PROJECT_ROOT", Path.cwd()))
OUTDIR = PROJECT_ROOT / "publication_tables_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_STAGE55_ROWS = 2153
EXPECTED_FINAL_ROWS = 2014

def find_file(name):
    hits = sorted(PROJECT_ROOT.rglob(name), key=lambda p:(len(p.parts),str(p).lower()))
    if not hits:
        raise FileNotFoundError(f"Cannot find {name} under {PROJECT_ROOT}")
    print(f"[READ] {name} <- {hits[0]}")
    return hits[0]

def load_parquet(name):
    try:
        return pd.read_parquet(find_file(name))
    except ImportError as exc:
        raise RuntimeError(
            "Install parquet support first:\n"
            "    pip install pyarrow\n"
            f"Original error: {exc}"
        )

def num(s):
    return pd.to_numeric(s, errors="coerce")

def median_iqr(s, digits=1, show_n=True):
    x = num(s).dropna()
    if len(x)==0:
        return "NA", 0
    q1, med, q3 = x.quantile([.25,.5,.75])
    f = f"{{:.{digits}f}}"
    txt = f"{f.format(med)} [{f.format(q1)}–{f.format(q3)}]"
    if show_n and len(x) < EXPECTED_FINAL_ROWS:
        txt += f" (n={len(x):,})"
    return txt, len(x)

def binary(s, show_n=True):
    x = num(s)
    valid = x.notna()
    den = int(valid.sum())
    if den==0:
        return "NA",0
    n = int((x[valid]==1).sum())
    txt = f"{n:,} ({100*n/den:.1f})"
    if show_n and den < EXPECTED_FINAL_ROWS:
        txt += f" (n={den:,})"
    return txt,den

def category(s, value, show_n=True):
    x=s.astype("string")
    valid=x.notna() & (x!="<NA>")
    den=int(valid.sum())
    n=int((x[valid]==value).sum())
    txt=f"{n:,} ({100*n/den:.1f})" if den else "NA"
    if show_n and den < EXPECTED_FINAL_ROWS:
        txt += f" (n={den:,})"
    return txt,den

# ---------------------------------------------------------------------
# Reconstruct exact frozen cohort
# ---------------------------------------------------------------------
s55 = load_parquet("stage5_5_analysis_cohort.parquet")
p7 = load_parquet("stage7_hourly_panel.parquet")

if len(s55) != EXPECTED_STAGE55_ROWS:
    raise RuntimeError(f"stage5_5 rows={len(s55)}, expected 2153")

for c in ["stay_id","eligible_primary","major_immunocompromise_proxy"]:
    if c not in s55.columns:
        raise RuntimeError(f"Missing required column: {c}")

mask = (
    num(s55["eligible_primary"]).fillna(0).eq(1)
    & num(s55["major_immunocompromise_proxy"]).fillna(0).eq(0)
)
frozen = s55.loc[mask].copy()

if len(frozen) != EXPECTED_FINAL_ROWS:
    raise RuntimeError(
        f"Final frozen cohort={len(frozen)}, expected 2014. STOP."
    )
if frozen["stay_id"].duplicated().any():
    raise RuntimeError("Duplicated stay_id in frozen cohort.")

hour0=p7.loc[num(p7["hour"]).eq(0)].copy()
hour0=hour0[hour0["stay_id"].isin(set(frozen["stay_id"]))].copy()
if hour0["stay_id"].nunique()!=EXPECTED_FINAL_ROWS:
    raise RuntimeError(
        f"Stage7 hour0 matched={hour0['stay_id'].nunique()}, expected 2014."
    )
if hour0["stay_id"].duplicated().any():
    raise RuntimeError("Duplicated stay_id at Stage7 hour0.")

dyn = [
    "stay_id",
    "dyn_rr_max6","dyn_map_min6","dyn_sf_ratio",
    "dyn_lactate_max12","dyn_ph_min12",
    "dyn_creatinine_last24","dyn_wbc_last24",
]
dyn=[c for c in dyn if c in hour0.columns]

a=frozen.merge(hour0[dyn], on="stay_id", how="left", validate="one_to_one")
N=len(a)

# ---------------------------------------------------------------------
# Main manuscript Table 1
# ---------------------------------------------------------------------
rows=[]

def row(section,label,value):
    rows.append({
        "Section":section,
        "Characteristic":label,
        f"Overall (N={N:,})":value,
    })

v,_=median_iqr(a["age"],1,False); row("Demographics","Age, years",v)

gender=a["gender"].astype("string").str.upper()
valid=gender.notna() & (gender!="<NA>")
female=(gender.str.startswith("F",na=False) & valid)
den=int(valid.sum()); nf=int(female.sum())
row("Demographics","Female sex",f"{nf:,} ({100*nf/den:.1f})")

v,_=median_iqr(a["weight_kg"],1,True); row("Demographics","Weight, kg",v)

# Race/ethnicity.
if "race_group" in a.columns:
    rg=a["race_group"].astype("string")
    valid=rg.notna() & (rg!="<NA>")
    den=int(valid.sum())
    known=[]
    for cat in ["White","Black","Asian"]:
        if (rg==cat).any():
            known.append(cat)
            val,_=category(rg,cat,False)
            row("Race/ethnicity",cat,val)
    n_other=int((valid & ~rg.isin(known)).sum())
    row("Race/ethnicity","Other/unknown",f"{n_other:,} ({100*n_other/den:.1f})")

# Clinically relevant comorbidities.
for col,label in [
    ("copd_history","Chronic obstructive pulmonary disease"),
    ("asthma_history","Asthma"),
    ("heart_failure","Heart failure"),
    ("diabetes","Diabetes mellitus"),
    ("chronic_kidney_disease","Chronic kidney disease"),
    ("chronic_liver_disease","Chronic liver disease"),
    ("malignancy","Malignancy"),
]:
    if col in a.columns:
        v,_=binary(a[col],False); row("Comorbidities",label,v)

# IMPORTANT: explicit time-zero support states from Stage5.5,
# not rolling support_imv/support_niv indicators from Stage7.
for col,label in [
    ("t0_invasive_device","Invasive respiratory support at respiratory time zero"),
    ("t0_niv","Noninvasive ventilation at respiratory time zero"),
    ("t0_hfnc_qualified","High-flow nasal oxygen at respiratory time zero"),
    ("shock_active_at_t0","Shock at respiratory time zero"),
]:
    if col in a.columns:
        v,_=binary(a[col],False); row("Clinical severity at respiratory time zero",label,v)

# Selected time-zero aligned physiology.
for col,label,d in [
    ("dyn_rr_max6","Respiratory rate, breaths/min",1),
    ("dyn_map_min6","Mean arterial pressure, mm Hg",1),
    ("dyn_sf_ratio","SpO2/FiO2 ratio",0),
    ("dyn_lactate_max12","Lactate, mmol/L",1),
    ("dyn_ph_min12","Arterial pH",2),
    ("dyn_creatinine_last24","Creatinine, mg/dL",2),
    ("dyn_wbc_last24","White blood cell count, 10^9/L",1),
]:
    if col in a.columns:
        v,_=median_iqr(a[col],d,True)
        row("Clinical severity at respiratory time zero",label,v)

main=pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Missingness appendix for the main Table 1 variables
# ---------------------------------------------------------------------
miss=[]
for c,label in [
    ("age","Age"),
    ("gender","Sex"),
    ("weight_kg","Weight"),
    ("dyn_rr_max6","Respiratory rate"),
    ("dyn_map_min6","Mean arterial pressure"),
    ("dyn_sf_ratio","SpO2/FiO2 ratio"),
    ("dyn_lactate_max12","Lactate"),
    ("dyn_ph_min12","Arterial pH"),
    ("dyn_creatinine_last24","Creatinine"),
    ("dyn_wbc_last24","White blood cell count"),
]:
    if c in a.columns:
        nmiss=int(a[c].isna().sum())
        miss.append({
            "Variable":label,
            "Available n":N-nmiss,
            "Missing n":nmiss,
            "Missing %":round(100*nmiss/N,1),
        })
missingness=pd.DataFrame(miss)

# ---------------------------------------------------------------------
# Comparison QA: explicit t0 support versus rolling Stage7 support
# ---------------------------------------------------------------------
compare=[]
for static,dynamic,label in [
    ("t0_invasive_device","support_imv","Invasive respiratory support"),
    ("t0_niv","support_niv","Noninvasive ventilation"),
    ("t0_hfnc_qualified","support_hfnc","High-flow nasal oxygen"),
]:
    if static in frozen.columns and dynamic in hour0.columns:
        st=num(frozen.set_index("stay_id")[static]).reindex(hour0["stay_id"]).reset_index(drop=True)
        dy=num(hour0[dynamic]).reset_index(drop=True)
        compare.append({
            "Characteristic":label,
            "Explicit_t0_n_pct":
                f"{int((st==1).sum()):,} ({100*(st==1).mean():.1f})",
            "Stage7_rolling_hour0_n_pct":
                f"{int((dy==1).sum()):,} ({100*(dy==1).mean():.1f})",
            "Preferred_for_Table1":"Explicit t0",
        })
comparison=pd.DataFrame(compare)

qa={
    "stage5_5_rows":len(s55),
    "final_frozen_rows":len(frozen),
    "unique_stay_ids":frozen["stay_id"].nunique(),
    "stage7_hour0_matched":hour0["stay_id"].nunique(),
    "table1_support_definition":
        "explicit respiratory-time-zero fields from Stage5.5",
    "dynamic_physiology_definition":
        "Stage7 hour==0 rolling history variables available at respiratory time zero",
}

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
main.to_csv(OUTDIR/"Table1_Main_Manuscript_V2.csv",index=False,encoding="utf-8-sig")
missingness.to_csv(OUTDIR/"Table1_Main_Missingness_V2.csv",index=False,encoding="utf-8-sig")
comparison.to_csv(OUTDIR/"Table1_QA_Support_Definition_Comparison.csv",index=False,encoding="utf-8-sig")
with open(OUTDIR/"Table1_QA_V2.json","w",encoding="utf-8") as f:
    json.dump(qa,f,ensure_ascii=False,indent=2)

with open(OUTDIR/"Table1_Main_Manuscript_V2.md","w",encoding="utf-8") as f:
    f.write("Table 1. Characteristics of the frozen MIMIC-IV primary cohort\n\n")
    f.write(main.to_markdown(index=False))
    f.write(
        "\n\nValues are median [IQR] or n (%). "
        "For variables with missing observations, the available sample size is shown in parentheses. "
        "Respiratory-support categories refer to the explicit respiratory time-zero state. "
        "Physiologic measurements are derived from information available at respiratory time zero. "
        "No hypothesis-test P values are reported.\n"
    )

print("\n"+"="*100)
print("TABLE 1 V2 COMPLETE")
print("="*100)
print(main.to_string(index=False))
print("\nMissingness:")
print(missingness.to_string(index=False))
print("\nSupport-definition QA:")
print(comparison.to_string(index=False))
print(f"\nOutputs: {OUTDIR}")

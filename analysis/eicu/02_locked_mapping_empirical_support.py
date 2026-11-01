from __future__ import annotations

"""
E2 — eICU locked external-validation mapping + outcome-blind positivity/balance gate

NO MORTALITY EFFECT IS ESTIMATED.

This stage freezes the eICU operational mapping BEFORE opening the external
steroid-outcome effect.

Frozen from MIMIC:
    - strategy: initiate systemic corticosteroid within 6h vs defer >=6h
    - sequential trial starts: 0,2,...,18h after respiratory t0
    - recent window: [3h,6h)
    - design family: sequential target-trial / pure clone-censor-weight
    - target: overlap / clinical-equipoise target
    - baseline overlap model: unpenalized logistic
    - defer censoring model: pooled 1h ridge logistic, C=0.5
    - recent-initiation model: unpenalized logistic

eICU-specific mapping locked here:
    - cohort seed comes from E1 explicit respiratory-t0 candidates
    - exclude any systemic steroid recorded before respiratory t0
    - require a hospital with observable medication interface
    - covariates use ONLY data available at/before each decision time
    - no APACHE first-day summaries are used in the time-varying PS because
      they can include post-treatment information

External outcome mapping is DECLARED here, but not analyzed:
    documented in-hospital death within 28 days from each trial baseline.
    eICU lacks reliable post-discharge vital status, so this is a partial
    external replication of the MIMIC 28-day all-cause endpoint.
"""

from pathlib import Path
import os
import re
import json
import math
import traceback
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
except Exception as exc:
    raise RuntimeError(
        "E2 requires scikit-learn.\n"
        "Run: pip install scikit-learn\n"
        f"Original error: {exc}"
    )

# =============================================================================
# PATHS
# =============================================================================
try:
    import config as _local_config
except ModuleNotFoundError:
    _local_config = None

_config_eicu_dir = getattr(_local_config, "EICU_DATA_DIR", None) if _local_config is not None else None
DATA_DIR = Path(os.environ.get("EICU_DATA_DIR", str(_config_eicu_dir) if _config_eicu_dir is not None else "PATH_TO_EICU_DATA"))
HERE = Path(__file__).resolve().parent

E1_DIR_CANDIDATES = [
    HERE / "outputs_eicu_E1",
    Path.cwd() / "outputs_eicu_E1",
]
E1_DIR = next((p for p in E1_DIR_CANDIDATES if p.exists()), E1_DIR_CANDIDATES[0])

OUTPUT_DIR = HERE / "outputs_eicu_E2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 300_000
PS_CLIP = 1e-5
SEED = 20261028

TRIAL_STARTS = list(range(0, 20, 2))  # 0,2,...,18 h
WINDOW_H = 6
RECENT_H0 = 3
RECENT_H1 = 6
PANEL_HOURS = list(range(0, 24))

# Outcome-blind site/interface rule declared before E2.
MIN_CANDIDATE_STAYS_PER_HOSPITAL = 20
MIN_ANY_MEDICATION_COVERAGE = 0.20

# Frozen support gates.
RECENT_ESS_FLOOR = 50
MAX_ABS_SMD = 0.10
REPRESENTATION_LOW = 0.90
REPRESENTATION_HIGH = 1.10

# External multicenter support diagnostics.
MIN_HOSPITALS_WITH_RECENT_INITIATION = 10
MAX_SINGLE_HOSPITAL_RECENT_SHARE = 0.30

# =============================================================================
# REGEX / PHENOTYPE HELPERS
# =============================================================================
VASO_RE = re.compile(
    r"\b("
    r"norepinephrine|noradrenaline|levophed|"
    r"epinephrine|adrenaline|"
    r"phenylephrine|neo[- ]?synephrine|"
    r"vasopressin|"
    r"dopamine"
    r")\b",
    re.I,
)

IMV_RE = re.compile(
    r"\b("
    r"mechanical\s+ventilation|"
    r"endotracheal|"
    r"intubat(?:e|ed|ion)|"
    r"ventilator"
    r")\b",
    re.I,
)
NIV_RE = re.compile(
    r"\b(non[- ]?invasive\s+ventilation|\bniv\b|\bbipap\b|\bcpap\b)\b",
    re.I,
)
HFNC_RE = re.compile(
    r"\b(high[- ]?flow|hfnc)\b",
    re.I,
)
RESP_NEGATIVE_RE = re.compile(
    r"\b(wean(?:ing|ed)?|extubat(?:e|ed|ion)|discontinu(?:e|ed|ation)|off\s+(?:the\s+)?vent)\b",
    re.I,
)

COMORBIDITY_REGEX = {
    "copd_history": re.compile(r"\b(copd|chronic obstructive)\b", re.I),
    "asthma_history": re.compile(r"\basthma\b", re.I),
    "heart_failure": re.compile(r"\b(heart failure|congestive heart failure|\bchf\b)\b", re.I),
    "coronary_disease": re.compile(r"\b(coronary artery disease|\bcad\b|ischemic heart disease)\b", re.I),
    "diabetes": re.compile(r"\b(diabetes|diabetic)\b", re.I),
    "chronic_kidney_disease": re.compile(r"\b(chronic kidney|chronic renal|\bckd\b|end stage renal|\besrd\b)\b", re.I),
    "chronic_liver_disease": re.compile(r"\b(cirrhosis|chronic liver|hepatic failure)\b", re.I),
    "malignancy": re.compile(r"\b(cancer|malignan|neoplasm|leukemia|lymphoma|myeloma|metast)\w*\b", re.I),
    "dementia": re.compile(r"\b(dementia|alzheimer)\w*\b", re.I),
}

# =============================================================================
# OUTPUT FILES
# =============================================================================
OUT_COHORT = OUTPUT_DIR / "E2_01_LOCKED_COHORT_COUNTS.csv"
OUT_HOSP = OUTPUT_DIR / "E2_02_HOSPITAL_INTERFACE_GATE.csv"
OUT_TRIAL = OUTPUT_DIR / "E2_03_SEQUENTIAL_TRIAL_COUNTS.csv"
OUT_BASE = OUTPUT_DIR / "E2_04_BASELINE_OVERLAP.csv"
OUT_DEFER_BAL = OUTPUT_DIR / "E2_05_DEFER_CENSORING_BALANCE.csv"
OUT_DEFER_SUM = OUTPUT_DIR / "E2_06_DEFER_HOURLY_SUMMARY.csv"
OUT_RECENT_BAL = OUTPUT_DIR / "E2_07_RECENT_INITIATOR_BALANCE.csv"
OUT_RECENT_SUM = OUTPUT_DIR / "E2_08_RECENT_INITIATOR_SUMMARY.csv"
OUT_SITE = OUTPUT_DIR / "E2_09_RECENT_INITIATION_BY_HOSPITAL.csv"
OUT_MISSING = OUTPUT_DIR / "E2_10_COVARIATE_MISSINGNESS.csv"
OUT_RECOMMEND = OUTPUT_DIR / "E2_11_RECOMMENDATION.csv"
OUT_LOCK = OUTPUT_DIR / "E2_12_EXTERNAL_PROTOCOL_LOCK.json"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_table(stem):
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"eICU folder not found: {DATA_DIR}\n"
            "Edit DATA_DIR at the top of the script if needed."
        )
    wanted = {norm(stem + ".csv.gz"), norm(stem + ".csv")}
    for p in DATA_DIR.iterdir():
        if p.is_file() and norm(p.name) in wanted:
            return p
    for p in DATA_DIR.iterdir():
        if p.is_file() and norm(p.name).startswith(norm(stem)):
            return p
    return None


def read_header(path):
    return pd.read_csv(path, nrows=0).columns.tolist()


def resolve_col(cols, name):
    m = {norm(c): c for c in cols}
    return m.get(norm(name))


def chunk_reader(stem, desired_cols=None):
    p = find_table(stem)
    if p is None:
        return
    cols = read_header(p)
    m = {norm(c): c for c in cols}
    use = None
    if desired_cols is not None:
        use = []
        for c in desired_cols:
            a = m.get(norm(c))
            if a is not None and a not in use:
                use.append(a)
    for ch in pd.read_csv(
        p,
        usecols=use,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        yield ch


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_int_set(s):
    x = pd.to_numeric(s, errors="coerce")
    return set(x[x.notna()].astype("int64").tolist())


def parse_numeric_text(s):
    x = s.astype("string").str.extract(
        r"([-+]?\d*\.?\d+)",
        expand=False,
    )
    return pd.to_numeric(x, errors="coerce")


def fit_unpenalized(X, y):
    attempts = [
        dict(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-9),
        dict(penalty="none", solver="lbfgs", max_iter=5000, tol=1e-9),
    ]
    last = None
    for kwargs in attempts:
        try:
            m = LogisticRegression(
                fit_intercept=True,
                random_state=SEED,
                **kwargs,
            )
            m.fit(X, y)
            return m
        except Exception as exc:
            last = exc
    raise RuntimeError(
        f"Frozen unpenalized logistic model failed: {last}"
    )


def fit_ridge(X, y, C=0.5):
    m = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    m.fit(X, y)
    return m


def ess(w):
    x = np.asarray(w, float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    return (x.sum() ** 2) / np.sum(x ** 2)


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


def weighted_smd(x1, w1, x0, w0):
    m1, v1 = weighted_mean_var(x1, w1)
    m0, v0 = weighted_mean_var(x0, w0)
    if not np.isfinite(m1) or not np.isfinite(m0):
        return np.nan
    den = math.sqrt((v1 + v0) / 2.0)
    return (m1 - m0) / den if den > 0 else 0.0


def require_e1_lock():
    rec_path = E1_DIR / "E1_11_RECOMMENDATION.csv"
    exp_path = E1_DIR / "E1_08_FROZEN_6H_EXPOSURE_FEASIBILITY.csv"
    if not rec_path.exists() or not exp_path.exists():
        raise FileNotFoundError(
            f"E1 outputs not found under {E1_DIR}\n"
            "Put the outputs_eicu_E1 folder next to this script."
        )
    rec = pd.read_csv(rec_path)
    if len(rec) != 1:
        raise RuntimeError("E1 recommendation must contain exactly one row.")
    row = rec.iloc[0]
    if str(row["decision"]) != "PROCEED_TO_E2_LOCKED_EXTERNAL_VALIDATION_MAPPING":
        raise RuntimeError(
            f"E2 abort: E1 decision was {row['decision']}"
        )
    if int(row["window_h"]) != 6 or str(row["recent_window"]) != "3-6h":
        raise RuntimeError(
            "E2 abort: E1 protocol does not match frozen 6h / 3-6h strategy."
        )
    print("[LOCK] E1 gate verified. 6h window / recent 3-6h remains frozen.")


def load_e1_seed():
    x = pd.read_csv(
        E1_DIR / "E1_08_FROZEN_6H_EXPOSURE_FEASIBILITY.csv"
    )
    need = {
        "patientunitstayid",
        "respiratory_t0_candidate_min",
        "first_systemic_steroid_offset_min",
        "steroid_relative_to_t0_min",
    }
    missing = need - set(x.columns)
    if missing:
        raise RuntimeError(
            f"E1 exposure feasibility missing columns: {sorted(missing)}"
        )
    x["patientunitstayid"] = to_num(x["patientunitstayid"]).astype("Int64")
    x["respiratory_t0_candidate_min"] = to_num(
        x["respiratory_t0_candidate_min"]
    )
    x["first_systemic_steroid_offset_min"] = to_num(
        x["first_systemic_steroid_offset_min"]
    )
    x["steroid_relative_to_t0_min"] = to_num(
        x["steroid_relative_to_t0_min"]
    )
    return x


def load_patient_mapping(seed_ids):
    p = find_table("patient")
    if p is None:
        raise FileNotFoundError("patient table missing.")
    cols = read_header(p)
    wanted = [
        "patientUnitStayID",
        "patientHealthSystemStayID",
        "hospitalID",
        "age",
        "gender",
        "ethnicity",
        "admissionWeight",
        "unitType",
        "unitAdmitSource",
        "unitVisitNumber",
        "unitDischargeOffset",
        "apacheAdmissionDx",
    ]
    actual = [resolve_col(cols, c) for c in wanted]
    actual = [c for c in actual if c is not None]
    df = pd.read_csv(p, usecols=actual, low_memory=False)

    idc = resolve_col(df.columns, "patientUnitStayID")
    df[idc] = to_num(df[idc]).astype("Int64")
    df = df[df[idc].isin(seed_ids)].copy()

    rename = {}
    for c in wanted:
        ac = resolve_col(df.columns, c)
        if ac:
            rename[ac] = norm(c)
    df = df.rename(columns=rename)

    # More readable canonical names.
    rename2 = {
        "patientunitstayid": "patientunitstayid",
        "patienthealthsystemstayid": "patienthealthsystemstayid",
        "hospitalid": "hospitalid",
        "admissionweight": "weight_kg",
        "unittype": "unit_type",
        "unitadmitsource": "unit_admit_source",
        "unitvisitnumber": "unit_visit_number",
        "unitdischargeoffset": "unit_discharge_offset_min",
        "apacheadmissiondx": "apache_admission_dx",
    }
    df = df.rename(columns=rename2)

    # age can include >89
    age = df["age"].astype("string").str.strip()
    age_num = pd.to_numeric(age, errors="coerce")
    age_num.loc[age.str.contains(r">\s*89", regex=True, na=False)] = 90
    df["age_years"] = age_num

    return df


def load_hospital():
    p = find_table("hospital")
    if p is None:
        return pd.DataFrame(columns=[
            "hospitalid", "hospital_region", "hospital_teaching",
            "hospital_beds"
        ])
    df = pd.read_csv(p, low_memory=False)
    out = pd.DataFrame()
    idc = resolve_col(df.columns, "hospitalID")
    out["hospitalid"] = to_num(df[idc]).astype("Int64")

    for src, dst in [
        ("region", "hospital_region"),
        ("teachingStatus", "hospital_teaching"),
        ("numBedsCategory", "hospital_beds"),
    ]:
        c = resolve_col(df.columns, src)
        out[dst] = (
            df[c].astype("string").fillna("Missing")
            if c else "Missing"
        )
    return out


def medication_interface_gate(patient_seed):
    """
    Outcome-blind data-interface gate.

    A hospital is eligible for the external treatment analysis only if:
      - >=20 E1 candidate t0 stays
      - >=20% of those candidate stays have at least one medication OR infusion record

    This prevents hospitals with structurally absent medication interfaces from
    masquerading as 'never treated'.
    """
    ids = safe_int_set(patient_seed["patientunitstayid"])
    any_med = set()

    for stem, idname in [
        ("medication", "patientUnitStayID"),
        ("infusionDrug", "patientUnitStayID"),
    ]:
        if find_table(stem) is None:
            continue
        for ch in chunk_reader(stem, [idname]):
            c = resolve_col(ch.columns, idname)
            if c is None:
                continue
            vals = to_num(ch[c])
            hit = vals[vals.isin(ids) & vals.notna()]
            any_med.update(hit.astype("int64").tolist())

    x = patient_seed[
        ["patientunitstayid", "hospitalid"]
    ].drop_duplicates().copy()
    x["has_any_med_interface_row"] = (
        x["patientunitstayid"].isin(any_med)
        .fillna(False)
        .astype("int8")
    )

    rows = []
    for hid, g in x.groupby("hospitalid", dropna=False):
        n = len(g)
        coverage = float(g["has_any_med_interface_row"].mean()) if n else 0
        passed = (
            n >= MIN_CANDIDATE_STAYS_PER_HOSPITAL
            and coverage >= MIN_ANY_MEDICATION_COVERAGE
        )
        rows.append({
            "hospitalid": hid,
            "candidate_explicit_t0_stays_n": n,
            "any_medication_interface_stays_n": int(
                g["has_any_med_interface_row"].sum()
            ),
            "any_medication_interface_coverage": coverage,
            "hospital_interface_pass": int(passed),
        })

    gate = pd.DataFrame(rows).sort_values(
        ["hospital_interface_pass", "candidate_explicit_t0_stays_n"],
        ascending=[False, False],
    )
    passed_hospitals = set(
        gate.loc[
            gate["hospital_interface_pass"] == 1,
            "hospitalid",
        ].dropna().astype(int)
    )
    return gate, passed_hospitals


def build_locked_cohort(seed, patient, hospital, passed_hospitals):
    x = seed.merge(
        patient,
        on="patientunitstayid",
        how="left",
        validate="one_to_one",
    )
    x = x.merge(
        hospital,
        on="hospitalid",
        how="left",
    )

    x["pre_t0_steroid"] = (
        x["steroid_relative_to_t0_min"].notna()
        & (x["steroid_relative_to_t0_min"] < 0)
    ).astype("int8")

    x["adult"] = (
        x["age_years"].ge(18)
        .fillna(False)
        .astype("int8")
    )

    x["hospital_interface_pass"] = (
        x["hospitalid"].isin(passed_hospitals)
        .fillna(False)
        .astype("int8")
    )

    x["still_in_icu_at_t0"] = (
        x["unit_discharge_offset_min"].isna()
        | (
            x["unit_discharge_offset_min"]
            > x["respiratory_t0_candidate_min"]
        )
    ).astype("int8")

    x["e2_locked_eligible"] = (
        (x["adult"] == 1)
        & (x["pre_t0_steroid"] == 0)
        & (x["hospital_interface_pass"] == 1)
        & (x["still_in_icu_at_t0"] == 1)
    ).astype("int8")

    return x


def comorbidity_flags(ids):
    ids = set(ids)
    result = {
        sid: {k: 0 for k in COMORBIDITY_REGEX}
        for sid in ids
    }

    for stem, textcols, offsetname in [
        ("admissionDx", ["admitDxPath", "admitDxName", "admitDxText"], "admitDxEnteredOffset"),
        ("diagnosis", ["diagnosisString", "ICD9Code"], "diagnosisOffset"),
    ]:
        if find_table(stem) is None:
            continue

        desired = ["patientUnitStayID", offsetname] + textcols
        for ch in chunk_reader(stem, desired):
            idc = resolve_col(ch.columns, "patientUnitStayID")
            oc = resolve_col(ch.columns, offsetname)
            if idc is None:
                continue
            sid = to_num(ch[idc])
            m = sid.notna() & sid.isin(ids)
            if not m.any():
                continue

            z = ch.loc[m].copy()
            sidz = to_num(z[idc])
            if oc:
                off = to_num(z[oc])
                if stem == "diagnosis":
                    z = z.loc[
                        off.between(0, 24*60, inclusive="both")
                        | off.isna()
                    ].copy()
                    sidz = to_num(z[idc])

            txt = pd.Series("", index=z.index, dtype="string")
            for tc in textcols:
                c = resolve_col(z.columns, tc)
                if c:
                    txt = (
                        txt
                        + " | "
                        + z[c].astype("string").fillna("")
                    )

            for flag, rx in COMORBIDITY_REGEX.items():
                hit = txt.str.contains(rx, na=False)
                for s in sidz[hit].dropna().astype("int64"):
                    if int(s) in result:
                        result[int(s)][flag] = 1

    rows = []
    for sid, d in result.items():
        row = {"patientunitstayid": sid}
        row.update(d)
        rows.append(row)
    return pd.DataFrame(rows)


def make_event_hours(df, cohort_t0, offset_col, id_col):
    z = df.copy()
    z["patientunitstayid"] = to_num(z[id_col]).astype("Int64")
    z["_offset"] = to_num(z[offset_col])
    z = z.merge(
        cohort_t0[["patientunitstayid", "respiratory_t0_candidate_min"]],
        on="patientunitstayid",
        how="inner",
    )
    z["_rel_min"] = (
        z["_offset"] - z["respiratory_t0_candidate_min"]
    )
    # The earliest integer decision hour at which the measurement is known.
    z["_avail_hour"] = np.ceil(z["_rel_min"] / 60.0)
    z = z[
        z["_avail_hour"].between(-24, 23, inclusive="both")
    ].copy()
    z["_avail_hour"] = z["_avail_hour"].astype(int)
    return z


def dense_hours(cohort):
    ids = cohort["patientunitstayid"].dropna().astype(int).tolist()
    grid = pd.MultiIndex.from_product(
        [ids, list(range(-24, 24))],
        names=["patientunitstayid", "hour"],
    ).to_frame(index=False)
    return grid


def rolling_from_hourly(hourly, cohort, spec):
    """
    spec:
      output_name -> (source_col, window, agg)
    agg in {"max","min","last","sum"}
    """
    grid = dense_hours(cohort)
    x = grid.merge(
        hourly,
        on=["patientunitstayid", "hour"],
        how="left",
    )
    x = x.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)

    out = x[["patientunitstayid", "hour"]].copy()

    for outname, (col, window, agg) in spec.items():
        vals = pd.to_numeric(x[col], errors="coerce")

        pieces = []
        for sid, idx in x.groupby("patientunitstayid", sort=False).groups.items():
            s = vals.loc[idx]
            if agg == "max":
                r = s.rolling(window, min_periods=1).max()
            elif agg == "min":
                r = s.rolling(window, min_periods=1).min()
            elif agg == "sum":
                r = s.fillna(0).rolling(window, min_periods=1).sum()
            elif agg == "last":
                # last observed value within the rolling window
                arr = s.to_numpy(float)
                res = np.full(len(arr), np.nan)
                for i in range(len(arr)):
                    lo = max(0, i-window+1)
                    w = arr[lo:i+1]
                    ok = np.isfinite(w)
                    if ok.any():
                        res[i] = w[np.where(ok)[0][-1]]
                r = pd.Series(res, index=s.index)
            else:
                raise ValueError(agg)
            pieces.append(r)

        joined = pd.concat(pieces).sort_index()
        out[outname] = joined.to_numpy()

    return out[out["hour"].between(0,23)].copy()


def extract_vitals(cohort):
    ids = safe_int_set(cohort["patientunitstayid"])
    t0 = cohort[["patientunitstayid","respiratory_t0_candidate_min"]]

    hourly_parts = []

    if find_table("vitalPeriodic") is not None:
        rows = []
        desired = [
            "patientUnitStayID", "observationOffset",
            "heartrate", "respiration", "systemicmean",
            "sao2", "temperature",
        ]
        for ch in chunk_reader("vitalPeriodic", desired):
            idc = resolve_col(ch.columns, "patientUnitStayID")
            oc = resolve_col(ch.columns, "observationOffset")
            sid = to_num(ch[idc])
            m = sid.notna() & sid.isin(ids)
            if not m.any():
                continue
            z = make_event_hours(ch.loc[m].copy(), t0, oc, idc)

            rec = pd.DataFrame({
                "patientunitstayid": z["patientunitstayid"],
                "hour": z["_avail_hour"],
                "hr": to_num(z[resolve_col(z.columns,"heartrate")])
                    if resolve_col(z.columns,"heartrate") else np.nan,
                "rr": to_num(z[resolve_col(z.columns,"respiration")])
                    if resolve_col(z.columns,"respiration") else np.nan,
                "map_periodic": to_num(z[resolve_col(z.columns,"systemicmean")])
                    if resolve_col(z.columns,"systemicmean") else np.nan,
                "spo2": to_num(z[resolve_col(z.columns,"sao2")])
                    if resolve_col(z.columns,"sao2") else np.nan,
                "temp": to_num(z[resolve_col(z.columns,"temperature")])
                    if resolve_col(z.columns,"temperature") else np.nan,
            })
            rows.append(rec)

        if rows:
            vp = pd.concat(rows, ignore_index=True)
            vp_hour = vp.groupby(
                ["patientunitstayid","hour"], as_index=False
            ).agg(
                hr=("hr","max"),
                rr=("rr","max"),
                map_periodic=("map_periodic","min"),
                spo2=("spo2","min"),
                temp=("temp","max"),
            )
        else:
            vp_hour = pd.DataFrame(columns=[
                "patientunitstayid","hour","hr","rr","map_periodic","spo2","temp"
            ])
    else:
        vp_hour = pd.DataFrame(columns=[
            "patientunitstayid","hour","hr","rr","map_periodic","spo2","temp"
        ])

    if find_table("vitalAperiodic") is not None:
        rows = []
        for ch in chunk_reader(
            "vitalAperiodic",
            ["patientUnitStayID","observationOffset","noninvasivemean"],
        ):
            idc=resolve_col(ch.columns,"patientUnitStayID")
            oc=resolve_col(ch.columns,"observationOffset")
            mc=resolve_col(ch.columns,"noninvasivemean")
            sid=to_num(ch[idc])
            m=sid.notna() & sid.isin(ids)
            if not m.any():
                continue
            z=make_event_hours(ch.loc[m].copy(),t0,oc,idc)
            rows.append(pd.DataFrame({
                "patientunitstayid":z["patientunitstayid"],
                "hour":z["_avail_hour"],
                "map_aperiodic":to_num(z[mc]),
            }))
        if rows:
            va=pd.concat(rows,ignore_index=True)
            va_hour=va.groupby(["patientunitstayid","hour"],as_index=False).agg(
                map_aperiodic=("map_aperiodic","min")
            )
        else:
            va_hour=pd.DataFrame(columns=["patientunitstayid","hour","map_aperiodic"])
    else:
        va_hour=pd.DataFrame(columns=["patientunitstayid","hour","map_aperiodic"])

    h=vp_hour.merge(va_hour,on=["patientunitstayid","hour"],how="outer")
    h["map_any"]=pd.concat(
        [h.get("map_periodic"),h.get("map_aperiodic")],
        axis=1
    ).min(axis=1,skipna=True)

    return rolling_from_hourly(
        h,
        cohort,
        {
            "dyn_hr_max6":("hr",6,"max"),
            "dyn_rr_max6":("rr",6,"max"),
            "dyn_map_min6":("map_any",6,"min"),
            "dyn_spo2_last6":("spo2",6,"last"),
            "dyn_temp_max6":("temp",6,"max"),
        },
    )


def extract_fio2_support(cohort):
    ids=safe_int_set(cohort["patientunitstayid"])
    t0=cohort[["patientunitstayid","respiratory_t0_candidate_min"]]

    fio_rows=[]
    if find_table("respiratoryCharting") is not None:
        for ch in chunk_reader(
            "respiratoryCharting",
            [
                "patientUnitStayID","respChartOffset",
                "respChartValueLabel","respChartValue",
                "respChartTypeCat",
            ],
        ):
            idc=resolve_col(ch.columns,"patientUnitStayID")
            oc=resolve_col(ch.columns,"respChartOffset")
            lc=resolve_col(ch.columns,"respChartValueLabel")
            vc=resolve_col(ch.columns,"respChartValue")
            if not all([idc,oc,vc]):
                continue
            sid=to_num(ch[idc])
            m=sid.notna() & sid.isin(ids)
            if not m.any():
                continue
            z=ch.loc[m].copy()
            label=(
                z[lc].astype("string").fillna("")
                if lc else pd.Series("",index=z.index,dtype="string")
            )
            typc=resolve_col(z.columns,"respChartTypeCat")
            typ=(
                z[typc].astype("string").fillna("")
                if typc else pd.Series("",index=z.index,dtype="string")
            )
            text=(label+" | "+typ).str.lower()
            mf=text.str.contains(r"fio2|fraction.*inspired|inspired.*oxygen",regex=True,na=False)
            if not mf.any():
                continue
            zz=make_event_hours(z.loc[mf].copy(),t0,oc,idc)
            val=parse_numeric_text(zz[vc])
            val=np.where((val>0)&(val<=1.5),val*100,val)
            val=pd.Series(val,index=zz.index)
            good=val.between(21,100,inclusive="both")
            if good.any():
                fio_rows.append(pd.DataFrame({
                    "patientunitstayid":zz.loc[good,"patientunitstayid"],
                    "hour":zz.loc[good,"_avail_hour"],
                    "fio2":val.loc[good],
                }))

    if fio_rows:
        f=pd.concat(fio_rows,ignore_index=True)
        f=f.sort_values(["patientunitstayid","hour"])
        fh=f.groupby(["patientunitstayid","hour"],as_index=False).agg(
            fio2=("fio2","last")
        )
    else:
        fh=pd.DataFrame(columns=["patientunitstayid","hour","fio2"])

    # Treatment-based support severity.
    supp_rows=[]
    if find_table("treatment") is not None:
        for ch in chunk_reader(
            "treatment",
            ["patientUnitStayID","treatmentOffset","treatmentString"],
        ):
            idc=resolve_col(ch.columns,"patientUnitStayID")
            oc=resolve_col(ch.columns,"treatmentOffset")
            tc=resolve_col(ch.columns,"treatmentString")
            sid=to_num(ch[idc])
            m=sid.notna() & sid.isin(ids)
            if not m.any():
                continue
            z=ch.loc[m].copy()
            txt=z[tc].astype("string").fillna("")
            pos=~txt.str.contains(RESP_NEGATIVE_RE,na=False)
            level=np.zeros(len(z),dtype=float)
            level[txt.str.contains(HFNC_RE,na=False)] = 1
            level[txt.str.contains(NIV_RE,na=False)] = np.maximum(
                level[txt.str.contains(NIV_RE,na=False)],2
            )
            level[txt.str.contains(IMV_RE,na=False)] = np.maximum(
                level[txt.str.contains(IMV_RE,na=False)],3
            )
            keep=pos.to_numpy() & (level>0)
            if keep.any():
                # IMPORTANT: attach support level BEFORE make_event_hours().
                # make_event_hours() filters rows by relative time, so using
                # the pre-filter numpy array after that step can create a
                # length mismatch (e.g. 1787 vs 796).
                zk=z.loc[keep].copy()
                zk["_support_level_event"]=level[keep]
                zz=make_event_hours(zk,t0,oc,idc)
                supp_rows.append(pd.DataFrame({
                    "patientunitstayid":zz["patientunitstayid"].to_numpy(),
                    "hour":zz["_avail_hour"].to_numpy(),
                    "support_level_event":pd.to_numeric(
                        zz["_support_level_event"],errors="coerce"
                    ).to_numpy(),
                }))

    if supp_rows:
        s=pd.concat(supp_rows,ignore_index=True)
        sh=s.groupby(["patientunitstayid","hour"],as_index=False).agg(
            support_level_event=("support_level_event","max")
        )
    else:
        sh=pd.DataFrame(columns=[
            "patientunitstayid","hour","support_level_event"
        ])

    h=fh.merge(sh,on=["patientunitstayid","hour"],how="outer")
    out=rolling_from_hourly(
        h,
        cohort,
        {
            "dyn_fio2_last6":("fio2",6,"last"),
            "support_level":("support_level_event",6,"max"),
        },
    )
    out["support_level"]=out["support_level"].fillna(0)
    out["support_imv"]=(out["support_level"]>=3).astype("int8")
    out["support_niv"]=(out["support_level"]==2).astype("int8")
    out["support_hfnc"]=(out["support_level"]==1).astype("int8")
    return out


def extract_labs(cohort):
    ids=safe_int_set(cohort["patientunitstayid"])
    t0=cohort[["patientunitstayid","respiratory_t0_candidate_min"]]

    lab_rows=[]
    if find_table("lab") is None:
        return dense_hours(cohort).query("hour>=0").copy()

    for ch in chunk_reader(
        "lab",
        ["patientUnitStayID","labResultOffset","labName","labResult","labResultText"],
    ):
        idc=resolve_col(ch.columns,"patientUnitStayID")
        oc=resolve_col(ch.columns,"labResultOffset")
        nc=resolve_col(ch.columns,"labName")
        rc=resolve_col(ch.columns,"labResult")
        sid=to_num(ch[idc])
        m=sid.notna() & sid.isin(ids)
        if not m.any():
            continue

        z=ch.loc[m].copy()
        zz=make_event_hours(z,t0,oc,idc)
        # Use columns carried through make_event_hours(); do not align a
        # pre-filter Series by the post-merge dataframe index.
        name=zz[nc].astype("string").str.lower().fillna("")
        val=to_num(zz[rc])

        var=np.full(len(zz),"",dtype=object)
        ns=name.astype(str).to_numpy()

        def assign(mask,label):
            var[mask]=label

        assign(np.array([bool(re.search(r"\blactate\b|lactic acid",x)) for x in ns]),"lactate")
        assign(np.array([bool(re.search(r"(^|[^a-z])ph([^a-z]|$)",x)) for x in ns]),"ph")
        assign(np.array([bool(re.search(r"pao2|po2.*arterial|arterial.*po2",x)) for x in ns]),"pao2")
        assign(np.array([bool(re.search(r"paco2|pco2.*arterial|arterial.*pco2",x)) for x in ns]),"paco2")
        assign(np.array([bool(re.search(r"\bcreatinine\b",x)) for x in ns]),"creatinine")
        assign(np.array([bool(re.search(r"bicarbonate|hco3",x)) for x in ns]),"bicarbonate")
        assign(np.array([bool(re.search(r"\bglucose\b",x)) for x in ns]),"glucose")
        assign(np.array([bool(re.search(r"\bwbc\b|white blood",x)) for x in ns]),"wbc")

        good=(var!="") & np.isfinite(val.to_numpy(float))
        if good.any():
            lab_rows.append(pd.DataFrame({
                "patientunitstayid":zz.loc[good,"patientunitstayid"],
                "hour":zz.loc[good,"_avail_hour"],
                "var":var[good],
                "value":val.loc[good].to_numpy(float),
            }))

    if not lab_rows:
        return dense_hours(cohort).query("hour>=0").copy()

    l=pd.concat(lab_rows,ignore_index=True)

    pieces=[]
    for variable,g in l.groupby("var"):
        # within-hour summary
        if variable in ["lactate"]:
            hg=g.groupby(["patientunitstayid","hour"],as_index=False).agg(value=("value","max"))
        elif variable in ["ph"]:
            hg=g.groupby(["patientunitstayid","hour"],as_index=False).agg(value=("value","min"))
        else:
            hg=g.sort_values(["patientunitstayid","hour"]).groupby(
                ["patientunitstayid","hour"],as_index=False
            ).tail(1)[["patientunitstayid","hour","value"]]
        hg=hg.rename(columns={"value":variable})
        pieces.append(hg)

    h=pieces[0]
    for p in pieces[1:]:
        h=h.merge(p,on=["patientunitstayid","hour"],how="outer")

    specs={}
    if "lactate" in h: specs["dyn_lactate_max12"]=("lactate",12,"max")
    if "ph" in h: specs["dyn_ph_min12"]=("ph",12,"min")
    if "pao2" in h: specs["dyn_pao2_last12"]=("pao2",12,"last")
    if "paco2" in h: specs["dyn_paco2_last12"]=("paco2",12,"last")
    if "creatinine" in h: specs["dyn_creatinine_last24"]=("creatinine",24,"last")
    if "bicarbonate" in h: specs["dyn_bicarbonate_last24"]=("bicarbonate",24,"last")
    if "glucose" in h: specs["dyn_glucose_last24"]=("glucose",24,"last")
    if "wbc" in h: specs["dyn_wbc_last24"]=("wbc",24,"last")

    if not specs:
        return dense_hours(cohort).query("hour>=0").copy()

    return rolling_from_hourly(h,cohort,specs)


def extract_vasopressors(cohort):
    ids=safe_int_set(cohort["patientunitstayid"])
    t0=cohort[["patientunitstayid","respiratory_t0_candidate_min"]]

    rows=[]
    if find_table("infusionDrug") is not None:
        for ch in chunk_reader(
            "infusionDrug",
            ["patientUnitStayID","infusionOffset","drugName"],
        ):
            idc=resolve_col(ch.columns,"patientUnitStayID")
            oc=resolve_col(ch.columns,"infusionOffset")
            nc=resolve_col(ch.columns,"drugName")
            sid=to_num(ch[idc])
            m=sid.notna() & sid.isin(ids)
            if not m.any():
                continue
            z=ch.loc[m].copy()
            txt=z[nc].astype("string").fillna("")
            vaso=txt.str.contains(VASO_RE,na=False)
            if not vaso.any():
                continue
            zz=make_event_hours(z.loc[vaso].copy(),t0,oc,idc)
            rows.append(pd.DataFrame({
                "patientunitstayid":zz["patientunitstayid"],
                "hour":zz["_avail_hour"],
                "vaso_event":1.0,
            }))

    if rows:
        v=pd.concat(rows,ignore_index=True)
        vh=v.groupby(["patientunitstayid","hour"],as_index=False).agg(
            vaso_event=("vaso_event","max")
        )
    else:
        vh=pd.DataFrame(columns=["patientunitstayid","hour","vaso_event"])

    out=rolling_from_hourly(
        vh,
        cohort,
        {
            "vasopressor_active":("vaso_event",6,"max"),
        },
    )
    out["vasopressor_active"]=out["vasopressor_active"].fillna(0).astype("int8")
    return out


def build_dynamic_panel(cohort):
    print("  extracting vitals...")
    v=extract_vitals(cohort)
    print("  extracting FiO2/support...")
    r=extract_fio2_support(cohort)
    print("  extracting labs...")
    l=extract_labs(cohort)
    print("  extracting vasopressors...")
    p=extract_vasopressors(cohort)

    panel=v.merge(r,on=["patientunitstayid","hour"],how="outer")
    panel=panel.merge(l,on=["patientunitstayid","hour"],how="outer")
    panel=panel.merge(p,on=["patientunitstayid","hour"],how="outer")

    panel["dyn_sf_ratio"]=(
        panel["dyn_spo2_last6"]
        / (panel["dyn_fio2_last6"]/100.0)
    )
    panel.loc[
        ~panel["dyn_sf_ratio"].between(50,1000,inclusive="both"),
        "dyn_sf_ratio"
    ]=np.nan

    return panel


def build_trials(cohort, panel):
    rows=[]
    for h in TRIAL_STARTS:
        x=cohort.copy()
        start_min=h*60

        steroid_rel=x["steroid_relative_to_t0_min"]
        untreated=steroid_rel.isna() | (steroid_rel >= start_min)
        still_icu=(
            x["unit_discharge_offset_min"].isna()
            | (
                x["unit_discharge_offset_min"]
                > x["respiratory_t0_candidate_min"]+start_min
            )
        )
        x=x[untreated & still_icu].copy()

        steroid_rel=x["steroid_relative_to_t0_min"]
        x["trial_hour"]=h
        x["trial_hour2"]=h*h
        x["actual_initiate_next6h"]=(
            steroid_rel.notna()
            & (steroid_rel >= start_min)
            & (steroid_rel < start_min+WINDOW_H*60)
        ).astype("int8")
        x["early_0_3h"]=(
            steroid_rel.notna()
            & (steroid_rel >= start_min)
            & (steroid_rel < start_min+RECENT_H0*60)
        ).astype("int8")
        x["recent_3_6h"]=(
            steroid_rel.notna()
            & (steroid_rel >= start_min+RECENT_H0*60)
            & (steroid_rel < start_min+RECENT_H1*60)
        ).astype("int8")

        rows.append(x)

    tr=pd.concat(rows,ignore_index=True)
    tr["trial_idx"]=np.arange(len(tr),dtype=int)

    tr=tr.merge(
        panel,
        left_on=["patientunitstayid","trial_hour"],
        right_on=["patientunitstayid","hour"],
        how="left",
    ).drop(columns=["hour"],errors="ignore")
    return tr


BASELINE_NUM=[
    "age_years","weight_kg",
]
BASELINE_BIN=list(COMORBIDITY_REGEX.keys())
BASELINE_CAT=[
    "gender","ethnicity","unit_type","unit_admit_source",
    "hospital_region","hospital_teaching","hospital_beds",
]
DYNAMIC_NUM=[
    "dyn_hr_max6","dyn_rr_max6","dyn_map_min6",
    "dyn_spo2_last6","dyn_fio2_last6","dyn_sf_ratio",
    "dyn_temp_max6","dyn_lactate_max12","dyn_ph_min12",
    "dyn_pao2_last12","dyn_paco2_last12",
    "dyn_creatinine_last24","dyn_bicarbonate_last24",
    "dyn_glucose_last24","dyn_wbc_last24",
    "support_level",
]
DYNAMIC_BIN=[
    "support_imv","support_niv","support_hfnc",
    "vasopressor_active",
]


def existing(cols,df):
    return [c for c in cols if c in df.columns]


def make_preprocessor(df, include_decision=False):
    nums=existing(BASELINE_NUM+BASELINE_BIN+DYNAMIC_NUM+DYNAMIC_BIN,df)
    timecols=["trial_hour","trial_hour2"]
    if include_decision:
        timecols += existing(
            ["interval_index","decision_hour","decision_hour2"],df
        )
    nums += [c for c in timecols if c in df.columns and c not in nums]
    cats=existing(BASELINE_CAT,df)

    transformers=[]
    if nums:
        transformers.append((
            "num",
            Pipeline([
                ("imputer",SimpleImputer(strategy="median",add_indicator=True)),
                ("scale",StandardScaler()),
            ]),
            nums,
        ))
    if cats:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=10)),
            ]),
            cats,
        ))
    return ColumnTransformer(transformers,remainder="drop")


def baseline_overlap(trials):
    prep=make_preprocessor(trials,False)
    X=prep.fit_transform(trials)
    y=trials["actual_initiate_next6h"].astype(int).to_numpy()
    model=fit_unpenalized(X,y)
    e=np.clip(model.predict_proba(X)[:,1],PS_CLIP,1-PS_CLIP)
    q=e*(1-e)
    return e,q


def build_defer_long(trials,panel):
    parts=[]
    for j in range(6):
        z=trials.copy()
        z["interval_index"]=j
        z["decision_hour"]=z["trial_hour"]+j
        z["decision_hour2"]=z["decision_hour"]**2

        abs_rel=z["decision_hour"]*60
        steroid=z["steroid_relative_to_t0_min"]
        at_risk=steroid.isna() | (steroid >= abs_rel)
        still_icu=(
            z["unit_discharge_offset_min"].isna()
            | (
                z["unit_discharge_offset_min"]
                > z["respiratory_t0_candidate_min"]+abs_rel
            )
        )
        z=z[at_risk & still_icu].copy()

        steroid=z["steroid_relative_to_t0_min"]
        z["start_next1h"]=(
            steroid.notna()
            & (steroid >= z["decision_hour"]*60)
            & (steroid < (z["decision_hour"]+1)*60)
        ).astype("int8")

        z=z.drop(columns=existing(DYNAMIC_NUM+DYNAMIC_BIN,z),errors="ignore")
        z=z.merge(
            panel,
            left_on=["patientunitstayid","decision_hour"],
            right_on=["patientunitstayid","hour"],
            how="left",
        ).drop(columns=["hour"],errors="ignore")
        parts.append(z)
    return pd.concat(parts,ignore_index=True)


def balance_frame(df):
    out={}
    for c in DYNAMIC_NUM+DYNAMIC_BIN+BASELINE_NUM+BASELINE_BIN+[
        "trial_hour","decision_hour"
    ]:
        if c not in df.columns or c in out:
            continue
        s=to_num(df[c])
        med=s.median()
        out[c]=s.fillna(med if pd.notna(med) else 0).astype(float)
        if s.isna().any():
            out[c+"__missing"]=s.isna().astype(float)
    for c in BASELINE_CAT:
        if c not in df.columns:
            continue
        s=df[c].astype("string").fillna("Missing")
        for lev in sorted(s.unique().tolist()):
            out[f"{c}=={lev}"]=(s==lev).astype(float)
    return pd.DataFrame(out,index=df.index)


def defer_audit(trials,long,q):
    prep=make_preprocessor(long,True)
    X=prep.fit_transform(long)
    y=long["start_next1h"].astype(int).to_numpy()
    model=fit_ridge(X,y,C=0.5)
    p=np.clip(model.predict_proba(X)[:,1],PS_CLIP,1-PS_CLIP)

    # map q by trial index
    q_by_idx=dict(zip(trials["trial_idx"].astype(int),q))

    rows=[]
    sums=[]

    # cumulative IPCW per trial
    long2=long.copy()
    long2["pstart"]=p
    long2=long2.sort_values(["trial_idx","interval_index"])
    long2["inv_no"]=1/(1-long2["pstart"])
    long2["prior_ipcw"]=(
        long2.groupby("trial_idx")["inv_no"]
        .cumprod()
        / long2["inv_no"]
    )
    long2["current_ipcw"]=(
        long2.groupby("trial_idx")["inv_no"].cumprod()
    )
    long2["q"]=long2["trial_idx"].map(q_by_idx)

    for j in range(6):
        z=long2[long2["interval_index"]==j].copy()
        # target = all at risk at start; survivors = no start this hour
        target=np.ones(len(z),dtype=bool)
        surv=z["start_next1h"].to_numpy(int)==0
        tw=(z["q"]*z["prior_ipcw"]).to_numpy(float)
        sw=(z["q"]*z["current_ipcw"]).to_numpy(float)

        bf=balance_frame(z)
        local=[]
        for c in bf.columns:
            x=bf[c].to_numpy(float)
            smd=weighted_smd(x[surv],sw[surv],x[target],tw[target])
            rr={
                "interval_index":j,
                "variable":c,
                "smd_after_ipcw":smd,
                "abs_smd_after_ipcw":abs(smd) if pd.notna(smd) else np.nan,
            }
            rows.append(rr); local.append(rr)
        b=pd.DataFrame(local)
        sums.append({
            "interval_index":j,
            "at_risk_person_hours_n":len(z),
            "starts_next1h_n":int(z["start_next1h"].sum()),
            "start_rate":float(z["start_next1h"].mean()),
            "max_abs_smd":b["abs_smd_after_ipcw"].max(),
            "n_abs_smd_gt_0_10":int((b["abs_smd_after_ipcw"]>.10).sum()),
            "n_abs_smd_gt_0_20":int((b["abs_smd_after_ipcw"]>.20).sum()),
            "hourly_no_start_ipcw_p99":float(np.quantile(z["inv_no"],.99)),
            "hourly_no_start_ipcw_max":float(z["inv_no"].max()),
            "cumulative_no_start_ipcw_p99":float(np.quantile(z["current_ipcw"],.99)),
            "cumulative_no_start_ipcw_max":float(z["current_ipcw"].max()),
        })
    return pd.DataFrame(rows),pd.DataFrame(sums)


def recent_audit(trials,panel,q):
    z=trials.copy()
    z["decision_hour"]=z["trial_hour"]+3
    z["decision_hour2"]=z["decision_hour"]**2

    steroid=z["steroid_relative_to_t0_min"]
    untreated=steroid.isna() | (steroid >= z["decision_hour"]*60)
    still_icu=(
        z["unit_discharge_offset_min"].isna()
        | (
            z["unit_discharge_offset_min"]
            > z["respiratory_t0_candidate_min"]+z["decision_hour"]*60
        )
    )
    z=z[untreated & still_icu].copy()
    steroid=z["steroid_relative_to_t0_min"]
    z["recent_initiate_3_6h"]=(
        steroid.notna()
        & (steroid >= z["decision_hour"]*60)
        & (steroid < (z["trial_hour"]+6)*60)
    ).astype("int8")

    z=z.drop(columns=existing(DYNAMIC_NUM+DYNAMIC_BIN,z),errors="ignore")
    z=z.merge(
        panel,
        left_on=["patientunitstayid","decision_hour"],
        right_on=["patientunitstayid","hour"],
        how="left",
    ).drop(columns=["hour"],errors="ignore")

    prep=make_preprocessor(z,True)
    X=prep.fit_transform(z)
    y=z["recent_initiate_3_6h"].astype(int).to_numpy()
    model=fit_unpenalized(X,y)
    p=np.clip(model.predict_proba(X)[:,1],PS_CLIP,1-PS_CLIP)

    q_by_idx=dict(zip(trials["trial_idx"].astype(int),q))
    tw=z["trial_idx"].map(q_by_idx).to_numpy(float)
    rw=tw*(1/p)

    positive=y==1

    bf=balance_frame(z)
    rows=[]
    for c in bf.columns:
        x=bf[c].to_numpy(float)
        smd=weighted_smd(x[positive],rw[positive],x,tw)
        rows.append({
            "variable":c,
            "smd_after_weighting":smd,
            "abs_smd_after_weighting":abs(smd) if pd.notna(smd) else np.nan,
        })
    bal=pd.DataFrame(rows)

    target_sum=float(tw.sum())
    recent_sum=float(rw[positive].sum())
    ratio=recent_sum/target_sum if target_sum>0 else np.nan

    summary=pd.DataFrame([{
        "recent_person_trials_n":int(positive.sum()),
        "recent_unique_patients_n":int(z.loc[positive,"patientunitstayid"].nunique()),
        "target_riskset_person_trials_n":len(z),
        "target_weight_sum":target_sum,
        "recent_weighted_sum":recent_sum,
        "representation_ratio":ratio,
        "recent_weight_ess":ess(rw[positive]),
        "max_abs_smd":bal["abs_smd_after_weighting"].max(),
        "mean_abs_smd":bal["abs_smd_after_weighting"].mean(),
        "n_abs_smd_gt_0_10":int((bal["abs_smd_after_weighting"]>.10).sum()),
        "n_abs_smd_gt_0_20":int((bal["abs_smd_after_weighting"]>.20).sum()),
        "p_recent_min_positive":float(p[positive].min()) if positive.any() else np.nan,
        "p_recent_median_positive":float(np.median(p[positive])) if positive.any() else np.nan,
        "recent_ipcw_p95":float(np.quantile(1/p[positive],.95)) if positive.any() else np.nan,
        "recent_ipcw_p99":float(np.quantile(1/p[positive],.99)) if positive.any() else np.nan,
        "recent_ipcw_max":float(np.max(1/p[positive])) if positive.any() else np.nan,
    }])

    site=(
        z.loc[positive]
        .groupby("hospitalid",dropna=False)
        .agg(
            recent_person_trials_n=("trial_idx","size"),
            recent_unique_patients_n=("patientunitstayid","nunique"),
        )
        .reset_index()
        .sort_values("recent_person_trials_n",ascending=False)
    )
    if len(site):
        total=site["recent_person_trials_n"].sum()
        site["recent_share"]=site["recent_person_trials_n"]/total
    else:
        site["recent_share"]=np.nan

    return bal,summary,site


def missingness_table(trials):
    rows=[]
    for c in DYNAMIC_NUM+DYNAMIC_BIN+BASELINE_NUM+BASELINE_BIN+BASELINE_CAT:
        if c not in trials.columns:
            rows.append({
                "variable":c,
                "present_in_mapping":0,
                "missing_pct":100.0,
            })
        else:
            rows.append({
                "variable":c,
                "present_in_mapping":1,
                "missing_pct":100*trials[c].isna().mean(),
            })
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    print("="*112)
    print("E2.1 — eICU locked mapping + outcome-blind positivity/balance gate (alignment fix)")
    print("="*112)
    print("NO mortality effect is estimated in E2.1.")
    print("[E2.1] Row-alignment fix enabled for respiratory support and lab extraction.")

    require_e1_lock()

    seed=load_e1_seed()
    seed_ids=safe_int_set(seed["patientunitstayid"])
    patient=load_patient_mapping(seed_ids)
    hospital=load_hospital()

    # Add comorbidities before interface gate/cohort.
    comorb=comorbidity_flags(seed_ids)
    patient=patient.merge(comorb,on="patientunitstayid",how="left")
    for c in COMORBIDITY_REGEX:
        patient[c]=patient[c].fillna(0).astype("int8")

    gate,passed_hospitals=medication_interface_gate(
        seed.merge(patient[["patientunitstayid","hospitalid"]],on="patientunitstayid",how="left")
    )
    cohort_all=build_locked_cohort(seed,patient,hospital,passed_hospitals)
    cohort=cohort_all[cohort_all["e2_locked_eligible"]==1].copy()

    counts=pd.DataFrame([
        {"step":"E1 explicit respiratory-t0 CAP-like seed","n":len(seed)},
        {"step":"exclude pre-t0 systemic steroid","n":int((cohort_all["pre_t0_steroid"]==0).sum())},
        {"step":"adult + still in ICU at t0","n":int(((cohort_all["pre_t0_steroid"]==0)&(cohort_all["adult"]==1)&(cohort_all["still_in_icu_at_t0"]==1)).sum())},
        {"step":"locked hospital medication-interface gate","n":len(cohort)},
        {"step":"hospitals passing interface gate","n":len(passed_hospitals)},
    ])
    counts.to_csv(OUT_COHORT,index=False,encoding="utf-8-sig")
    gate.to_csv(OUT_HOSP,index=False,encoding="utf-8-sig")

    if len(cohort)<500:
        raise RuntimeError(
            f"Locked eICU cohort only {len(cohort)} stays after outcome-blind interface/eligibility rules."
        )

    print(f"Locked eICU cohort: {len(cohort)} stays across {cohort.hospitalid.nunique()} hospitals")
    print("Building time-updated covariates (no outcome data)...")
    panel=build_dynamic_panel(cohort)

    trials=build_trials(cohort,panel)
    trial_counts=(
        trials.groupby("trial_hour",as_index=False)
        .agg(
            eligible_person_trials_n=("trial_idx","size"),
            initiate_within6h_n=("actual_initiate_next6h","sum"),
            early_0_3h_n=("early_0_3h","sum"),
            recent_3_6h_n=("recent_3_6h","sum"),
        )
    )
    trial_counts["initiate_within6h_rate"]=(
        trial_counts["initiate_within6h_n"]
        / trial_counts["eligible_person_trials_n"]
    )
    trial_counts.to_csv(OUT_TRIAL,index=False,encoding="utf-8-sig")

    print(f"Sequential person-trials: {len(trials)}; 6h initiations: {int(trials.actual_initiate_next6h.sum())}")

    # Frozen baseline overlap model: unpenalized, no fallback/model shopping.
    e,q=baseline_overlap(trials)
    base_rows=[]
    y=trials["actual_initiate_next6h"].to_numpy(int)
    for label,mask in [
        ("all",np.ones(len(y),bool)),
        ("6h_initiators",y==1),
        ("6h_defer",y==0),
    ]:
        pp=e[mask]
        qq=q[mask]
        base_rows.append({
            "group":label,
            "n":int(mask.sum()),
            "e6_min":float(pp.min()),
            "e6_p01":float(np.quantile(pp,.01)),
            "e6_median":float(np.median(pp)),
            "e6_p99":float(np.quantile(pp,.99)),
            "e6_max":float(pp.max()),
            "q_sum":float(qq.sum()),
            "q_ess":ess(qq),
        })
    pd.DataFrame(base_rows).to_csv(OUT_BASE,index=False,encoding="utf-8-sig")

    long=build_defer_long(trials,panel)
    defer_bal,defer_sum=defer_audit(trials,long,q)
    defer_bal.to_csv(OUT_DEFER_BAL,index=False,encoding="utf-8-sig")
    defer_sum.to_csv(OUT_DEFER_SUM,index=False,encoding="utf-8-sig")

    recent_bal,recent_sum,site=recent_audit(trials,panel,q)
    recent_bal.to_csv(OUT_RECENT_BAL,index=False,encoding="utf-8-sig")
    recent_sum.to_csv(OUT_RECENT_SUM,index=False,encoding="utf-8-sig")
    site.to_csv(OUT_SITE,index=False,encoding="utf-8-sig")

    miss=missingness_table(trials)
    miss.to_csv(OUT_MISSING,index=False,encoding="utf-8-sig")

    r=recent_sum.iloc[0]
    defer_pass=bool(
        (defer_sum["max_abs_smd"]<MAX_ABS_SMD).all()
        and (defer_sum["n_abs_smd_gt_0_10"]==0).all()
    )
    recent_pass=bool(
        REPRESENTATION_LOW <= r["representation_ratio"] <= REPRESENTATION_HIGH
        and r["recent_weight_ess"] >= RECENT_ESS_FLOOR
        and r["max_abs_smd"] < MAX_ABS_SMD
        and int(r["n_abs_smd_gt_0_10"]) == 0
    )

    hospitals_recent=int(site["hospitalid"].nunique()) if len(site) else 0
    max_share=float(site["recent_share"].max()) if len(site) else np.nan
    multicenter_pass=bool(
        hospitals_recent >= MIN_HOSPITALS_WITH_RECENT_INITIATION
        and np.isfinite(max_share)
        and max_share <= MAX_SINGLE_HOSPITAL_RECENT_SHARE
    )

    proceed=defer_pass and recent_pass and multicenter_pass

    rec=pd.DataFrame([{
        "external_validation_stage":"E2",
        "mimic_design_frozen":1,
        "locked_eicu_cohort_stays_n":len(cohort),
        "locked_hospitals_n":cohort["hospitalid"].nunique(),
        "sequential_person_trials_n":len(trials),
        "initiate_within6h_person_trials_n":int(trials["actual_initiate_next6h"].sum()),
        "recent_3_6h_person_trials_n":int(r["recent_person_trials_n"]),
        "recent_unique_patients_n":int(r["recent_unique_patients_n"]),
        "recent_representation_ratio":float(r["representation_ratio"]),
        "recent_weight_ess":float(r["recent_weight_ess"]),
        "recent_max_abs_smd":float(r["max_abs_smd"]),
        "recent_n_abs_smd_gt_0_10":int(r["n_abs_smd_gt_0_10"]),
        "defer_all_6_hours_pass":int(defer_pass),
        "recent_model_pass":int(recent_pass),
        "hospitals_with_recent_initiation_n":hospitals_recent,
        "largest_hospital_recent_share":max_share,
        "multicenter_support_pass":int(multicenter_pass),
        "external_outcome_predeclared":"documented in-hospital death within 28 days from trial baseline",
        "outcome_difference_from_mimic":"eICU lacks reliable post-discharge vital status; partial endpoint replication",
        "effect_opened_in_E2":0,
        "decision":(
            "PROCEED_TO_E3_ONCE_LOCKED_EXTERNAL_EFFECT"
            if proceed else
            "HOLD_EXTERNAL_EFFECT_SUPPORT_OR_BALANCE_LIMITED"
        ),
    }])
    rec.to_csv(OUT_RECOMMEND,index=False,encoding="utf-8-sig")

    lock={
        "locked_before_external_effect":True,
        "development_cohort":"MIMIC-IV",
        "external_cohort":"eICU-CRD v2.0",
        "population_mapping":{
            "seed":"E1 high-specificity CAP-like with explicit respiratory-support t0",
            "exclude_pre_t0_systemic_steroid":True,
            "adult_only":True,
            "must_remain_in_icu_at_t0":True,
            "hospital_interface_gate":{
                "min_candidate_stays":MIN_CANDIDATE_STAYS_PER_HOSPITAL,
                "min_any_medication_coverage":MIN_ANY_MEDICATION_COVERAGE,
            },
        },
        "strategy":{
            "initiate_window_h":6,
            "defer_window_h":6,
            "recent_window_h":[3,6],
            "sequential_trial_starts_h":TRIAL_STARTS,
        },
        "nuisance_models":{
            "baseline_overlap":"unpenalized logistic",
            "defer_censoring":"pooled 1h ridge logistic C=0.5",
            "recent_initiation":"unpenalized logistic at +3h predicting initiation in [3h,6h)",
        },
        "target":"ATO clinical-equipoise / overlap population",
        "external_outcome":{
            "name":"documented in-hospital death within 28 days from trial baseline",
            "note":"partial replication because post-discharge mortality is unavailable in eICU",
        },
        "effect_opened_in_E2":False,
        "recommendation":rec.iloc[0].to_dict(),
    }
    with open(OUT_LOCK,"w",encoding="utf-8") as f:
        json.dump(lock,f,ensure_ascii=False,indent=2,default=str)

    print("\n"+"="*112)
    print("E2 COMPLETE — NO MORTALITY EFFECT WAS OPENED")
    print("="*112)
    print(rec.to_string(index=False))
    print("\nGenerated outputs are available in the outputs_eicu_E2 folder.")


if __name__=="__main__":
    try:
        main()
    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise

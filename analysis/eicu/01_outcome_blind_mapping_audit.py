from __future__ import annotations

"""
E1 — eICU external-validation frozen-protocol feasibility + schema audit

IMPORTANT SCIENTIFIC RULE
=========================
This script DOES NOT estimate the steroid-mortality effect.

MIMIC development is finished. The external-validation protocol is now locked:
    - clinical question: initiate systemic corticosteroid within 6h vs defer >=6h
    - recent-initiation window: 3–6h
    - sequential target-trial logic / pure CCW
    - 28-day mortality target if eICU permits sufficiently faithful outcome mapping

E1 only answers:
    1) Can eICU v2.0 operationalize the frozen cohort and time zero?
    2) Which tables/columns/interfaces are available?
    3) Is systemic corticosteroid timing observable?
    4) Is respiratory support timing observable?
    5) Is outcome timing/status available?
    6) Is there enough support to proceed WITHOUT opening the treatment effect?

No outcome-by-treatment comparison is generated in E1.
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

# =============================================================================
# USER PATH — already filled from your screenshot.
# Change ONLY this line if your folder moves.
# =============================================================================
DATA_DIR = Path(os.environ.get("EICU_DATA_DIR", "PATH_TO_EICU_DATA"))

# Outputs are written beside this script, not inside the raw eICU folder.
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs_eicu_E1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 300_000
MAX_SAMPLE_ROWS_PER_TABLE = 250

# Frozen external-validation target (DO NOT CHANGE AFTER E1).
FROZEN_WINDOW_H = 6
FROZEN_RECENT_H0 = 3
FROZEN_RECENT_H1 = 6
RESPIRATORY_T0_SEARCH_MIN = 0
RESPIRATORY_T0_SEARCH_MAX = 24 * 60

# Conservative respiratory support markers.
RESP_SUPPORT_RE = re.compile(
    r"\b("
    r"mechanical\s+ventilation|"
    r"invasive\s+ventilation|"
    r"ventilator|"
    r"endotracheal|"
    r"intubat(?:e|ed|ion)|"
    r"non[- ]?invasive\s+ventilation|"
    r"\bniv\b|"
    r"\bbipap\b|"
    r"\bcpap\b|"
    r"high[- ]?flow|"
    r"\bhfnc\b"
    r")\b",
    re.I,
)

# Exclude purely historical/weaning/extubation references when possible.
RESP_NEGATIVE_RE = re.compile(
    r"\b("
    r"wean(?:ing|ed)?|"
    r"extubat(?:e|ed|ion)|"
    r"discontinu(?:e|ed|ation)|"
    r"off\s+(?:the\s+)?vent"
    r")\b",
    re.I,
)

PNEUMONIA_RE = re.compile(
    r"\b("
    r"pneumonia|"
    r"community[- ]acquired\s+pneumonia|"
    r"\bcap\b"
    r")\b",
    re.I,
)

# High-specificity exclusions for an external CAP-like cohort.
ASPIRATION_RE = re.compile(r"\baspirat(?:ion|e|ed)\b", re.I)
PJP_RE = re.compile(
    r"\b(pneumocystis|pneumocystis\s+jirovecii|\bpjp\b|\bpcp\b)\b",
    re.I,
)
TB_RE = re.compile(r"\b(tuberculosis|\btb\b)\b", re.I)
INFLUENZA_RE = re.compile(r"\b(influenza|flu\s+pneumonia)\b", re.I)

STEROID_RE = re.compile(
    r"\b("
    r"hydrocortisone|"
    r"methylprednisolone|methylpred|solu[- ]?medrol|"
    r"prednisone|prednisolone|"
    r"dexamethasone|decadron|"
    r"betamethasone|"
    r"cortisone"
    r")\b",
    re.I,
)

# For diagnosis text indicating acute respiratory failure.
RESP_FAILURE_RE = re.compile(
    r"\b("
    r"acute\s+respiratory\s+failure|"
    r"respiratory\s+failure|"
    r"acute\s+hypoxemic\s+respiratory\s+failure|"
    r"hypoxemic\s+respiratory\s+failure"
    r")\b",
    re.I,
)

EXPECTED_TABLES = [
    "patient",
    "admissionDx",
    "diagnosis",
    "medication",
    "infusionDrug",
    "admissionDrug",
    "treatment",
    "respiratoryCharting",
    "respiratoryCare",
    "nurseCharting",
    "vitalPeriodic",
    "vitalAperiodic",
    "lab",
    "apacheApsVar",
    "apachePatientResult",
    "hospital",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_table(stem: str) -> Path | None:
    """
    Case-insensitive finder for:
        table.csv.gz
        table.csv
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"eICU folder does not exist:\n{DATA_DIR}\n"
            "Edit DATA_DIR at the top of this script if the path changed."
        )

    wanted = {norm(stem + ".csv.gz"), norm(stem + ".csv")}
    for p in DATA_DIR.iterdir():
        if p.is_file() and norm(p.name) in wanted:
            return p

    # Fallback to stem-only normalized comparison.
    for p in DATA_DIR.iterdir():
        if not p.is_file():
            continue
        nm = norm(p.name)
        if nm.startswith(norm(stem)) and (
            nm.endswith(norm(".csv.gz")) or nm.endswith(norm(".csv"))
        ):
            return p
    return None


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def colmap(cols) -> dict[str, str]:
    return {norm(c): c for c in cols}


def resolve_col(cols, *names) -> str | None:
    m = colmap(cols)
    for name in names:
        if norm(name) in m:
            return m[norm(name)]
    return None


def safe_read_table(stem, usecols=None, nrows=None):
    p = find_table(stem)
    if p is None:
        return None
    if usecols is not None:
        cols = read_header(p)
        actual = []
        cm = colmap(cols)
        for c in usecols:
            ac = cm.get(norm(c))
            if ac is not None and ac not in actual:
                actual.append(ac)
        usecols = actual
    return pd.read_csv(
        p,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )


def chunk_reader(stem, desired_cols=None):
    p = find_table(stem)
    if p is None:
        return
    cols = read_header(p)
    cm = colmap(cols)

    actual = None
    if desired_cols is not None:
        actual = []
        for c in desired_cols:
            ac = cm.get(norm(c))
            if ac is not None and ac not in actual:
                actual.append(ac)

    for ch in pd.read_csv(
        p,
        usecols=actual,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        yield ch


def clean_text_series(df, candidates):
    cols = []
    for c in candidates:
        ac = resolve_col(df.columns, c)
        if ac is not None:
            cols.append(ac)
    if not cols:
        return pd.Series("", index=df.index, dtype="string")
    s = df[cols[0]].astype("string").fillna("")
    for c in cols[1:]:
        s = s + " | " + df[c].astype("string").fillna("")
    return s.str.lower()


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_int_values(s):
    """Return a plain int Series after dropping NA/non-numeric IDs."""
    x = pd.to_numeric(s, errors="coerce")
    x = x[x.notna()]
    return x.astype("int64")


def safe_id_value_map(df, id_col, value_col):
    """
    Build {patientUnitStayID: numeric_value} safely.
    Rows with missing/non-numeric ID or value are ignored.
    """
    if df is None or len(df) == 0:
        return {}
    if id_col not in df.columns or value_col not in df.columns:
        return {}

    ids = pd.to_numeric(df[id_col], errors="coerce")
    vals = pd.to_numeric(df[value_col], errors="coerce")
    ok = ids.notna() & vals.notna()

    if not ok.any():
        return {}

    return dict(
        zip(
            ids.loc[ok].astype("int64"),
            vals.loc[ok].astype(float),
        )
    )


def parse_age(s):
    x = s.astype("string").str.strip()
    out = pd.to_numeric(x, errors="coerce")
    over89 = x.str.contains(r">\s*89", regex=True, na=False)
    out.loc[over89] = 90.0
    return out


def save(df, name):
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def schema_audit():
    rows = []
    for stem in EXPECTED_TABLES:
        p = find_table(stem)
        if p is None:
            rows.append({
                "table": stem,
                "found": 0,
                "filename": "",
                "size_mb": np.nan,
                "columns_n": 0,
                "columns": "",
            })
            continue
        try:
            cols = read_header(p)
            rows.append({
                "table": stem,
                "found": 1,
                "filename": p.name,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                "columns_n": len(cols),
                "columns": " | ".join(cols),
            })
        except Exception as exc:
            rows.append({
                "table": stem,
                "found": 1,
                "filename": p.name,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                "columns_n": -1,
                "columns": f"READ_ERROR: {exc}",
            })
    return pd.DataFrame(rows)


def load_patient_base():
    p = find_table("patient")
    if p is None:
        raise FileNotFoundError("patient.csv.gz not found.")

    df = pd.read_csv(p, low_memory=False)
    cm = colmap(df.columns)

    idc = cm.get(norm("patientUnitStayID"))
    hsid = cm.get(norm("patientHealthSystemStayID"))
    pid = cm.get(norm("uniquePID"))
    agec = cm.get(norm("age"))
    hospid = cm.get(norm("hospitalID"))
    apdx = cm.get(norm("apacheAdmissionDx"))
    hdstatus = cm.get(norm("hospitalDischargeStatus"))
    hdoff = cm.get(norm("hospitalDischargeOffset"))
    ustatus = cm.get(norm("unitDischargeStatus"))
    uoff = cm.get(norm("unitDischargeOffset"))

    if idc is None:
        raise RuntimeError("patient table lacks patientUnitStayID.")

    out = pd.DataFrame({
        "patientunitstayid": to_num(df[idc]).astype("Int64"),
    })

    out["patienthealthsystemstayid"] = (
        to_num(df[hsid]).astype("Int64") if hsid else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    out["uniquepid"] = (
        df[pid].astype("string") if pid else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["hospitalid"] = (
        to_num(df[hospid]).astype("Int64") if hospid else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    out["age_years"] = (
        parse_age(df[agec]) if agec else np.nan
    )
    out["apache_admission_dx"] = (
        df[apdx].astype("string").fillna("") if apdx else ""
    )
    out["hospital_discharge_status"] = (
        df[hdstatus].astype("string") if hdstatus else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["hospital_discharge_offset_min"] = (
        to_num(df[hdoff]) if hdoff else np.nan
    )
    out["unit_discharge_status"] = (
        df[ustatus].astype("string") if ustatus else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["unit_discharge_offset_min"] = (
        to_num(df[uoff]) if uoff else np.nan
    )

    # Adult unit stays.
    out["adult"] = (out["age_years"] >= 18).fillna(False).astype("int8")

    # First ICU stay within the health-system/hospital encounter if available.
    # We do NOT use this to silently exclude data yet; it is reported for E2 protocol mapping.
    if out["patienthealthsystemstayid"].notna().any():
        out["_row"] = np.arange(len(out))
        first_idx = (
            out.sort_values(["patienthealthsystemstayid", "_row"])
            .groupby("patienthealthsystemstayid", dropna=False)
            .head(1)
            .index
        )
        out["first_unit_stay_in_hospital_encounter"] = 0
        out.loc[first_idx, "first_unit_stay_in_hospital_encounter"] = 1
        out = out.drop(columns="_row")
    else:
        out["first_unit_stay_in_hospital_encounter"] = np.nan

    return out


def diagnosis_audit(patient_ids):
    hit_ids = set()
    exclusion_ids = set()
    resp_failure_ids = set()
    samples = []
    top_terms = []

    # Patient.apacheAdmissionDx is handled separately in main.

    for stem, textcols, offsetcol in [
        ("admissionDx", ["admitDxPath", "admitDxName", "admitDxText"], "admitDxEnteredOffset"),
        ("diagnosis", ["diagnosisString", "ICD9Code"], "diagnosisOffset"),
    ]:
        p = find_table(stem)
        if p is None:
            continue

        for ch in chunk_reader(
            stem,
            ["patientUnitStayID", offsetcol] + textcols,
        ):
            idc = resolve_col(ch.columns, "patientUnitStayID")
            oc = resolve_col(ch.columns, offsetcol)
            if idc is None:
                continue

            ids = to_num(ch[idc])
            mask_id = ids.isin(patient_ids)
            if not mask_id.any():
                continue
            x = ch.loc[mask_id].copy()
            ids = to_num(x[idc])
            txt = clean_text_series(x, textcols)
            off = to_num(x[oc]) if oc else pd.Series(np.nan, index=x.index)

            # AdmissionDx is inherently admission-focused.
            # Diagnosis is restricted to first 24h for phenotype construction.
            if stem == "diagnosis" and oc is not None:
                timeok = off.between(0, 24 * 60, inclusive="both")
            else:
                timeok = pd.Series(True, index=x.index)

            pneu = txt.str.contains(PNEUMONIA_RE, na=False) & timeok
            excl = (
                txt.str.contains(ASPIRATION_RE, na=False)
                | txt.str.contains(PJP_RE, na=False)
                | txt.str.contains(TB_RE, na=False)
                | txt.str.contains(INFLUENZA_RE, na=False)
            ) & timeok
            rf = txt.str.contains(RESP_FAILURE_RE, na=False) & timeok

            hit_ids.update(safe_int_values(ids[pneu]).tolist())
            exclusion_ids.update(safe_int_values(ids[excl]).tolist())
            resp_failure_ids.update(safe_int_values(ids[rf]).tolist())

            if pneu.any() and len(samples) < MAX_SAMPLE_ROWS_PER_TABLE:
                tmp = pd.DataFrame({
                    "table": stem,
                    "patientunitstayid": ids[pneu].astype("Int64"),
                    "offset_min": off[pneu],
                    "text": txt[pneu],
                }).head(MAX_SAMPLE_ROWS_PER_TABLE - len(samples))
                samples.extend(tmp.to_dict("records"))

    return hit_ids, exclusion_ids, resp_failure_ids, pd.DataFrame(samples)


def treatment_respiratory_hits(patient_ids):
    hit_rows = []
    first_support = {}

    # treatment: high-value structured active treatment terms.
    if find_table("treatment") is not None:
        for ch in chunk_reader(
            "treatment",
            ["patientUnitStayID", "treatmentOffset", "treatmentString"],
        ):
            idc = resolve_col(ch.columns, "patientUnitStayID")
            oc = resolve_col(ch.columns, "treatmentOffset")
            tc = resolve_col(ch.columns, "treatmentString")
            if not all([idc, oc, tc]):
                continue

            ids = to_num(ch[idc])
            off = to_num(ch[oc])
            txt = ch[tc].astype("string").fillna("")

            m = (
                ids.notna()
                & ids.isin(patient_ids)
                & off.notna()
                & off.between(RESPIRATORY_T0_SEARCH_MIN, RESPIRATORY_T0_SEARCH_MAX)
                & txt.str.contains(RESP_SUPPORT_RE, na=False)
                & ~txt.str.contains(RESP_NEGATIVE_RE, na=False)
            ).fillna(False)
            if not bool(m.any()):
                continue

            xx = pd.DataFrame({
                "table": "treatment",
                "patientunitstayid": ids[m].astype("Int64"),
                "offset_min": off[m],
                "label": txt[m],
                "value": "",
            })

            for r in xx.itertuples(index=False):
                sid = int(r.patientunitstayid)
                val = float(r.offset_min)
                if sid not in first_support or val < first_support[sid]:
                    first_support[sid] = val

            if len(hit_rows) < MAX_SAMPLE_ROWS_PER_TABLE:
                remain = MAX_SAMPLE_ROWS_PER_TABLE - len(hit_rows)
                hit_rows.extend(xx.head(remain).to_dict("records"))

    # respiratoryCharting: identify explicit device/mode/value references.
    if find_table("respiratoryCharting") is not None:
        for ch in chunk_reader(
            "respiratoryCharting",
            [
                "patientUnitStayID",
                "respChartOffset",
                "respChartTypeCat",
                "respChartValueLabel",
                "respChartValue",
            ],
        ):
            idc = resolve_col(ch.columns, "patientUnitStayID")
            oc = resolve_col(ch.columns, "respChartOffset")
            lc = resolve_col(ch.columns, "respChartValueLabel")
            vc = resolve_col(ch.columns, "respChartValue")
            tc = resolve_col(ch.columns, "respChartTypeCat")
            if idc is None or oc is None:
                continue

            ids = to_num(ch[idc])
            off = to_num(ch[oc])
            label = ch[lc].astype("string").fillna("") if lc else ""
            value = ch[vc].astype("string").fillna("") if vc else ""
            typ = ch[tc].astype("string").fillna("") if tc else ""
            txt = (
                pd.Series(label, index=ch.index).astype("string")
                + " | "
                + pd.Series(value, index=ch.index).astype("string")
                + " | "
                + pd.Series(typ, index=ch.index).astype("string")
            )

            m = (
                ids.notna()
                & ids.isin(patient_ids)
                & off.notna()
                & off.between(RESPIRATORY_T0_SEARCH_MIN, RESPIRATORY_T0_SEARCH_MAX)
                & txt.str.contains(RESP_SUPPORT_RE, na=False)
                & ~txt.str.contains(RESP_NEGATIVE_RE, na=False)
            ).fillna(False)
            if not bool(m.any()):
                continue

            xx = pd.DataFrame({
                "table": "respiratoryCharting",
                "patientunitstayid": ids[m].astype("Int64"),
                "offset_min": off[m],
                "label": pd.Series(label, index=ch.index)[m].astype("string"),
                "value": pd.Series(value, index=ch.index)[m].astype("string"),
            })

            for r in xx.itertuples(index=False):
                sid = int(r.patientunitstayid)
                val = float(r.offset_min)
                if sid not in first_support or val < first_support[sid]:
                    first_support[sid] = val

            if len(hit_rows) < MAX_SAMPLE_ROWS_PER_TABLE:
                remain = MAX_SAMPLE_ROWS_PER_TABLE - len(hit_rows)
                hit_rows.extend(xx.head(remain).to_dict("records"))

    first_df = pd.DataFrame(
        [{"patientunitstayid": k, "respiratory_t0_candidate_min": v}
         for k, v in first_support.items()]
    )
    return first_df, pd.DataFrame(hit_rows)


def steroid_medication_audit(patient_ids):
    all_hits = []
    first_steroid = {}
    names = {}

    p = find_table("medication")
    if p is not None:
        for ch in chunk_reader(
            "medication",
            [
                "patientUnitStayID",
                "drugStartOffset",
                "drugOrderOffset",
                "drugStopOffset",
                "drugOrderCancelled",
                "drugName",
                "dosage",
                "routeAdmin",
                "frequency",
            ],
        ):
            idc = resolve_col(ch.columns, "patientUnitStayID")
            sc = resolve_col(ch.columns, "drugStartOffset")
            nc = resolve_col(ch.columns, "drugName")
            cc = resolve_col(ch.columns, "drugOrderCancelled")
            if idc is None or sc is None or nc is None:
                continue

            ids = to_num(ch[idc])
            start = to_num(ch[sc])
            name = ch[nc].astype("string").fillna("")
            cancelled = (
                ch[cc].astype("string").str.lower().eq("yes")
                if cc else pd.Series(False, index=ch.index)
            )

            m = (
                ids.notna()
                & ids.isin(patient_ids)
                & ~cancelled
                & start.notna()
                & name.str.contains(STEROID_RE, na=False)
            )
            if not bool(m.fillna(False).any()):
                continue

            mm = m.fillna(False)
            sid_vals = safe_int_values(ids[mm]).to_numpy()
            off_vals = pd.to_numeric(start[mm], errors="coerce").to_numpy(dtype=float)
            nm_vals = name[mm].astype("string").fillna("").astype(str).to_numpy()

            for sid, off, nm in zip(
                sid_vals,
                off_vals,
                nm_vals,
            ):
                if sid not in first_steroid or off < first_steroid[sid]:
                    first_steroid[sid] = off
                names[nm] = names.get(nm, 0) + 1

            if len(all_hits) < MAX_SAMPLE_ROWS_PER_TABLE:
                remain = MAX_SAMPLE_ROWS_PER_TABLE - len(all_hits)
                keepcols = [c for c in [
                    idc, sc, nc,
                    resolve_col(ch.columns, "dosage"),
                    resolve_col(ch.columns, "routeAdmin"),
                    resolve_col(ch.columns, "frequency"),
                ] if c is not None]
                tmp = ch.loc[mm, keepcols].head(remain).copy()
                tmp.insert(0, "source_table", "medication")
                all_hits.extend(tmp.to_dict("records"))

    # InfusionDrug can capture directly charted steroid infusions.
    p = find_table("infusionDrug")
    if p is not None:
        for ch in chunk_reader(
            "infusionDrug",
            [
                "patientUnitStayID",
                "infusionOffset",
                "drugName",
                "drugRate",
                "infusionRate",
            ],
        ):
            idc = resolve_col(ch.columns, "patientUnitStayID")
            oc = resolve_col(ch.columns, "infusionOffset")
            nc = resolve_col(ch.columns, "drugName")
            if idc is None or oc is None or nc is None:
                continue

            ids = to_num(ch[idc])
            off = to_num(ch[oc])
            name = ch[nc].astype("string").fillna("")
            m = (
                ids.notna()
                & ids.isin(patient_ids)
                & off.notna()
                & name.str.contains(STEROID_RE, na=False)
            )
            if not bool(m.fillna(False).any()):
                continue

            mm = m.fillna(False)
            sid_vals = safe_int_values(ids[mm]).to_numpy()
            off_vals = pd.to_numeric(off[mm], errors="coerce").to_numpy(dtype=float)
            nm_vals = name[mm].astype("string").fillna("").astype(str).to_numpy()

            for sid, of, nm in zip(
                sid_vals,
                off_vals,
                nm_vals,
            ):
                if sid not in first_steroid or of < first_steroid[sid]:
                    first_steroid[sid] = of
                names[nm] = names.get(nm, 0) + 1

            if len(all_hits) < MAX_SAMPLE_ROWS_PER_TABLE:
                remain = MAX_SAMPLE_ROWS_PER_TABLE - len(all_hits)
                keepcols = [c for c in [
                    idc, oc, nc,
                    resolve_col(ch.columns, "drugRate"),
                    resolve_col(ch.columns, "infusionRate"),
                ] if c is not None]
                tmp = ch.loc[m, keepcols].head(remain).copy()
                tmp.insert(0, "source_table", "infusionDrug")
                all_hits.extend(tmp.to_dict("records"))

    first_df = pd.DataFrame(
        [{"patientunitstayid": k, "first_systemic_steroid_offset_min": v}
         for k, v in first_steroid.items()]
    )
    name_df = pd.DataFrame(
        [{"drug_name_raw": k, "rows_n": v} for k, v in names.items()]
    )
    if len(name_df):
        name_df = name_df.sort_values("rows_n", ascending=False)

    return first_df, pd.DataFrame(all_hits), name_df


def hospital_interface_coverage(patient, respiratory_t0, steroid):
    """
    Hospital-level interface coverage audit.
    Lack of a table interface must not be interpreted as absence of care.
    """
    x = patient[["patientunitstayid", "hospitalid"]].copy()
    resp_ids = set(
        safe_int_values(
            respiratory_t0.get(
                "patientunitstayid",
                pd.Series(dtype="Int64"),
            )
        ).tolist()
    )
    steroid_ids = set(
        safe_int_values(
            steroid.get(
                "patientunitstayid",
                pd.Series(dtype="Int64"),
            )
        ).tolist()
    )

    x["has_resp_support_hit_0_24h"] = (
        x["patientunitstayid"].isin(resp_ids).fillna(False).astype("int8")
    )
    x["has_systemic_steroid_record_anytime"] = (
        x["patientunitstayid"].isin(steroid_ids).fillna(False).astype("int8")
    )

    rows = []
    for hid, g in x.groupby("hospitalid", dropna=False):
        rows.append({
            "hospitalid": hid,
            "unit_stays_n": len(g),
            "resp_support_hit_pct": 100 * g["has_resp_support_hit_0_24h"].mean(),
            "steroid_record_pct": 100 * g["has_systemic_steroid_record_anytime"].mean(),
        })
    return pd.DataFrame(rows).sort_values("unit_stays_n", ascending=False)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 110)
    print("E1.2 — eICU frozen external-validation feasibility + schema audit (all-NA-safe fix)")
    print("=" * 110)
    print(f"Raw eICU folder: {DATA_DIR}")
    print("[E1.2] Full nullable compatibility enabled for age/ID/offset conversions.")
    print(
        "No treatment-effect estimate will be produced. "
        "This run is outcome-blind with respect to steroid effect."
    )

    schema = schema_audit()
    save(schema, "E1_00_SCHEMA_AUDIT.csv")

    patient = load_patient_base()
    adult = patient[patient["adult"] == 1].copy()
    adult_ids = set(safe_int_values(adult["patientunitstayid"]).tolist())

    # Pneumonia from patient.apacheAdmissionDx.
    apdx_txt = adult["apache_admission_dx"].astype("string").fillna("")
    apdx_pneu = apdx_txt.str.contains(PNEUMONIA_RE, na=False)
    apdx_excl = (
        apdx_txt.str.contains(ASPIRATION_RE, na=False)
        | apdx_txt.str.contains(PJP_RE, na=False)
        | apdx_txt.str.contains(TB_RE, na=False)
        | apdx_txt.str.contains(INFLUENZA_RE, na=False)
    )
    apdx_rf = apdx_txt.str.contains(RESP_FAILURE_RE, na=False)

    dx_pneu, dx_excl, dx_rf, dx_samples = diagnosis_audit(adult_ids)

    pneumonia_ids = set(
        safe_int_values(adult.loc[apdx_pneu, "patientunitstayid"])
    ) | dx_pneu

    exclusion_ids = set(
        safe_int_values(adult.loc[apdx_excl, "patientunitstayid"])
    ) | dx_excl

    resp_failure_dx_ids = set(
        safe_int_values(adult.loc[apdx_rf, "patientunitstayid"])
    ) | dx_rf

    cap_like_ids = pneumonia_ids - exclusion_ids

    # Respiratory support candidate t0 within first 24h ICU.
    resp_t0, resp_samples = treatment_respiratory_hits(adult_ids)
    resp_support_ids = set(
        safe_int_values(
            resp_t0.get(
                "patientunitstayid",
                pd.Series(dtype="Int64"),
            )
        ).tolist()
    )

    severe_resp_ids = resp_support_ids | resp_failure_dx_ids

    # Candidate external severe CAP phenotype.
    candidate_ids = cap_like_ids & severe_resp_ids

    steroid_first, steroid_samples, steroid_names = steroid_medication_audit(adult_ids)

    steroid_map = safe_id_value_map(
        steroid_first,
        "patientunitstayid",
        "first_systemic_steroid_offset_min",
    )

    resp_map = safe_id_value_map(
        resp_t0,
        "patientunitstayid",
        "respiratory_t0_candidate_min",
    )

    # Frozen 6h exposure FEASIBILITY ONLY, relative to candidate respiratory t0.
    # This is NOT linked to mortality in E1.
    n_candidate_with_t0 = 0
    n_steroid_0_6h = 0
    n_recent_3_6h = 0
    n_early_0_3h = 0
    exposure_rows = []

    for sid in sorted(candidate_ids):
        if sid not in resp_map:
            continue
        n_candidate_with_t0 += 1
        t0 = resp_map[sid]
        st = steroid_map.get(sid, np.nan)
        rel = st - t0 if np.isfinite(st) else np.nan
        if np.isfinite(rel) and 0 <= rel < 6 * 60:
            n_steroid_0_6h += 1
            if rel < 3 * 60:
                n_early_0_3h += 1
            else:
                n_recent_3_6h += 1
        exposure_rows.append({
            "patientunitstayid": sid,
            "respiratory_t0_candidate_min": t0,
            "first_systemic_steroid_offset_min": st,
            "steroid_relative_to_t0_min": rel,
            "steroid_within6h": int(np.isfinite(rel) and 0 <= rel < 360),
            "steroid_early_0_3h": int(np.isfinite(rel) and 0 <= rel < 180),
            "steroid_recent_3_6h": int(np.isfinite(rel) and 180 <= rel < 360),
        })

    exposure_feas = pd.DataFrame(exposure_rows)

    # Outcome availability ONLY — no treatment stratification.
    hstat = adult["hospital_discharge_status"].astype("string").str.lower()
    dead = hstat.str.contains("expired|dead|death", regex=True, na=False)

    outcome_audit = pd.DataFrame([{
        "adult_unit_stays_n": len(adult),
        "hospital_discharge_status_nonmissing_n": int(
            adult["hospital_discharge_status"].notna().sum()
        ),
        "hospital_discharge_offset_nonmissing_n": int(
            pd.Series(adult["hospital_discharge_offset_min"]).notna().sum()
        ),
        "hospital_death_status_rows_n": int(dead.sum()),
        "unit_discharge_status_nonmissing_n": int(
            adult["unit_discharge_status"].notna().sum()
        ),
        "note": (
            "Availability audit only. No steroid-outcome comparison was performed. "
            "eICU hospitalDischargeOffset is relative to ICU admission and may permit "
            "28-day in-hospital death mapping, but post-discharge 28-day mortality "
            "is not assumed available."
        ),
    }])

    # Core counts.
    summary = pd.DataFrame([
        {"metric": "all_unit_stays", "n": len(patient)},
        {"metric": "adult_unit_stays", "n": len(adult)},
        {"metric": "adult_pneumonia_any_admission_or_first24h_dx", "n": len(pneumonia_ids)},
        {"metric": "adult_high_specificity_CAP_like_after_exclusions", "n": len(cap_like_ids)},
        {"metric": "adult_resp_support_hit_first24h", "n": len(resp_support_ids)},
        {"metric": "adult_early_resp_failure_dx", "n": len(resp_failure_dx_ids)},
        {"metric": "adult_severe_resp_any_support_or_resp_failure_dx", "n": len(severe_resp_ids)},
        {"metric": "candidate_severe_CAP_like_intersection", "n": len(candidate_ids)},
        {"metric": "candidate_with_explicit_resp_t0_support_marker", "n": n_candidate_with_t0},
        {"metric": "candidate_first_steroid_within6h_of_resp_t0", "n": n_steroid_0_6h},
        {"metric": "candidate_steroid_early_0_3h", "n": n_early_0_3h},
        {"metric": "candidate_steroid_recent_3_6h", "n": n_recent_3_6h},
    ])

    # Simple proceed gate — operational, not treatment-effect based.
    required = {
        "patient": find_table("patient") is not None,
        "medication": find_table("medication") is not None,
        "treatment_or_respchart": (
            find_table("treatment") is not None
            or find_table("respiratoryCharting") is not None
        ),
        "pneumonia_source": (
            find_table("admissionDx") is not None
            or find_table("diagnosis") is not None
        ),
    }

    mapping_gaps = []
    if not required["patient"]:
        mapping_gaps.append("patient table missing")
    if not required["medication"]:
        mapping_gaps.append("medication table missing")
    if not required["treatment_or_respchart"]:
        mapping_gaps.append("no treatment/respiratoryCharting source")
    if not required["pneumonia_source"]:
        mapping_gaps.append("no admissionDx/diagnosis source")
    if len(candidate_ids) < 200:
        mapping_gaps.append(
            "candidate severe CAP-like cohort <200; phenotype/support may be too sparse"
        )
    if n_steroid_0_6h < 50:
        mapping_gaps.append(
            "fewer than 50 candidate 6h steroid initiations; external causal replication may be under-supported"
        )
    if n_recent_3_6h < 20:
        mapping_gaps.append(
            "fewer than 20 recent 3–6h steroid initiations; grace-end IPCW likely under-supported"
        )

    proceed = len(mapping_gaps) == 0

    decision = pd.DataFrame([{
        "external_validation_stage": "E1",
        "mimic_protocol_frozen": 1,
        "window_h": FROZEN_WINDOW_H,
        "recent_window": "3-6h",
        "candidate_severe_CAP_like_n": len(candidate_ids),
        "candidate_with_explicit_resp_t0_n": n_candidate_with_t0,
        "steroid_initiation_0_6h_n": n_steroid_0_6h,
        "recent_steroid_initiation_3_6h_n": n_recent_3_6h,
        "proceed_to_E2_mapping_and_locked_analysis": int(proceed),
        "decision": (
            "PROCEED_TO_E2_LOCKED_EXTERNAL_VALIDATION_MAPPING"
            if proceed
            else "HOLD_EXTERNAL_CAUSAL_EFFECT_RESOLVE_MAPPING_SUPPORT"
        ),
        "mapping_or_support_gaps": " | ".join(mapping_gaps),
        "scientific_note": (
            "E1 did not estimate a steroid-mortality effect. "
            "Any eICU-specific phenotype adjustments required after E1 must be "
            "documented before opening the external treatment effect."
        ),
    }])

    coverage = hospital_interface_coverage(
        adult,
        resp_t0,
        steroid_first,
    )

    # Save outputs.
    save(summary, "E1_01_FEASIBILITY_COUNTS.csv")
    save(dx_samples, "E1_02_PNEUMONIA_DIAGNOSIS_SAMPLES.csv")
    save(resp_t0, "E1_03_RESPIRATORY_T0_CANDIDATES.csv")
    save(resp_samples, "E1_04_RESPIRATORY_SUPPORT_SAMPLES.csv")
    save(steroid_first, "E1_05_STEROID_FIRST_OFFSET.csv")
    save(steroid_samples, "E1_06_STEROID_RECORD_SAMPLES.csv")
    save(steroid_names, "E1_07_STEROID_RAW_NAME_COUNTS.csv")
    save(exposure_feas, "E1_08_FROZEN_6H_EXPOSURE_FEASIBILITY.csv")
    save(outcome_audit, "E1_09_OUTCOME_FIELD_AVAILABILITY.csv")
    save(coverage, "E1_10_HOSPITAL_INTERFACE_COVERAGE.csv")
    save(decision, "E1_11_RECOMMENDATION.csv")

    # A compact JSON with exact file/column mapping for the next locked stage.
    mapping = {
        "data_dir": str(DATA_DIR),
        "protocol": {
            "window_h": 6,
            "recent_window_h": [3, 6],
            "effect_opened_in_E1": False,
        },
        "tables": {
            r["table"]: {
                "found": bool(r["found"]),
                "filename": r["filename"],
                "columns": r["columns"],
            }
            for _, r in schema.iterrows()
        },
        "decision": decision.iloc[0].to_dict(),
    }
    with open(
        OUTPUT_DIR / "E1_12_SCHEMA_AND_PROTOCOL_MAP.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 110)
    print("E1 COMPLETE — NO TREATMENT EFFECT WAS OPENED")
    print("=" * 110)
    print(summary.to_string(index=False))
    print("\nRecommendation:")
    print(decision.to_string(index=False))

    print("\nGenerated outputs are available in the outputs_eicu_E1 folder:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FAILED] Full traceback:\n")
        traceback.print_exc()
        raise

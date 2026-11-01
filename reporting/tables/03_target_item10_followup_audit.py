from __future__ import annotations

"""
TARGET item 10 — frozen clone follow-up / artificial-censoring audit

PURPOSE
-------
Generate exact, strategy-specific counts describing why cloned person-trials
stop contributing follow-up in the frozen MIMIC-IV and eICU analyses.

THIS SCRIPT DOES NOT:
    - fit or refit propensity/censoring models
    - change the 6-hour treatment window
    - change the recent [3,6) h window
    - change the cohort
    - change the estimand
    - estimate a new treatment effect
    - bootstrap anything

It reconstructs the exact event/artificial-censoring geometry used by the
already-frozen estimators.

Tie rules are copied from the frozen analysis code:
    - MIMIC: death wins a tie with artificial censoring.
    - eICU: a clinical event (death or alive hospital discharge) wins a tie
      with artificial censoring.

Outputs include a candidate Supplementary Table S11 and manuscript text
for TARGET item 10.
"""

from pathlib import Path
import os
import json
import zipfile
import hashlib
import numpy as np
import pandas as pd

# =============================================================================
# USER PATH
# =============================================================================
PROJECT_ROOT = Path(os.environ.get("SCAP_PROJECT_ROOT", Path.cwd()))

OUTDIR = PROJECT_ROOT / "TARGET10_followup_audit"
CACHE = OUTDIR / "_materialized_sources"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

FOLLOWUP_H = 28 * 24.0
WINDOW_H = 6.0
TRIAL_STARTS = list(range(0, 20, 2))

EXPECTED_MIMIC_STAYS = 2014
EXPECTED_MIMIC_TRIALS = 18487
EXPECTED_MIMIC_INIT6 = 744

EXPECTED_EICU_STAYS = 5394
EXPECTED_EICU_CLUSTERS = 4794
EXPECTED_EICU_TRIALS = 47923
EXPECTED_EICU_INIT6 = 1341

DEATH_TIME_CANDIDATES = [
    "death_hours_full",
    "death_hours_from_t0",
    "death_hours_from_respiratory_t0",
    "death_rel_h",
    "death_hours",
    "hours_to_death",
    "death_time_hours",
    "death_h",
]


# =============================================================================
# Source discovery / materialization
# =============================================================================
def _extract_member(zp: Path, member: str) -> Path:
    token = hashlib.md5((str(zp) + "::" + member).encode("utf-8")).hexdigest()[:10]
    out = CACHE / f"{token}_{Path(member).name}"
    if not out.exists():
        with zipfile.ZipFile(zp, "r") as z, z.open(member) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out


def find_all_exact(basename: str) -> list[Path]:
    hits = sorted(
        PROJECT_ROOT.rglob(basename),
        key=lambda p: (len(p.parts), str(p).lower()),
    )

    # Also search ZIP members because several frozen stages were delivered as ZIPs.
    for zp in sorted(PROJECT_ROOT.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zp, "r") as z:
                members = [m for m in z.namelist() if Path(m).name == basename]
            for member in members:
                hits.append(_extract_member(zp, member))
        except zipfile.BadZipFile:
            continue

    # Deduplicate resolved paths.
    uniq = []
    seen = set()
    for p in hits:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def find_one(basename: str) -> Path:
    hits = find_all_exact(basename)
    if not hits:
        raise FileNotFoundError(
            f"Could not find {basename} below {PROJECT_ROOT}, including ZIP files."
        )
    print(f"[SOURCE] {basename} <- {hits[0]}")
    return hits[0]


def read_csv_optional(basename: str):
    hits = find_all_exact(basename)
    if not hits:
        return None
    print(f"[SOURCE] {basename} <- {hits[0]}")
    return pd.read_csv(hits[0], encoding="utf-8-sig")


# =============================================================================
# MIMIC frozen cohort + exact frozen final 6-hour analysis trial reconstruction
# =============================================================================
def normalize_mimic_death(panel: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve the frozen MIMIC death-time variable.

    In the actual Stage 7 hourly panel used by this project, the field is
    `death_hours`. Earlier audit code incorrectly checked only
    `death_hours_full` in the panel before falling back to the Stage 5.5
    cohort. This function now searches the Stage 7 panel first using all
    known candidate names.

    The selected field is normalized internally to `death_hours_full`
    without changing its values.
    """
    # 1) Preferred source: Stage 7 panel, because this is the exact dynamic
    # panel used to construct the sequential trials.
    panel_source = None

    if "death_hours_full" in panel.columns:
        panel_source = "death_hours_full"
    else:
        for c in DEATH_TIME_CANDIDATES:
            if c in panel.columns:
                panel_source = c
                break

    if panel_source is not None:
        print(f"[MIMIC] death-time source = Stage7 panel column `{panel_source}`")
        out = panel.copy()
        vals = pd.to_numeric(out[panel_source], errors="coerce")

        if panel_source != "death_hours_full":
            # Avoid carrying two competing death-time fields internally.
            out["death_hours_full"] = vals
        else:
            out["death_hours_full"] = vals

        return out

    # 2) Fallback: patient-level Stage 5.5 cohort, only if the Stage 7 panel
    # genuinely lacks a death-time field.
    cohort_source = None
    for c in DEATH_TIME_CANDIDATES:
        if c in cohort.columns:
            cohort_source = c
            break

    if cohort_source is None:
        suffix_hits = [
            c for c in cohort.columns
            if c.startswith("death_hours_full")
        ]
        if suffix_hits:
            cohort_source = suffix_hits[0]

    if cohort_source is None:
        panel_deathish = [
            c for c in panel.columns
            if any(k in c.lower() for k in ["death", "mort", "dod", "expire"])
        ]
        cohort_deathish = [
            c for c in cohort.columns
            if any(k in c.lower() for k in ["death", "mort", "dod", "expire"])
        ]
        raise RuntimeError(
            "Could not identify a MIMIC death-time column.\n"
            f"Death-like Stage7 columns: {panel_deathish}\n"
            f"Death-like Stage5.5 columns: {cohort_deathish}\n"
            "STOPPED rather than guessing a time variable."
        )

    print(f"[MIMIC] death-time source = Stage5.5 cohort column `{cohort_source}`")
    death = cohort[["stay_id", cohort_source]].copy()
    death[cohort_source] = pd.to_numeric(death[cohort_source], errors="coerce")
    death = death.rename(columns={cohort_source: "death_hours_full"})
    death = death.drop_duplicates("stay_id")

    return panel.merge(
        death,
        on="stay_id",
        how="left",
        validate="many_to_one",
    )


def build_mimic_trials() -> pd.DataFrame:
    cohort = pd.read_parquet(find_one("stage5_5_analysis_cohort.parquet"))
    panel = pd.read_parquet(find_one("stage7_hourly_panel.parquet"))

    print(
        "[MIMIC] Stage7 death-like columns:",
        [c for c in panel.columns if any(k in c.lower() for k in ["death", "mort", "dod", "expire"])]
    )

    needed = {
        "stay_id",
        "eligible_primary",
        "major_immunocompromise_proxy",
    }
    missing = needed - set(cohort.columns)
    if missing:
        raise RuntimeError(f"MIMIC cohort missing required columns: {sorted(missing)}")

    final_mask = (
        pd.to_numeric(cohort["eligible_primary"], errors="coerce")
        .fillna(0).eq(1)
        &
        pd.to_numeric(cohort["major_immunocompromise_proxy"], errors="coerce")
        .fillna(0).eq(0)
    )
    frozen = cohort.loc[final_mask].copy()
    if len(frozen) != EXPECTED_MIMIC_STAYS:
        raise RuntimeError(
            f"MIMIC frozen cohort={len(frozen)}, expected {EXPECTED_MIMIC_STAYS}. STOP."
        )
    if frozen["stay_id"].duplicated().any():
        raise RuntimeError("MIMIC frozen cohort has duplicated stay_id.")

    final_ids = set(frozen["stay_id"])
    panel = panel[panel["stay_id"].isin(final_ids)].copy()
    panel = normalize_mimic_death(panel, frozen)

    required_panel = {"stay_id", "hour", "any_init_h", "death_hours_full"}
    missing = required_panel - set(panel.columns)
    if missing:
        raise RuntimeError(f"MIMIC Stage7 panel missing: {sorted(missing)}")

    if panel[["stay_id", "hour"]].duplicated().any():
        raise RuntimeError("MIMIC Stage7 panel has duplicate stay_id/hour rows.")

    parts = []
    for h in TRIAL_STARTS:
        x = panel.loc[pd.to_numeric(panel["hour"], errors="coerce").eq(h)].copy()
        if x.empty:
            continue

        init = pd.to_numeric(x["any_init_h"], errors="coerce")
        death = pd.to_numeric(x["death_hours_full"], errors="coerce")

        # Exact frozen 6-hour feasibility analysis/frozen final 6-hour analysis eligibility:
        # alive and not previously initiated at the trial baseline.
        eligible = (
            (death.isna() | (death > h))
            & (init.isna() | (init >= h))
        )
        x = x.loc[eligible, ["stay_id", "any_init_h", "death_hours_full"]].copy()

        init = pd.to_numeric(x["any_init_h"], errors="coerce")
        death = pd.to_numeric(x["death_hours_full"], errors="coerce")

        x["trial_hour"] = float(h)
        x["actual_initiate_next6h"] = (
            init.notna()
            & (init >= h)
            & (init < h + WINDOW_H)
            & (death.isna() | (init <= death))
        ).astype(int)

        parts.append(x)

    trials = pd.concat(parts, ignore_index=True)
    trials["trial_idx"] = np.arange(len(trials), dtype=np.int64)

    if trials["stay_id"].nunique() != EXPECTED_MIMIC_STAYS:
        raise RuntimeError(
            f"MIMIC reconstructed unique stays={trials['stay_id'].nunique()}, "
            f"expected {EXPECTED_MIMIC_STAYS}."
        )
    if len(trials) != EXPECTED_MIMIC_TRIALS:
        raise RuntimeError(
            f"MIMIC reconstructed person-trials={len(trials)}, "
            f"expected {EXPECTED_MIMIC_TRIALS}. STOP."
        )
    if int(trials["actual_initiate_next6h"].sum()) != EXPECTED_MIMIC_INIT6:
        raise RuntimeError(
            f"MIMIC reconstructed initiate<=6h={int(trials['actual_initiate_next6h'].sum())}, "
            f"expected {EXPECTED_MIMIC_INIT6}. STOP."
        )

    # Cross-check against the official frozen final 6-hour analysis effect CSV if available.
    eff = read_csv_optional("final_effect_estimates.csv")
    if eff is not None:
        r = eff.iloc[0]
        checks = {
            "unique_patients_n": EXPECTED_MIMIC_STAYS,
            "person_trials_n": EXPECTED_MIMIC_TRIALS,
            "initiate_within6h_person_trials_n": EXPECTED_MIMIC_INIT6,
        }
        for c, expected in checks.items():
            if c in r.index and int(r[c]) != expected:
                raise RuntimeError(
                    f"Official frozen final 6-hour analysis {c}={r[c]} but expected {expected}. STOP."
                )

    print(
        f"[PASS] MIMIC exact frozen trials: {trials['stay_id'].nunique()} stays; "
        f"{len(trials)} person-trials; initiate<=6h={int(trials['actual_initiate_next6h'].sum())}"
    )
    return trials


def mimic_geometry(trials: pd.DataFrame):
    h0 = trials["trial_hour"].to_numpy(float)
    init_abs = pd.to_numeric(trials["any_init_h"], errors="coerce").to_numpy(float)
    death_abs = pd.to_numeric(trials["death_hours_full"], errors="coerce").to_numpy(float)
    y6 = trials["actual_initiate_next6h"].astype(int).to_numpy()

    init_rel = init_abs - h0
    death_rel = death_abs - h0

    death_end = np.where(np.isfinite(death_rel), death_rel, np.inf)
    horizon = np.full(len(trials), FOLLOWUP_H, float)

    # Frozen frozen final 6-hour analysis DEFER geometry.
    art0 = np.where(y6 == 1, init_rel, np.inf)
    end0 = np.minimum.reduce([horizon, death_end, art0])
    event0 = (
        np.isfinite(death_rel)
        & (death_rel > 0)
        & (death_rel <= FOLLOWUP_H)
        & (death_rel <= art0)  # death wins ties
    )
    end0 = np.where(event0, death_rel, end0)

    # Frozen frozen final 6-hour analysis INITIATE geometry.
    art1 = np.where(y6 == 1, np.inf, WINDOW_H)
    end1 = np.minimum.reduce([horizon, death_end, art1])
    event1 = (
        np.isfinite(death_rel)
        & (death_rel > 0)
        & (death_rel <= FOLLOWUP_H)
        & (death_rel <= art1)  # death wins ties
    )
    end1 = np.where(event1, death_rel, end1)

    return {
        "y6": y6,
        "init_rel": init_rel,
        "death_rel": death_rel,
        "art0": art0,
        "art1": art1,
        "end0": end0,
        "event0": event0,
        "end1": end1,
        "event1": event1,
    }


# =============================================================================
# eICU exact frozen E3 geometry
# =============================================================================
def choose_eicu_trials() -> pd.DataFrame:
    candidates = find_all_exact("e3_trials.parquet")
    if not candidates:
        raise FileNotFoundError("Could not find e3_trials.parquet.")

    accepted = []
    for p in candidates:
        try:
            x = pd.read_parquet(p)
        except Exception:
            continue

        if "patientunitstayid" not in x.columns:
            continue

        nstay = x["patientunitstayid"].nunique()
        ntrial = len(x)
        ninit = (
            int(pd.to_numeric(x["actual_initiate_next6h"], errors="coerce").fillna(0).sum())
            if "actual_initiate_next6h" in x.columns else -1
        )
        ncluster = x["cluster_id"].nunique() if "cluster_id" in x.columns else -1

        if (
            nstay == EXPECTED_EICU_STAYS
            and ntrial == EXPECTED_EICU_TRIALS
            and ninit == EXPECTED_EICU_INIT6
            and ncluster == EXPECTED_EICU_CLUSTERS
        ):
            accepted.append((p, x))

    if not accepted:
        raise RuntimeError(
            "No e3_trials.parquet matched the frozen E3 counts "
            f"({EXPECTED_EICU_STAYS} stays, {EXPECTED_EICU_CLUSTERS} clusters, "
            f"{EXPECTED_EICU_TRIALS} person-trials, {EXPECTED_EICU_INIT6} initiations)."
        )

    p, trials = accepted[0]
    print(f"[PASS] eICU frozen E3 trials <- {p}")
    print(
        f"       {trials['patientunitstayid'].nunique()} stays; "
        f"{trials['cluster_id'].nunique()} patient clusters; "
        f"{len(trials)} person-trials; "
        f"initiate<=6h={int(trials['actual_initiate_next6h'].sum())}"
    )

    eff = read_csv_optional("external_effect_estimates.csv")
    if eff is not None:
        r = eff.iloc[0]
        checks = {
            "locked_unit_stays_n": EXPECTED_EICU_STAYS,
            "patient_clusters_n": EXPECTED_EICU_CLUSTERS,
            "person_trials_n": EXPECTED_EICU_TRIALS,
            "initiate_within6h_person_trials_n": EXPECTED_EICU_INIT6,
        }
        for c, expected in checks.items():
            if c in r.index and int(r[c]) != expected:
                raise RuntimeError(
                    f"Official E3 {c}={r[c]} but expected {expected}. STOP."
                )

    return trials.reset_index(drop=True).copy()


def eicu_geometry(trials: pd.DataFrame):
    required = {
        "patientunitstayid",
        "cluster_id",
        "trial_hour",
        "respiratory_t0_candidate_min",
        "hospital_discharge_offset_min",
        "hospital_death",
        "steroid_relative_to_t0_min",
        "actual_initiate_next6h",
    }
    missing = required - set(trials.columns)
    if missing:
        raise RuntimeError(f"eICU frozen trials missing: {sorted(missing)}")

    baseline_min = (
        pd.to_numeric(trials["respiratory_t0_candidate_min"], errors="coerce").to_numpy(float)
        + pd.to_numeric(trials["trial_hour"], errors="coerce").to_numpy(float) * 60
    )

    hosp_end_abs = pd.to_numeric(
        trials["hospital_discharge_offset_min"], errors="coerce"
    ).to_numpy(float)
    hosp_rel_h = (hosp_end_abs - baseline_min) / 60.0

    death = trials["hospital_death"].astype(int).to_numpy() == 1
    cause = np.where(death, 1, 2)  # 1 death; 2 alive hospital discharge

    valid_event = (
        np.isfinite(hosp_rel_h)
        & (hosp_rel_h > 0)
        & (hosp_rel_h <= FOLLOWUP_H)
    )
    event_time = np.where(valid_event, hosp_rel_h, np.inf)
    event_cause = np.where(valid_event, cause, 0)

    init_rel_h = (
        pd.to_numeric(trials["steroid_relative_to_t0_min"], errors="coerce").to_numpy(float)
        / 60.0
        - pd.to_numeric(trials["trial_hour"], errors="coerce").to_numpy(float)
    )
    y6 = trials["actual_initiate_next6h"].astype(int).to_numpy()

    horizon = np.full(len(trials), FOLLOWUP_H, float)
    art0 = np.where(y6 == 1, init_rel_h, np.inf)
    art1 = np.where(y6 == 1, np.inf, WINDOW_H)

    def arm_geom(art):
        end = np.minimum.reduce([horizon, event_time, art])
        clinical = (
            (event_cause > 0)
            & (event_time <= horizon)
            & (event_time <= art)  # clinical event wins ties
        )
        etype = np.where(clinical, event_cause, 0)
        end = np.where(clinical, event_time, end)
        return end, etype

    end0, type0 = arm_geom(art0)
    end1, type1 = arm_geom(art1)

    return {
        "y6": y6,
        "hosp_rel_h": hosp_rel_h,
        "event_time": event_time,
        "event_cause": event_cause,
        "init_rel_h": init_rel_h,
        "art0": art0,
        "art1": art1,
        "end0": end0,
        "type0": type0,
        "end1": end1,
        "type1": type1,
    }


# =============================================================================
# Reason classification
# =============================================================================
def summarize_mimic(trials: pd.DataFrame, g: dict):
    rows = []
    detail = []

    for strategy, event, end, art, artificial_reason in [
        (
            "Defer systemic corticosteroid for at least 6 h",
            g["event0"], g["end0"], g["art0"],
            "Artificial censoring at corticosteroid initiation during [0,6) h",
        ),
        (
            "Initiate systemic corticosteroid within 6 h",
            g["event1"], g["end1"], g["art1"],
            "Artificial censoring at 6 h because corticosteroid was not initiated",
        ),
    ]:
        if strategy.startswith("Defer"):
            art_mask = (~event) & (g["y6"] == 1)
        else:
            art_mask = (~event) & (g["y6"] == 0)

        death_mask = event
        admin_mask = ~(art_mask | death_mask)

        masks = [
            (artificial_reason, art_mask),
            ("Death within 28 d before strategy deviation", death_mask),
            ("Administrative end at day 28 without prior death/deviation", admin_mask),
        ]

        for reason, mask in masks:
            rows.append({
                "Cohort": "MIMIC-IV",
                "Strategy": strategy,
                "End-of-follow-up reason": reason,
                "Person-trial clones, n": int(mask.sum()),
                "Percent of strategy clones": round(100 * mask.mean(), 2),
                "Unique ICU stays contributing, n": int(
                    trials.loc[mask, "stay_id"].nunique()
                ),
                "Unique patient clusters, n": np.nan,
            })

        if int((art_mask | death_mask | admin_mask).sum()) != len(trials):
            raise RuntimeError("MIMIC reason partition failed.")

        d = trials[["stay_id", "trial_hour"]].copy()
        d["Cohort"] = "MIMIC-IV"
        d["Strategy"] = strategy
        d["Reason"] = np.select(
            [art_mask, death_mask],
            [artificial_reason, "Death within 28 d before strategy deviation"],
            default="Administrative end at day 28 without prior death/deviation",
        )
        detail.append(d)

    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True)


def summarize_eicu(trials: pd.DataFrame, g: dict):
    rows = []
    detail = []

    for strategy, etype, art_reason in [
        (
            "Defer systemic corticosteroid for at least 6 h",
            g["type0"],
            "Artificial censoring at corticosteroid initiation during [0,6) h",
        ),
        (
            "Initiate systemic corticosteroid within 6 h",
            g["type1"],
            "Artificial censoring at 6 h because corticosteroid was not initiated",
        ),
    ]:
        if strategy.startswith("Defer"):
            art_mask = (etype == 0) & (g["y6"] == 1)
        else:
            art_mask = (etype == 0) & (g["y6"] == 0)

        death_mask = etype == 1
        discharge_mask = etype == 2
        admin_mask = ~(art_mask | death_mask | discharge_mask)

        masks = [
            (art_reason, art_mask),
            ("Documented in-hospital death within 28 d", death_mask),
            ("Alive hospital discharge within 28 d (competing event)", discharge_mask),
            ("Administrative end at day 28 without prior clinical event/deviation", admin_mask),
        ]

        for reason, mask in masks:
            rows.append({
                "Cohort": "eICU",
                "Strategy": strategy,
                "End-of-follow-up reason": reason,
                "Person-trial clones, n": int(mask.sum()),
                "Percent of strategy clones": round(100 * mask.mean(), 2),
                "Unique ICU stays contributing, n": int(
                    trials.loc[mask, "patientunitstayid"].nunique()
                ),
                "Unique patient clusters, n": int(
                    trials.loc[mask, "cluster_id"].nunique()
                ),
            })

        if int((art_mask | death_mask | discharge_mask | admin_mask).sum()) != len(trials):
            raise RuntimeError("eICU reason partition failed.")

        d = trials[
            ["patientunitstayid", "cluster_id", "trial_hour"]
        ].copy()
        d["Cohort"] = "eICU"
        d["Strategy"] = strategy
        d["Reason"] = np.select(
            [art_mask, death_mask, discharge_mask],
            [
                art_reason,
                "Documented in-hospital death within 28 d",
                "Alive hospital discharge within 28 d (competing event)",
            ],
            default="Administrative end at day 28 without prior clinical event/deviation",
        )
        detail.append(d)

    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True)


def artificial_by_hour_mimic(trials, g):
    mask = (~g["event0"]) & (g["y6"] == 1)
    rel = g["init_rel"][mask]
    bins = [0, 1, 2, 3, 4, 5, 6]
    labels = ["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5)", "[5,6)"]
    cat = pd.cut(rel, bins=bins, labels=labels, right=False, include_lowest=True)
    out = (
        pd.Series(cat)
        .value_counts(sort=False)
        .rename_axis("Relative initiation interval, h")
        .reset_index(name="Artificially censored defer clones, n")
    )
    out["Cohort"] = "MIMIC-IV"
    return out[["Cohort", "Relative initiation interval, h", "Artificially censored defer clones, n"]]


def artificial_by_hour_eicu(trials, g):
    mask = (g["type0"] == 0) & (g["y6"] == 1)
    rel = g["init_rel_h"][mask]
    bins = [0, 1, 2, 3, 4, 5, 6]
    labels = ["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5)", "[5,6)"]
    cat = pd.cut(rel, bins=bins, labels=labels, right=False, include_lowest=True)
    out = (
        pd.Series(cat)
        .value_counts(sort=False)
        .rename_axis("Relative initiation interval, h")
        .reset_index(name="Artificially censored defer clones, n")
    )
    out["Cohort"] = "eICU"
    return out[["Cohort", "Relative initiation interval, h", "Artificially censored defer clones, n"]]


def by_trial_baseline(detail: pd.DataFrame):
    return (
        detail
        .groupby(["Cohort", "Strategy", "trial_hour", "Reason"], dropna=False)
        .size()
        .reset_index(name="Person-trial clones, n")
        .rename(columns={"trial_hour": "Trial baseline, h"})
        .sort_values(["Cohort", "Strategy", "Trial baseline, h", "Reason"])
    )


# =============================================================================
# Manuscript text
# =============================================================================
def lookup(summary, cohort, strategy_prefix, reason_prefix):
    x = summary[
        summary["Cohort"].eq(cohort)
        & summary["Strategy"].str.startswith(strategy_prefix)
        & summary["End-of-follow-up reason"].str.startswith(reason_prefix)
    ]
    if len(x) != 1:
        raise RuntimeError(
            f"Could not uniquely find {cohort} / {strategy_prefix} / {reason_prefix}"
        )
    return int(x.iloc[0]["Person-trial clones, n"])


def make_patch(summary):
    md_art = lookup(summary, "MIMIC-IV", "Defer", "Artificial censoring")
    md_death = lookup(summary, "MIMIC-IV", "Defer", "Death within")
    md_admin = lookup(summary, "MIMIC-IV", "Defer", "Administrative")

    mi_art = lookup(summary, "MIMIC-IV", "Initiate", "Artificial censoring")
    mi_death = lookup(summary, "MIMIC-IV", "Initiate", "Death within")
    mi_admin = lookup(summary, "MIMIC-IV", "Initiate", "Administrative")

    ed_art = lookup(summary, "eICU", "Defer", "Artificial censoring")
    ed_death = lookup(summary, "eICU", "Defer", "Documented in-hospital death")
    ed_dis = lookup(summary, "eICU", "Defer", "Alive hospital discharge")
    ed_admin = lookup(summary, "eICU", "Defer", "Administrative")

    ei_art = lookup(summary, "eICU", "Initiate", "Artificial censoring")
    ei_death = lookup(summary, "eICU", "Initiate", "Documented in-hospital death")
    ei_dis = lookup(summary, "eICU", "Initiate", "Alive hospital discharge")
    ei_admin = lookup(summary, "eICU", "Initiate", "Administrative")

    return f"""TARGET ITEM 10 — MANUSCRIPT/SUPPLEMENT PATCH

Suggested Methods sentence
--------------------------
At each sequential trial baseline, each eligible person-trial contributed one
clone to each strategy. Follow-up for a clone ended at the first strategy
deviation requiring artificial censoring, the relevant clinical endpoint, or
the 28-day administrative horizon. For the defer-at-least-6-hours strategy,
initiation of systemic corticosteroids during the 6-hour grace period triggered
artificial censoring at the observed initiation time. For the initiate-within-6-hours
strategy, clones without corticosteroid initiation were artificially censored at
6 hours. In eICU, documented in-hospital death and alive hospital discharge were
treated as competing clinical events; alive discharge was not treated as ordinary
censoring.

Suggested Results sentence — MIMIC-IV
--------------------------------------
Across {EXPECTED_MIMIC_TRIALS:,} MIMIC-IV sequential person-trials, each trial
was cloned to both strategies. In the defer-at-least-6-hours strategy,
{md_art:,} clones were artificially censored at corticosteroid initiation,
{md_death:,} ended in death within 28 days before strategy deviation, and
{md_admin:,} reached the 28-day administrative horizon without prior death or
strategy deviation. In the initiate-within-6-hours strategy, {mi_art:,} clones
were artificially censored at 6 hours because corticosteroids had not been
initiated, {mi_death:,} ended in death within 28 days before artificial censoring,
and {mi_admin:,} reached day 28 without prior death or strategy deviation.

Suggested Results sentence — eICU
----------------------------------
Across {EXPECTED_EICU_TRIALS:,} eICU sequential person-trials, in the
defer-at-least-6-hours strategy, {ed_art:,} clones were artificially censored at
corticosteroid initiation, {ed_death:,} ended in documented in-hospital death,
{ed_dis:,} ended in alive hospital discharge as a competing event, and
{ed_admin:,} reached the 28-day administrative horizon without a prior clinical
event or strategy deviation. In the initiate-within-6-hours strategy,
{ei_art:,} clones were artificially censored at 6 hours because corticosteroids
had not been initiated, {ei_death:,} ended in documented in-hospital death,
{ei_dis:,} ended in alive hospital discharge, and {ei_admin:,} reached day 28
without a prior clinical event or strategy deviation.

Suggested Supplement title
--------------------------
Table S11. Strategy-specific artificial censoring and end-of-follow-up reasons
in the sequential target-trial emulations.

TARGET item 10 status after insertion
-------------------------------------
READY — strategy-specific reasons for artificial censoring and end of follow-up
are reported for both MIMIC-IV and eICU.

Important interpretation note
-----------------------------
These are unweighted descriptive counts of cloned person-trial follow-up
termination, not treatment-effect estimates. They do not alter the frozen
causal analyses.
"""


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 110)
    print("TARGET ITEM 10 — FROZEN FOLLOW-UP / ARTIFICIAL-CENSORING AUDIT")
    print("=" * 110)

    mimic_trials = build_mimic_trials()
    mg = mimic_geometry(mimic_trials)
    mimic_summary, mimic_detail = summarize_mimic(mimic_trials, mg)

    eicu_trials = choose_eicu_trials()
    eg = eicu_geometry(eicu_trials)
    eicu_summary, eicu_detail = summarize_eicu(eicu_trials, eg)

    summary = pd.concat([mimic_summary, eicu_summary], ignore_index=True)
    detail = pd.concat([mimic_detail, eicu_detail], ignore_index=True)

    by_hour = pd.concat(
        [
            artificial_by_hour_mimic(mimic_trials, mg),
            artificial_by_hour_eicu(eicu_trials, eg),
        ],
        ignore_index=True,
    )

    baseline = by_trial_baseline(detail)

    # eICU source-data chronology context already addressed by Table S10.
    baseline_min = (
        pd.to_numeric(eicu_trials["respiratory_t0_candidate_min"], errors="coerce")
        + pd.to_numeric(eicu_trials["trial_hour"], errors="coerce") * 60
    )
    invalid_discharge = (
        pd.to_numeric(eicu_trials["hospital_discharge_offset_min"], errors="coerce").notna()
        & (
            pd.to_numeric(eicu_trials["hospital_discharge_offset_min"], errors="coerce")
            <= baseline_min
        )
    )

    # Hard partition QAs.
    qa_rows = []
    for cohort, expected in [
        ("MIMIC-IV", EXPECTED_MIMIC_TRIALS),
        ("eICU", EXPECTED_EICU_TRIALS),
    ]:
        for strategy in summary.loc[summary["Cohort"].eq(cohort), "Strategy"].unique():
            n = int(
                summary.loc[
                    summary["Cohort"].eq(cohort)
                    & summary["Strategy"].eq(strategy),
                    "Person-trial clones, n",
                ].sum()
            )
            qa_rows.append({
                "Cohort": cohort,
                "Strategy": strategy,
                "Expected strategy clones, n": expected,
                "Classified strategy clones, n": n,
                "Partition pass": int(n == expected),
            })
            if n != expected:
                raise RuntimeError(
                    f"Reason counts for {cohort} / {strategy} sum to {n}, expected {expected}."
                )

    qa = pd.DataFrame(qa_rows)

    summary.to_csv(
        OUTDIR / "TARGET10_01_Table_S11_Strategy_Specific_End_of_Followup.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_hour.to_csv(
        OUTDIR / "TARGET10_02_Defer_Artificial_Censoring_By_Relative_Hour.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline.to_csv(
        OUTDIR / "TARGET10_03_End_of_Followup_By_Trial_Baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    qa.to_csv(
        OUTDIR / "TARGET10_04_PARTITION_QA.csv",
        index=False,
        encoding="utf-8-sig",
    )

    patch = make_patch(summary)
    (OUTDIR / "TARGET10_MANUSCRIPT_PATCH.txt").write_text(
        patch,
        encoding="utf-8",
    )

    manifest = {
        "analysis_changed": False,
        "models_fit_or_refit": False,
        "bootstrap_run": False,
        "mimic_frozen_stays": EXPECTED_MIMIC_STAYS,
        "mimic_frozen_person_trials": EXPECTED_MIMIC_TRIALS,
        "mimic_initiate_within6h_person_trials": EXPECTED_MIMIC_INIT6,
        "eicu_frozen_stays": EXPECTED_EICU_STAYS,
        "eicu_frozen_patient_clusters": EXPECTED_EICU_CLUSTERS,
        "eicu_frozen_person_trials": EXPECTED_EICU_TRIALS,
        "eicu_initiate_within6h_person_trials": EXPECTED_EICU_INIT6,
        "eicu_trials_with_recorded_hospital_discharge_at_or_before_trial_baseline": int(
            invalid_discharge.sum()
        ),
        "note_on_eicu_chronology": (
            "The small source-data chronology issue is already reported separately "
            "in Supplementary Table S10 and its post-hoc sensitivity analysis."
        ),
        "target_item_10_after_inserting_outputs": "READY",
    }
    (OUTDIR / "TARGET10_QA_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nStrategy-specific end-of-follow-up counts:")
    print(summary.to_string(index=False))

    print("\nPartition QA:")
    print(qa.to_string(index=False))

    print("\n" + "=" * 110)
    print("TARGET ITEM 10 AUDIT COMPLETE")
    print("=" * 110)
    print(f"Outputs: {OUTDIR}")
    print("\nSend back the entire TARGET10_followup_audit folder.")


if __name__ == "__main__":
    main()

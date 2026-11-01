# Reproduction workflow

## Data access

Obtain authorized access to MIMIC-IV and the eICU Collaborative Research
Database independently through PhysioNet. Do not place restricted data in this
repository.

The frozen MIMIC project used MIMIC-IV v3.1. If MIMIC-IV-ED is available locally,
the baseline-covariate repair stage can use it to improve pretreatment vital-sign
coverage.

## MIMIC-IV: authorized source data to frozen primary analysis

Copy `analysis/mimic/config.py.example` to `analysis/mimic/config.py`, set
`DATA_ROOT` (or the `MIMIC_DATA_ROOT` environment variable), and run from the
repository root:

```bash
python analysis/mimic/00_check_data.py
python analysis/mimic/01_run_feasibility.py
python analysis/mimic/08_build_trial_eligible_cohort.py
python analysis/mimic/09_extract_baseline_covariates.py
python analysis/mimic/10_repair_baseline_covariates.py
python analysis/mimic/12_run_time_updated_ccw.py
```

These stages create restricted local intermediate files including:

- `analysis/mimic/outputs/stage5_5_analysis_cohort.parquet`
- `analysis/mimic/outputs/stage7_hourly_panel.parquet`

Do not commit or redistribute these files.

Optional exposure-description audits retained in the repository are:

```bash
python analysis/mimic/06_exposure_audit.py
python analysis/mimic/07_emar_exposure_audit.py
```

### Outcome-blind 2-hour design-development audit

The source chain is retained because the study reports why the 2-hour candidate
strategy was not carried forward. After the hourly panel has been built, run:

```bash
python analysis/mimic/14_run_high_specificity_ccw.py
python analysis/mimic/06_two_hour_weight_model_diagnostic.py
python analysis/mimic/07_two_hour_overlap_diagnostic.py
python analysis/mimic/08_two_hour_overlap_bootstrap_diagnostic.py
python analysis/mimic/09_two_hour_hourly_ipcw_diagnostic.py
python analysis/mimic/10_two_hour_pure_ccw_support_audit.py
python analysis/mimic/11_two_hour_candidate_model_audit.py
```

The 2-hour design-development decision uses treatment/censoring-process support
and balance diagnostics rather than mortality outcomes. The unsupported strategy
is not carried forward to a formal mortality-effect analysis.

### Frozen 6-hour strategy

Run the outcome-blind support gate before the final mortality analysis:

```bash
python analysis/mimic/12_six_hour_empirical_support.py
python analysis/mimic/13_final_6h_ccw_effect.py
```

The final script checks the frozen support decision before estimating the primary
mortality effect.

Expected frozen MIMIC counts:

- 2,014 unique ICU stays
- 18,487 sequential person-trials
- 744 person-trials with initiation within 6 hours

Published point estimate (approximately):

- defer risk 0.3661
- initiate risk 0.4233
- risk difference +0.0573
- risk ratio 1.156

## eICU external evaluation

Copy `analysis/eicu/config.py.example` to `analysis/eicu/config.py` and set
`EICU_DATA_DIR` (or set the environment variable `EICU_DATA_DIR`). Then run in
order from the repository root:

```bash
python analysis/eicu/01_outcome_blind_mapping_audit.py
python analysis/eicu/02_locked_mapping_empirical_support.py
python analysis/eicu/03_final_external_effect.py
python analysis/eicu/04_inference_stability.py
python analysis/eicu/05_chronology_sensitivity.py
```

Expected final external counts:

- 5,394 ICU stays
- 4,794 patient clusters
- 47,923 sequential person-trials
- 1,341 person-trials with initiation within 6 hours

Published point estimate (approximately):

- defer risk 0.2147
- initiate risk 0.2383
- risk difference +0.0236
- risk ratio 1.11

The external endpoint is documented in-hospital death within 28 days from each
trial baseline, with alive hospital discharge treated as a competing event. It
is a partial endpoint replication because eICU does not provide reliable
post-discharge 28-day vital status.

## Reporting

Table and figure code is under `reporting/`. The TARGET item 10 audit reconstructs
strategy-specific artificial censoring and end-of-follow-up counts without
refitting causal models.

## Published-output verification

After both final effect files are available, run:

```bash
python tools/verify_published_outputs.py \
  --mimic-effect analysis/mimic/outputs/final_effect_estimates.csv \
  --eicu-effect analysis/eicu/outputs_eicu_E3/external_effect_estimates.csv
```

All discrete cohort/trial counts are required to match exactly. The MIMIC effect
check allows only the small cross-environment numerical drift documented during
clean-environment reproduction; the eICU E3 effect is checked at tight
floating-point tolerance.

## Reproducibility boundaries

The release reproduces the scientific workflow while respecting database-access
restrictions. Patient-level intermediate datasets are intentionally excluded.
Aggregate example outputs are provided for cross-checking.

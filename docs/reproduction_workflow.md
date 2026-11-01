# Reproduction workflow

## Data access

Obtain authorized access to MIMIC-IV and the eICU Collaborative Research
Database independently through PhysioNet. Do not place restricted data in this
repository.

The frozen MIMIC project used MIMIC-IV v3.1. If MIMIC-IV-ED is available
locally, Stage 5.5 can use it to improve pretreatment vital-sign coverage.

## MIMIC-IV: from authorized source data to frozen primary analysis

Copy `analysis/mimic/config.py.example` to `analysis/mimic/config.py` and set
`DATA_ROOT` (or the `MIMIC_DATA_ROOT` environment variable).

Run from `analysis/mimic/`:

```bash
python 00_check_data.py
python 01_run_feasibility.py
python 08_build_trial_eligible_cohort.py
python 09_extract_baseline_covariates.py
python 10_repair_baseline_covariates.py
python 12_run_time_updated_ccw.py
```

These stages create restricted local intermediate files including:

- `outputs/stage5_5_analysis_cohort.parquet`
- `outputs/stage7_hourly_panel.parquet`

Do not commit or redistribute them.

### Outcome-blind 2-hour design-development audit

The full source dependency chain is retained because the published supplement
reports why the 2-hour candidate strategy was abandoned:

```bash
python 14_run_high_specificity_ccw.py
python 06_two_hour_weight_model_diagnostic.py
python 07_two_hour_overlap_diagnostic.py
python 08_two_hour_overlap_bootstrap_diagnostic.py
python 09_two_hour_hourly_ipcw_diagnostic.py
python 10_two_hour_pure_ccw_support_audit.py
python 16outcome-blind 2-hour design audit_stage10_hour2_model_audit.py
```

Model choice in the outcome-blind 2-hour design audit audit uses treatment/censoring-process support
diagnostics and not mortality outcomes.

### Frozen 6-hour strategy

```bash
python 12_six_hour_empirical_support.py
python 16frozen final 6-hour analysis_stage10_final_6h_ccw.py
```

The frozen 6-hour feasibility analysis support gate is outcome-blind. frozen final 6-hour analysis checks the frozen decision before
estimating the primary mortality effect.

Expected MIMIC frozen analysis counts:

- 2,014 unique ICU stays
- 18,487 sequential person-trials
- 744 person-trials with initiation within 6 h

Published point estimate approximately:

- defer risk 0.3661
- initiate risk 0.4233
- RD +0.0573
- RR 1.156

## eICU external evaluation

Copy `analysis/eicu/config.py.example` to `analysis/eicu/config.py`, point it to
an authorized local eICU copy, then run in order:

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
- 1,341 person-trials with initiation within 6 h

Published point estimate approximately:

- defer risk 0.2147
- initiate risk 0.2383
- RD +0.0236
- RR 1.11

## Reporting

Table and figure code is under `reporting/`. The TARGET item 10 audit reconstructs
strategy-specific artificial censoring and end-of-follow-up counts without
refitting causal models.

## Reproducibility boundaries

The code release reproduces the scientific workflow while respecting database
access restrictions. Patient-level intermediate datasets are intentionally
excluded. Aggregate example outputs are provided for cross-checking.

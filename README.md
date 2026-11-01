# Timing of corticosteroid initiation in severe community-acquired pneumonia

Reproducibility code for a sequential target-trial emulation evaluating
**initiation of systemic corticosteroids within 6 hours versus deferral for at
least 6 hours** among patients with severe community-acquired pneumonia.

MIMIC-IV is used for design development and primary causal estimation; the eICU
Collaborative Research Database is used for locked external evaluation.

## Scientific design

- Sequential trial baselines: 0, 2, 4, ..., 18 hours after respiratory time zero.
- Strategies: initiate systemic corticosteroids within the next 6 hours vs defer
  initiation for at least 6 hours.
- Recent-initiation interval for adherence modeling: [3, 6) hours.
- Target population: overlap/equipoise population using q(X)=e(X)[1-e(X)].
- MIMIC-IV outcome: 28-day all-cause mortality from each trial baseline.
- eICU outcome: documented in-hospital death within 28 days, with alive hospital
  discharge treated as a competing event.
- A 2-hour candidate strategy was evaluated outcome-blind and was not carried
  forward because empirical-support criteria were not satisfied.
- The 6-hour design was frozen before the primary mortality effect analysis.

## Data are not included

This repository **does not distribute MIMIC-IV, eICU, or any patient-level
derived dataset**. Users must independently obtain authorized access through
PhysioNet and comply with all applicable credentialing and data-use agreements.

Do not commit patient-level source data, derived cohort/trial files, caches,
identifier lists, weights, or credentials. The included `.gitignore` blocks the
main local data/output formats used by this project.

## Repository layout

```text
analysis/
  mimic/
    00_check_data.py
    01_run_feasibility.py
    08_build_trial_eligible_cohort.py
    09_extract_baseline_covariates.py
    10_repair_baseline_covariates.py
    12_run_time_updated_ccw.py
    12_six_hour_empirical_support.py
    13_final_6h_ccw_effect.py
    14_run_high_specificity_ccw.py
    06_two_hour_weight_model_diagnostic.py
    07_two_hour_overlap_diagnostic.py
    08_two_hour_overlap_bootstrap_diagnostic.py
    09_two_hour_hourly_ipcw_diagnostic.py
    10_two_hour_pure_ccw_support_audit.py
    11_two_hour_candidate_model_audit.py
  eicu/
    01_outcome_blind_mapping_audit.py
    02_locked_mapping_empirical_support.py
    03_final_external_effect.py
    04_inference_stability.py
    05_chronology_sensitivity.py
reporting/
  tables/
  figures/
example_outputs/
  mimic/   # aggregate, non-patient-level outputs only
  eicu/    # aggregate, non-patient-level outputs only
docs/
tools/
```

Additional MIMIC exposure-audit scripts (`06_exposure_audit.py` and
`07_emar_exposure_audit.py`) are retained for auditability but are not required
to run the frozen primary analysis chain.

## Reproduction workflow

See [`docs/reproduction_workflow.md`](docs/reproduction_workflow.md).

The release intentionally preserves the outcome-blind design-development logic.
The unsupported 2-hour strategy is documented rather than hidden.

## Configuration

### MIMIC-IV

Copy `analysis/mimic/config.py.example` to `analysis/mimic/config.py` and set
`DATA_ROOT` to an authorized local MIMIC-IV installation. The frozen project used
MIMIC-IV v3.1. Local outputs and caches remain under `analysis/mimic/` by default
and are ignored by Git.

The upstream scripts create restricted local intermediate files including
`stage5_5_analysis_cohort.parquet` and `stage7_hourly_panel.parquet`. These files
must remain local and must not be redistributed.

### eICU

Copy `analysis/eicu/config.py.example` to `analysis/eicu/config.py` and set
`EICU_DATA_DIR` to an authorized local eICU data directory. The environment
variable `EICU_DATA_DIR` can also be used.

## Software

The frozen causal analyses used Python 3.11.9, pandas 3.1.0, NumPy 2.4.6,
scikit-learn 1.9.1, and pyarrow 26.0.0. See `requirements-analysis.txt` and
`environment.yml`.

## Output verification

After reproducing the final MIMIC and eICU effect files, run:

```bash
python tools/verify_published_outputs.py \
  --mimic-effect analysis/mimic/outputs/final_effect_estimates.csv \
  --eicu-effect analysis/eicu/outputs_eicu_E3/external_effect_estimates.csv
```

MIMIC effect values are checked using the small numerical-concordance tolerance
established during clean-environment reproduction; all discrete cohort/trial
counts must match exactly. eICU E3 values are checked at tight floating-point
tolerance.

## Release

Version 1.0.0 is the curated reproducibility release associated with the study.
Aggregate example outputs are included for cross-checking; restricted source data
and patient-level intermediate datasets are not.

## License

Code is released under the MIT License. The license applies to code in this
repository only and does not alter the access terms or data-use agreements for
MIMIC-IV or eICU.

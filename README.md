# Timing of corticosteroid initiation in severe community-acquired pneumonia

Reproducibility code for a sequential target-trial emulation evaluating
**initiation of systemic corticosteroids within 6 hours versus deferral for at
least 6 hours** among patients with severe community-acquired pneumonia.

The study uses MIMIC-IV for design development and primary causal estimation and
the eICU Collaborative Research Database for external evaluation.

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

## Data are NOT included

This repository **does not distribute MIMIC-IV, eICU, or any patient-level
derived dataset**. Users must independently obtain authorized access through
PhysioNet and comply with all applicable credentialing and data-use agreements.

Never commit:
- `.parquet`, `.feather`, `.pkl`, `.pickle`, database extracts, or caches;
- subject/stay ID lists;
- patient-level weights or trial panels;
- PhysioNet credentials or tokens.

## Repository layout

```text
analysis/
  mimic/
    01_feasibility_scan.py
    02_construct_trial_eligible_cohort.py
    03_construct_baseline_cohort.py
    04_repair_baseline_covariates.py
    05_construct_dynamic_hourly_panel.py
    06_two_hour_weight_model_diagnostic.py
    07_two_hour_overlap_diagnostic.py
    08_two_hour_overlap_bootstrap_diagnostic.py
    09_two_hour_hourly_ipcw_diagnostic.py
    10_two_hour_pure_ccw_support_audit.py
    11_two_hour_candidate_model_audit.py
    12_six_hour_empirical_support.py
    13_final_6h_ccw_effect.py
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

## Reproduction workflow

See [`docs/reproduction_workflow.md`](docs/reproduction_workflow.md).

The public release intentionally preserves the outcome-blind design-development
logic. The unsupported 2-hour strategy is documented rather than hidden.

## Configuration

### MIMIC-IV

Copy:

```text
analysis/mimic/config.py.example
```

to:

```text
analysis/mimic/config.py
```

and set `OUTPUT_DIR` to your **restricted local** analysis-output directory.

The repository includes the upstream scripts that create the restricted local
intermediate files `stage5_5_analysis_cohort.parquet` and
`stage7_hourly_panel.parquet` from authorized MIMIC-IV data. These derived
patient-level files remain local and must not be redistributed.

### eICU

Copy:

```text
analysis/eicu/config.py.example
```

to:

```text
analysis/eicu/config.py
```

and set `EICU_DATA_DIR` to your authorized local eICU data directory.

Alternatively, set the environment variable `EICU_DATA_DIR`.

## Software

The frozen causal analyses used Python 3.11.9, pandas 3.1.0, NumPy 2.4.6,
scikit-learn 1.9.1, and pyarrow 26.0.0.

See `requirements-analysis.txt`.

## Release status

This repository is the curated **v1.0.0 public reproducibility release**.

The release includes the upstream MIMIC pipeline required to build the frozen
patient-level intermediate files locally from authorized MIMIC-IV data, the
outcome-blind 2-hour design-development audit, the frozen 6-hour primary
analysis, the locked eICU external evaluation, sensitivity analyses, and
reporting code.

The original local helper module `src/utils.py` was not present in the upstream
export bundle. The public repository therefore includes a narrowly reconstructed
compatibility implementation for file discovery and DuckDB connection setup.
It contains no scientific definitions or causal-analysis rules; this is
documented in `SOURCE_PROVENANCE.csv`.

## License

Code is released under the MIT License. The license applies to code in this
repository only and does not alter the access terms or data-use agreements for
MIMIC-IV or eICU.

# Design-development and freeze record

The design-development sequence is part of the scientific record.

## Unsupported 2-hour candidate strategy

A 2-hour initiation strategy was evaluated using outcome-blind empirical-support
diagnostics. The candidate failed the prespecified support/balance criteria
(maximum absolute standardized mean difference approximately 0.227, with
multiple covariates exceeding 0.10). It was therefore not carried forward to a
formal mortality effect analysis.

The source-backed aggregate diagnostic files are included in
`example_outputs/mimic/`.

## Frozen 6-hour strategy

The 6-hour strategy passed the outcome-blind support gate:
- initiation within the next 6 hours vs deferral for at least 6 hours;
- sequential baselines every 2 hours from 0 to 18 hours;
- recent-initiation interval [3,6) hours;
- overlap/equipoise target population;
- time-updated deferral censoring model;
- recent-initiation adherence model.

The design was frozen before the primary mortality effect was estimated.

## External evaluation

eICU mapping and empirical support were locked outcome-blind before the external
effect estimate was opened. Later chronology and inference-stability analyses
are explicitly labeled as sensitivity/data-quality analyses and do not replace
the frozen external point estimate.


The source code for the outcome-blind 2-hour design-development diagnostic chain is retained under `analysis/mimic/` for auditability.

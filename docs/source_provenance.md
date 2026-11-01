# Source provenance

`SOURCE_PROVENANCE.csv` records the source package/member and SHA-256 hashes for
curated scripts.

Most scientific scripts are copied directly from frozen local analysis packages
or the local upstream candidate export, with path/configuration changes only
where documented.

One exception is `analysis/mimic/src/utils.py`: the original local helper was not
included in the upstream export. The public compatibility implementation was
reconstructed from its call signatures and usage in the supplied scripts. It
only handles file discovery, DuckDB connection setup, SQL source expressions,
path quoting, and percentage formatting. It does not define scientific
eligibility, exposure, outcome, censoring, weighting, or estimand logic.

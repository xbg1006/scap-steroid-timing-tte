# Public naming policy

Public-facing filenames and documentation use scientific/descriptive names
rather than internal development identifiers.

Examples:
- `final_effect_estimates.csv`
- `weight_support_diagnostics.csv`
- `two_hour_design_model_selection.csv`
- `external_effect_estimates.csv`

The original research workspace used internal labels such as script/stage/version
numbers during development. Those labels are retained **only** in
`SOURCE_PROVENANCE.csv` so that public files can be traced back to the frozen
research artifacts.

Internal labels in `SOURCE_PROVENANCE.csv` are not database release numbers,
manuscript versions, or alternative published analyses.

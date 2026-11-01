# Target-trial specification

| Component | Specification |
|---|---|
| Population | Severe community-acquired pneumonia population meeting frozen eligibility criteria; major immunocompromise excluded from the primary MIMIC population |
| Trial baselines | 0, 2, 4, ..., 18 h after respiratory time zero |
| Strategy 1 | Initiate systemic corticosteroids within the next 6 h |
| Strategy 2 | Defer systemic corticosteroid initiation for at least 6 h |
| Assignment in target trial | Random assignment at each eligible trial baseline |
| Observational emulation | Cloning, artificial censoring, and weighting |
| Follow-up | Up to 28 d from each trial baseline |
| MIMIC outcome | 28-d all-cause mortality |
| eICU outcome | Documented in-hospital death within 28 d; alive hospital discharge as competing event |
| Effect measures | Absolute risk, risk difference, risk ratio |
| Target population | Clinical-equipoise/overlap population |

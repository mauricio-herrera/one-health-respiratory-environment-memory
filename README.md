# One-Health respiratory-environment memory modeling — reproducibility release v1.0.0

This release contains clean, aggregated, publication-level datasets and frozen
derived outputs supporting a regional One-Health analysis of daily respiratory
urgent-care burden, environmental exposures, vulnerability, and finite-memory
features.

## Scope

Historical:
- 2019–2024
- 35,072 region-day rows
- 2,192 days
- 16 canonical units
- accepted M0–M3 temporal-validation outputs

External 2025:
- 4,015 region-day rows
- 365 days
- 11 units
- repaired prospective M0–M2 fits trained only on 2019–2024
- frozen SADU urgent-respiratory outcome
- pooled/regional metrics
- paired seven-day moving-block bootstrap (B=5000; seed=20260822)

## Frozen interpretation

M2 has lower pooled point MAE, RMSE and mean Poisson deviance than M0 and M1,
but all nine paired 95% bootstrap confidence intervals include zero. External
M0–M2 superiority is therefore not established.

The original formal external success gate belongs to M3 and remains deferred
until GRD 2025 becomes available.

No operational EWS, causal, or weather-forecasting claim is supported by this
release.

See `docs/MANUAL_PUBLICATION_CHECKLIST.md` before public deposition.


## Complete-data publication policy

This release contains the **complete integrated research dataset**. Original
source observations are attributable to their respective public source
organizations; the authors do not claim ownership over those observations.

The research contribution is the integrated, cleaned, harmonized, audited, and
model-ready dataset together with its derived variables, models, predictions,
validation outputs, and reproducibility layer.

See:
- `docs/DATA_OWNERSHIP_INTEGRATION_AND_REUSE.md`
- `docs/SOURCE_RIGHTS_POLICY.csv`
- `docs/SOURCE_ATTRIBUTIONS.md`
- `docs/SINCA_ACADEMIC_USE_NOTE.md`

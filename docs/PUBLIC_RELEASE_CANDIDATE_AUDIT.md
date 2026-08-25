# Public release candidate audit

Candidate: `one-health-memory-publication-release-v1.0.0-rc1`

This RC was derived from the computationally validated publication release
without changing scientific data, predictions, metrics, bootstrap outputs, or
interpretations.

## Portability changes

- workstation-absolute paths were removed from public `SOURCE_PROVENANCE.csv`;
- exact local provenance was retained outside the public candidate in
  `internal_audit_not_for_public_deposition/`;
- hard-coded project-root defaults in frozen reference scripts were replaced by
  `/PATH/TO/ONE_HEALTH_PROJECT`;
- release manifest/checksums were regenerated after sanitization.

## Scientific state

- historical 2019–2024: CLOSED / ACCEPTED;
- external M0–M2 2025: CLOSED / SUPPLEMENTAL;
- external M3: DEFERRED until GRD 2025.

## Public-deposition state

**MANUAL_HOLD**

The candidate is portable and internally auditable, but public deposition must
wait until:
1. data redistribution rights are reviewed source-by-source;
2. the final data license is selected;
3. complete author metadata is inserted;
4. GitHub repository URL and, later, Zenodo DOI are inserted.

No model or inferential change is authorized during that metadata/licensing step.

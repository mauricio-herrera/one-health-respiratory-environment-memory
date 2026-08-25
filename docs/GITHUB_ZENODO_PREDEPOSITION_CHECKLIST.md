# GitHub → Zenodo pre-deposition checklist

## Rights

- [ ] Exact SADU CC non-commercial licence variant confirmed.
- [ ] Exact DEIS source licence/terms bound.
- [ ] Exact GRD source licence/terms bound.
- [ ] Exact SINCA source licence/terms bound, if SINCA-derived fields are included.
- [ ] INE CC BY-SA attribution/share-alike implications reviewed.
- [ ] CASEN derived-data redistribution treatment reviewed.
- [ ] Copernicus ERA5 attribution notice included.
- [ ] Final mixed-license/custom-rights strategy selected.

## Metadata

- [ ] Complete creator list and order.
- [ ] ORCIDs.
- [ ] Affiliations.
- [ ] Corresponding/contact creator.
- [ ] Final release title.
- [ ] Manuscript title/status.
- [ ] GitHub repository URL.
- [ ] Funding statement.
- [ ] Keywords.
- [ ] Data availability wording.

## GitHub

- [ ] Create clean repository from final candidate only.
- [ ] Add final `CITATION.cff`.
- [ ] Add `.zenodo.json` only after metadata is complete.
- [ ] Validate all relative links.
- [ ] Run `code/validate_release.py`.
- [ ] Commit and tag `v1.0.0`.
- [ ] Create GitHub release `v1.0.0`.

## Zenodo

- [ ] Confirm file count <= 100.
- [ ] Confirm total upload size <= 50 GB.
- [ ] Create/save a draft first.
- [ ] Select Dataset as primary resource type unless strategy changes.
- [ ] Add verified standard licences and/or custom rights statement.
- [ ] Reserve DOI if it should be inserted into repository metadata before publication.
- [ ] Preview the record.
- [ ] Publish only after rights and metadata review are complete.

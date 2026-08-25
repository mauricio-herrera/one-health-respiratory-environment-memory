# Zenodo deposition and licence strategy

## Why no blanket data licence is selected yet

The release combines software and derived data from source families with
different reuse conditions. Current evidence includes:

- SADU: portal label indicating a Creative Commons non-commercial licence;
- ERA5: Copernicus licence permitting redistribution/adaptation with attribution;
- INE: CC BY-SA 4.0;
- CASEN: public research/publication use documented, but derivative
  redistribution licence not yet frozen;
- DEIS/GRD/SINCA: exact source licences still require binding.

Applying a single permissive licence to all data could accidentally grant rights
that the depositor does not possess.

## Recommended Zenodo approach

Zenodo currently requires a licence field, but supports **multiple licences** and
**custom licences** for mixed-license uploads.

For the final record:

1. Keep software/code under MIT.
2. Use source-specific notices for source-derived data.
3. Add all applicable standard licences that are exact and verified.
4. If necessary, add a custom data-rights statement describing the mixed-source
   derivative dataset and referring to `SOURCE_RIGHTS_LEDGER.csv`.
5. Do not select Zenodo's default CC BY 4.0 for the whole deposit unless the
   source-rights compatibility review explicitly supports it.

## Record type

The principal contribution is the dataset/reproducibility package, so `Dataset`
is a reasonable primary Zenodo resource type. If the software itself becomes a
major standalone contribution, consider a separate Software record linked to the
Dataset record.

## GitHub integration

Do not create a root `.zenodo.json` until author metadata and rights metadata are
final. If both `.zenodo.json` and `CITATION.cff` exist, Zenodo's GitHub
integration prioritizes `.zenodo.json`.

The templates in `docs/templates/` are intentionally non-active.

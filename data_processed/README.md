# data_processed

This directory is reserved for processed cohort-level data generated from ADNI and NACC source files.

## Note on data availability

The public version of this repository does **not** include patient-level processed data.

The following types of files are therefore not distributed here:

- cohort tables
- label tables
- module feature tables
- train/validation/test split files
- external evaluation input tables

This decision is made because the project relies on controlled-access clinical datasets (ADNI and NACC), and patient-level processed derivatives should not be redistributed in this public repository.

## How to reproduce processed data

Users who have obtained authorized access to the relevant source datasets may reproduce the processed data by running the cohort, label, module, and split generation scripts provided in this repository.

Relevant code is located in:

- `src/data/`
- `scripts/`

Typical processing steps include:

1. build ADNI cohort
2. build survival and auxiliary labels
3. build baseline / structure / state / dynamics modules
4. create train / validation / test splits
5. build harmonized NACC cohort for external validation

## Public repository policy

In the public release, this directory may contain only lightweight placeholder files or documentation, but not the actual processed patient-level tables.

Model checkpoint is not included in the current public release. It can be shared upon reasonable request or added in a later reproducibility release.
# Data access note

This repository contains code, lightweight training summaries, manuscript-ready figures, and summary tables for a modular, availability-aware clinical AI framework for MCI progression modeling.

## Controlled-access datasets

The full pipeline was developed using data derived from:

- **ADNI** (Alzheimer’s Disease Neuroimaging Initiative)
- **NACC** (National Alzheimer’s Coordinating Center)

These datasets are subject to controlled-access or governed data-use conditions. Therefore, this public repository does **not** redistribute:

- raw source data
- patient-level processed cohort tables
- patient-level label tables
- patient-level module feature tables
- train/validation/test split files derived from controlled-access data
- patient-level embedding outputs
- full model checkpoint files

## What is included in this public release

This repository does include:

- framework implementation code
- baseline comparison scripts
- missingness robustness scripts
- figure files
- summary result tables
- lightweight training metadata and preprocessing summaries

## How to reproduce the pipeline

Users who have obtained authorized access to ADNI and/or NACC may reproduce the full workflow by:

1. obtaining access through the official data governance channels
2. placing source files locally according to the expected project structure
3. rebuilding cohorts, labels, modules, and splits using the provided scripts
4. rerunning training, evaluation, and robustness analysis

Relevant code is primarily located in:

- `src/data/`
- `src/models/`
- `src/training/`
- `src/evaluation/`
- `scripts/`

## Purpose of this release

The goal of this public repository is to document the framework design, experimental logic, and reproducible code structure without redistributing controlled clinical data.
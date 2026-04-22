# AD_MCI_framework_v3

A modular, availability-aware clinical AI framework for progression modeling in mild cognitive impairment (MCI) using routine longitudinal follow-up data.

## Overview

This repository presents a clinically motivated framework for survival-oriented MCI progression modeling from routinely collected longitudinal follow-up data.

Unlike many prediction pipelines that depend on expensive or inconsistently available biomarkers such as PET, CSF, or MRI, this framework focuses on **routine clinical follow-up information** and is designed for realistic settings where visits are irregular and module-level data may be incomplete.

To preserve clinical interpretability, longitudinal information is organized into four explicit modules:

- **Baseline** — baseline patient profile  
- **Structure** — follow-up pattern and visit organization  
- **State** — current clinical status  
- **Dynamics** — recent short-term change over time  

Each module is encoded independently and fused under an **availability-aware mechanism**, enabling progression modeling and patient-level representation learning under incomplete inputs.

## Framework overview

![Framework overview](results/figures/figure1_overview.png)

**Figure 1.** Overview of the proposed modular, availability-aware clinical framework. Routine longitudinal follow-up data are organized into clinically interpretable modules, encoded independently, and fused under explicit module availability to produce a unified patient representation for downstream prediction and analysis.

## Study setting

- **Development cohort:** ADNI  
- **External validation cohort:** NACC  
- **Primary tasks:** survival risk prediction, 3-year progression classification, high-risk subgroup identification, subgroup discovery, and external validation  

Because ADNI and NACC are controlled-access datasets, raw data and patient-level processed derivatives are **not distributed** in this repository.

For access notes, see [`docs/data_access_note.md`](docs/data_access_note.md).

## Key results

- **Internal test performance (ADNI):** C-index **0.792**  
- **External validation (NACC):** C-index **0.746**  
- Designed for **incomplete longitudinal clinical inputs**  
- Uses **routine follow-up data** without requiring PET / CSF / MRI as mandatory inputs  

Additional figures and summary tables are provided in:

- [`results/figures/`](results/figures/)
- [`results/tables/`](results/tables/)

## Repository structure

```text
AD_MCI_framework_v3/
├── README.md
├── requirements.txt
├── environment.yml
├── docs/
│   └── data_access_note.md
├── data_processed/
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
├── scripts/
└── src/
    ├── data/
    ├── models/
    └── training/

  Setup
conda env create -f environment.yml
conda activate <env_name>

Dependencies are listed in requirements.txt and environment.yml.

Pipeline
Build ADNI cohort
Construct labels
Build modules (baseline / structure / state / dynamics)
Train model
Internal evaluation + KM analysis
External validation (NACC)
Baseline comparison and robustness
Scripts

Main pipeline:

01_build_adni_cohort.py
02_build_labels.py
03_build_modules.py
04_make_splits.py
05_export_test_outputs.py
06_cluster_and_km.py
07_build_nacc_cohort.py
Outputs
results/figures/
results/tables/
results/models/

Includes performance, KM curves, external validation, and robustness analysis.

Reproducibility

Code, figures, and tables are provided.
ADNI/NACC data are not included (controlled access required).

Contact

📧 liqirui019@gmail.com
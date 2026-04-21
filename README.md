# AD_MCI_framework_v3

A modular, availability-aware clinical AI framework for progression modeling in mild cognitive impairment (MCI) using routine longitudinal follow-up data.

## Overview

This repository implements a four-module clinical framework for survival-oriented MCI progression modeling **without relying on expensive biomarker modalities** such as PET, CSF, or MRI.

The central idea is that routine longitudinal clinical follow-up data should not be treated as one undifferentiated feature matrix. Instead, patient information is organized into four clinically interpretable modules:

- **Baseline** — who the patient is at baseline
- **Structure** — how the patient has been followed over time
- **State** — the patient’s current clinical status
- **Dynamics** — how the patient has recently changed

Each module is embedded separately and then fused under an explicit **availability-aware** mechanism, allowing the framework to preserve module identity while handling incomplete inputs.

This repository should be understood as a **framework paper repository**, not a claim of raw benchmark superiority over all simpler baselines. Its main value lies in:

- modular clinical organization of longitudinal data
- explicit handling of missing module availability
- survival-oriented progression modeling
- external transportability across cohorts
- patient representation learning for downstream stratification

## Clinical motivation

Predicting progression in MCI is difficult because:

- disease trajectories are heterogeneous
- longitudinal follow-up is often incomplete or irregular
- many predictive approaches depend on expensive biomarkers that are not routinely available

This project asks a practical question:

**Can routine longitudinal clinical follow-up data support meaningful progression modeling when they are explicitly organized into clinically interpretable modules and fused under module-availability constraints?**

## Framework design

### Four clinical modules

#### 1. Baseline module
- age
- sex
- education
- APOE4
- baseline MMSE
- baseline CDR global

#### 2. Structure module
- number of visits
- follow-up span
- visits per year
- median visit interval
- variability of visit interval

#### 3. State module
- state MMSE
- state CDR global
- state CDR-SB
- state ADAS
- state FAQ

#### 4. Dynamics module
- MMSE delta at 6 and 12 months
- CDR-SB delta at 6 and 12 months
- ADAS delta at 6 and 12 months
- FAQ delta at 6 and 12 months

## Model architecture

The framework consists of five main stages:

1. **Module-specific embedding**  
   Each clinical module is encoded independently.

2. **Module identity preservation**  
   Module-type embeddings indicate whether a token corresponds to baseline, structure, state, or dynamics.

3. **Availability-aware fusion**  
   Module embeddings are fused using an attention-based transformer-lite encoder under an explicit availability mask.

4. **Gated aggregation**  
   Fusion weights are learned over available modules to generate a unified patient representation.

5. **Multi-task prediction**  
   The fused representation is used for:
   - survival risk prediction
   - 3-year progression classification
   - high-risk classification

The same learned representation is also used for:

- latent-space visualization
- clustering
- Kaplan–Meier subgroup analysis

## Data sources

This project uses:

- **ADNI** for model development and internal evaluation
- **NACC** for external validation

Because ADNI and NACC are controlled-access datasets, raw data and patient-level processed derivatives are **not distributed** in this public repository.

See [`docs/data_access_note.md`](docs/data_access_note.md) for details.

## Main results snapshot

### Internal ADNI results
- Best validation C-index: **0.801**
- Test C-index: **0.792**
- 3-year progression accuracy: **0.730**
- High-risk classification accuracy: **0.690**

### External NACC validation
- External C-index: **0.746**
- Clear Kaplan–Meier separation by model-predicted risk group in the external cohort

### Conventional survival baselines on the same ADNI split
- Cox proportional hazards: **0.817**
- Random survival forest: **0.811**
- DeepSurv: **0.811**
- Proposed modular framework: **0.792**

These results indicate that the repository should not be interpreted as claiming raw predictive superiority over all simpler baselines. Instead, its contribution lies in structured longitudinal modeling, tolerance to incomplete inputs, external transportability, and representation-level value.

## Missingness robustness

### Simulated state-module missingness
Under removal of the entire **state** module in the ADNI test cohort:

- Cox: **0.7403**  (Δ = -0.0767)
- RSF: **0.6953**  (Δ = -0.1157)
- DeepSurv: **0.6710**  (Δ = -0.1400)
- Proposed framework: **0.7154**  (Δ = -0.0766)

This indicates that the modular framework showed degradation comparable to Cox and smaller degradation than RSF and DeepSurv under loss of the clinically important state module.

### Simulated random 30% feature missingness
Under random masking of 30% of test-set feature entries:

- Cox: **0.7949**  (Δ = -0.0221)
- RSF: **0.8010**  (Δ = -0.0100)
- DeepSurv: **0.7865**  (Δ = -0.0245)
- Proposed framework: **0.7807**  (Δ = -0.0113)

This indicates that under distributed partial missingness, the proposed framework retained discrimination comparable to the strongest baseline and showed greater stability than Cox and DeepSurv.

## Repository structure

```text
AD_MCI_framework_v3/
├── README.md
├── .gitignore
├── requirements.txt
├── environment.yml
├── data_processed/
│   └── README.md
├── docs/
│   └── data_access_note.md
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
├── scripts/
└── src/
    ├── data/
    ├── evaluation/
    ├── models/
    ├── training/
    ├── utils/
    └── visualization/

    Directory roles
src/data/
Cohort construction, label generation, module building, dataset loading, preprocessing
src/models/
Module embedding, fusion encoder, prediction heads, unified framework model
src/training/
Training logic, losses, and masking strategy
src/evaluation/
Export of test outputs, clustering, Kaplan–Meier analysis, and robustness evaluation
src/utils/
Lightweight utility helpers
src/visualization/
Plotting utilities used for figures
scripts/
Runnable experiment entry points and baseline comparison scripts
results/figures/
Final figure files used in the manuscript
results/tables/
Final summary tables used in the manuscript
results/models/
Lightweight training summaries and preprocessing statistics
Minimal workflow

A typical workflow is:

build ADNI cohort
build labels
build modules
create train/validation/test splits
train the modular framework
export internal test outputs
run external validation
run missingness robustness analyses
Key tables included in this release
Main tables
table1_cohort_characteristics.csv
table2_internal_performance.csv
table3_baseline_comparison.csv
table4_module_missingness_robustness.csv
Supplementary tables
tableS1_ablation.csv
tableS2_cluster_characteristics.csv
tableS3_random30_missingness.csv
Interpretation and scope

This repository should be understood as a research framework for modular clinical progression modeling.

It is not currently:

a deployed clinical decision support tool
a production-ready software package
a claim of universal superiority over conventional complete-data baselines

Instead, it demonstrates the feasibility of:

organizing routine longitudinal data into clinically meaningful modules
explicitly modeling incomplete module availability
generating survival-oriented progression risk
learning patient representations for downstream stratification
transferring a frozen framework to an external cohort
Limitations

Current limitations include:

retrospective study design
imperfect cross-cohort harmonization between ADNI and NACC
incomplete external feature overlap
no claim of complete-data superiority over all simpler baselines
dynamics features based on simple delta engineering may contribute unstable value
no prospective validation or deployment-focused calibration analysis yet
Reproducibility

This public repository includes code, figure files, and summary tables. It does not include raw ADNI/NACC data, patient-level processed derivatives, patient-level embeddings, or full model checkpoints.

To reproduce the full pipeline, users must:

obtain authorized access to ADNI and/or NACC
place source files locally according to the expected structure
rebuild cohorts and modules using the provided scripts
rerun training and evaluation steps
Citation

If you use this code or framework, please cite the associated manuscript once available.
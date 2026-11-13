# Reproducible code for the pancreatic cancer prediction study

## Overview

This repository contains R scripts for training, tuning and evaluating XGBoost models,
producing SHAP explanations and assessing calibration in the held-out
case-control test sample.

**The CPRD patient-level data are not included.** Access to CPRD data is
restricted by the data access agreement and the approved study application.
Researchers wishing to reproduce the analyses must obtain their own authorised
access to the relevant CPRD data and satisfy the applicable governance
requirements.

## Repository structure

```text
project/
  00_project_config.R
  01_model_training_and_evaluation.R
  02_model_evaluation.R (optional evaluation of saved models)
  03_shap_analysis.R
  04_calibration_assessment.R
  data/                 # authorised local datasets; do not publish
  models/               # fitted XGBoost models
  metadata/
    feature_labels.xlsx
  results/              # generated outputs
```

## Input data

The scripts expect one dataset per six-month time-window prior to PC diagnosis, named:

`dataset_0_to_6.txt`, `dataset_1_to_7.txt`, ..., `dataset_18_to_24.txt`

Each dataset should contain the binary outcome (`casInd`, 1 = case, 0 =
control), matched-set identifier (`casenum`), identifiers (`e_patid`,
`e_pracid`) and the predictor variables used in model development.

The trained XGBoost models are saved as JSON files:

`xgboost_model_0_to_6.json`, ..., `xgboost_model_18_to_24.json`

The feature label workbook must contain `variable` and `label` columns.

## Running the scripts

1. Install the required R packages listed in `00_project_config.R`.
2. Obtain authorised access to the CPRD data and construct the analysis
   datasets according to the study protocol.
3. Place the datasets, model files and feature label workbook in the folders
   shown above.
4. Edit `PROJECT_DIR` in `00_project_config.R`.
5. Run the scripts in order:
   - `01_model_training_and_evaluation.R`
   - `02_model_evaluation.R` (if you wish to evaluate previously saved models)
   - `03_shap_analysis.R`
   - `04_calibration_assessment.R`

## Important methodological notes

- Matched sets are assigned wholly to either the training or test set.
- Age is standardised using training-set mean and standard deviation.
- Sex coding and predictor column order must match the original model
  development pipeline.
- Conventional ROC AUC is calculated from continuous predicted probabilities.
  Threshold-specific performance is reported as sensitivity, specificity and
  balanced accuracy; it is not labelled as AUC.
- SHAP values describe the contribution of features to model predictions and
  should not be interpreted as causal effects.
- The calibration script assesses the held-out matched case-control sample.
  Its predicted probabilities are not population-level absolute risks. Population
  calibration requires validation in a representative cohort with observed
  outcomes and an appropriate calibration procedure.

## Scope and reproducibility of this release

This release includes R scripts covering the model-training and evaluation pipeline, 
SHAP-based model interpretation, and calibration assessment. The model-training script 
implements XGBoost hyperparameter tuning using the specified parameter grid and 
five-fold cross-validation, with matched sets retained within the same folds to 
avoid information leakage between training and validation data. Early stopping is 
also incorporated to help limit overfitting.

The scripts are intended to support reproducibility of the modelling and analysis 
procedures. Researchers should use the documented preprocessing, matched-set 
partitioning, hyperparameter settings and model-selection procedures consistently
across prediction windows.

The original CPRD patient-level data cannot be shared publicly because access is 
restricted by the applicable data-use agreement and approved study application. 
Consequently, researchers will need their own authorised access to the relevant 
data to reproduce the analyses. Any differences in data preparation, software versions 
or preprocessing should be documented, as these may affect the resulting models 
and performance estimates.


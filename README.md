# Machine Learning for Preoperative Prediction of Impacted Ureteral Stones

## Overview

This repository contains R scripts supporting the development, validation, and interpretation of machine learning models for the preoperative prediction of impacted ureteral stones in patients undergoing ureteroscopy.

The study used a two-center retrospective cohort design, including 545 patients in the development cohort and 101 patients in the independent external-validation cohort.

Nine candidate prediction algorithms were evaluated using stratified five-times-repeated five-fold nested cross-validation. The random forest (RF) model was selected as the final prediction model and refitted using the complete development cohort before independent external validation.

The final RF model incorporated six predictors:

1. Flank pain
2. Stone attenuation
3. Proximal ureteral width
4. Ureteral wall thickness
5. Stone length
6. Hydronephrosis area

The repository includes scripts for model development, performance assessment, calibration analysis, model interpretation, exploratory analyses, and preparation of publication figures and tables.

## Repository Contents

### 1. Model Development

**`impacted_nested_lasso_9models (2).R`**

Implements the nested cross-validation workflow for comparing nine candidate prediction algorithms, including feature selection, hyperparameter tuning, and out-of-fold performance estimation.

**`impacted_finalize_selected_model.R`**

Fits the selected prediction model using the complete development cohort and prepares the final model for independent external validation.

### 2. Main Figures

**`Figure_2_from_nested_results.R`**

Generates Figure 2, presenting model comparison results derived from nested cross-validation.

**`Figure3_RF_nested_OOF_external.R`**

Generates Figure 3, visualizing the predictive performance of the random forest model using development-cohort out-of-fold predictions and independent external-validation results.

**`Figure4_RF_SHAP_error_analysis.R`**

Generates Figure 4, presenting SHAP-based feature importance, individual model explanations, and descriptive analyses of external-validation prediction errors.

### 3. Main Tables

**`Table1_S1_development_and_Table1_six_predictors.R`**

Generates Table 1 and Supplementary Table S1, summarizing baseline characteristics and the distribution of candidate and final predictors.

**`Table2.R`**

Generates Table 2, reporting comparative performance estimates for the candidate machine learning algorithms evaluated during model development.

**`Table3_final_RF_performance.R`**

Generates Table 3, summarizing the predictive performance of the final random forest model, including discrimination, calibration, and classification metrics.

### 4. Supplementary Figures

**`FigureS1_RF_feature_selection_publication.R`**

Generates Supplementary Figure S1, illustrating the feature-selection results used to support model development.

**`FigureS3_RF_exploratory_RCS.R`**

Generates Supplementary Figure S3, presenting exploratory restricted cubic spline analyses to investigate potential nonlinear relationships between selected continuous predictors and stone impaction.

### 5. Supplementary Tables and Additional Analyses

**`Table_S2_nine_algorithms_with_calibration_CI_fixed.R`**

Generates Supplementary Table S2, providing additional comparative performance and calibration estimates with confidence intervals for the nine candidate algorithms.

**`table_S4_ICC.R`**

Calculates intraclass correlation coefficients (ICCs) to assess the reproducibility of imaging measurements.

**`Table_S5_external_error_cases.R`**

Generates Supplementary Table S5, summarizing misclassified patients in the external-validation cohort and supporting descriptive analyses of false-positive and false-negative predictions.

### 6. Software and Package Versions

**`R_package_versions.csv`**

Provides a record of installed R packages and their corresponding versions to document the computational environment.

The repository contains R scripts for statistical analysis, machine learning model development, validation, and visualization. Package requirements can be identified from the corresponding scripts.

## Model Validation Strategy

Model development and internal validation were conducted using stratified five-times-repeated five-fold nested cross-validation.

Feature selection, preprocessing, and hyperparameter tuning were performed within the appropriate training partitions to reduce information leakage.

After model selection, the final random forest model was refitted using all 545 development-cohort patients and evaluated in the independent external-validation cohort of 101 patients.

The external-validation cohort was not used for model development, feature selection, hyperparameter optimization, model updating, or threshold selection.

A fixed probability threshold of 0.50 was used for descriptive classification performance assessment.

## Model Interpretation

SHapley Additive exPlanations (SHAP) were used to investigate feature contributions to predicted probabilities of ureteral stone impaction.

Global and local SHAP analyses were performed to describe model behavior. External-validation prediction errors were further explored using predicted probabilities and representative individual explanations.

These post-hoc analyses describe model predictions and should not be interpreted as evidence of causal relationships.

## Data Availability

Individual-level clinical and imaging data are not publicly available in this repository because of patient confidentiality and institutional data-sharing restrictions.

The analytical scripts are provided to support methodological transparency. Reproduction of the original numerical results requires access to the underlying study datasets and corresponding analytical inputs.

## Intended Use

The repository accompanies a clinical prediction modeling study and is intended for research, methodological review, and reproducibility assessment.

The prediction model has not been established as a standalone clinical decision-making tool and requires further evaluation before routine clinical implementation.

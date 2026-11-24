# Jupyter Notebooks

This directory contains the Jupyter notebooks used for data preparation,
exploratory data analysis (EDA), and machine learning model development
for the UMBC DATA606 Capstone Project.

## Notebook Overview

| Notebook | Description |
|---|---|
| [nhanes_data_preparation.ipynb](nhanes_data_preparation.ipynb) | NHANES data collection, integration, cleaning, feature engineering, and preparation of the analytical dataset. |
| [nhanes_eda.ipynb](nhanes_eda.ipynb) | Exploratory data analysis, descriptive statistics, missing-value analysis, predictor distributions, and relationships with elevated HbA1c. |
| [nhanes_modeling.ipynb](nhanes_modeling.ipynb) | Machine learning model development, cross-validation, model comparison, threshold selection, final evaluation, feature importance, and model export. |

## Recommended Execution Order

### 1. Data Preparation

**Notebook:** `nhanes_data_preparation.ipynb`

- Collect NHANES data from four survey cycles (2011–2018).
- Merge relevant NHANES data components.
- Apply participant eligibility criteria.
- Clean and standardize variables.
- Perform feature engineering.
- Define the elevated HbA1c target variable.
- Prepare and export the analytical dataset.

### 2. Exploratory Data Analysis

**Notebook:** `nhanes_eda.ipynb`

- Examine dataset structure and descriptive statistics.
- Analyze missing values.
- Examine HbA1c distribution and target class balance.
- Explore demographic, anthropometric, behavioral,
  cardiovascular, and socioeconomic characteristics.
- Investigate relationships between predictors and elevated HbA1c.
- Generate supporting visualizations.

### 3. Machine Learning Modeling

**Notebook:** `nhanes_modeling.ipynb`

- Prepare training and held-out test datasets.
- Develop preprocessing pipelines.
- Establish an Age + BMI Logistic Regression baseline.
- Develop and compare Logistic Regression, Random Forest,
  and Gradient Boosting models.
- Perform cross-validation and hyperparameter tuning.
- Select the classification threshold.
- Evaluate the final model on held-out test data.
- Analyze predictor importance.
- Evaluate the contribution of behavioral and socioeconomic predictors.
- Evaluate model performance among non-obese adults.
- Address the four research questions.
- Export and verify the final model for Streamlit deployment.

## Directory Structure

    notebooks/
    ├── README.md
    ├── nhanes_data_preparation.ipynb
    ├── nhanes_eda.ipynb
    └── nhanes_modeling.ipynb

## Final Model

| Component | Description |
|---|---|
| Selected Model | Expanded Logistic Regression |
| Number of Predictors | 13 |
| Target Variable | Elevated HbA1c (HbA1c ≥ 5.7%) |
| Classification Threshold | 0.29 |
| Model Artifact | `elevated_hba1c_logistic.joblib` |

## Requirements

Refer to [requirements.txt](../requirements.txt) for project dependencies.

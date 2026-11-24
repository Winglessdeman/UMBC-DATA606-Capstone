# UMBC-DATA606-Capstone

# NHANES HbA1c Screening — UMBC DATA606 Capstone

**Identifying Hidden Glycemic Risk Using Machine Learning: An Analysis of U.S. Adults Without Diagnosed Diabetes**

This capstone evaluates whether non-laboratory demographic, socioeconomic, anthropometric, behavioral, and health-history information can help identify elevated HbA1c among U.S. adults without a previously reported diabetes diagnosis. It uses NHANES data from 2011–2018 and compares predictive models for HbA1c ≥ 5.7%.

The analysis includes 17,156 adults and evaluates 13 non-laboratory predictors. The deployed application demonstrates a model-based screening estimate for educational purposes; it is **not a medical diagnosis or a substitute for laboratory testing**.

## Project resources

- **[Project report](docs/report.md)** 
- **[EDA notebook](notebooks/nhanes_eda.ipynb)** 
- **[Modeling notebook](notebooks/nhanes_modeling.ipynb)** 
- **[Streamlit application](https://nhanes-hba1c-screening.streamlit.app/)**
- **[Final presentation](https://docs.google.com/presentation/d/1RcuCA3hMLdgMUhZl3_jZcrc8IVwGSDwQ/edit)**

## Research questions

1. Can non-laboratory characteristics identify elevated HbA1c among adults without a previously reported diabetes diagnosis?
2. Which non-laboratory characteristics are most important?
3. Do behavioral and socioeconomic characteristics improve prediction beyond age and BMI?
4. Can elevated HbA1c be identified among adults who are not obese?

## Dataset at a glance

| Characteristic | Value |
|---|---:|
| Survey cycles | 2011–2012, 2013–2014, 2015–2016, 2017–2018 |
| Participants | 17,156 |
| Analysis-ready variables | 32 |
| Participants with HbA1c ≥ 5.7% | 5,463 (31.84%) |
| Participants with HbA1c < 5.7% | 11,693 (68.16%) |


## Notes

NHANES is a complex survey. Descriptive figures presented here are sample-based unless explicitly identified as survey-weighted population estimates. Relationships identified through EDA are associations, not evidence of causation.

**Author:** Shristi Pokharel · UMBC Master in Applied Data Science · DATA606 Capstone

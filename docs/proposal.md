# Identifying Hidden Glycemic Risk using Machine Learning: An Analysis of U.S. Adults without diagnosed diabetes. 

- **Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang 
- **Author**: Shristi Pokharel
- **GitHub repository**: https://github.com/winglessdeman/UMBC-DATA606-Capstone
- **LinkedIn:** 
- **PowerPoint Presentation:** 
- **YouTube Presentation:** 

# 2. Background

### What is it about?
Diabetes and prediabetes are usually evaluated using HbA1c : a laboratory measurement that reflects average blood glucose over time. 
Some individuals might have elevated HbA1c even though they have not been officially diagnosed with diabetes.
This project uses data from the  (NHANES) National Health and Nutrition Examination Survey that is conducted by the (CDC)
National Center for Health Statistics within the Centers for Disease Control and Prevention. 
NHANES combines interviews, physical examinations and laboratory testing thus providing demographic, socioeconomic, 
anthropometric, behavioral, cardiovascular, and laboratory information for the U.S participants.

In this project I will focus on adults without a previously reported diabetes diagnosis. 
Laboratory-measured HbA1c will be used as the outcome source, the machine learning predictors 
will be limited primarily to variables that are available without performing HbA1c laboratory test. 
This project will investigate whether non-laboratory characteristics can be used to distinguish 
participants with normal HbA1c from the participants that have elevated HbA1c. 

### Why does it matter?

HbA1c testing can identify elevated glycemic levels, but it requires a laboratory test.
In this project I explore whether the commonly available non-laboratory information such as age, BMI, waist circumference, physical activity, smoking behavior, sleep duration, income, education, hypertension, and high cholesterol can help to identify patterns associated with elevated HbA1c. If these characteristics can successfully distinguish individuals with elevated HbA1c, then the machine-learning model could demonstrate how routinely available health and lifestyle information may be used to support screening decisions and identify individuals who may benefit from further laboratory testing . Although the model cannot be used to diagnose diabetes or replace HbA1c testing it can be a risk-screening and educational tool for individuals wanting to make changes in their lifestyle and seek medical attention.   

For the final Streamlit application, I would design it as a simple glycemic-risk screening demonstration where an user would enter non-laboratory information similar to the variables used in the model, and the app would return the model's estimated likelihood/classification of elevated HbA1c.


### What are your research questions?

#### Research Question 1
**Can non-laboratory demographic, Biometric, behavioral, cardiovascular, and socioeconomic characteristics be used to identify elevated HbA1c among adults without a previously reported diabetes diagnosis?**


#### Research Question 2
**Which non-laboratory characteristics are most important for identifying elevated HbA1c?**

#### Research Question 3
**Do behavioral and socioeconomic characteristics improve the prediction of elevated HbA1c beyond basic factors such as age and BMI?**

#### Research Question 4
**Can elevated HbA1c be identified among adults who are not obese?**

# 3. Data

### Data Source: 
The project uses publicly available data from the CDC/NCHS National Health and Nutrition Examination Survey (NHANES). The data and documentation are available from the following links:

- NHANES website: https://www.cdc.gov/nchs/nhanes/
- NHANES data search and documentation: https://wwwn.cdc.gov/nchs/nhanes/

### Data size:
The project uses 32 SAS Transport (`.xpt`) files: eight NHANES components
from each of four survey cycles. The combined raw file size is approximately 65.20 MB.

### Data shape:
- After combining the four NHANES cycles and merging the eight components using
`SEQN`, the working merged dataset contains:

- **39,156 rows**
- **30 columns**
- **39,156 unique participants**
- **0 duplicate participant identifiers**

For the primary analysis, the study population is restricted to participants who:

- are age 20 years or older;
- have a valid laboratory HbA1c measurement; and
- reported no previous diabetes diagnosis (DIQ010 = 2).

After applying these criteria, the primary analysis population contains
**17,156 participants**.

Participants reporting diagnosed diabetes (DIQ010 = 1), borderline diabetes
(DIQ010 = 3), or missing/unknown diabetes status are excluded from the
primary analysis.


### Time Period:
It includes four continuous pre-pandemic NHANES survey cycles. 

- 2011–2012
- 2013–2014
- 2015–2016
- 2017–2018

### What does each row represent?
- Each row represents the unique participant identifiers from the survey cycle.
- **SEQN**  is used as the unique participant identifier and merge key across the eight data components.


### NHANES Data Components:

| Component | Description | Main Use in Project |
|---|---|---|
| `DEMO` | Demographics | Age, sex, race/ethnicity, education, income, and survey-design variables |
| `GHB` | Glycohemoglobin | Laboratory-measured HbA1c and source of the ML target |
| `BMX` | Body Measures | BMI, waist circumference, and body weight |
| `DIQ` | Diabetes Questionnaire | Identification of previously reported diabetes |
| `PAQ` | Physical Activity | Recreational activity and sedentary behavior |
| `SMQ` | Smoking - Cigarette Use | Lifetime and current smoking behavior |
| `SLQ` | Sleep Disorders | Usual sleep duration |
| `BPQ` | Blood Pressure & Cholesterol Questionnaire | Self-reported hypertension and high cholesterol |

### Data dictionary:

| Column                               | Type        | Definition                             | Potential Values / Units       | Role               |
| ------------------------------------ | ----------- | -------------------------------------- | ------------------------------ | ------------------ |
| `age`                                | Numeric     | Participant age                        | Years, 20–80 in primary sample | Predictor          |
| `sex`                                | Categorical | Participant sex                        | NHANES coded categories        | Predictor          |
| `race_ethnicity`                     | Categorical | Race/Hispanic origin                   | NHANES coded categories        | Predictor          |
| `education`                          | Categorical | Adult educational attainment           | NHANES education categories    | Predictor          |
| `income_poverty_ratio`               | Numeric     | Family income-to-poverty ratio         | 0–5                            | Predictor          |
| `hba1c`                              | Numeric     | Laboratory HbA1c                       | Percent                        | Target source      |
| `bmi`                                | Numeric     | Body Mass Index                        | kg/m²                          | Predictor          |
| `waist_cm`                           | Numeric     | Waist circumference                    | cm                             | Predictor          |
| `weight_kg`                          | Numeric     | Body weight                            | kg                             | Possible predictor |
| `diabetes_report`                    | Categorical | Self-reported diabetes diagnosis       | 1 Yes, 2 No, 3 Borderline      | Eligibility        |
| `vigorous_minutes_week`              | Numeric     | Derived vigorous recreational activity | Minutes/week                   | Predictor          |
| `moderate_minutes_week`              | Numeric     | Derived moderate recreational activity | Minutes/week                   | Predictor          |
| `recreational_activity_minutes_week` | Numeric     | Total recreational activity            | Minutes/week                   | Predictor          |
| `sedentary_minutes`                  | Numeric     | Sedentary time                         | Minutes/day                    | Predictor          |
| `smoking_status`                     | Categorical | Derived smoking status                 | Never, Former, Current         | Predictor          |
| `sleep_hours`                        | Numeric     | Weekday/workday sleep duration         | Hours                          | Predictor          |
| `hypertension`                       | Categorical | Reported hypertension                  | Yes/No                         | Predictor          |
| `high_cholesterol`                   | Categorical | Reported high cholesterol              | Yes/No                         | Predictor          |


### Target/label:
The machine-learning target will be derived from:

`LBXGH` is the laboratory-measured HbA1c percentage.

The proposed primary binary target is:

| Target Class | HbA1c |
|---|---|
| Normal | `< 5.7%` |
| Elevated | `>= 5.7%` |

The project will use HbA1c as the outcome to investigate hidden glycemic risk among adults who do not report diagnosed diabetes. An HbA1c value alone will not be treated as a clinical diagnosis.

Within the primary analysis population of 17,156 participants, 11,693
(68.16%) have normal HbA1c and 5,463 (31.84%) have elevated HbA1c.

### Features/predictors:

Potential predictors are grouped by domain:

#### Demographic

- Age
- Sex
- Race/ethnicity

#### Socioeconomic

- Education
- Family income-to-poverty ratio

#### Anthropometric

- BMI
- Waist circumference
- Body weight

#### Behavioral

- Vigorous recreational activity
- Moderate recreational activity
- Total recreational activity
- Sedentary time
- Lifetime smoking history
- Current smoking status
- Sleep duration

#### Cardiovascular / Health History

- Self-reported hypertension
- Self-reported high cholesterol

`DIQ010` (Doctor Told You Have Diabetes) is primarily used as an eligibility
variable to identify participants with a previously reported diabetes diagnosis.

`WTMEC2YR` (2-Year MEC Examination Weight), `SDMVPSU` (Masked Variance
Pseudo-Primary Sampling Unit), and `SDMVSTRA` (Masked Variance Pseudo-Stratum)
are NHANES survey-design variables. They are retained for survey-aware
descriptive analysis and are not intended to be used as ordinary
machine-learning predictors.

The final predictor set will be determined after exploratory data analysis, missing-data assessment, 
evaluation of redundancy among variables, and review of model performance.

Official CDC/NHANES documentation for each component is recorded in the accompanying Jupyter Notebook, 
including cycle-specific codebooks for DEMO, GHB, BMX, DIQ, PAQ, SMQ, SLQ, and BPQ.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Elevated HbA1c Risk Screening",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. FILE PATHS
# ============================================================

# Application directory
APP_DIR = Path(__file__).resolve().parent

# Project root directory
PROJECT_DIR = APP_DIR.parent

# Visualization directory
ASSETS_DIR = APP_DIR / "assets"

# Model directory
MODELS_DIR = PROJECT_DIR / "models"

# Saved model artifact
MODEL_PATH = MODELS_DIR / "elevated_hba1c_logistic.joblib"


# ============================================================
# INPUT CATEGORY MAPPINGS
# ============================================================
# Study eligibility response options
DIABETES_HISTORY_OPTIONS = [
    "Yes",
    "No",
    "Don't know",
    "Prefer not to answer"
]

# Age options
AGE_OPTIONS = {
    str(age): float(age)
    for age in range(20, 80)
}

AGE_OPTIONS["80 or older"] = 80.0


# Sex categories
SEX_OPTIONS = {
    "Male": 1.0,
    "Female": 2.0
}


# Race/Ethnicity categories
RACE_ETHNICITY_OPTIONS = {
    "Mexican American": 1.0,
    "Other Hispanic": 2.0,
    "Non-Hispanic White": 3.0,
    "Non-Hispanic Black": 4.0,
    "Non-Hispanic Asian": 6.0,
    "Other Race, Including Multiracial": 7.0
}

# Cardiovascular response categories
CARDIOVASCULAR_OPTIONS = {
    "Yes": 1.0,
    "No": 2.0,
    "Don't know": np.nan,
    "Prefer not to answer": np.nan
}

# Smoking status categories
SMOKING_OPTIONS = {
    "Never smoker": "Never",
    "Former smoker": "Former",
    "Current smoker": "Current",
    "Don't know": np.nan,
    "Prefer not to answer": np.nan
}


# Physical activity response options
ACTIVITY_OPTIONS = [
    "Yes",
    "No",
    "Don't know",
    "Prefer not to answer"
]

# ============================================================
# SOCIOECONOMIC CATEGORY MAPPINGS
# ============================================================

# Education categories
EDUCATION_OPTIONS = {
    "Less than 9th grade": 1.0,
    "9th–12th grade, no diploma": 2.0,
    "High school graduate or GED": 3.0,
    "Some college or associate degree": 4.0,
    "College graduate or above": 5.0,
    "Don't know": np.nan,
    "Prefer not to answer": np.nan
}


# NHANES annual family income ranges
# Bounded ranges are represented by rounded midpoints

INCOME_OPTIONS = {
    "$0–$4,999": 2500.0,
    "$5,000–$9,999": 7500.0,
    "$10,000–$14,999": 12500.0,
    "$15,000–$19,999": 17500.0,
    "$20,000–$24,999": 22500.0,
    "$25,000–$34,999": 30000.0,
    "$35,000–$44,999": 40000.0,
    "$45,000–$54,999": 50000.0,
    "$55,000–$64,999": 60000.0,
    "$65,000–$74,999": 70000.0,
    "$75,000–$99,999": 87500.0,
    "$100,000 and above": None,
    "Don't know": np.nan,
    "Prefer not to answer": np.nan
}


# Additional income categories for the application
HIGH_INCOME_OPTIONS = {
    "$100,000–$124,999": 112500.0,
    "$125,000–$149,999": 137500.0,
    "$150,000–$199,999": 175000.0,
    "$200,000–$249,999": 225000.0,
    "$250,000–$499,999": 375000.0,
    "$500,000 and above": 500000.0
}


# Federal poverty guideline regions
REGION_OPTIONS = [
    "48 contiguous states and Washington, D.C.",
    "Alaska",
    "Hawaii"
]


# Official 2026 HHS poverty guidelines
POVERTY_GUIDELINE_YEAR = 2026

POVERTY_GUIDELINES = {
    "48 contiguous states and Washington, D.C.": {
        "base": 15960,
        "additional_person": 5680
    },
    "Alaska": {
        "base": 19950,
        "additional_person": 7100
    },
    "Hawaii": {
        "base": 18360,
        "additional_person": 6530
    }
}

# Sleep duration options (half-hour increments)
# Sleep duration options: 2–14.5 hours
SLEEP_OPTIONS = {
    f"{hours / 2:g} hours": float(hours / 2)
    for hours in range(4,30)
}

SLEEP_OPTIONS["Don't know"] = np.nan
SLEEP_OPTIONS["Prefer not to answer"] = np.nan

# ============================================================
# MODEL INPUT SPECIFICATIONS
# ============================================================

NUMERIC_FEATURES = [
    "age",
    "income_poverty_ratio",
    "bmi",
    "waist_cm",
    "recreational_activity_minutes_week",
    "sedentary_minutes",
    "sleep_hours"
]


CATEGORICAL_FEATURES = [
    "sex",
    "race_ethnicity",
    "education",
    "smoking_status",
    "hypertension",
    "high_cholesterol"
]


# Human-readable predictor names
FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "income_poverty_ratio": "Income-to-Poverty Ratio",
    "bmi": "Body Mass Index (BMI)",
    "waist_cm": "Waist Circumference",
    "recreational_activity_minutes_week": "Recreational Physical Activity",
    "sedentary_minutes": "Sedentary Time",
    "sleep_hours": "Sleep Duration",
    "sex": "Sex",
    "race_ethnicity": "Race / Ethnicity",
    "education": "Education Level",
    "smoking_status": "Smoking Status",
    "hypertension": "High Blood Pressure",
    "high_cholesterol": "High Cholesterol"
}

# ============================================================
# Helper function for displaying saved visualizations
# ============================================================

def display_visualization(filename):

    image_path = ASSETS_DIR / filename

    if image_path.is_file():

        st.image(
            str(image_path),
            width="stretch"
        )

    else:

        st.error(
            f"Visualization not found: {filename}"
        )

        st.caption(
            f"Expected file location: {image_path}"
        )


# ============================================================
# MODEL INPUT DATAFRAME CONSTRUCTION
# ============================================================

def build_model_input(input_values, features):

    # Check for missing predictor names
    missing_features = set(features) - set(input_values.keys())

    if missing_features:

        raise ValueError(
            f"Missing required predictors: {missing_features}"
        )

    # Check for unexpected predictor names
    unexpected_features = (
        set(input_values.keys()) - set(features)
    )

    if unexpected_features:

        raise ValueError(
            f"Unexpected predictors: {unexpected_features}"
        )

    # Create one-row DataFrame in the exact model feature order
    input_df = pd.DataFrame(
        [input_values],
        columns=features
    )

    # Convert numerical predictors to float64
    for column in NUMERIC_FEATURES:

        input_df[column] = (
            pd.to_numeric(
                input_df[column],
                errors="raise"
            )
            .astype("float64")
        )

    # Preserve categorical predictors as object
    for column in CATEGORICAL_FEATURES:

        input_df[column] = input_df[column].astype("object")

    # Validate DataFrame shape
    if input_df.shape != (1, len(features)):

        raise ValueError(
            "Unexpected model input dimensions."
        )

    # Validate numerical values
    numeric_values = input_df[
        NUMERIC_FEATURES
    ].to_numpy(dtype=float)

    if np.isinf(numeric_values).any():

        raise ValueError(
            "Numerical inputs cannot contain infinite values."
        )

    return input_df

# ============================================================
# MODEL INPUT VALIDATION
# ============================================================

def validate_model_inputs(input_values):

    errors = []

    # --------------------------------------------------------
    # Age eligibility
    # --------------------------------------------------------

    age_value = input_values["age"]

    if pd.isna(age_value):

        errors.append(
            "Age must be provided."
        )

    elif not 20 <= age_value <= 80:

        errors.append(
            "Age must fall within the supported age categories."
        )

    # --------------------------------------------------------
    # Required demographic values
    # --------------------------------------------------------

    if pd.isna(input_values["sex"]):

        errors.append(
            "Sex must be selected."
        )

    if pd.isna(input_values["race_ethnicity"]):

        errors.append(
            "Race / Ethnicity must be selected."
        )

    # --------------------------------------------------------
    # BMI validation
    # --------------------------------------------------------

    bmi_value = input_values["bmi"]

    if pd.notna(bmi_value) and bmi_value <= 0:

        errors.append(
            "BMI must be greater than zero."
        )

    # --------------------------------------------------------
    # Waist circumference validation
    # --------------------------------------------------------

    waist_value = input_values["waist_cm"]

    if pd.notna(waist_value) and waist_value <= 0:

        errors.append(
            "Waist circumference must be greater than zero."
        )

    # --------------------------------------------------------
    # Income-to-poverty ratio validation
    # --------------------------------------------------------

    pir_value = input_values["income_poverty_ratio"]

    if pd.notna(pir_value):

        if not 0 <= pir_value <= 5:

            errors.append(
                "Income-to-poverty ratio must be between "
                "0 and 5."
            )

    # --------------------------------------------------------
    # Sleep duration validation
    # --------------------------------------------------------

    sleep_value = input_values["sleep_hours"]

    if pd.notna(sleep_value):

        if not 2 <= sleep_value <= 14.5:

            errors.append(
                "Sleep duration must be between "
                "2 and 14.5 hours."
            )

    # --------------------------------------------------------
    # Sedentary time validation
    # --------------------------------------------------------

    sedentary_value = input_values["sedentary_minutes"]

    if pd.notna(sedentary_value):

        if not 0 <= sedentary_value <= 1440:

            errors.append(
                "Sedentary time must be between "
                "0 and 1,440 minutes per day."
            )

    # --------------------------------------------------------
    # Recreational activity validation
    # --------------------------------------------------------

    activity_value = input_values[
        "recreational_activity_minutes_week"
    ]

    if pd.notna(activity_value):

        if not 0 <= activity_value <= 10080:

            errors.append(
                "Weekly recreational activity must be "
                "between 0 and 10,080 minutes."
            )

    # --------------------------------------------------------
    # Sleep and sedentary time consistency
    # --------------------------------------------------------

    if (
        pd.notna(sleep_value)
        and pd.notna(sedentary_value)
    ):

        sleep_minutes = sleep_value * 60

        if sleep_minutes + sedentary_value > 1440:

            errors.append(
                "Combined sleep duration and sedentary time "
                "cannot exceed 24 hours per day."
            )

    # --------------------------------------------------------
    # Validate categorical values
    # --------------------------------------------------------

    valid_categories = {
        "sex": [1.0, 2.0],
        "race_ethnicity": [1.0, 2.0, 3.0, 4.0, 6.0, 7.0],
        "education": [1.0, 2.0, 3.0, 4.0, 5.0],
        "smoking_status": ["Never", "Former", "Current"],
        "hypertension": [1.0, 2.0],
        "high_cholesterol": [1.0, 2.0]
    }

    for feature, allowed_values in valid_categories.items():

        value = input_values[feature]

        if pd.notna(value) and value not in allowed_values:

            errors.append(
                f"Invalid response for "
                f"{FEATURE_DISPLAY_NAMES[feature]}."
            )

    return errors

# ============================================================
# MODEL PREDICTION FUNCTION
# ============================================================

def generate_screening_prediction(
    model,
    model_input,
    threshold
):

    # Validate the classification threshold
    if not 0 <= threshold <= 1:

        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    # Generate predicted probabilities
    probabilities = model.predict_proba(
        model_input
    )[0]

    # Identify the positive class
    positive_class_index = list(
        model.classes_
    ).index(1)

    # Extract probability of elevated HbA1c
    elevated_probability = float(
        probabilities[positive_class_index]
    )

    # Validate probability
    if not np.isfinite(elevated_probability):

        raise ValueError(
            "The model returned an invalid probability."
        )

    if not 0 <= elevated_probability <= 1:

        raise ValueError(
            "Predicted probability must be between 0 and 1."
        )

    # Calculate probability of normal HbA1c
    normal_probability = 1 - elevated_probability

    # Apply saved classification threshold
    prediction = int(
        elevated_probability >= threshold
    )

    # Return prediction results
    return {
        "elevated_probability": elevated_probability,
        "normal_probability": normal_probability,
        "prediction": prediction,
        "threshold": threshold
    }

# ============================================================
# SCREENING PROBABILITY VISUALIZATION
# ============================================================

def display_probability_chart(probability, threshold):

    fig, ax = plt.subplots(figsize=(10, 2.5))

    # Below-threshold region
    ax.axvspan(
        0,
        threshold,
        color="#4880b8",
        alpha=0.25,
        label="Below Screening Threshold"
    )

    # At-or-above-threshold region
    ax.axvspan(
        threshold,
        1,
        color="#f28e2b",
        alpha=0.25,
        label="At or Above Screening Threshold"
    )

    # Screening threshold
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5
    )

    # Predicted probability marker
    ax.scatter(
        probability,
        0.5,
        color="#333333",
        s=150,
        zorder=5
    )

    # Probability annotation
    ax.text(
        probability,
        0.78,
        f"{probability:.2%}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

    # Threshold annotation
    ax.text(
        threshold,
        0.12,
        f"Threshold: {threshold:.0%}",
        ha="center",
        fontsize=10
    )

    # Axis formatting
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)

    ax.set_xticks(
        [0, 0.25, 0.50, 0.75, 1.00]
    )

    ax.set_xticklabels(
        ["0%", "25%", "50%", "75%", "100%"]
    )

    ax.set_yticks([])

    ax.set_xlabel(
        "Model-Estimated Probability of Elevated HbA1c",
        fontsize=11
    )

    ax.set_title(
        "Estimated Probability and Screening Threshold",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    # Remove unnecessary borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        fontsize=9,
        frameon=False
    )

    fig.tight_layout()

    # Display chart in Streamlit
    st.pyplot(fig)

    # Close figure to release memory
    plt.close(fig)

# ============================================================
# MODEL LOADING FUNCTION
# ============================================================

@st.cache_resource
def load_model_artifact():

    # Check whether the model file exists
    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Load the saved model artifact
    artifact = joblib.load(MODEL_PATH)

    # Validate required artifact keys
    required_keys = {
        "model",
        "threshold",
        "features",
        "target_labels",
        "model_name",
        "target_definition"
    }

    if not isinstance(artifact, dict):

        raise TypeError(
            "The saved model artifact must be a dictionary."
        )

    missing_keys = required_keys - set(artifact.keys())

    if missing_keys:

        raise ValueError(
            f"Missing artifact keys: {missing_keys}"
        )

    return artifact

# ============================================================
# ANTHROPOMETRIC CALCULATION FUNCTIONS
# ============================================================

def convert_height_to_cm(feet, inches):

    return (feet * 12 + inches) * 2.54


def convert_weight_to_kg(weight_lbs):

    return weight_lbs * 0.45359237


def convert_inches_to_cm(inches):

    return inches * 2.54


def calculate_bmi(weight_kg, height_cm):

    # Handle missing or invalid measurements
    if pd.isna(weight_kg) or pd.isna(height_cm):
        return np.nan

    if weight_kg <= 0 or height_cm <= 0:
        return np.nan

    # Convert height from centimeters to meters
    height_m = height_cm / 100

    # Calculate BMI
    bmi = weight_kg / (height_m ** 2)

    return bmi

# ============================================================
# BEHAVIORAL CALCULATION FUNCTIONS
# ============================================================

def calculate_sedentary_minutes(hours, minutes):

    if hours is None or minutes is None:
        return np.nan

    total_minutes = hours * 60 + minutes

    if total_minutes < 0 or total_minutes > 1440:
        return np.nan

    return float(total_minutes)


def calculate_weekly_activity(response, days, minutes):

    # No participation
    if response == "No":
        return 0.0

    # Missing information
    if response != "Yes":
        return np.nan

    if days is None or minutes is None:
        return np.nan

    if not 1 <= days <= 7:
        return np.nan

    if not 10 <= minutes <= 1440:
        return np.nan

    return float(days * minutes)

# ============================================================
# SOCIOECONOMIC CALCULATION FUNCTIONS
# ============================================================

def calculate_poverty_guideline(family_size, region):

    # Validate inputs
    if family_size is None or region is None:
        return np.nan

    if family_size < 1:
        return np.nan

    if region not in POVERTY_GUIDELINES:
        return np.nan

    # Retrieve regional poverty guidelines
    guideline = POVERTY_GUIDELINES[region]

    # Calculate guideline based on family size
    poverty_guideline = (
        guideline["base"]
        + (family_size - 1)
        * guideline["additional_person"]
    )

    return float(poverty_guideline)


def calculate_income_poverty_ratio(
    estimated_income,
    family_size,
    region
):

    # Handle missing income
    if pd.isna(estimated_income):
        return np.nan

    if estimated_income < 0:
        return np.nan

    # Calculate applicable poverty guideline
    poverty_guideline = calculate_poverty_guideline(
        family_size,
        region
    )

    if pd.isna(poverty_guideline):
        return np.nan

    if poverty_guideline <= 0:
        return np.nan

    # Calculate income-to-poverty ratio
    pir = estimated_income / poverty_guideline

    # Cap the ratio at 5.00 following NHANES conventions
    pir = min(pir, 5.0)

    return round(pir, 2)

# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("NHANES HbA1c")

st.sidebar.caption("Non-Laboratory Risk Screening")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Research Findings",
        "HbA1c Risk Screening"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("**UMBC DATA606 Capstone**")

st.sidebar.caption(
    "Master of Professional Studies in Data Science"
)

st.sidebar.caption("NHANES 2011–2018")


# ============================================================
# 4. PROJECT OVERVIEW PAGE
# ============================================================

if page == "Project Overview":

    # --------------------------------------------------------
    # 4.1 Project Title
    # --------------------------------------------------------

    st.title("Elevated HbA1c Risk Screening")

    st.subheader(
        "Identifying Elevated HbA1c Using "
        "Non-Laboratory Characteristics"
    )

    st.caption(
        "UMBC DATA606 Capstone Project | "
        "Master of Professional Studies in Data Science"
    )

    st.divider()

    # --------------------------------------------------------
    # 4.2 Key Project Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "NHANES Survey Cycles",
        "4",
        help="Four NHANES survey cycles from 2011 to 2018."
    )

    col2.metric(
        "Non-Laboratory Predictors",
        "13",
        help="Number of predictors in the final Expanded model."
    )

    col3.metric(
        "Test ROC-AUC",
        "0.777",
        help=(
            "Measures the model's ability to distinguish "
            "between normal and elevated HbA1c."
        )
    )

    col4.metric(
        "Test Recall",
        "78.3%",
        help=(
            "Percentage of participants with elevated HbA1c "
            "correctly identified by the final model."
        )
    )

    st.divider()

    # --------------------------------------------------------
    # 4.3 Project Background
    # --------------------------------------------------------

    st.header("Project Background")

    st.write(
        """
        Hemoglobin A1c (HbA1c) reflects average blood glucose
        levels over approximately the previous two to three
        months and is commonly used to assess glucose regulation.

        Some adults may have elevated HbA1c without a previously
        reported diabetes diagnosis. Identifying these individuals
        may help support earlier screening and further clinical
        evaluation.

        This project investigates whether routinely available
        non-laboratory characteristics can be used to identify
        adults who may have elevated HbA1c without requiring
        laboratory measurements as model inputs.
        """
    )

    # --------------------------------------------------------
    # 4.4 Project Objective
    # --------------------------------------------------------

    st.header("Project Objective")

    st.write(
        """
        The objective of this project is to develop and evaluate
        a machine learning model that identifies elevated HbA1c
        among adults without a previously reported diabetes
        diagnosis using demographic, anthropometric, behavioral,
        cardiovascular, and socioeconomic characteristics.
        """
    )

    # --------------------------------------------------------
    # 4.5 Data Source
    # --------------------------------------------------------

    st.header("Data Source")

    st.markdown(
        """
        This study uses publicly available data from the
        **National Health and Nutrition Examination Survey
        (NHANES)**, conducted by the National Center for
        Health Statistics (NCHS) within the Centers for
        Disease Control and Prevention (CDC).
        """
    )

    st.subheader("Selected Survey Cycles")

    cycle1, cycle2, cycle3, cycle4 = st.columns(4)

    cycle1.metric("Cycle G", "2011–12")
    cycle2.metric("Cycle H", "2013–14")
    cycle3.metric("Cycle I", "2015–16")
    cycle4.metric("Cycle J", "2017–18")

    st.markdown(
        "[Visit the official NHANES website]"
        "(https://www.cdc.gov/nchs/nhanes/)"
    )

    # Selected NHANES components

    with st.expander("Selected NHANES Data Components"):

        st.markdown(
            """
            | Component | Description |
            |---|---|
            | DEMO | Demographic Characteristics |
            | GHB | Glycohemoglobin (HbA1c) |
            | BMX | Body Measurements |
            | DIQ | Diabetes Questionnaire |
            | PAQ | Physical Activity Questionnaire |
            | SMQ | Smoking Questionnaire |
            | SLQ | Sleep Questionnaire |
            | BPQ | Blood Pressure and Cholesterol |
            """
        )

    # --------------------------------------------------------
    # 4.6 Study Population
    # --------------------------------------------------------

    st.header("Study Population")

    st.markdown(
        """
        The study focuses on adults from the selected NHANES
        survey cycles who:

        - Meet the study's adult age eligibility criteria.
        - Have a valid measured HbA1c value.
        - Do not have a previously reported diabetes diagnosis.

        A subgroup of adults with BMI below 30 kg/m² is also
        evaluated to assess model performance among
        non-obese individuals.
        """
    )

    # --------------------------------------------------------
    # 4.7 Target Variable
    # --------------------------------------------------------

    st.header("Target Variable")

    st.write(
        """
        The prediction target is a binary classification
        derived from measured HbA1c values.
        """
    )

    target_col1, target_col2 = st.columns(2)

    with target_col1:

        st.success("Class 0 — Normal HbA1c")

        st.markdown("**HbA1c < 5.7%**")

        st.write(
            "Participants whose measured HbA1c falls "
            "below the elevated HbA1c threshold."
        )

    with target_col2:

        st.warning("Class 1 — Elevated HbA1c")

        st.markdown("**HbA1c ≥ 5.7%**")

        st.write(
            "Participants whose measured HbA1c falls "
            "within or above the prediabetes range."
        )

    st.caption(
        "The binary target combines HbA1c values in the "
        "prediabetes and diabetes ranges. The model does "
        "not distinguish between these two ranges."
    )

    st.divider()

    # --------------------------------------------------------
    # 4.8 Research Questions
    # --------------------------------------------------------

    st.header("Research Questions")

    st.markdown(
        """
        **RQ1. Predictive Performance**

        Can non-laboratory demographic, anthropometric,
        behavioral, cardiovascular, and socioeconomic
        characteristics be used to identify elevated HbA1c
        among adults without a previously reported
        diabetes diagnosis?

        **RQ2. Feature Importance**

        Which non-laboratory characteristics are most
        important for identifying elevated HbA1c?

        **RQ3. Contribution of Additional Features**

        Do behavioral and socioeconomic characteristics
        improve the prediction of elevated HbA1c beyond
        basic factors such as age and BMI?

        **RQ4. Non-Obese Population**

        Can elevated HbA1c be identified among adults
        who are not obese?
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 4.9 Non-Laboratory Predictors
    # --------------------------------------------------------

    st.header("Non-Laboratory Predictors")

    st.write(
        """
        The final Expanded Logistic Regression model uses
        13 predictors organized into five categories.
        No laboratory measurements are used as predictors.
        """
    )

    predictor_col1, predictor_col2 = st.columns(2)

    with predictor_col1:

        st.markdown(
            """
            #### Demographic (3)

            - Age
            - Sex
            - Race/Ethnicity

            #### Anthropometric (2)

            - Body Mass Index (BMI)
            - Waist Circumference

            #### Cardiovascular (2)

            - Hypertension
            - High Cholesterol
            """
        )

    with predictor_col2:

        st.markdown(
            """
            #### Behavioral (4)

            - Smoking Status
            - Sleep Duration
            - Sedentary Time
            - Recreational Physical Activity

            #### Socioeconomic (2)

            - Education Level
            - Income-to-Poverty Ratio
            """
        )

    st.divider()

    # --------------------------------------------------------
    # 4.10 Research Methodology
    # --------------------------------------------------------

    st.header("Research Methodology")

    with st.expander(
        "View Research and Modeling Workflow",
        expanded=True
    ):

        st.markdown(
            """
            **1. Data Collection and Integration**

            Selected NHANES components were collected
            and combined across four survey cycles.

            **2. Data Cleaning and Preparation**

            Variables were harmonized, eligibility criteria
            were applied, and the binary HbA1c target
            was constructed.

            **3. Exploratory Data Analysis**

            Participant characteristics, HbA1c distributions,
            missing values, and relationships between
            predictors and the target were examined.

            **4. Model Development**

            Logistic Regression, Random Forest, and
            Gradient Boosting models were developed
            and compared.

            **5. Model Evaluation and Selection**

            Cross-validation and multiple evaluation
            metrics were used to compare candidate models.

            **6. Classification Threshold Selection**

            A classification threshold was selected to
            improve the identification of participants
            with elevated HbA1c.

            **7. Final Held-Out Test Evaluation**

            The selected model was evaluated on the
            held-out test dataset.

            **8. Research Question Analysis**

            Permutation feature importance, feature-group
            comparisons, and non-obese subgroup evaluation
            were used to address the research questions.

            **9. Application Development**

            The final model and preprocessing pipeline
            were exported for integration into this
            Streamlit application.
            """
        )

    # --------------------------------------------------------
    # 4.11 Final Selected Model
    # --------------------------------------------------------

    st.header("Final Selected Model")

    st.markdown(
        """
        | Model Component | Description |
        |---|---|
        | Algorithm | Logistic Regression |
        | Feature Set | Expanded |
        | Number of Predictors | 13 |
        | Target Outcome | HbA1c ≥ 5.7% |
        | Classification Type | Binary Classification |
        | Classification Threshold | 0.29 |
        | Evaluation Dataset | Held-Out Test Dataset |
        """
    )

    st.subheader("Final Held-Out Test Performance")

    performance_col1, performance_col2, performance_col3 = (
        st.columns(3)
    )

    performance_col1.metric(
        "Accuracy",
        "68.4%"
    )

    performance_col2.metric(
        "Precision",
        "50.2%"
    )

    performance_col3.metric(
        "Recall",
        "78.3%"
    )

    performance_col4, performance_col5, performance_col6 = (
        st.columns(3)
    )

    performance_col4.metric(
        "F1-Score",
        "0.612"
    )

    performance_col5.metric(
        "ROC-AUC",
        "0.777"
    )

    performance_col6.metric(
        "Average Precision",
        "0.586"
    )

    st.info(
        """
        The selected classification threshold of 0.29
        prioritizes identifying individuals with elevated
        HbA1c, accepting a higher number of false positives
        in exchange for improved sensitivity.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 4.12 Intended Use and Limitations
    # --------------------------------------------------------

    st.header("Intended Use and Limitations")

    st.write(
        """
        This application demonstrates the potential use
        of machine learning for preliminary HbA1c risk
        screening using non-laboratory characteristics.
        """
    )

    st.markdown(
        """
        - The model estimates the likelihood of elevated
          HbA1c; it does not predict an exact HbA1c value.
        - The model does not diagnose prediabetes or diabetes.
        - Feature importance does not establish causation.
        - Screening estimates may contain false positives
          and false negatives.
        - Performance may differ across populations.
        - Clinical validation is required before real-world
          healthcare use.
        """
    )

    st.warning(
        """
        This application is intended for research and
        educational purposes only. Laboratory testing
        and professional medical evaluation are required
        for clinical diagnosis.
        """
    )

# ============================================================
# 5. RESEARCH FINDINGS PAGE
# ============================================================

elif page == "Research Findings":

    # --------------------------------------------------------
    # 5.1 Page Introduction
    # --------------------------------------------------------

    st.title("Research Findings")

    st.write(
        """
        Explore the results of the four research questions
        investigating the identification of elevated HbA1c
        using non-laboratory characteristics.
        """
    )

    st.caption(
        "Final Model: Expanded Logistic Regression | "
        "13 Predictors | Classification Threshold: 0.29"
    )

    st.divider()

    # --------------------------------------------------------
    # 5.2 Research Question Navigation
    # --------------------------------------------------------

    rq1, rq2, rq3, rq4 = st.tabs(
        [
            "RQ1: Predictive Performance",
            "RQ2: Feature Importance",
            "RQ3: Feature Contribution",
            "RQ4: Non-Obese Adults"
        ]
    )

    # ========================================================
    # RQ1 — PREDICTIVE PERFORMANCE
    # ========================================================

    with rq1:

        st.header("RQ1 — Predictive Performance")

        st.markdown(
            """
            **Research Question**

            Can non-laboratory demographic, anthropometric,
            behavioral, cardiovascular, and socioeconomic
            characteristics be used to identify elevated HbA1c
            among adults without a previously reported
            diabetes diagnosis?
            """
        )

        st.subheader("Purpose of Analysis")

        st.write(
            """
            Evaluate whether the final Logistic Regression
            model can distinguish between adults with normal
            and elevated HbA1c using only non-laboratory
            predictors.
            """
        )

        # ----------------------------------------------------
        # RQ1 Key Performance Metrics
        # ----------------------------------------------------

        st.subheader("Final Held-Out Test Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            "68.4%",
            help=(
                "Proportion of all test participants "
                "correctly classified."
            )
        )

        col2.metric(
            "Precision",
            "50.2%",
            help=(
                "Proportion of participants predicted to "
                "have elevated HbA1c who actually have it."
            )
        )

        col3.metric(
            "Recall",
            "78.3%",
            help=(
                "Proportion of participants with elevated "
                "HbA1c correctly identified by the model."
            )
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "F1-Score",
            "0.612",
            help=(
                "Harmonic mean of precision and recall."
            )
        )

        col5.metric(
            "ROC-AUC",
            "0.777",
            help=(
                "Measures how well the model distinguishes "
                "between normal and elevated HbA1c across "
                "classification thresholds."
            )
        )

        col6.metric(
            "Average Precision",
            "0.586",
            help=(
                "Summarizes precision across recall levels "
                "as the classification threshold varies."
            )
        )

        st.divider()

        # ----------------------------------------------------
        # RQ1 Visualization
        # ----------------------------------------------------

        st.subheader("Confusion Matrix")

        display_visualization(
            "confusion_matrix.png"
        )

        st.caption(
            "Final Logistic Regression model evaluated on "
            "the held-out test dataset using a classification "
            "threshold of 0.29."
        )

        # ----------------------------------------------------
        # RQ1 Confusion Matrix Interpretation
        # ----------------------------------------------------

        st.subheader("Classification Outcomes")

        st.markdown(
            """
            | Classification | Participants | Interpretation |
            |---|---:|---|
            | True Negative | 1,490 | Correctly identified normal HbA1c |
            | False Positive | 849 | Normal HbA1c incorrectly flagged as elevated |
            | False Negative | 237 | Elevated HbA1c missed by the model |
            | True Positive | 856 | Correctly identified elevated HbA1c |
            """
        )

        st.subheader("Key Findings")

        st.markdown(
            """
            - The model was evaluated on 3,432 held-out
              test participants.

            - The model correctly identified 856 of the
              1,093 participants with elevated HbA1c.

            - The selected classification threshold of 0.29
              prioritizes sensitivity.

            - Increased sensitivity comes at the expense
              of more false-positive predictions.
            """
        )

        st.subheader("Conclusion")

        st.write(
            """
            Non-laboratory characteristics demonstrate
            potential for identifying elevated HbA1c
            among adults without a previously reported
            diabetes diagnosis.

            The model may be useful as a preliminary
            research screening approach, rather than
            a replacement for laboratory testing.
            """
        )

        st.caption(
            "Limitation: False-positive and false-negative "
            "predictions remain possible. External and clinical "
            "validation are required before healthcare use."
        )


    # ========================================================
    # RQ2 — FEATURE IMPORTANCE
    # ========================================================

    with rq2:

        st.header("RQ2 — Feature Importance")

        st.markdown(
            """
            **Research Question**

            Which non-laboratory characteristics are most
            important for identifying elevated HbA1c?
            """
        )

        st.subheader("Purpose of Analysis")

        st.write(
            """
            Identify which of the 13 non-laboratory predictors
            contribute most strongly to the predictive
            performance of the final Logistic Regression model.
            """
        )

        # ----------------------------------------------------
        # RQ2 Key Metrics
        # ----------------------------------------------------

        st.subheader("Top Three Predictors")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "1. Age",
            "0.180",
            help="Mean decrease in average precision."
        )

        col2.metric(
            "2. Race / Ethnicity",
            "0.051",
            help="Mean decrease in average precision."
        )

        col3.metric(
            "3. Body Mass Index",
            "0.039",
            help="Mean decrease in average precision."
        )

        st.divider()

        # ----------------------------------------------------
        # RQ2 Visualization
        # ----------------------------------------------------

        st.subheader("Permutation Feature Importance")

        display_visualization(
            "feature_importance.png"
        )

        st.caption(
            "Permutation feature importance measures the "
            "decrease in model performance when a predictor's "
            "values are randomly shuffled. Higher values "
            "indicate greater predictive importance. Error "
            "bars represent one standard deviation across "
            "repeated permutations."
        )

        # ----------------------------------------------------
        # RQ2 Method Explanation
        # ----------------------------------------------------

        with st.expander(
            "How is permutation feature importance calculated?"
        ):

            st.markdown(
                """
                1. Evaluate the model using the original data.
                2. Randomly shuffle the values of one predictor.
                3. Evaluate the model again.
                4. Calculate the decrease in average precision.
                5. Repeat the process to estimate the mean
                   importance and its variability.
                """
            )

            st.latex(
                r"\text{Importance} = AP_{\text{original}} "
                r"- AP_{\text{permuted}}"
            )

            st.write(
                """
                A larger decrease in performance indicates
                that the model relies more strongly on
                that predictor.
                """
            )

        # ----------------------------------------------------
        # RQ2 Interpretation
        # ----------------------------------------------------

        st.subheader("Key Findings")

        st.markdown(
            """
            - **Age** is the strongest predictor of elevated
              HbA1c within the final model.

            - **Race/Ethnicity** is the second most important
              predictor.

            - **BMI** is the third most important predictor.

            - The remaining predictors contribute smaller
              individual improvements to predictive performance.

            - Sex, recreational physical activity, and
              hypertension demonstrate near-zero individual
              permutation importance.
            """
        )

        st.subheader("Conclusion")

        st.write(
            """
            Demographic and anthropometric characteristics,
            particularly age, race/ethnicity, and BMI, are the
            most influential predictors within the final
            Logistic Regression model.
            """
        )

        st.caption(
            "Limitation: Predictive importance does not "
            "establish causation. Race/ethnicity may reflect "
            "underlying social, environmental, or healthcare "
            "inequities. Correlated predictors may also share "
            "predictive information."
        )


    # ========================================================
    # RQ3 — CONTRIBUTION OF ADDITIONAL FEATURE GROUPS
    # ========================================================

    with rq3:

        st.header("RQ3 — Contribution of Additional Features")

        st.markdown(
            """
            **Research Question**

            Do behavioral and socioeconomic characteristics
            improve the prediction of elevated HbA1c beyond
            basic factors such as age and BMI?
            """
        )

        st.subheader("Purpose of Analysis")

        st.write(
            """
            Compare the cross-validation performance of
            three Logistic Regression models with
            progressively expanded predictor sets.
            """
        )

        # ----------------------------------------------------
        # RQ3 Model Definitions
        # ----------------------------------------------------

        st.subheader("Model Definitions")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Baseline Model",
                "2 Predictors"
            )

            st.markdown(
                """
                - Age
                - BMI
                """
            )

        with col2:

            st.metric(
                "Core Model",
                "7 Predictors"
            )

            st.markdown(
                """
                Baseline predictors plus:

                - Sex
                - Race/Ethnicity
                - Waist Circumference
                - Hypertension
                - High Cholesterol
                """
            )

        with col3:

            st.metric(
                "Expanded Model",
                "13 Predictors"
            )

            st.markdown(
                """
                Core predictors plus:

                - Smoking Status
                - Sleep Duration
                - Sedentary Time
                - Recreational Physical Activity
                - Education Level
                - Income-to-Poverty Ratio
                """
            )

        st.divider()

        # ----------------------------------------------------
        # RQ3 Visualization
        # ----------------------------------------------------

        st.subheader("Cross-Validation Performance Comparison")

        display_visualization(
            "rq3_model_comparison.png"
        )

        st.caption(
            "Mean cross-validation performance of the "
            "Baseline, Core, and Expanded Logistic Regression "
            "models. Higher scores indicate better performance "
            "according to the corresponding evaluation metric."
        )

        # ----------------------------------------------------
        # RQ3 Incremental Performance
        # ----------------------------------------------------

        st.subheader("Contribution of Behavioral and "
                     "Socioeconomic Features")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "ROC-AUC",
            "0.772",
            delta="+0.004",
            help=(
                "Improvement from the Core model "
                "to the Expanded model."
            )
        )

        col2.metric(
            "Average Precision",
            "0.578",
            delta="+0.007",
            help=(
                "Improvement from the Core model "
                "to the Expanded model."
            )
        )

        col3.metric(
            "Recall",
            "0.410",
            delta="+0.006",
            help=(
                "Improvement from the Core model "
                "to the Expanded model."
            )
        )

        st.caption(
            "Values represent mean cross-validation scores. "
            "Changes are based on displayed rounded results."
        )

        # ----------------------------------------------------
        # RQ3 Interpretation
        # ----------------------------------------------------

        st.subheader("Key Findings")

        st.markdown(
            """
            - The transition from the Baseline model to the
              Core model produced the largest improvements
              in predictive performance.

            - Adding behavioral and socioeconomic
              characteristics provided modest additional
              improvements in recall, F1-score, ROC-AUC,
              and average precision.

            - Accuracy remained unchanged at the displayed
              precision between the Core and Expanded models.

            - Precision decreased slightly after adding
              behavioral and socioeconomic predictors.

            - The Expanded model achieved the highest
              ROC-AUC and average precision among the
              three feature sets.
            """
        )

        st.subheader("Conclusion")

        st.write(
            """
            Behavioral and socioeconomic characteristics
            provide modest additional predictive value
            beyond the demographic, anthropometric, and
            cardiovascular characteristics included in the
            Core model.

            The Expanded model provides the most comprehensive
            non-laboratory predictor set and improves several
            important evaluation metrics.
            """
        )

        st.caption(
            "Limitation: Differences in mean cross-validation "
            "scores alone do not establish statistical "
            "significance."
        )


    # ========================================================
    # RQ4 — NON-OBESE POPULATION
    # ========================================================

    with rq4:

        st.header("RQ4 — Performance Among Non-Obese Adults")

        st.markdown(
            """
            **Research Question**

            Can elevated HbA1c be identified among adults
            who are not obese?
            """
        )

        st.subheader("Purpose of Analysis")

        st.write(
            """
            Evaluate whether the final Logistic Regression
            model maintains meaningful predictive performance
            among adults with BMI below 30 kg/m².
            """
        )

        # ----------------------------------------------------
        # RQ4 Population Definitions
        # ----------------------------------------------------

        st.subheader("Evaluation Populations")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Overall Test Population**")

            st.write(
                "All participants in the held-out "
                "test dataset, regardless of BMI."
            )

        with col2:

            st.markdown("**Non-Obese Population**")

            st.write(
                "A subset of the held-out test population "
                "with BMI below 30 kg/m²."
            )

        st.caption(
            "Both populations are evaluated using the same "
            "final Logistic Regression model and classification "
            "threshold of 0.29."
        )

        # ----------------------------------------------------
        # RQ4 Key Performance Metrics
        # ----------------------------------------------------

        st.subheader("Non-Obese Population Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            "72.4%",
            delta="+4.0 percentage points"
        )

        col2.metric(
            "Recall",
            "71.9%",
            delta="-6.4 percentage points"
        )

        col3.metric(
            "ROC-AUC",
            "0.796",
            delta="+0.019"
        )

        st.caption(
            "Changes are relative to the overall test population."
        )

        st.divider()

        # ----------------------------------------------------
        # RQ4 Visualization
        # ----------------------------------------------------

        st.subheader("Overall vs. Non-Obese Performance")

        display_visualization(
            "rq4_population_comparison.png"
        )

        st.caption(
            "Performance comparison between the complete "
            "held-out test population and the non-obese subgroup."
        )

        # ----------------------------------------------------
        # RQ4 Interpretation
        # ----------------------------------------------------

        st.subheader("Key Findings")

        st.markdown(
            """
            - The model correctly classified 72.4% of
              non-obese test participants.

            - The model identified 71.9% of non-obese
              participants with elevated HbA1c.

            - ROC-AUC increased from 0.777 in the overall
              population to 0.796 in the non-obese subgroup.

            - Precision, recall, F1-score, and average
              precision were lower in the non-obese
              subgroup than in the overall population.

            - The model maintained meaningful discriminatory
              performance among adults with BMI below 30.
            """
        )

        st.subheader("Conclusion")

        st.write(
            """
            Non-laboratory characteristics demonstrate
            potential for identifying elevated HbA1c
            among adults who are not obese.

            The results support exploring screening approaches
            that consider multiple characteristics rather
            than relying solely on obesity status.
            """
        )

        st.caption(
            "Limitation: BMI remains a model predictor, so "
            "these findings do not establish that prediction "
            "is independent of BMI. Differences in outcome "
            "prevalence may also affect performance comparisons."
        )

# ============================================================
# 6. HbA1c RISK SCREENING PAGE
# ============================================================

elif page == "HbA1c Risk Screening":

    st.title("HbA1c Risk Screening")

    st.write(
        """
        Enter demographic, anthropometric, cardiovascular,
        behavioral, and socioeconomic characteristics
        to generate a model-based screening estimate.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 6.1 Load Model Artifact
    # --------------------------------------------------------

    try:

        model_artifact = load_model_artifact()

        model = model_artifact["model"]

        threshold = model_artifact["threshold"]

        features = model_artifact["features"]

        target_labels = model_artifact["target_labels"]

        model_name = model_artifact["model_name"]

        target_definition = model_artifact["target_definition"]

    except Exception as error:

        st.error("Unable to load the saved model.")

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # 6.2 Model Status
    # --------------------------------------------------------

    st.success("Model loaded successfully.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model",
        "Logistic Regression"
    )

    col2.metric(
        "Predictors",
        len(features)
    )

    col3.metric(
        "Classification Threshold",
        f"{threshold:.2f}"
    )

    # --------------------------------------------------------
    # 6.3 Model Information
    # --------------------------------------------------------

    with st.expander(
        "View Model Information",
        expanded=False
    ):

        st.write("**Model name:**", model_name)

        st.write("**Target definition:**", target_definition)

        st.write("**Target labels:**", target_labels)

        st.write(
            "**Classification threshold:**",
            threshold
        )

        st.write(
            "**Model classes:**",
            model.classes_.tolist()
        )

        st.write("**Model file:**", MODEL_PATH.name)

        st.subheader("Required Predictors")

        feature_table = pd.DataFrame({
            "Number": range(1, len(features) + 1),
            "Predictor": features
        })

        st.dataframe(
            feature_table,
            hide_index=True,
            width="stretch"
        )

    st.divider()

    # --------------------------------------------------------
    # 6.4 Predictor Input Form
    # --------------------------------------------------------

    st.header("Participant Characteristics")

    st.write(
        "Please provide the following information. "
        "Fields marked with an asterisk (*) are required."
    )

    # ========================================================
    # 6.4.0 STUDY ELIGIBILITY
    # ========================================================

    st.subheader("Study Eligibility")

    st.write(
        "This screening model was developed for adults "
        "aged 20 years and older without a previously "
        "reported diabetes diagnosis."
    )

    diabetes_history = st.selectbox(
        "Have you ever been diagnosed with diabetes by "
        "a healthcare professional? *",
        options=DIABETES_HISTORY_OPTIONS,
        index=None,
        placeholder="Select response",
        help=(
            "This question determines whether the screening "
            "model is appropriate for you. It is not used "
            "as a predictor by the model."
        ),
        key="diabetes_history_input"
    )

    eligibility_confirmed = diabetes_history == "No"

    if diabetes_history == "Yes":

        st.warning(
            "This model was developed for adults without "
            "a previously reported diabetes diagnosis. "
            "It is not intended to assess individuals "
            "with an existing diabetes diagnosis."
        )

    elif diabetes_history in [
        "Don't know",
        "Prefer not to answer"
    ]:

        st.warning(
            "Study eligibility cannot be confirmed. "
            "A screening estimate cannot be generated."
        )

    elif eligibility_confirmed:

        st.success(
            "Diabetes history eligibility criterion satisfied."
        )

    else:

        st.caption(
            "Please answer the eligibility question."
        )

    st.divider()

    # ========================================================
    # 6.4.1 DEMOGRAPHIC CHARACTERISTICS
    # ========================================================

    st.subheader("1. Demographic Characteristics")

    demo_col1, demo_col2 = st.columns(2)

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    with demo_col1:

        age_selection = st.selectbox(
            "Age (years) *",
            options=list(AGE_OPTIONS.keys()),
            index=None,
            placeholder="Select your age",
            help=(
                "This model was developed for adults aged "
                "20 years and older. Ages 80 and above are "
                "recorded as 80, following NHANES conventions."
            ),
            key="age_input"
        )

    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------

    with demo_col2:

        sex_selection = st.selectbox(
            "Sex *",
            options=list(SEX_OPTIONS.keys()),
            index=None,
            placeholder="Select sex",
            help=(
                "Select the applicable category. The available "
                "categories reflect the original NHANES "
                "variable used during model training."
            ),
            key="sex_input"
        )

    # --------------------------------------------------------
    # Race/Ethnicity
    # --------------------------------------------------------

    race_ethnicity_selection = st.selectbox(
        "Race / Ethnicity *",
        options=list(RACE_ETHNICITY_OPTIONS.keys()),
        index=None,
        placeholder="Select race/ethnicity",
        help=(
            "Select the category that best describes your "
            "race or ethnicity. These categories follow the "
            "NHANES classification used during model development."
        ),
        key="race_ethnicity_input"
    )

    # --------------------------------------------------------
    # Convert selected labels into model values
    # --------------------------------------------------------

    age = (
        AGE_OPTIONS[age_selection]
        if age_selection is not None
        else np.nan
    )

    sex = (
        SEX_OPTIONS[sex_selection]
        if sex_selection is not None
        else np.nan
    )

    race_ethnicity = (
        RACE_ETHNICITY_OPTIONS[race_ethnicity_selection]
        if race_ethnicity_selection is not None
        else np.nan
    )

    # --------------------------------------------------------
    # Check demographic field completion
    # --------------------------------------------------------

    demographic_complete = all([
        age_selection is not None,
        sex_selection is not None,
        race_ethnicity_selection is not None
    ])

    if demographic_complete:

        st.success(
            "Demographic information completed."
        )

    else:

        st.caption(
            "Please complete all required demographic fields."
        )

    st.divider()

    # ========================================================
    # 6.4.2 ANTHROPOMETRIC CHARACTERISTICS
    # ========================================================

    st.subheader("2. Anthropometric Characteristics")

    st.write(
        "Provide your height, weight, and waist circumference. "
        "BMI and unit conversions will be calculated automatically."
    )

    # --------------------------------------------------------
    # Measurement system
    # --------------------------------------------------------

    measurement_system = st.radio(
        "Measurement System",
        options=[
            "US Customary (ft, in, lbs)",
            "Metric (cm, kg)"
        ],
        horizontal=True,
        help=(
            "Select your preferred measurement system. "
            "All measurements are converted to metric units "
            "before being passed to the model."
        ),
        key="measurement_system"
    )

    use_us_units = (
        measurement_system == "US Customary (ft, in, lbs)"
    )

    # --------------------------------------------------------
    # Height and Weight
    # --------------------------------------------------------

    st.markdown("#### Height and Weight")

    unknown_height_weight = st.checkbox(
        "I do not know my height or weight",
        help=(
            "If selected, BMI will be treated as missing. "
            "The trained model will use its fitted median "
            "imputation procedure."
        ),
        key="unknown_height_weight"
    )

    # Initialize measurements
    height_cm = np.nan
    weight_kg = np.nan
    bmi = np.nan

    if not unknown_height_weight:

        if use_us_units:

            height_col1, height_col2, weight_col = st.columns(3)

            # Height in feet
            with height_col1:

                height_feet = st.selectbox(
                    "Height (feet)",
                    options=list(range(3, 8)),
                    index=None,
                    placeholder="Select feet",
                    key="height_feet"
                )

            # Additional height in inches
            with height_col2:

                height_inches = st.selectbox(
                    "Height (inches)",
                    options=list(range(12)),
                    index=None,
                    placeholder="Select inches",
                    help=(
                        "Enter the remaining inches after "
                        "selecting your height in feet."
                    ),
                    key="height_inches"
                )

            # Weight in pounds
            with weight_col:

                weight_lbs = st.number_input(
                    "Weight (lbs)",
                    min_value=40.0,
                    max_value=1000.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    placeholder="Enter weight",
                    help="Weight in pounds before unit conversion.",
                    key="weight_lbs"
                )

            # Convert height to centimeters
            if (
                height_feet is not None
                and height_inches is not None
            ):

                height_cm = convert_height_to_cm(
                    height_feet,
                    height_inches
                )

            # Convert weight to kilograms
            if weight_lbs is not None:

                weight_kg = convert_weight_to_kg(
                    weight_lbs
                )

        else:

            metric_col1, metric_col2 = st.columns(2)

            # Height in centimeters
            with metric_col1:

                height_cm_input = st.number_input(
                    "Height (cm)",
                    min_value=90.0,
                    max_value=250.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    placeholder="Enter height",
                    key="height_cm"
                )

            # Weight in kilograms
            with metric_col2:

                weight_kg_input = st.number_input(
                    "Weight (kg)",
                    min_value=18.0,
                    max_value=454.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    placeholder="Enter weight",
                    key="weight_kg"
                )

            if height_cm_input is not None:

                height_cm = height_cm_input

            if weight_kg_input is not None:

                weight_kg = weight_kg_input

        # Validate converted measurements
        height_weight_valid = (
            pd.notna(height_cm)
            and pd.notna(weight_kg)
            and 90 <= height_cm <= 250
            and 18 <= weight_kg <= 454
        )

        # Calculate BMI
        if height_weight_valid:

            bmi = calculate_bmi(
                weight_kg,
                height_cm
            )

            # Display converted measurements and BMI
            bmi_col1, bmi_col2, bmi_col3 = st.columns(3)

            bmi_col1.metric(
                "Height",
                f"{height_cm:.1f} cm"
            )

            bmi_col2.metric(
                "Weight",
                f"{weight_kg:.1f} kg"
            )

            bmi_col3.metric(
                "Calculated BMI",
                f"{bmi:.2f} kg/m²",
                help=(
                    "BMI = Weight (kg) / Height (m)². "
                    "Height is converted to meters and "
                    "weight is converted to kilograms "
                    "before calculation."
                )
            )

            # Warn about values outside observed test range
            if bmi < 13.4 or bmi > 82.9:

                st.warning(
                    "The calculated BMI falls outside the "
                    "range observed in the held-out test "
                    "dataset. Please verify your measurements. "
                    "Model performance may be less reliable "
                    "for values outside the observed range."
                )

        elif (
            pd.notna(height_cm)
            and pd.notna(weight_kg)
        ):

            st.warning(
                "Please verify that the entered height "
                "and weight are correct."
            )

        else:

            st.caption(
                "Complete both height and weight "
                "to calculate BMI."
            )

    else:

        st.info(
            "BMI will be treated as missing and handled "
            "by the trained preprocessing pipeline."
        )

    st.divider()

    # --------------------------------------------------------
    # Waist Circumference
    # --------------------------------------------------------

    st.markdown("#### Waist Circumference")

    unknown_waist = st.checkbox(
        "I do not know my waist circumference",
        help=(
            "If selected, waist circumference will be "
            "treated as missing and handled by the "
            "trained preprocessing pipeline."
        ),
        key="unknown_waist"
    )

    # Initialize waist circumference
    waist_cm = np.nan

    if not unknown_waist:

        if use_us_units:

            waist_inches = st.number_input(
                "Waist Circumference (inches)",
                min_value=0.1,
                value=None,
                step=0.1,
                format="%.1f",
                placeholder="Enter waist circumference",
                help=(
                    "Measure at the uppermost lateral border "
                    "of the hip bone. Keep the measuring tape "
                    "horizontal and measure after breathing "
                    "out normally. "
                    "Conversion: inches × 2.54 = centimeters."
                ),
                key="waist_inches"
            )

            if waist_inches is not None:

                waist_cm = convert_inches_to_cm(
                    waist_inches
                )

        else:

            waist_cm_input = st.number_input(
                "Waist Circumference (cm)",
                min_value=0.1,
                value=None,
                step=0.1,
                format="%.1f",
                placeholder="Enter waist circumference",
                help=(
                    "Measure at the uppermost lateral border "
                    "of the hip bone. Keep the measuring tape "
                    "horizontal and measure after breathing "
                    "out normally."
                ),
                key="waist_cm"
            )

            if waist_cm_input is not None:

                waist_cm = waist_cm_input

        # Display waist circumference
        if pd.notna(waist_cm):

            st.metric(
                "Waist Circumference",
                f"{waist_cm:.1f} cm",
                help=(
                    "The final model expects waist "
                    "circumference in centimeters. "
                    "1 inch = 2.54 centimeters."
                )
            )

            # Warn about values outside observed test range
            if waist_cm < 56.2 or waist_cm > 169.5:

                st.warning(
                    "The entered waist circumference falls "
                    "outside the range observed in the "
                    "held-out test dataset. Please verify "
                    "your measurement. Model performance "
                    "may be less reliable outside this range."
                )

    else:

        st.info(
            "Waist circumference will be treated as missing "
            "and handled by the trained preprocessing pipeline."
        )

    # --------------------------------------------------------
    # Anthropometric field completion
    # --------------------------------------------------------

    bmi_complete = (
        unknown_height_weight
        or pd.notna(bmi)
    )

    waist_complete = (
        unknown_waist
        or pd.notna(waist_cm)
    )

    anthropometric_complete = (
        bmi_complete
        and waist_complete
    )

    if anthropometric_complete:

        if unknown_height_weight or unknown_waist:

            st.warning(
                "Anthropometric section completed with "
                "missing information. Missing measurements "
                "will be imputed by the trained model, "
                "which may reduce how representative the "
                "screening estimate is for you."
            )

        else:

            st.success(
                "Anthropometric information completed."
            )

    else:

        st.caption(
            "Please complete the measurements or indicate "
            "which measurements are unknown."
        )

    st.divider()

        # ========================================================
    # 6.4.3 CARDIOVASCULAR CHARACTERISTICS
    # ========================================================

    st.subheader("3. Cardiovascular Characteristics")

    st.write(
        "Provide information about your history of "
        "high blood pressure and high cholesterol."
    )

    # --------------------------------------------------------
    # Hypertension
    # --------------------------------------------------------

    hypertension_selection = st.selectbox(
        "Have you ever been told by a healthcare professional "
        "that you have high blood pressure?",
        options=list(CARDIOVASCULAR_OPTIONS.keys()),
        index=None,
        placeholder="Select response",
        help=(
            "Answer based on whether a healthcare professional "
            "has previously informed you that you have high "
            "blood pressure. Answer Yes even if your blood "
            "pressure is currently controlled through "
            "medication or lifestyle changes."
        ),
        key="hypertension_input"
    )

    # --------------------------------------------------------
    # High Cholesterol
    # --------------------------------------------------------

    high_cholesterol_selection = st.selectbox(
        "Have you ever been told by a healthcare professional "
        "that you have high cholesterol?",
        options=list(CARDIOVASCULAR_OPTIONS.keys()),
        index=None,
        placeholder="Select response",
        help=(
            "Answer based on whether a healthcare professional "
            "has previously informed you that you have high "
            "cholesterol. Answer Yes even if your cholesterol "
            "is currently controlled through medication "
            "or lifestyle changes."
        ),
        key="high_cholesterol_input"
    )

    # --------------------------------------------------------
    # Convert selections into model values
    # --------------------------------------------------------

    hypertension = (
        CARDIOVASCULAR_OPTIONS[hypertension_selection]
        if hypertension_selection is not None
        else np.nan
    )

    high_cholesterol = (
        CARDIOVASCULAR_OPTIONS[high_cholesterol_selection]
        if high_cholesterol_selection is not None
        else np.nan
    )

    # --------------------------------------------------------
    # Check cardiovascular field completion
    # --------------------------------------------------------

    cardiovascular_complete = all([
        hypertension_selection is not None,
        high_cholesterol_selection is not None
    ])

    # Check whether any selected responses are missing
    cardiovascular_missing = any([
        pd.isna(hypertension),
        pd.isna(high_cholesterol)
    ])

    # --------------------------------------------------------
    # Display completion status
    # --------------------------------------------------------

    if cardiovascular_complete:

        if cardiovascular_missing:

            st.warning(
                "Cardiovascular section completed with "
                "missing information. Missing responses "
                "will be handled by the trained preprocessing "
                "pipeline, which may reduce how representative "
                "the screening estimate is for you."
            )

        else:

            st.success(
                "Cardiovascular information completed."
            )

    else:

        st.caption(
            "Please complete both cardiovascular questions."
        )

    st.divider()

        # ========================================================
    # 6.4.4 BEHAVIORAL CHARACTERISTICS
    # ========================================================

    st.subheader("4. Behavioral Characteristics")

    st.write(
        "Provide information about your smoking, sleep, "
        "sedentary behavior, and recreational physical activity."
    )

    # --------------------------------------------------------
    # Smoking Status
    # --------------------------------------------------------

    st.markdown("#### Smoking Status")

    smoking_selection = st.selectbox(
        "What is your cigarette smoking status?",
        options=list(SMOKING_OPTIONS.keys()),
        index=None,
        placeholder="Select smoking status",
        help=(
            "Never smoker: smoked fewer than 100 cigarettes "
            "during their lifetime. "
            "Former smoker: smoked at least 100 cigarettes "
            "but no longer smokes. "
            "Current smoker: smoked at least 100 cigarettes "
            "and currently smokes every day or some days."
        ),
        key="smoking_status_input"
    )

    smoking_status = (
        SMOKING_OPTIONS[smoking_selection]
        if smoking_selection is not None
        else np.nan
    )

    # --------------------------------------------------------
    # Sleep Duration
    # --------------------------------------------------------

    st.markdown("#### Sleep Duration")

    sleep_selection = st.selectbox(
        "How many hours do you usually sleep at night?",
        options=list(SLEEP_OPTIONS.keys()),
        index=None,
        placeholder="Select sleep duration",
        help=(
            "Select your usual nighttime sleep duration. "
            "Available options range from 2 to 14.5 hours "
            "in half-hour increments, based on the observed "
            "study test dataset range."
        ),
        key="sleep_hours_input"
    )

    sleep_hours = (
        SLEEP_OPTIONS[sleep_selection]
        if sleep_selection is not None
        else np.nan
    )

    st.divider()

    # --------------------------------------------------------
    # Sedentary Time
    # --------------------------------------------------------

    st.markdown("#### Daily Sedentary Time")

    st.write(
        "Estimate how much time you usually spend sitting "
        "or reclining during a typical day, excluding sleep."
    )

    unknown_sedentary = st.checkbox(
        "I do not know or prefer not to answer",
        help=(
            "If selected, sedentary time will be treated "
            "as missing and handled by the fitted "
            "preprocessing pipeline."
        ),
        key="unknown_sedentary"
    )

    sedentary_minutes = np.nan

    if not unknown_sedentary:

        sedentary_col1, sedentary_col2 = st.columns(2)

        with sedentary_col1:

            sedentary_hours = st.selectbox(
                "Sedentary hours per day",
                options=list(range(25)),
                index=None,
                placeholder="Select hours",
                key="sedentary_hours_input"
            )

        with sedentary_col2:

            sedentary_extra_minutes = st.selectbox(
                "Additional minutes",
                options=list(range(60)),
                index=None,
                placeholder="Select minutes",
                key="sedentary_minutes_input"
            )

        # Calculate total sedentary minutes
        sedentary_minutes = calculate_sedentary_minutes(
            sedentary_hours,
            sedentary_extra_minutes
        )

        if pd.notna(sedentary_minutes):

            st.metric(
                "Calculated Daily Sedentary Time",
                f"{sedentary_minutes:.0f} minutes/day",
                help=(
                    "Daily sedentary minutes = "
                    "(Hours × 60) + Additional minutes."
                )
            )

            # Warn about values outside observed test range
            if sedentary_minutes < 1 or sedentary_minutes > 1200:

                st.warning(
                    "The calculated sedentary time falls outside "
                    "the observed test dataset range "
                    "(1–1,200 minutes per day). "
                    "Please verify your response."
                )

        elif (
            sedentary_hours is not None
            and sedentary_extra_minutes is not None
        ):

            st.error(
                "Total sedentary time cannot exceed "
                "24 hours per day."
            )

    else:

        st.info(
            "Sedentary time will be treated as missing."
        )

    st.divider()

    # --------------------------------------------------------
    # Recreational Physical Activity
    # --------------------------------------------------------

    st.markdown("#### Recreational Physical Activity")

    st.write(
        "Include sports, exercise, fitness, and other "
        "recreational activities performed for at least "
        "10 continuous minutes. Exclude work-related "
        "activity and transportation."
    )

    # --------------------------------------------------------
    # Moderate-intensity recreational activity
    # --------------------------------------------------------

    st.markdown("**Moderate-Intensity Activity**")

    moderate_response = st.selectbox(
        "Do you participate in moderate-intensity "
        "recreational physical activity?",
        options=ACTIVITY_OPTIONS,
        index=None,
        placeholder="Select response",
        help=(
            "Moderate-intensity activity causes a small "
            "increase in breathing or heart rate. "
            "Examples include brisk walking, recreational "
            "cycling, and moderate swimming."
        ),
        key="moderate_activity_response"
    )

    moderate_days = None
    moderate_minutes = None

    if moderate_response == "Yes":

        moderate_col1, moderate_col2 = st.columns(2)

        with moderate_col1:

            moderate_days = st.selectbox(
                "Moderate activity days per week",
                options=list(range(1, 8)),
                index=None,
                placeholder="Select days",
                key="moderate_days_input"
            )

        with moderate_col2:

            moderate_minutes = st.number_input(
                "Moderate activity minutes per active day",
                min_value=10,
                max_value=1440,
                value=None,
                step=1,
                placeholder="Enter minutes",
                help=(
                    "Enter the total minutes spent doing "
                    "moderate recreational activity "
                    "on a typical active day."
                ),
                key="moderate_minutes_input"
            )

    moderate_weekly_minutes = calculate_weekly_activity(
        moderate_response,
        moderate_days,
        moderate_minutes
    )

    if pd.notna(moderate_weekly_minutes):

        st.caption(
            f"Moderate activity: "
            f"{moderate_weekly_minutes:.0f} minutes/week"
        )

    # --------------------------------------------------------
    # Vigorous-intensity recreational activity
    # --------------------------------------------------------

    st.markdown("**Vigorous-Intensity Activity**")

    vigorous_response = st.selectbox(
        "Do you participate in vigorous-intensity "
        "recreational physical activity?",
        options=ACTIVITY_OPTIONS,
        index=None,
        placeholder="Select response",
        help=(
            "Vigorous-intensity activity causes a large "
            "increase in breathing or heart rate. "
            "Examples include running, fast cycling, "
            "and vigorous aerobic exercise."
        ),
        key="vigorous_activity_response"
    )

    vigorous_days = None
    vigorous_minutes = None

    if vigorous_response == "Yes":

        vigorous_col1, vigorous_col2 = st.columns(2)

        with vigorous_col1:

            vigorous_days = st.selectbox(
                "Vigorous activity days per week",
                options=list(range(1, 8)),
                index=None,
                placeholder="Select days",
                key="vigorous_days_input"
            )

        with vigorous_col2:

            vigorous_minutes = st.number_input(
                "Vigorous activity minutes per active day",
                min_value=10,
                max_value=1440,
                value=None,
                step=1,
                placeholder="Enter minutes",
                help=(
                    "Enter the total minutes spent doing "
                    "vigorous recreational activity "
                    "on a typical active day."
                ),
                key="vigorous_minutes_input"
            )

    vigorous_weekly_minutes = calculate_weekly_activity(
        vigorous_response,
        vigorous_days,
        vigorous_minutes
    )

    if pd.notna(vigorous_weekly_minutes):

        st.caption(
            f"Vigorous activity: "
            f"{vigorous_weekly_minutes:.0f} minutes/week"
        )

    # --------------------------------------------------------
    # Calculate Total Weekly Recreational Activity
    # --------------------------------------------------------

    recreational_activity_minutes_week = np.nan

    activity_total_valid = True

    if (
        pd.notna(moderate_weekly_minutes)
        and pd.notna(vigorous_weekly_minutes)
    ):

        total_weekly_minutes = (
            moderate_weekly_minutes
            + vigorous_weekly_minutes
        )

        if total_weekly_minutes > 10080:

            activity_total_valid = False

            st.error(
                "Total weekly activity cannot exceed "
                "the number of minutes in a week. "
                "Please verify your responses."
            )

        else:

            recreational_activity_minutes_week = (
                total_weekly_minutes
            )

            st.metric(
                "Total Weekly Recreational Physical Activity",
                f"{recreational_activity_minutes_week:.0f} "
                "minutes/week",
                help=(
                    "Total weekly recreational activity = "
                    "(Moderate days × Moderate minutes per day) + "
                    "(Vigorous days × Vigorous minutes per day). "
                    "Moderate and vigorous minutes are "
                    "added without intensity weighting."
                )
            )

            # Warn about values outside observed test range
            if recreational_activity_minutes_week > 3300:

                st.warning(
                    "The total weekly activity exceeds "
                    "the maximum observed in the held-out "
                    "test dataset (3,300 minutes per week). "
                    "Please verify your responses."
                )

    # --------------------------------------------------------
    # Behavioral Field Completion
    # --------------------------------------------------------

    smoking_complete = smoking_selection is not None

    sleep_complete = sleep_selection is not None

    sedentary_complete = (
        unknown_sedentary
        or pd.notna(sedentary_minutes)
    )

    moderate_complete = (
        moderate_response is not None
        and (
            moderate_response != "Yes"
            or pd.notna(moderate_weekly_minutes)
        )
    )

    vigorous_complete = (
        vigorous_response is not None
        and (
            vigorous_response != "Yes"
            or pd.notna(vigorous_weekly_minutes)
        )
    )

    behavioral_complete = all([
        smoking_complete,
        sleep_complete,
        sedentary_complete,
        moderate_complete,
        vigorous_complete,
        activity_total_valid
    ])

    # Check for missing information
    behavioral_missing = any([
        pd.isna(smoking_status),
        pd.isna(sleep_hours),
        pd.isna(sedentary_minutes),
        pd.isna(recreational_activity_minutes_week)
    ])

    # --------------------------------------------------------
    # Display Completion Status
    # --------------------------------------------------------

    if behavioral_complete:

        if behavioral_missing:

            st.warning(
                "Behavioral section completed with missing "
                "information. Missing responses will be "
                "handled by the trained preprocessing "
                "pipeline, which may reduce how representative "
                "the screening estimate is for you."
            )

        else:

            st.success(
                "Behavioral information completed."
            )

    else:

        st.caption(
            "Please complete all behavioral questions "
            "or indicate which responses are unknown."
        )

    st.divider()

        # ========================================================
    # 6.4.5 SOCIOECONOMIC CHARACTERISTICS
    # ========================================================

    st.subheader("5. Socioeconomic Characteristics")

    st.write(
        "Provide information about your education level "
        "and annual family income range."
    )

    # --------------------------------------------------------
    # Education Level
    # --------------------------------------------------------

    st.markdown("#### Education Level")

    education_selection = st.selectbox(
        "What is the highest level of education you completed?",
        options=list(EDUCATION_OPTIONS.keys()),
        index=None,
        placeholder="Select education level",
        help=(
            "Select your highest completed education level. "
            "The available categories follow the NHANES "
            "adult education classification."
        ),
        key="education_input"
    )

    # Convert selection to model value
    education = (
        EDUCATION_OPTIONS[education_selection]
        if education_selection is not None
        else np.nan
    )

    st.divider()

    # --------------------------------------------------------
    # Income-to-Poverty Ratio
    # --------------------------------------------------------

    st.markdown("#### Income-to-Poverty Ratio")

    st.write(
        "The income-to-poverty ratio compares estimated "
        "annual family income with the applicable federal "
        "poverty guideline. You do not need to disclose "
        "your exact income."
    )

    st.caption(
        f"Federal poverty guideline year: "
        f"{POVERTY_GUIDELINE_YEAR}"
    )

    # --------------------------------------------------------
    # Annual Family Income Range
    # --------------------------------------------------------

    income_selection = st.selectbox(
        "Annual family income range",
        options=list(INCOME_OPTIONS.keys()),
        index=None,
        placeholder="Select annual family income range",
        help=(
            "Select the range representing your combined "
            "annual family income before taxes. "
            "The midpoint of the selected income range "
            "will be used to estimate income, following "
            "the NHANES midpoint calculation approach."
        ),
        key="income_range_input"
    )

    # Initialize values
    estimated_income = np.nan
    family_size = None
    region = None
    poverty_guideline = np.nan
    income_poverty_ratio = np.nan

    high_income_selection = None

    # Check whether income was explicitly withheld
    income_unknown = income_selection in [
        "Don't know",
        "Prefer not to answer"
    ]

    # --------------------------------------------------------
    # Process Income Range
    # --------------------------------------------------------

    if income_selection is not None and not income_unknown:

        # Additional categories for income above $100,000
        if income_selection == "$100,000 and above":

            high_income_selection = st.selectbox(
                "Select a more specific income range",
                options=list(HIGH_INCOME_OPTIONS.keys()),
                index=None,
                placeholder="Select additional income range",
                help=(
                    "Additional income ranges are provided "
                    "because the highest original NHANES "
                    "income category is open-ended."
                ),
                key="high_income_range_input"
            )

            if high_income_selection is not None:

                estimated_income = HIGH_INCOME_OPTIONS[
                    high_income_selection
                ]

                if high_income_selection == "$500,000 and above":

                    st.caption(
                        "For this open-ended income range, "
                        "$500,000 is used as a lower-bound "
                        "income estimate."
                    )

        else:

            estimated_income = INCOME_OPTIONS[
                income_selection
            ]

        # ----------------------------------------------------
        # Family Size and Geographic Region
        # ----------------------------------------------------

        income_col1, income_col2 = st.columns(2)

        with income_col1:

            family_size = st.selectbox(
                "Number of family members",
                options=list(range(1, 21)),
                index=None,
                placeholder="Select family size",
                help=(
                    "Include yourself and the family members "
                    "whose income is included in the reported "
                    "combined family income. Do not automatically "
                    "include unrelated roommates."
                ),
                key="family_size_input"
            )

        with income_col2:

            region = st.selectbox(
                "Geographic region",
                options=REGION_OPTIONS,
                index=None,
                placeholder="Select region",
                help=(
                    "Federal poverty guidelines differ "
                    "between the 48 contiguous states "
                    "and Washington, D.C., Alaska, "
                    "and Hawaii."
                ),
                key="poverty_region_input"
            )

        # ----------------------------------------------------
        # Calculate Poverty Guideline
        # ----------------------------------------------------

        poverty_guideline = calculate_poverty_guideline(
            family_size,
            region
        )

        # ----------------------------------------------------
        # Calculate Income-to-Poverty Ratio
        # ----------------------------------------------------

        income_poverty_ratio = calculate_income_poverty_ratio(
            estimated_income,
            family_size,
            region
        )

        # ----------------------------------------------------
        # Display Calculated Values
        # ----------------------------------------------------

        if pd.notna(income_poverty_ratio):

            pir_col1, pir_col2, pir_col3 = st.columns(3)

            pir_col1.metric(
                "Estimated Annual Income",
                f"${estimated_income:,.0f}",
                help=(
                    "Estimated using the midpoint of the "
                    "selected bounded income range. "
                    "The highest open-ended range uses "
                    "its lower bound."
                )
            )

            pir_col2.metric(
                "Federal Poverty Guideline",
                f"${poverty_guideline:,.0f}",
                help=(
                    "The applicable federal poverty guideline "
                    "is determined by family size, geographic "
                    "region, and guideline year."
                )
            )

            pir_col3.metric(
                "Estimated Income-to-Poverty Ratio",
                f"{income_poverty_ratio:.2f}",
                help=(
                    "Income-to-Poverty Ratio = "
                    "Estimated annual family income divided "
                    "by the applicable federal poverty "
                    "guideline. Values at or above 5.00 "
                    "are capped at 5.00."
                )
            )

            # Calculate percentage of poverty guideline
            poverty_percentage = (
                estimated_income / poverty_guideline
            ) * 100

            st.caption(
                f"Estimated family income is "
                f"{poverty_percentage:.1f}% "
                f"of the applicable federal poverty guideline."
            )

            # Explain top-coding
            if estimated_income / poverty_guideline >= 5:

                st.caption(
                    "The income-to-poverty ratio is capped "
                    "at 5.00 to match NHANES conventions."
                )

            # ------------------------------------------------
            # Formula Explanation
            # ------------------------------------------------

            with st.expander(
                "How is the income-to-poverty ratio calculated?"
            ):

                st.markdown(
                    """
                    The ratio compares estimated annual
                    family income with the applicable
                    federal poverty guideline.
                    """
                )

                st.latex(
                    r"PIR = "
                    r"\frac{\text{Estimated Annual Family Income}}"
                    r"{\text{Federal Poverty Guideline}}"
                )

                st.latex(
                    rf"PIR = "
                    rf"\frac{{{estimated_income:,.0f}}}"
                    rf"{{{poverty_guideline:,.0f}}}"
                    rf" = "
                    rf"{estimated_income / poverty_guideline:.2f}"
                )

                st.write(
                    f"Final model input: "
                    f"{income_poverty_ratio:.2f}"
                )

                st.caption(
                    "This is an estimate based on an income "
                    "range, not an exact income amount or "
                    "an official benefits eligibility determination."
                )

                st.markdown(
                    "[Official HHS Poverty Guidelines]"
                    "(https://aspe.hhs.gov/poverty-guidelines)"
                )

                st.markdown(
                    "[NHANES Income Variable Documentation]"
                    "(https://wwwn.cdc.gov/nchs/data/nhanes/"
                    "public/2017/datafiles/DEMO_J.htm)"
                )

        else:

            st.caption(
                "Complete the income range, family size, "
                "and geographic region to calculate the "
                "estimated income-to-poverty ratio."
            )

    # --------------------------------------------------------
    # Missing Income Response
    # --------------------------------------------------------

    elif income_unknown:

        st.info(
            "Income information will be treated as missing "
            "and handled by the trained preprocessing pipeline."
        )

    # --------------------------------------------------------
    # Socioeconomic Field Completion
    # --------------------------------------------------------

    education_complete = (
        education_selection is not None
    )

    income_complete = (
        income_unknown
        or pd.notna(income_poverty_ratio)
    )

    socioeconomic_complete = all([
        education_complete,
        income_complete
    ])

    socioeconomic_missing = any([
        pd.isna(education),
        pd.isna(income_poverty_ratio)
    ])

    # --------------------------------------------------------
    # Display Completion Status
    # --------------------------------------------------------

    if socioeconomic_complete:

        if socioeconomic_missing:

            st.warning(
                "Socioeconomic section completed with "
                "missing information. Missing responses "
                "will be handled by the trained preprocessing "
                "pipeline, which may reduce how representative "
                "the screening estimate is for you."
            )

        else:

            st.success(
                "Socioeconomic information completed."
            )

    else:

        st.caption(
            "Please complete the socioeconomic questions "
            "or indicate which responses are unknown."
        )

    st.divider()

        # ========================================================
    # 6.5 INPUT REVIEW AND VALIDATION
    # ========================================================

    st.header("Review and Generate Screening Estimate")

    st.write(
        "Review the completion status of each section "
        "before generating your screening estimate."
    )

    # --------------------------------------------------------
    # Section completion status
    # --------------------------------------------------------

    section_status = {
        "Study Eligibility": eligibility_confirmed,
        "Demographic Characteristics": demographic_complete,
        "Anthropometric Characteristics": anthropometric_complete,
        "Cardiovascular Characteristics": cardiovascular_complete,
        "Behavioral Characteristics": behavioral_complete,
        "Socioeconomic Characteristics": socioeconomic_complete
    }

    completed_sections = sum(
        bool(status)
        for status in section_status.values()
    )

    total_sections = len(section_status)

    # Display completion progress
    st.progress(
        completed_sections / total_sections
    )

    st.caption(
        f"{completed_sections} of {total_sections} "
        f"sections completed or confirmed."
    )

    # Display section status
    status_col1, status_col2 = st.columns(2)

    for i, (section, complete) in enumerate(
        section_status.items()
    ):

        column = (
            status_col1
            if i % 2 == 0
            else status_col2
        )

        if complete:

            column.markdown(
                f"✅ {section}"
            )

        else:

            column.markdown(
                f"⚪ {section}"
            )

    # Check whether every section is complete
    all_sections_complete = all(
        section_status.values()
    )

    st.divider()

    # --------------------------------------------------------
    # Combine all 13 predictor values
    # --------------------------------------------------------

    input_values = {
        "age": age,
        "income_poverty_ratio": income_poverty_ratio,
        "bmi": bmi,
        "waist_cm": waist_cm,
        "recreational_activity_minutes_week":
            recreational_activity_minutes_week,
        "sedentary_minutes": sedentary_minutes,
        "sleep_hours": sleep_hours,
        "sex": sex,
        "race_ethnicity": race_ethnicity,
        "education": education,
        "smoking_status": smoking_status,
        "hypertension": hypertension,
        "high_cholesterol": high_cholesterol
    }

    # Initialize validation variables
    validation_errors = []
    unusual_measurements = []
    missing_predictors = []

    missing_confirmed = True
    unusual_confirmed = True

    model_input = None

    # --------------------------------------------------------
    # Validate completed sections
    # --------------------------------------------------------

    if all_sections_complete:

        validation_errors = validate_model_inputs(
            input_values
        )

        # ----------------------------------------------------
        # Identify missing predictor values
        # ----------------------------------------------------

        missing_predictors = [
            FEATURE_DISPLAY_NAMES[feature]
            for feature, value in input_values.items()
            if pd.isna(value)
        ]

        # ----------------------------------------------------
        # Identify unusual measurements
        # Based on observed held-out test dataset ranges
        # ----------------------------------------------------

        if pd.notna(bmi):

            if bmi < 13.4 or bmi > 82.9:

                unusual_measurements.append(
                    f"BMI: {bmi:.2f} kg/m² "
                    "(observed range: 13.4–82.9)"
                )

        if pd.notna(waist_cm):

            if waist_cm < 56.2 or waist_cm > 169.5:

                unusual_measurements.append(
                    f"Waist circumference: {waist_cm:.1f} cm "
                    "(observed range: 56.2–169.5 cm)"
                )

        if pd.notna(sedentary_minutes):

            if sedentary_minutes < 1 or sedentary_minutes > 1200:

                unusual_measurements.append(
                    f"Sedentary time: {sedentary_minutes:.0f} "
                    "minutes/day "
                    "(observed range: 1–1,200)"
                )

        if pd.notna(recreational_activity_minutes_week):

            if recreational_activity_minutes_week > 3300:

                unusual_measurements.append(
                    f"Recreational activity: "
                    f"{recreational_activity_minutes_week:.0f} "
                    "minutes/week "
                    "(observed range: 0–3,300)"
                )

        # ----------------------------------------------------
        # Additional time-consistency warning
        # ----------------------------------------------------

        if (
            pd.notna(sleep_hours)
            and pd.notna(sedentary_minutes)
            and pd.notna(recreational_activity_minutes_week)
        ):

            estimated_weekly_time = (
                sleep_hours * 60 * 7
                + sedentary_minutes * 7
                + recreational_activity_minutes_week
            )

            if estimated_weekly_time > 10080:

                unusual_measurements.append(
                    "Combined estimated sleep, sedentary "
                    "time and recreational activity exceeds "
                    "the available minutes in a week. "
                    "Please review the reported durations."
                )

        # ----------------------------------------------------
        # Display validation errors
        # ----------------------------------------------------

        if validation_errors:

            st.error(
                "Please correct the following inputs:"
            )

            for error in validation_errors:

                st.write(f"- {error}")

        # ----------------------------------------------------
        # Display missing-information warning
        # ----------------------------------------------------

        if missing_predictors:

            st.warning(
                f"{len(missing_predictors)} of the 13 predictors "
                "contain missing information."
            )

            for predictor in missing_predictors:

                st.write(f"- {predictor}")

            st.write(
                "The saved preprocessing pipeline replaces "
                "missing numerical values with training medians "
                "and missing categorical values with the most "
                "frequent training categories."
            )

            st.caption(
                "These replacement values may not represent "
                "your actual characteristics and may affect "
                "the screening estimate."
            )

            # Reset confirmation when missing fields change
            missing_confirmation_key = (
                "confirm_missing_"
                + "_".join(missing_predictors)
            )

            missing_confirmed = st.checkbox(
                "I understand that missing information "
                "will be imputed by the model.",
                key=missing_confirmation_key
            )

        # ----------------------------------------------------
        # Display unusual-measurement warning
        # ----------------------------------------------------

        if unusual_measurements:

            st.warning(
                "Some responses require additional review."
            )

            for measurement in unusual_measurements:

                st.write(f"- {measurement}")

            st.caption(
                "Model performance may be less reliable "
                "for values outside the observed dataset range."
            )

            # Reset confirmation when unusual values change
            unusual_confirmation_key = (
                "confirm_unusual_"
                + "_".join(unusual_measurements)
            )

            unusual_confirmed = st.checkbox(
                "I have reviewed these responses and confirm "
                "that they are correct.",
                key=unusual_confirmation_key
            )

    else:

        st.info(
            "Complete all required sections and confirm "
            "study eligibility before generating a "
            "screening estimate."
        )

    # --------------------------------------------------------
    # Determine whether the form is ready
    # --------------------------------------------------------

    form_ready = (
        all_sections_complete
        and not validation_errors
        and missing_confirmed
        and unusual_confirmed
    )

    # --------------------------------------------------------
    # Construct model input DataFrame
    # --------------------------------------------------------

    if form_ready:

        try:

            model_input = build_model_input(
                input_values,
                features
            )

            # Verify exact predictor order
            if list(model_input.columns) != list(features):

                raise ValueError(
                    "Model predictor order does not match."
                )

            # Confirm consistency with fitted pipeline
            if hasattr(model, "feature_names_in_"):

                if list(model_input.columns) != list(
                    model.feature_names_in_
                ):

                    raise ValueError(
                        "Input predictors do not match "
                        "the fitted model."
                    )

            st.success(
                "All required information has been reviewed. "
                "Your inputs are ready for screening."
            )

        except Exception as error:

            form_ready = False
            model_input = None

            st.error(
                "Unable to prepare model inputs."
            )

            st.exception(error)

    st.divider()

    # ========================================================
    # 6.6 SCREENING SUBMISSION
    # ========================================================

    st.caption(
        "This screening estimate is intended for educational "
        "and research purposes only. It is not a clinical "
        "diagnosis."
    )

    generate_estimate = st.button(
        "Generate Screening Estimate",
        type="primary",
        width="stretch",
        disabled=not form_ready,
        key="generate_screening_estimate"
    )

        # ========================================================
    # 6.7 SCREENING PREDICTION AND RESULTS
    # ========================================================

    if generate_estimate and form_ready and model_input is not None:

        try:

            # ------------------------------------------------
            # Generate Model Prediction
            # ------------------------------------------------

            with st.spinner("Generating screening estimate..."):

                results = generate_screening_prediction(
                    model,
                    model_input,
                    threshold
                )

            # Extract results
            elevated_probability = results[
                "elevated_probability"
            ]

            normal_probability = results[
                "normal_probability"
            ]

            prediction = results["prediction"]

            screening_threshold = results["threshold"]

            st.divider()

            # ------------------------------------------------
            # Screening Results Header
            # ------------------------------------------------

            st.header("Your HbA1c Screening Results")

            st.caption(
                f"Model: {model_name} | "
                f"Predictors: {len(features)} | "
                f"Threshold: {screening_threshold:.2f}"
            )

            # ------------------------------------------------
            # Display Probability Metrics
            # ------------------------------------------------

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Estimated Probability of Elevated HbA1c",
                f"{elevated_probability:.2%}",
                help=(
                    "The model-estimated probability that "
                    "HbA1c is 5.7% or higher based on "
                    "the provided characteristics. "
                    "This probability has not been "
                    "clinically validated or confirmed "
                    "to be calibrated for individual use."
                )
            )

            result_col2.metric(
                "Screening Threshold",
                f"{screening_threshold:.2%}",
                help=(
                    "The selected classification threshold "
                    "from model development. Probabilities "
                    "at or above this value are classified "
                    "as flagged for further screening."
                )
            )

            # ------------------------------------------------
            # Probability Visualization
            # ------------------------------------------------

            display_probability_chart(
                elevated_probability,
                screening_threshold
            )

            # ------------------------------------------------
            # Screening Classification
            # ------------------------------------------------

            st.subheader("Screening Classification")

            if prediction == 1:

                st.warning(
                    "Flagged for Further HbA1c Screening"
                )

                st.write(
                    f"The model-estimated probability of "
                    f"elevated HbA1c is "
                    f"**{elevated_probability:.2%}**, "
                    f"which meets or exceeds the selected "
                    f"screening threshold of "
                    f"**{screening_threshold:.2%}**."
                )

                st.write(
                    "Based on this research model, the "
                    "provided characteristics meet the "
                    "criteria for a positive screening flag."
                )

                st.write(
                    "Consider discussing appropriate HbA1c "
                    "testing with a healthcare professional."
                )

            else:

                st.info(
                    "Below Screening Threshold"
                )

                st.write(
                    f"The model-estimated probability of "
                    f"elevated HbA1c is "
                    f"**{elevated_probability:.2%}**, "
                    f"which is below the selected screening "
                    f"threshold of "
                    f"**{screening_threshold:.2%}**."
                )

                st.write(
                    "Based on this research model, the "
                    "provided characteristics do not meet "
                    "the selected screening threshold."
                )

                st.write(
                    "This result does not rule out elevated "
                    "HbA1c, prediabetes, or diabetes. "
                    "Continue recommended routine screening "
                    "and consult a healthcare professional "
                    "when appropriate."
                )

            # ------------------------------------------------
            # Missing Information Disclosure
            # ------------------------------------------------

            if missing_predictors:

                st.subheader("Missing Information")

                st.warning(
                    f"{len(missing_predictors)} of the "
                    f"{len(features)} predictors were missing "
                    "and handled by the fitted preprocessing "
                    "pipeline."
                )

                for predictor in missing_predictors:

                    st.write(f"- {predictor}")

                st.caption(
                    "Imputed values are based on the training "
                    "dataset and may not represent your actual "
                    "characteristics. Interpret the screening "
                    "estimate with additional caution."
                )

            # ------------------------------------------------
            # Unusual Measurement Disclosure
            # ------------------------------------------------

            if unusual_measurements:

                st.subheader("Input Range Considerations")

                st.warning(
                    "Some provided responses fall outside "
                    "observed dataset ranges or require "
                    "additional interpretation."
                )

                for measurement in unusual_measurements:

                    st.write(f"- {measurement}")

                st.caption(
                    "The model may be less reliable for "
                    "characteristics outside the ranges "
                    "represented in the study data."
                )

            # ------------------------------------------------
            # Income Estimation Disclosure
            # ------------------------------------------------

            if pd.notna(income_poverty_ratio):

                st.caption(
                    "The income-to-poverty ratio was "
                    "estimated using the selected family "
                    "income range and applicable federal "
                    "poverty guideline."
                )

            st.divider()

            # ------------------------------------------------
            # Explain Screening Threshold
            # ------------------------------------------------

            with st.expander(
                "Why is the screening threshold 0.29?"
            ):

                st.write(
                    """
                    During model development, the classification
                    threshold was selected to prioritize the
                    identification of participants with elevated
                    HbA1c.

                    A lower classification threshold generally
                    identifies more positive cases, but may also
                    increase false-positive predictions.
                    """
                )

                st.markdown(
                    """
                    **Final held-out test performance**
                    """
                )

                threshold_col1, threshold_col2 = st.columns(2)

                threshold_col1.metric(
                    "Sensitivity / Recall",
                    "78.3%"
                )

                threshold_col2.metric(
                    "Specificity",
                    "63.7%"
                )

                st.caption(
                    "These values describe performance on "
                    "the held-out NHANES test dataset, "
                    "not guaranteed performance for an "
                    "individual user."
                )

                st.write(
                    """
                    The classification threshold is a model
                    decision cutoff, not a clinical HbA1c
                    threshold or a medically established
                    risk category.
                    """
                )

            # ------------------------------------------------
            # Explain What the Model Predicts
            # ------------------------------------------------

            with st.expander(
                "What does this model predict?"
            ):

                st.markdown(
                    """
                    The model estimates the probability
                    of belonging to one of two categories:

                    | Category | HbA1c Definition |
                    |---|---|
                    | Normal | Below 5.7% |
                    | Elevated | 5.7% or higher |

                    The elevated category combines HbA1c
                    values in the prediabetes and diabetes
                    ranges.

                    The application does not:

                    - Predict an exact HbA1c measurement.
                    - Distinguish prediabetes from diabetes.
                    - Predict when diabetes will develop.
                    - Establish a clinical diagnosis.
                    - Replace laboratory testing.
                    """
                )

            # ------------------------------------------------
            # Model Limitations
            # ------------------------------------------------

            st.subheader("Important Limitations")

            st.markdown(
                """
                - The model was developed using NHANES
                  data collected between 2011 and 2018.

                - The application relies on self-reported
                  information and estimated measurements.

                - False-positive and false-negative
                  predictions are possible.

                - The estimated probability is not a
                  clinically validated individual risk score.

                - Performance may differ across demographic
                  groups and present-day populations.

                - External validation and clinical evaluation
                  are required before healthcare use.
                """
            )

            st.warning(
                "This screening result is intended for "
                "educational and research purposes only. "
                "It is not a medical diagnosis and should "
                "not be used to make treatment decisions."
            )

        except Exception as error:

            st.error(
                "Unable to generate the screening estimate."
            )

            st.exception(error)

    st.divider()

# ============================================================
# 7. FOOTER AND DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "UMBC DATA606 Capstone Project | NHANES 2011–2018"
)

st.caption(
    "This application is intended for educational and "
    "research purposes only. It is not a clinically validated "
    "diagnostic tool and does not replace laboratory testing "
    "or professional medical advice."
)
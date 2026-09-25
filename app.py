
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Gym Churn Prediction",
    page_icon="🏋️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666666;
    font-size: 18px;
    margin-bottom: 30px;
}

.risk-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: #f1f5f9;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🏋️ Gym Membership Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered member retention dashboard</div>',
    unsafe_allow_html=True
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("gym_churn_us.csv")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # -----------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------

    # Difference between historical and current attendance
    df["attendance_change"] = (
        df["Avg_class_frequency_total"]
        - df["Avg_class_frequency_current_month"]
    )

    # Engagement score
    df["engagement_score"] = (
        df["Avg_class_frequency_current_month"]
        / (df["Lifetime"] + 1)
    )

    return df


df = load_data()


# =========================================================
# MODEL TRAINING
# =========================================================

@st.cache_resource
def train_models(df):

    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -----------------------------------------------------
    # 70% Train / 15% Validation / 15% Test
    # -----------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # =====================================================
    # MODEL 1 — LOGISTIC REGRESSION
    # =====================================================

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # =====================================================
    # MODEL 2 — DECISION TREE
    # =====================================================

    tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    )

    # =====================================================
    # MODEL 3 — RANDOM FOREST
    # =====================================================

    rf = RandomForestClassifier(
        random_state=42
    )

    models = {
        "Logistic Regression": logistic,
        "Decision Tree": tree,
        "Random Forest": rf
    }

    # -----------------------------------------------------
    # Train models
    # -----------------------------------------------------

    for name, model in models.items():
        model.fit(X_train, y_train)

    # =====================================================
    # VALIDATION COMPARISON
    # =====================================================

    results = []

    for name, model in models.items():

        predictions = model.predict(X_val)

        probabilities = model.predict_proba(X_val)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_val,
                predictions
            ),
            "Precision": precision_score(
                y_val,
                predictions
            ),
            "Recall": recall_score(
                y_val,
                predictions
            ),
            "F1": f1_score(
                y_val,
                predictions
            ),
            "ROC-AUC": roc_auc_score(
                y_val,
                probabilities
            )
        })

    results_df = pd.DataFrame(results)

    # =====================================================
    # RANDOM FOREST HYPERPARAMETER TUNING
    # =====================================================

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        RandomForestClassifier(
            random_state=42
        ),
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # =====================================================
    # FINAL TEST EVALUATION
    # =====================================================

    test_predictions = best_model.predict(X_test)

    test_probabilities = (
        best_model.predict_proba(X_test)[:, 1]
    )

    test_metrics = {

        "Accuracy": accuracy_score(
            y_test,
            test_predictions
        ),

        "Precision": precision_score(
            y_test,
            test_predictions
        ),

        "Recall": recall_score(
            y_test,
            test_predictions
        ),

        "F1": f1_score(
            y_test,
            test_predictions
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            test_probabilities
        )
    }

    return (
        best_model,
        results_df,
        test_metrics,
        grid.best_params_
    )


# Train models
model, results_df, test_metrics, best_params = train_models(df)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📊 Model Information")

st.sidebar.write("Final Model")

st.sidebar.success(
    "Tuned Random Forest"
)

st.sidebar.write("Dataset")

st.sidebar.info(
    f"{len(df):,} members"
)

st.sidebar.write("Input Features")

st.sidebar.info(
    f"{len(df.columns) - 1} features"
)


# =========================================================
# MEMBER INFORMATION
# =========================================================

st.header("👤 Member Information")

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    near_location = st.selectbox(
        "Near Location?",
        ["Yes", "No"]
    )

    partner = st.selectbox(
        "Partner?",
        ["Yes", "No"]
    )

    promo_friends = st.selectbox(
        "Promo Friends?",
        ["Yes", "No"]
    )

    phone = st.selectbox(
        "Phone Available?",
        ["Yes", "No"]
    )


# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with col2:

    contract_period = st.number_input(
        "Contract Period (months)",
        min_value=1,
        max_value=24,
        value=6
    )

    group_visits = st.selectbox(
        "Group Visits?",
        ["Yes", "No"]
    )

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=30
    )

    additional_charges = st.number_input(
        "Average Additional Charges",
        min_value=0.0,
        value=100.0
    )


# ---------------------------------------------------------
# COLUMN 3
# ---------------------------------------------------------

with col3:

    months_to_contract = st.number_input(
        "Months to End Contract",
        min_value=0,
        max_value=24,
        value=6
    )

    lifetime = st.number_input(
        "Lifetime (months)",
        min_value=0,
        max_value=120,
        value=12
    )

    total_frequency = st.number_input(
        "Average Class Frequency - Total",
        min_value=0.0,
        value=2.0
    )

    current_frequency = st.number_input(
        "Average Class Frequency - Current Month",
        min_value=0.0,
        value=2.0
    )


# =========================================================
# CONVERT USER INPUT
# =========================================================

gender_value = 1 if gender == "Male" else 0

near_location_value = (
    1 if near_location == "Yes" else 0
)

partner_value = (
    1 if partner == "Yes" else 0
)

promo_friends_value = (
    1 if promo_friends == "Yes" else 0
)

phone_value = (
    1 if phone == "Yes" else 0
)

group_visits_value = (
    1 if group_visits == "Yes" else 0
)


# =========================================================
# FEATURE ENGINEERING FOR NEW MEMBER
# =========================================================

attendance_change = (
    total_frequency - current_frequency
)

engagement_score = (
    current_frequency / (lifetime + 1)
)


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({

    "gender": [gender_value],

    "Near_Location": [
        near_location_value
    ],

    "Partner": [
        partner_value
    ],

    "Promo_friends": [
        promo_friends_value
    ],

    "Phone": [
        phone_value
    ],

    "Contract_period": [
        contract_period
    ],

    "Group_visits": [
        group_visits_value
    ],

    "Age": [
        age
    ],

    "Avg_additional_charges_total": [
        additional_charges
    ],

    "Month_to_end_contract": [
        months_to_contract
    ],

    "Lifetime": [
        lifetime
    ],

    "Avg_class_frequency_total": [
        total_frequency
    ],

    "Avg_class_frequency_current_month": [
        current_frequency
    ],

    "attendance_change": [
        attendance_change
    ],

    "engagement_score": [
        engagement_score
    ]
})


# =========================================================
# PREDICTION
# =========================================================

st.divider()

if st.button(
    "🔮 Predict Churn Risk",
    use_container_width=True
):

    # Churn probability
    probability = model.predict_proba(
        input_data
    )[0][1]

    # Prediction using 0.50 threshold
    prediction = (
        1 if probability >= 0.50 else 0
    )

    st.header("📈 Prediction Result")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

        st.progress(
            float(probability)
        )


    # -----------------------------------------------------
    # RISK CATEGORY
    # -----------------------------------------------------

    with col2:

        if probability >= 0.70:

            st.error(
                "🔴 HIGH CHURN RISK"
            )

            st.write(
                "This member may require "
                "immediate retention attention."
            )

        elif probability >= 0.40:

            st.warning(
                "🟡 MEDIUM CHURN RISK"
            )

            st.write(
                "This member should be monitored "
                "and engaged."
            )

        else:

            st.success(
                "🟢 LOW CHURN RISK"
            )

            st.write(
                "The member currently shows "
                "relatively lower churn risk."
            )


    # =====================================================
    # RETENTION RECOMMENDATION
    # =====================================================

    st.subheader(
        "💡 Retention Recommendation"
    )

    if probability >= 0.70:

        st.write("""
        - Contact the member personally
        - Offer a personalized training session
        - Check attendance issues
        - Consider a retention promotion
        - Encourage group activities
        """)

    elif probability >= 0.40:

        st.write("""
        - Send an engagement message
        - Encourage regular class attendance
        - Recommend group activities
        - Monitor attendance over the next few weeks
        """)

    else:

        st.write("""
        - Continue normal engagement
        - Maintain communication
        - Encourage consistent attendance
        """)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.header("🤖 Model Performance")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1": "{:.3f}",
        "ROC-AUC": "{:.3f}"
    }),
    use_container_width=True
)


# =========================================================
# FINAL MODEL RESULTS
# =========================================================

st.subheader(
    "🏆 Tuned Random Forest Test Results"
)

metric_cols = st.columns(5)

for i, (metric, value) in enumerate(
    test_metrics.items()
):

    metric_cols[i].metric(
        metric,
        f"{value:.3f}"
    )


# =========================================================
# BEST HYPERPARAMETERS
# =========================================================

st.subheader(
    "⚙️ Best Hyperparameters"
)

st.json(best_params)


# =========================================================
# DATA QUALITY
# =========================================================

st.divider()

st.header("🧹 Data Quality")

missing_values = (
    df.isnull().sum().sum()
)

duplicate_rows = (
    df.duplicated().sum()
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    f"{len(df):,}"
)

col2.metric(
    "Missing Values",
    missing_values
)

col3.metric(
    "Duplicate Rows",
    duplicate_rows
)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.header("ℹ️ About This Project")

st.write("""
This machine learning system predicts the probability that a
gym member may churn.

The system compares Logistic Regression, Decision Tree and
Random Forest models. A tuned Random Forest is used as the
final prediction model.

The dashboard is designed to support proactive member
retention by identifying members who may require additional
engagement.
""")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Gym Membership Churn Prediction and Retention System | "
    "Machine Learning Project"
)

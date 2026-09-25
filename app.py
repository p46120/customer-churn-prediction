import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ===================================================
# PAGE CONFIGURATION
# ===================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ===================================================
# LOAD MODEL
# ===================================================

model = joblib.load("churn_model.pkl")
features = joblib.load("feature_info.pkl")


# ===================================================
# TITLE
# ===================================================

st.title("📊 Customer Churn Prediction Tool")

st.write(
    """
    This tool estimates the probability that a telecom customer
    may churn based on their historical usage, subscription,
    service and customer characteristics.
    """
)

st.info(
    """
    **Important:** This model was trained using historical telecom
    customer data. It is intended as a decision-support tool and
    should not replace managerial judgement.
    """
)


# ===================================================
# CUSTOMER INFORMATION
# ===================================================

st.header("1. Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Customer Age",
        min_value=15,
        max_value=100,
        value=30,
        step=1
    )

    age_group = st.selectbox(
        "Age Group",
        options=[1, 2, 3, 4, 5],
        index=1,
        help="""
        Age group used in the original dataset.
        1–5 represent the dataset's age categories.
        """
    )


with col2:

    subscription_length = st.number_input(
        "Subscription Length",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

    tariff_plan = st.selectbox(
        "Tariff Plan",
        options=[1, 2],
        index=0
    )


with col3:

    charge_amount = st.number_input(
        "Charge Amount",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

    customer_value = st.number_input(
        "Customer Value",
        min_value=0.0,
        value=200.0,
        step=10.0
    )


# ===================================================
# USAGE INFORMATION
# ===================================================

st.header("2. Customer Usage")

col1, col2, col3 = st.columns(3)


with col1:

    seconds_use = st.number_input(
        "Seconds of Use",
        min_value=0,
        value=3000,
        step=100
    )


with col2:

    frequency_use = st.number_input(
        "Frequency of Use",
        min_value=0,
        value=30,
        step=1
    )


with col3:

    frequency_sms = st.number_input(
        "Frequency of SMS",
        min_value=0,
        value=20,
        step=1
    )


# ===================================================
# SERVICE EXPERIENCE
# ===================================================

st.header("3. Service Experience")

col1, col2 = st.columns(2)


with col1:

    call_failure = st.number_input(
        "Number of Call Failures",
        min_value=0,
        value=5,
        step=1
    )


with col2:

    distinct_called_numbers = st.number_input(
        "Distinct Numbers Called",
        min_value=0,
        value=20,
        step=1
    )


complains = st.selectbox(
    "Has the customer made a complaint?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)


# ===================================================
# PREDICTION BUTTON
# ===================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    use_container_width=True
)


# ===================================================
# PREDICTION
# ===================================================

if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Call  Failure": call_failure,
        "Complains": complains,
        "Subscription  Length": subscription_length,
        "Charge  Amount": charge_amount,
        "Seconds of Use": seconds_use,
        "Frequency of use": frequency_use,
        "Frequency of SMS": frequency_sms,
        "Distinct Called Numbers": distinct_called_numbers,
        "Age Group": age_group,
        "Tariff Plan": tariff_plan,
        "Age": age,
        "Customer Value": customer_value
    }])

    # Make probability prediction
    probability = model.predict_proba(input_data)[0][1]

    probability_percentage = probability * 100


    # ===================================================
    # RISK CATEGORY
    # ===================================================

    if probability < 0.30:
        risk_level = "LOW"
        risk_message = "The customer has a relatively low predicted churn probability."

    elif probability < 0.60:
        risk_level = "MEDIUM"
        risk_message = "The customer shows a moderate predicted churn probability."

    else:
        risk_level = "HIGH"
        risk_message = "The customer shows a relatively high predicted churn probability."


    # ===================================================
    # RESULT DISPLAY
    # ===================================================

    st.header("4. Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated Churn Probability",
            f"{probability_percentage:.1f}%"
        )

    with col2:

        if risk_level == "HIGH":
            st.error(f"🔴 {risk_level} CHURN RISK")

        elif risk_level == "MEDIUM":
            st.warning(f"🟠 {risk_level} CHURN RISK")

        else:
            st.success(f"🟢 {risk_level} CHURN RISK")


    st.write(risk_message)


    # ===================================================
    # EXPLANATION
    # ===================================================

    st.header("5. Why did the model make this prediction?")

    st.write(
        """
        The model estimates churn probability using patterns learned
        from historical customer data. The factors below show which
        variables contributed relatively more toward the individual
        prediction.
        """
    )


    # Get logistic regression coefficients
    logistic_model = model.named_steps["logistic_regression"]
    scaler = model.named_steps["scaler"]

    # Standardized input values
    scaled_input = scaler.transform(input_data)

    coefficients = logistic_model.coef_[0]

    contributions = scaled_input[0] * coefficients

    explanation_df = pd.DataFrame({
        "Variable": features,
        "Contribution": contributions
    })

    explanation_df["Absolute Contribution"] = (
        explanation_df["Contribution"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute Contribution",
        ascending=False
    )


    # Show top 5 factors
    top_factors = explanation_df.head(5)


    for _, row in top_factors.iterrows():

        variable = row["Variable"]
        contribution = row["Contribution"]

        if contribution > 0:

            st.write(
                f"🔴 **{variable}** → contributed toward higher predicted churn risk"
            )

        else:

            st.write(
                f"🟢 **{variable}** → contributed toward lower predicted churn risk"
            )


    # ===================================================
    # MANAGERIAL INTERPRETATION
    # ===================================================

    st.header("6. Managerial Interpretation")

    st.write(
        """
        The prediction can be used as an additional input in customer
        retention decisions. A high predicted probability may indicate
        that the customer deserves closer review or proactive attention.
        However, the final decision should also consider managerial
        judgement and other customer information.
        """
    )


    # ===================================================
    # MODEL LIMITATION
    # ===================================================

    with st.expander("ℹ️ About this prediction"):

        st.write(
            """
            This prediction is generated using a Logistic Regression
            model trained on historical telecom customer data.

            A probability does not mean that the customer will
            definitely churn.

            The model is specific to the population and variables
            represented in the training data. It should not be directly
            transferred to another industry without retraining the
            model using relevant industry-specific data.
            """
        )

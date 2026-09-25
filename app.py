import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load("churn_model.pkl")
    features = joblib.load("feature_info.pkl")

except Exception as e:
    st.error("The prediction model could not be loaded.")
    st.error(f"Technical details: {e}")
    st.stop()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction Tool</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Estimate how likely a telecom customer is to leave the company
    and understand the key factors behind the prediction.
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    """
    **Decision-support tool:** This model uses patterns learned from
    historical telecom customer data. It is intended to support
    managerial decision-making and should not replace human judgment.
    """
)

# ============================================================
# HOW TO USE
# ============================================================

with st.expander("ℹ️ How to use this tool"):

    st.write(
        """
        **Step 1:** Enter the customer's information.

        **Step 2:** Click **Predict Customer Churn**.

        **Step 3:** Review the estimated churn probability, risk level,
        key factors and managerial interpretation.

        **Important:** This model was trained on telecom customer data.
        It should not be used directly for another industry without
        retraining and validation using relevant data.
        """
    )

# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("👤 1. Customer Information")

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    age = st.number_input(
        "Customer Age",
        min_value=15,
        max_value=100,
        value=30,
        step=1,
        help="Enter the customer's age in years."
    )

    age_group = st.selectbox(
        "Age Group",
        options=[1, 2, 3, 4, 5],
        index=1,
        format_func=lambda x: f"Age Group {x}",
        help=(
            "Select the age-group category recorded for the customer "
            "in the company's customer data."
        )
    )

# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    subscription_length = st.number_input(
        "Customer Tenure (months)",
        min_value=0,
        max_value=120,
        value=20,
        step=1,
        help=(
            "How many months the customer has been subscribed "
            "to the telecom service."
        )
    )

    tariff_plan = st.selectbox(
        "Tariff Plan",
        options=[1, 2],
        format_func=lambda x: (
            "1 – Pay as you go"
            if x == 1
            else "2 – Contractual"
        ),
        help="Select the customer's tariff/service plan."
    )

# ------------------------------------------------------------
# COLUMN 3
# ------------------------------------------------------------

with col3:

    charge_amount = st.number_input(
        "Customer Charge Level (0–9)",
        min_value=0,
        max_value=9,
        value=3,
        step=1,
        help=(
            "This is not the rupee bill amount. "
            "It is the charge-level scale used in the dataset: "
            "0 = lowest charge level and 9 = highest charge level."
        )
    )

    customer_value = st.number_input(
        "Customer Value",
        min_value=0.0,
        value=200.0,
        step=10.0,
        help=(
            "Enter the customer value recorded/calculated by the company. "
            "This is the customer-value measure used by the model."
        )
    )

# ============================================================
# CUSTOMER USAGE
# ============================================================

st.header("📱 2. Customer Usage")

col1, col2, col3 = st.columns(3)

with col1:

    seconds_use = st.number_input(
        "Total Call Usage (seconds)",
        min_value=0,
        value=3000,
        step=100,
        help=(
            "Total number of seconds the customer has used "
            "for calls during the recorded period."
        )
    )

with col2:

    frequency_use = st.number_input(
        "Number of Calls",
        min_value=0,
        value=30,
        step=1,
        help=(
            "Total number of calls made by the customer "
            "during the recorded period."
        )
    )

with col3:

    frequency_sms = st.number_input(
        "Number of SMS",
        min_value=0,
        value=20,
        step=1,
        help=(
            "Total number of SMS messages sent by the customer "
            "during the recorded period."
        )
    )

# ============================================================
# SERVICE EXPERIENCE
# ============================================================

st.header("☎️ 3. Service Experience")

col1, col2, col3 = st.columns(3)

with col1:

    call_failure = st.number_input(
        "Number of Call Failures",
        min_value=0,
        value=5,
        step=1,
        help=(
            "Number of unsuccessful or failed calls "
            "recorded for the customer."
        )
    )

with col2:

    distinct_called_numbers = st.number_input(
        "Different Numbers Called",
        min_value=0,
        value=20,
        step=1,
        help=(
            "Number of different phone numbers contacted "
            "by the customer."
        )
    )

with col3:

    complains = st.selectbox(
        "Has the Customer Made a Complaint?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Select Yes if the customer has registered a complaint."
    )

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True,
    type="primary"
)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "Call Failure": call_failure,
        "Complains": complains,
        "Subscription Length": subscription_length,
        "Charge Amount": charge_amount,
        "Seconds of Use": seconds_use,
        "Frequency of use": frequency_use,
        "Frequency of SMS": frequency_sms,
        "Distinct Called Numbers": distinct_called_numbers,
        "Age Group": age_group,
        "Tariff Plan": tariff_plan,
        "Age": age,
        "Customer Value": customer_value
    }])

    # --------------------------------------------------------
    # IMPORTANT:
    # EXACT FEATURE ORDER USED DURING MODEL TRAINING
    # --------------------------------------------------------

    model_input = input_data[
        [
            "Call Failure",
            "Complains",
            "Subscription Length",
            "Charge Amount",
            "Seconds of Use",
            "Frequency of use",
            "Frequency of SMS",
            "Distinct Called Numbers",
            "Age Group",
            "Tariff Plan",
            "Age",
            "Customer Value"
        ]
    ]

    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    try:

        # Convert DataFrame to NumPy array.
        # This avoids the feature-name mismatch error
        # from the saved sklearn model.

        probability = float(
            model.predict_proba(model_input.values)[0][1]
        )

    except Exception as e:

        st.error("The prediction could not be generated.")

        st.error(
            "Please check that the model files match the training "
            "data and that all input values are valid."
        )

        st.error(f"Technical details: {e}")

        st.stop()

    probability_percentage = probability * 100

    # --------------------------------------------------------
    # PROBABILITY DISPLAY
    # --------------------------------------------------------

    if probability_percentage < 0.1:

        probability_display = f"{probability_percentage:.3f}%"

    else:

        probability_display = f"{probability_percentage:.2f}%"

    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    if probability < 0.30:

        risk_level = "LOW"

        risk_message = (
            "The model estimates a relatively low likelihood "
            "of churn for this customer."
        )

    elif probability < 0.60:

        risk_level = "MEDIUM"

        risk_message = (
            "The model estimates a moderate likelihood "
            "of churn for this customer."
        )

    else:

        risk_level = "HIGH"

        risk_message = (
            "The model estimates a relatively high likelihood "
            "of churn for this customer."
        )

    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.header("📊 4. Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated Churn Probability",
            probability_display
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        st.caption(
            "Estimated probability that the customer may churn "
            "based on the information entered."
        )

    with col2:

        if risk_level == "HIGH":

            st.error(
                "🔴 HIGH CHURN RISK"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "🟠 MEDIUM CHURN RISK"
            )

        else:

            st.success(
                "🟢 LOW CHURN RISK"
            )

        st.write(risk_message)

    # --------------------------------------------------------
    # SIMPLE MANAGERIAL MESSAGE
    # --------------------------------------------------------

    if probability < 0.30:

        st.success(
            """
            **Managerial interpretation:** The model does not identify
            this customer as a high-priority churn case based on the
            information entered.
            """
        )

    elif probability < 0.60:

        st.warning(
            """
            **Managerial interpretation:** This customer may deserve
            additional review. Consider service history, complaints,
            usage and customer value before deciding on any retention action.
            """
        )

    else:

        st.error(
            """
            **Managerial interpretation:** This customer may deserve
            closer attention from the retention team. Review the factors
            below before taking any action.
            """
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    st.header("🔍 5. Why did the model make this prediction?")

    st.write(
        """
        The model learned patterns from historical telecom customers.
        The factors below show which characteristics pushed the model's
        estimated churn risk relatively higher or lower for this customer.
        """
    )

    # --------------------------------------------------------
    # MODEL EXPLANATION
    # --------------------------------------------------------

    try:

        logistic_model = model.named_steps["logistic_regression"]

        scaler = model.named_steps["scaler"]

        # IMPORTANT:
        # Use the same NumPy input used for prediction.

        scaled_input = scaler.transform(
            model_input.values
        )

        coefficients = logistic_model.coef_[0]

        contributions = (
            scaled_input[0] * coefficients
        )

        explanation_df = pd.DataFrame({
            "Variable": [
                "Call Failure",
                "Complains",
                "Subscription Length",
                "Charge Amount",
                "Seconds of Use",
                "Frequency of use",
                "Frequency of SMS",
                "Distinct Called Numbers",
                "Age Group",
                "Tariff Plan",
                "Age",
                "Customer Value"
            ],
            "Contribution": contributions
        })

        explanation_df["Absolute Contribution"] = (
            explanation_df["Contribution"].abs()
        )

        explanation_df = explanation_df.sort_values(
            "Absolute Contribution",
            ascending=False
        )

        top_factors = explanation_df.head(5)

        # ----------------------------------------------------
        # FRIENDLY NAMES
        # ----------------------------------------------------

        friendly_names = {

            "Call Failure":
                "Call failures",

            "Complains":
                "Customer complaint",

            "Subscription Length":
                "Customer tenure",

            "Charge Amount":
                "Charge level",

            "Seconds of Use":
                "Call usage",

            "Frequency of use":
                "Number of calls",

            "Frequency of SMS":
                "Number of SMS",

            "Distinct Called Numbers":
                "Different numbers called",

            "Age Group":
                "Age group",

            "Tariff Plan":
                "Tariff plan",

            "Age":
                "Customer age",

            "Customer Value":
                "Customer value"
        }

        # ----------------------------------------------------
        # DISPLAY TOP FACTORS
        # ----------------------------------------------------

        for _, row in top_factors.iterrows():

            variable = row["Variable"]

            contribution = row["Contribution"]

            friendly_variable = friendly_names.get(
                variable,
                variable
            )

            if contribution > 0:

                st.write(
                    f"🔴 **{friendly_variable}** — "
                    "pushed the model's estimated churn risk higher."
                )

            elif contribution < 0:

                st.write(
                    f"🟢 **{friendly_variable}** — "
                    "pushed the model's estimated churn risk lower."
                )

            else:

                st.write(
                    f"⚪ **{friendly_variable}** — "
                    "had little effect on this prediction."
                )

    except Exception as e:

        st.warning(
            "The prediction was generated, but the explanation "
            "section could not be displayed."
        )

        st.caption(
            f"Technical details: {e}"
        )

    # ========================================================
    # MANAGERIAL INTERPRETATION
    # ========================================================

    st.header("💼 6. Managerial Interpretation")

    if probability < 0.30:

        st.write(
            """
            **What this means:** The model estimates a relatively low
            probability of churn for this customer.

            **How a manager can use it:** Combine this result with
            customer complaints, service quality, recent interactions
            and other business information before making a decision.
            """
        )

    elif probability < 0.60:

        st.write(
            """
            **What this means:** The model estimates a moderate
            probability of churn.

            **How a manager can use it:** Review the customer's service
            history, complaints, usage patterns and customer value before
            deciding whether proactive retention action is appropriate.
            """
        )

    else:

        st.write(
            """
            **What this means:** The model estimates a relatively high
            probability of churn.

            **How a manager can use it:** The customer may deserve closer
            attention from the retention team. Review the underlying
            customer information and business context before taking action.
            """
        )

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.header("📝 7. Customer Information Summary")

    summary_data = pd.DataFrame({

        "Customer Information": [

            "Age",
            "Tenure",
            "Tariff Plan",
            "Charge Level",
            "Customer Value",
            "Call Usage",
            "Number of Calls",
            "Number of SMS",
            "Call Failures",
            "Different Numbers Called",
            "Complaint"
        ],

        "Entered Value": [

            f"{age} years",

            f"{subscription_length} months",

            (
                "Pay as you go"
                if tariff_plan == 1
                else "Contractual"
            ),

            charge_amount,

            customer_value,

            f"{seconds_use} seconds",

            frequency_use,

            frequency_sms,

            call_failure,

            distinct_called_numbers,

            (
                "Yes"
                if complains == 1
                else "No"
            )
        ]
    })

    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ABOUT THE MODEL
    # ========================================================

    with st.expander("ℹ️ About this prediction"):

        st.write(
            """
            **Model:** Logistic Regression

            **Purpose:** Estimate the probability that a telecom customer
            may churn based on historical customer characteristics,
            usage and service information.

            **Important:** A probability is not a certainty.

            For example, a 60% predicted probability does not mean that
            the customer will definitely leave.

            This model was trained on a specific telecom customer dataset.
            Its results should not be directly transferred to another
            industry without retraining and validating the model using
            relevant industry-specific data.

            The prediction is intended to support managerial decision-making,
            not replace managerial judgment.
            """
        )

    # ========================================================
    # TECHNICAL INFORMATION
    # ========================================================

    with st.expander("🔧 Technical Information"):

        st.write(
            "Raw model probability:"
        )

        st.code(
            f"{probability:.8f}"
        )

        st.caption(
            "This value is shown for transparency and testing. "
            "It can be removed from the final version if desired."
        )

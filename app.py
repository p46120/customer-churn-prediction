import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
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

    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a8a 55%,
            #2563eb 100%
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.90;
        margin-bottom: 15px;
    }

    .hero-tag {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background-color: rgba(255,255,255,0.14);
        font-size: 0.85rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* Prediction card */
    .prediction-card {
        padding: 22px;
        border-radius: 15px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    /* Small text */
    .small-text {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* Workflow */
    .workflow {
        padding: 15px;
        border-radius: 12px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROFESSIONAL HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            📡 Customer Retention Intelligence
        </div>

        <div class="hero-subtitle">
            AI-powered customer churn assessment for telecom decision-making
        </div>

        <div class="hero-tag">
            Logistic Regression • Predictive Analytics • Decision Support
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DECISION SUPPORT NOTICE
# ============================================================

st.info(
    """
    **Managerial Decision-Support Tool**

    This application estimates the likelihood of customer churn using
    patterns learned from historical telecom customer data. The result
    should be used as an additional input for managerial decisions and
    should not replace human judgment.
    """
)


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander("🔎 How does this tool work?"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="workflow">
            👤<br>
            <b>Customer Data</b><br>
            <span class="small-text">
            Enter customer information
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="workflow">
            ⚙️<br>
            <b>AI Model</b><br>
            <span class="small-text">
            Analyse customer patterns
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="workflow">
            📊<br>
            <b>Churn Probability</b><br>
            <span class="small-text">
            Estimate likelihood of churn
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="workflow">
            💼<br>
            <b>Managerial Review</b><br>
            <span class="small-text">
            Combine AI with judgment
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.markdown(
    '<div class="section-title">👤 1. Customer Profile</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# CUSTOMER AGE
# ------------------------------------------------------------

with col1:

    age = st.selectbox(
        "Customer Age",
        options=[15, 25, 30, 45, 55],
        index=2,
        format_func=lambda x: f"{x} years",
        help=(
            "Select the customer's age from the age values represented "
            "in the historical dataset."
        )
    )

    # Convert age to the Age Group used by the trained model.
    age_group_mapping = {
        15: 1,
        25: 2,
        30: 3,
        45: 4,
        55: 5
    }

    age_group = age_group_mapping[age]


# ------------------------------------------------------------
# CUSTOMER TENURE
# ------------------------------------------------------------

with col2:

    subscription_length = st.number_input(
        "Customer Tenure (months)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help=(
            "Number of months the customer has been subscribed "
            "to the telecom service."
        )
    )


# ------------------------------------------------------------
# TARIFF PLAN
# ------------------------------------------------------------

with col3:

    tariff_plan = st.selectbox(
        "Tariff Plan",
        options=[1, 2],
        format_func=lambda x: (
            "1 – Pay as you go"
            if x == 1
            else "2 – Contractual"
        ),
        help=(
            "Select the type of tariff/service plan used by "
            "the customer."
        )
    )


# ============================================================
# CUSTOMER VALUE & CHARGES
# ============================================================

col1, col2 = st.columns(2)


with col1:

    charge_amount = st.number_input(
        "Customer Charge Level (0–9)",
        min_value=0,
        max_value=9,
        value=3,
        step=1,
        help=(
            "This is not the customer's rupee bill. "
            "It is the charge-level scale used in the historical "
            "dataset: 0 = lowest and 9 = highest."
        )
    )


with col2:

    customer_value = st.number_input(
        "Customer Value",
        min_value=0.0,
        max_value=5000.0,
        value=200.0,
        step=10.0,
        help=(
            "Enter the customer value recorded or calculated by "
            "the company. This is the customer-value measure "
            "used by the model."
        )
    )


# ============================================================
# CUSTOMER USAGE
# ============================================================

st.markdown(
    '<div class="section-title">📱 2. Customer Usage</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    seconds_use = st.number_input(
        "Total Call Usage (seconds)",
        min_value=0,
        max_value=50000,
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
        max_value=500,
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
        max_value=1000,
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

st.markdown(
    '<div class="section-title">☎️ 3. Service Experience</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    call_failure = st.number_input(
        "Number of Call Failures",
        min_value=0,
        max_value=100,
        value=5,
        step=1,
        help=(
            "Number of unsuccessful or failed calls "
            "recorded for this customer."
        )
    )


with col2:

    distinct_called_numbers = st.number_input(
        "Different Numbers Called",
        min_value=0,
        max_value=150,
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
        help=(
            "Select Yes if the customer has registered a complaint."
        )
    )


# ============================================================
# PREDICT BUTTON
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
    # CREATE MODEL INPUT
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
    # EXACT FEATURE ORDER USED DURING TRAINING
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

        probability = float(
            model.predict_proba(
                model_input.values
            )[0][1]
        )

    except Exception as e:

        st.error(
            "The prediction could not be generated."
        )

        st.error(
            "Please check the input values and model files."
        )

        st.error(
            f"Technical details: {e}"
        )

        st.stop()


    probability_percentage = probability * 100


    # ========================================================
    # PROBABILITY DISPLAY
    # ========================================================

    if probability_percentage < 0.1:

        probability_display = (
            f"{probability_percentage:.3f}%"
        )

    else:

        probability_display = (
            f"{probability_percentage:.2f}%"
        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

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

    st.markdown(
        '<div class="section-title">🎯 4. Churn Assessment</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    with col1:

        st.metric(
            "Estimated Churn Probability",
            probability_display
        )

        st.progress(
            min(
                max(probability, 0.0),
                1.0
            )
        )

        st.caption(
            "Estimated likelihood that this customer may churn "
            "based on the information entered."
        )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

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


    # ========================================================
    # MANAGERIAL INTERPRETATION
    # ========================================================

    if probability < 0.30:

        st.success(
            """
            **Managerial interpretation:** The model does not identify
            this customer as a high-priority churn case based on the
            information entered. Managers can still consider other
            customer information before making a decision.
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
            below and consider the broader customer context before action.
            """
        )


    # ========================================================
    # MODEL EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 5. Why did the model make this prediction?</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The model learned patterns from historical telecom customers.
        The factors below show which characteristics pushed the model's
        estimated churn risk relatively higher or lower for this customer.

        These are model associations, not proof that a particular factor
        directly caused the customer to churn.
        """
    )


    try:

        # ----------------------------------------------------
        # GET LOGISTIC REGRESSION AND SCALER
        # ----------------------------------------------------

        logistic_model = (
            model.named_steps[
                "logistic_regression"
            ]
        )

        scaler = (
            model.named_steps[
                "scaler"
            ]
        )


        # IMPORTANT:
        # Use the exact same input used for prediction.

        scaled_input = scaler.transform(
            model_input.values
        )


        coefficients = (
            logistic_model.coef_[0]
        )


        contributions = (
            scaled_input[0] * coefficients
        )


        # ----------------------------------------------------
        # NORMALISE FEATURE NAMES
        # ----------------------------------------------------

        clean_features = [
            str(feature)
            .replace("  ", " ")
            .strip()
            for feature in features
        ]


        explanation_df = pd.DataFrame({

            "Variable": clean_features,

            "Contribution": contributions

        })


        explanation_df[
            "Absolute Contribution"
        ] = (
            explanation_df[
                "Contribution"
            ].abs()
        )


        explanation_df = (
            explanation_df.sort_values(
                "Absolute Contribution",
                ascending=False
            )
        )


        top_factors = (
            explanation_df.head(5)
        )


        # ----------------------------------------------------
        # FRIENDLY VARIABLE NAMES
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
        # DISPLAY FACTORS
        # ----------------------------------------------------

        for _, row in top_factors.iterrows():

            variable = row["Variable"]

            contribution = (
                row["Contribution"]
            )

            friendly_variable = (
                friendly_names.get(
                    variable,
                    variable
                )
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
            "could not be displayed."
        )

        st.caption(
            f"Technical details: {e}"
        )


    # ========================================================
    # MANAGERIAL USE
    # ========================================================

    st.markdown(
        '<div class="section-title">💼 6. Managerial Use</div>',
        unsafe_allow_html=True
    )


    if probability < 0.30:

        st.write(
            """
            **Possible managerial use:** The customer can remain under
            routine monitoring. Managers can combine the model output
            with customer interactions, service history and other
            information available to them.
            """
        )

    elif probability < 0.60:

        st.write(
            """
            **Possible managerial use:** The customer may warrant
            additional review. Managers can examine recent complaints,
            service quality, usage patterns and customer value.
            """
        )

    else:

        st.write(
            """
            **Possible managerial use:** The customer may warrant
            closer attention from the retention team. The model output
            can be used to prioritise further investigation, while the
            final decision remains with the manager.
            """
        )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    with st.expander(
        "📝 View Customer Information Summary"
    ):

        summary_data = pd.DataFrame({

            "Customer Information": [

                "Customer Age",
                "Customer Tenure",
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
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "ℹ️ About the Model & Responsible Use"
    ):

        st.write(
            """
            **Model:** Logistic Regression

            **Objective:** Estimate the probability that a telecom
            customer may churn using customer characteristics, usage
            behaviour and service experience.

            **Interpretation:** The output is a probability estimate,
            not a certainty. A high probability does not mean that the
            customer will definitely churn.

            **Scope:** The model was trained on a specific historical
            telecom customer dataset. It should not be directly applied
            to FMCG, banking, manufacturing or another industry without
            retraining and validating the model using relevant data.

            **Managerial role:** The tool is intended to support human
            decision-making rather than replace managerial judgment.
            """
        )


    # ========================================================
    # TECHNICAL TRANSPARENCY
    # ========================================================

    with st.expander(
        "🔧 Technical Information"
    ):

        st.write(
            "Raw probability returned by the model:"
        )

        st.code(
            f"{probability:.8f}"
        )

        st.caption(
            "This is provided for transparency and testing."
        )

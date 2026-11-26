import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="BankInsight | Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("bank_model.pkl")

model = load_model()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit top deploy bar, menu, and default footer */
#MainMenu, header, footer {
    visibility: hidden;
}

.stApp {
    background-color: #F4F7FB;
    color: #173452;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

[data-testid="stSidebar"] {
    background-color: #101D35;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    color: #FFFFFF;
    padding: 12px 0 5px 0;
}

.brand span {
    color: #38BDF8;
}

.sidebar-caption {
    color: #AAB8CC !important;
    font-size: 12px;
    line-height: 1.7;
}

.main-header {
    background: linear-gradient(120deg, #142541, #234C79);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.main-header h1 {
    color: white !important;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}

.main-header p {
    color: #D5E5F6;
    font-size: 14px;
    margin-bottom: 0;
}

.section-heading {
    color: #193451;
    font-size: 18px;
    font-weight: 700;
    margin: 12px 0 16px 0;
}

.info-card {
    background: white;
    border: 1px solid #E3EAF2;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 15px;
}

.info-label {
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* Fixed wrapping on Classification / Subscription */
.info-value {
    color: #173452;
    font-size: clamp(18px, 1.8vw, 23px);
    font-weight: 800;
    margin-top: 5px;
    white-space: nowrap;
}

/* Fix invisible labels and headings inside the white form card */
div[data-testid="stForm"] {
    background: white;
    border: 1px solid #E2EAF2;
    border-radius: 16px;
    padding: 22px;
}

div[data-testid="stForm"] h3,
div[data-testid="stForm"] label,
div[data-testid="stForm"] [data-testid="stWidgetLabel"] p {
    color: #173452 !important;
    font-weight: 600 !important;
}

div[data-testid="stForm"] h3 {
    font-weight: 700 !important;
    font-size: 18px !important;
}

.result-card {
    background: white;
    border: 1px solid #E2EAF2;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.result-title {
    color: #183653;
    font-size: 21px;
    font-weight: 800;
}

.result-yes {
    background: #E8F8F0;
    color: #087443;
    padding: 18px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 800;
    text-align: center;
}

.result-no {
    background: #FFF3E8;
    color: #B45309;
    padding: 18px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 800;
    text-align: center;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #087EAB, #1265A0);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 700;
    min-height: 48px;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #124B78;
    color: white !important;
    border: none;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2EAF2;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #173452 !important;
}

hr {
    border-color: #E2EAF2;
}

.footer {
    text-align: center;
    color: #718096;
    font-size: 12px;
    padding: 25px 0;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">Bank<span>Insight.</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sidebar-caption">CUSTOMER INTELLIGENCE PLATFORM</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠  Customer Prediction")
    st.markdown("📊  Model Overview")
    st.markdown("⚙️  Application Settings")

    st.divider()

    st.markdown("### Model Information")

    st.markdown("""
    <div class="sidebar-caption">
        Model: Random Forest Classifier<br><br>
        Task: Binary Classification<br><br>
        Prediction: Term Deposit Subscription<br><br>
        Version: 1.0
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<p class="sidebar-caption">Powered by Python, Scikit-learn & Streamlit</p>',
        unsafe_allow_html=True
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>Customer Intelligence Dashboard</h1>
    <p>
        Predict customer subscription behaviour using
        Machine Learning and historical banking data.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TOP INFORMATION CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">ML Algorithm</div>
        <div class="info-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Prediction Type</div>
        <div class="info-value">Classification</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Target Variable</div>
        <div class="info-value">Subscription</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="section-heading">Enter Customer Information</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter information available in the bank's customer "
    "records and marketing campaign system."
)

# --------------------------------------------------
# CUSTOMER INPUT FORM
# --------------------------------------------------

with st.form("customer_form"):

    st.markdown("### 👤 Personal Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

        job = st.selectbox(
            "Occupation",
            [
                "management", "technician", "entrepreneur",
                "blue-collar", "unknown", "retired",
                "admin.", "services", "self-employed",
                "unemployed", "housemaid", "student"
            ]
        )

    with c2:
        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )

        education = st.selectbox(
            "Education",
            ["primary", "secondary", "tertiary", "unknown"]
        )

    with c3:
        default = st.selectbox(
            "Credit in Default",
            ["no", "yes"]
        )

        balance = st.number_input(
            "Account Balance",
            value=1000
        )

    st.divider()

    st.markdown("### 💳 Financial Information")

    c1, c2 = st.columns(2)

    with c1:
        housing = st.selectbox(
            "Housing Loan",
            ["no", "yes"]
        )

    with c2:
        loan = st.selectbox(
            "Personal Loan",
            ["no", "yes"]
        )

    st.divider()

    st.markdown("### 📞 Marketing Campaign Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        contact = st.selectbox(
            "Contact Method",
            ["cellular", "telephone", "unknown"]
        )

        day = st.number_input(
            "Last Contact Day",
            min_value=1,
            max_value=31,
            value=15
        )

    with c2:
        month = st.selectbox(
            "Campaign Month",
            [
                "jan", "feb", "mar", "apr",
                "may", "jun", "jul", "aug",
                "sep", "oct", "nov", "dec"
            ],
            index=4
        )

        campaign = st.number_input(
            "Contacts in Current Campaign",
            min_value=1,
            value=1
        )

    with c3:
        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=-1,
            value=-1,
            help="-1 indicates the customer was not previously contacted."
        )

        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            value=0
        )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "other", "failure", "success"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔍  Generate Customer Prediction",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    customer_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    prediction = model.predict(customer_data)[0]
    probabilities = model.predict_proba(customer_data)[0]
    classes = list(model.classes_)

    yes_index = None
    for candidate in ("yes", "Yes", "YES", 1, True):
        if candidate in classes:
            yes_index = classes.index(candidate)
            break

    if yes_index is None:
        yes_index = 1 if len(classes) == 2 else 0

    no_index = 1 - yes_index if len(classes) == 2 else 0

    yes_probability = probabilities[yes_index]
    no_probability = probabilities[no_index]
    is_subscribed = (prediction == classes[yes_index])

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Prediction Analysis</div>',
        unsafe_allow_html=True
    )

    r1, r2 = st.columns([1, 1])

    with r1:

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="result-title">Model Prediction</div>',
            unsafe_allow_html=True
        )

        st.write("")

        if is_subscribed:
            st.markdown(
                '<div class="result-yes">✓ Likely to Subscribe</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-no">✕ Unlikely to Subscribe</div>',
                unsafe_allow_html=True
            )

        st.write("")
        st.metric(
            "Predicted Class",
            "YES" if is_subscribed else "NO"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with r2:

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="result-title">Prediction Probability</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.metric(
            "Subscription Probability",
            f"{yes_probability * 100:.2f}%"
        )

        st.progress(float(yes_probability))

        st.caption(
            f"Not subscribed: {no_probability * 100:.2f}%"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Probability Breakdown")

    probability_df = pd.DataFrame({
        "Outcome": ["Not Subscribed", "Subscribed"],
        "Probability": [no_probability, yes_probability]
    })

    st.bar_chart(
        probability_df.set_index("Outcome"),
        horizontal=True
    )

    st.info(
        "This is a model-generated estimate based on historical "
        "data. It is not a guarantee of customer behaviour. "
        "The prediction should support, not replace, human "
        "marketing decisions."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    BANKINSIGHT | CUSTOMER INTELLIGENCE PLATFORM<br>
    Machine Learning Demonstration Project
</div>
""", unsafe_allow_html=True)
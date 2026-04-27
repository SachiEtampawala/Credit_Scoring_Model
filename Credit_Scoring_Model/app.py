import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE ----------------
st.set_page_config(page_title="Credit Scoring App", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    font-size: 18px;
}

h1 {
    text-align: center;
    color: white;
    margin-top: 30px;
    margin-bottom: 28px;
    font-size: 46px;
    line-height: 1.3;
}

.subtitle {
    text-align: center;
    color: white;
    margin-bottom: 72px;
    font-size: 22px;
    line-height: 2;
}

.section-title {
    color: white;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 26px;
    line-height: 1.5;
}

.block-container {
    padding-left: 3rem;
    padding-right: 3rem;
}

.result-box {
    padding: 5px;
    border-radius: 15px;
    text-align: center;
    font-size: 20px;
    margin-top: 20px;
}

.output-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.18);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.output-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 28px 60px rgba(0,0,0,0.28);
    animation: bounce 0.55s ease;
}

@keyframes bounce {
    0% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
    60% { transform: translateY(0); }
    100% { transform: translateY(-4px); }
}

.output-row > div[data-testid="column"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.18);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.output-row > div[data-testid="column"]:hover {
    transform: translateY(-8px);
    box-shadow: 0 28px 60px rgba(0,0,0,0.28);
}

div.stButton > button {
    width: 100%;
    border-radius: 20px;
    height: 50px;
    font-size: 25px;
    font-weight: bold;
    color: white;
    background: linear-gradient(90deg, #0b132b, #1c2541);
    border: none;
    padding: 10px 20px;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #1e3a8a, #1d4ed8);
}

div[data-testid="column"] {
    padding: 0 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def train_model():
    data = pd.read_csv("data/credit_data.csv")

    data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    data = pd.get_dummies(data)

    X = data.drop('loan_status', axis=1)
    y = data['loan_status']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns

model, columns = train_model()

# ---------------- TITLE ----------------
st.title("💳 Credit Scoring Prediction")
st.markdown("<p class='subtitle'>Enter details to evaluate loan approval risk</p>", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p class='section-title'>👤 Customer Details</p>", unsafe_allow_html=True)
    age = st.number_input("Customer age", min_value=18, max_value=100)

with col2:
    st.markdown("<p class='section-title'>💸 Financial Details</p>", unsafe_allow_html=True)

    try:
        income = float(st.text_input("Monthly income",))
    except:
        income = 0.0

    emp_length = st.number_input("Employment length (years)", min_value=0)

    home = st.selectbox("Home ownership", ["Own", "Rent", "Mortage", "Other"])
    intent = st.selectbox("Loan purpose", ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Other"])

with col3:
    st.markdown("<p class='section-title'>💰 Credit Details</p>", unsafe_allow_html=True)

    try:
        loan_amount = float(st.text_input("Loan amount",))
    except:
        loan_amount = 0.0

    try:
        interest = float(st.text_input("Interest rate (%)",))
    except:
        interest = 0.0

    # Auto-calculated loan percent of income (non-editable, styled like others)
    percent_income = (loan_amount / income) if income != 0 and income > 0 else 0.0
    st.number_input("Loan % of income", value=round(percent_income * 100, 2), format="%.2f", disabled=True, step=0.01)

    default = st.selectbox("Default history", ["No", "Yes"])
    default_val = 1 if default == "Yes" else 0

    credit_length = st.number_input("Credit history length (years)", min_value=0)
    grade = st.selectbox("Loan grade", ["A","B","C","D","E","F","G"])

# ---------------- BUTTON ----------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict"):

    if income <= 0 or loan_amount <= 0:
        st.write("⚠️ Please enter valid income and loan amount.")
    else:
        with st.spinner("Analyzing the profile..."):
            time.sleep(1.5)

            input_data = pd.DataFrame(0, index=[0], columns=columns)

            input_data['person_age'] = age
            input_data['person_income'] = income
            input_data['person_emp_length'] = emp_length
            input_data['loan_amnt'] = loan_amount
            input_data['loan_int_rate'] = interest
            input_data['loan_percent_income'] = percent_income
            input_data['cb_person_default_on_file'] = default_val
            input_data['cb_person_cred_hist_length'] = credit_length

            if f'person_home_ownership_{home}' in input_data.columns:
                input_data[f'person_home_ownership_{home}'] = 1

            if f'loan_intent_{intent}' in input_data.columns:
                input_data[f'loan_intent_{intent}'] = 1

            if f'loan_grade_{grade}' in input_data.columns:
                input_data[f'loan_grade_{grade}'] = 1

            result = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0]

            risk = prob[1]
            safe = prob[0]

            # RESULT CARD
            if result == 1:
                st.markdown(
                    f"<div class='result-box'>❌ Rejected (High Risk)<br>🔴 Risk: {risk*100:.0f}%</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-box'>✅ Approved (Low Risk)<br>🟢 Safe: {safe*100:.0f}%</div>",
                    unsafe_allow_html=True
                )

            # ---------------- OUTPUT (3 COLUMNS) ----------------
            st.markdown("---")
            st.markdown("<div class='output-row'>", unsafe_allow_html=True)
            colA, colB, colC = st.columns([1.1,1,1.1])

            # AI EXPLANATION
            with colA:
                ratio = loan_amount / income if income != 0 else 0
                st.markdown(f"""
                    <div class='output-card'>
                        <p class='section-title'>📌 AI Explanation</p>
                        <ul>
                            <li>Loan burden: {ratio*100:.0f}% of income</li>
                            <li>Interest rate: {interest}%</li>
                            <li>Credit profile: {'Default history present' if default_val else 'No defaults'}</li>
                            <li>Employment stability: {emp_length} years</li>
                            <li>Credit history length: {credit_length} years</li>
                            <li>{'Model indicates high probability of default' if risk > 0.6 else 'Model indicates acceptable risk level'}</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            # DONUT CHART
            with colB:
                st.markdown("""
                    <div class='output-card'>
                        <p class='section-title'>📊 Risk Analysis</p>
                    """, unsafe_allow_html=True)

                fig = go.Figure(data=[go.Pie(
                    labels=["Safe", "Risk"],
                    values=[safe, risk],
                    hole=0.6,
                    marker=dict(colors=["#0B1236", "#4c4d59"], line=dict(color='white', width=1))
                )])

                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                    </div>
                """, unsafe_allow_html=True)

            # SMART INSIGHTS (only if loan is rejected)
            with colC:
                if result == 1:
                    st.markdown(f"""
                        <div class='output-card'>
                            <p class='section-title'>💡 Smart Insights</p>
                            <ul>
                                {''.join([
                                    '<li>Reduce loan-to-income ratio</li>' if ratio > 0.5 else '',
                                    '<li>Improve employment stability</li>' if emp_length < 2 else '',
                                    '<li>Build a longer credit history</li>' if credit_length < 3 else '',
                                    '<li>Maintain clean repayment history</li>' if default_val == 1 else '',
                                    '<li>Consider lower interest options</li>' if interest > 15 else '',
                                    '<li>Keep total debt burden below 35% of income</li>' if 0.35 < ratio <= 0.5 else '',
                                    '<li>Financial profile appears stable</li>' if risk <= 0.6 else ''
                                ])}
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Optional: show a friendly message when approved
                    st.markdown("""
                        <div class='output-card'>
                            <p class='section-title'>💡 Smart Insights</p>
                            <p style="color: #aaa;">✅ Your profile looks less risky – no critical issues found.</p>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
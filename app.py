import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Deposit Predictor", page_icon="Infinitive Bug.JPG", layout="wide")
# ------------------ LOAD MODEL ------------------
model = xgb.XGBClassifier()
model.load_model("xgb_deposit_model.json")

# ------------------ GLOBAL STYLING ------------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: Verdana, sans-serif !important;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #D1CFFE, #3B36C9);
        background-attachment: fixed;
    }
    .subtitle {
        text-align: center;
        font-size: 35px;
        color: #070384;
        margin-top: 5px;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(to bottom, #FDDFB2, #FDA51E);
        height: 100vh;
        overflow-y: auto;
        box-shadow: 0 0 20px rgba(0,0,0,0.2);
    }
    div.stButton > button {
        background-color: #E16600;
        color: white;
        border-radius: 20px;
        font-weight: bold;
        border: none;
        transition: color 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        color: #3B36C9;
        font-weight: bold;
    }
            
    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 0 20px rgba(0,0,0,0.2);
        color: #3B36C9
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown(
    "<h1 style='text-align:center; font-size: 80px; color: #E16600; font-weight: bold;'>INFINIREACH</h1>",
    unsafe_allow_html=True
)

st.markdown("<p class='subtitle'>Predict deposit subscription likelihood based on customer demographics and campaign data. Powered by machine learning.</p>", unsafe_allow_html=True)

# ------------------ INPUT LOGIC (SIDEBAR) ------------------
st.sidebar.markdown(
    "<span style='font-size: 28px; color: #3B36C9; font-weight: bold;'>Deposit Predictor</span>",
    unsafe_allow_html=True
)



job_options = {
    "Admin.": 0,
    "Blue Collar": 1,
    "Entrepreneur": 2,
    "Housemaid": 3,
    "Management": 4,
    "Retired": 5,
    "Self-Employed": 6,
    "Services": 7,
    "Student": 8,
    "Technician": 9,
    "Unemployed": 10,
    "Unknown": 11
}

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35, help="Customer age in years")
job = st.sidebar.selectbox("Job", list(job_options.keys()), help="Customer job category")
job_encoded = job_options[job]
balance = st.sidebar.number_input("Balance", value=1000, help="Customer average yearly balance")
month = st.sidebar.slider("Month", 1, 12, 5, help="Last contact month (1-12)")
day = st.sidebar.slider("Day", 1, 31, 15, help="Last contact day of month")
duration = st.sidebar.number_input("Duration", value=100, help="Last contact duration (sec.)")
campaign = st.sidebar.slider("Campaign", 1, 20, 2, help="Number of contacts during campaign")
pdays = st.sidebar.number_input("Days Since", value=999, help="Days since previous campaign")

# ------------------ FEATURE ENGINEERING ------------------
contact_effort_level = campaign * duration
balance_per_contact = balance / (campaign + 1)
campaign_month_num = month

X_input = pd.DataFrame([{
    "age": age,
    "job": job_encoded,
    "balance": balance,
    "month": month,
    "day": day,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "campaign_month_num": campaign_month_num,
    "contact_effort_level": contact_effort_level,
    "balance_per_contact": balance_per_contact
}])

predict_btn = st.sidebar.button("Predict")
prediction = None
proba = None

if predict_btn:
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

# ------------------ PREDICTION + CHART ------------------

##col_center = st.columns([1, 6, 1])[1]  # center the content in the middle column

with st.expander("📊 Probability Breakdown", expanded=True):
    if predict_btn and prediction is not None:

        result_text = "✔️ Likely to Subscribe" if prediction == 1 else "❌ Unlikely to Subscribe"

        # Show title and result above the pie chart
        st.markdown(
            f"""
            <div style='font-size: 28px; display: flex; align-items: center; gap: 12px; margin-bottom: 10px;'>
                <h2 style='color: #3B36C9; font-size: 25px; font-weight: bold; margin: 0;'>Subscription Likelihood:</h2>
                <span style='font-size: 25px; font-weight: bold; color: #333;'>{result_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )



        # --- Draw horizontal bar chart ---
        
        labels = ['Will Not Subscribe', 'Will Subscribe']
        colors = ['#E16600', '#3B36C9']

        fig, ax = plt.subplots(figsize=(6, 2))
        bars = ax.barh(labels, proba, color=colors)

# Style the chart
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")

# Add value labels next to bars
        for bar, p in zip(bars, proba):
            ax.text(p + 0.02, bar.get_y() + bar.get_height()/2, f"{p:.1%}", va='center', fontsize=10)

# Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        st.pyplot(fig)
    else:
        st.info("👈 Enter inputs and click **Predict** to view probability breakdown.")

with st.expander("🧠 Campaign Suggestions", expanded=False):
    if predict_btn and prediction is not None:
        suggestions = []

        if prediction == 0:
            st.error("😥 This customer is unlikely to subscribe. Try implementing the following strategies.")
            if duration < 300:
                suggestions.append("📞 Increase call duration, previous call was too short.")
            if campaign > 20:
                suggestions.append("🔁 Reduce contact frequency, customer may feel overwhelmed.")
            if campaign < 10:
                suggestions.append("🔁 Increase contact frequency, customer may feel neglected.")
            if pdays > 100:
                suggestions.append("🕓 Try re-engaging, previous campaign was too long ago.")
            if balance < 1100:
                suggestions.append("💰 Offer incentives, low balance customers may need stronger motivation.")

            if suggestions:
                for s in suggestions:
                    st.markdown(f"- {s}")
            else:
                st.markdown("✅ Campaign strategy is generally strong. Consider refining message tone or communication type.")

        else:
            st.success("😆 This customer is likely to subscribe! Keep doing what works.")
            st.markdown("""
            - 💼 Maintain current outreach approach
            - 🥇 Consider offering a loyalty bonus
            - 💰 Upsell or cross-sell with other financial products
            """)
    else:
        st.info("👈 Enter inputs and click **Predict** to view campaign suggestions.")
                

# Feature importance visual
with st.expander("🔍 Feature Importance"):
    fig_imp, ax_imp = plt.subplots()
    xgb.plot_importance(model, ax=ax_imp)
    st.pyplot(fig_imp)

with st.expander("⚙️ Model Performance", expanded=False):
    st.markdown("""
    - **Accuracy:** 84%  
    - **Precision:**
        - No Deposit: 86%
        - Deposit: 81%
    - **Recall:**
        - No Deposit: 83%
        - Deposit: 85%
    - **F1-Score:**
        - No Deposit: 85%
        - Deposit: 83%
    - **Macro Average F1-Score:** 84%
    - **Weighted Average F1-Score:** 84%
    - **Support:** 2,233 records
    """)



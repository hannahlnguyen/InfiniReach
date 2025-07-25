import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure page
# Set app icon
st.set_page_config(page_title="Deposit Predictor", page_icon="Infinitive Bug.JPG", layout="wide")

# Load XGBoost model
model = xgb.XGBClassifier()
model.load_model("xgb_deposit_model.json")

# Global CSS styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: Verdana, sans-serif !important;
            overflow-x: hidden !important;
    }
    h1 {
        font-size: 10vw !important;
        text-align: center;
    }
    .subtitle {
        font-size: 2.2vw !important;
        text-align: center;
        color: #070384;
        margin-top: 5px;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #D1CFFE, #3B36C9);
        background-attachment: fixed;
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
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important
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
    canvas {
        max-width: 100% !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    </style>
""", unsafe_allow_html=True)

# Name of app
st.markdown(
    "<h1 style='text-align:center; font-size: 80px; color: #E16600; font-weight: bold;'>INFINIREACH</h1>",
    unsafe_allow_html=True
)

# Web app description
st.markdown("<p class='subtitle'>Predict deposit subscription likelihood based on customer demographics and campaign data. Powered by machine learning.</p>", unsafe_allow_html=True)

# Create 2 tabs
app_tabs = st.tabs(["Customer Analysis", "Model Insights"]) 

# Customer Analysis tab
with app_tabs[0]:
    # Tab title
    st.markdown("<span style='font-size: 28px; color: #E16600; font-weight: bold;'>Customer Analysis</span>", unsafe_allow_html=True)

    # Sidebar form title
    st.sidebar.markdown(
        "<span style='font-size: 28px; color: #3B36C9; font-weight: bold;'>Deposit Predictor</span>",
        unsafe_allow_html=True
    )

    # Match job category name in form to its encoded value
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

    # Sidebar form input
    with st.sidebar.form("predict_form"):
        age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35, help="Customer age in years")
        job = st.sidebar.selectbox("Job Category", list(job_options.keys()), help="Customer job category")
        job_encoded = job_options[job]
        balance = st.sidebar.number_input("Avg. Yearly Balance", value=1000, help="Customer average yearly balance")
        month = st.sidebar.slider("Month", 1, 12, 5, help="Last contact month (1-12)")
        day = st.sidebar.slider("Day", 1, 31, 15, help="Last contact day of month")
        duration = st.sidebar.number_input("Contact Duration", value=100, help="Length of last contact duration with customer (sec.)")
        campaign = st.sidebar.slider("Campaign", 1, 20, 2, help="Number of contacts to customer during campaign")
        pdays = st.sidebar.number_input("Days Since", value=999, help="Amount of days since previous campaign")

    # Feature Engineering
    # Campaign intensity
    contact_effort_level = campaign * duration
    # Campaign effort vs finances
    balance_per_contact = balance / (campaign + 1)

    # Connect input to dataset columns
    X_input = pd.DataFrame([{
        "age": age,
        "job": job_encoded,
        "balance": balance,
        "month": month,
        "day": day,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "contact_effort_level": contact_effort_level,
        "balance_per_contact": balance_per_contact
    }])

    # Prediction form button
    predict_btn = st.sidebar.button("Predict")
    prediction = None
    proba = None
    if predict_btn:
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

    # Summary of inputted customer stats
    with st.expander("👤 Customer Profile", expanded=True):
        st.markdown(f"""
            <table style='width: 100%; border-collapse: collapse; font-size: 16px; color: #070384;'>
                <tr>
                    <th style='text-align: left; padding: 8px; color: #E16600;'>Feature</th>
                    <th style='text-align: left; padding: 8px; color: #E16600;'>Value</th>
                </tr>
                <tr><td style='padding: 8px;'>Age</td><td style='padding: 8px;'>{age}</td></tr>
                <tr><td style='padding: 8px;'>Job Category</td><td style='padding: 8px;'>{job}</td></tr>
                <tr><td style='padding: 8px;'>Avg. Yearly Balance</td><td style='padding: 8px;'>${balance:,.2f}</td></tr>
                <tr><td style='padding: 8px;'>Month</td><td style='padding: 8px;'>{month}</td></tr>
                <tr><td style='padding: 8px;'>Day</td><td style='padding: 8px;'>{day}</td></tr>
                <tr><td style='padding: 8px;'>Contact Duration (sec.)</td><td style='padding: 8px;'>{duration}</td></tr>
                <tr><td style='padding: 8px;'>Campaign</td><td style='padding: 8px;'>{campaign}</td></tr>
                <tr><td style='padding: 8px;'>Days Since</td><td style='padding: 8px;'>{pdays}</td></tr>
            </table>
        """, unsafe_allow_html=True)

    # Model prediction results visual
    with st.expander("📊 Probability Prediction", expanded=True):
        if predict_btn and prediction is not None:
            result_text = "✔️ Likely to Subscribe" if prediction == 1 else "❌ Unlikely to Subscribe"
            # Bar graph title and prediction result
            st.markdown(
                f"""
                <div style='font-size: 28px; display: flex; align-items: center; gap: 12px; margin-bottom: 10px;'>
                    <h2 style='color: #070384; font-size: 25px; font-weight: bold; margin: 0;'>Subscription Likelihood:</h2>
                    <span style='font-size: 25px; font-weight: bold; color: #070384;'>{result_text}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Horizontal bar chart (deposit subscription likelihood)
            labels = ['Will Not Subscribe', 'Will Subscribe']
            colors = ['#E16600', '#3B36C9']
            fig, ax = plt.subplots(figsize=(6, 2))
            bars = ax.barh(labels, proba, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            for bar, p in zip(bars, proba):
                ax.text(p + 0.02, bar.get_y() + bar.get_height()/2, f"{p:.1%}", va='center', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            st.pyplot(fig)
        # Instructions for user to complete form
        else:
            st.info("👈 Enter inputs and click **Predict** to view probability breakdown.")

    # Campaign suggestions
    with st.expander("🧠 Campaign Suggestions", expanded=False):
        if predict_btn and prediction is not None:
            suggestions = []
            # Suggestions if deposit subscription is unlikely
            if prediction == 0:
                st.error("😥 This customer is unlikely to subscribe. Try implementing the following strategies.")
                if duration < 500:
                    suggestions.append("📞 Increase call duration, previous call was too short.")
                if campaign >= 15:
                    suggestions.append("🔁 Reduce contact frequency, customer may feel overwhelmed.")
                if campaign < 15:
                    suggestions.append("🔁 Increase contact frequency, customer may feel neglected.")
                if pdays > 100:
                    suggestions.append("🕓 Try re-engaging, previous campaign was too long ago.")
                if balance < 2000:
                    suggestions.append("💰 Offer incentives, low balance customers may need stronger motivation.")
                if suggestions:
                    for s in suggestions:
                        st.markdown(f"<p style='color:#070384; font-size:16px;'>• {s}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("✅ Campaign strategy is generally strong. Consider refining message tone or communication type.")
            # Suggestions if deposit subscription is likely
            else:
                st.success("😆 This customer is likely to subscribe! Keep doing what works.")
                st.markdown("""
                - 💼 Maintain current outreach approach
                - 🥇 Consider offering a loyalty bonus
                - 💰 Upsell or cross-sell with other financial products
                """)
        # Instructions for user to complete form
        else:
            st.info("👈 Enter inputs and click **Predict** to view campaign suggestions.")

# Model Insights tab         
with app_tabs[1]:
    # Tab title
    st.markdown("<span style='font-size: 28px; color: #E16600; font-weight: bold;'>Model Insights</span>", unsafe_allow_html=True)

    # Feature importance visual
    with st.expander("🔍 Feature Importance", expanded=True):
        importance_info = {
        "weight": {
            "title": "Feature Importance by Weight",
            "desc": "How often each feature was used in the model's decision-making process (split count)."
        },
        "gain": {
            "title": "Feature Importance by Gain",
            "desc": "How much each feature improved model accuracy when it was used (information gain)."
        },
        "cover": {
            "title": "Feature Importance by Cover",
            "desc": "How many samples each feature helped split across all trees (data coverage)."
        }
    }

        # Loop and display all importance types
        for imp_type, info in importance_info.items():
            st.markdown(f"<h3 style='color:#E16600;'>{info['title']}</h3>", unsafe_allow_html=True)
            # Get feature importances
            booster = model.get_booster()
            score = booster.get_score(importance_type=imp_type)

            # Convert to sorted DataFrame
            importance_df = pd.DataFrame.from_dict(score, orient='index', columns=['Importance'])
            importance_df.index.name = 'Feature'
            importance_df = importance_df.sort_values(by='Importance', ascending=True)
            importance_df['Importance'] = importance_df['Importance'].round(1)

            # Plot with annotations
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(importance_df.index, importance_df['Importance'], color='#3B36C9')
            ax.set_xlabel('Importance')
            ax.set_title(info['title'])

            # Annotate each bar with its value
            for bar in bars:
                width = bar.get_width()
                ax.text(width - 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{width:.1f}', va='center', ha='right', fontsize=9, color='black')


            st.pyplot(fig)


            st.markdown(f"<p style='color:#070384;'>{info['desc']}</p>", unsafe_allow_html=True)

    # Model performance stats
    with st.expander("⚙️ Model Performance", expanded=False):
        st.markdown("""
        <div style='color:#070384; font-size:16px;'>
            <ul>
                <li><b>Accuracy:</b> 84%</li>
                <li><b>Precision:</b>
                    <ul>
                        <li>No Deposit: 86%</li>
                        <li>Deposit: 81%</li>
                    </ul>
                </li>
                <li><b>Recall:</b>
                    <ul>
                        <li>No Deposit: 83%</li>
                        <li>Deposit: 85%</li>
                    </ul>
                </li>
                <li><b>F1-Score:</b>
                    <ul>
                        <li>No Deposit: 84%</li>
                        <li>Deposit: 83%</li>
                    </ul>
                </li>
                <li><b>Macro Average F1-Score:</b> 84%</li>
                <li><b>Weighted Average F1-Score:</b> 84%</li>
                <li><b>Support:</b> 2,233 records</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # How the model works
    with st.expander("ℹ️ How This Works", expanded=False):
        st.markdown("""
        <div style='color:#070384; font-size:16px;'>
            InfiniReach uses an XGBoost machine learning model trained on historical marketing campaign data from InfiniBank to predict if a customer is likely to subscribe to a term deposit.<br><br>
            Predictions are based on contact timing, frequency, past outcomes, and customer profile features like age, balance, and job type.
        </div>
        """, unsafe_allow_html=True)


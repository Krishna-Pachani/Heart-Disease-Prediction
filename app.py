import streamlit as st
import pandas as pd
import joblib

# ---------- Page config ----------
st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #e53935 0%, #ff7043 100%);
        padding: 2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(229, 57, 53, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; color: white; }
    .hero p  { margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.95; }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .result-card {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        margin-top: 1rem;
        color: white;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .result-high { background: linear-gradient(135deg, #c62828, #ef5350); }
    .result-low  { background: linear-gradient(135deg, #2e7d32, #66bb6a); }
    .result-card h2 { margin: 0; color: white; }
    .result-card p  { margin: 0.4rem 0 0 0; font-size: 1.05rem; }

    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #e53935, #ff7043);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.7rem 0;
        border: none;
        border-radius: 12px;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        filter: brightness(1.08);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load model files (cached so it loads only once) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns


model, scaler, expected_columns = load_artifacts()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ❤️ About")
    st.write(
        "Aa app ek **KNN machine learning model** use kare che je "
        "tamara health details parthi heart disease no risk predict kare che."
    )
    st.markdown("### How to use")
    st.markdown(
        "1. Badhi details bharo\n"
        "2. **Predict Risk** button dabavo\n"
        "3. Result ane risk % juo"
    )
    st.info("Made by **Krishna** ❤️")
    st.warning(
        "⚠️ Aa sirf educational purpose mate che. "
        "Medical advice mate hamesha doctor ne maLo."
    )

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Stroke Prediction</h1>
        <p>Tamari health details bharo ane heart disease no risk check karo</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Input form ----------
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="section-title">👤 Personal Info</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
        sex = st.selectbox(
            "Gender",
            ["M", "F"],
            format_func=lambda x: "Male" if x == "M" else "Female",
        )
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            format_func=lambda x: {
                "ATA": "ATA - Atypical Angina",
                "NAP": "NAP - Non-Anginal Pain",
                "TA": "TA - Typical Angina",
                "ASY": "ASY - Asymptomatic",
            }[x],
        )

    with col2:
        st.markdown('<div class="section-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with col3:
        st.markdown('<div class="section-title">📈 ECG & Exercise</div>', unsafe_allow_html=True)
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, step=1)
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
        )
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, format="%.1f")
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("🔍 Predict Risk")

# ---------- Prediction ----------
if submitted:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Fill missing columns with 0 and reorder
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    with st.spinner("Analyzing your data..."):
        prediction = model.predict(scaled_input)[0]

        risk_pct = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_input)[0]
            classes = list(model.classes_)
            if 1 in classes:
                risk_pct = float(probs[classes.index(1)]) * 100

    st.markdown("---")
    st.subheader("📊 Result")

    if prediction == 1:
        st.markdown(
            """
            <div class="result-card result-high">
                <h2>⚠️ High Risk of Heart Disease</h2>
                <p>Tamari details mujab risk vadhare dekhay che. Please doctor ne consult karo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card result-low">
                <h2>✅ Low Risk of Heart Disease</h2>
                <p>Tamari details mujab risk ochho che. Healthy lifestyle chalu rakho!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    m1, m2, m3 = st.columns(3)
    if risk_pct is not None:
        m1.metric("Risk Probability", f"{risk_pct:.1f}%")
    m2.metric("Max Heart Rate", f"{max_hr} bpm")
    m3.metric("Blood Pressure", f"{resting_bp} mm Hg")

    if risk_pct is not None:
        st.progress(min(int(risk_pct), 100), text=f"Risk level: {risk_pct:.1f}%")

    with st.expander("📋 Tame aapeli details juo"):
        summary = pd.DataFrame(
            {
                "Field": [
                    "Age", "Gender", "Chest Pain Type", "Resting BP",
                    "Cholesterol", "Fasting BS > 120", "Resting ECG",
                    "Max HR", "Exercise Angina", "Oldpeak", "ST Slope",
                ],
                "Value": [
                    age, "Male" if sex == "M" else "Female", chest_pain, resting_bp,
                    cholesterol, "Yes" if fasting_bs == 1 else "No", resting_ecg,
                    max_hr, "Yes" if exercise_angina == "Y" else "No", oldpeak, st_slope,
                ],
            }
        )

        st.dataframe(summary, hide_index=True, use_container_width=True)

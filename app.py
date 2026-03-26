import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Title
st.markdown('<p class="title">🩺 Disease Prediction System</p>', unsafe_allow_html=True)

st.warning("⚠️ This is not a medical diagnosis. Consult a doctor.")

# Card start
st.markdown('<div class="card">', unsafe_allow_html=True)

# Symptoms list
symptoms_list = [
    "fever", "headache", "nausea", "chills", "sweating",
    "cough", "breathlessness", "sore_throat",
    "blurred_vision", "dizziness",
    "vomiting", "diarrhea", "abdominal_pain",
    "rash", "joint_pain",
    "fatigue", "weight_loss", "high_sugar",
    "chest_pain", "tiredness"
]

# Layout columns
col1, col2, col3 = st.columns(3)

with col1:
    symptom1 = st.selectbox("Symptom 1", symptoms_list)

with col2:
    symptom2 = st.selectbox("Symptom 2", symptoms_list)

with col3:
    symptom3 = st.selectbox("Symptom 3", symptoms_list)

# Button centered
predict = st.button("🔍 Predict Disease")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if predict:

    if len(set([symptom1, symptom2, symptom3])) < 3:
        st.error("⚠️ Please select 3 different symptoms")

    else:
        input_data = pd.DataFrame([[symptom1, symptom2, symptom3]],
                                  columns=["Symptom_1", "Symptom_2", "Symptom_3"])

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        probs = model.predict_proba(input_encoded)[0]
        top_indices = probs.argsort()[-3:][::-1]

        st.markdown("### 🧠 Prediction Results")

        for i in top_indices:
            disease_name = label_encoder.inverse_transform([i])[0]
            probability = probs[i] * 100

            st.markdown(f"""
                <div class="card">
                    <h4>{disease_name}</h4>
                    <p>Probability: <b>{probability:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

        st.success("✅ Prediction Complete!")

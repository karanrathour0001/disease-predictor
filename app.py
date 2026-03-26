import os
print(os.listdir())

import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Disease Predictor", page_icon="🩺")
st.warning("⚠️ This is not a medical diagnosis. Please consult a doctor.")

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🩺 Disease Prediction System")
st.write("Select symptoms and get predicted disease")

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

# User input
symptom1 = st.selectbox("Select Symptom 1", symptoms_list)
symptom2 = st.selectbox("Select Symptom 2", symptoms_list)
symptom3 = st.selectbox("Select Symptom 3", symptoms_list)

# Predict button
if st.button("Predict Disease"):

    # Check duplicate symptoms
    if len(set([symptom1, symptom2, symptom3])) < 3:
        st.error("Please select 3 different symptoms")

    else:
        # Create input dataframe
        input_data = pd.DataFrame([[symptom1, symptom2, symptom3]],
                                  columns=["Symptom_1", "Symptom_2", "Symptom_3"])

        # One-hot encoding
        input_encoded = pd.get_dummies(input_data)

        # Match training columns
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        # 🔥 Prediction with probability
        probs = model.predict_proba(input_encoded)[0]

        # Top 3 predictions
        top_indices = probs.argsort()[-3:][::-1]

        st.subheader("Top Predictions:")

        for i in top_indices:
            disease_name = label_encoder.inverse_transform([i])[0]
            probability = probs[i] * 100
            st.write(f"{disease_name} : {probability:.2f}%")

        st.success("Prediction complete!")
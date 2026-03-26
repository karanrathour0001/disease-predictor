import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Disease Predictor", page_icon="🩺")

st.warning("⚠️ This is not a medical diagnosis. Please consult a doctor.")

# Load files
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🩺 Smart Disease Predictor (NLP Enabled)")

# 🧠 Symptom dictionary (SMART NLP)
symptom_dict = {
    "fever": ["fever", "high temperature", "temperature", "bukhar"],
    "headache": ["headache", "head pain", "migraine"],
    "nausea": ["nausea", "feeling sick"],
    "chills": ["chills", "shivering"],
    "sweating": ["sweating", "sweat"],
    "cough": ["cough", "khansi"],
    "breathlessness": ["breathlessness", "shortness of breath"],
    "sore_throat": ["sore throat", "throat pain"],
    "blurred_vision": ["blurred vision", "vision problem"],
    "dizziness": ["dizziness", "lightheaded"],
    "vomiting": ["vomiting", "vomit", "puke"],
    "diarrhea": ["diarrhea", "loose motion"],
    "abdominal_pain": ["stomach pain", "abdominal pain"],
    "rash": ["rash", "skin rash"],
    "joint_pain": ["joint pain", "body pain"],
    "fatigue": ["fatigue", "weakness", "tired"],
    "weight_loss": ["weight loss", "losing weight"],
    "high_sugar": ["high sugar", "diabetes"],
    "chest_pain": ["chest pain"],
    "tiredness": ["tiredness", "exhausted"]
}

user_input = st.text_area("Describe your symptoms (e.g., I have high temperature and headache)")

if st.button("Predict Disease"):

    if user_input.strip() == "":
        st.error("❌ Please enter symptoms")
    else:
        text = user_input.lower()

        detected_symptoms = []

        # 🔍 Smart matching
        for key, synonyms in symptom_dict.items():
            for word in synonyms:
                if word in text:
                    detected_symptoms.append(key)
                    break

        if len(detected_symptoms) == 0:
            st.error("❌ No symptoms detected")
        else:
            st.write("🧠 Detected Symptoms:", detected_symptoms)

            # Take max 3
            selected = detected_symptoms[:3]

            # Fill if less than 3
            while len(selected) < 3:
                selected.append(selected[0])

            # Dataframe
            input_data = pd.DataFrame([selected],
                                     columns=["Symptom_1", "Symptom_2", "Symptom_3"])

            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

            probs = model.predict_proba(input_encoded)[0]
            top_indices = probs.argsort()[-3:][::-1]

            st.subheader("🧾 Top Predictions:")

            for i in top_indices:
                disease_name = label_encoder.inverse_transform([i])[0]
                probability = probs[i] * 100
                st.write(f"👉 {disease_name} : {probability:.2f}%")

            st.success("✅ Prediction complete!")

import streamlit as st
import pickle
import pandas as pd
from rapidfuzz import fuzz

st.set_page_config(page_title="AI Disease Predictor", page_icon="🧠", layout="centered")

# 🎨 Custom CSS (UI MAGIC)
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.chat-container {
    padding: 10px;
}

.user-msg {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    width: fit-content;
    margin-left: auto;
}

.bot-msg {
    background-color: #262730;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    width: fit-content;
}

.title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# 🧠 Header
st.markdown('<div class="title">🤖 AI Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Describe your symptoms like chatting with a doctor</div>', unsafe_allow_html=True)

st.warning("⚠️ This is not a medical diagnosis. Please consult a doctor.")

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Symptom dictionary
symptom_dict = {
    "fever": ["fever", "temperature", "bukhar"],
    "headache": ["headache", "head pain"],
    "nausea": ["nausea", "feeling sick"],
    "chills": ["chills", "shivering"],
    "sweating": ["sweating"],
    "cough": ["cough", "khansi"],
    "breathlessness": ["shortness of breath"],
    "sore_throat": ["throat pain"],
    "blurred_vision": ["vision problem"],
    "dizziness": ["dizziness"],
    "vomiting": ["vomit", "puke"],
    "diarrhea": ["loose motion"],
    "abdominal_pain": ["stomach pain"],
    "rash": ["skin rash"],
    "joint_pain": ["body pain"],
    "fatigue": ["weakness", "tired"],
    "weight_loss": ["losing weight"],
    "high_sugar": ["diabetes"],
    "chest_pain": ["chest pain"],
    "tiredness": ["exhausted"]
}

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input
prompt = st.chat_input("Type your symptoms...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    text = prompt.lower()
    detected = []

    for key, synonyms in symptom_dict.items():
        for word in synonyms:
            if fuzz.partial_ratio(word, text) > 80:
                detected.append(key)
                break

    detected = list(set(detected))

    if len(detected) == 0:
        reply = "❌ I couldn't detect symptoms. Try again."
    else:
        selected = detected[:3]
        while len(selected) < 3:
            selected.append(selected[0])

        input_data = pd.DataFrame([selected],
                                 columns=["Symptom_1", "Symptom_2", "Symptom_3"])

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        probs = model.predict_proba(input_encoded)[0]
        top_indices = probs.argsort()[-3:][::-1]

        reply = "🧾 Possible diseases:\n"
        for i in top_indices:
            disease = label_encoder.inverse_transform([i])[0]
            prob = probs[i] * 100
            reply += f"\n👉 {disease} ({prob:.2f}%)"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

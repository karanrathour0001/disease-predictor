import streamlit as st
import pickle
import pandas as pd
from rapidfuzz import fuzz

st.set_page_config(page_title="AI Disease Predictor", page_icon="🧠")

st.title("🤖 AI Disease Predictor (Chat Style)")
st.caption("Type your symptoms like chatting with a doctor")

st.warning("⚠️ This is not a medical diagnosis. Please consult a doctor.")

# Load files
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

all_words = []
for words in symptom_dict.values():
    all_words.extend(words)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
prompt = st.chat_input("Describe your symptoms...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    text = prompt.lower()

    detected = []

    # 🔥 Fuzzy matching
    for key, synonyms in symptom_dict.items():
        for word in synonyms:
            score = fuzz.partial_ratio(word, text)
            if score > 80:
                detected.append(key)
                break

    # Remove duplicates
    detected = list(set(detected))

    if len(detected) == 0:
        reply = "❌ Sorry, I couldn't detect symptoms. Try describing differently."
    else:
        selected = detected[:3]
        while len(selected) < 3:
            selected.append(selected[0])

        # Prepare input
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

    # Show bot reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)

import streamlit as st
import pickle
import pandas as pd
from rapidfuzz import fuzz
import time

st.set_page_config(page_title="AI Disease Predictor", page_icon="🧠", layout="centered")

# 🎨 CSS
st.markdown("""
<style>
.user-msg {
    background-color: #4CAF50;
    color: white;
    padding: 10px;
    border-radius: 15px;
    margin: 5px 0;
    width: fit-content;
    margin-left: auto;
}
.bot-msg {
    background-color: #262730;
    color: white;
    padding: 10px;
    border-radius: 15px;
    margin: 5px 0;
    width: fit-content;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Disease Predictor")

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Symptom dictionary
symptom_dict = {
    "fever": ["fever", "temperature", "bukhar"],
    "headache": ["headache", "head pain"],
    "cough": ["cough", "khansi"],
    "fatigue": ["weakness", "tired"],
    "vomiting": ["vomit", "puke"],
    "abdominal_pain": ["stomach pain"],
}

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# Chat display
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# Input
prompt = st.chat_input("Type your symptoms...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🤖 AI is thinking..."):
        time.sleep(1.5)

        text = prompt.lower()
        detected = []

        for key, synonyms in symptom_dict.items():
            for word in synonyms:
                if fuzz.partial_ratio(word, text) > 80:
                    detected.append(key)
                    break

        detected = list(set(detected))

        if len(detected) == 0:
            reply = "❌ No symptoms detected"
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
            result_data = []

            for i in top_indices:
                disease = label_encoder.inverse_transform([i])[0]
                prob = probs[i] * 100
                reply += f"\n👉 {disease} ({prob:.2f}%)"
                result_data.append((disease, prob))

            # Save history
            st.session_state.history.append(result_data)

    # Typing effect
    final_text = ""
    placeholder = st.empty()
    for char in reply:
        final_text += char
        placeholder.markdown(f'<div class="bot-msg">{final_text}</div>', unsafe_allow_html=True)
        time.sleep(0.01)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# 📊 DASHBOARD
st.divider()
st.subheader("📊 Prediction Dashboard")

if st.session_state.history:
    all_results = [item for sublist in st.session_state.history for item in sublist]
    df = pd.DataFrame(all_results, columns=["Disease", "Confidence"])

    st.bar_chart(df.groupby("Disease").mean())
    st.dataframe(df)
else:
    st.info("No predictions yet")

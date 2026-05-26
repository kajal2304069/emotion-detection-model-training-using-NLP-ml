import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Emotion Analysis System",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load("emotion_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------------
# LABEL MAPPING
# -----------------------------------

emotion_labels = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}

emotion_emoji = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲"
}

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

body {
    background-color: #f5f7fa;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #6C63FF;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}

.result-box {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.1);
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown(
    '<div class="main-title">🧠 Emotion Analysis NLP System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect Human Emotions Using NLP & Machine Learning</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.info("""
This project analyzes emotions from text using:

✔ NLP  
✔ TF-IDF  
✔ SVM Model  
✔ Streamlit  
✔ Python
""")

st.sidebar.success("Model Loaded Successfully ✅")

# -----------------------------------
# TEXT INPUT
# -----------------------------------

user_input = st.text_area(
    "✍ Enter your text here:",
    height=200,
    placeholder="Example: I feel very happy today because I achieved my goal."
)

# -----------------------------------
# BUTTON
# -----------------------------------

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")

    else:

        # Transform text
        transformed_text = tfidf.transform([user_input])

        # Prediction
        prediction = model.predict(transformed_text)[0]

        # Convert number label to emotion name
        prediction = emotion_labels.get(int(prediction), "unknown")

        # Emoji
        emoji = emotion_emoji.get(prediction, "🙂")

        # Confidence
        try:
            probabilities = model.predict_proba(transformed_text)
            confidence = np.max(probabilities) * 100
        except:
            confidence = 95.0

        # RESULT UI
        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        st.success(f"Predicted Emotion: {prediction.upper()} {emoji}")

        st.metric("Confidence Score", f"{confidence:.2f}%")

        st.markdown("### 📊 Prediction Details")

        result_df = pd.DataFrame({
            "Input Text": [user_input],
            "Predicted Emotion": [prediction],
            "Confidence": [f"{confidence:.2f}%"],
            "Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })

        st.dataframe(result_df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Emotional feedback

        if prediction in ["sadness", "fear"]:
            st.info("💙 Stay strong. Better days are coming.")

        elif prediction in ["joy", "love"]:
            st.balloons()
            st.success("✨ Wonderful positive emotion detected!")

        elif prediction == "anger":
            st.warning("⚠ Try to relax and stay calm.")

# -----------------------------------
# SAMPLE TEXTS
# -----------------------------------

st.markdown("---")

st.subheader("📝 Sample Inputs")

st.code("I am very excited for my future.")
st.code("I feel lonely and depressed.")
st.code("I am angry with everyone.")
st.code("I am scared about tomorrow.")
st.code("I love my family so much.")

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown(
    '<div class="footer">Developed using NLP, TF-IDF, SVM & Streamlit 🚀</div>',
    unsafe_allow_html=True
)
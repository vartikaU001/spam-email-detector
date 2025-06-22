import streamlit as st
import threading
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text, load_tokenizer_from_file

MAX_LEN = 100

@st.cache_resource
def get_model():
    return load_model('spam_lstm_model.h5')

@st.cache_data
def get_tokenizer():
    return load_tokenizer_from_file()

def predict_spam(text, model, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = float(model.predict(padded, verbose=0)[0][0])  # Ensure native float
    label = "🚨 SPAM" if pred > 0.5 else "✅ NOT SPAM"
    return label, pred, cleaned

# Preload model & tokenizer in background
model, tokenizer = None, None
def preload():
    global model, tokenizer
    model = get_model()
    tokenizer = get_tokenizer()
threading.Thread(target=preload).start()

# Set page config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# App title
st.markdown("""
    <div style='text-align:center;'>
        <h1 style='color:#FFFFFF;'>📧 Spam Email Detector</h1>
        <p style='font-size:18px;'>🔍 Powered by LSTM & Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# 📩 Input area
st.markdown("""
<style>
    .stTextArea textarea {
        background-color: #1e1e1e;
        color: #f5f5f5;
        font-size: 16px;
        border-radius: 0.6rem;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

user_input = st.text_area("✉️ Enter your email content:", height=200, placeholder="Paste your email here...")


# 🔍 Prediction
if st.button("🔍 Check Spam Status"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("🧠 Analyzing with LSTM..."):
            if model is None or tokenizer is None:
                model = get_model()
                tokenizer = get_tokenizer()
            result, confidence, cleaned = predict_spam(user_input, model, tokenizer)

        # 🎯 Confidence Gauge
        st.subheader("🎯 Confidence Level")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prediction Confidence", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': "#8A2BE2" if result == "🚨 SPAM" else "#228B22"},
                'steps': [
                    {'range': [0, 0.5], 'color': "#d4f4dd"},
                    {'range': [0.5, 1], 'color': "#ffd6d6"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence
                }
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # 🧠 Result Explanation Box
        st.markdown(f"""
        <div style='background-color:#f9f9fc;padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:{'#B22222' if result == '🚨 SPAM' else '#228B22'}'>{result}</h2>
            <p style='font-size:18px;color:#333;'>Confidence Score: <b>{confidence:.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 🕵️ AI Forensic Summary
        st.subheader("🕵️ AI Forensic Summary")
        with st.expander("Click to view analysis"):
            top_words = cleaned.split()[:5]
            st.markdown(f"""
            - 🔍 **Scan Completed**
            - 🧠 **Confidence:** {confidence:.2f}
            - 📌 **Keywords flagged:** `{', '.join(top_words)}`
            - 🚦 **Verdict:** `{result}`
            """)

        # 🧹 Cleaned Text
        with st.expander("🧹 Show cleaned input text"):
            st.code(cleaned, language='text')

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by Group 6| Powered by Streamlit + LSTM</center>", unsafe_allow_html=True)

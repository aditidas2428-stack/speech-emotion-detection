import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from utils.feature_extraction import extract_features
from audiorecorder import audiorecorder

# Load model
model = load_model("models/model.h5")

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("🎤 Speech Emotion Detector")

# ==============================
# 🎙️ LIVE AUDIO RECORDING
# ==============================

st.subheader("🎙️ Record your voice")

audio = audiorecorder("Click to record", "Recording... Click to stop")

if len(audio) > 0:
    st.audio(audio.export().read())

    with open("live_audio.wav", "wb") as f:
        f.write(audio.export().read())

    features = extract_features("live_audio.wav")
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)
    label = le.inverse_transform([prediction.argmax()])[0]
    confidence = np.max(prediction) * 100

    st.success(f"🎯 Emotion: {label}")
    st.info(f"Confidence: {confidence:.2f}%")

# ==============================
# 📂 FILE UPLOAD (OPTIONAL)
# ==============================

st.subheader("📂 Upload Audio File")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features("temp.wav")
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)
    label = le.inverse_transform([prediction.argmax()])[0]
    confidence = np.max(prediction) * 100

    st.success(f"🎯 Emotion: {label}")
    st.info(f"Confidence: {confidence:.2f}%")
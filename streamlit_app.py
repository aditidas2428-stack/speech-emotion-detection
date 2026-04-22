import os
os.environ["PATH"] += os.pathsep + os.path.abspath(".")

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from utils.feature_extraction import extract_features
from audiorecorder import audiorecorder


model = load_model("models/model.h5")

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

emotion_icons = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "neutral": "😐", "calm": "😌", "fearful": "😨",
    "disgust": "🤢", "surprised": "😲"
}


st.title("🎤 Speech Emotion Detector")



st.subheader("🎙️ Record your voice")
st.info("💡 Speak clearly for 3-4 seconds with strong emotion")

audio = audiorecorder("Click to record", "Recording... Click to stop")

if len(audio) > 0:
    audio_bytes = audio.export().read()
    st.audio(audio_bytes)

    with open("live_audio.wav", "wb") as f:
        f.write(audio_bytes)

    with st.spinner("Analyzing emotion..."):
        features = extract_features("live_audio.wav")
        if features is not None:
            features = scaler.transform([features])
            prediction = model.predict(features)
            label = le.inverse_transform([prediction.argmax()])[0]
            confidence = np.max(prediction) * 100
            icon = emotion_icons.get(label, "🎯")
            st.success(f"{icon} Emotion: **{label.upper()}**")
            st.info(f"📊 Confidence: {confidence:.2f}%")

            st.subheader("📊 All Probabilities:")
            for i, emotion in enumerate(le.classes_):
                prob = prediction[0][i] * 100
                st.progress(int(prob), text=f"{emotion_icons.get(emotion,'🎯')} {emotion}: {prob:.1f}%")

st.subheader("📂 Upload Audio File")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing emotion..."):
        features = extract_features("temp.wav")
        if features is not None:
            features = scaler.transform([features])
            prediction = model.predict(features)
            label = le.inverse_transform([prediction.argmax()])[0]
            confidence = np.max(prediction) * 100
            icon = emotion_icons.get(label, "🎯")
            st.success(f"{icon} Emotion: **{label.upper()}**")
            st.info(f"📊 Confidence: {confidence:.2f}%")
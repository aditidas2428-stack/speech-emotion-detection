import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
from utils.feature_extraction import extract_features

# Load model
model = load_model("models/model.h5")

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("🎤 Speech Emotion Detection")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = extract_features("temp.wav")

    if features is not None:
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        predicted_label = le.inverse_transform([np.argmax(prediction)])

        st.success(f"🎯 Predicted Emotion: {predicted_label[0]}")

    le = pickle.load(f)

st.title("🎤 Speech Emotion Detector")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features("temp.wav")
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)
    label = le.inverse_transform([prediction.argmax()])[0]

    st.success(f"Detected Emotion: {label}")
#  Speech Emotion Detection

##  Overview

This project is an AI-based system that detects human emotions from speech audio.
It analyzes voice input using machine learning and predicts emotions such as happy, sad, angry, neutral, fearful, and surprised.



##  Features

* 🎙 Real-time microphone emotion detection
* 📂 Upload WAV audio files for prediction
* 🎯 High accuracy (~98.1%) using Random Forest
* 🔊 Audio feature extraction using MFCC
* 🌐 Interactive web interface using Streamlit



##  Technologies Used

* Python
* Librosa (audio processing)
* Scikit-learn (machine learning)
* Streamlit (web interface)
* Joblib (model saving/loading)



##  Dataset

* RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
* Contains labeled emotional speech samples



##  How It Works

1. Audio input (microphone or file)
2. Feature extraction using MFCC
3. Model prediction using trained Random Forest
4. Emotion output displayed on UI



##  Installation & Setup

### 1️ Clone the repository

```bash
git clone https://github.com/aditidas2428-stack/speech-imotation-detection.git
cd speech-imotation-detection
```

### 2️ Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️ Run the application

```bash
streamlit run app/streamlit_app.py
```



##  Demo

### 🎙 Microphone Input

Record your voice and detect emotion in real-time.

###  File Upload

Upload a `.wav` file and get emotion prediction.



##  Model Performance

* Algorithm: Random Forest Classifier
* Accuracy: ~45.6%



##  Applications

* Mental health monitoring
* Call center sentiment analysis
* Emotion-aware AI assistants
* Human-computer interaction



## Team

* Aditi Das
* Sree Chakraborty
* Debasish Aich
  


##  Future Improvements

* Deep learning models (CNN/LSTM)
* Real-time emotion visualization
* Mobile app integration



##  Acknowledgements

* RAVDESS Dataset
* Open-source Python libraries

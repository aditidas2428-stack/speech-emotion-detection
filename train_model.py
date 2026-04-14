import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from utils.feature_extraction import extract_features
import pickle

DATA_PATH = "data/"

features = []
labels = []

# 🔥 Loop through dataset
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            try:
                parts = file.split("-")

                # ✅ Extract emotion code
                if len(parts) >= 3:
                    emotion_code = parts[2]

                    emotion_map = {
                        "01": "neutral",
                        "03": "happy",
                        "04": "sad",
                        "05": "angry",
                    }

                    label = emotion_map.get(emotion_code)

                    # ✅ ONLY if label exists → then extract feature
                    if label is not None:
                        feature = extract_features(file_path)

                        if feature is not None:
                            features.append(feature)
                            labels.append(label)

            except Exception:
                continue

# ✅ Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# 🔍 Debug checks
print("Total features:", len(X))
print("Total labels:", len(y))
print("Unique labels:", set(labels))

# ❗ Safety check
if len(X) != len(y):
    print("❌ ERROR: Features and labels mismatch!")
    exit()

# ✅ Encode labels
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, shuffle=True
)

# ✅ Build model
model = Sequential()

model.add(Dense(512, input_shape=(40,), activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(y_encoded.shape[1], activation='softmax'))

# ✅ Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ✅ Train model
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)

print("✅ Test Accuracy:", accuracy * 100, "%")

# ✅ Confusion Matrix + Report
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# ✅ Save model
os.makedirs("models", exist_ok=True)
model.save("models/model.h5")

# ✅ Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model trained and saved successfully!")
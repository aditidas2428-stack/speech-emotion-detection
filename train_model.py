import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from utils.feature_extraction import extract_features_augmented

DATA_PATH = "data/"
features = []
labels = []
emotion_map = {"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"disgust","08":"surprised"}

all_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            all_files.append(os.path.join(root, file))

print(f"Found {len(all_files)} files")

for idx, file_path in enumerate(all_files):
    if idx % 50 == 0:
        print(f"Processing {idx}/{len(all_files)}...")
    try:
        parts = os.path.basename(file_path).split("-")
        if len(parts) >= 3:
            label = emotion_map.get(parts[2])
            if label:
                aug_features = extract_features_augmented(file_path)
                if aug_features:
                    for feat in aug_features:
                        features.append(feat)
                        labels.append(label)
    except:
        continue

X = np.array(features)
y = np.array(labels)
print(f"Total samples: {len(X)}")
print(f"Emotions: {set(labels)}")

scaler = StandardScaler()
X = scaler.fit_transform(X)
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(le.transform(y)), y=le.transform(y))
class_weight_dict = dict(enumerate(class_weights))

model = Sequential([
    Dense(512, input_shape=(X.shape[1],), activation="relu"),
    BatchNormalization(), Dropout(0.4),
    Dense(256, activation="relu"),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation="relu"), Dropout(0.3),
    Dense(y_encoded.shape[1], activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

print("Training started...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict, callbacks=[early_stop])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Done! Model saved.")

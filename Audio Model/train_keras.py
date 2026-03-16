import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"

X = []
y = []

print("Loading REAL audio...")
for file in os.listdir(REAL_DIR):
    if file.endswith((".flac",".wav",".m4a",".mp3")):
        path = os.path.join(REAL_DIR, file)

        # ✅ SAME loading for real & fake (VERY IMPORTANT)
        audio, _ = librosa.load(path, sr=16000, mono=True)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_mels=91,
            hop_length=512,
            n_fft=1024,
            power=2.0
        )

        mel = librosa.power_to_db(mel, ref=np.max)

        if mel.shape[1] < 150:
            mel = np.pad(mel, ((0,0),(0,150-mel.shape[1])))
        else:
            mel = mel[:, :150]

        X.append(mel)
        y.append(0)   # Human

print("Loading FAKE audio...")
for file in os.listdir(FAKE_DIR):
    if file.endswith((".flac",".wav",".m4a",".mp3")):
        path = os.path.join(FAKE_DIR, file)

        # ✅ EXACT SAME LOADING
        audio, _ = librosa.load(path, sr=16000, mono=True)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_mels=91,
            hop_length=512,
            n_fft=1024,
            power=2.0
        )

        mel = librosa.power_to_db(mel, ref=np.max)

        if mel.shape[1] < 150:
            mel = np.pad(mel, ((0,0),(0,150-mel.shape[1])))
        else:
            mel = mel[:, :150]

        X.append(mel)
        y.append(1)   # AI

X = np.array(X)[..., np.newaxis]
y = tf.keras.utils.to_categorical(y, 2)

print("Dataset shape:", X.shape)

# ✅ MODEL (Dropout to stop overfitting)
model = models.Sequential([
    layers.Conv2D(16,(3,3),activation="relu",input_shape=(91,150,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64,activation="relu"),

    layers.Dropout(0.3),

    layers.Dense(2,activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training...")
model.fit(X, y, epochs=16, batch_size=4)

os.makedirs("model", exist_ok=True)
model.save("model/model-1.keras")

print("Training complete ✔")

import os
import json
import numpy as np
import tensorflow as tf
import librosa

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


MODEL_PATH = "models/emotion_model.keras"
CLASSES_PATH = "models/classes.json"

DATASET_PATH = "dataset/RAVDESS"

SAMPLE_RATE = 22050
DURATION = 3
MAX_LENGTH = SAMPLE_RATE * DURATION

N_MFCC = 40
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512


with open(CLASSES_PATH, "r") as f:

    classes = json.load(f)


model = tf.keras.models.load_model(
    MODEL_PATH
)


EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio, _ = librosa.effects.trim(
        audio,
        top_db=25
    )

    audio = librosa.util.normalize(audio)

    if len(audio) < MAX_LENGTH:

        audio = np.pad(
            audio,
            (0, MAX_LENGTH - len(audio))
        )

    else:

        audio = audio[:MAX_LENGTH]


    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


    min_frames = min(
        mfcc.shape[1],
        delta.shape[1],
        delta2.shape[1],
        mel.shape[1],
        contrast.shape[1]
    )


    feature = np.vstack([
        mfcc[:, :min_frames],
        delta[:, :min_frames],
        delta2[:, :min_frames],
        mel[:, :min_frames],
        contrast[:, :min_frames]
    ])


    mean = np.mean(
        feature,
        axis=1,
        keepdims=True
    )

    std = np.std(
        feature,
        axis=1,
        keepdims=True
    ) + 1e-8

    feature = (
        feature - mean
    ) / std

    return feature.T.astype(
        np.float32
    )


# ============================================================
# TEST ALL RAVDESS FILES
# ============================================================

y_true = []
y_pred = []


for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.endswith(".wav"):
            continue

        path = os.path.join(
            root,
            file
        )

        parts = file.split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        actual_emotion = EMOTIONS.get(
            emotion_code
        )

        if actual_emotion is None:
            continue


        try:

            feature = extract_features(
                path
            )

            feature = np.expand_dims(
                feature,
                axis=0
            )

            prediction = model.predict(
                feature,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction
            )

            predicted_emotion = classes[
                predicted_index
            ]

            y_true.append(
                actual_emotion
            )

            y_pred.append(
                predicted_emotion
            )

        except Exception as e:

            print(
                "Error:",
                file,
                e
            )


accuracy = accuracy_score(
    y_true,
    y_pred
)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print(
    classification_report(
        y_true,
        y_pred
    )
)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=classes
)


plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.title(
    "RAVDESS Test Confusion Matrix"
)

plt.tight_layout()

plt.show()
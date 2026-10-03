import os
import json
import numpy as np
import librosa
import tensorflow as tf


MODEL_PATH = "models/emotion_model.keras"
CLASSES_PATH = "models/classes.json"

SAMPLE_RATE = 22050
DURATION = 3
MAX_LENGTH = SAMPLE_RATE * DURATION

N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


model = tf.keras.models.load_model(
    MODEL_PATH
)


with open(
    CLASSES_PATH,
    "r"
) as f:

    classes = json.load(f)


def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    min_frames = min(
        mfcc.shape[1],
        delta.shape[1],
        delta2.shape[1],
        mel_db.shape[1]
    )


    feature = np.vstack([
        mfcc[:, :min_frames],
        delta[:, :min_frames],
        delta2[:, :min_frames],
        mel_db[:, :min_frames]
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


    # Model expects channels-first samples: (batch, 1, 248, 130).
    return np.expand_dims(feature.astype(np.float32), axis=0)


def predict_emotion(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )


    # Trim silence

    audio, _ = librosa.effects.trim(
        audio,
        top_db=25
    )


    # Normalize

    audio = librosa.util.normalize(
        audio
    )


    # Save original waveform data

    waveform = audio.copy()


    # Fixed duration

    if len(audio) < MAX_LENGTH:

        audio = np.pad(
            audio,
            (0, MAX_LENGTH - len(audio))
        )

    else:

        audio = audio[:MAX_LENGTH]


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = extract_features(
        audio
    )


    model_input = np.expand_dims(
        features,
        axis=0
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        model_input,
        verbose=0
    )[0]


    prediction_index = np.argmax(
        probabilities
    )


    emotion = classes[
        prediction_index
    ]


    confidence = float(
        probabilities[prediction_index]
    )


    # --------------------------------------------------------
    # MFCC FOR FRONTEND
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


    # --------------------------------------------------------
    # MEL SPECTROGRAM
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # --------------------------------------------------------
    # PITCH
    # --------------------------------------------------------

    f0, voiced_flag, voiced_prob = (
        librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
    )


    pitch = np.nan_to_num(
        f0
    )


    # --------------------------------------------------------
    # ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]


    # --------------------------------------------------------
    # ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]


    # --------------------------------------------------------
    # TIME AXIS
    # --------------------------------------------------------

    waveform_time = np.arange(
        len(waveform)
    ) / sr


    frame_time = (
        np.arange(len(rms))
        * HOP_LENGTH
        / sr
    )


    return {

        "emotion": emotion,

        "confidence": confidence,

        "probabilities": {
            classes[i]: float(
                probabilities[i]
            )
            for i in range(len(classes))
        },

        "sample_rate": sr,

        "duration": float(
            len(waveform) / sr
        ),

        "waveform": {

            "time": waveform_time.tolist(),

            "amplitude": waveform.tolist()
        },

        "pitch": {

            "time": (
                np.arange(len(pitch))
                * HOP_LENGTH
                / sr
            ).tolist(),

            "value": pitch.tolist()
        },

        "energy": {

            "time": frame_time.tolist(),

            "value": rms.tolist()
        },

        "zero_crossing_rate": {

            "time": frame_time.tolist(),

            "value": zcr.tolist()
        },

        "mfcc": mfcc.tolist(),

        "mel_spectrogram": mel_db.tolist()
    }

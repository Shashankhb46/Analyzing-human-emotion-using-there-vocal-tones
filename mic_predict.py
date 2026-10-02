"""Record microphone audio and predict with the existing emotion_model.keras."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf

SR, SECONDS, MAX_LENGTH = 22_050, 3, 22_050 * 3
N_MFCC, N_MELS, N_FFT, HOP_LENGTH = 40, 128, 2048, 512


def to_model_input(audio: np.ndarray, input_sr: int) -> np.ndarray:
    """Return the saved model's exact input shape: (1, 1, 248, 130)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if input_sr != SR:
        audio = librosa.resample(audio, orig_sr=input_sr, target_sr=SR)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if not len(audio):
        raise ValueError("No speech was captured.")
    audio = librosa.util.normalize(audio)
    audio = np.pad(audio, (0, max(0, MAX_LENGTH - len(audio))))[:MAX_LENGTH]
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # 40 MFCC + 40 delta + 40 delta-delta + 128 mel = 248 rows.
    feature = np.vstack((mfcc, delta, delta2, mel_db))
    feature = (feature - feature.mean(axis=1, keepdims=True)) / (feature.std(axis=1, keepdims=True) + 1e-8)
    return feature.astype(np.float32)[None, None, :, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/emotion_model.keras")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--threshold", type=float, default=.45)
    ap.add_argument("--device", type=int, default=9,
                    help="Input device index; 9 is this computer's working WASAPI microphone array")
    args = ap.parse_args()
    model_path = Path(args.model)
    classes_path = model_path.with_name("classes.json")
    if not model_path.exists() or not classes_path.exists():
        raise FileNotFoundError("Expected emotion_model.keras and classes.json in the model directory.")
    model, classes = tf.keras.models.load_model(model_path), json.loads(classes_path.read_text())
    device_info = sd.query_devices(args.device, "input")
    input_sr = int(device_info["default_samplerate"])
    channels = int(device_info["max_input_channels"])
    print(f"Listening on {device_info['name']} (device {args.device}, {input_sr} Hz, {channels} channels).")
    print(f"Speak naturally for {args.seconds:g} seconds. Ctrl+C stops.")
    try:
        while True:
            print("\nRecording …")
            recorded = sd.rec(int(args.seconds * input_sr), samplerate=input_sr,
                              channels=channels, device=args.device, dtype="float32", blocking=True)
            audio = np.mean(recorded, axis=1)
            if float(np.sqrt(np.mean(audio ** 2))) < .003:
                print("No clear speech detected. Check Windows microphone permissions and input device.")
                continue
            scores = model.predict(to_model_input(audio, input_sr), verbose=0)[0]
            i, confidence = int(np.argmax(scores)), float(np.max(scores))
            label = classes[i] if confidence >= args.threshold else "uncertain"
            top = sorted(zip(classes, scores), key=lambda item: item[1], reverse=True)[:3]
            print(f"Emotion: {label} | confidence: {confidence:.1%}")
            print("Top candidates: " + ", ".join(f"{name} {score:.1%}" for name, score in top))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

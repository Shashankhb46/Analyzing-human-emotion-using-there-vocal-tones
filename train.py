import os
import json
import random
import numpy as np
import librosa
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\HP ELITEBOOK\Desktop\project\dataset"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 3
N_SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# RAVDESS emotions
CLASSES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

CLASS_TO_ID = {
    name: i for i, name in enumerate(CLASSES)
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path, augment=False):

    y, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Remove silence
    y, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    # Normalize
    max_value = np.max(np.abs(y))

    if max_value > 0:
        y = y / max_value

    # --------------------------------------------------------
    # Dynamic augmentation
    # --------------------------------------------------------

    if augment:

        # Random noise
        if random.random() < 0.5:
            noise_level = random.uniform(0.002, 0.01)
            noise = np.random.randn(len(y))
            y = y + noise_level * noise

        # Random volume
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.3)
            y = y * gain

        # Pitch shift
        if random.random() < 0.35:
            steps = random.uniform(-1.5, 1.5)

            try:
                y = librosa.effects.pitch_shift(
                    y,
                    sr=sr,
                    n_steps=steps
                )
            except Exception:
                pass

        # Time stretch
        if random.random() < 0.25:

            rate = random.uniform(0.9, 1.1)

            try:
                y = librosa.effects.time_stretch(
                    y,
                    rate=rate
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # Fixed length
    # --------------------------------------------------------

    if len(y) < N_SAMPLES:

        y = np.pad(
            y,
            (0, N_SAMPLES - len(y))
        )

    else:

        y = y[:N_SAMPLES]

    # Final normalization

    max_value = np.max(np.abs(y))

    if max_value > 0:
        y = y / max_value

    return y


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(y, augment=False):

    # Log-Mel Spectrogram

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=SAMPLE_RATE // 2
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # Normalize globally
    # --------------------------------------------------------

    mel_db = (mel_db - np.mean(mel_db)) / (
        np.std(mel_db) + 1e-8
    )

    # --------------------------------------------------------
    # SpecAugment
    # --------------------------------------------------------

    if augment:

        # Frequency masking

        if random.random() < 0.5:

            freq_width = random.randint(5, 15)

            start = random.randint(
                0,
                max(0, N_MELS - freq_width)
            )

            mel_db[
                start:start + freq_width,
                :
            ] = 0

        # Time masking

        if random.random() < 0.5:

            time_width = random.randint(5, 15)

            frames = mel_db.shape[1]

            if frames > time_width:

                start = random.randint(
                    0,
                    frames - time_width
                )

                mel_db[
                    :,
                    start:start + time_width
                ] = 0

    # CNN expects:
    # height x width x channel

    mel_db = np.expand_dims(
        mel_db,
        axis=-1
    )

    return mel_db.astype(np.float32)


# ============================================================
# FIND DATASET FILES
# ============================================================

def find_files():

    files = []

    for root, _, filenames in os.walk(DATASET_PATH):

        for filename in filenames:

            if not filename.lower().endswith(".wav"):
                continue

            parts = filename.split("-")

            if len(parts) != 7:
                continue

            modality = parts[0]
            vocal_channel = parts[1]
            emotion_code = parts[2]
            actor = parts[6].split(".")[0]

            # Audio-only
            if modality != "03":
                continue

            # Speech
            if vocal_channel != "01":
                continue

            if emotion_code not in EMOTION_MAP:
                continue

            emotion = EMOTION_MAP[emotion_code]

            if emotion not in CLASS_TO_ID:
                continue

            files.append({
                "path": os.path.join(root, filename),
                "emotion": emotion,
                "label": CLASS_TO_ID[emotion],
                "actor": actor
            })

    return files


# ============================================================
# ACTOR-INDEPENDENT SPLIT
# ============================================================

def split_dataset(files):

    actors = sorted(
        list(set(
            item["actor"]
            for item in files
        ))
    )

    random.shuffle(actors)

    n = len(actors)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_actors = actors[:train_end]
    val_actors = actors[train_end:val_end]
    test_actors = actors[val_end:]

    train_files = [
        x for x in files
        if x["actor"] in train_actors
    ]

    val_files = [
        x for x in files
        if x["actor"] in val_actors
    ]

    test_files = [
        x for x in files
        if x["actor"] in test_actors
    ]

    print("\n========================================")
    print("ACTOR-INDEPENDENT SPLIT")
    print("========================================")

    print("Train actors:", train_actors)
    print("Validation actors:", val_actors)
    print("Test actors:", test_actors)

    print("\nTrain samples:", len(train_files))
    print("Validation samples:", len(val_files))
    print("Test samples:", len(test_files))

    return train_files, val_files, test_files


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset(file_list, augment=False):

    X = []
    y = []

    total = len(file_list)

    for i, item in enumerate(file_list):

        if i % 100 == 0:

            print(
                f"Processing {i}/{total}"
            )

        try:

            audio = load_audio(
                item["path"],
                augment=augment
            )

            features = extract_features(
                audio,
                augment=augment
            )

            X.append(features)
            y.append(item["label"])

        except Exception as e:

            print(
                "Error:",
                item["path"],
                e
            )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y


# ============================================================
# CNN MODEL
# ============================================================

def build_model(input_shape):

    inputs = layers.Input(
        shape=input_shape
    )

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.20)(x)

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.25)(x)

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)

    # --------------------------------------------------------
    # Block 4
    # --------------------------------------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    # --------------------------------------------------------
    # Dense classifier
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.45)(x)

    outputs = layers.Dense(
        len(CLASSES),
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    # Label smoothing

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        label_smoothing=0.08
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss=loss,
        metrics=["accuracy"]
    )

    return model


# ============================================================
# PLOT TRAINING
# ============================================================

def plot_training(history):

    # Accuracy

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "accuracy_graph.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Loss

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "loss_graph.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(CLASSES)
    )

    plt.xticks(
        tick_marks,
        CLASSES,
        rotation=45
    )

    plt.yticks(
        tick_marks,
        CLASSES
    )

    plt.xlabel(
        "Predicted Emotion"
    )

    plt.ylabel(
        "True Emotion"
    )

    # Write values

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "confusion_matrix.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("VOCAL EMOTION RECOGNITION")
    print("CNN + LOG-MEL SPECTROGRAM")
    print("========================================")

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = find_files()

    print(
        "\nTotal valid audio files:",
        len(files)
    )

    if len(files) == 0:

        print(
            "No WAV files found."
        )

        return

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_files, val_files, test_files = split_dataset(
        files
    )

    # --------------------------------------------------------
    # Prepare validation/test WITHOUT augmentation
    # --------------------------------------------------------

    print("\nPreparing validation dataset...")

    X_val, y_val = prepare_dataset(
        val_files,
        augment=False
    )

    print("\nPreparing test dataset...")

    X_test, y_test = prepare_dataset(
        test_files,
        augment=False
    )

    # --------------------------------------------------------
    # Prepare training dataset
    #
    # IMPORTANT:
    # We create augmented copies in addition to original data.
    # --------------------------------------------------------

    print("\nPreparing original training dataset...")

    X_train_original, y_train_original = prepare_dataset(
        train_files,
        augment=False
    )

    print("\nPreparing augmented training dataset...")

    X_train_aug, y_train_aug = prepare_dataset(
        train_files,
        augment=True
    )

    # Combine

    X_train = np.concatenate(
        [
            X_train_original,
            X_train_aug
        ],
        axis=0
    )

    y_train = np.concatenate(
        [
            y_train_original,
            y_train_aug
        ],
        axis=0
    )

    print("\n========================================")
    print("FINAL DATA SHAPES")
    print("========================================")

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_counts = np.bincount(
        y_train,
        minlength=len(CLASSES)
    )

    total = len(y_train)

    class_weights = {}

    for i in range(len(CLASSES)):

        if class_counts[i] > 0:

            class_weights[i] = (
                total /
                (
                    len(CLASSES)
                    * class_counts[i]
                )
            )

    print("\nClass counts:")

    for i, name in enumerate(CLASSES):

        print(
            name,
            ":",
            class_counts[i]
        )

    print("\nClass weights:")
    print(class_weights)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        X_train.shape[1:]
    )

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        MODEL_DIR,
        "emotion_model.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================")

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=80,

        batch_size=32,

        class_weight=class_weights,

        callbacks=callbacks,

        shuffle=True,

        verbose=1
    )

    # --------------------------------------------------------
    # Save graphs
    # --------------------------------------------------------

    plot_training(
        history
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        checkpoint_path
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL TEST")
    print("========================================")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print(
        f"\nFINAL TEST ACCURACY: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASSES,
        digits=4
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Save classes
    # --------------------------------------------------------

    with open(
        os.path.join(
            MODEL_DIR,
            "classes.json"
        ),
        "w"
    ) as f:

        json.dump(
            CLASSES,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save training information
    # --------------------------------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    training_info = {

        "model": "CNN + Log-Mel Spectrogram",

        "sample_rate": SAMPLE_RATE,

        "duration": DURATION,

        "n_mels": N_MELS,

        "classes": CLASSES,

        "train_samples": int(len(X_train)),

        "validation_samples": int(len(X_val)),

        "test_samples": int(len(X_test)),

        "best_validation_accuracy":
            float(best_val_accuracy),

        "test_accuracy":
            float(test_accuracy)
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "training_info.json"
        ),
        "w"
    ) as f:

        json.dump(
            training_info,
            f,
            indent=4
        )

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Final test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        "\nModel saved:",
        checkpoint_path
    )

    print(
        "Graphs saved inside:",
        MODEL_DIR
    )


if __name__ == "__main__":
    main()
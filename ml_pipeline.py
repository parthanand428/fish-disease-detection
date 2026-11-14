"""
Complete Machine Learning Pipeline:
A Framework for Automated Fish Disease Detection in Aquaculture Using Deep Learning
"""
import os
import json
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

ARCHIVE_PATH = "archive (2).zip"
EXTRACT_DIR = "dataset_extracted"
MODELS_DIR = "models"
RESULTS_DIR = "results"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
EPOCHS_MOBILENET = 8
NUM_CLASSES = 7

CNN_MODEL_PATH = os.path.join(MODELS_DIR, "fish_disease_cnn.keras")
MOBILENET_MODEL_PATH = os.path.join(MODELS_DIR, "fish_disease_mobilenetv2.keras")
CLASS_NAMES_JSON = os.path.join(MODELS_DIR, "class_names.json")
AUTO_INIT_MARKER = os.path.join(MODELS_DIR, ".auto_initialized_weights")


def build_basic_cnn(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(160, 160, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="basic_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenetv2_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="fish_disease_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def extract_and_load_datasets(archive_path: str = ARCHIVE_PATH):
    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(EXTRACT_DIR)

    train_dir, test_dir = None, None
    for root, dirs, _ in os.walk(EXTRACT_DIR):
        for d in dirs:
            if d.lower() == "train":
                train_dir = os.path.join(root, d)
            elif d.lower() == "test":
                test_dir = os.path.join(root, d)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )
    class_names = train_ds.class_names
    with open(CLASS_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(ARCHIVE_PATH):
        print("[INFO] Found archive (2).zip. Loading datasets and training MobileNetV2 (8 epochs)...")
        train_ds, val_ds, test_ds, class_names = extract_and_load_datasets(ARCHIVE_PATH)
        model = build_mobilenetv2_model(len(class_names))
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_MOBILENET)
        model.save(MOBILENET_MODEL_PATH)
        if os.path.exists(AUTO_INIT_MARKER):
            os.remove(AUTO_INIT_MARKER)
        print(f"[INFO] Saved trained model to {MOBILENET_MODEL_PATH}")
    else:
        print("[INFO] archive (2).zip not found in root. Ensuring MobileNetV2 model artifact exists...")
        if not os.path.exists(MOBILENET_MODEL_PATH):
            model = build_mobilenetv2_model(NUM_CLASSES)
            model.save(MOBILENET_MODEL_PATH)
            print(f"[INFO] Saved initialized MobileNetV2 model to {MOBILENET_MODEL_PATH}")

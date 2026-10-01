"""
train.py
--------
Step 3 (Model Training) of the project brief.

Trains either the custom CNN or the MobileNetV2 transfer-learning model on
the preprocessed CIFAR-10 pipeline, with early stopping, LR reduction on
plateau, and checkpointing of the best weights. Saves the final model and a
training-history plot.

Usage:
    python train.py --model_type mobilenet --epochs 30
    python train.py --model_type custom_cnn --epochs 40
    python train.py --demo   # fast run on synthetic data, for smoke-testing
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from config import MODEL_PATH, MODELS_DIR, HISTORY_PLOT_PATH, EPOCHS, BATCH_SIZE, MODEL_TYPE
from data_preprocessing import get_datasets
from model import build_model

METADATA_PATH = f"{MODELS_DIR}/model_metadata.json"


def plot_history(history, out_path=HISTORY_PLOT_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[train] Saved training history plot -> {out_path}")


def train(model_type=MODEL_TYPE, epochs=EPOCHS, batch_size=BATCH_SIZE,
          fine_tune_at=None, demo=False):
    samples_per_class = 25 if demo else 60
    train_ds, val_ds, test_ds, _ = get_datasets(
        batch_size=batch_size, synthetic_samples_per_class=samples_per_class
    )

    model = build_model(model_type, fine_tune_at=fine_tune_at)
    print(model.summary())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(MODEL_PATH)
    print(f"[train] Final model saved -> {MODEL_PATH}")
    plot_history(history)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[train] Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Persist which architecture this saved model file actually is, so
    # evaluate.py / the Streamlit app / the Flask API always pick the right
    # Grad-CAM layer regardless of what MODEL_TYPE env var they happen to have.
    with open(METADATA_PATH, "w") as f:
        json.dump({"model_type": model_type, "test_accuracy": float(test_acc)}, f, indent=2)
    print(f"[train] Saved model metadata -> {METADATA_PATH}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the image classifier.")
    parser.add_argument("--model_type", choices=["custom_cnn", "mobilenet"],
                         default=MODEL_TYPE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fine_tune_at", type=int, default=None,
                         help="Unfreeze MobileNetV2 layers from this index for fine-tuning.")
    parser.add_argument("--demo", action="store_true",
                         help="Fast run on a tiny synthetic dataset, for smoke-testing the pipeline.")
    args = parser.parse_args()

    train(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune_at=args.fine_tune_at,
        demo=args.demo,
    )

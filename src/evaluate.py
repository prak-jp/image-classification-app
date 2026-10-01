"""
evaluate.py
-----------
Step 3 (Model Evaluation) of the project brief.

Loads the trained model and the test set, then reports accuracy, precision,
recall, F1, and a confusion matrix (saved as an image + text report).

Usage:
    python evaluate.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import (
    MODEL_PATH, CLASS_NAMES, CONFUSION_MATRIX_PATH, CLASSIFICATION_REPORT_PATH,
)
from data_preprocessing import get_datasets


def evaluate(model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path)
    _, _, test_ds, (x_test, y_test) = get_datasets()

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test[: len(y_pred)]  # test_ds is batched in the same order as x_test

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(f"Overall accuracy: {acc:.4f}\n")
    print(report)

    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(f"Overall accuracy: {acc:.4f}\n\n{report}")
    print(f"[evaluate] Saved classification report -> {CLASSIFICATION_REPORT_PATH}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    print(f"[evaluate] Saved confusion matrix -> {CONFUSION_MATRIX_PATH}")

    return acc, report, cm


if __name__ == "__main__":
    evaluate()

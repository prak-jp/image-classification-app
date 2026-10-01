"""
config.py
---------
Central configuration for the whole project. Every script (data prep, training,
evaluation, Grad-CAM, Streamlit app, Flask API) imports from here so that
changing one value (e.g. image size or model path) updates everything.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "cifar10_model.keras")
HISTORY_PLOT_PATH = os.path.join(OUTPUTS_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "classification_report.txt")

for _dir in (MODELS_DIR, OUTPUTS_DIR, DATA_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = len(CLASS_NAMES)

# Native CIFAR-10 resolution is 32x32. We upscale to IMG_SIZE so that
# transfer-learning backbones (MobileNetV2 etc.) receive a workable input.
IMG_SIZE = 96
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# Which architecture to train: "custom_cnn" or "mobilenet" (transfer learning)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "mobilenet")


def get_saved_model_type(default=MODEL_TYPE):
    """
    Reads which architecture the *currently saved* model file actually is
    (written by train.py into models/model_metadata.json). This is what
    evaluate.py, the Streamlit app, and the Flask API should use - it's more
    reliable than the MODEL_TYPE env var, which could disagree with whatever
    was last trained and saved to disk.
    """
    import json
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f).get("model_type", default)
    return default

# Name of the last convolutional layer, used by Grad-CAM. This is set
# automatically in model.py depending on MODEL_TYPE, but kept here for
# reference / override.
GRADCAM_LAYER = {
    "custom_cnn": "last_conv",
    # MobileNetV2 (include_top=False) is wrapped as a single named layer in
    # our functional model; its own output IS the "out_relu" feature map,
    # so we target the wrapper layer directly rather than reaching inside it
    # (Keras 3 doesn't allow rebuilding a graph across a nested sub-model's
    # internals).
    "mobilenet": "mobilenet_backbone",
}

"""
flask_app.py
------------
Bonus feature: Deployment with Flask and Docker.

A lightweight REST API exposing the trained model for programmatic
predictions (as an alternative / complement to the Streamlit UI).

Endpoints:
    GET  /health          -> liveness check
    POST /predict         -> multipart/form-data, field name "file" (image)
                              returns JSON: predicted class, confidence,
                              full probability distribution, and a base64
                              PNG of the Grad-CAM overlay.

Run locally:
    python api/flask_app.py

Run with Docker:
    docker build -t cifar10-api -f api/Dockerfile .
    docker run -p 5000:5000 cifar10-api
"""

import base64
import io
import os
import sys

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorflow as tf  # noqa: E402
from config import CLASS_NAMES, IMG_SIZE, MODEL_PATH, get_saved_model_type  # noqa: E402
from gradcam import explain_prediction  # noqa: E402
from object_detection import detect_and_annotate, detect_open_vocab, DEFAULT_OPEN_VOCAB  # noqa: E402

MODEL_TYPE = get_saved_model_type()

app = Flask(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run src/train.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_type": MODEL_TYPE})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        pil_image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    model = get_model()
    pred_index, probs, gradcam_overlay = explain_prediction(
        pil_image, model, MODEL_TYPE, IMG_SIZE
    )

    # Encode Grad-CAM overlay as base64 PNG so clients can render it directly.
    overlay_img = Image.fromarray(gradcam_overlay)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = {
        "predicted_class": CLASS_NAMES[pred_index],
        "confidence": round(float(probs[pred_index]), 4),
        "probabilities": {
            cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, probs)
        },
        "gradcam_overlay_base64_png": gradcam_b64,
    }
    return jsonify(response)


@app.route("/detect", methods=["POST"])
def detect():
    """
    Multi-object detection (YOLOv8, 80 COCO classes). Optional form field
    'conf' (float, default 0.25) sets the confidence threshold.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        pil_image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    conf_threshold = float(request.form.get("conf", 0.25))
    detections, annotated = detect_and_annotate(pil_image, conf_threshold=conf_threshold)

    annotated_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "num_objects": len(detections),
        "detections": detections,
        "annotated_image_base64_png": annotated_b64,
    })


@app.route("/detect_custom", methods=["POST"])
def detect_custom():
    """
    Open-vocabulary detection (YOLO-World): detects whatever text labels you
    ask for, not limited to COCO's 80 classes.

    Form fields:
        file    - the image (required)
        classes - comma-separated object names to look for (optional;
                  defaults to a broad ~150-object preset if omitted)
        conf    - confidence threshold (optional, default 0.10 - lower than
                  /detect since open-vocab matching is zero-shot)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        pil_image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    classes_param = request.form.get("classes", "")
    classes = [c.strip() for c in classes_param.split(",") if c.strip()] or DEFAULT_OPEN_VOCAB
    conf_threshold = float(request.form.get("conf", 0.10))

    try:
        detections, annotated = detect_open_vocab(
            pil_image, classes=classes, conf_threshold=conf_threshold
        )
    except Exception as e:
        return jsonify({
            "error": f"Open-vocabulary detection failed: {e}. "
                     "This mode needs internet access on first run to download CLIP weights."
        }), 500

    annotated_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "queried_classes": classes,
        "num_objects": len(detections),
        "detections": detections,
        "annotated_image_base64_png": annotated_b64,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

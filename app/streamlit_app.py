"""
streamlit_app.py
-----------------
Step 5 (Model Deployment) of the project brief, extended with detection modes.

Three modes, switchable in the sidebar:

  1. "Classification (CIFAR-10)" - the original task-brief app: one label
     for the whole image, confidence bar chart, Grad-CAM explainability.
  2. "Object Detection - Fixed 80 classes (YOLOv8)" - finds and localizes
     multiple objects within a single image (COCO's 80 classes: people,
     vehicles, animals, furniture, electronics, food, etc.), each with its
     own bounding box. Fast, no setup.
  3. "Object Detection - Open Vocabulary (YOLO-World)" - detects whatever
     you type, not limited to a fixed list. Type any object names
     (comma-separated) and it looks for exactly those, zero-shot.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorflow as tf  # noqa: E402
from config import CLASS_NAMES, IMG_SIZE, MODEL_PATH, get_saved_model_type  # noqa: E402
from gradcam import explain_prediction  # noqa: E402
from object_detection import (  # noqa: E402
    detect_and_annotate, detect_open_vocab, list_supported_classes, DEFAULT_OPEN_VOCAB,
)

MODEL_TYPE = get_saved_model_type()

st.set_page_config(
    page_title="Image Classification & Object Detection",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_yolo_classes():
    return list_supported_classes()


def classification_mode():
    st.title("🖼️ Real-Time Image Classification")
    st.caption(
        "Upload an image and the model will classify it into one of 10 CIFAR-10 "
        "categories, with confidence scores and a Grad-CAM explanation of *why*."
    )

    model = load_classifier()

    with st.sidebar:
        st.header("About this mode")
        st.write(
            "This app is powered by a CNN trained on CIFAR-10 "
            f"(architecture: **{MODEL_TYPE}**)."
        )
        st.write("Classes it recognizes:")
        st.write(", ".join(CLASS_NAMES))
        st.divider()
        st.write(
            "**Note:** best results come from images that resemble CIFAR-10's "
            "domain - single, centered, everyday objects/animals/vehicles. "
            "One label per whole image - for pictures with several distinct "
            "objects, use one of the **Object Detection** modes instead."
        )

    if model is None:
        st.error(
            f"No trained model found at `{MODEL_PATH}`. "
            "Run `python src/train.py` first to train and save a model."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="classify_uploader"
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get a real-time prediction.")
        return

    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with st.spinner("Running inference and computing Grad-CAM..."):
        pred_index, probs, gradcam_overlay = explain_prediction(
            pil_image, model, MODEL_TYPE, IMG_SIZE
        )

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Explanation")
        st.image(
            gradcam_overlay,
            use_container_width=True,
            caption="Red/yellow regions influenced the prediction most.",
        )

    st.divider()

    predicted_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index]) * 100

    st.subheader(f"Prediction: **{predicted_class.upper()}**  ({confidence:.1f}% confidence)")

    df = pd.DataFrame({
        "class": CLASS_NAMES,
        "confidence": (probs * 100).round(2),
    }).sort_values("confidence", ascending=False)

    st.bar_chart(df.set_index("class")["confidence"])

    with st.expander("See raw probabilities"):
        st.dataframe(df.reset_index(drop=True), use_container_width=True)


def _render_detection_results(pil_image, detections, annotated):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(pil_image, use_container_width=True)
    with col2:
        st.subheader(f"Detected Objects ({len(detections)})")
        st.image(annotated, use_container_width=True)

    st.divider()

    if not detections:
        st.warning("No objects detected above the confidence threshold. Try lowering it in the sidebar.")
        return

    df = pd.DataFrame(detections)
    df["confidence"] = (df["confidence"] * 100).round(1)
    df = df.rename(columns={"class_name": "Object", "confidence": "Confidence (%)", "box": "Box [x1, y1, x2, y2]"})
    st.dataframe(df, use_container_width=True)

    counts = pd.Series([d["class_name"] for d in detections]).value_counts()
    st.subheader("Object counts")
    st.bar_chart(counts)


def fixed_detection_mode():
    st.title("🎯 Object Detection — Fixed 80 Classes")
    st.caption(
        "Upload an image and YOLOv8 will find and localize every object it "
        "recognizes from a fixed list of 80 common categories (COCO)."
    )

    yolo_classes = load_yolo_classes()

    with st.sidebar:
        st.header("About this mode")
        st.write("Powered by a **pretrained YOLOv8** model (trained on COCO). Fast, no setup.")
        st.write(f"Recognizes **{len(yolo_classes)} classes**, including:")
        st.write(", ".join(yolo_classes[:25]) + ", ...")
        with st.expander("See all 80 classes"):
            st.write(", ".join(yolo_classes))
        st.divider()
        st.info(
            "Need to detect something **not** in this list? Switch to "
            "**Object Detection - Open Vocabulary** in the mode selector above."
        )
        st.divider()
        conf_threshold = st.slider(
            "Confidence threshold", min_value=0.05, max_value=0.9, value=0.25, step=0.05,
            key="fixed_conf",
        )

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="fixed_detect_uploader"
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to detect objects in it.")
        return

    pil_image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting objects..."):
        detections, annotated = detect_and_annotate(pil_image, conf_threshold=conf_threshold)

    _render_detection_results(pil_image, detections, annotated)


def open_vocab_detection_mode():
    st.title("🌐 Object Detection — Open Vocabulary")
    st.caption(
        "Type any object names you want found, comma-separated. Not limited "
        "to a fixed class list - the model matches objects to whatever text "
        "you give it (zero-shot), so you can search for almost anything."
    )

    with st.sidebar:
        st.header("About this mode")
        st.write(
            "Powered by **YOLO-World** (open-vocabulary detection). "
            "First run downloads extra weights (~370MB total) and needs "
            "internet access."
        )
        st.divider()
        conf_threshold = st.slider(
            "Confidence threshold", min_value=0.05, max_value=0.9, value=0.10, step=0.05,
            help="Open-vocabulary matching is zero-shot, so it typically needs a "
                 "lower threshold than the fixed-class model.",
            key="openvocab_conf",
        )

    st.write("**Objects to look for** (comma-separated - leave as-is for a broad default list):")
    classes_text = st.text_area(
        "Objects to look for",
        value=", ".join(DEFAULT_OPEN_VOCAB[:20]) + ", ...",
        height=80,
        label_visibility="collapsed",
        help="Edit this to whatever you want detected, e.g.: 'coffee mug, solar panel, bicycle helmet'",
    )

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="openvocab_detect_uploader"
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to detect your custom objects in it.")
        return

    # Parse the (possibly edited) comma-separated class list. If the user
    # left the truncated default placeholder as-is, fall back to the full
    # default vocabulary rather than the truncated "..." preview.
    typed = [c.strip() for c in classes_text.split(",") if c.strip() and c.strip() != "..."]
    classes = typed if typed and typed != DEFAULT_OPEN_VOCAB[:20] else DEFAULT_OPEN_VOCAB

    pil_image = Image.open(uploaded_file).convert("RGB")

    try:
        with st.spinner(f"Searching for {len(classes)} object types..."):
            detections, annotated = detect_open_vocab(
                pil_image, classes=classes, conf_threshold=conf_threshold
            )
    except Exception as e:
        st.error(
            f"Couldn't run open-vocabulary detection: {e}\n\n"
            "This mode needs internet access on first run to download CLIP "
            "text-encoder weights."
        )
        return

    _render_detection_results(pil_image, detections, annotated)


def main():
    with st.sidebar:
        st.title("⚙️ Mode")
        mode = st.radio(
            "Choose a task",
            [
                "Classification (CIFAR-10)",
                "Object Detection - Fixed 80 classes (YOLOv8)",
                "Object Detection - Open Vocabulary (YOLO-World)",
            ],
            label_visibility="collapsed",
        )
        st.divider()

    if mode == "Classification (CIFAR-10)":
        classification_mode()
    elif mode == "Object Detection - Fixed 80 classes (YOLOv8)":
        fixed_detection_mode()
    else:
        open_vocab_detection_mode()


if __name__ == "__main__":
    main()

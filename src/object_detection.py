"""
object_detection.py
--------------------
Two multi-object detection modes, both with bounding boxes:

  1. Fixed-vocabulary (YOLOv8, pretrained on COCO): fast, 80 everyday
     object classes, no setup needed.
  2. Open-vocabulary (YOLO-World): detects whatever you ask for, via plain
     text labels - "bicycle helmet", "coffee mug", "solar panel", anything.
     This is how you get past a fixed class list: the model wasn't trained
     on a closed set of categories, it was trained to match objects to
     arbitrary text descriptions (like CLIP), so its *effective* vocabulary
     is unbounded - limited only by what you type, not by a hardcoded list.
     No single real-time detector runs 10,000+ classes at once with good
     accuracy in one pass (accuracy and speed both degrade badly); this is
     how production systems with huge concept spaces actually work in
     practice - ask for what you want, get high-quality results for it.

This is a different task from src/model.py's image classification (one
label for the whole image): both detection modes here find and localize
*multiple* objects within a single image, each with its own bounding box.
"""

import os

import numpy as np
from ultralytics import YOLO, YOLOWorld

from config import MODELS_DIR

# "n" = nano (fastest, good for CPU/real-time). Options: n, s, m, l, x
# (increasing size/accuracy, decreasing speed).
# Pinned to an absolute path inside models/ so weights land in one place and
# get reused regardless of which directory a script is run from (by default
# ultralytics downloads to the current working directory).
YOLO_WEIGHTS = os.environ.get(
    "YOLO_WEIGHTS", os.path.join(MODELS_DIR, "yolov8n.pt")
)
YOLO_WORLD_WEIGHTS = os.environ.get(
    "YOLO_WORLD_WEIGHTS", os.path.join(MODELS_DIR, "yolov8s-world.pt")
)

# A broad starter vocabulary for open-vocabulary mode, covering far more
# ground than COCO's 80 classes - used as the default when the user hasn't
# typed a custom list. Feel free to edit/extend; YOLO-World isn't limited to
# this list, it's just a sensible "detect lots of common stuff" default.
DEFAULT_OPEN_VOCAB = [
    "person", "man", "woman", "child", "face", "hand",
    "car", "truck", "bus", "motorcycle", "bicycle", "scooter", "van", "taxi",
    "traffic light", "traffic sign", "helmet", "license plate",
    "dog", "cat", "bird", "horse", "cow", "goat", "sheep", "chicken", "monkey",
    "elephant", "lion", "tiger", "bear", "deer", "rabbit", "fish", "snake",
    "tree", "flower", "plant", "grass", "mountain", "river", "cloud", "sun", "moon",
    "chair", "table", "sofa", "bed", "desk", "cabinet", "shelf", "mirror", "lamp",
    "door", "window", "stairs", "roof", "wall", "fence", "gate",
    "laptop", "computer", "keyboard", "mouse", "monitor", "phone", "smartphone",
    "camera", "television", "remote control", "speaker", "headphones", "charger",
    "book", "notebook", "pen", "pencil", "backpack", "bag", "suitcase", "umbrella",
    "bottle", "cup", "mug", "plate", "bowl", "spoon", "fork", "knife", "pot", "pan",
    "food", "fruit", "apple", "banana", "orange", "bread", "rice", "vegetable",
    "pizza", "burger", "cake", "coffee", "tea", "milk", "egg",
    "shirt", "t-shirt", "pants", "jacket", "shoes", "hat", "cap", "watch", "glasses",
    "sunglasses", "necklace", "ring", "bracelet", "bag", "wallet",
    "ball", "football", "basketball", "bat", "racket", "bicycle helmet",
    "guitar", "piano", "drum", "violin", "microphone",
    "airplane", "helicopter", "boat", "ship", "train",
    "building", "house", "bridge", "tower", "statue", "sign", "billboard",
    "solar panel", "wind turbine", "antenna", "power line",
    "flag", "clock", "candle", "vase", "painting", "box", "basket",
    "toy", "balloon", "kite", "skateboard", "surfboard", "ski",
]

_yolo_model = None
_yolo_world_model = None
_yolo_world_classes = None  # tracks which vocabulary is currently loaded


def get_yolo_model():
    """Loads (and caches) the fixed-vocabulary (COCO-80) YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_WEIGHTS)
    return _yolo_model


def get_yolo_world_model(classes):
    """
    Loads (and caches) the open-vocabulary YOLO-World model, configured for
    the given list of text class labels. Re-applies set_classes() whenever
    the requested vocabulary changes (cheap - it's just re-embedding the
    text prompts, not reloading the whole model).

    Note: on first use this downloads a CLIP text encoder (~350MB) in
    addition to the YOLO-World weights - needs internet access once.
    """
    global _yolo_world_model, _yolo_world_classes
    if _yolo_world_model is None:
        _yolo_world_model = YOLOWorld(YOLO_WORLD_WEIGHTS)
    if classes != _yolo_world_classes:
        _yolo_world_model.set_classes(classes)
        _yolo_world_classes = classes
    return _yolo_world_model


def _extract_detections(model, result):
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "class_name": model.names[cls_id],
            "confidence": round(conf, 4),
            "box": [round(v, 1) for v in xyxy],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def detect_objects(pil_image, conf_threshold=0.25):
    """
    Fixed-vocabulary detection (COCO-80). Returns a list of detections, each:
        {"class_name": str, "confidence": float, "box": [x1, y1, x2, y2]}
    """
    model = get_yolo_model()
    results = model.predict(pil_image, conf=conf_threshold, verbose=False)
    return _extract_detections(model, results[0])


def detect_and_annotate(pil_image, conf_threshold=0.25):
    """
    Fixed-vocabulary (COCO-80) detect + draw. Returns (detections, annotated
    RGB uint8 image).
    """
    model = get_yolo_model()
    results = model.predict(pil_image, conf=conf_threshold, verbose=False)
    result = results[0]
    annotated_rgb = result.plot()[..., ::-1]  # BGR -> RGB
    return _extract_detections(model, result), annotated_rgb.astype(np.uint8)


def detect_open_vocab(pil_image, classes=None, conf_threshold=0.1):
    """
    Open-vocabulary detection (YOLO-World): detects whatever text labels you
    give it, not limited to a fixed 80-class list.

    classes: list of text labels to look for (e.g. ["coffee mug", "solar
        panel", "bicycle helmet"]). Defaults to DEFAULT_OPEN_VOCAB (~150
        common objects) if not given. Can be as long as you like, though
        very large lists (thousands) will slow inference and reduce
        precision - shorter, targeted lists give the best results.
    conf_threshold: open-vocabulary matching tends to need a lower threshold
        than the fixed-vocab model since it's zero-shot; 0.1 is a reasonable
        starting point, raise it if you get too many false positives.

    Returns (detections, annotated RGB uint8 image), same shape as
    detect_and_annotate().
    """
    classes = classes or DEFAULT_OPEN_VOCAB
    model = get_yolo_world_model(classes)
    results = model.predict(pil_image, conf=conf_threshold, verbose=False)
    result = results[0]
    annotated_rgb = result.plot()[..., ::-1]
    return _extract_detections(model, result), annotated_rgb.astype(np.uint8)


def list_supported_classes():
    """Returns the 80 fixed COCO class names the standard YOLO model detects."""
    model = get_yolo_model()
    return list(model.names.values())


if __name__ == "__main__":
    # Quick smoke test using ultralytics' own bundled sample image.
    import ultralytics
    sample = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")

    print("=== Fixed-vocabulary (COCO-80) ===")
    dets, annotated = detect_and_annotate(sample)
    for d in dets:
        print(f"  {d['class_name']:15s} conf={d['confidence']:.2f}  box={d['box']}")

    print("\n=== Open-vocabulary (YOLO-World) - requires internet for CLIP weights ===")
    try:
        dets, annotated = detect_open_vocab(sample, classes=["person", "bus", "stop sign", "wheel", "window"])
        for d in dets:
            print(f"  {d['class_name']:15s} conf={d['confidence']:.2f}  box={d['box']}")
    except Exception as e:
        print(f"  Skipped (expected in network-restricted sandboxes): {e}")

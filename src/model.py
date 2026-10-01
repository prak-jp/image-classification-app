"""
model.py
--------
Step 3 (Model Development) and Step 4 (Model Optimization / Transfer Learning)
of the project brief.

Provides two selectable architectures:
  - "custom_cnn": a CNN built from scratch (Conv/BatchNorm/Dropout blocks).
  - "mobilenet" : transfer learning on top of MobileNetV2 pretrained on
                   ImageNet, with a fine-tuning option.

Both expose the last conv layer under a fixed name so Grad-CAM can find it
without extra config.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, GRADCAM_LAYER


def build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """A CNN built from scratch: conv blocks + batchnorm + dropout + dense head."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Named so Grad-CAM can always find the last convolutional feature map.
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    return model


def build_mobilenet_transfer(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                              fine_tune_at=None):
    """
    Transfer learning on MobileNetV2 (pretrained on ImageNet).

    fine_tune_at: if given, unfreezes layers from this index onward for
    fine-tuning (Step 4 - Transfer Learning). If None, the whole backbone
    stays frozen (feature-extraction only), which is faster to train.
    """
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    except Exception as e:
        # Falls back to random init if ImageNet weights can't be downloaded
        # (e.g. no internet access). Training still works, just without the
        # transfer-learning head start - useful for offline smoke tests.
        print(f"[model] Could not download ImageNet weights ({e}). "
              f"Building MobileNetV2 with random initialization instead.")
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights=None
        )
        fine_tune_at = 0  # nothing pretrained to freeze
    base._name = "mobilenet_backbone"

    if fine_tune_at is None:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False if fine_tune_at is None else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenet_transfer")
    return model


def build_model(model_type="mobilenet", fine_tune_at=None):
    """Factory used by train.py / evaluate.py / gradcam.py."""
    if model_type == "custom_cnn":
        model = build_custom_cnn()
    elif model_type == "mobilenet":
        model = build_mobilenet_transfer(fine_tune_at=fine_tune_at)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_gradcam_layer_name(model_type):
    """Returns the layer name Grad-CAM should hook into for the given architecture."""
    return GRADCAM_LAYER[model_type]


if __name__ == "__main__":
    for mtype in ("custom_cnn", "mobilenet"):
        m = build_model(mtype)
        print(f"\n=== {mtype} ===")
        m.summary()

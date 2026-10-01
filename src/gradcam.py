"""
gradcam.py
----------
Bonus feature: Model Explainability via Grad-CAM.

Produces a heatmap over the input image showing which regions most
influenced the model's predicted class. Works transparently with both the
custom CNN and the MobileNetV2 transfer-learning model, including looking
inside nested sub-models (MobileNetV2 is wrapped as a single layer in our
functional model, so we search inside it for the target conv layer too).
"""

import numpy as np
import cv2
import tensorflow as tf

from model import get_gradcam_layer_name


def _find_layer(model, layer_name):
    """
    Finds a layer by name at the top level of the model. For MobileNetV2 the
    target is the backbone wrapper layer itself (its own output already IS
    the last conv feature map we need) rather than an internal layer, since
    Keras 3 can't rebuild a graph reaching inside a nested sub-model.

    Keras auto-names the MobileNetV2 sub-model (e.g. "mobilenetv2_1.00_96")
    rather than honoring a manual rename in every TF version, so if the
    configured name isn't found we fall back to locating the single nested
    Functional/Model layer automatically.
    """
    try:
        return model.get_layer(layer_name)
    except ValueError:
        nested = [l for l in model.layers if isinstance(l, tf.keras.Model)]
        if len(nested) == 1:
            return nested[0]
        raise ValueError(
            f"Layer '{layer_name}' not found, and could not unambiguously "
            f"locate a nested backbone model as a fallback."
        )


def make_gradcam_heatmap(img_array, model, model_type, pred_index=None):
    """
    img_array: preprocessed batch of shape (1, H, W, 3), values in [0, 1].
    Returns: heatmap (H', W') normalized to [0, 1], and the predicted class index.

    Implementation note: rather than rebuilding a new Functional Model from
    intermediate tensors (which Keras 3 can refuse to reconnect for a model
    that's been saved/loaded, especially across a nested sub-model boundary
    like MobileNetV2), we replay the model's own layers one by one in eager
    mode inside a GradientTape. This works uniformly for any linear-stack
    architecture (both our custom CNN and the MobileNetV2 wrapper) and
    sidesteps graph-reconnection issues entirely.
    """
    layer_name = get_gradcam_layer_name(model_type)
    target_layer = _find_layer(model, layer_name)
    x = tf.convert_to_tensor(img_array, dtype=tf.float32)

    conv_outputs = None
    with tf.GradientTape() as tape:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if layer is target_layer:
                tape.watch(x)
                conv_outputs = x
        predictions = x
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(original_img_uint8, heatmap, alpha=0.4):
    """
    original_img_uint8: HxWx3 uint8 RGB image (the un-normalized display image).
    heatmap: 2D array in [0, 1] from make_gradcam_heatmap.
    Returns: HxWx3 uint8 RGB image with the heatmap overlaid.
    """
    h, w = original_img_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlaid = np.uint8(colored * alpha + original_img_uint8 * (1 - alpha))
    return overlaid


def explain_prediction(pil_image, model, model_type, img_size):
    """
    Convenience wrapper for the apps: takes a PIL image, returns
    (predicted_class_index, probabilities, gradcam_overlay_uint8_image).
    """
    from PIL import Image

    resized = pil_image.convert("RGB").resize((img_size, img_size))
    display_img = np.array(resized)
    img_array = display_img.astype(np.float32) / 255.0
    batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(batch, verbose=0)[0]
    pred_index = int(np.argmax(preds))

    heatmap, _ = make_gradcam_heatmap(batch, model, model_type, pred_index=pred_index)
    overlay = overlay_heatmap(display_img, heatmap)

    return pred_index, preds, overlay

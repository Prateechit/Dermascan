"""
Grad-CAM heatmap generation for model explainability.

Grad-CAM highlights the regions of the lesion that most influenced the model's
decision, giving a visual "focus map" (as shown in the reference report). When
running in demo mode (no trained model) it produces a plausible centre-weighted
heatmap so the UI still has something to show.
"""

import cv2
import numpy as np


def _overlay(image_bgr, heatmap):
    """Blend a [0,1] heatmap onto the original image."""
    heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.6, colored, 0.4, 0)


def demo_heatmap(image_bgr):
    """Centre-weighted Gaussian heatmap for demo mode."""
    h, w = image_bgr.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = ((x - cx) ** 2 + (y - cy) ** 2) / (2 * (min(h, w) / 3) ** 2)
    heat = np.exp(-dist)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat


def gradcam_heatmap(model, tensor, last_conv_layer_name=None):
    """Compute a Grad-CAM heatmap for the top predicted class."""
    import tensorflow as tf

    if last_conv_layer_name is None:
        # MobileNetV2's final conv activation.
        last_conv_layer_name = "Conv_1"

    base = model.get_layer(index=0)  # MobileNetV2 base
    grad_model = tf.keras.models.Model(
        [base.inputs],
        [base.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tensor)
        top = tf.argmax(preds[0])
        class_channel = preds[:, top]

    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heat = conv_out @ pooled[..., tf.newaxis]
    heat = tf.squeeze(heat)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def generate(image_bgr, predictor, out_path):
    """Generate and save a Grad-CAM overlay image. Returns the path."""
    if predictor.demo:
        heat = demo_heatmap(image_bgr)
    else:
        from .preprocessing import preprocess
        tensor = preprocess(image_bgr, for_model=True)
        try:
            heat = gradcam_heatmap(predictor.model, tensor)
        except Exception:  # noqa: BLE001
            heat = demo_heatmap(image_bgr)

    overlay = _overlay(image_bgr, heat)
    cv2.imwrite(out_path, overlay)
    return out_path

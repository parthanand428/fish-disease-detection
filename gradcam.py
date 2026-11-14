from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


def _find_target_conv_layer(model: tf.keras.Model) -> Tuple[Optional[tf.keras.Model], str]:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            if hasattr(layer, "layers"):
                for sublayer in reversed(layer.layers):
                    try:
                        out_shape = sublayer.output.shape
                        if len(out_shape) == 4:
                            return layer, sublayer.name
                    except Exception:
                        continue

    for layer in reversed(model.layers):
        try:
            out_shape = layer.output.shape
            if len(out_shape) == 4 and "rescaling" not in layer.name.lower():
                return None, layer.name
        except Exception:
            continue

    raise ValueError("Could not locate a 4D convolutional layer for Grad-CAM.")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    class_index: Optional[int] = None,
) -> np.ndarray:
    inputs = tf.cast(img_tensor, tf.float32)
    nested_base, target_layer_name = _find_target_conv_layer(model)

    if nested_base is not None:
        pre_layers = []
        post_layers = []
        found_base = False
        for layer in model.layers:
            if layer.name == nested_base.name:
                found_base = True
                continue
            if not found_base:
                pre_layers.append(layer)
            else:
                post_layers.append(layer)

        target_conv_output = nested_base.get_layer(target_layer_name).output
        base_grad_model = tf.keras.Model(
            inputs=nested_base.inputs,
            outputs=[target_conv_output, nested_base.output],
        )

        with tf.GradientTape() as tape:
            x = inputs
            for pre_layer in pre_layers:
                x = pre_layer(x, training=False)

            conv_outputs, base_out = base_grad_model(x, training=False)
            tape.watch(conv_outputs)

            y = base_out
            for post_layer in post_layers:
                y = post_layer(y, training=False)

            preds = y
            if class_index is None:
                class_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, class_index]

    else:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(target_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(inputs, training=False)
            tape.watch(conv_outputs)
            if class_index is None:
                class_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        # Fallback to mean feature map activation if gradients vanish
        heatmap = tf.reduce_mean(tf.maximum(conv_outputs, 0), axis=-1)
        m2 = tf.math.reduce_max(heatmap)
        if m2 > 0:
            heatmap = heatmap / m2

    return heatmap.numpy()


def generate_gradcam_overlay(
    model: tf.keras.Model,
    original_pil_image: Image.Image,
    img_tensor: np.ndarray,
    class_index: Optional[int] = None,
    alpha: float = 0.42,
) -> Tuple[Image.Image, Image.Image]:
    heatmap_2d = compute_gradcam_heatmap(model, img_tensor, class_index=class_index)

    orig_rgb = np.array(original_pil_image.convert("RGB"))
    height, width = orig_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap_2d, (width, height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(orig_rgb, 1.0 - alpha, jet_rgb, alpha, 0)

    overlay_pil = Image.fromarray(superimposed)
    heatmap_pil = Image.fromarray(jet_rgb)
    return overlay_pil, heatmap_pil

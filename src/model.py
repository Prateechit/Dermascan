"""
Model definition and loading.

Architecture: transfer learning on MobileNetV2 (ImageNet weights) with a small
classification head for the 7 HAM10000 classes. MobileNetV2 is lightweight
enough to run on a laptop or a free Colab GPU while still giving good accuracy.

The rest of the application imports `load_model()` and `CLASS_NAMES` from here.
If TensorFlow is not installed or no trained weights are present, `load_model()`
returns None and the app falls back to a deterministic demo predictor (see
predict.py) so the project is always runnable.
"""

import os

# HAM10000 label order. Keep this consistent with train/train.py.
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (224, 224, 3)

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "skin_model.h5",
)


def build_model():
    """Build the (untrained) transfer-learning model. Used by train.py."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze for the first training phase

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model


def load_model():
    """Load trained weights if available; otherwise return None (demo mode)."""
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[model] Could not load trained model ({exc}). Using demo mode.")
        return None


def model_path():
    return _MODEL_PATH

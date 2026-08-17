"""
Prediction / inference.

Loads, in order of preference:
  1. A lightweight TensorFlow-Lite model (models/skin_model.tflite or
     skin_model.tflite at the repo root) -> runs in very little RAM, so it fits
     free hosting tiers like Render.
  2. A full Keras model (models/skin_model.h5) -> used when running locally with
     TensorFlow installed.
  3. DEMO mode -> deterministic placeholder, so the app always runs even before
     a model is trained.

The active backend is exposed as `self.backend` ("tflite" | "keras" | "demo").
"""

import hashlib
import json
import os

import numpy as np

from .model import CLASS_NAMES
from .preprocessing import preprocess

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PATH = os.path.join(_BASE, "data", "disease_info.json")

_TFLITE_CANDIDATES = [
    os.path.join(_BASE, "models", "skin_model.tflite"),
    os.path.join(_BASE, "skin_model.tflite"),
]
_H5_CANDIDATES = [
    os.path.join(_BASE, "models", "skin_model.h5"),
    os.path.join(_BASE, "skin_model.h5"),
]

with open(_DATA_PATH, "r", encoding="utf-8") as fh:
    DISEASE_INFO = json.load(fh)

CONFIDENCE_THRESHOLD = 0.40


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _load_tflite(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception:
            from tensorflow.lite import Interpreter
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


class Predictor:
    def __init__(self):
        self.backend = "demo"
        self.interp = None
        self.model = None

        tflite_path = _first_existing(_TFLITE_CANDIDATES)
        h5_path = _first_existing(_H5_CANDIDATES)

        if tflite_path:
            try:
                self.interp = _load_tflite(tflite_path)
                self._in = self.interp.get_input_details()[0]
                self._out = self.interp.get_output_details()[0]
                self.backend = "tflite"
                print(f"[predict] Using TFLite model: {tflite_path}")
            except Exception as exc:
                print(f"[predict] TFLite load failed ({exc}); trying next.")

        if self.backend == "demo" and h5_path:
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(h5_path)
                self.backend = "keras"
                print(f"[predict] Using Keras model: {h5_path}")
            except Exception as exc:
                print(f"[predict] Keras load failed ({exc}); using demo mode.")

        self.demo = self.backend == "demo"

    def _demo_probs(self, image_bgr):
        digest = hashlib.sha256(image_bgr.tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        logits = rng.normal(size=len(CLASS_NAMES))
        logits[np.argmax(logits)] += 2.0
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def _probs(self, image_bgr):
        if self.backend == "tflite":
            tensor = preprocess(image_bgr, for_model=True).astype(np.float32)
            self.interp.set_tensor(self._in["index"], tensor)
            self.interp.invoke()
            return self.interp.get_tensor(self._out["index"])[0]
        if self.backend == "keras":
            tensor = preprocess(image_bgr, for_model=True)
            return self.model.predict(tensor, verbose=0)[0]
        return self._demo_probs(image_bgr)

    def predict(self, image_bgr):
        probs = np.asarray(self._probs(image_bgr), dtype=float)
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        top_key = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])

        info_key = top_key if top_conf >= CONFIDENCE_THRESHOLD else "unknown"
        info = DISEASE_INFO[info_key]

        top3 = [
            {
                "key": CLASS_NAMES[i],
                "name": DISEASE_INFO[CLASS_NAMES[i]]["short"],
                "confidence": round(float(probs[i]) * 100, 1),
            }
            for i in order[:3]
        ]

        return {
            "demo_mode": self.demo,
            "backend": self.backend,
            "disease_key": info_key,
            "disease_name": info["name"],
            "confidence": round(top_conf * 100, 1),
            "severity": info["severity"],
            "malignant": info["malignant"],
            "description": info["description"],
            "precautions": info["precautions"],
            "top3": top3,
        }

"""
Prediction / inference.

Exposes a single Predictor class. On construction it tries to load the trained
model. If none is available it runs in DEMO mode: a deterministic pseudo-model
that derives a stable prediction from the image content. Demo mode lets the
whole application (upload -> preprocess -> predict -> chatbot -> map) be
demonstrated end-to-end before you train on HAM10000, and it is clearly flagged
in the response so it is never mistaken for a real diagnosis.
"""

import hashlib
import json
import os

import numpy as np

from .model import CLASS_NAMES, load_model
from .preprocessing import preprocess

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "disease_info.json",
)

with open(_DATA_PATH, "r", encoding="utf-8") as fh:
    DISEASE_INFO = json.load(fh)

# Below this top-probability the model is treated as "uncertain".
CONFIDENCE_THRESHOLD = 0.40


class Predictor:
    def __init__(self):
        self.model = load_model()
        self.demo = self.model is None

    def _demo_probs(self, image_bgr):
        """Deterministic pseudo-probabilities from the image bytes.

        Same image -> same result, so demos are reproducible. This is NOT a
        real classifier; it only exists so the UI works before training.
        """
        digest = hashlib.sha256(image_bgr.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=len(CLASS_NAMES))
        # Bias slightly toward a confident top class for a nicer demo.
        logits[np.argmax(logits)] += 2.0
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def predict(self, image_bgr):
        """Return a structured prediction dict for a raw BGR image."""
        if self.demo:
            probs = self._demo_probs(image_bgr)
        else:
            tensor = preprocess(image_bgr, for_model=True)
            probs = self.model.predict(tensor, verbose=0)[0]

        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        top_key = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])

        if top_conf < CONFIDENCE_THRESHOLD:
            info_key = "unknown"
        else:
            info_key = top_key

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
            "disease_key": info_key,
            "disease_name": info["name"],
            "confidence": round(top_conf * 100, 1),
            "severity": info["severity"],
            "malignant": info["malignant"],
            "description": info["description"],
            "precautions": info["precautions"],
            "top3": top3,
        }

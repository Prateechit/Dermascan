"""
Flask web application for AI-Based Skin Disease Detection.

Routes
------
GET  /                 -> single-page UI
POST /api/analyze      -> upload image, preprocess, predict, Grad-CAM
POST /api/chat         -> combine prediction + questionnaire -> guidance
GET  /api/questions    -> chatbot screening questions

Run:  python app.py   then open http://localhost:8080
"""

import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from src.chatbot import QUESTIONS, generate_ai_reply, generate_reply
from src.gradcam import generate as generate_gradcam
from src.predict import Predictor
from src.preprocessing import save_processed_preview
from src.recommendations import recommend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Load the predictor once at startup (loads model or enters demo mode).
predictor = Predictor()

# Simple in-memory store of the last prediction per session id.
_last_prediction = {}


def _read_image(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html", demo_mode=predictor.demo)


@app.route("/api/questions")
def questions():
    return jsonify(QUESTIONS)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image_bgr = _read_image(request.files["image"])
    if image_bgr is None:
        return jsonify({"error": "Could not read the image."}), 400

    session_id = request.form.get("session_id") or str(uuid.uuid4())
    stamp = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # Save original, processed preview, and Grad-CAM overlay.
    orig_name = f"orig_{stamp}.jpg"
    proc_name = f"proc_{stamp}.jpg"
    cam_name = f"cam_{stamp}.jpg"

    cv2.imwrite(os.path.join(UPLOAD_DIR, orig_name), image_bgr)
    save_processed_preview(image_bgr, os.path.join(UPLOAD_DIR, proc_name))
    generate_gradcam(image_bgr, predictor, os.path.join(UPLOAD_DIR, cam_name))

    prediction = predictor.predict(image_bgr)
    _last_prediction[session_id] = prediction

    lat = request.form.get("lat", type=float)
    lng = request.form.get("lng", type=float)
    rec = recommend(prediction["severity"], lat, lng)

    return jsonify({
        "session_id": session_id,
        "prediction": prediction,
        "recommendation": rec,
        "images": {
            "original": f"/static/uploads/{orig_name}",
            "processed": f"/static/uploads/{proc_name}",
            "gradcam": f"/static/uploads/{cam_name}",
        },
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    session_id = payload.get("session_id")
    answers = payload.get("answers", {})
    user_question = payload.get("question", "")

    prediction = _last_prediction.get(session_id)
    if prediction is None:
        return jsonify({"error": "Analyze an image first."}), 400

    reply = generate_ai_reply(prediction, answers, user_question)
    structured = generate_reply(prediction, answers)
    return jsonify({"reply": reply, "urgency": structured["urgency"]})


@app.route("/static/uploads/<path:filename>")
def uploaded(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    mode = "DEMO (no trained model found)" if predictor.demo else "TRAINED model"
    print(f" * Skin Disease Detection running in {mode} mode")
    app.run(host="0.0.0.0", port=8080, debug=True)

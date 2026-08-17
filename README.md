# DermaScan — AI-Based Skin Disease Detection & Dermatology Assistant

**Live demo:** https://dermascan-194r.onrender.com
> ⏳ Hosted on a free tier — the first visit after it's been idle takes ~50 seconds to wake up, then it's fast.

> ⚠️ Educational screening tool only — **not** a medical diagnosis.

DermaScan screens a skin photo with a deep-learning model trained on the HAM10000
dataset, shows the predicted condition with a confidence score, asks a short set
of health questions through an AI chatbot, gives precautions, and points the user
to a nearby dermatologist.

---

## Features

| # | Requirement | Where it's implemented |
|---|-------------|------------------------|
| 1 | Upload or capture a skin image | `templates/index.html`, `static/js/app.js` |
| 2 | OpenCV preprocessing (hair removal, colour constancy, denoise) | `src/preprocessing.py` |
| 3 | Prediction with a trained deep-learning model | `src/model.py`, `src/predict.py` |
| 4 | Disease name + confidence score | shown in the UI |
| 5 | AI chatbot asks 5 health questions | `src/chatbot.py` |
| 6 | Precautions based on the prediction | `data/disease_info.json` |
| 7 | Nearest dermatologist (Google Maps + clinic list) | `src/recommendations.py` |

## How it works

```
Skin image → OpenCV preprocessing → CNN prediction → confidence + focus map
           → AI chatbot (symptoms) → guidance & precautions → dermatologist referral
```

- **Model:** MobileNetV2 transfer learning, trained on HAM10000 (7 lesion classes).
- **Serving:** the trained model is converted to TensorFlow-Lite (`skin_model.tflite`)
  so it runs in a few MB of RAM and fits a free hosting tier.
- **Explainability:** a focus map highlights the region the model attended to.

## Tech stack

Python · Flask · OpenCV · TensorFlow / Keras (training) · TensorFlow-Lite (serving) · HTML/CSS/JS

## Dataset

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) —
10,000+ dermatoscopic images across 7 classes: akiec, bcc, bkl, df, mel, nv, vasc.

## Run locally

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:8080
```
Without a model file it runs in a demo mode so the UI is always explorable; with
`skin_model.tflite` (or `models/skin_model.h5`) present it uses the trained model.

## Train the model (Google Colab)

1. Turn on a GPU runtime.
2. Download HAM10000 from Kaggle.
3. Prepare and train:
   ```bash
   python train/prepare_data.py --metadata HAM10000_metadata.csv \
       --images HAM10000_images_part_1 HAM10000_images_part_2 --out data/ham10000
   python train/train.py --data data/ham10000 --epochs 15
   ```
4. Convert to TensorFlow-Lite and place `skin_model.tflite` in the repo.

## Deployment (Render, free)

- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
- Environment variable: `PYTHON_VERSION = 3.11.9`

## Disease classes

akiec (actinic keratosis) · bcc (basal cell carcinoma) · bkl (benign keratosis) ·
df (dermatofibroma) · mel (melanoma) · nv (melanocytic nevus) · vasc (vascular lesion)

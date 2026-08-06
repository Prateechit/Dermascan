# DermaScan — AI-Based Skin Disease Detection & Dermatology Assistant

A mini-project that screens a skin photo with a deep-learning model, explains
its decision with a Grad-CAM focus map, asks a few health questions through an
AI chatbot, shows precautions, and points the user to a nearby dermatologist.

> ⚠️ Educational screening tool only — **not** a medical diagnosis.

## Features

| Requirement | Where it's implemented |
|---|---|
| Upload or capture a skin image | `templates/index.html`, `static/js/app.js` |
| OpenCV preprocessing | `src/preprocessing.py` (hair removal, colour constancy, denoise) |
| Deep-learning prediction | `src/model.py` (MobileNetV2 transfer learning), `src/predict.py` |
| Disease name + confidence | `src/predict.py` → shown in the UI |
| Grad-CAM explainability | `src/gradcam.py` |
| AI chatbot (4–5 questions) | `src/chatbot.py` |
| Precautions | `data/disease_info.json` |
| Nearest dermatologist | `src/recommendations.py` (Google Maps link + fallback list) |

## Quick start (runs immediately in demo mode)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install Flask opencv-python-headless numpy Pillow
python app.py
```

Open <http://localhost:8080>. With no trained model present, the app runs in
**demo mode** — the full pipeline works (upload → preprocess → predict →
chatbot → map), but predictions are illustrative placeholders.

## Train the real model (HAM10000)

1. Install everything: `pip install -r requirements.txt`
2. Download HAM10000 from Kaggle:
   <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>
3. Sort it into class folders:
   ```bash
   python train/prepare_data.py \
       --metadata HAM10000_metadata.csv \
       --images HAM10000_images_part_1 HAM10000_images_part_2 \
       --out data/ham10000
   ```
4. Train (GPU / Colab recommended):
   ```bash
   python train/train.py --data data/ham10000 --epochs 15
   ```
   This saves `models/skin_model.h5`.
5. Restart `python app.py`. It now uses the trained model automatically.

## Optional: generative chatbot with Claude

Set an API key and the chatbot uses Claude for natural, generative guidance
(otherwise it uses the built-in rule-based engine):

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Project structure

```
skin-disease-detection/
├── app.py                 # Flask app (routes)
├── requirements.txt
├── src/
│   ├── preprocessing.py   # OpenCV pipeline
│   ├── model.py           # MobileNetV2 architecture + loader
│   ├── predict.py         # inference + confidence + demo fallback
│   ├── gradcam.py         # explainability heatmap
│   ├── chatbot.py         # questionnaire + guidance
│   └── recommendations.py # dermatologist / Google Maps
├── train/
│   ├── prepare_data.py    # sort HAM10000 into folders
│   └── train.py           # transfer-learning training
├── templates/index.html   # UI
├── static/css/style.css
├── static/js/app.js
├── data/disease_info.json # class metadata + precautions
└── models/                # trained model goes here
```

## Disease classes (HAM10000)

akiec · bcc · bkl · df · mel · nv · vasc
(actinic keratosis, basal cell carcinoma, benign keratosis, dermatofibroma,
melanoma, melanocytic nevus, vascular lesion)

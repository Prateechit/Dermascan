# Deploying DermaScan to a permanent public link (Render.com — free)

This gives you a fixed URL like https://dermascan.onrender.com that anyone can
open, any time, without you running anything.

## One-time prerequisite: put the code on GitHub
1. Create a free account at https://github.com and click "New repository".
2. Name it e.g. `dermascan`, keep it Public, click "Create repository".
3. On the repo page click "Add file" -> "Upload files", then drag in ALL the
   files and folders from this project (app.py, src/, templates/, static/,
   data/, requirements.txt, Procfile, runtime.txt, etc.). Commit.

## Deploy on Render
1. Go to https://render.com and sign up (you can log in with GitHub — free, no card).
2. Click "New +" -> "Web Service".
3. Connect your GitHub and pick the `dermascan` repo.
4. Render auto-detects the settings from the Procfile. Confirm:
   - Runtime: Python 3
   - Build command:  pip install -r requirements.txt
   - Start command:  gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
   - Instance type:  Free
5. Click "Create Web Service" and wait ~2–5 minutes for the first build.
6. When it says "Live", your link is at the top, e.g. https://dermascan.onrender.com
   Share that link — anyone can open it.

## Notes
- Free instances sleep after ~15 min of no traffic. The first visit after sleep
  takes ~30–50 seconds to wake up, then it's fast. Fine for a demo/submission.
- This deploys in DEMO mode (light and reliable). To serve your REAL trained
  model, add `tensorflow-cpu==2.16.1` to requirements.txt and commit your
  models/skin_model.h5 — but TensorFlow needs more than the free 512 MB of RAM,
  so use a paid Render instance OR Hugging Face Spaces (16 GB RAM free), which
  is the better home for the full model.

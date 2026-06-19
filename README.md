---
title: Traffic Violation Detection
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🚦 Traffic Violation Detection Dashboard

A single Gradio dashboard to run and compare several YOLO (v10 / v11) traffic
violation detection models. Pick a model, upload an image, tune the confidence
threshold, and get an annotated image plus a per-class detection summary.

## Models

| # | Model | Framework | Weights | Classes |
|---|-------|-----------|---------|---------|
| 1 | License Plate Detection | YOLOv11n | `models/license.onnx` | License Plate |
| 2 | Helmet Violation Detection | YOLOv11m | `models/helmet.pt` *(add later)* | Plate, WithHelmet, WithoutHelmet |
| 3 | Driver Monitoring System (DMS) | YOLOv8n | `models/driver.onnx` | Open Eye, Closed Eye, Cigarette, Phone, Seatbelt |
| 4 | Illegal Parking Detection | YOLOv11m | `models/illegal_parking.pt` *(add later)* | Empty, Occupied |

All weights — `.pt` or `.onnx` — are loaded via Ultralytics `YOLO(...)`, so a
single code path handles both formats. Class names are read from a per-model
`data.yaml` in `configs/`, falling back to the names embedded in the model.

The dashboard **loads every available model at startup** (so switching is
instant) and **gracefully handles missing weights** — the Illegal Parking model
appears in the dropdown marked `(model not loaded)` until you drop its weights
file in `models/`.

## Folder structure

```
traffic-violation/
├── app.py                  # Gradio dashboard (entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file (also the HF Space config)
├── .gitattributes          # Git LFS tracking for *.pt / *.onnx
├── .gitignore
├── models/                 # Model weights
│   ├── license.onnx
│   ├── driver.onnx
│   ├── helmet.pt                    # <- add this later (see helmet_model_note.txt)
│   ├── helmet_model_note.txt
│   ├── illegal_parking.pt           # <- add this later (see link file)
│   └── illegal_parking_model_link.txt
├── configs/                # Class names per model (data.yaml style)
│   ├── license.yaml
│   ├── helmet.yaml
│   ├── driver.yaml
│   └── illegal_parking.yaml
└── notebooks/              # Original training notebooks (reference)
    ├── license.ipynb
    ├── helmetviolation.ipynb
    └── illegal-parking-detection.ipynb
```

### Adding the Helmet model later

The original `helmet.pt` was committed as a broken Git LFS pointer (the 117 MB
object was never uploaded), so it has been removed to keep clones/pulls working.
Copy your trained `helmet.pt` into `models/` and commit it with working LFS —
see `models/helmet_model_note.txt` for the exact commands.

### Adding the Illegal Parking model later

1. Download the weights from the Google Drive link in
   `models/illegal_parking_model_link.txt`.
2. Save the file as `models/illegal_parking.pt`.
3. Restart the app — it will be picked up automatically.

To add **any** new model: drop the weights in `models/`, add a matching
`configs/<id>.yaml` with its class `names`, and append one entry to the
`MODELS` list at the top of `app.py`.

## Run locally

```bash
# 1. Clone (with Git LFS so the .pt / .onnx weights download)
git lfs install
git clone <your-repo-url>
cd traffic-violation
git lfs pull            # ensure weights are real files, not LFS pointers

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
python app.py
```

Open the printed URL (default http://localhost:7860). The app forces dark mode.

> Tip: if a model shows `(model not loaded)`, its weights file is missing or is
> an un-pulled Git LFS pointer. Run `git lfs pull` and restart.

## Deploy to Hugging Face Spaces (with Git LFS)

`.pt` files are large, so the weights must go through Git LFS.

```bash
# 0. (Once) install the HF CLI and log in
pip install huggingface_hub
huggingface-cli login          # paste a write token from hf.co/settings/tokens

# 1. Create a Gradio Space (UI: hf.co/new-space, SDK = Gradio)
#    e.g. https://huggingface.co/spaces/<username>/traffic-violation

# 2. Add the Space as a git remote
git remote add space https://huggingface.co/spaces/<username>/traffic-violation

# 3. Make sure Git LFS is initialised and tracks the weights
git lfs install
git lfs track "*.pt" "*.onnx"
git add .gitattributes

# 4. Stage everything (LFS picks up the weights via .gitattributes)
git add .
git commit -m "Deploy traffic violation dashboard"

# 5. Push to the Space (LFS objects upload automatically)
git push space <your-branch>:main
```

The Space rebuilds automatically using:
- `README.md` front-matter (`sdk: gradio`, `app_file: app.py`)
- `requirements.txt` for dependencies

### Notes for Spaces
- Confirm `.pt` / `.onnx` files are LFS-tracked before pushing:
  `git lfs ls-files` should list them.
- If you already committed weights as normal git blobs, migrate them with
  `git lfs migrate import --include="*.pt,*.onnx"` then force-push.
- Free CPU Spaces are fine for these models; pick a GPU Space for faster
  inference if needed.
- `opencv-python-headless` (already in `requirements.txt`) avoids missing
  system-library errors on Spaces.

---
title: Traffic Violation Detection
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
python_version: "3.11"
pinned: false
---

# 🚦 Traffic Violation Detection Dashboard

A Gradio dashboard that runs several traffic-violation models on **images and
video**, either all together or one at a time. Every run produces an annotated
result, a per-class summary, and a **downloadable CSV** of all detections.

## Highlights

- **Two modes, no dropdowns** — a *Run All Models* tab (one upload → combined
  overlay) and an *Individual Models* tab (each model has its own description,
  upload and results in a collapsible section).
- **Image *and* video** — upload either; video is sampled frame-by-frame, run
  through the models, and re-encoded as an annotated MP4 (capped at
  `MAX_VIDEO_FRAMES = 200` so CPU Spaces stay responsive).
- **Downloadable CSV** on every run — one row per detection with model, class,
  confidence and bounding box.
- **Detectors + a classifier** — four YOLO detectors draw boxes; the vehicle
  CNN labels the whole frame with a banner.

## Models

| # | Model | Framework | Weights | Output |
|---|-------|-----------|---------|--------|
| 1 | Helmet Violation Detection | YOLOv11m | `models/helmet.pt` | Plate, WithHelmet, WithoutHelmet |
| 2 | Driver Monitoring System (DMS) | YOLOv8n | `models/driver.onnx` | Open/Closed Eye, Cigarette, Phone, Seatbelt |
| 3 | Illegal Parking Detection | YOLOv11m (COCO) | `models/illegalpark.pt` | car, motorcycle, bus, truck, bicycle |
| 4 | Traffic Signal & Sign Violations | YOLOv8m | `models/stopwait.pt` | red-light, stop-line, wrong-side, signs |
| 5 | Vehicle Type Classification (CNN) | Keras (224×224) | `models/complete_model_model.h5` | 15 vehicle types (whole-image) |

The four YOLO weights (`.pt` / `.onnx`) load through Ultralytics `YOLO(...)`.
The vehicle classifier is a Keras CNN loaded with TensorFlow; instead of boxes
it predicts one of **15 classes** — Ambulance, Bicycle, Boat, Bus, Car,
Helicopter, Limousine, Motorcycle, PickUp, Segway, Snowmobile, Tank, Taxi,
Truck, Van — and draws a label banner on the frame.

> **Note on Illegal Parking:** `illegalpark.pt` is a COCO-pretrained detector,
> so its raw output is filtered down to vehicle classes in `app.py`
> (`"keep"` set) for a parking-enforcement use-case.

The dashboard **loads every available model at startup** and **gracefully
handles missing weights** — a model with no real weights file simply shows up
marked `(not loaded)` instead of crashing the app.

## Folder structure

```
traffic-violation/
├── app.py                       # Gradio dashboard (entry point)
├── requirements.txt             # Python dependencies
├── README.md                    # This file (also the HF Space config)
├── .gitattributes               # Git LFS tracking for *.pt / *.onnx / *.h5
├── models/
│   ├── helmet.pt                # YOLOv11m helmet detector (LFS)
│   ├── driver.onnx              # YOLOv8n driver-monitoring (LFS)
│   ├── illegalpark.pt           # COCO YOLOv11m, filtered to vehicles (LFS)
│   ├── stopwait.pt              # YOLOv8m signal/sign detector (LFS)
│   └── complete_model_model.h5  # Keras vehicle classifier (LFS)
├── configs/                     # Class names per detector (data.yaml style)
│   ├── helmet.yaml
│   ├── driver.yaml
│   └── illegal_parking.yaml
└── notebooks/                   # Original training notebooks (reference)
    ├── helmetviolation.ipynb
    ├── illegal-parking-detection.ipynb
    ├── stopwait.ipynb
    └── classification_vehicle/  # CNN training + prediction notebooks
```

### Adding or swapping a model

Drop the weights into `models/`, add an entry to the `MODELS` list at the top
of `app.py` (set `"type": "detect"` for YOLO or `"type": "classify"` for a
Keras CNN), and — for detectors — optionally add a `configs/<id>.yaml` with the
class `names`. The model then appears in both tabs automatically.

## Run locally

```bash
# 1. Clone with Git LFS so the weights download as real files
git lfs install
git clone <your-repo-url>
cd traffic-violation
git lfs pull            # ensure weights are real files, not LFS pointers

# 2. Environment + dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Launch
python app.py
```

Open the printed URL (default http://localhost:7860).

> If a model shows `(not loaded)`, its weights file is missing or is an
> un-pulled Git LFS pointer — run `git lfs pull` and restart.

## Deploy to Hugging Face Spaces

Weights are large, so they must go through Git LFS — **Spaces rejects non-LFS
files over 10 MB**. The four YOLO weights are already LFS-tracked. The 27 MB
vehicle classifier (`complete_model_model.h5`) is committed as a regular git
blob here, so **before pushing to a Space you must migrate it into LFS**:

```bash
# 0. (Once) install the HF CLI and log in with a write token
pip install huggingface_hub
huggingface-cli login

# 1. Create a Gradio Space at hf.co/new-space and add it as a remote
git remote add space https://huggingface.co/spaces/<username>/traffic-violation

# 2. Track *.h5 with LFS and rewrite the .h5 blob into an LFS object
git lfs install
git lfs track "*.pt" "*.onnx" "*.h5"
git lfs migrate import --include="*.h5" --everything
git add .gitattributes

# 3. Push to the Space (LFS objects upload automatically)
git push space <your-branch>:main
```

The Space rebuilds from the `README.md` front-matter (`sdk: gradio`,
`app_file: app.py`, `python_version: "3.11"`) and `requirements.txt`.

### Notes for Spaces
- `python_version: "3.11"` is pinned because the stdlib `audioop` module
  (used transitively by Gradio) was removed in 3.13.
- `tensorflow-cpu==2.15.0` ships Keras 2.x, which loads the legacy `.h5`
  classifier; `numpy` is pinned `<2.0` for TensorFlow compatibility.
- Confirm weights are LFS-tracked before pushing — `git lfs ls-files` should
  list every `.pt` / `.onnx` / `.h5`.
- Free CPU Spaces are fine; pick a GPU Space for faster video processing.

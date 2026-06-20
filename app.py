"""
Traffic Violation Detection — Gradio Dashboard
==============================================

Run several traffic-violation models on an **image or a video**, either all
together or one model at a time. Every run produces an annotated image/video,
a per-class summary and a downloadable CSV of detections.

Models:
    1. Helmet Violation Detection            (YOLOv11m)      -> models/helmet.pt
    2. Driver Monitoring System (DMS)        (YOLOv8n)       -> models/driver.onnx
    3. Illegal Parking Detection             (YOLOv11m/COCO) -> models/illegalpark.pt
    4. Traffic Signal & Sign Violations      (YOLOv8m)       -> models/stopwait.pt
    5. Vehicle Type Classification           (Keras CNN)     -> models/complete_model_model.h5

The YOLO weights (.pt / .onnx) load through Ultralytics' ``YOLO`` loader. The
vehicle classifier is a 224×224 Keras CNN loaded with TensorFlow; instead of
boxes it labels the whole frame with its most likely vehicle type.
"""

from __future__ import annotations

import csv
import os
import tempfile
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import yaml
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CONFIGS_DIR = BASE_DIR / "configs"

DEFAULT_CONF = 0.25
# Cap processed video frames so CPU Spaces don't time out.
MAX_VIDEO_FRAMES = 200

FONT = cv2.FONT_HERSHEY_SIMPLEX

# 15 vehicle classes the CNN was trained on (alphabetical — matches the
# training generator's class order; see notebooks/classification_vehicle).
VEHICLE_LABELS = [
    "Ambulance", "Bicycle", "Boat", "Bus", "Car", "Helicopter", "Limousine",
    "Motorcycle", "PickUp", "Segway", "Snowmobile", "Tank", "Taxi", "Truck",
    "Van",
]

MODELS: list[dict] = [
    {
        "id": "helmet",
        "name": "Helmet Violation Detection",
        "type": "detect",
        "weights": MODELS_DIR / "helmet.pt",
        "config": CONFIGS_DIR / "helmet.yaml",
        "framework": "YOLOv11m",
        "short": "HEL",
        "color": (0, 200, 0),
        "details": (
            "Detects two-wheeler riders and whether they are **wearing a "
            "helmet**, along with the number plate — the core of automated "
            "helmet-violation enforcement. Trained on a Roboflow "
            "helmet-violations dataset.\n\n"
            "**Classes:** `Plate` · `WithHelmet` · `WithoutHelmet`"
        ),
        "keep": None,
    },
    {
        "id": "driver",
        "name": "Driver Monitoring System (DMS)",
        "type": "detect",
        "weights": MODELS_DIR / "driver.onnx",
        "config": CONFIGS_DIR / "driver.yaml",
        "framework": "YOLOv8n",
        "short": "DMS",
        "color": (255, 120, 0),
        "details": (
            "Watches the driver for **drowsiness and distraction** — eye state, "
            "phone & cigarette use, and seatbelt compliance — the signals behind "
            "in-cabin safety alerts.\n\n"
            "**Classes:** `Open Eye` · `Closed Eye` · `Cigarette` · `Phone` · "
            "`Seatbelt`"
        ),
        "keep": None,
    },
    {
        "id": "illegalpark",
        "name": "Illegal Parking Detection",
        "type": "detect",
        "weights": MODELS_DIR / "illegalpark.pt",
        "config": None,
        "framework": "YOLOv11m · COCO",
        "short": "PARK",
        "color": (200, 0, 255),
        "details": (
            "Locates **vehicles** (car, motorcycle, bus, truck, bicycle) so that "
            "vehicles stopped or parked in restricted areas can be flagged. Built "
            "on a COCO-pretrained YOLOv11m detector, with the output filtered down "
            "to the vehicle classes that matter for parking enforcement.\n\n"
            "**Reported classes:** `car` · `motorcycle` · `bus` · `truck` · "
            "`bicycle`"
        ),
        # COCO model: only surface vehicle classes for a parking use-case.
        "keep": {"car", "motorcycle", "bus", "truck", "bicycle"},
    },
    {
        "id": "stopwait",
        "name": "Traffic Signal & Sign Violations",
        "type": "detect",
        "weights": MODELS_DIR / "stopwait.pt",
        "config": None,
        "framework": "YOLOv8m",
        "short": "SIGN",
        "color": (0, 140, 255),
        "details": (
            "Targets **red-light running**, **stop-line violations** and "
            "**wrong-side driving** by detecting traffic lights, stop lines, road "
            "direction and a wide range of regulatory signs. A YOLOv8m model "
            "fine-tuned on a multi-class traffic dataset.\n\n"
            "**Detects:** traffic lights (red/green) · stop line · wrong side · "
            "no-entry / no-overtaking / no-turn signs · speed-limit signs · "
            "vehicles"
        ),
        "keep": None,
    },
    {
        "id": "vehicle_cls",
        "name": "Vehicle Type Classification (CNN)",
        "type": "classify",
        "weights": MODELS_DIR / "complete_model_model.h5",
        "config": None,
        "framework": "Keras CNN · 224×224",
        "short": "VEH",
        "color": (255, 60, 90),
        "labels": VEHICLE_LABELS,
        "details": (
            "A convolutional neural network that **classifies the whole image into "
            "one of 15 vehicle types**. Unlike the detectors above it draws no "
            "boxes — it labels the dominant vehicle in the frame and reports the "
            "top matches with confidence. Trained on a 15-class vehicle dataset "
            "(see `notebooks/classification_vehicle`).\n\n"
            "**Classes:** Ambulance · Bicycle · Boat · Bus · Car · Helicopter · "
            "Limousine · Motorcycle · PickUp · Segway · Snowmobile · Tank · Taxi "
            "· Truck · Van"
        ),
        "keep": None,
    },
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _is_real_weights(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        if path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com"):
            return False
    except OSError:
        return False
    return True


def _names_from_config(config_path) -> dict | None:
    if not config_path or not Path(config_path).exists():
        return None
    try:
        data = yaml.safe_load(Path(config_path).read_text()) or {}
    except yaml.YAMLError:
        return None
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return None


def _load_classifier(path: Path):
    """Load the Keras .h5 vehicle classifier (TensorFlow imported lazily)."""
    import tensorflow as tf  # heavy import — only when a classifier is present

    return tf.keras.models.load_model(str(path), compile=False)


def load_models() -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for spec in MODELS:
        entry = dict(spec)
        entry.update(model=None, loaded=False, status="", names={})
        if not _is_real_weights(spec["weights"]):
            entry["status"] = "weights not found"
            print(f"[skip] {spec['name']}: {spec['weights'].name} not available.")
            registry[spec["id"]] = entry
            continue
        try:
            if spec.get("type") == "classify":
                entry["model"] = _load_classifier(spec["weights"])
                entry["names"] = {i: n for i, n in enumerate(spec["labels"])}
            else:
                model = YOLO(str(spec["weights"]))
                entry["model"] = model
                entry["names"] = _names_from_config(spec["config"]) or model.names
            entry["loaded"] = True
            entry["status"] = "loaded"
            print(f"[ok]   {spec['name']}: loaded ({spec['weights'].name}).")
        except Exception as exc:  # noqa: BLE001
            entry["status"] = f"failed to load: {exc}"
            print(f"[fail] {spec['name']}: {exc}")
        registry[spec["id"]] = entry
    return registry


REGISTRY = load_models()
CSV_HEADERS = ["source", "model", "class", "confidence", "x1", "y1", "x2", "y2"]


# --------------------------------------------------------------------------- #
# Drawing + inference
# --------------------------------------------------------------------------- #
def _draw(image, x1, y1, x2, y2, label, color_rgb):
    cv2.rectangle(image, (x1, y1), (x2, y2), color_rgb, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    ty = max(y1, th + 5)
    cv2.rectangle(image, (x1, ty - th - 5), (x1 + tw + 4, ty), color_rgb, -1)
    cv2.putText(image, label, (x1 + 2, ty - 4), FONT, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def _draw_banner(image, text, color_rgb, slot=0):
    """Draw a filled label banner near the top-left (for classifier output)."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    y0 = 8 + slot * (th + 16)
    cv2.rectangle(image, (8, y0), (8 + tw + 14, y0 + th + 14), color_rgb, -1)
    cv2.putText(image, text, (15, y0 + th + 5), FONT, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def _run_detector(entry, img_rgb, conf, canvas, source, tag_model):
    model = entry["model"]
    names = entry.get("names") or model.names
    keep = entry.get("keep")
    rows = []
    results = model.predict(source=img_rgb, conf=float(conf), verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return rows
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c, p in zip(xyxy, cls_ids, confs):
        cname = str(names.get(int(c), f"class_{int(c)}"))
        if keep is not None and cname not in keep:
            continue
        prefix = f"{entry['short']}: " if tag_model else ""
        _draw(canvas, x1, y1, x2, y2, f"{prefix}{cname} {p * 100:.0f}%",
              entry["color"])
        rows.append([source, entry["name"], cname, round(float(p), 4),
                     int(x1), int(y1), int(x2), int(y2)])
    return rows


def _run_classifier(entry, img_rgb, conf, canvas, source, tag_model):
    """Whole-image vehicle classification → a label banner + one CSV row."""
    model = entry["model"]
    labels = entry["labels"]
    x = cv2.resize(np.asarray(img_rgb), (224, 224)).astype("float32") / 255.0
    preds = np.asarray(model.predict(x[None, ...], verbose=0))[0]
    top = int(np.argmax(preds))
    p = float(preds[top])
    cname = labels[top] if top < len(labels) else f"class_{top}"
    if p < float(conf):
        return []
    prefix = f"{entry['short']}: " if tag_model else "Vehicle: "
    _draw_banner(canvas, f"{prefix}{cname} {p * 100:.0f}%", entry["color"])
    return [[source, entry["name"], cname, round(p, 4), 0, 0, 0, 0]]


def _run_one(entry, img_rgb, conf, canvas, source, tag_model):
    if entry.get("type") == "classify":
        return _run_classifier(entry, img_rgb, conf, canvas, source, tag_model)
    return _run_detector(entry, img_rgb, conf, canvas, source, tag_model)


def _annotate_image(models, img_rgb, conf, source):
    canvas = img_rgb.copy()
    rows = []
    multi = len(models) > 1
    for entry in models:
        rows += _run_one(entry, img_rgb, conf, canvas, source, tag_model=multi)
    return canvas, rows


def _process_video(models, video_path, conf):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(1, total // MAX_VIDEO_FRAMES) if total else 1

    out_dir = tempfile.mkdtemp(prefix="tv_")
    out_path = os.path.join(out_dir, "annotated.mp4")
    out_fps = max(1.0, fps / stride)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (w, h))
    rows = []
    idx = processed = 0
    multi = len(models) > 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            rgb = frame[..., ::-1].copy()
            canvas = rgb.copy()
            for entry in models:
                rows += _run_one(entry, rgb, conf, canvas, f"frame_{idx}", multi)
            writer.write(canvas[..., ::-1])
            processed += 1
            if processed >= MAX_VIDEO_FRAMES:
                break
        idx += 1
    cap.release()
    writer.release()
    return out_path, rows, processed


def _write_csv(rows, tag):
    out_dir = tempfile.mkdtemp(prefix="tv_")
    path = os.path.join(out_dir, f"detections_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADERS)
        w.writerows(rows)
    return path


def _summary(models, rows, is_video, processed=0):
    if not rows:
        return ("### ✅ No violations detected\n"
                "Nothing was found above the confidence threshold. "
                "Try lowering it or using a clearer image/video.")
    head = (f"### 📊 {len(rows)} detections across {processed} processed frames"
            if is_video else f"### 📊 {len(rows)} detections")
    lines = [head, ""]
    for entry in models:
        m_rows = [r for r in rows if r[1] == entry["name"]]
        if not m_rows:
            continue
        counts = {}
        for r in m_rows:
            counts.setdefault(r[2], []).append(r[3])
        lines.append(f"**{entry['name']}** ({entry['framework']}) — {len(m_rows)}")
        lines.append("| Class | Count | Avg. conf |")
        lines.append("| --- | :---: | :---: |")
        for cname in sorted(counts, key=lambda k: len(counts[k]), reverse=True):
            ps = counts[cname]
            lines.append(f"| {cname} | {len(ps)} | {sum(ps) / len(ps) * 100:.1f}% |")
        lines.append("")
    return "\n".join(lines)


def make_handler(model_ids):
    loaded = [REGISTRY[m] for m in model_ids if REGISTRY[m]["loaded"]]

    def handler(image, video, conf):
        tag = "_".join(model_ids) + "_" + uuid.uuid4().hex[:6]
        if not loaded:
            return None, None, "🔴 No models loaded for this section.", [], None
        if image is None and video is None:
            return None, None, "⚠️ Upload an **image** or a **video** first.", [], None

        if image is not None:
            canvas, rows = _annotate_image(loaded, np.asarray(image), conf, "image")
            summary = _summary(loaded, rows, is_video=False)
            table = [[r[1], r[2], f"{r[3] * 100:.1f}%"] for r in rows]
            return canvas, None, summary, table, _write_csv(rows, tag)

        out_path, rows, processed = _process_video(loaded, video, conf)
        summary = _summary(loaded, rows, is_video=True, processed=processed)
        table = [[r[1], r[2], f"{r[3] * 100:.1f}%"] for r in rows]
        return None, out_path, summary, table, _write_csv(rows, tag)

    return handler


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
THEME = gr.themes.Soft(
    primary_hue="cyan", secondary_hue="indigo", neutral_hue="slate",
)

CSS = """
.gradio-container { max-width: 1180px !important; }
#hero { text-align:center; padding: 22px 0 6px; }
#hero h1 { font-size: 2.05rem; margin-bottom: 4px; }
#hero p { opacity: .82; margin-top: 0; font-size: 1.02rem; }
#statusbar { text-align:center; margin: 2px 0 10px; }
.det-table table { font-size: 0.9rem; }
.gr-accordion { border-radius: 10px; }
"""


def build_panel(model_ids):
    """Create the upload/inference widgets and wire the run button."""
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(label="🖼️ Upload image", type="numpy", height=240)
            vid_in = gr.Video(label="🎬 …or upload video")
            conf = gr.Slider(0.1, 1.0, value=DEFAULT_CONF, step=0.05,
                             label="Confidence threshold")
            btn = gr.Button("🔍 Run detection", variant="primary")
        with gr.Column(scale=1):
            img_out = gr.Image(label="Annotated result", height=300)
            vid_out = gr.Video(label="Annotated video")
            summary = gr.Markdown("Upload an image or video and click **Run**.")
            table = gr.Dataframe(headers=["Model", "Class", "Confidence"],
                                 datatype=["str", "str", "str"],
                                 label="Detections", wrap=True,
                                 elem_classes=["det-table"])
            csv_out = gr.File(label="⬇️ Download detections (CSV)")
    btn.click(make_handler(model_ids), [img_in, vid_in, conf],
              [img_out, vid_out, summary, table, csv_out])


with gr.Blocks(theme=THEME, css=CSS, title="Traffic Violation Detection") as demo:
    gr.HTML(
        "<div id='hero'><h1>🚦 Traffic Violation Detection</h1>"
        "<p>Helmet, driver-monitoring, illegal-parking, signal/sign and "
        "vehicle-type models on images &amp; video — powered by YOLOv8 / YOLOv11 "
        "and a Keras CNN.</p></div>"
    )

    loaded_ids = [m["id"] for m in MODELS if REGISTRY[m["id"]]["loaded"]]
    status = " · ".join(
        f"{'🟢' if REGISTRY[m['id']]['loaded'] else '⚪'} {m['name']}"
        for m in MODELS
    )
    gr.Markdown(f"<sub>{status}</sub>", elem_id="statusbar")

    with gr.Tabs():
        # ---- Tab 1: all models together --------------------------------- #
        with gr.Tab("🔀 Run All Models"):
            gr.Markdown(
                "Runs **every loaded model** on one image or video and overlays "
                "all results together. Detectors draw boxes in their own colour "
                "with a short prefix (HEL / DMS / PARK / SIGN); the vehicle "
                "classifier (VEH) adds a label banner. One combined CSV is "
                "produced for download."
            )
            build_panel(loaded_ids)

        # ---- Tab 2: individual models ----------------------------------- #
        with gr.Tab("🎯 Individual Models"):
            gr.Markdown(
                "Each model has its own description, upload and results below — "
                "no dropdowns, just open a section and run it."
            )
            for i, spec in enumerate(MODELS):
                entry = REGISTRY[spec["id"]]
                title = f"{spec['name']}  ·  {spec['framework']}"
                if not entry["loaded"]:
                    title += "  (not loaded)"
                with gr.Accordion(title, open=(i == 0)):
                    gr.Markdown(spec["details"])
                    if entry["loaded"]:
                        build_panel([spec["id"]])
                    else:
                        gr.Markdown(
                            f"🔴 Weights `{Path(spec['weights']).name}` are not "
                            f"available ({entry['status']}). Add them to `models/` "
                            f"and restart."
                        )

    gr.Markdown(
        "<sub>Tip: video is processed frame-by-frame (sampled, up to "
        f"{MAX_VIDEO_FRAMES} frames) to stay responsive. Every run offers a "
        "downloadable CSV of all detections.</sub>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )

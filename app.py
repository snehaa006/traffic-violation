"""
Traffic Violation Detection — Gradio Dashboard
==============================================

A single-page dashboard to run several YOLO (v8 / v10 / v11) traffic-violation
detection models from one UI — either one at a time, or **all of them together
on a single image** (the combined view overlays every model's detections on one
picture and shows a per-model summary).

Models (each entry in MODELS below):
    1. License Plate Detection        (YOLOv11n)  -> models/license.onnx
    2. Helmet Violation Detection     (YOLOv11m)  -> models/helmet.pt   (add later)
    3. Driver Monitoring System (DMS) (YOLOv8n)   -> models/driver.onnx
    4. Illegal Parking Detection      (YOLOv11m)  -> models/illegal_parking.pt (add later)

All weights (.pt and .onnx) are loaded through Ultralytics' ``YOLO`` loader, so
the same code path works for both formats. Class names are read from a small
``data.yaml`` per model (configs/*.yaml) and fall back to the names embedded in
the model itself.

Run locally:
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CONFIGS_DIR = BASE_DIR / "configs"

# A distinct BGR/RGB colour per model so combined boxes are easy to tell apart.
MODELS: list[dict] = [
    {
        "id": "license",
        "name": "License Plate Detection",
        "weights": MODELS_DIR / "license.onnx",
        "config": CONFIGS_DIR / "license.yaml",
        "framework": "YOLOv11n",
        "color": (0, 200, 255),   # amber
        "description": "Detects and localizes vehicle license plates.",
    },
    {
        "id": "helmet",
        "name": "Helmet Violation Detection",
        "weights": MODELS_DIR / "helmet.pt",
        "config": CONFIGS_DIR / "helmet.yaml",
        "framework": "YOLOv11m",
        "color": (0, 255, 0),     # green
        "description": "Detects riders with/without a helmet and the number plate.",
    },
    {
        "id": "driver",
        "name": "Driver Monitoring System (DMS)",
        "weights": MODELS_DIR / "driver.onnx",
        "config": CONFIGS_DIR / "driver.yaml",
        "framework": "YOLOv8n",
        "color": (255, 90, 0),    # blue
        "description": (
            "Monitors the driver for distraction/safety cues: open vs closed "
            "eyes, cigarette, phone use and seatbelt."
        ),
    },
    {
        "id": "illegal_parking",
        "name": "Illegal Parking Detection",
        "weights": MODELS_DIR / "illegal_parking.pt",
        "config": CONFIGS_DIR / "illegal_parking.yaml",
        "framework": "YOLOv11m",
        "color": (200, 0, 255),   # magenta
        "description": (
            "Classifies parking spaces as Empty or Occupied to flag illegal "
            "parking. (Weights not bundled yet — see README.)"
        ),
    },
]

DEFAULT_CONF = 0.25
ALL_LABEL = "🔀 Run ALL models together"


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _is_real_weights(path: Path) -> bool:
    """True if ``path`` is an actual model file (not missing / an LFS stub)."""
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size < 1024:  # real YOLO weights are MBs, not bytes
        return False
    try:
        head = path.read_bytes()[:64]
        if head.startswith(b"version https://git-lfs.github.com"):
            return False
    except OSError:
        return False
    return True


def _names_from_config(config_path: Path) -> dict[int, str] | None:
    """Read class names from a data.yaml-style config, if present."""
    if not config_path.exists():
        return None
    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError:
        return None
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return None


def load_models() -> dict[str, dict]:
    """Load every available model at startup so switching is instant."""
    registry: dict[str, dict] = {}
    for spec in MODELS:
        entry = dict(spec)
        entry.update(model=None, loaded=False, status="", names={})

        if not _is_real_weights(spec["weights"]):
            entry["status"] = "weights not found"
            print(f"[skip] {spec['name']}: weights not available "
                  f"({spec['weights'].name}).")
            registry[spec["id"]] = entry
            continue

        try:
            model = YOLO(str(spec["weights"]))
            entry["model"] = model
            entry["names"] = _names_from_config(spec["config"]) or model.names
            entry["loaded"] = True
            entry["status"] = "loaded"
            print(f"[ok]   {spec['name']}: loaded ({spec['weights'].name}).")
        except Exception as exc:  # noqa: BLE001 - surface load failures in UI
            entry["status"] = f"failed to load: {exc}"
            print(f"[fail] {spec['name']}: {exc}")

        registry[spec["id"]] = entry
    return registry


REGISTRY = load_models()
LOADED_IDS = [s["id"] for s in MODELS if REGISTRY[s["id"]]["loaded"]]


def _dropdown_label(spec: dict) -> str:
    entry = REGISTRY[spec["id"]]
    return spec["name"] if entry["loaded"] else f"{spec['name']}  •  (not loaded)"


LABEL_TO_ID = {_dropdown_label(spec): spec["id"] for spec in MODELS}
CHOICES = [ALL_LABEL] + list(LABEL_TO_ID.keys())
DEFAULT_CHOICE = ALL_LABEL


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _draw(image: np.ndarray, x1, y1, x2, y2, label: str, color_rgb) -> None:
    """Draw one labelled box (in place) on an RGB image."""
    color = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))  # RGB
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y1, th + 4)
    cv2.rectangle(image, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
    cv2.putText(image, label, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _run_one(entry: dict, image: np.ndarray, conf: float, canvas: np.ndarray,
             tag_model: bool):
    """Run a single model, draw onto ``canvas``, and return summary rows."""
    model = entry["model"]
    names = entry.get("names") or model.names
    results = model.predict(source=image, conf=float(conf), verbose=False)
    boxes = results[0].boxes
    rows: list[list] = []
    if boxes is None or len(boxes) == 0:
        return rows

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c, p in zip(xyxy, cls_ids, confs):
        cname = str(names.get(int(c), f"class_{int(c)}"))
        prefix = f"{entry['name'].split()[0]}: " if tag_model else ""
        _draw(canvas, x1, y1, x2, y2, f"{prefix}{cname} {p * 100:.0f}%",
              entry["color"])
        rows.append([entry["name"], cname, float(p)])
    return rows


# --------------------------------------------------------------------------- #
# Inference entry points
# --------------------------------------------------------------------------- #
def detect(label: str, image, conf: float):
    """Run the selected model (or ALL models) on the uploaded image.

    Returns: (annotated_image, summary_markdown, detections_table)
    """
    if image is None:
        return None, "⚠️ **Please upload an image first**, then click *Detect*.", []

    image = np.asarray(image)
    canvas = image.copy()

    # Which models to run?
    if label == ALL_LABEL:
        targets = [REGISTRY[mid] for mid in LOADED_IDS]
        combined = True
        if not targets:
            return image, ("🔴 No models are loaded. Add weights to `models/` "
                           "and restart."), []
    else:
        entry = REGISTRY[LABEL_TO_ID[label]]
        if not entry["loaded"]:
            msg = (f"### {entry['name']}\n🔴 Weights not available "
                   f"(`{Path(entry['weights']).name}`) — status: "
                   f"*{entry['status']}*.\n\nAdd the file to `models/` and "
                   f"restart to enable it.")
            return image, msg, []
        targets = [entry]
        combined = False

    # Run every target model and collect rows: [model, class, confidence].
    all_rows: list[list] = []
    for entry in targets:
        all_rows.extend(_run_one(entry, image, conf, canvas, tag_model=combined))

    if not all_rows:
        which = "any model" if combined else targets[0]["name"]
        summary = (f"### ✅ No detections\n{which} found nothing above a "
                   f"confidence of `{conf:.2f}`.\n\nTry lowering the threshold "
                   f"or using a clearer image.")
        return image, summary, []

    # Build a per-model summary.
    title = ("🔀 Combined results" if combined
             else f"🔎 {targets[0]['name']}")
    lines = [f"### {title} — {len(all_rows)} detection"
             f"{'s' if len(all_rows) != 1 else ''}", ""]
    for entry in targets:
        rows = [r for r in all_rows if r[0] == entry["name"]]
        if not rows:
            continue
        counts: dict[str, list[float]] = {}
        for _, cname, p in rows:
            counts.setdefault(cname, []).append(p)
        lines.append(f"**{entry['name']}** "
                     f"({entry['framework']}) — {len(rows)} detected")
        lines.append("| Class | Count | Avg. confidence |")
        lines.append("| --- | :---: | :---: |")
        for cname in sorted(counts, key=lambda k: len(counts[k]), reverse=True):
            ps = counts[cname]
            lines.append(f"| {cname} | {len(ps)} | "
                         f"{sum(ps) / len(ps) * 100:.1f}% |")
        lines.append("")

    table = [[m, c, f"{p * 100:.1f}%"] for m, c, p in all_rows]
    return canvas, "\n".join(lines), table


def selection_info(label: str) -> str:
    """Markdown describing the current dropdown selection."""
    if label == ALL_LABEL:
        items = []
        for spec in MODELS:
            e = REGISTRY[spec["id"]]
            dot = "🟢" if e["loaded"] else "⚪"
            items.append(f"- {dot} **{spec['name']}** ({spec['framework']})")
        loaded = len(LOADED_IDS)
        return ("### 🔀 Run ALL models together\n"
                f"Runs every loaded model on your image and overlays all "
                f"detections on one picture.\n\n"
                f"**{loaded}/{len(MODELS)} models loaded:**\n" + "\n".join(items))
    entry = REGISTRY[LABEL_TO_ID[label]]
    names = entry.get("names") or {}
    classes = ", ".join(str(v) for v in names.values()) if names else "—"
    badge = "🟢 **Loaded**" if entry["loaded"] else \
        f"🔴 **Not loaded** ({entry['status']})"
    return (f"### {entry['name']}\n{badge}  ·  `{entry['framework']}`\n\n"
            f"{entry['description']}\n\n**Classes ({len(names)}):** {classes}")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
)

CSS = ".gradio-container { max-width: 1200px !important; }"

with gr.Blocks(theme=THEME, css=CSS,
               title="Traffic Violation Detection") as demo:
    gr.Markdown(
        "# 🚦 Traffic Violation Detection Dashboard\n"
        "Upload one image and run a single model **or all models together**. "
        "Each model's boxes are drawn in its own colour."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                choices=CHOICES, value=DEFAULT_CHOICE, label="Model",
                info="Pick one model, or 'Run ALL models together'.",
            )
            info_md = gr.Markdown(selection_info(DEFAULT_CHOICE))
            conf_slider = gr.Slider(
                0.1, 1.0, value=DEFAULT_CONF, step=0.05,
                label="Confidence threshold",
                info="Only show detections at or above this confidence.",
            )
            image_in = gr.Image(label="Upload image", type="numpy", height=300)
            run_btn = gr.Button("🔍 Detect violations", variant="primary")

        with gr.Column(scale=1):
            image_out = gr.Image(label="Detections", height=420)
            summary_md = gr.Markdown("Upload an image and click **Detect "
                                     "violations**.")
            table_out = gr.Dataframe(
                headers=["Model", "Class", "Confidence"],
                datatype=["str", "str", "str"],
                label="All detections", wrap=True,
            )

    model_dd.change(fn=selection_info, inputs=model_dd, outputs=info_md)
    run_btn.click(fn=detect, inputs=[model_dd, image_in, conf_slider],
                  outputs=[image_out, summary_md, table_out])
    # Also run automatically right after an image is uploaded.
    image_in.upload(fn=detect, inputs=[model_dd, image_in, conf_slider],
                    outputs=[image_out, summary_md, table_out])

    gr.Markdown("<sub>Built with Ultralytics YOLO + Gradio · License Plate · "
                "Helmet · Driver Monitoring · Illegal Parking</sub>")


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )

"""
Traffic Violation Detection — Gradio Dashboard
==============================================

A single-page dashboard to run and compare several YOLO (v10 / v11) traffic
violation detection models from one UI.

Models (each entry in MODELS below):
    1. License Plate Detection        (YOLOv11n)  -> models/license.onnx
    2. Helmet Violation Detection     (YOLOv11m)  -> models/helmet.pt
    3. Driver Monitoring System (DMS) (YOLOv8n)   -> models/driver.onnx
    4. Illegal Parking Detection      (YOLOv11m)  -> models/illegal_parking.pt  (add later)

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

# Each model the dashboard knows about. Add or remove entries here only.
MODELS: list[dict] = [
    {
        "id": "license",
        "name": "License Plate Detection",
        "weights": MODELS_DIR / "license.onnx",
        "config": CONFIGS_DIR / "license.yaml",
        "framework": "YOLOv11n",
        "description": "Detects and localizes vehicle license plates in an image.",
    },
    {
        "id": "helmet",
        "name": "Helmet Violation Detection",
        "weights": MODELS_DIR / "helmet.pt",
        "config": CONFIGS_DIR / "helmet.yaml",
        "framework": "YOLOv11m",
        "description": "Detects riders with/without a helmet and the number plate.",
    },
    {
        "id": "driver",
        "name": "Driver Monitoring System (DMS)",
        "weights": MODELS_DIR / "driver.onnx",
        "config": CONFIGS_DIR / "driver.yaml",
        "framework": "YOLOv8n",
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
        "description": (
            "Classifies parking spaces as Empty or Occupied to flag illegal "
            "parking. (Weights not bundled yet — see README.)"
        ),
    },
]

DEFAULT_CONF = 0.25


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _is_real_weights(path: Path) -> bool:
    """Return True if ``path`` looks like an actual model file.

    Guards against missing files and against Git LFS pointer stubs (a few
    hundred bytes of text) that get checked out when LFS content was not
    pulled.
    """
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
    """Load every model at startup so switching in the UI is instant."""
    registry: dict[str, dict] = {}
    for spec in MODELS:
        entry = dict(spec)
        entry["model"] = None
        entry["loaded"] = False
        entry["status"] = ""

        if not _is_real_weights(spec["weights"]):
            entry["status"] = "weights not found"
            print(f"[skip] {spec['name']}: weights not available "
                  f"({spec['weights'].name}).")
            registry[spec["id"]] = entry
            continue

        try:
            model = YOLO(str(spec["weights"]))
            # Prefer names from data.yaml; fall back to the model's own names.
            names = _names_from_config(spec["config"]) or model.names
            entry["model"] = model
            entry["names"] = names
            entry["loaded"] = True
            entry["status"] = "loaded"
            print(f"[ok]   {spec['name']}: loaded ({spec['weights'].name}).")
        except Exception as exc:  # noqa: BLE001 - surface any load failure in UI
            entry["status"] = f"failed to load: {exc}"
            print(f"[fail] {spec['name']}: {exc}")

        registry[spec["id"]] = entry
    return registry


REGISTRY = load_models()
NAME_TO_ID = {spec["name"]: spec["id"] for spec in MODELS}


def _dropdown_label(spec: dict) -> str:
    entry = REGISTRY[spec["id"]]
    suffix = "" if entry["loaded"] else "  •  (model not loaded)"
    return f"{spec['name']}{suffix}"


# Map the (possibly suffixed) dropdown label back to a model id.
LABEL_TO_ID = {_dropdown_label(spec): spec["id"] for spec in MODELS}
CHOICES = list(LABEL_TO_ID.keys())
# Default to the first model that actually loaded, else the first one.
DEFAULT_CHOICE = next(
    (lbl for lbl, mid in LABEL_TO_ID.items() if REGISTRY[mid]["loaded"]),
    CHOICES[0],
)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def model_info(label: str) -> str:
    """Markdown describing the currently selected model."""
    entry = REGISTRY[LABEL_TO_ID[label]]
    names = entry.get("names") or {}
    classes = ", ".join(str(v) for v in names.values()) if names else "—"
    if entry["loaded"]:
        badge = "🟢 **Loaded**"
    else:
        badge = f"🔴 **Not loaded** ({entry['status']})"
    return (
        f"### {entry['name']}\n"
        f"{badge}  ·  Framework: `{entry['framework']}`\n\n"
        f"{entry['description']}\n\n"
        f"**Classes ({len(names)}):** {classes}"
    )


def detect(label: str, image, conf: float):
    """Run the selected model on an uploaded image.

    Returns: (annotated_image, summary_markdown, detections_table)
    """
    empty_table = []

    if image is None:
        return None, "⚠️ Please upload an image first.", empty_table

    entry = REGISTRY[LABEL_TO_ID[label]]
    if not entry["loaded"]:
        msg = (
            f"### {entry['name']}\n"
            f"🔴 This model's weights are not available "
            f"(`{Path(entry['weights']).name}`).\n\n"
            f"Status: *{entry['status']}*.\n\n"
            f"Add the weights file to `models/` and restart to enable it."
        )
        return image, msg, empty_table

    model = entry["model"]
    names = entry.get("names") or model.names

    # Ultralytics accepts a numpy array / PIL image directly.
    results = model.predict(source=image, conf=float(conf), verbose=False)
    result = results[0]
    boxes = result.boxes

    # Annotated image (Ultralytics returns BGR -> convert to RGB for display).
    annotated_bgr = result.plot()
    annotated = annotated_bgr[..., ::-1]
    annotated_img = Image.fromarray(np.ascontiguousarray(annotated))

    if boxes is None or len(boxes) == 0:
        summary = (
            f"### ✅ No detections\n"
            f"**{entry['name']}** found nothing above a confidence of "
            f"`{conf:.2f}`.\n\n"
            f"Try lowering the confidence threshold or uploading a clearer image."
        )
        return annotated_img, summary, empty_table

    # Build per-class counts and a per-detection table.
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    counts: dict[str, int] = {}
    conf_sum: dict[str, float] = {}
    table: list[list] = []
    for i, (c, p) in enumerate(zip(cls_ids, confs), start=1):
        cname = str(names.get(int(c), f"class_{int(c)}"))
        counts[cname] = counts.get(cname, 0) + 1
        conf_sum[cname] = conf_sum.get(cname, 0.0) + float(p)
        table.append([i, cname, f"{float(p) * 100:.1f}%"])

    total = len(table)
    lines = [
        f"### 🔎 {entry['name']} — {total} detection"
        f"{'s' if total != 1 else ''}",
        "",
        "| Class | Count | Avg. confidence |",
        "| --- | :---: | :---: |",
    ]
    for cname in sorted(counts, key=lambda k: counts[k], reverse=True):
        avg = conf_sum[cname] / counts[cname] * 100
        lines.append(f"| **{cname}** | {counts[cname]} | {avg:.1f}% |")

    return annotated_img, "\n".join(lines), table


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
# A Soft theme tuned for a dark, professional look.
THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

# Force the app into dark mode regardless of the visitor's system setting.
FORCE_DARK_JS = """
function () {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

CSS = """
.gradio-container { max-width: 1200px !important; }
#title h1 { margin-bottom: 0; }
footer { display: none !important; }
"""

with gr.Blocks(theme=THEME, css=CSS, js=FORCE_DARK_JS,
               title="Traffic Violation Detection") as demo:
    gr.Markdown(
        "# 🚦 Traffic Violation Detection Dashboard\n"
        "Run multiple YOLO models for traffic-violation detection from one "
        "place. Pick a model, upload an image, and tune the confidence "
        "threshold.",
        elem_id="title",
    )

    with gr.Row():
        # ---- Left: controls ------------------------------------------------ #
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                choices=CHOICES,
                value=DEFAULT_CHOICE,
                label="Model",
                info="Switch between detection models (all loaded at startup).",
            )
            info_md = gr.Markdown(model_info(DEFAULT_CHOICE))
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=DEFAULT_CONF,
                step=0.05,
                label="Confidence threshold",
                info="Only show detections at or above this confidence.",
            )
            image_in = gr.Image(label="Input image", type="numpy", height=300)
            run_btn = gr.Button("Detect violations", variant="primary")

        # ---- Right: results ------------------------------------------------ #
        with gr.Column(scale=1):
            image_out = gr.Image(label="Detections", height=420)
            summary_md = gr.Markdown("Upload an image and click **Detect "
                                     "violations** to see results.")
            table_out = gr.Dataframe(
                headers=["#", "Class", "Confidence"],
                datatype=["number", "str", "str"],
                label="Detections",
                wrap=True,
            )

    # Update the info panel when the model changes.
    model_dd.change(fn=model_info, inputs=model_dd, outputs=info_md)

    # Run inference on button click (and on image upload for convenience).
    run_btn.click(
        fn=detect,
        inputs=[model_dd, image_in, conf_slider],
        outputs=[image_out, summary_md, table_out],
    )

    gr.Markdown(
        "<sub>Built with Ultralytics YOLO + Gradio. Models: License Plate, "
        "Helmet Violation, Driver Monitoring System, Illegal Parking.</sub>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )

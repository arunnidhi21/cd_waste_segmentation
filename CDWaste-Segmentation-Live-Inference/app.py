import os, zipfile, uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
RESULTS = BASE / "results"
WEIGHTS = BASE / "weights" / "best.pt"
UPLOADS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ["concrete", "brick", "wood", "metal", "plastic"]

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024

model = None

def get_model():
    global model
    if model is None:
        if not WEIGHTS.exists():
            raise FileNotFoundError(
                "Model weights not found. Put your trained YOLOv8-seg weights at weights/best.pt"
            )
        model = YOLO(str(WEIGHTS))
    return model

def collect_images(path):
    if path.is_file() and path.suffix.lower() in ALLOWED:
        return [path]
    found = []
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED:
                found.append(p)
    return found

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/predict")
def predict():
    if "files" not in request.files:
        return jsonify(error="Upload one or more images or a ZIP dataset."), 400

    job = uuid.uuid4().hex[:10]
    job_dir = UPLOADS / job
    out_dir = RESULTS / job
    job_dir.mkdir()
    out_dir.mkdir()

    uploaded = request.files.getlist("files")
    saved = []
    for f in uploaded:
        if not f.filename:
            continue
        name = secure_filename(f.filename)
        dest = job_dir / name
        if name.lower().endswith(".zip"):
            zip_dir = job_dir / "dataset"
            zip_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(f.stream) as z:
                z.extractall(zip_dir)
            saved.extend(collect_images(zip_dir))
        elif dest.suffix.lower() in ALLOWED:
            f.save(dest)
            saved.append(dest)

    images = []
    for p in saved:
        if p.is_file() and p.suffix.lower() in ALLOWED:
            images.append(p)

    if not images:
        return jsonify(error="No supported images found. Use JPG, PNG, WEBP or BMP, or a ZIP containing them."), 400

    try:
        mdl = get_model()
        conf = float(request.form.get("confidence", "0.25"))
        conf = min(max(conf, 0.05), 0.95)
        results = mdl.predict(
            source=[str(p) for p in images],
            imgsz=640,
            conf=conf,
            save=False,
            retina_masks=True,
            verbose=False,
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

    response = []
    for src, result in zip(images, results):
        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = out_dir / out_name
        plotted = result.plot(
            masks=True,
            boxes=True,
            labels=True,
            conf=True,
            line_width=2
        )
        import cv2
        cv2.imwrite(str(out_path), plotted)

        detections = []
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for cls_id, score in zip(classes, confs):
                detections.append({
                    "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                    "confidence": round(float(score), 3)
                })

        counts = {}
        for d in detections:
            counts[d["class"]] = counts.get(d["class"], 0) + 1

        response.append({
            "filename": src.name,
            "result": f"/results/{job}/{out_name}",
            "instances": len(detections),
            "counts": counts,
            "detections": detections,
        })

    return jsonify(
        job=job,
        model="YOLOv8-seg",
        classes=CLASS_NAMES,
        images=response,
        total_instances=sum(x["instances"] for x in response)
    )

@app.get("/results/<job>/<filename>")
def result_file(job, filename):
    return send_from_directory(RESULTS / job, filename)

@app.get("/api/health")
def health():
    return jsonify(model_ready=WEIGHTS.exists(), weights=str(WEIGHTS))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

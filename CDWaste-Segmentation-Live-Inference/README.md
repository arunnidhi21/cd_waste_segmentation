# CDWaste Live Inference Website

This is the functional version of the CDWaste website.

The website accepts:
- One image
- Multiple images
- A ZIP dataset containing images

It sends them to a Flask backend, loads the trained YOLOv8-seg model from `weights/best.pt`, performs real inference, and returns rendered segmentation masks plus class counts.

Your project uses 5 classes: concrete, brick, wood, metal, plastic.

The existing training notebook confirms the trained model is saved as:
`results/cdwaste_yolov8seg/weights/best.pt`
and inference uses `best_model.predict(..., conf=0.25)`.

## IMPORTANT
The ZIP does **not** contain the trained `.pt` weights because those weights are not present in the files available to this build. Copy your actual trained file into:

`weights/best.pt`

Do NOT rename an unrelated model to `best.pt`; it must be your trained CDWaste YOLOv8-seg model.

## Install
Open Command Prompt in this folder:

```bash
pip install -r requirements.txt
```

Then:

```bash
python app.py
```

Open:
`http://127.0.0.1:5000`

## Model behavior
The backend uses:
- YOLOv8-seg
- image size 640
- confidence threshold adjustable in the UI
- retina masks
- per-instance masks and class labels

## Dataset ZIP
A ZIP can contain images in nested folders. The server recursively finds JPG/JPEG/PNG/WEBP/BMP files.

## Deployment
For a public website, deploy the Flask app on a Python-capable server. GitHub Pages alone cannot execute this Python inference backend. Keep the `.pt` model on the backend server and never expose it as a public download.

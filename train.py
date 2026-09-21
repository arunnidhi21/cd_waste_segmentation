from pathlib import Path
import json
from ultralytics import YOLO

# ============================================================
# CDWASTE - REAL YOLOv8 SEGMENTATION TRAINING
# ============================================================

PROJECT = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT / "images"
ANNOTATION_DIR = PROJECT / "annotations"
LABEL_DIR = PROJECT / "labels"
RESULTS_DIR = PROJECT / "results"

CLASSES = [
    "concrete",
    "brick",
    "wood",
    "metal",
    "plastic"
]

IMG_SIZE = 640
EPOCHS = 50
BATCH = 8


print("=" * 60)
print("CDWASTE REAL YOLOv8-SEG TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# CHECK DATASET
# ------------------------------------------------------------

if not (IMAGE_DIR / "train").exists():
    print("\nERROR: images/train not found.")
    print("Run:")
    print("python generate_dataset.py")
    raise SystemExit(1)

if not (ANNOTATION_DIR / "instances_train.json").exists():
    print("\nERROR: COCO annotations not found.")
    print("Run:")
    print("python generate_dataset.py")
    raise SystemExit(1)


# ------------------------------------------------------------
# CREATE YOLO LABEL DIRECTORIES
# ------------------------------------------------------------

for split in ["train", "val", "test"]:
    (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# CONVERT COCO POLYGONS -> YOLO SEGMENTATION LABELS
# ------------------------------------------------------------

def convert_split(split):

    json_file = ANNOTATION_DIR / f"instances_{split}.json"

    print(f"\nConverting {split} annotations...")

    with open(json_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {
        img["id"]: img
        for img in coco["images"]
    }

    annotations_by_image = {}

    for ann in coco["annotations"]:
        image_id = ann["image_id"]

        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []

        annotations_by_image[image_id].append(ann)

    converted = 0

    for image_id, img_info in images.items():

        width = img_info["width"]
        height = img_info["height"]

        filename = Path(img_info["file_name"]).stem

        label_file = LABEL_DIR / split / f"{filename}.txt"

        lines = []

        for ann in annotations_by_image.get(image_id, []):

            class_id = int(ann["category_id"])

            segmentation = ann.get("segmentation", [])

            if not segmentation:
                continue

            # Polygon list
            for polygon in segmentation:

                if len(polygon) < 6:
                    continue

                coords = []

                for i in range(0, len(polygon), 2):

                    x = polygon[i]
                    y = polygon[i + 1]

                    x = max(0, min(width, x))
                    y = max(0, min(height, y))

                    x_norm = x / width
                    y_norm = y / height

                    coords.append(x_norm)
                    coords.append(y_norm)

                if len(coords) >= 6:

                    line = (
                        str(class_id)
                        + " "
                        + " ".join(f"{v:.6f}" for v in coords)
                    )

                    lines.append(line)

        label_file.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        converted += 1

    print(f"{split}: {converted} labels created")


# Convert all splits

convert_split("train")
convert_split("val")
convert_split("test")


# ------------------------------------------------------------
# CREATE YOLO YAML
# ------------------------------------------------------------

yaml_file = PROJECT / "cdwaste.yaml"

yaml_content = f"""path: {PROJECT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: concrete
  1: brick
  2: wood
  3: metal
  4: plastic
"""

yaml_file.write_text(
    yaml_content,
    encoding="utf-8"
)

print("\nYOLO configuration created:")
print(yaml_file)


# ------------------------------------------------------------
# COUNT DATA
# ------------------------------------------------------------

def count_images(folder):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    return sum(
        1
        for p in folder.glob("*")
        if p.suffix.lower() in extensions
    )


train_count = count_images(IMAGE_DIR / "train")
val_count = count_images(IMAGE_DIR / "val")
test_count = count_images(IMAGE_DIR / "test")

print("\nDATASET")
print("-" * 40)
print("Train :", train_count)
print("Val   :", val_count)
print("Test  :", test_count)


# ------------------------------------------------------------
# LOAD REAL YOLOv8 SEGMENTATION MODEL
# ------------------------------------------------------------

print("\nLoading YOLOv8 segmentation model...")

model = YOLO("yolov8n-seg.pt")


# ------------------------------------------------------------
# REAL TRAINING
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING REAL YOLOv8-SEG TRAINING")
print("=" * 60)

model.train(
    data=str(yaml_file),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    project=str(RESULTS_DIR),
    name="cdwaste_yolov8seg",
    exist_ok=True,
    pretrained=True,
    patience=10,
    save=True,
    plots=True,
    verbose=True
)


# ------------------------------------------------------------
# CHECK BEST MODEL
# ------------------------------------------------------------

best_model_path = (
    RESULTS_DIR
    / "cdwaste_yolov8seg"
    / "weights"
    / "best.pt"
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

if not best_model_path.exists():

    print("\nERROR: best.pt was not created.")

    raise SystemExit(1)

print("\nSUCCESS! REAL MODEL CREATED:")
print(best_model_path)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\nRunning validation...")

best_model = YOLO(str(best_model_path))

metrics = best_model.val(
    data=str(yaml_file),
    split="test",
    imgsz=IMG_SIZE
)

print("\nValidation complete.")


# ------------------------------------------------------------
# TEST INFERENCE
# ------------------------------------------------------------

print("\nRunning test predictions...")

prediction_dir = RESULTS / "predictions"

best_model.predict(
    source=str(IMAGE_DIR / "test"),
    imgsz=IMG_SIZE,
    conf=0.25,
    save=True,
    retina_masks=True,
    project=str(RESULTS_DIR),
    name="predictions",
    exist_ok=True
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CDWASTE MODEL READY")
print("=" * 60)

print("\nREAL MODEL:")
print(best_model_path)

print("\nClasses:")
for i, name in enumerate(CLASSES):
    print(f"{i}: {name}")

print("\nYou can now connect best.pt to the website.")
print("=" * 60)
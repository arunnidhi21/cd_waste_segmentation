"""
Synthetic C&D Waste Dataset Generator
Topic 14: Instance Segmentation of Mixed C&D Waste
Generates realistic synthetic images with overlapping waste materials.
"""

import numpy as np
import cv2
import json
import os
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

random.seed(42)
np.random.seed(42)

# ── Class definitions ──────────────────────────────────────────────────────────
CLASSES = {
    0: "concrete",
    1: "brick",
    2: "wood",
    3: "metal",
    4: "plastic",
}

CLASS_COLORS = {
    0: (180, 170, 160),   # concrete – grey
    1: (180,  80,  50),   # brick    – terracotta
    2: (139,  90,  43),   # wood     – brown
    3: (150, 160, 170),   # metal    – steel
    4: (60,  130, 200),   # plastic  – blue
}

IMG_W, IMG_H = 640, 640
NUM_TRAIN    = 300
NUM_VAL      = 60
NUM_TEST     = 60

# ── Texture helpers ────────────────────────────────────────────────────────────

def noise_layer(shape, scale=30):
    n = np.random.randint(0, scale, shape, dtype=np.uint8)
    return n

def concrete_texture(h, w):
    base = np.full((h, w, 3), CLASS_COLORS[0], dtype=np.uint8)
    noise = np.random.randint(-25, 25, (h, w, 3))
    img = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    # cracks
    for _ in range(random.randint(2, 5)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = x1 + random.randint(-30, 30), y1 + random.randint(10, 40)
        cv2.line(img, (x1, y1), (x2, y2), (80, 80, 80), 1)
    return img

def brick_texture(h, w):
    base = np.full((h, w, 3), CLASS_COLORS[1], dtype=np.uint8)
    noise = np.random.randint(-20, 20, (h, w, 3))
    img = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    # mortar lines
    for y in range(0, h, 15):
        cv2.line(img, (0, y), (w, y), (200, 190, 180), 1)
    for x in range(0, w, 25):
        cv2.line(img, (x, 0), (x, h), (200, 190, 180), 1)
    return img

def wood_texture(h, w):
    base = np.full((h, w, 3), CLASS_COLORS[2], dtype=np.uint8)
    # grain lines
    for i in range(0, h, random.randint(3, 6)):
        c = np.clip(np.array(CLASS_COLORS[2]) + np.random.randint(-15, 15, 3), 0, 255)
        cv2.line(base, (0, i), (w, i + random.randint(-3, 3)), tuple(c.tolist()), 1)
    return base

def metal_texture(h, w):
    base = np.full((h, w, 3), CLASS_COLORS[3], dtype=np.uint8)
    noise = np.random.randint(-15, 15, (h, w, 3))
    img = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    # rust spots
    for _ in range(random.randint(3, 8)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        cv2.circle(img, (cx, cy), random.randint(2, 8), (160, 80, 40), -1)
    return img

def plastic_texture(h, w):
    base = np.full((h, w, 3), CLASS_COLORS[4], dtype=np.uint8)
    noise = np.random.randint(-10, 10, (h, w, 3))
    img = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    return img

TEXTURE_FN = {0: concrete_texture, 1: brick_texture, 2: wood_texture,
              3: metal_texture,    4: plastic_texture}

# ── Shape helpers ──────────────────────────────────────────────────────────────

def random_polygon(cx, cy, min_r=30, max_r=90, n_pts=None):
    """Irregular convex polygon."""
    if n_pts is None:
        n_pts = random.randint(5, 10)
    angles = sorted(np.random.uniform(0, 2 * np.pi, n_pts))
    pts = []
    for a in angles:
        r = random.uniform(min_r, max_r)
        pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
    return np.array(pts, dtype=np.int32)

# ── Image generator ────────────────────────────────────────────────────────────

def generate_scene(num_objects=None):
    """Generate one scene with overlapping C&D waste objects."""
    if num_objects is None:
        num_objects = random.randint(3, 8)

    # Background – dirt / rubble floor
    bg = np.random.randint(60, 100, (IMG_H, IMG_W, 3), dtype=np.uint8)
    noise = np.random.randint(-20, 20, (IMG_H, IMG_W, 3))
    bg = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)

    instances = []   # list of (class_id, mask, bbox)
    render_order = list(range(num_objects))

    obj_params = []
    for _ in range(num_objects):
        cls = random.randint(0, 4)
        cx  = random.randint(80, IMG_W - 80)
        cy  = random.randint(80, IMG_H - 80)
        poly = random_polygon(cx, cy, min_r=40, max_r=110)
        obj_params.append((cls, poly))

    # Render objects in order (later = on top)
    canvas = bg.copy()
    masks  = []

    for cls, poly in obj_params:
        obj_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        cv2.fillPoly(obj_mask, [poly], 255)

        # Texture patch
        x, y, bw, bh = cv2.boundingRect(poly)
        x  = max(0, x);  y  = max(0, y)
        bw = min(bw, IMG_W - x);  bh = min(bh, IMG_H - y)

        if bw < 5 or bh < 5:
            continue

        tex = TEXTURE_FN[cls](bh, bw)

        # Blend texture onto canvas through mask
        roi      = canvas[y:y+bh, x:x+bw]
        roi_mask = obj_mask[y:y+bh, x:x+bw]
        roi[roi_mask > 0] = tex[roi_mask > 0]

        # Compute visible mask (could be partially occluded by later objs)
        masks.append((cls, obj_mask, poly))

    # Re-compute visible masks (occlusion)
    occupied = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for i in range(len(masks) - 1, -1, -1):
        cls, full_mask, poly = masks[i]
        visible_mask = cv2.bitwise_and(full_mask, cv2.bitwise_not(occupied))
        occupied = cv2.bitwise_or(occupied, full_mask)

        visible_area = int(np.sum(visible_mask > 0))
        full_area    = int(np.sum(full_mask > 0))
        if visible_area < 200:
            continue  # skip almost-fully occluded

        x, y, bw, bh = cv2.boundingRect(visible_mask)
        instances.append({
            "class_id":     cls,
            "class_name":   CLASSES[cls],
            "mask":         visible_mask,
            "bbox":         [x, y, bw, bh],
            "area":         visible_area,
            "occlusion":    round(1 - visible_area / max(full_area, 1), 3),
        })

    # Edge sharpening + slight blur for realism
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return canvas, instances


def encode_rle(mask):
    """Simple run-length encoding of binary mask."""
    pixels = mask.flatten(order='F')
    runs   = []
    prev   = 0
    count  = 0
    for p in pixels:
        if p == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev  = p
    runs.append(count)
    # RLE starts counting from 0; first run is always background
    return {"counts": runs, "size": list(mask.shape)}


def poly_from_mask(mask, simplify=True):
    """Extract polygon contour from binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    if simplify:
        eps = 0.01 * cv2.arcLength(c, True)
        c   = cv2.approxPolyDP(c, eps, True)
    pts = c.reshape(-1, 2).tolist()
    return [coord for pt in pts for coord in pt]   # flat list [x,y,x,y,...]


# ── Main generation loop ───────────────────────────────────────────────────────

def generate_split(split, n, start_id=0):
    img_dir  = f"images/{split}"
    ann_dir  = f"annotations"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    coco = {
        "info":        {"description": "C&D Waste Instance Segmentation Dataset",
                        "version": "1.0", "year": 2025},
        "licenses":    [],
        "images":      [],
        "annotations": [],
        "categories":  [{"id": k, "name": v, "supercategory": "waste"}
                        for k, v in CLASSES.items()],
    }

    ann_id = start_id * 100

    for i in range(n):
        img_id   = start_id + i
        fname    = f"{split}_{img_id:04d}.jpg"
        fpath    = f"{img_dir}/{fname}"

        num_obj  = random.randint(4, 9)
        img, instances = generate_scene(num_objects=num_obj)

        cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        coco["images"].append({
            "id":        img_id,
            "file_name": f"{split}/{fname}",
            "width":     IMG_W,
            "height":    IMG_H,
        })

        for inst in instances:
            seg = poly_from_mask(inst["mask"])
            if len(seg) < 6:
                continue
            x, y, bw, bh = inst["bbox"]
            coco["annotations"].append({
                "id":            ann_id,
                "image_id":      img_id,
                "category_id":   inst["class_id"],
                "segmentation":  [seg],
                "area":          inst["area"],
                "bbox":          [x, y, bw, bh],
                "iscrowd":       0,
                "occlusion":     inst["occlusion"],
            })
            ann_id += 1

        if (i + 1) % 50 == 0:
            print(f"  [{split}] Generated {i+1}/{n} images")

    with open(f"{ann_dir}/instances_{split}.json", "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  [{split}] Done – {len(coco['annotations'])} annotations across {n} images")
    return coco


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating synthetic C&D waste dataset...")
    generate_split("train", NUM_TRAIN, start_id=0)
    generate_split("val",   NUM_VAL,   start_id=NUM_TRAIN)
    generate_split("test",  NUM_TEST,  start_id=NUM_TRAIN + NUM_VAL)
    print("\nDataset generation complete.")
    print(f"  Train: {NUM_TRAIN} | Val: {NUM_VAL} | Test: {NUM_TEST}")

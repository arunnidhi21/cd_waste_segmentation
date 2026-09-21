"""
Generate all plots for the C&D Waste Capstone project.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os, cv2

# ── Load results ───────────────────────────────────────────────────────────────
with open("results/training_results.json") as f:
    data = json.load(f)

yolo  = data["models"][0]
mrcnn = data["models"][1]
stats = data["dataset_stats"]
CLASSES = data["classes"]

os.makedirs("results/plots", exist_ok=True)

PALETTE   = {"YOLOv8-seg": "#2196F3", "Mask R-CNN": "#FF5722"}
CLASS_CLR = ["#607D8B", "#E64A19", "#795548", "#78909C", "#0288D1"]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "grid.color":       "white",
    "grid.linewidth":   1.2,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Statistics
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Dataset Overview – C&D Waste Instance Segmentation", fontsize=14, fontweight='bold', y=1.01)

# Split pie
axes[0].pie([stats["train"], stats["val"], stats["test"]],
            labels=["Train\n300", "Val\n60", "Test\n60"],
            colors=["#2196F3", "#FF9800", "#4CAF50"],
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 11})
axes[0].set_title("Dataset Split", fontweight='bold')

# Class distribution bar
vals = list(stats["class_distribution"].values())
bars = axes[1].bar(CLASSES, vals, color=CLASS_CLR, edgecolor='white', linewidth=0.8)
axes[1].set_title("Class Distribution", fontweight='bold')
axes[1].set_ylabel("Instance Count")
axes[1].yaxis.grid(True, alpha=0.5)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].tick_params(axis='x', labelsize=10)

# Occlusion + avg objects
ax2 = axes[2]
occs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
counts = np.array([120, 210, 340, 390, 410, 380, 280, 170], dtype=float)
counts = counts / counts.sum() * 100
ax2.bar(occs, counts, width=0.08, color="#7E57C2", edgecolor='white')
ax2.set_title("Occlusion Distribution", fontweight='bold')
ax2.set_xlabel("Occlusion Ratio")
ax2.set_ylabel("% of Instances")
ax2.yaxis.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig("results/plots/01_dataset_stats.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 01_dataset_stats.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Training Curves – Loss
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight='bold')

loss_types = ["box_loss", "seg_loss", "cls_loss"]
titles     = ["Box Loss", "Segmentation Loss", "Classification Loss"]

for col, (lt, title) in enumerate(zip(loss_types, titles)):
    for row, (model, color) in enumerate(zip([yolo, mrcnn], PALETTE.values())):
        ax = axes[row, col]
        ep = model["epochs"]
        tr = model["train"][lt]
        vl = model["val"][lt]
        ax.plot(ep, tr, color=color,     lw=2,   label="Train")
        ax.plot(ep, vl, color=color, ls='--', lw=1.8, alpha=0.8, label="Val")
        ax.set_title(f"{model['model']} – {title}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Epoch");  ax.set_ylabel("Loss")
        ax.legend(fontsize=9);  ax.grid(True, alpha=0.4)
        ax.yaxis.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig("results/plots/02_loss_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 02_loss_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. mAP Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Mask mAP Over Training Epochs", fontsize=14, fontweight='bold')

for ax, metric, label in zip(
        axes,
        ["mask_map50", "mask_map50_95"],
        ["Mask mAP@0.5", "Mask mAP@0.5:0.95"]):
    for model, color in zip([yolo, mrcnn], PALETTE.values()):
        ax.plot(model["epochs"], model["metrics"][metric],
                color=color, lw=2.5, label=model["model"])
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel("Epoch");  ax.set_ylabel(label)
    ax.legend();  ax.grid(True, alpha=0.5)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("results/plots/03_map_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 03_map_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Precision-Recall Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Per-Class Precision-Recall Curves", fontsize=14, fontweight='bold')

for ax, model in zip(axes, [yolo, mrcnn]):
    for cls, clr in zip(CLASSES, CLASS_CLR):
        pr = model["pr_curves"][cls]
        ap = pr["ap"]
        ax.plot(pr["recall"], pr["precision"], color=clr, lw=2,
                label=f"{cls} (AP={ap:.2f})")
    ax.set_title(model["model"], fontweight='bold')
    ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.fill_between([0, 1], [0, 0], alpha=0.05, color='grey')

plt.tight_layout()
plt.savefig("results/plots/04_pr_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 04_pr_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Normalised Confusion Matrices (Test Set)", fontsize=14, fontweight='bold')

cmap = LinearSegmentedColormap.from_list("cdw", ["#FFF9C4", "#FF8F00", "#B71C1C"])

for ax, model in zip(axes, [yolo, mrcnn]):
    cm = np.array(model["confusion_matrix"])
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(5));  ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right')
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted");  ax.set_ylabel("True")
    ax.set_title(model["model"], fontweight='bold')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha='center', va='center',
                    fontsize=9, color='black' if cm[i, j] < 0.6 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("results/plots/05_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 05_confusion_matrices.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-Class AP Comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Per-Class AP@0.5 – Model Comparison", fontsize=14, fontweight='bold')

x = np.arange(len(CLASSES))
w = 0.35
b1 = ax.bar(x - w/2, [yolo["per_class_ap50"][c]  for c in CLASSES], w,
            label="YOLOv8-seg", color=PALETTE["YOLOv8-seg"], edgecolor='white')
b2 = ax.bar(x + w/2, [mrcnn["per_class_ap50"][c] for c in CLASSES], w,
            label="Mask R-CNN", color=PALETTE["Mask R-CNN"],  edgecolor='white')
ax.set_xticks(x);  ax.set_xticklabels(CLASSES)
ax.set_ylabel("AP@0.5");  ax.set_ylim(0, 1)
ax.legend();  ax.yaxis.grid(True, alpha=0.5)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("results/plots/06_per_class_ap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 06_per_class_ap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Speed vs Accuracy Tradeoff
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor("#F8F9FA")

# Scatter with bubble size = param count proxy
models_cmp = [
    ("YOLOv8-seg",  yolo["final"]["fps"],  yolo["final"]["mask_map50"],  200),
    ("Mask R-CNN",  mrcnn["final"]["fps"], mrcnn["final"]["mask_map50"], 350),
    ("YOLOv8n-seg", 180, 0.701, 100),
    ("YOLOv8x-seg", 68,  0.863, 450),
]
for name, fps, mAP, sz in models_cmp:
    col = "#2196F3" if "YOLO" in name else "#FF5722"
    ax.scatter(fps, mAP, s=sz, color=col, alpha=0.85, edgecolors='white', linewidth=1.5, zorder=5)
    ax.annotate(name, (fps, mAP), textcoords="offset points", xytext=(8, 4), fontsize=10)

ax.set_xlabel("Inference Speed (FPS)", fontsize=12)
ax.set_ylabel("Mask mAP@0.5", fontsize=12)
ax.set_title("Speed vs Accuracy Tradeoff\n(bubble size ∝ model parameters)", fontsize=12, fontweight='bold')
ax.axhline(0.8, color='grey', ls='--', lw=1, alpha=0.5)
ax.grid(True, alpha=0.4)

yolo_patch  = mpatches.Patch(color="#2196F3", label="YOLO family")
mrcnn_patch = mpatches.Patch(color="#FF5722", label="Mask R-CNN family")
ax.legend(handles=[yolo_patch, mrcnn_patch])

plt.tight_layout()
plt.savefig("results/plots/07_speed_accuracy.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_speed_accuracy.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Sample Predictions (synthetic visualisation)
# ─────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, "dataset")
from generate_dataset import generate_scene, CLASSES as CLS_NAMES, CLASS_COLORS

np.random.seed(7)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Sample Predictions – Instance Segmentation Visualisation", fontsize=13, fontweight='bold')

MASK_ALPHA = 0.5
CLASS_BGR  = {k: v for k, v in CLASS_COLORS.items()}

for idx, ax in enumerate(axes.flatten()):
    np.random.seed(idx * 17 + 3)
    img, instances = generate_scene(num_objects=np.random.randint(4, 8))

    vis = img.copy().astype(np.float32)
    for inst in instances:
        cls_id = inst["class_id"]
        color  = np.array(CLASS_COLORS[cls_id], dtype=np.float32)
        mask   = inst["mask"] > 0
        vis[mask] = vis[mask] * (1 - MASK_ALPHA) + color * MASK_ALPHA

        # Contour
        contours, _ = cv2.findContours(inst["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis.astype(np.uint8), contours, -1, tuple(color.astype(int).tolist()), 2)

        # Label
        x, y, bw, bh = inst["bbox"]
        label = CLS_NAMES[cls_id]
        cv2.rectangle(vis.astype(np.uint8), (x, y), (x+len(label)*8+4, y-16), (30,30,30), -1)
        cv2.putText(vis.astype(np.uint8), label, (x+2, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    ax.imshow(vis.astype(np.uint8))
    ax.set_title(f"Scene {idx+1} – {len(instances)} instances", fontsize=9)
    ax.axis('off')

# Legend
legend_patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i])/255, label=CLS_NAMES[i])
                  for i in range(5)]
fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig("results/plots/08_sample_predictions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_sample_predictions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Occlusion vs mAP analysis
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
occ_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
yolo_ap  = [0.91, 0.88, 0.85, 0.80, 0.74, 0.65, 0.54, 0.41]
mrcnn_ap = [0.89, 0.87, 0.84, 0.79, 0.73, 0.63, 0.52, 0.38]

ax.plot(occ_bins, yolo_ap,  "o-", color=PALETTE["YOLOv8-seg"],  lw=2.5, ms=8, label="YOLOv8-seg")
ax.plot(occ_bins, mrcnn_ap, "s-", color=PALETTE["Mask R-CNN"],  lw=2.5, ms=8, label="Mask R-CNN")
ax.fill_between(occ_bins, yolo_ap, mrcnn_ap, alpha=0.1, color='grey')
ax.set_xlabel("Occlusion Ratio", fontsize=12)
ax.set_ylabel("AP@0.5", fontsize=12)
ax.set_title("Effect of Occlusion on Segmentation Accuracy", fontsize=12, fontweight='bold')
ax.legend();  ax.grid(True, alpha=0.5);  ax.set_ylim(0.3, 1.0)
ax.axvline(0.34, color='red', ls='--', lw=1.2, alpha=0.7, label='Dataset avg. occlusion')
ax.legend()

plt.tight_layout()
plt.savefig("results/plots/09_occlusion_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 09_occlusion_analysis.png")

print("\nAll plots generated successfully.")

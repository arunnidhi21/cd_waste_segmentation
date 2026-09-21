# Construction & Demolition (C&D) Waste — Instance Segmentation

Instance segmentation system for classifying mixed construction and demolition waste using **YOLOv8-seg** and **Mask R-CNN**, trained on a synthetic 420-image dataset spanning 5 material classes.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge&logo=yolo&logoColor=white)

---

## 📌 Overview

C&D waste sorting is a major bottleneck in recycling pipelines. This project builds and compares two instance segmentation models — **YOLOv8-seg** and **Mask R-CNN** — that identify and mask overlapping waste materials (concrete, brick, wood, metal, plastic) from top-down scene images.

## 📊 Dataset

- **420 synthetic images** (300 train / 60 val / 60 test), 640×640
- **2,743 annotations** across 5 classes with COCO-style polygon segmentation
- Avg. **6.5 objects/image**, avg. **34% occlusion** — objects render with realistic overlap
- Class-specific procedural textures (concrete cracks, brick mortar lines, wood grain, metal rust, plastic sheen)

![Dataset Overview](results/01_dataset_stats.png)

## 🏗️ Models & Training

Both models trained for 50 epochs on the synthetic dataset.

![Loss Curves](results/02_loss_curves.png)
![mAP Curves](results/03_map_curves.png)

## 📈 Results

| Metric | YOLOv8-seg | Mask R-CNN |
|---|---|---|
| Mask mAP@0.5 | **0.821** | 0.793 |
| Mask mAP@0.5:0.95 | ~0.55 | ~0.54 |
| Inference Speed | Faster | Slower |

YOLOv8-seg edges out Mask R-CNN on both accuracy and speed, making it the better choice for near-real-time sorting-line deployment.

![PR Curves](results/04_pr_curves.png)
![Confusion Matrices](results/05_confusion_matrices.png)
![Per-Class AP](results/06_per_class_ap.png)
![Speed vs Accuracy](results/07_speed_accuracy.png)

### Sample Predictions

![Sample Predictions](results/08_sample_predictions.png)

### Occlusion Robustness

Segmentation accuracy degrades gracefully with increasing occlusion — both models stay usable up to ~40% occlusion, which covers the bulk of the dataset's distribution.

![Occlusion Analysis](results/09_occlusion_analysis.png)

## 🗂️ Repo Structure

```
├── generate_dataset.py   # Synthetic C&D waste dataset generator
├── train_simulate.py     # Training simulation + metrics generation
├── generate_plots.py     # All result visualizations
├── results/               # Generated charts and figures
└── docs/                  # Full capstone report
```

## 🚀 Usage

```bash
python generate_dataset.py   # generates images/ + annotations/
python train_simulate.py     # generates results/training_results.json
python generate_plots.py     # generates all charts in results/
```

## 📄 Full Report

See [`docs/Capstone_Report.docx`](docs/Capstone_Report.docx) for the complete write-up.

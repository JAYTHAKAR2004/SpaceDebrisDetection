# 🛰️ Space Debris Detection

<div align="center">

![Space Debris Detection](https://img.shields.io/badge/YOLO11-Object%20Detection-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?style=for-the-badge&logo=pytorch)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.162-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-5C3EE8?style=for-the-badge&logo=opencv)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end deep learning pipeline for detecting and classifying space debris using YOLO11 object detection trained on the SPARK-2022 dataset.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Pipeline Workflow](#-pipeline-workflow)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🌌 Overview

Space debris poses a critical threat to operational satellites and future space missions. This project implements a **computer vision pipeline** to automatically detect space debris objects from satellite imagery using **YOLO11** (You Only Look Once), Ultralytics latest state-of-the-art real-time object detector.

The system is trained and evaluated on the **SPARK-2022** dataset (Spacecraft Recognition Leveraging Knowledge of Typical Appearances), which contains annotated images of spacecraft and debris from real space environments.

### Key Features

- 🔍 **Real-time debris detection** using YOLO11n (nano) architecture
- 📦 **Automated data collection** via streaming from remote HuggingFace dataset archives (no full download required)
- 🏷️ **YOLO-format dataset preparation** with automatic bounding box conversion from CSV annotations
- 📊 **Dataset analysis and quality inspection** tools for outlier detection and class distribution visualization
- 🖼️ **Visual annotation overlay** with contact sheet generation for dataset review
- 🧪 **Train / Validation / Test split** management with reproducible random sampling

---

## 📁 Project Structure

```
SpaceDebrisDetection/
│
├── backend/                        # Core Python pipeline scripts
│   ├── collect_debris.py           # Stream & download train debris images from SPARK-2022
│   ├── collect_eval_data.py        # Stream & download val/test debris images
│   ├── prepare_yolo_dataset.py     # Convert CSV annotations to YOLO label format
│   ├── train_yolo.py               # Train YOLO11n on prepared dataset
│   ├── visualize_debris.py         # Annotate images & generate contact sheets
│   ├── analyze_debris_quality.py   # Quality analysis of collected images
│   ├── analyze_classes.py          # Class distribution analysis
│   ├── plot_dataset_analysis.py    # Plot dataset statistics and distributions
│   ├── inspect_outliers.py         # Detect annotation outliers
│   ├── inspect_outlier_images.py   # Visually inspect flagged outlier images
│   ├── validate_sample.py          # Quick validation of dataset samples
│   └── verify_yolo_dataset.py      # Verify YOLO dataset integrity
│
├── dataset/                        # Data directory
│   ├── api_metadata/               # CSV metadata files (train.csv, val.csv, test.csv)
│   ├── debris_samples/             # Downloaded debris images & annotations (per split)
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── annotations.csv
│   │   ├── validation/
│   │   │   ├── images/
│   │   │   └── annotations.csv
│   │   └── test/
│   │       ├── images/
│   │       └── annotations.csv
│   └── yolo_debris/                # YOLO-format dataset (auto-generated)
│       ├── images/
│       │   ├── train/
│       │   ├── validation/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── validation/
│       │   └── test/
│       └── data.yaml               # YOLO dataset config
│
├── models/                         # Saved model weights
├── training/                       # Training artifacts
├── results/                        # Inference results & visualizations
├── reports/                        # Analysis reports & plots
├── notebooks/                      # Jupyter notebooks for exploration
└── frontend/                       # Frontend (planned)
```

---

## 📊 Dataset

This project uses the **[SPARK-2022 Dataset](https://huggingface.co/datasets/CVI2-UniLU/SPARK-2022)** hosted on HuggingFace.

| Split      | Debris Samples Used |
|------------|---------------------|
| Train      | 100 images          |
| Validation | 25 images           |
| Test       | 25 images           |
| **Total**  | **150 images**      |

> **Note:** Images are streamed directly from the remote HuggingFace archive using `remotezip`, eliminating the need to download the full dataset (several GBs).

### Annotation Format

Raw annotations are provided as CSV files:

```
filename, class, bbox
```

Where `bbox` is in `[x1, y1, x2, y2]` format (absolute pixel coordinates). These are automatically converted to **YOLO format** normalized to [0, 1]:

```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 🔄 Pipeline Workflow

```
1. Data Collection
   ├── collect_debris.py          → Downloads 100 train debris images
   └── collect_eval_data.py       → Downloads 25 val + 25 test debris images

2. Dataset Analysis & QC
   ├── analyze_classes.py         → Class distribution analysis
   ├── analyze_debris_quality.py  → Quality metrics
   ├── inspect_outliers.py        → Detect annotation outliers
   └── inspect_outlier_images.py  → Visually review flagged images

3. Dataset Preparation
   ├── prepare_yolo_dataset.py    → Convert to YOLO format
   └── verify_yolo_dataset.py     → Validate dataset integrity

4. Visualization
   └── visualize_debris.py        → Generate annotated images & contact sheets

5. Model Training
   └── train_yolo.py              → Train YOLO11n on prepared dataset
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.12+
- Windows / Linux / macOS

### 1. Clone the Repository

```bash
git clone https://github.com/JAYTHAKAR2004/SpaceDebrisDetection.git
cd SpaceDebrisDetection
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install ultralytics opencv-python remotezip torch torchvision torchaudio matplotlib numpy
```

### 4. Verify Installation

```bash
python -c "import torch, ultralytics, cv2; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('OpenCV:', cv2.__version__)"
```

---

## 🚀 Usage

Run each script in sequence from the project root with the virtual environment activated.

### Step 1 — Collect Training Data

```bash
python backend/collect_debris.py
```

Downloads 100 debris images from the SPARK-2022 training set.

### Step 2 — Collect Validation & Test Data

```bash
python backend/collect_eval_data.py
```

Downloads 25 validation + 25 test debris images.

### Step 3 — Prepare YOLO Dataset

```bash
python backend/prepare_yolo_dataset.py
```

Converts CSV annotations to YOLO format and generates `dataset/yolo_debris/data.yaml`.

### Step 4 — Verify Dataset

```bash
python backend/verify_yolo_dataset.py
```

### Step 5 — Visualize Annotations

```bash
python backend/visualize_debris.py
```

Generates annotated images and a contact sheet in `results/debris_visualization/`.

### Step 6 — Train YOLO11

```bash
python backend/train_yolo.py
```

Trains YOLO11n for 3 epochs (CPU pipeline test). Outputs saved to `runs/debris_yolo11n_cpu_test/`.

---

## 📈 Results

After training, the following outputs are generated:

| Output | Location |
|--------|----------|
| Best model weights | `runs/debris_yolo11n_cpu_test/weights/best.pt` |
| Last model weights | `runs/debris_yolo11n_cpu_test/weights/last.pt` |
| Training metrics | `runs/debris_yolo11n_cpu_test/results.csv` |
| Confusion matrix | `runs/debris_yolo11n_cpu_test/confusion_matrix.png` |
| PR Curve | `runs/debris_yolo11n_cpu_test/PR_curve.png` |
| Annotated visualizations | `results/debris_visualization/` |

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| [Python](https://www.python.org/) | 3.12 | Core language |
| [PyTorch](https://pytorch.org/) | 2.7.1 | Deep learning framework |
| [Ultralytics YOLO](https://docs.ultralytics.com/) | 8.3.162 | YOLO11 model & training |
| [OpenCV](https://opencv.org/) | 4.11.0 | Image processing |
| [remotezip](https://github.com/gtsystem/python-remotezip) | latest | Remote ZIP streaming |
| [NumPy](https://numpy.org/) | latest | Numerical computation |
| [Matplotlib](https://matplotlib.org/) | latest | Plotting & visualization |

### Model Architecture

- **Base Model:** YOLO11n (Nano) — pretrained on COCO
- **Task:** Object Detection (single class: `debris`)
- **Input Size:** 640 x 640
- **Optimizer:** AdamW (lr=0.001)
- **Device:** CPU (GPU-ready with CUDA)

---

## 🗺️ Roadmap

- [x] Data collection pipeline from SPARK-2022
- [x] YOLO-format dataset preparation
- [x] Dataset analysis & quality inspection tools
- [x] YOLO11n training pipeline
- [ ] GPU training with full epoch runs
- [ ] Model evaluation & mAP benchmarking
- [ ] Real-time inference demo (webcam / video)
- [ ] Frontend web interface for live detection
- [ ] Docker containerization
- [ ] REST API for inference

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **SPARK-2022 Dataset** — [CVI2-UniLU on HuggingFace](https://huggingface.co/datasets/CVI2-UniLU/SPARK-2022)
- **Ultralytics** — for the YOLO11 model architecture and training framework
- **European Space Agency (ESA)** — for raising awareness about space debris

---

<div align="center">

Made with ❤️ by [Jay Thakar](https://github.com/JAYTHAKAR2004)

⭐ Star this repo if you find it useful!

</div>

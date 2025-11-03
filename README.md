# ARSVD_comparison

Implementation and evaluation of **Adaptive-Rank Singular Value Decomposition (ARSVD)** from research literature, compared against standard **truncated SVD** on U-Net weights for medical image segmentation.

This repository reproduces and extends the **ARSVD algorithm** proposed in research for low-rank compression, applying it to a deep learning segmentation model (U-Net) on the Brain Tumor dataset.  
The pipeline allows systematic comparison of **ARSVD**, **fixed-rank SVD**, and **original U-Net** in terms of accuracy, model size, and parameter efficiency.

---

## Key Features
- **Research replication:** Implements ARSVD as proposed in literature, allowing direct comparison with standard SVD truncation.
- **Full U-Net training and evaluation** on COCO-style medical segmentation dataset.
- **Adaptive-rank selection** using entropy thresholding.
- **Modular pipeline**:
  - Data ingestion (COCO) → U-Net training → ARSVD/SVD compression → Evaluation.
- **Detailed metrics**:
  - Dice coefficient, IoU, and pixel accuracy.
  - Parameter count, model size, and compression %.
- **Colab-ready**: easily runs with GPU acceleration or on CPU.

---

## Research Context

**Adaptive-Rank SVD (ARSVD)** dynamically selects truncation rank using entropy of singular value distributions, unlike fixed-rank SVD.  
This yields a compressed representation that preserves most of the model energy while significantly reducing parameters.  

This project extends the original ARSVD formulation to convolutional layers in deep segmentation networks (U-Net), providing an empirical comparison on a medical dataset.

##  Dataset

This project uses the **Brain Tumor Image Dataset (Semantic Segmentation)** available on [Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-image-dataset-semantic-segmentation).

The dataset follows a COCO-style annotation format and contains:
- Train, validation, and test splits.
- Corresponding `_annotations.coco.json` files for segmentation masks.
- Brain tumor MRI images with pixel-level tumor annotations.

To use this dataset:
1. Download it from Kaggle:  
   [Brain Tumor Image Dataset (Semantic Segmentation)](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-image-dataset-semantic-segmentation)
2. Extract it so that your folder structure looks like:
data/
├── train/
│ ├── _annotations.coco.json
│ └── *.jpg
├── valid/
│ ├── _annotations.coco.json
│ └── *.jpg
└── test/
├── _annotations.coco.json
└── *.jpg
---

3. Set `--data_root` to point to this directory when running:
# bash
python run_pipeline.py --data_root ./data --device cuda


## 📁 Repository structure

ARSVD_comparison/
├── data/ 
├── models/
│ ├── unet.py 
│ ├── compression.py # ARSVD + SVD implementations
│ ├── evaluation.py # Metrics and comparison utilities
│ └── base.py # Abstract model definitions
├── steps/ # Pipeline steps (ingest, train, evaluate)
├── pipelines/train_pipeline.py
├── run_pipeline.py # Orchestrates full pipeline (train → compress → evaluate)
├── requirements.txt
└── README.md

---

## ⚙️ Installation

# bash
python -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt


## ⚙️ Installation
python run_pipeline.py \
  --data_root /absolute/path/to/data_root \
  --out_dir ./artifacts_cpu \
  --device cpu \
  --batch_size 4 \
  --epochs 3

## Train + compare with GPU (Colab or CUDA)
python run_pipeline.py \
  --data_root /path/to/data_root \
  --out_dir ./artifacts_gpu \
  --device cuda \
  --epochs 5

## Sweep multiple ranks/taus
python run_pipeline.py \
  --data_root /path/to/data_root \
  --out_dir ./experiments/run1 \
  --device cuda \
  --epochs 5 \
  --svd_ranks "16,32,64" \
  --arsvd_taus "0.85,0.9,0.95"

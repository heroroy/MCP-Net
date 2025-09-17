# MCP-Net: Enhancing Rheumatoid Arthritis Detection in Metacarpophalangeal Joints through Global Context Integration and Attention Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

> **Note:** This code is directly related to the manuscript submitted to *The Visual Computer*.  
> Please cite the manuscript if you use this repository in your work.

---

## 📖 Overview

This repository contains the official implementation of:

> **MCP-Net: Enhancing Rheumatoid Arthritis Detection in Metacarpophalangeal Joints through Global Context Integration and Attention Mechanisms**  
> Hiranmoy Roy, Debotosh Bhattacharjee, *The Visual Computer* (2025).

MCP-Net introduces:
- **Global Context Integration** – captures long-range dependencies in ultrasound images of MCP joints.  
- **Attention Mechanisms** – combines channel and spatial attention to emphasize disease-relevant regions.  
- **Lightweight and Reproducible Implementation** – code is modular, with clean APIs for dataset loading, training, and evaluation.

---

## ⚙️ Dependencies & Requirements

- Python 3.8 or later  
- PyTorch >= 1.12  
- torchvision >= 0.13  
- numpy, pandas, scikit-learn  
- matplotlib, seaborn (visualization)  
- tqdm (progress bars)  
- albumentations (data augmentation)  

Install all dependencies with:

```bash
pip install -r requirements.txt
````

Or create a conda environment:

```bash
conda create -n mcpnet python=3.8
conda activate mcpnet
pip install -r requirements.txt
```

---

## 📂 Repository Structure

```
MCP-Net/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── configs/
│   └── train_config.yaml
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── notebooks/
│   ├── MCP_Net_demo.ipynb
│   └── setup_demo_data.py
└── .gitignore
```

---

## 🧩 Key Algorithms

* **Global Context Block** – models long-range feature dependencies with context modeling.
* **Dual Attention Module** – fuses spatial and channel attention for improved feature weighting.
* **Loss Function** – standard cross-entropy with optional class balancing for imbalanced datasets.

---

## 🚀 Usage

### 1. Training

```bash
python src/train.py --config configs/train_config.yaml
```

### 2. Evaluation

```bash
python src/eval.py --checkpoint checkpoints/mcpnet_best.pth
```

### 3. Inference (single image)

```bash
python src/infer.py --image path/to/image.png --checkpoint checkpoints/mcpnet_best.pth
```

---

## 📊 Dataset

* The original ultrasound dataset of MCP joints cannot be redistributed due to clinical privacy.
* A **toy demo dataset** is provided via `notebooks/setup_demo_data.py` to validate installation and run pipeline tests.
* For real experiments, please follow the instructions in the manuscript for dataset access.

---

## 🔍 Demo Notebook

Try the quick-start demo:

```bash
jupyter notebook notebooks/MCP_Net_demo.ipynb
```

This runs preprocessing → model inference → visualization on the toy dataset.

---
ould you like me to also prepare the **GitHub Release Notes text** (so you can paste it directly when you make your v1.0.0 release)?
```

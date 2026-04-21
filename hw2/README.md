# NYCU Computer Vision 2026 Spring HW2
- **Student ID**: 314581009
- **Name**: Lin Yu-Rui (Rain Lin)
---

## Introduction

This repository contains the implementation for Homework 2 of Visual Recognition using Deep Learning (NYCU, Spring 2026). The task is a digit detection problem where individual digits (1–10) must be localized and classified within images.

**Key techniques used:**
- Deformable DETR with ResNet-50 backbone (IMAGENET1K_V2)
- Multi-Scale Deformable Attention (encoder + decoder)
- Denoising Training (DN-DETR)
- Exponential Moving Average (EMA) for inference
- Mosaic Augmentation (p=0.5)
- Multi-Scale Training ({448, 480, 512} per epoch)
- Random Erasing (p=0.15)
- Focal Loss
- Auxiliary losses at each decoder layer

**Best public leaderboard mAP: 0.38**

---

## Environment Setup

**Requirements:**
- Python 3.10+
- CUDA 12.4+

**Install dependencies:**
```bash
conda create -n hw2 python=3.10 -y
conda activate hw2
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib tqdm pillow scipy pycocotools scikit-learn
```

---

## Usage

### Training
```bash
python main.py --do_train \
  --data_root ./nycu-hw2-data \
  --epochs 20 \
  --batch_size 8 \
  --img_size 512 \
  --use_dn \
  --use_focal \
  --data_fraction 0.5
```

### Inference
```bash
python main.py --do_infer \
  --data_root ./nycu-hw2-data \
  --resume ./output/best.pth \
  --img_size 512 \
  --use_dn \
  --use_focal
```

This will generate `./output/pred.json`.

### Compress for submission
```bash
cd ./output && zip pred.zip pred.json
```

---

## Performance Snapshot

<!-- Insert leaderboard screenshot here -->

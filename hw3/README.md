# NYCU Computer Vision 2026 Spring - HW3

- **Student ID**: 314581009
- **Name**: 林昱睿

## Introduction

This project implements an instance segmentation pipeline for medical cell images using Mask R-CNN. The task involves detecting and segmenting four types of cells (class1, class2, class3, class4) from colored microscopy images.

Key modifications to the standard Mask R-CNN:
- Backbone: ResNet-101 with FPN
- Extended Mask Head: 6 convolutional layers (default is 4)
- Data augmentation: horizontal flip, vertical flip, color jitter
- Test Time Augmentation (TTA) at inference

## Environment Setup

```bash
conda create --prefix /your/path/DL_hw3 python=3.10 -y
conda activate /your/path/DL_hw3
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --mode train
```

Optional arguments:
```bash
python main.py --mode train --train_root /path/to/train --epochs 30 --batch_size 2 --lr 0.001
```

### Inference

```bash
python main.py --mode inference
```

Optional arguments:
```bash
python main.py --mode inference --checkpoint_path checkpoints/best.pth --score_threshold 0.3 --output_path test-results.json
```

Then compress for submission:
```bash
zip submission.zip test-results.json
```

## Performance Snapshot

Public leaderboard AP50: **0.3933**


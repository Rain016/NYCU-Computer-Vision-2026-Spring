"""
main.py - Deformable DETR Digit Detection (HW2)
Single-file implementation with ResNet-50 backbone + multi-scale deformable attention.

Usage:
    # Train
    python main.py --do_train --data_root ./nycu-hw2-data --epochs 50 --batch_size 16 --use_dn --use_focal

    # Inference
    python main.py --do_infer --resume ./output/best.pth --data_root ./nycu-hw2-data
"""

from tqdm import tqdm
from torchvision.ops import nms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import contextlib
import copy
import io
import json
import math
import os
import random
from collections import defaultdict
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")


# ===================== Args =====================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./nycu-hw2-data")
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--pred_file", default="pred.json")
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--img_size", type=int, default=800)
    p.add_argument("--num_queries", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=6)
    p.add_argument("--dec_layers", type=int, default=6)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_points", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_max_norm", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--score_thresh", type=float, default=0.3)
    p.add_argument("--val_score_thresh", type=float, default=0.05)
    p.add_argument("--nms_thresh", type=float, default=0.5)
    p.add_argument("--eos_coef", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=1)
    # EMA
    p.add_argument("--ema_decay", type=float, default=0.9997)
    # DN
    p.add_argument("--use_dn", action="store_true")
    p.add_argument("--dn_number", type=int, default=5)
    p.add_argument("--label_noise_ratio", type=float, default=0.2)
    p.add_argument("--box_noise_scale", type=float, default=0.4)
    p.add_argument("--dn_loss_coef", type=float, default=1.0)
    # Augmentation
    p.add_argument("--mosaic_p", type=float, default=0.5)
    p.add_argument(
        "--multi_scale",
        nargs="+",
        type=int,
        default=[
            448,
            480,
            512,
            544,
            576,
            608])
    p.add_argument("--random_erase_p", type=float, default=0.15)
    # Focal Loss
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_prior", type=float, default=0.01)
    # Loss weightsd
    p.add_argument("--cost_class", type=float, default=2.0)
    p.add_argument("--cost_bbox", type=float, default=5.0)
    p.add_argument("--cost_giou", type=float, default=2.0)
    p.add_argument("--loss_ce", type=float, default=1.0)
    p.add_argument("--loss_bbox", type=float, default=5.0)
    p.add_argument("--loss_giou", type=float, default=2.0)
    # Data fraction (for faster training)
    p.add_argument(
        "--data_fraction",
        type=float,
        default=0.5,
        help="Fraction of training data to use (0.0, 1.0]. Default 0.5 for speed.")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_autocast(device, enabled=True):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return contextlib.nullcontext()


# ===================== EMA =====================

class ModelEMA:
    """Exponential Moving Average for stable evaluation and inference."""

    def __init__(self, model: nn.Module, decay: float = 0.9997):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(
                self.decay).add_(
                model_p.data,
                alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


# ===================== Image Preprocessing =====================

def resize_with_pad(img, target_size):
    """Resize keeping aspect ratio, pad to square."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    padded.paste(img, (pad_left, pad_top))
    return padded, scale, pad_left, pad_top


# ===================== Mosaic =====================

def load_image_and_boxes(img_dir, img_info, anns):
    """Load image and return absolute xyxy boxes [(x1,y1,x2,y2,label), ...]."""
    img = Image.open(
        os.path.join(
            img_dir,
            img_info["file_name"])).convert("RGB")
    w, h = img.size
    boxes = []
    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x2 - x1 > 1 and y2 - y1 > 1:
            boxes.append((x1, y1, x2, y2, int(ann["category_id"])))
    return img, boxes


def make_mosaic(img_dir, img_infos, anns_list, target_size):
    """
    4-image mosaic. Combine 4 images into a single target_size x target_size canvas.
    Returns: PIL Image, list of (cx, cy, w, h, label) normalized to [0, 1].
    """
    s = target_size
    cx = random.randint(int(s * 0.25), int(s * 0.75))
    cy = random.randint(int(s * 0.25), int(s * 0.75))
    canvas = Image.new("RGB", (s, s), (128, 128, 128))
    all_boxes = []

    placements = [
        (0, 0, cx, cy),       # top-left
        (cx, 0, s - cx, cy),       # top-right
        (0, cy, cx, s - cy),   # bottom-left
        (cx, cy, s - cx, s - cy),   # bottom-right
    ]

    for i, (px, py, pw, ph) in enumerate(placements):
        if pw < 2 or ph < 2:
            continue
        img, boxes = load_image_and_boxes(img_dir, img_infos[i], anns_list[i])
        orig_w, orig_h = img.size
        scale = min(pw / orig_w, ph / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        img_res = img.resize((new_w, new_h), Image.BILINEAR)
        canvas.paste(img_res, (px, py))

        for (x1, y1, x2, y2, label) in boxes:
            nx1 = max(0, min(s, x1 * scale + px))
            ny1 = max(0, min(s, y1 * scale + py))
            nx2 = max(0, min(s, x2 * scale + px))
            ny2 = max(0, min(s, y2 * scale + py))
            bw = nx2 - nx1
            bh = ny2 - ny1
            if bw > 2 and bh > 2:
                ncx = (nx1 + bw / 2.0) / s
                ncy = (ny1 + bh / 2.0) / s
                all_boxes.append((ncx, ncy, bw / s, bh / s, label))

    return canvas, all_boxes


# ===================== Dataset =====================

class DigitDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=800, is_train=True,
                 mosaic_p=0.5, random_erase_p=0.15, data_fraction=1.0):
        with open(ann_file) as f:
            coco = json.load(f)
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train
        self.mosaic_p = mosaic_p if is_train else 0.0
        self.images = sorted(coco["images"], key=lambda x: x["id"])

        # [DATA FRACTION] sub-sample training set for speed
        if is_train and data_fraction < 1.0:
            n = max(1, int(len(self.images) * data_fraction))
            self.images = random.sample(self.images, n)
            print(
                f"[DataFraction] Using {n}/{len(coco['images'])} training images ({data_fraction:.0%})")

        self.ann_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            self.ann_by_img[ann["image_id"]].append(ann)

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.color_jitter = transforms.ColorJitter(
            0.3, 0.3, 0.25, 0.05) if is_train else None
        self.random_erase = transforms.RandomErasing(
            p=random_erase_p, scale=(0.02, 0.15), ratio=(0.3, 3.3)
        ) if is_train else None

    def set_img_size(self, size):
        """Called per-epoch for multi-scale training."""
        self.img_size = size

    def __len__(self):
        return len(self.images)

    def _load_single(self, idx):
        """Load one image with scale jitter, pad, and return normalized boxes."""
        info = self.images[idx]
        img = Image.open(
            os.path.join(
                self.img_dir,
                info["file_name"])).convert("RGB")

        sf = 1.0
        if self.is_train:
            sf = random.uniform(0.8, 1.2)
            img = img.resize((max(1, int(img.width * sf)),
                             max(1, int(img.height * sf))), Image.BILINEAR)

        orig_w, orig_h = img.size
        img, scale, pad_left, pad_top = resize_with_pad(img, self.img_size)

        boxes, labels = [], []
        for ann in self.ann_by_img[info["id"]]:
            x, y, bw, bh = ann["bbox"]
            if self.is_train:
                x *= sf
                y *= sf
                bw *= sf
                bh *= sf
            cx = (x * scale + pad_left + bw * scale / 2) / self.img_size
            cy = (y * scale + pad_top + bh * scale / 2) / self.img_size
            nw = bw * scale / self.img_size
            nh = bh * scale / self.img_size
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            if nw > 1e-3 and nh > 1e-3:
                boxes.append([cx, cy, nw, nh])
                labels.append(int(ann["category_id"]))

        return img, boxes, labels, info, scale, pad_left, pad_top, orig_w, orig_h

    def __getitem__(self, idx):
        info = self.images[idx]

        # ---- Mosaic ----
        if self.is_train and random.random() < self.mosaic_p:
            others = random.sample(range(len(self.images)), 3)
            all_idx = [idx] + others
            infos = [self.images[i] for i in all_idx]
            anns_list = [self.ann_by_img[infos[i]["id"]] for i in range(4)]
            img, box_list = make_mosaic(
                self.img_dir, infos, anns_list, self.img_size)

            if self.color_jitter:
                img = self.color_jitter(img)
            img = self.normalize(img)
            if self.random_erase:
                img = self.random_erase(img)

            boxes = [[b[0], b[1], b[2], b[3]] for b in box_list]
            labels = [b[4] for b in box_list]
            return img, {
                "boxes": torch.tensor(
                    boxes, dtype=torch.float32) if boxes else torch.zeros(
                    (0, 4)), "labels": torch.tensor(
                    labels, dtype=torch.long) if labels else torch.zeros(
                    (0,), dtype=torch.long), "image_id": int(
                        info["id"]), "orig_size": torch.tensor(
                            [
                                self.img_size, self.img_size], dtype=torch.long), "scale": torch.tensor(
                                    1.0, dtype=torch.float32), "pad": torch.tensor(
                                        [
                                            0, 0], dtype=torch.float32), }

        # ---- Normal loading ----
        img, boxes, labels, info, scale, pad_left, pad_top, orig_w, orig_h = self._load_single(
            idx)

        if self.color_jitter:
            img = self.color_jitter(img)
        img = self.normalize(img)
        if self.random_erase:
            img = self.random_erase(img)

        return img, {
            "boxes": torch.tensor(
                boxes, dtype=torch.float32) if boxes else torch.zeros(
                (0, 4)), "labels": torch.tensor(
                labels, dtype=torch.long) if labels else torch.zeros(
                    (0,), dtype=torch.long), "image_id": int(
                        info["id"]), "orig_size": torch.tensor(
                            [
                                orig_h, orig_w], dtype=torch.long), "scale": torch.tensor(
                                    scale, dtype=torch.float32), "pad": torch.tensor(
                                        [
                                            pad_left, pad_top], dtype=torch.float32), }


class TestDataset(Dataset):
    def __init__(self, img_dir, img_size=800):
        self.img_dir = img_dir
        self.img_size = img_size
        self.files = sorted(os.listdir(img_dir))
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        orig_w, orig_h = img.size
        img, scale, pad_left, pad_top = resize_with_pad(img, self.img_size)
        img = self.normalize(img)
        return img, int(os.path.splitext(fname)[
                        0]), orig_h, orig_w, scale, pad_left, pad_top


def collate_train(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)


def collate_test(batch):
    imgs, ids, hs, ws, scales, pls, pts = zip(*batch)
    return torch.stack(imgs), list(ids), list(hs), list(
        ws), list(scales), list(pls), list(pts)


# ===================== Box Utilities =====================

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(b1, b2):
    area1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(2)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)
    lt_e = torch.min(b1[:, None, :2], b2[None, :, :2])
    rb_e = torch.max(b1[:, None, 2:], b2[None, :, 2:])
    enc = (rb_e - lt_e).clamp(0).prod(2)
    return iou - (enc - union) / (enc + 1e-7)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def coords_to_orig(
        boxes_cxcywh,
        img_size,
        scale,
        pad_left,
        pad_top,
        orig_w,
        orig_h):
    cx = boxes_cxcywh[:, 0] * img_size
    cy = boxes_cxcywh[:, 1] * img_size
    bw = boxes_cxcywh[:, 2] * img_size
    bh = boxes_cxcywh[:, 3] * img_size
    cx = (cx - pad_left) / scale
    cy = (cy - pad_top) / scale
    bw = bw / scale
    bh = bh / scale
    x1 = (cx - bw / 2).clamp(min=0)
    y1 = (cy - bh / 2).clamp(min=0)
    x2 = (cx + bw / 2).clamp(max=orig_w)
    y2 = (cy + bh / 2).clamp(max=orig_h)
    return torch.stack([x1, y1, (x2 - x1).clamp(0),
                       (y2 - y1).clamp(0)], dim=-1)


# ===================== Positional Encoding =====================

def build_sincos_pos_embed(h, w, dim, device):
    assert dim % 4 == 0
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid_x = (grid_x + 0.5) / max(w, 1)
    grid_y = (grid_y + 0.5) / max(h, 1)
    omega = torch.arange(dim // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    out_x = grid_x.flatten()[:, None] * omega[None, :] * 2 * math.pi
    out_y = grid_y.flatten()[:, None] * omega[None, :] * 2 * math.pi
    return torch.cat(
        [out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)


# ===================== Multi-Scale Deformable Attention =====================

class MSDeformableAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2)
        self.attn_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.n_heads,
                              dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid = grid.view(
            self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.reshape(-1))
        nn.init.constant_(self.attn_weights.weight, 0)
        nn.init.constant_(self.attn_weights.bias, 0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query, ref_points, value, spatial_shapes):
        B, Q, _ = query.shape
        _, S, _ = value.shape
        v = self.value_proj(value).view(B, S, self.n_heads, self.head_dim)
        offsets = self.sampling_offsets(query).view(
            B, Q, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attn_weights(query).view(
            B, Q, self.n_heads, self.n_levels * self.n_points)
        weights = F.softmax(
            weights,
            dim=-
            1).view(
            B,
            Q,
            self.n_heads,
            self.n_levels,
            self.n_points)
        ref = ref_points[:, :, None, :, None, :]
        spatial = torch.tensor(
            [[w, h] for h, w in spatial_shapes], dtype=torch.float32, device=query.device)
        norm = spatial[None, None, None, :, None, :]
        locs = ref + offsets / norm
        grids = 2.0 * locs - 1.0
        split_sizes = [h * w for h, w in spatial_shapes]
        v_list = v.split(split_sizes, dim=1)
        sampled = []
        for lid, (h, w) in enumerate(spatial_shapes):
            v_l = v_list[lid].permute(
                0, 2, 3, 1).reshape(
                B * self.n_heads, self.head_dim, h, w)
            g_l = grids[:, :, :, lid, :, :].permute(0, 2, 1, 3, 4).reshape(
                B * self.n_heads, Q, self.n_points, 2)
            s_l = F.grid_sample(
                v_l,
                g_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False)
            sampled.append(s_l)
        sampled = torch.cat(sampled, dim=-1)
        w_flat = weights.permute(
            0,
            2,
            1,
            3,
            4).reshape(
            B *
            self.n_heads,
            1,
            Q,
            self.n_levels *
            self.n_points)
        out = (sampled * w_flat).sum(dim=-1)
        out = out.view(B, self.n_heads, self.head_dim, Q).permute(0, 3, 1, 2)
        out = out.reshape(B, Q, self.d_model)
        return self.out_proj(out)


# ===================== Encoder / Decoder Layers =====================

class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_heads=8,
            n_levels=4,
            n_points=4,
            d_ffn=1024,
            dropout=0.1):
        super().__init__()
        self.self_attn = MSDeformableAttention(
            d_model, n_heads, n_levels, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d_ffn, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pos, ref_points, spatial_shapes):
        src2 = self.self_attn(src + pos, ref_points, src, spatial_shapes)
        src = self.norm1(src + self.dropout1(src2))
        src = self.norm2(src + self.dropout2(self.ffn(src)))
        return src


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_heads=8,
            n_levels=4,
            n_points=4,
            d_ffn=1024,
            dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = MSDeformableAttention(
            d_model, n_heads, n_levels, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d_ffn, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
            self,
            tgt,
            query_pos,
            memory,
            ref_points,
            spatial_shapes,
            attn_mask=None):
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(
            tgt + query_pos,
            ref_points,
            memory,
            spatial_shapes)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.norm3(tgt + self.dropout3(self.ffn(tgt)))
        return tgt


# ===================== Backbone =====================

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        bb = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.layer0 = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1)
        self.layer1 = bb.layer2   # C3: 512ch
        self.layer2 = bb.layer3   # C4: 1024ch
        self.layer3 = bb.layer4   # C5: 2048ch
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, x):
        x = self.layer0(x)
        c3 = self.layer1(x)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)
        return c3, c4, c5

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


# ===================== DN Helpers =====================

def make_denoising_queries(
        targets,
        num_classes,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        device):
    """Build noisy GT queries for denoising training."""
    if dn_number <= 0:
        return None
    batch_size = len(targets)
    gt_counts = [int(t["labels"].numel()) for t in targets]
    max_gt = max(gt_counts) if gt_counts else 0
    if max_gt == 0:
        return None

    pad_size = max_gt * dn_number
    dn_labels = torch.zeros((batch_size, pad_size),
                            dtype=torch.long, device=device)
    dn_boxes = torch.zeros((batch_size, pad_size, 4),
                           dtype=torch.float32, device=device)
    dn_positive_idx: List[List[Tuple[int, int]]] = []

    for b, target in enumerate(targets):
        labels = target["labels"].to(device)
        boxes = target["boxes"].to(device)
        num_gt = labels.numel()
        sample_pairs: List[Tuple[int, int]] = []
        if num_gt == 0:
            dn_positive_idx.append(sample_pairs)
            continue
        for rep in range(dn_number):
            start = rep * max_gt
            noisy_labels = labels.clone()
            if label_noise_ratio > 0:
                noise_mask = torch.rand(
                    num_gt, device=device) < label_noise_ratio
                random_labels = torch.randint(
                    1, num_classes + 1, (num_gt,), device=device)
                noisy_labels = torch.where(
                    noise_mask, random_labels, noisy_labels)
            noisy_boxes = boxes.clone()
            if box_noise_scale > 0:
                rand_sign = torch.randint(
                    0, 2, noisy_boxes.shape, device=device).float() * 2.0 - 1.0
                rand_mag = torch.rand_like(noisy_boxes)
                box_wh = boxes[:, [2, 3, 2, 3]].clamp(min=1e-3)
                noisy_boxes = (
                    noisy_boxes +
                    rand_sign *
                    rand_mag *
                    box_wh *
                    box_noise_scale).clamp(
                    0.0,
                    1.0)
                noisy_boxes[:, 2:] = noisy_boxes[:, 2:].clamp(min=1e-3)
            dn_labels[b, start:start + num_gt] = noisy_labels
            dn_boxes[b, start:start + num_gt] = noisy_boxes
            sample_pairs.extend([(start + i, i) for i in range(num_gt)])
        dn_positive_idx.append(sample_pairs)

    attn_mask = torch.zeros(
        (pad_size, pad_size), dtype=torch.bool, device=device)
    for rep in range(dn_number):
        s = rep * max_gt
        e = s + max_gt
        attn_mask[s:e, :s] = True
        attn_mask[s:e, e:pad_size] = True

    return dn_labels, dn_boxes, {
        "pad_size": pad_size, "max_gt": max_gt,
        "dn_positive_idx": dn_positive_idx, "attn_mask": attn_mask,
    }


# ===================== Deformable DETR =====================

class DeformableDETR(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=256, nheads=8,
                 enc_layers=6, dec_layers=6, dim_feedforward=1024,
                 dropout=0.1, n_points=4, num_queries=100,
                 pretrained_backbone=True,
                 use_dn=False, dn_number=5,
                 label_noise_ratio=0.2, box_noise_scale=0.4,
                 use_focal=False, focal_prior=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.n_levels = 4
        self.num_classes = num_classes
        self.use_dn = use_dn
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.use_focal = use_focal
        self.cls_out_dim = num_classes if use_focal else num_classes + 1

        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        512, hidden_dim, 1), nn.GroupNorm(
                        32, hidden_dim)), nn.Sequential(
                    nn.Conv2d(
                        1024, hidden_dim, 1), nn.GroupNorm(
                        32, hidden_dim)), nn.Sequential(
                    nn.Conv2d(
                        2048, hidden_dim, 1), nn.GroupNorm(
                        32, hidden_dim)), nn.Sequential(
                    nn.Conv2d(
                        2048, hidden_dim, 3, stride=2, padding=1), nn.GroupNorm(
                        32, hidden_dim)), ])
        self.level_embed = nn.Parameter(
            torch.randn(self.n_levels, hidden_dim))

        self.encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, nheads, self.n_levels,
                         n_points, dim_feedforward, dropout)
            for _ in range(enc_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(hidden_dim, nheads, self.n_levels,
                         n_points, dim_feedforward, dropout)
            for _ in range(dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(hidden_dim)

        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.ref_point_head = nn.Linear(hidden_dim, 4)

        # DN embeddings
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim)
        self.box_enc = nn.Sequential(
            nn.Linear(
                4, hidden_dim), nn.ReLU(
                inplace=True), nn.Linear(
                hidden_dim, hidden_dim))
        self.dn_qpos = nn.Sequential(
            nn.Linear(
                4, hidden_dim), nn.ReLU(
                inplace=True), nn.Linear(
                hidden_dim, hidden_dim))

        self.class_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.cls_out_dim) for _ in range(dec_layers)
        ])
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 4),
            ) for _ in range(dec_layers)
        ])

        # Focal loss bias init
        if use_focal:
            bias_value = -math.log((1 - focal_prior) / focal_prior)
            for head in self.class_heads:
                nn.init.constant_(head.bias, bias_value)

    def _encoder_ref_points(self, spatial_shapes, device):
        refs = []
        for h, w in spatial_shapes:
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, device=device) / h,
                torch.linspace(0.5, w - 0.5, w, device=device) / w,
                indexing="ij",
            )
            refs.append(torch.stack(
                [ref_x.flatten(), ref_y.flatten()], dim=-1))
        refs = torch.cat(refs, dim=0)
        return refs[:, None, :].expand(-1, len(spatial_shapes), -1)

    def _build_dn_inputs(self, targets, device):
        if not self.training or not self.use_dn or targets is None:
            return None, None, None, None
        out = make_denoising_queries(
            targets,
            self.num_classes,
            self.dn_number,
            self.label_noise_ratio,
            self.box_noise_scale,
            device)
        if out is None:
            return None, None, None, None
        dn_labels, dn_boxes, dn_meta = out
        dn_tgt = self.label_enc(dn_labels) + self.box_enc(dn_boxes)
        dn_qpos = self.dn_qpos(dn_boxes)
        return dn_tgt, dn_qpos, dn_boxes, dn_meta

    def forward(self, x, targets=None):
        B = x.shape[0]
        device = x.device

        c3, c4, c5 = self.backbone(x)
        raw = [c3, c4, c5, c5]

        srcs, poss, spatial_shapes = [], [], []
        for lid, feat in enumerate(raw):
            src = self.input_proj[lid](feat)
            _, _, h, w = src.shape
            spatial_shapes.append((h, w))
            pos = build_sincos_pos_embed(h, w, self.hidden_dim, device)
            pos = pos.unsqueeze(0).expand(B, -1, -1) + self.level_embed[lid]
            srcs.append(src.flatten(2).permute(0, 2, 1))
            poss.append(pos)

        src_flat = torch.cat(srcs, dim=1)
        pos_flat = torch.cat(poss, dim=1)

        enc_ref = self._encoder_ref_points(
            spatial_shapes, device).unsqueeze(0).expand(
            B, -1, -1, -1)
        memory = src_flat
        for layer in self.encoder:
            memory = layer(memory, pos_flat, enc_ref, spatial_shapes)

        qe = self.query_embed.weight
        query_pos, query_content = qe.split(self.hidden_dim, dim=-1)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)
        tgt = query_content.unsqueeze(0).expand(B, -1, -1)
        ref_pts = self.ref_point_head(query_pos).sigmoid()

        dn_tgt, dn_qpos, dn_ref, dn_meta = self._build_dn_inputs(
            targets, device)
        total_attn_mask = None
        if dn_tgt is not None:
            tgt = torch.cat([dn_tgt, tgt], dim=1)
            query_pos = torch.cat([dn_qpos, query_pos], dim=1)
            ref_pts = torch.cat([dn_ref, ref_pts], dim=1)
            total_len = tgt.size(1)
            pad_size = dn_meta["pad_size"]
            total_attn_mask = torch.zeros(
                (total_len, total_len), dtype=torch.bool, device=device)
            total_attn_mask[:pad_size, :pad_size] = dn_meta["attn_mask"]
            total_attn_mask[pad_size:, :pad_size] = True

        outputs_cls, outputs_box = [], []
        for lid, layer in enumerate(self.decoder):
            ref_for_attn = ref_pts[:, :, None,
                                   :2].expand(-1, -1, self.n_levels, -1)
            tgt = layer(
                tgt,
                query_pos,
                memory,
                ref_for_attn,
                spatial_shapes,
                attn_mask=total_attn_mask)
            tgt_norm = self.dec_norm(tgt)
            logits = self.class_heads[lid](tgt_norm)
            pred_boxes = (
                self.bbox_heads[lid](tgt_norm) +
                inverse_sigmoid(ref_pts)).sigmoid()
            outputs_cls.append(logits)
            outputs_box.append(pred_boxes)
            ref_pts = pred_boxes.detach()

        out = {
            "pred_logits": outputs_cls[-1],
            "pred_boxes": outputs_box[-1],
            "aux_outputs": [
                {"pred_logits": outputs_cls[i], "pred_boxes": outputs_box[i]}
                for i in range(len(outputs_cls) - 1)
            ],
        }
        if dn_meta is not None:
            out["dn_meta"] = dn_meta
        return out


# ===================== Focal Loss =====================

def sigmoid_focal_loss(
        logits,
        targets_onehot,
        alpha=0.25,
        gamma=2.0,
        num_boxes=1):
    """Focal loss for multi-label classification."""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(
        logits, targets_onehot, reduction="none")
    p_t = prob * targets_onehot + (1 - prob) * (1 - targets_onehot)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
        loss = alpha_t * loss
    return loss.mean(1).sum() / max(num_boxes, 1)


# ===================== Hungarian Matcher =====================

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets, use_focal=False):
        indices = []
        B = outputs["pred_logits"].shape[0]
        for b in range(B):
            tgt_labels = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes"]
            if tgt_labels.numel() == 0:
                indices.append(
                    (torch.empty(
                        0, dtype=torch.long), torch.empty(
                        0, dtype=torch.long)))
                continue
            pred_box = outputs["pred_boxes"][b]
            if use_focal:
                out_prob = outputs["pred_logits"][b].sigmoid()
                neg_cost = self.focal_alpha * \
                    (out_prob ** self.focal_gamma) * \
                    (-(1 - out_prob + 1e-8).log())
                pos_cost = (1 - self.focal_alpha) * ((1 - out_prob)
                                                     ** self.focal_gamma) * (-(out_prob + 1e-8).log())
                cost_cls = pos_cost[:, tgt_labels - 1] - \
                    neg_cost[:, tgt_labels - 1]
            else:
                cost_cls = - \
                    outputs["pred_logits"][b].softmax(-1)[:, tgt_labels]
            cost = (self.cost_class * cost_cls
                    + self.cost_bbox * torch.cdist(pred_box, tgt_boxes, p=1)
                    + self.cost_giou * (-generalized_box_iou(
                        box_cxcywh_to_xyxy(pred_box),
                        box_cxcywh_to_xyxy(tgt_boxes),
                    )))
            row, col = linear_sum_assignment(cost.cpu().numpy())
            indices.append((torch.as_tensor(row, dtype=torch.long),
                            torch.as_tensor(col, dtype=torch.long)))
        return indices


# ===================== Loss =====================

class SetCriterion(nn.Module):
    def __init__(self, num_classes=10, eos_coef=0.1,
                 w_ce=1.0, w_bbox=5.0, w_giou=2.0, dn_loss_coef=1.0,
                 use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        self.w_ce = w_ce
        self.w_bbox = w_bbox
        self.w_giou = w_giou
        self.dn_loss_coef = dn_loss_coef
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[0] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _compute(self, outputs, targets, indices):
        device = outputs["pred_logits"].device
        num_boxes = max(sum(len(s) for s, _ in indices), 1)

        if self.use_focal:
            B, Q, C = outputs["pred_logits"].shape
            tgt_oh = torch.zeros(
                (B, Q, C), dtype=outputs["pred_logits"].dtype, device=device)
            for b, (si, ti) in enumerate(indices):
                if len(si) > 0:
                    tgt_oh[b, si, targets[b]["labels"]
                           [ti].to(device) - 1] = 1.0
            loss_ce = sigmoid_focal_loss(
                outputs["pred_logits"],
                tgt_oh,
                self.focal_alpha,
                self.focal_gamma,
                num_boxes)
        else:
            B, Q, _ = outputs["pred_logits"].shape
            tgt_classes = torch.zeros(B, Q, dtype=torch.long, device=device)
            for b, (src_idx, tgt_idx) in enumerate(indices):
                if len(src_idx):
                    tgt_classes[b, src_idx] = targets[b]["labels"][tgt_idx].to(
                        device)
            loss_ce = F.cross_entropy(
                outputs["pred_logits"].permute(
                    0, 2, 1), tgt_classes, weight=self.empty_weight)

        src_boxes, tgt_boxes = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx):
                src_boxes.append(outputs["pred_boxes"][b][src_idx])
                tgt_boxes.append(targets[b]["boxes"][tgt_idx].to(device))
        if src_boxes:
            sb = torch.cat(src_boxes)
            tb = torch.cat(tgt_boxes)
            loss_bbox = F.l1_loss(sb, tb, reduction="sum") / num_boxes
            loss_giou = (1 - generalized_box_iou(
                box_cxcywh_to_xyxy(sb), box_cxcywh_to_xyxy(tb)
            ).diag()).sum() / num_boxes
        else:
            loss_bbox = loss_giou = torch.tensor(0.0, device=device)

        total = self.w_ce * loss_ce + self.w_bbox * loss_bbox + self.w_giou * loss_giou
        return total, loss_ce.item(), loss_bbox.item(), loss_giou.item()

    def _compute_dn_loss(self, outputs, targets, dn_meta):
        device = outputs["pred_logits"].device
        pad_size = int(dn_meta["pad_size"])
        if pad_size <= 0:
            return torch.tensor(0.0, device=device)
        pred_logits = outputs["pred_logits"][:, :pad_size]
        pred_boxes = outputs["pred_boxes"][:, :pad_size]
        src_l, tgt_l, src_b, tgt_b = [], [], [], []
        for b, pairs in enumerate(dn_meta["dn_positive_idx"]):
            if not pairs:
                continue
            si = torch.tensor([p[0] for p in pairs],
                              dtype=torch.long, device=device)
            ti = torch.tensor([p[1] for p in pairs],
                              dtype=torch.long, device=device)
            src_l.append(pred_logits[b, si])
            tgt_l.append(targets[b]["labels"][ti].to(device))
            src_b.append(pred_boxes[b, si])
            tgt_b.append(targets[b]["boxes"][ti].to(device))
        if not src_l:
            return torch.tensor(0.0, device=device)
        src_b = torch.cat(src_b)
        tgt_b = torch.cat(tgt_b)
        nb = max(src_b.shape[0], 1)
        if self.use_focal:
            src_l_cat = torch.cat(src_l)
            tgt_oh = torch.zeros_like(src_l_cat)
            tgt_l_cat = torch.cat(tgt_l)
            for i, lbl in enumerate(tgt_l_cat):
                tgt_oh[i, int(lbl.item()) - 1] = 1.0
            loss_ce = sigmoid_focal_loss(
                src_l_cat, tgt_oh, self.focal_alpha, self.focal_gamma, nb)
        else:
            src_l_cat = torch.cat(src_l)
            tgt_l_cat = torch.cat(tgt_l)
            loss_ce = F.cross_entropy(
                src_l_cat, tgt_l_cat, weight=self.empty_weight)
        loss_bbox = F.l1_loss(src_b, tgt_b, reduction="sum") / nb
        loss_giou = (1.0 - generalized_box_iou(
            box_cxcywh_to_xyxy(src_b), box_cxcywh_to_xyxy(tgt_b)
        ).diag()).sum() / nb
        return (self.w_ce * loss_ce + self.w_bbox * loss_bbox +
                self.w_giou * loss_giou) * self.dn_loss_coef

    def _split_match_outputs(self, outputs):
        dn_meta = outputs.get("dn_meta")
        if dn_meta is None:
            return outputs, None
        pad = int(dn_meta["pad_size"])
        match_out = {
            "pred_logits": outputs["pred_logits"][:, pad:],
            "pred_boxes": outputs["pred_boxes"][:, pad:],
        }
        if "aux_outputs" in outputs:
            match_out["aux_outputs"] = [
                {"pred_logits": a["pred_logits"][:, pad:],
                    "pred_boxes": a["pred_boxes"][:, pad:]}
                for a in outputs["aux_outputs"]
            ]
        return match_out, dn_meta

    def forward(self, outputs, targets):
        match_out, dn_meta = self._split_match_outputs(outputs)
        indices = self.matcher(match_out, targets, use_focal=self.use_focal)
        loss, ce, bbox, giou = self._compute(match_out, targets, indices)

        if "aux_outputs" in match_out:
            for aux in match_out["aux_outputs"]:
                aux_idx = self.matcher(aux, targets, use_focal=self.use_focal)
                aux_loss, _, _, _ = self._compute(aux, targets, aux_idx)
                loss = loss + aux_loss

        dn_loss = torch.tensor(0.0, device=loss.device)
        if dn_meta is not None:
            dn_loss = self._compute_dn_loss(outputs, targets, dn_meta)
            loss = loss + dn_loss

        return loss, {
            "loss": loss.item(),
            "loss_ce": ce,
            "loss_bbox": bbox,
            "loss_giou": giou,
            "loss_dn": float(dn_loss.item()),
        }, indices


# ===================== Postprocess =====================

def postprocess_single(
        logits,
        boxes,
        img_size,
        scale,
        pad_left,
        pad_top,
        orig_w,
        orig_h,
        score_thresh,
        nms_thresh,
        use_focal=False):
    """Shared postprocess for eval and inference."""
    if use_focal:
        probs = logits.sigmoid()
        scores, ci = probs.max(dim=-1)
        cls_ids = ci + 1
    else:
        probs = logits.softmax(-1)
        scores, ci = probs[:, 1:].max(dim=-1)
        cls_ids = ci + 1

    keep = scores > score_thresh
    scores = scores[keep]
    cls_ids = cls_ids[keep]
    boxes = boxes[keep]
    if scores.numel() == 0:
        return []

    bx_xywh = coords_to_orig(
        boxes.cpu(),
        img_size,
        scale,
        pad_left,
        pad_top,
        orig_w,
        orig_h)
    bx_xyxy = bx_xywh.clone()
    bx_xyxy[:, 2] += bx_xyxy[:, 0]
    bx_xyxy[:, 3] += bx_xyxy[:, 1]

    ki = []
    for cls in cls_ids.unique():
        cm = (cls_ids == cls).nonzero(as_tuple=True)[0]
        ki.append(
            cm[nms(bx_xyxy[cm].float(), scores[cm].float(), nms_thresh)])
    if not ki:
        return []
    ki = torch.cat(ki)
    return [{"bbox": bx_xywh[i].tolist(), "score": float(
        scores[i]), "category_id": int(cls_ids[i])} for i in ki.tolist()]


# ===================== LR Scheduler =====================

def build_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * \
            (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================== Train =====================

def train_one_epoch(
        model,
        criterion,
        loader,
        optimizer,
        device,
        scaler,
        ema,
        use_dn,
        clip_max_norm):
    model.train()
    criterion.train()
    total = 0.0
    nb = 0
    pbar = tqdm(loader, desc="Train")
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with get_autocast(device):
            outputs = model(imgs, targets if use_dn else None)
            loss, ld, _ = criterion(outputs, targets)
        scaler.scale(loss).backward()
        if clip_max_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        total += ld["loss"]
        nb += 1
        pbar.set_postfix(
            loss=f"{ld['loss']:.4f}",
            ce=f"{ld['loss_ce']:.3f}",
            bbox=f"{ld['loss_bbox']:.3f}",
            giou=f"{ld['loss_giou']:.3f}",
            dn=f"{ld['loss_dn']:.3f}",
        )
    return total / max(nb, 1)


# ===================== Evaluate =====================

@torch.no_grad()
def evaluate(model, criterion, loader, device, img_size, output_dir, epoch,
             num_classes, val_score_thresh, nms_thresh, use_focal):
    model.eval()
    total_loss = 0.0
    nb = 0
    all_pred_cls, all_gt_cls = [], []
    all_preds_coco = []
    all_gt_coco = {
        "images": [], "annotations": [],
        "categories": [{"id": i} for i in range(1, num_classes + 1)],
    }
    gt_ann_id = 0

    for imgs, targets in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]
        with get_autocast(device):
            outputs = model(imgs)
            loss, _, indices = criterion(outputs, targets)
        total_loss += loss.item()
        nb += 1

        # Strip DN queries before accuracy check
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        if "dn_meta" in outputs:
            ps = int(outputs["dn_meta"]["pad_size"])
            pred_logits = pred_logits[:, ps:]
            pred_boxes = pred_boxes[:, ps:]

        if use_focal:
            pred_cls = pred_logits.sigmoid().argmax(dim=-1) + 1
        else:
            pred_cls = pred_logits.argmax(dim=-1)

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                all_pred_cls.extend(pred_cls[b][src_idx].cpu().tolist())
                all_gt_cls.extend(
                    targets[b]["labels"][tgt_idx].cpu().tolist())

        for b in range(imgs.size(0)):
            img_id = int(targets[b]["image_id"])
            orig_h, orig_w = targets[b]["orig_size"].tolist()
            scale = float(targets[b]["scale"].item())
            pad_l, pad_t = targets[b]["pad"].tolist()

            all_gt_coco["images"].append(
                {"id": img_id, "width": int(orig_w), "height": int(orig_h)})
            for j in range(len(targets[b]["labels"])):
                box_abs = coords_to_orig(
                    targets[b]["boxes"][j:j + 1].cpu(), img_size, scale, pad_l, pad_t, orig_w, orig_h)
                bx = box_abs[0].tolist()
                all_gt_coco["annotations"].append({
                    "id": gt_ann_id, "image_id": img_id,
                    "category_id": int(targets[b]["labels"][j].item()),
                    "bbox": bx, "area": float(bx[2] * bx[3]), "iscrowd": 0,
                })
                gt_ann_id += 1

            preds = postprocess_single(
                pred_logits[b].cpu(), pred_boxes[b].cpu(),
                img_size, scale, pad_l, pad_t, orig_w, orig_h,
                val_score_thresh, nms_thresh, use_focal,
            )
            for p in preds:
                all_preds_coco.append({
                    "image_id": img_id,
                    "category_id": p["category_id"],
                    "bbox": [round(v, 2) for v in p["bbox"]],
                    "score": round(p["score"], 6),
                })

    mAP = 0.0
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO()
        coco_gt.dataset = all_gt_coco
        coco_gt.createIndex()
        if all_preds_coco:
            coco_dt = coco_gt.loadRes(all_preds_coco)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()
            coco_eval.summarize()
            mAP = float(coco_eval.stats[0])
    except Exception as e:
        print(f"mAP eval failed: {e}")

    acc = (sum(p == g for p, g in zip(all_pred_cls, all_gt_cls)) /
           len(all_pred_cls) if all_pred_cls else 0.0)
    return total_loss / max(nb, 1), acc, mAP, all_pred_cls, all_gt_cls


# ===================== Plotting =====================

def plot_curves(history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    ep = [i for i, v in enumerate(history["val_loss"]) if v is not None]
    vl = [v for v in history["val_loss"] if v is not None]
    if vl:
        axes[0, 1].plot(ep, vl, marker="o", markersize=3, label="Val Loss")
    axes[0, 1].set_title("Val Loss")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    ea = [i for i, v in enumerate(history["val_acc"]) if v is not None]
    va = [v for v in history["val_acc"] if v is not None]
    if va:
        axes[1, 0].plot(ea, va, marker="o", markersize=3, label="Val Acc")
    axes[1, 0].set_title("Val Matched Accuracy")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    em = [i for i, v in enumerate(history["val_mAP"]) if v is not None]
    vm = [v for v in history["val_mAP"] if v is not None]
    if vm:
        axes[1, 1].plot(em, vm, marker="o", markersize=3,
                        label="mAP @[.5:.95]")
    axes[1, 1].set_title("Val mAP @[.5:.95]")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curves.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(pred_cls, gt_cls, output_dir, epoch, num_classes=10):
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        labels = list(range(1, num_classes + 1))
        cm = confusion_matrix(gt_cls, pred_cls, labels=labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            cm, display_labels=[
                str(i) for i in labels])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix (Epoch {epoch})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"confusion_matrix_ep{epoch}.png"),
            dpi=150)
        plt.close(fig)
    except ImportError:
        print("sklearn not installed, skipping confusion matrix")


# ===================== Inference =====================

@torch.no_grad()
def run_inference(
        model,
        test_dir,
        img_size,
        device,
        score_thresh,
        nms_thresh,
        use_focal):
    model.eval()
    ds = TestDataset(test_dir, img_size)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_test,
        pin_memory=(
            device.type == "cuda"))
    results = []
    for imgs, img_ids, hs, ws, scales, pls, pts in tqdm(
            loader, desc="Inference"):
        imgs = imgs.to(device, non_blocking=True)
        with get_autocast(device):
            outputs = model(imgs)
        for i in range(imgs.shape[0]):
            preds = postprocess_single(
                outputs["pred_logits"][i].cpu(),
                outputs["pred_boxes"][i].cpu(),
                img_size,
                scales[i],
                pls[i],
                pts[i],
                ws[i],
                hs[i],
                score_thresh,
                nms_thresh,
                use_focal,
            )
            for p in preds:
                results.append({
                    "image_id": int(img_ids[i]),
                    "bbox": [round(v, 2) for v in p["bbox"]],
                    "score": round(p["score"], 6),
                    "category_id": p["category_id"],
                })
    return results


# ===================== Main =====================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | img_size={args.img_size} | use_dn={args.use_dn} | use_focal={args.use_focal}")
    print(
        f"Mosaic p={args.mosaic_p} | Multi-scale={args.multi_scale} | RandomErase p={args.random_erase_p}")
    print(
        f"Data fraction={args.data_fraction} | batch_size={args.batch_size}")

    model = DeformableDETR(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        n_points=args.n_points,
        num_queries=args.num_queries,
        pretrained_backbone=True,
        use_dn=args.use_dn,
        dn_number=args.dn_number,
        label_noise_ratio=args.label_noise_ratio,
        box_noise_scale=args.box_noise_scale,
        use_focal=args.use_focal,
        focal_prior=args.focal_prior,
    ).to(device)

    ema = ModelEMA(model, decay=args.ema_decay)

    matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou,
                               args.focal_alpha, args.focal_gamma)
    criterion = SetCriterion(
        num_classes=args.num_classes,
        eos_coef=args.eos_coef,
        w_ce=args.loss_ce,
        w_bbox=args.loss_bbox,
        w_giou=args.loss_giou,
        dn_loss_coef=args.dn_loss_coef,
        use_focal=args.use_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    ).to(device)
    criterion.matcher = matcher

    start_epoch = 0
    best_mAP = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mAP": []}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mAP = ckpt.get("best_mAP", 0.0)
        history = ckpt.get("history", history)
        print(f"Resumed from epoch {start_epoch}, best_mAP={best_mAP:.4f}")

    if args.do_train:
        train_ds = DigitDataset(
            os.path.join(args.data_root, "train"),
            os.path.join(args.data_root, "train.json"),
            img_size=args.img_size, is_train=True,
            mosaic_p=args.mosaic_p,
            random_erase_p=args.random_erase_p,
            data_fraction=args.data_fraction,
        )
        val_ds = DigitDataset(
            os.path.join(args.data_root, "valid"),
            os.path.join(args.data_root, "valid.json"),
            img_size=args.img_size, is_train=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_train,
            pin_memory=True,
            drop_last=True)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_train,
            pin_memory=True)

        backbone_params = ("backbone.layer0", "backbone.layer1",
                           "backbone.layer2", "backbone.layer3")
        param_dicts = [
            {"params": [p for n, p in model.named_parameters()
                        if not n.startswith(backbone_params) and p.requires_grad], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters()
                        if n.startswith(backbone_params) and p.requires_grad], "lr": args.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, weight_decay=args.weight_decay)
        scheduler = build_scheduler(
            optimizer,
            args.warmup_epochs,
            args.epochs,
            args.min_lr_ratio)
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if args.resume:
            ckpt = torch.load(
                args.resume,
                map_location=device,
                weights_only=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])

        for epoch in range(start_epoch, args.epochs):
            # ---- Multi-scale: pick random resolution per epoch ----
            if args.multi_scale and len(args.multi_scale) > 1:
                ms = random.choice(args.multi_scale)
                train_ds.set_img_size(ms)
                print(f"  [Multi-scale] {ms}x{ms}")

            train_loss = train_one_epoch(model, criterion, train_loader,
                                         optimizer, device, scaler, ema,
                                         args.use_dn, args.clip_max_norm)
            scheduler.step()
            history["train_loss"].append(train_loss)

            # Reset to default size before eval
            train_ds.set_img_size(args.img_size)

            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                val_loss, acc, mAP, pred_cls, gt_cls = evaluate(
                    ema.ema, criterion, val_loader, device,
                    args.img_size, args.output_dir, epoch + 1,
                    args.num_classes, args.val_score_thresh, args.nms_thresh,
                    args.use_focal,
                )
                history["val_loss"].append(val_loss)
                history["val_acc"].append(acc)
                history["val_mAP"].append(mAP)
                print(f"Epoch {epoch+1}/{args.epochs} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                      f"Acc: {acc:.4f} | mAP(EMA): {mAP:.4f}")
                if pred_cls:
                    plot_confusion_matrix(pred_cls, gt_cls, args.output_dir,
                                          epoch + 1, args.num_classes)
            else:
                history["val_loss"].append(None)
                history["val_acc"].append(None)
                history["val_mAP"].append(None)
                print(
                    f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.4f}")

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mAP": best_mAP,
                "history": history,
            }
            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                if mAP > best_mAP:
                    best_mAP = mAP
                    ckpt["best_mAP"] = best_mAP
                    torch.save(ckpt, os.path.join(
                        args.output_dir, "best.pth"))
                    print(f"  ✓ Best saved (mAP={mAP:.4f})")
                plot_curves(history, args.output_dir)

            torch.save(ckpt, os.path.join(args.output_dir, "latest.pth"))

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        plot_curves(history, args.output_dir)
        print(f"Training complete! Best mAP: {best_mAP:.4f}")

    if args.do_infer:
        ckpt_path = args.resume or os.path.join(args.output_dir, "best.pth")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ema.ema.load_state_dict(ckpt["ema"])
        print(f"Loaded EMA from {ckpt_path}")

        results = run_inference(
            ema.ema,
            os.path.join(args.data_root, "test"),
            args.img_size, device,
            args.score_thresh, args.nms_thresh,
            args.use_focal,
        )
        pred_path = os.path.join(args.output_dir, args.pred_file)
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"Saved {len(results)} predictions to {pred_path}")


if __name__ == "__main__":
    main()

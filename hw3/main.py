# main.py
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import json
import argparse
import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib
matplotlib.use("Agg")


# ========================
# Dataset
# ========================

class CellDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = root
        self.augment = augment
        self.samples = sorted(os.listdir(root))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root, self.samples[idx])

        img = Image.open(os.path.join(sample_dir, "image.tif")).convert("RGB")
        W, H = img.size

        boxes = []
        masks = []
        labels = []

        class_map = {"class1": 1, "class2": 2, "class3": 3, "class4": 4}

        for class_name, class_id in class_map.items():
            mask_path = os.path.join(sample_dir, f"{class_name}.tif")
            if not os.path.exists(mask_path):
                continue

            mask = tifffile.imread(mask_path).astype(np.int32)
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]

            for inst_id in instance_ids:
                binary_mask = (mask == inst_id).astype(np.uint8)

                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(W - 1, int(x_max))
                y_max = min(H - 1, int(y_max))

                if x_max <= x_min or y_max <= y_min:
                    continue

                boxes.append([x_min, y_min, x_max, y_max])
                masks.append(binary_mask)
                labels.append(class_id)

        masks_np = np.array(masks) if masks else np.zeros(
            (0, H, W), dtype=np.uint8)

        if self.augment:
            if random.random() > 0.5:
                img = F.hflip(img)
                masks_np = masks_np[:, :, ::-1].copy()
                boxes = [[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in boxes]

            if random.random() > 0.5:
                img = F.vflip(img)
                masks_np = masks_np[:, ::-1, :].copy()
                boxes = [[x1, H - y2, x2, H - y1] for x1, y1, x2, y2 in boxes]

            if random.random() > 0.5:
                img = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                )(img)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks_np, dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        img = transforms.ToTensor()(img)

        target = {
            "boxes": boxes,
            "masks": masks,
            "labels": labels,
            "image_id": torch.tensor(idx),
        }

        return img, target


# ========================
# Model
# ========================

def get_model(num_classes):
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights="ResNet101_Weights.DEFAULT"
    )

    mask_head = MaskRCNNHeads(
        in_channels=256,
        layers=(256, 256, 256, 256, 256, 256),
        dilation=1
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        mask_head=mask_head
    )

    return model


# ========================
# Evaluate on val set
# ========================

@torch.no_grad()
def evaluate(model, val_loader, val_dataset, device, output_dir, epoch):
    model.eval()

    # 建立 GT COCO 格式
    coco_gt_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i} for i in range(1, 5)]
    }
    ann_id = 0
    for idx in range(len(val_dataset)):
        _, target = val_dataset[idx]
        image_id = int(target["image_id"].item())
        H_img, W_img = target["masks"].shape[-2:
                                             ] if target["masks"].shape[0] > 0 else (512, 512)  # noqa: E501
        coco_gt_dict["images"].append(
            {"id": image_id, "height": int(H_img), "width": int(W_img)})

        for i in range(len(target["labels"])):
            binary_mask = target["masks"][i].numpy().astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = target["boxes"][i].tolist()
            coco_gt_dict["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(target["labels"][i].item()),
                "segmentation": rle,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1

    # 跑推理
    coco_dt_list = []
    all_gt_labels = []
    all_pred_labels = []

    for imgs, targets in tqdm(
            val_loader, desc=f"Val Epoch {epoch}", leave=False):
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            masks = output["masks"].cpu().numpy()

            keep = scores >= 0.3
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            masks = masks[keep]

            for i in range(len(labels)):
                binary_mask = (masks[i, 0] >= 0.5).astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                x1, y1, x2, y2 = boxes[i]
                coco_dt_list.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "segmentation": {"size": list(binary_mask.shape), "counts": rle["counts"]},  # noqa: E501
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # noqa: E501
                    "score": float(scores[i]),
                })
                all_pred_labels.append(int(labels[i]))

            for label in target["labels"].tolist():
                all_gt_labels.append(int(label))

    # 計算 AP50
    ap50 = 0.0
    if coco_dt_list:
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(coco_dt_list)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.params.iouThrs = np.array([0.5])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap50 = float(coco_eval.stats[0])

    print(f"Epoch {epoch} | Val AP50: {ap50:.4f}")

    # Confusion matrix
    if all_gt_labels and all_pred_labels:
        min_len = min(len(all_gt_labels), len(all_pred_labels))
        cm = confusion_matrix(
            all_gt_labels[:min_len],
            all_pred_labels[:min_len],
            labels=[1, 2, 3, 4]
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            cm, display_labels=[
                "class1", "class2", "class3", "class4"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix (Epoch {epoch})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"confusion_matrix_ep{epoch}.png"),
            dpi=150)
        plt.close()

    return ap50


# ========================
# Plot curves
# ========================

def plot_curves(loss_history, ap_history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        range(
            1,
            len(loss_history) +
            1),
        loss_history,
        marker="o",
        markersize=3)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    ep = [i + 1 for i, v in enumerate(ap_history) if v is not None]
    ap = [v for v in ap_history if v is not None]
    if ap:
        axes[1].plot(ep, ap, marker="o", markersize=3, color="orange")
    axes[1].set_title("Validation AP50")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AP50")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {output_dir}/training_curves.png")


# ========================
# Train
# ========================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    full_dataset = CellDataset(root=args.train_root, augment=True)

    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset.augment = False

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_set, batch_size=1,
        shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model(num_classes=5)
    model.to(device)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {total_params / 1e6:.1f}M")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
    )

    loss_history = []
    ap_history = []
    best_ap = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        valid_batches = 0

        for imgs, targets in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += losses.item()
            valid_batches += 1

        lr_scheduler.step()
        avg_loss = total_loss / max(valid_batches, 1)
        loss_history.append(avg_loss)
        print(
            f"Epoch {epoch+1} | loss: {avg_loss:.4f} | valid: {valid_batches}/{len(train_loader)}")  # noqa: E501

        # 每 5 epoch 或最後一個 epoch 評估
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            ap50 = evaluate(
                model,
                val_loader,
                val_set.dataset,
                device,
                args.output_dir,
                epoch + 1)
            ap_history.append(ap50)

            if ap50 > best_ap:
                best_ap = ap50
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.checkpoint_dir,
                        "best.pth"))
                print(f"Best model saved! AP50: {best_ap:.4f}")
        else:
            ap_history.append(None)

        torch.save(
            model.state_dict(),
            os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pth")
        )

        # 每 epoch 更新 curves
        plot_curves(loss_history, ap_history, args.output_dir)

    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump({"loss": loss_history, "ap50": ap_history}, f)

    print(f"訓練完成！Best AP50: {best_ap:.4f}")


# ========================
# Inference with TTA
# ========================

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    with open(args.json_path, "r") as f:
        name_to_ids = json.load(f)

    filename_to_id = {item["file_name"]: item["id"] for item in name_to_ids}

    model = get_model(num_classes=5)
    model.load_state_dict(
        torch.load(
            args.checkpoint_path,
            map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.ToTensor()
    results = []

    test_files = sorted(os.listdir(args.test_root))
    print(f"共 {len(test_files)} 張測試圖片")

    for fname in test_files:
        if fname not in filename_to_id:
            print(f"找不到 {fname} 的 image_id，跳過")
            continue

        image_id = filename_to_id[fname]
        img_path = os.path.join(args.test_root, fname)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        tta_imgs = [
            transform(img),
            transform(F.hflip(img)),
            transform(F.vflip(img)),
        ]

        all_boxes = []
        all_scores = []
        all_labels = []
        all_masks = []

        with torch.no_grad():
            for i, img_tensor in enumerate(tta_imgs):
                output = model([img_tensor.to(device)])[0]

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                masks = output["masks"].cpu().numpy()

                if i == 1:
                    boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                    masks = masks[:, :, :, ::-1].copy()
                elif i == 2:
                    boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
                    masks = masks[:, :, ::-1, :].copy()

                keep = scores >= args.score_threshold
                all_boxes.append(boxes[keep])
                all_scores.append(scores[keep])
                all_labels.append(labels[keep])
                all_masks.append(masks[keep])

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        for i in range(len(all_labels)):
            binary_mask = (all_masks[i, 0] >= 0.5).astype(np.uint8)
            H_m, W_m = binary_mask.shape

            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            segmentation = {"size": [H_m, W_m], "counts": rle["counts"]}

            x1, y1, x2, y2 = all_boxes[i]
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            results.append({
                "image_id": image_id,
                "category_id": int(all_labels[i]),
                "bbox": bbox,
                "score": float(all_scores[i]),
                "segmentation": segmentation
            })

    with open(args.output_path, "w") as f:
        json.dump(results, f)

    print(f"完成！共 {len(results)} 個預測，存到 {args.output_path}")


# ========================
# Main
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train",
            "inference"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--train_root",
        type=str,
        default=os.path.expanduser("~/nycu-2/DL/hw3/train"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--test_root",
        type=str,
        default=os.path.expanduser("~/nycu-2/DL/hw3/test_release"))
    parser.add_argument(
        "--json_path",
        type=str,
        default=os.path.expanduser("~/nycu-2/DL/hw3/test_image_name_to_ids.json"))  # noqa: E501
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/best.pth")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--output_path", type=str, default="test-results.json")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)

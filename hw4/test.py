import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset
from model import build_model


def pad_to_multiple(x, multiple=8):
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w


@torch.no_grad()
def infer(model, img, device, tile_size=None, overlap=32):
    """
    Full-image inference with optional tiling for very large images.
    img: [1, 3, H, W] float tensor on CPU or device
    Returns: [1, 3, H, W] float tensor, clamped to [0,1]
    """
    img = img.to(device)
    _, _, H, W = img.shape

    if tile_size is None or (H <= tile_size and W <= tile_size):
        img_pad, h, w = pad_to_multiple(img)
        out = model(img_pad)[:, :, :h, :w]
        return out.clamp(0, 1)

    # Tiled inference
    stride = tile_size - overlap
    out_acc = torch.zeros(1, 3, H, W, device=device)
    cnt = torch.zeros(1, 1, H, W, device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y, H - tile_size)
            x1 = min(x, W - tile_size)
            y2 = y1 + tile_size
            x2 = x1 + tile_size

            tile = img[:, :, y1:y2, x1:x2]
            tile_pad, th, tw = pad_to_multiple(tile)
            tile_out = model(tile_pad)[:, :, :th, :tw]

            out_acc[:, :, y1:y2, x1:x2] += tile_out
            cnt[:, :, y1:y2, x1:x2] += 1

    return (out_acc / cnt).clamp(0, 1)


def _aug(x, k):
    """Apply k-th D4 augmentation to [1,C,H,W] tensor."""
    if k == 0:
        return x
    if k == 1:
        return x.flip(-1)
    if k == 2:
        return x.flip(-2)
    if k == 3:
        return x.flip(-1).flip(-2)
    if k == 4:
        return torch.rot90(x, 1, [2, 3])
    if k == 5:
        return torch.rot90(x, 1, [2, 3]).flip(-1)
    if k == 6:
        return torch.rot90(x, 1, [2, 3]).flip(-2)
    if k == 7:
        return torch.rot90(x, 3, [2, 3])


def _deaug(x, k):
    """Inverse of _aug(x, k)."""
    if k == 0:
        return x
    if k == 1:
        return x.flip(-1)
    if k == 2:
        return x.flip(-2)
    if k == 3:
        return x.flip(-1).flip(-2)
    if k == 4:
        return torch.rot90(x, 3, [2, 3])
    if k == 5:
        return torch.rot90(x.flip(-1), 3, [2, 3])
    if k == 6:
        return torch.rot90(x.flip(-2), 3, [2, 3])
    if k == 7:
        return torch.rot90(x, 1, [2, 3])


@torch.no_grad()
def tta_infer(model, img, device, tile_size=None, overlap=32):
    """8-fold TTA over D4 symmetries, then average."""
    img = img.to(device)
    acc = None
    for k in range(8):
        pred = infer(model, _aug(img, k), device, tile_size, overlap)
        pred = _deaug(pred, k)
        acc = pred if acc is None else acc + pred
    return (acc / 8).clamp(0, 1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = build_model().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    epoch_str = ckpt.get("epoch", "?")
    psnr_val = ckpt.get("psnr", 0)
    print(f'Loaded checkpoint: epoch={epoch_str}, PSNR={psnr_val:.2f}')

    # Dataset
    test_ds = TestDataset(args.data_dir)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    print(f'Test images: {len(test_ds)}')

    infer_fn = tta_infer if args.tta else infer
    results = {}
    for img, (filename,) in tqdm(loader, desc='Inference'):
        pred = infer_fn(model, img, device,
                        tile_size=args.tile_size if args.tile else None,
                        overlap=args.overlap)
        # [1, 3, H, W] -> [3, H, W] uint8
        pred_np = (pred.squeeze(0).cpu().numpy()
                   * 255).round().astype(np.uint8)
        results[filename] = pred_np

    np.savez(args.output, **results)
    print(f'Saved {len(results)} predictions to {args.output}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='data/hw4_realse_dataset')
    p.add_argument('--checkpoint', default='checkpoints/best.pth')
    p.add_argument('--output',     default='pred.npz')
    p.add_argument('--tile',       action='store_true',
                   help='Use tiled inference (for large images)')
    p.add_argument('--tile_size',  type=int, default=256)
    p.add_argument('--overlap',    type=int, default=32)
    p.add_argument('--tta',        action='store_true',
                   help='8-fold test-time augmentation (D4 symmetries)')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())

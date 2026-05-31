import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import TrainDataset, make_train_val_split
from model import build_model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred.float(), target.float())
    return (
        20 * torch.log10(torch.tensor(max_val))
        - 10 * torch.log10(mse)
    ).item()


def pad_to_multiple(x, multiple=8):
    """Reflect-pad so H and W are divisible by `multiple`."""
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_psnr = 0.0
    for deg, cln in val_loader:
        deg, cln = deg.to(device), cln.to(device)
        deg_pad, h, w = pad_to_multiple(deg)
        pred = model(deg_pad)[:, :, :h, :w].clamp(0, 1)
        total_psnr += psnr(pred, cln)
    return total_psnr / len(val_loader)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- Data ----
    train_files, val_files = make_train_val_split(
        args.data_dir, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f'Train: {len(train_files)}  Val: {len(val_files)}')

    train_ds = TrainDataset(
        args.data_dir, file_list=train_files, patch_size=args.patch_size)
    val_ds = TrainDataset(
        args.data_dir, file_list=val_files,
        patch_size=args.patch_size * 2)

    # Use a simpler val dataset that loads full images
    class FullImageDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds):
            self.files = base_ds.degraded_files
            self.clean_dir = base_ds.clean_dir
            self._get_clean = base_ds._get_clean_path
            import torchvision.transforms.functional as TF
            self.to_tensor = TF.to_tensor

        def __len__(self): return len(self.files)

        def __getitem__(self, idx):
            from PIL import Image
            dp = self.files[idx]
            cp = self._get_clean(dp)
            deg = self.to_tensor(Image.open(dp).convert('RGB'))
            cln = self.to_tensor(Image.open(cp).convert('RGB'))
            return deg, cln

    val_full_ds = FullImageDataset(val_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_full_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ---- Model ----
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model params: {n_params:.1f}M')

    # ---- Optimizer & Scheduler ----
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = GradScaler('cuda')
    criterion = nn.L1Loss()

    # Resume if checkpoint exists
    start_epoch = 1
    best_psnr = 0.0
    latest_ckpt = os.path.join(args.ckpt_dir, 'latest.pth')
    if args.resume and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f'Resumed from epoch {ckpt["epoch"]}, best PSNR={best_psnr:.2f}')

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'[{epoch}/{args.epochs}]', leave=False)
        for deg, cln in pbar:
            deg, cln = deg.to(device), cln.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                pred = model(deg)
                loss = criterion(pred, cln)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]['lr']

        # Save latest
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
        }, latest_ckpt)

        # Validate
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            val_psnr = validate(model, val_loader, device)
            print(
                f'Epoch {epoch:4d} | loss={avg_loss:.4f}'
                f' | lr={lr_now:.2e} | val PSNR={val_psnr:.2f} dB')

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'psnr': best_psnr,
                }, os.path.join(args.ckpt_dir, 'best.pth'))
                print(f'  -> Saved best (PSNR={best_psnr:.2f})')
        else:
            print(f'Epoch {epoch:4d} | loss={avg_loss:.4f} | lr={lr_now:.2e}')


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',     default='data/hw4_realse_dataset')
    p.add_argument('--ckpt_dir',     default='checkpoints')
    p.add_argument('--epochs',       type=int,   default=300)
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--patch_size',   type=int,   default=128)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--val_ratio',    type=float, default=0.05)
    p.add_argument('--val_freq',     type=int,   default=5)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--resume',       action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TARGET_KEYS = ("Dw", "rho")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_nii(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W)


def bbox_from_mask(mask: np.ndarray, margin: int = 12):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1
    z0 = max(z0 - margin, 0)
    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    z1 = min(z1 + margin, mask.shape[0])
    y1 = min(y1 + margin, mask.shape[1])
    x1 = min(x1 + margin, mask.shape[2])
    return (z0, z1, y0, y1, x0, x1)


def crop(arr: np.ndarray, bb) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bb
    return arr[z0:z1, y0:y1, x0:x1]


def resize_3d(arr: np.ndarray, out_shape: Sequence[int]) -> np.ndarray:
    t = torch.from_numpy(arr)[None, None, ...]
    t = F.interpolate(t, size=tuple(out_shape), mode="trilinear", align_corners=False)
    return t[0, 0].numpy()


def get_targets(sample_dir: Path, target_keys: Sequence[str], log_dw: bool = True) -> np.ndarray:
    params = json.loads((sample_dir / "parameters.json").read_text())
    vals = []
    for k in target_keys:
        v = float(params[k])
        if k == "Dw" and log_dw:
            v = math.log(v)
        vals.append(v)
    return np.array(vals, dtype=np.float32)


class TumorSegDatasetCached(Dataset):
    def __init__(
        self,
        sample_dirs: Sequence[Path],
        out_shape=(96, 96, 96),
        target_keys: Sequence[str] = TARGET_KEYS,
        y_mean: np.ndarray | None = None,
        y_std: np.ndarray | None = None,
        log_dw: bool = True,
        verbose: bool = True,
    ):
        self.sample_dirs = list(sample_dirs)
        self.out_shape = tuple(out_shape)
        self.target_keys = tuple(target_keys)
        self.y_mean = y_mean
        self.y_std = y_std
        self.log_dw = log_dw

        self.cache = []
        if verbose:
            print(f"Caching {len(self.sample_dirs)} samples...")

        for i, sd in enumerate(self.sample_dirs):
            x, y = self._load_one(sd)
            self.cache.append((x, y))
            if verbose and (i + 1) % 25 == 0:
                print(f"  cached {i + 1}/{len(self.sample_dirs)}")

        if verbose:
            print("Caching done.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]

    def _load_one(self, sd: Path):
        y_raw = get_targets(sd, self.target_keys, log_dw=self.log_dw)
        if self.y_mean is None or self.y_std is None:
            y_norm = y_raw
        else:
            y_norm = (y_raw - self.y_mean) / self.y_std
        y = torch.tensor(y_norm, dtype=torch.float32)

        segm = read_nii(sd / "segm.nii.gz")
        mask = (segm > 0).astype(np.float32)

        bb = bbox_from_mask(mask > 0, margin=12)
        if bb is not None:
            mask = crop(mask, bb)

        mask = resize_3d(mask, self.out_shape)
        x = torch.from_numpy(mask[None, ...]).float()  # (1,D,H,W)
        return x, y


class CNN3DRegressor(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))


def split_samples(samples: Sequence[Path], seed: int):
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    train = samples[: int(0.7 * n)]
    val = samples[int(0.7 * n) : int(0.85 * n)]
    test = samples[int(0.85 * n) :]
    return train, val, test


def mae(pred, y):
    return (pred - y).abs().mean()


def evaluate_metrics(preds: np.ndarray, trues: np.ndarray, keys: Sequence[str]):
    out = {}
    for i, k in enumerate(keys):
        err = preds[:, i] - trues[:, i]
        mae_i = float(np.abs(err).mean())
        rmse_i = float(np.sqrt((err**2).mean()))
        denom = np.maximum(np.abs(trues[:, i]), 1e-8)
        mape_i = float((np.abs(err) / denom).mean() * 100.0)
        corr_i = float(np.corrcoef(preds[:, i], trues[:, i])[0, 1]) if len(preds) > 1 else float("nan")
        out[k] = {
            "mae": mae_i,
            "rmse": rmse_i,
            "mape_percent": mape_i,
            "corr": corr_i,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Phase 3 baseline: seg-only 3D CNN for Dw/rho regression")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/phase3_segm_baseline")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--d", type=int, default=96)
    parser.add_argument("--h", type=int, default=96)
    parser.add_argument("--w", type=int, default=96)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-dw", action="store_true", default=True)
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = sorted([p for p in data_root.iterdir() if p.is_dir() and (p.name.startswith("sample_") or p.name.startswith("synthetic1T_run"))])
    if not samples:
        raise RuntimeError(f"No sample folders found in {data_root} (expected names like sample_* or synthetic1T_run*)")

    train, val, test = split_samples(samples, seed=args.split_seed)
    print(f"Samples: total={len(samples)} train={len(train)} val={len(val)} test={len(test)}")

    y_train = np.stack([get_targets(s, TARGET_KEYS, log_dw=args.log_dw) for s in train], axis=0)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0) + 1e-6

    out_shape = (args.d, args.h, args.w)
    train_ds = TumorSegDatasetCached(train, out_shape=out_shape, target_keys=TARGET_KEYS, y_mean=y_mean, y_std=y_std, log_dw=args.log_dw)
    val_ds = TumorSegDatasetCached(val, out_shape=out_shape, target_keys=TARGET_KEYS, y_mean=y_mean, y_std=y_std, log_dw=args.log_dw)
    test_ds = TumorSegDatasetCached(test, out_shape=out_shape, target_keys=TARGET_KEYS, y_mean=y_mean, y_std=y_std, log_dw=args.log_dw)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = CNN3DRegressor(in_channels=1, out_dim=len(TARGET_KEYS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_mae = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.item())
            tr_mae += float(mae(pred.detach(), y).item())

        model.eval()
        va_loss = va_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    pred = model(x)
                    loss = loss_fn(pred, y)
                va_loss += float(loss.item())
                va_mae += float(mae(pred, y).item())

        val_mse = va_loss / max(len(val_loader), 1)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | train MSE {tr_loss/max(len(train_loader),1):.4f} MAE {tr_mae/max(len(train_loader),1):.4f} "
            f"| val MSE {val_mse:.4f} MAE {va_mae/max(len(val_loader),1):.4f} | best {best_val_mse:.4f} @ {best_epoch:03d}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_norm, trues_norm = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).cpu().numpy()[0]
            preds_norm.append(pred)
            trues_norm.append(y.numpy()[0])

    preds_norm = np.array(preds_norm)
    trues_norm = np.array(trues_norm)

    preds = preds_norm * y_std + y_mean
    trues = trues_norm * y_std + y_mean

    dw_idx = TARGET_KEYS.index("Dw")
    if args.log_dw:
        preds[:, dw_idx] = np.exp(preds[:, dw_idx])
        trues[:, dw_idx] = np.exp(trues[:, dw_idx])

    metrics = evaluate_metrics(preds, trues, TARGET_KEYS)

    print("\nTest metrics:")
    for k in TARGET_KEYS:
        m = metrics[k]
        print(
            f"  {k}: MAE={m['mae']:.6f} RMSE={m['rmse']:.6f} "
            f"MAPE={m['mape_percent']:.2f}% Corr={m['corr']:.4f}"
        )

    ckpt_path = out_dir / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "target_keys": TARGET_KEYS,
            "out_shape": out_shape,
            "y_mean": y_mean,
            "y_std": y_std,
            "log_dw": args.log_dw,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "data_root": str(data_root),
            "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        },
        ckpt_path,
    )

    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    preds_csv = out_dir / "test_predictions.csv"
    with preds_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "true_Dw", "pred_Dw", "true_rho", "pred_rho"])
        for sd, t, p in zip(test, trues, preds):
            writer.writerow([sd.name, t[0], p[0], t[1], p[1]])

    split_path = out_dir / "split_manifest.json"
    split_path.write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "seed": args.split_seed,
                "train": [p.name for p in train],
                "val": [p.name for p in val],
                "test": [p.name for p in test],
            },
            indent=2,
        )
    )

    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Saved metrics:    {metrics_path}")
    print(f"Saved preds:      {preds_csv}")
    print(f"Saved split:      {split_path}")


if __name__ == "__main__":
    main()

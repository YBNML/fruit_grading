#!/usr/bin/env python3
"""
Multi-view + multi-task trainer (Phase 2).

One model per crop predicts all 4 size measurements (weight, height, max_w, min_w)
at once. Takes 3 views (t1, t2, b1) as input, shares backbone across views, and
the final head emits 4 scalars in normalized (z-score) space. Target stats are
computed on the train split and stored as model buffers so checkpoints are
self-contained.

Replaces 4 separate Phase-3 single-task runs with one combined run per crop;
feature sharing may regularize weaker targets (e.g. height) via the stronger ones.

Examples:
  python train_mv_mt.py --label label_peach.csv     --out runs_mv_mt/peach
  python train_mv_mt.py --label label_tangerine.csv --out runs_mv_mt/tangerine --epochs 25
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiViewMultiTaskDataset, load_label_csv
from model import MultiViewMultiTaskModel

DEFAULT_TARGETS = ("weight", "height", "max_w", "min_w")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_by_fruit(rows, test_ratio=0.3, seed=2020):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["date"], r["fruit_idx"])].append(r)
    fruits = list(groups.keys())
    rng.shuffle(fruits)
    n_test = int(len(fruits) * test_ratio)
    test_set = set(fruits[:n_test])
    train, test = [], []
    for k, items in groups.items():
        (test if k in test_set else train).extend(items)
    return train, test


def build_transforms(img_size, crop_size):
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.ToPILImage(),
        T.CenterCrop(crop_size),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        norm,
    ])
    test_tf = T.Compose([
        T.ToPILImage(),
        T.CenterCrop(crop_size),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        norm,
    ])
    return train_tf, test_tf


def compute_target_stats(dataset, target_keys):
    # Sweep one row per fruit (target is identical across views).
    vals = np.zeros((len(dataset.fruits), len(target_keys)), dtype=np.float32)
    for i, views in enumerate(dataset.fruits):
        r0 = views[MultiViewMultiTaskDataset.VIEW_ORDER[0]]
        for j, k in enumerate(target_keys):
            vals[i, j] = float(r0[k])
    means = vals.mean(axis=0)
    stds = vals.std(axis=0)
    # Guard against zero std to avoid divide-by-zero during normalization.
    stds = np.where(stds < 1e-6, 1.0, stds)
    return means, stds


def r2_score(y_true, y_pred):
    mean_y = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - mean_y) ** 2).sum()
    return 0.0 if ss_tot < 1e-8 else float(1 - ss_res / ss_tot)


def per_target_metrics(y_true, y_pred, target_keys):
    # y_*: (N, T) arrays in original units
    out = {}
    maes = []
    mapes = []
    for j, k in enumerate(target_keys):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        err = yp - yt
        mae = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        r2 = r2_score(yt, yp)
        mape = float((np.abs(err) / np.maximum(np.abs(yt), 1e-8)).mean() * 100)
        out[k] = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
        maes.append(mae)
        mapes.append(mape)
    out["avg_mape"] = float(np.mean(mapes))
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_norm = []
    targets_raw = []
    for imgs, tgt, _ in loader:
        imgs = imgs.to(device)
        preds_norm.append(model(imgs).cpu())
        targets_raw.append(tgt)
    pred_n = torch.cat(preds_norm, dim=0)
    pred = model.denormalize(pred_n.to(device)).cpu().numpy()
    target = torch.cat(targets_raw, dim=0).numpy()
    return pred, target


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "densenet201"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop-size", type=int, default=560)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--mask-bg", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")
    print(f"targets: {args.targets}")

    rows = load_label_csv(args.label)
    rows = [r for r in rows if all(r.get(k) is not None for k in args.targets)]
    print(f"loaded {len(rows)} rows with all {len(args.targets)} targets valid")

    train_rows, test_rows = split_by_fruit(rows, test_ratio=args.test_ratio, seed=args.seed)

    train_tf, test_tf = build_transforms(args.img_size, args.crop_size)
    train_ds = MultiViewMultiTaskDataset(train_rows, args.repo_root, args.targets,
                                         transform=train_tf, mask_bg=args.mask_bg)
    test_ds = MultiViewMultiTaskDataset(test_rows, args.repo_root, args.targets,
                                        transform=test_tf, mask_bg=args.mask_bg)
    print(f"fruit-level samples: train={len(train_ds)}  test={len(test_ds)}")

    means, stds = compute_target_stats(train_ds, args.targets)
    print(f"target stats (from train):")
    for k, m, s in zip(args.targets, means, stds):
        print(f"  {k:<8} mean={m:.3f}  std={s:.3f}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    model = MultiViewMultiTaskModel(args.targets, backbone=args.backbone,
                                    pretrained=not args.no_pretrained).to(device)
    model.set_norm(torch.from_numpy(means), torch.from_numpy(stds))
    means_t = model.target_means
    stds_t = model.target_stds

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_metric = float("inf")  # minimize avg MAPE
    stagnant = 0
    stopped_at = args.epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for imgs, tgt, _ in pbar:
            imgs = imgs.to(device)
            tgt = tgt.to(device)
            tgt_norm = (tgt - means_t) / stds_t
            optimizer.zero_grad()
            pred_norm = model(imgs)
            loss = criterion(pred_norm, tgt_norm)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * tgt.size(0)
            n += tgt.size(0)
            pbar.set_postfix(loss=f"{run_loss/max(n,1):.4f}")
        scheduler.step()
        train_loss = run_loss / max(n, 1)

        pred, target = evaluate(model, test_loader, device)
        metrics = per_target_metrics(target, pred, args.targets)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        print(f"epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  avg_MAPE={metrics['avg_mape']:.2f}%")
        for k in args.targets:
            m = metrics[k]
            print(f"  {k:<8} MAE={m['mae']:.3f}  R2={m['r2']:.3f}  MAPE={m['mape']:.2f}%")

        cur = metrics["avg_mape"]
        if cur < best_metric:
            best_metric = cur
            stagnant = 0
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "metrics": metrics,
                "target_keys": args.targets,
                "target_means": means.tolist(),
                "target_stds": stds.tolist(),
            }, out_dir / "best.pt")
            print(f"  -> saved best (avg_MAPE={cur:.2f}%)")
        else:
            stagnant += 1
            if args.patience > 0 and stagnant >= args.patience:
                stopped_at = epoch
                print(f"early stopping at epoch {epoch}")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump({
            "best_avg_mape": best_metric,
            "stopped_at": stopped_at,
            "target_keys": args.targets,
            "target_means": means.tolist(),
            "target_stds": stds.tolist(),
            "args": vars(args),
            "history": history,
        }, f, indent=2, default=str)
    print(f"done. best avg_MAPE={best_metric:.2f}%  stopped@{stopped_at}  -> {out_dir}")


if __name__ == "__main__":
    main()

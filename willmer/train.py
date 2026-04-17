#!/usr/bin/env python3
"""
Per-crop classifier (3-grade quality) or regressor (weight/height/...).

Examples:
  # classification
  python train.py --label label_peach.csv --out runs/peach_cls --task cls --target grade

  # regression
  python train.py --label label_peach.csv     --out runs/peach_weight --task reg --target weight
  python train.py --label label_tangerine.csv --out runs/tan_height  --task reg --target height
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import FruitDataset, load_label_csv
from model import build_model

CLS_TARGETS = {"grade"}
REG_TARGETS = {"weight", "height", "max_w", "min_w", "brix"}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_by_fruit(rows, test_ratio=0.3, seed=2020):
    # Split at the fruit level, not the row level: every fruit is present as 3
    # rows (t1/t2/b1 views with identical labels). Row-level random split would
    # leak the same fruit into both train and test and inflate val metrics.
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


def make_sampler(rows, num_classes=3):
    labels = np.array([r["grade"] for r in rows])
    counts = np.bincount(labels, minlength=num_classes)
    w_per_class = 1.0 / np.maximum(counts, 1)
    sample_w = w_per_class[labels]
    return WeightedRandomSampler(sample_w.tolist(), len(sample_w), replacement=True)


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


def r2_score(y_true, y_pred):
    mean_y = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - mean_y) ** 2).sum()
    return 0.0 if ss_tot < 1e-8 else float(1 - ss_res / ss_tot)


def reg_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": r2_score(y_true, y_pred),
        "mape": float((np.abs(err) / np.maximum(np.abs(y_true), 1e-8)).mean() * 100),
    }


@torch.no_grad()
def evaluate(model, loader, device, task, num_classes=3):
    model.eval()
    yt, yp, fids = [], [], []
    for imgs, labels, fruit_ids in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        if task == "cls":
            yp.extend(out.argmax(1).cpu().tolist())
            yt.extend(labels.tolist())
        else:
            yp.extend(out.squeeze(1).cpu().tolist())
            yt.extend([float(v) for v in labels.tolist()])
        fids.extend(fruit_ids)

    yt_arr = np.array(yt, dtype=float)
    yp_arr = np.array(yp, dtype=float)

    if task == "cls":
        yt_i = yt_arr.astype(int)
        yp_i = yp_arr.astype(int)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(yt_i, yp_i):
            cm[t, p] += 1
        return {"acc": float((yt_i == yp_i).mean()), "confusion": cm.tolist()}

    m = reg_metrics(yt_arr, yp_arr)
    # Aggregate per-view predictions back to per-fruit predictions by averaging
    # the 3 views. This mirrors deployment (sorter sees 3 cameras per fruit) and
    # acts as a free noise-reducing ensemble. Expect fruit_* to beat sample-level.
    preds = defaultdict(list)
    trues = {}
    for fid, t, p in zip(fids, yt_arr, yp_arr):
        preds[fid].append(p)
        trues[fid] = t
    f_true = np.array([trues[k] for k in preds])
    f_pred = np.array([np.mean(v) for v in preds.values()])
    fm = reg_metrics(f_true, f_pred)
    m["fruit_mae"] = fm["mae"]
    m["fruit_rmse"] = fm["rmse"]
    m["fruit_r2"] = fm["r2"]
    m["fruit_mape"] = fm["mape"]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="cls", choices=["cls", "reg"])
    ap.add_argument("--target", default=None,
                    help="cls: grade (default). reg: weight|height|max_w|min_w|brix (default=weight)")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--model", default="resnet50", choices=["resnet50", "densenet201"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5,
                    help="early stop after this many epochs without improvement (0 disables)")
    ap.add_argument("--batch-size", type=int, default=16)
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

    if args.target is None:
        args.target = "grade" if args.task == "cls" else "weight"
    if args.task == "cls" and args.target not in CLS_TARGETS:
        ap.error(f"cls task requires target in {sorted(CLS_TARGETS)}")
    if args.task == "reg" and args.target not in REG_TARGETS:
        ap.error(f"reg task requires target in {sorted(REG_TARGETS)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")
    print(f"task: {args.task}  target: {args.target}")

    rows = load_label_csv(args.label)
    if args.task == "reg":
        rows = [r for r in rows if r.get(args.target) is not None]
    print(f"loaded {len(rows)} rows")
    if args.task == "cls":
        print(f"  class dist: {dict(sorted(Counter(r['grade'] for r in rows).items()))}")
    else:
        vals = np.array([r[args.target] for r in rows])
        print(f"  {args.target} stats: min={vals.min():.2f} max={vals.max():.2f} "
              f"mean={vals.mean():.2f} std={vals.std():.2f}")

    train_rows, test_rows = split_by_fruit(rows, test_ratio=args.test_ratio, seed=args.seed)
    print(f"split (by fruit_id): train={len(train_rows)}  test={len(test_rows)}")

    train_tf, test_tf = build_transforms(args.img_size, args.crop_size)
    is_float = args.task == "reg"
    train_ds = FruitDataset(train_rows, args.repo_root, target_key=args.target,
                            target_is_float=is_float, transform=train_tf, mask_bg=args.mask_bg)
    test_ds = FruitDataset(test_rows, args.repo_root, target_key=args.target,
                           target_is_float=is_float, transform=test_tf, mask_bg=args.mask_bg)

    pin = device.type == "cuda"
    if args.task == "cls":
        sampler = make_sampler(train_rows)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    num_outputs = 3 if args.task == "cls" else 1
    model = build_model(args.model, num_outputs=num_outputs,
                        pretrained=not args.no_pretrained).to(device)

    if args.task == "cls":
        criterion = nn.CrossEntropyLoss()
        key_metric, higher_is_better = "acc", True
        best_metric = -float("inf")
    else:
        criterion = nn.SmoothL1Loss()
        key_metric, higher_is_better = "mae", False
        best_metric = float("inf")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    stagnant = 0
    stopped_at = args.epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device)
            # MPS does not support float64; DataLoader collates Python floats into
            # float64 tensors by default, so cast to float32 before the device copy.
            labels = labels.float().to(device) if args.task == "reg" else labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            if args.task == "reg":
                loss = criterion(out.squeeze(1), labels)
            else:
                loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            pbar.set_postfix(loss=f"{run_loss/max(n,1):.4f}")
        scheduler.step()
        train_loss = run_loss / max(n, 1)

        metrics = evaluate(model, test_loader, device, args.task)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if args.task == "cls":
            print(f"epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  val_acc={metrics['acc']*100:.2f}%")
            print("  confusion (rows=true, cols=pred):")
            for row in metrics["confusion"]:
                print("   ", row)
        else:
            print(f"epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
                  f"MAE={metrics['mae']:.3f}  R2={metrics['r2']:.3f}  MAPE={metrics['mape']:.2f}%")
            print(f"  fruit-avg: MAE={metrics['fruit_mae']:.3f}  R2={metrics['fruit_r2']:.3f}  "
                  f"MAPE={metrics['fruit_mape']:.2f}%")

        cur = metrics[key_metric]
        improved = cur > best_metric if higher_is_better else cur < best_metric
        if improved:
            best_metric = cur
            stagnant = 0
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "metrics": metrics,
            }, out_dir / "best.pt")
            print(f"  -> saved best ({key_metric}={cur:.4f})")
        else:
            stagnant += 1
            if args.patience > 0 and stagnant >= args.patience:
                stopped_at = epoch
                print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump({"best_metric": best_metric, "key_metric": key_metric,
                   "stopped_at": stopped_at, "args": vars(args), "history": history},
                  f, indent=2, default=str)
    print(f"done. best {key_metric}={best_metric:.4f}  stopped@{stopped_at}  -> {out_dir}")


if __name__ == "__main__":
    main()

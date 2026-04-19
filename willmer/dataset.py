#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUMERIC_FIELDS = ("brix", "weight", "height", "max_w", "min_w")


def segment_fruit(img_rgb):
    # HSV background removal tuned for the original sorter rig (blue-tinted backdrop
    # + white/gray surfaces). Masks out: Hue >= 90 (blue side) and Saturation < 90
    # (washed-out surfaces). New capture setups with different backgrounds will need
    # the thresholds re-tuned; disabled by default in training.
    x = cv2.GaussianBlur(img_rgb, (15, 15), 0)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    bg_blue = cv2.inRange(x, (90, 0, 0), (255, 255, 255))
    bg_gray = cv2.inRange(x, (0, 0, 0), (255, 90, 255))
    keep = cv2.bitwise_not(cv2.bitwise_or(bg_blue, bg_gray))
    return cv2.bitwise_and(img_rgb, img_rgb, mask=keep)


def load_label_csv(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if "grade" in r:
            r["grade"] = None if r["grade"] == "" else int(r["grade"])
        for k in NUMERIC_FIELDS:
            if k in r:
                if r[k] == "":
                    r[k] = None
                else:
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        r[k] = None
    return rows


class FruitDataset(Dataset):
    def __init__(self, rows, repo_root, target_key="grade", target_is_float=False,
                 transform=None, mask_bg=False):
        self.rows = rows
        self.repo_root = Path(repo_root)
        self.target_key = target_key
        self.target_is_float = target_is_float
        self.transform = transform
        self.mask_bg = mask_bg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = self.repo_root / r["img_path"]
        with open(path, "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))
        if self.mask_bg:
            img = segment_fruit(img)
        if self.transform is not None:
            img = self.transform(img)
        raw = r[self.target_key]
        target = float(raw) if self.target_is_float else int(raw)
        # fruit_id lets the eval loop average the 3 views (t1/t2/b1) of a single
        # fruit back together, since the same fruit is stored as 3 separate rows.
        fruit_id = f"{r['date']}__{r['fruit_idx']}"
        return img, target, fruit_id


class MultiViewFruitDataset(Dataset):
    """Groups rows into one sample per fruit; returns a stacked (3, C, H, W) tensor.

    Views are ordered t1 -> t2 -> b1 so the downstream model can learn per-view
    features by position. Fruits missing any of the three views, or missing the
    target field in any row, are dropped.
    """

    VIEW_ORDER = ("t1", "t2", "b1")

    def __init__(self, rows, repo_root, target_key="weight", target_is_float=True,
                 transform=None, mask_bg=False):
        self.repo_root = Path(repo_root)
        self.target_key = target_key
        self.target_is_float = target_is_float
        self.transform = transform
        self.mask_bg = mask_bg

        groups = defaultdict(dict)
        for r in rows:
            groups[(r["date"], r["fruit_idx"])][r["view"]] = r

        self.fruits = []
        for views in groups.values():
            if not all(v in views for v in self.VIEW_ORDER):
                continue
            if any(views[v].get(target_key) is None for v in self.VIEW_ORDER):
                continue
            self.fruits.append(views)

    def __len__(self):
        return len(self.fruits)

    def _load(self, row):
        path = self.repo_root / row["img_path"]
        with open(path, "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))
        if self.mask_bg:
            img = segment_fruit(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        views = self.fruits[idx]
        imgs = torch.stack([self._load(views[v]) for v in self.VIEW_ORDER], dim=0)
        r0 = views[self.VIEW_ORDER[0]]
        raw = r0[self.target_key]
        target = float(raw) if self.target_is_float else int(raw)
        fruit_id = f"{r0['date']}__{r0['fruit_idx']}"
        return imgs, target, fruit_id


class MultiViewMultiTaskDataset(Dataset):
    """Multi-view dataset that returns multiple regression targets per fruit.

    Returns (imgs, targets, fruit_id) where `targets` is a float tensor of
    shape (len(target_keys),) in the same order as passed in. Fruits missing
    any view or any target field are dropped.
    """

    VIEW_ORDER = ("t1", "t2", "b1")

    def __init__(self, rows, repo_root, target_keys, transform=None, mask_bg=False):
        self.repo_root = Path(repo_root)
        self.target_keys = list(target_keys)
        self.transform = transform
        self.mask_bg = mask_bg

        groups = defaultdict(dict)
        for r in rows:
            groups[(r["date"], r["fruit_idx"])][r["view"]] = r

        self.fruits = []
        for views in groups.values():
            if not all(v in views for v in self.VIEW_ORDER):
                continue
            r0 = views[self.VIEW_ORDER[0]]
            if any(r0.get(k) is None for k in self.target_keys):
                continue
            self.fruits.append(views)

    def __len__(self):
        return len(self.fruits)

    def _load(self, row):
        path = self.repo_root / row["img_path"]
        with open(path, "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))
        if self.mask_bg:
            img = segment_fruit(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        views = self.fruits[idx]
        imgs = torch.stack([self._load(views[v]) for v in self.VIEW_ORDER], dim=0)
        r0 = views[self.VIEW_ORDER[0]]
        targets = torch.tensor([float(r0[k]) for k in self.target_keys],
                               dtype=torch.float32)
        fruit_id = f"{r0['date']}__{r0['fruit_idx']}"
        return imgs, targets, fruit_id

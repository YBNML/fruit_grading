#!/usr/bin/env python3
"""
Build label_peach.csv / label_tangerine.csv from DB/.

Quality grade (0/1/2 = 하/중/상) is derived from per-crop `grade_field`
tertiles as a proxy -- no explicit quality label exists in the raw data.
Switch `--grade-field` to use height/max_w/etc instead of weight.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "DB"
OUT_DIR = Path(__file__).resolve().parent

FIELDS = [
    "img_path", "crop", "date", "view", "fruit_idx",
    "brix", "weight", "height", "max_w", "min_w", "grade",
]


def _split_path(raw_path):
    """Pull (date, view, filename) out of an absolute path that contains 'database'."""
    # Raw labels use absolute paths from the capture machine (e.g. /home/ybnml/database/...
    # or /database/...). Anchor on the 'database' segment so we can remap regardless of
    # which machine's prefix was recorded.
    parts = Path(raw_path).parts
    try:
        i = parts.index("database")
    except ValueError:
        return None
    tail = parts[i + 1:]
    if len(tail) != 3:
        return None
    return tail  # (date, view, filename)


def load_peach():
    root = DB / "peach_천중도"
    rows = []
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        jp = date_dir / f"{date_dir.name}_labeling_01.json"
        if not jp.exists():
            continue
        with open(jp) as f:
            d = json.load(f)
        for info, meas in zip(d["image_info"], d["measurement_info"]):
            tail = _split_path(info["image_path"])
            if tail is None:
                continue
            date, view, fname = tail
            if view not in ("t1", "t2", "b1"):
                continue
            rel = (root / date / view / fname).relative_to(REPO)
            try:
                rows.append({
                    "img_path": str(rel),
                    "crop": "peach",
                    "date": date,
                    "view": view,
                    "fruit_idx": Path(fname).stem.split("_")[1],
                    "brix": float(meas["brix"]),
                    "weight": float(meas["weight"]),
                    "height": float(meas["fruit_height"]),
                    "max_w": float(meas["fruit_max_width"]),
                    "min_w": float(meas["fruit_min_width"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_tangerine():
    root = DB / "귤_황금향"
    rows = []
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cp = date_dir / "labeling.csv"
        if not cp.exists():
            continue
        data = None
        for enc in ("utf-8", "cp949"):
            try:
                with open(cp, encoding=enc) as f:
                    data = list(csv.reader(f))
                break
            except UnicodeDecodeError:
                continue
        if data is None:
            continue
        for row in data[1:]:
            if len(row) < 7 or not row[0]:
                continue
            tail = _split_path(row[0])
            if tail is None:
                continue
            date, view, fname = tail
            if view not in ("t1", "t2", "b1"):
                continue
            rel = (root / date / view / fname).relative_to(REPO)
            try:
                rows.append({
                    "img_path": str(rel),
                    "crop": "tangerine",
                    "date": date,
                    "view": view,
                    "fruit_idx": Path(fname).stem.split("_")[1],
                    "brix": float(row[1]),
                    "weight": float(row[2]),
                    "height": float(row[3]),
                    "max_w": float(row[4]),
                    "min_w": float(row[5]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def filter_existing(rows):
    return [r for r in rows if (REPO / r["img_path"]).exists()]


def assign_grade(rows, field):
    # No explicit quality label exists in the raw data; we derive 3 classes as
    # per-crop tertiles of a size-related field. Keeps class distribution balanced
    # by construction. Swap `field` to choose the proxy (weight/height/max_w/...).
    vals = sorted(r[field] for r in rows)
    n = len(vals)
    if n < 3:
        raise RuntimeError(f"not enough rows to compute tertiles: {n}")
    t1, t2 = vals[n // 3], vals[2 * n // 3]
    for r in rows:
        v = r[field]
        r["grade"] = 0 if v < t1 else (1 if v < t2 else 2)
    return t1, t2


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grade-field", default="weight",
                    choices=["weight", "height", "max_w", "min_w", "brix"])
    ap.add_argument("--crops", nargs="+", default=["peach", "tangerine"],
                    choices=["peach", "tangerine"])
    args = ap.parse_args()

    loaders = {"peach": load_peach, "tangerine": load_tangerine}
    for crop in args.crops:
        raw = loaders[crop]()
        rows = filter_existing(raw)
        if not rows:
            print(f"[{crop}] no rows with existing images - skipped")
            continue
        t1, t2 = assign_grade(rows, field=args.grade_field)
        out = OUT_DIR / f"label_{crop}.csv"
        write_csv(rows, out)
        dist = Counter(r["grade"] for r in rows)
        print(f"[{crop}] total={len(raw)}  exists={len(rows)}  "
              f"{args.grade_field} tertiles=({t1:.2f}, {t2:.2f})  "
              f"grade dist={dict(sorted(dist.items()))}  -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

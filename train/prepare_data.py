"""
Prepare the HAM10000 dataset into the train/val folder structure expected by
train.py.

Download HAM10000 from Kaggle first:
    https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

You need:
    HAM10000_metadata.csv
    HAM10000_images_part_1/  and  HAM10000_images_part_2/   (the .jpg files)

Usage:
    python train/prepare_data.py \
        --metadata HAM10000_metadata.csv \
        --images HAM10000_images_part_1 HAM10000_images_part_2 \
        --out data/ham10000 --val-split 0.15
"""

import argparse
import csv
import os
import random
import shutil

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def find_image(image_id, image_dirs):
    for d in image_dirs:
        path = os.path.join(d, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="data/ham10000")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Read metadata: image_id -> dx (diagnosis label).
    rows = []
    with open(args.metadata, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["dx"] in CLASS_NAMES:
                rows.append((row["image_id"], row["dx"]))

    random.shuffle(rows)

    # Create folders.
    for split in ("train", "val"):
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)

    n_val = int(len(rows) * args.val_split)
    copied, missing = 0, 0
    for i, (image_id, dx) in enumerate(rows):
        src = find_image(image_id, args.images)
        if src is None:
            missing += 1
            continue
        split = "val" if i < n_val else "train"
        dst = os.path.join(args.out, split, dx, image_id + ".jpg")
        shutil.copy(src, dst)
        copied += 1

    print(f"Copied {copied} images ({missing} missing). Output: {args.out}")


if __name__ == "__main__":
    main()

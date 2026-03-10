"""
Write ImageNet-1k from HuggingFace Hub to FFCV .beton format.

Run this ONCE before starting any training job.  The resulting files are
reused by all three training scripts.

Usage
-----
python src/train/write_imagenet_ffcv.py
python src/train/write_imagenet_ffcv.py \\
    --train-out data/imagenet_ffcv/train.beton \\
    --val-out   data/imagenet_ffcv/val.beton   \\
    --num-workers 16

The default output directory is <project-root>/data/imagenet_ffcv/.
Writing full ImageNet-1k (1.28 M train + 50 k val) typically takes
30–90 minutes depending on HF download speed and disk throughput.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

# Common module lives in the same directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from resnet_imagenet_common import HFImageNetRawDataset, BASE_DIR


def write_split(
    split: str,
    out_path: Path,
    num_workers: int,
    token: str | None,
    max_resolution: int,
    jpeg_quality: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  {out_path} already exists — skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  Writing split='{split}' -> {out_path}")
    print(f"{'='*60}")

    # HF 'validation' split maps to val in our naming
    hf_split = "validation" if split == "val" else "train"
    ds = HFImageNetRawDataset(split=hf_split, token=token)

    writer = DatasetWriter(
        str(out_path),
        {
            "image": RGBImageField(
                max_resolution=max_resolution,
                jpeg_quality=jpeg_quality,
            ),
            "label": IntField(),
        },
        num_workers=num_workers,
    )
    writer.from_indexed_dataset(ds)
    print(f"  Done — {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert HF ImageNet-1k to FFCV .beton files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_dir = str(BASE_DIR / "data" / "imagenet_ffcv")
    p.add_argument("--train-out",      type=str,
                   default=str(Path(default_dir) / "train.beton"))
    p.add_argument("--val-out",        type=str,
                   default=str(Path(default_dir) / "val.beton"))
    p.add_argument("--num-workers",    type=int, default=16,
                   help="Parallel workers for dataset writing")
    p.add_argument("--max-resolution", type=int, default=500,
                   help="Maximum stored image dimension (px); FFCV decodes to 224 at runtime")
    p.add_argument("--jpeg-quality",   type=int, default=90,
                   help="JPEG quality for stored images")
    p.add_argument("--hf-token",       type=str, default=None,
                   help="HF access token (falls back to HF_TOKEN env var)")
    p.add_argument("--skip-train",     action="store_true")
    p.add_argument("--skip-val",       action="store_true")
    args = p.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or None

    if not args.skip_train:
        write_split("train", Path(args.train_out),
                    args.num_workers, token,
                    args.max_resolution, args.jpeg_quality)

    if not args.skip_val:
        write_split("val", Path(args.val_out),
                    args.num_workers, token,
                    args.max_resolution, args.jpeg_quality)

    print("\nAll splits written successfully.")
    print(f"  train : {args.train_out}")
    print(f"  val   : {args.val_out}")


if __name__ == "__main__":
    main()

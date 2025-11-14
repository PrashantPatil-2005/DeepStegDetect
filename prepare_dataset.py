"""
Preprocess images: resize, convert to 3-channel, and save them into a
temporary structure before creating train/val/test splits.

Usage:
    python prepare_dataset.py --cover dataset/BOSSBase --stego dataset/SUNIWARD --out processed --size 224
"""

import argparse
from pathlib import Path
from utils import resize_and_save, create_splits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover', required=True, help='Path to cover images directory')
    parser.add_argument('--stego', required=True, help='Path to stego images directory')
    parser.add_argument('--out', required=True, help='Output folder for processed dataset')
    parser.add_argument('--size', type=int, default=224, help='Resize dimension (default: 224)')
    return parser.parse_args()


def preprocess_and_save_pairs(cover_dir, stego_dir, tmp_dir, size=224):
    """
    Resize cover and stego images, convert to 3-channel RGB,
    and store temporary files with .png extension.
    """
    tmp_cover = Path(tmp_dir) / 'cover'
    tmp_stego = Path(tmp_dir) / 'stego'

    tmp_cover.mkdir(parents=True, exist_ok=True)
    tmp_stego.mkdir(parents=True, exist_ok=True)

    cover_dir = Path(cover_dir)
    stego_dir = Path(stego_dir)

    cover_files = sorted([p for p in cover_dir.iterdir() if p.is_file()])

    for c in cover_files:
        stem = c.stem
        # match cover → corresponding stego image
        candidates = list(stego_dir.glob(stem + '*'))
        if not candidates:
            continue  # no stego version found
        s = candidates[0]

        dst_cover = tmp_cover / (stem + '.png')
        dst_stego = tmp_stego / (s.stem + '.png')

        # resize & save
        resize_and_save(c, dst_cover, size)
        resize_and_save(s, dst_stego, size)

    print("Preprocessing complete.")


if __name__ == '__main__':
    args = parse_args()

    tmp_dir = 'tmp_processed'

    # Step 1: preprocess images & resize them
    preprocess_and_save_pairs(args.cover, args.stego, tmp_dir, args.size)

    # Step 2: create train/val/test splits
    create_splits(
        cover_dir=f'{tmp_dir}/cover',
        stego_dir=f'{tmp_dir}/stego',
        out_dir=args.out
    )

    print("✅ Dataset prepared successfully at:", args.out)

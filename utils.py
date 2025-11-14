import os
import random
from pathlib import Path
from PIL import Image


def read_pgm(path):
    """Read PGM image and return PIL Image."""
    return Image.open(path)


def ensure_three_channel(img_pil):
    """
    Convert grayscale image (L mode) to 3-channel RGB.
    """
    if img_pil.mode == 'L':
        return Image.merge('RGB', (img_pil, img_pil, img_pil))
    elif img_pil.mode == 'RGB':
        return img_pil
    else:
        return img_pil.convert('RGB')


def resize_and_save(src_path, dst_path, size):
    """
    Resize image to (size x size), convert to 3-channel, then save.
    """
    img = read_pgm(src_path)
    img = ensure_three_channel(img)
    img = img.resize((size, size), Image.LANCZOS)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path)


def create_splits(cover_dir, stego_dir, out_dir,
                  train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create train/val/test folders with cover and stego images.
    
    Folder structure created:
        out_dir/
            train/cover/
            train/stego/
            val/cover/
            val/stego/
            test/cover/
            test/stego/
    """
    random.seed(seed)

    cover_dir = Path(cover_dir)
    stego_dir = Path(stego_dir)
    out_dir = Path(out_dir)

    cover_files = sorted([p for p in cover_dir.iterdir() if p.is_file()])

    # Match cover and stego by filename stem
    pairs = []
    for cover_path in cover_files:
        stem = cover_path.stem
        candidates = list(stego_dir.glob(stem + '*'))  # handles "_stego" suffix too
        if candidates:
            pairs.append((cover_path, candidates[0]))

    random.shuffle(pairs)

    N = len(pairs)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    splits = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:]
    }

    # Create output structure
    for split_name, items in splits.items():
        for cover_p, stego_p in items:
            dst_cover = out_dir / split_name / 'cover' / cover_p.name
            dst_stego = out_dir / split_name / 'stego' / stego_p.name

            # Make directories
            dst_cover.parent.mkdir(parents=True, exist_ok=True)
            dst_stego.parent.mkdir(parents=True, exist_ok=True)

            # Copy files
            if not dst_cover.exists():
                os.system(f'cp "{cover_p}" "{dst_cover}"')
            if not dst_stego.exists():
                os.system(f'cp "{stego_p}" "{dst_stego}"')

    print("Created splits with sizes:", {k: len(v) for k, v in splits.items()})

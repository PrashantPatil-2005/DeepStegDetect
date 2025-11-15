import os
import random
import shutil
from pathlib import Path
from PIL import Image


def read_pgm(path):
    return Image.open(path)


def ensure_three_channel(img_pil):
    if img_pil.mode == 'L':
        return Image.merge('RGB', (img_pil, img_pil, img_pil))
    elif img_pil.mode == 'RGB':
        return img_pil
    else:
        return img_pil.convert('RGB')


def resize_and_save(src_path, dst_path, size):
    img = read_pgm(src_path)
    img = ensure_three_channel(img)
    img = img.resize((size, size), Image.LANCZOS)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path)


def create_splits(cover_dir, stego_dir, out_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)

    cover_files = sorted([p for p in Path(cover_dir).iterdir() if p.is_file()])

    pairs = []
    for c in cover_files:
        stem = c.stem
        candidates = list(Path(stego_dir).glob(stem + '*'))
        if candidates:
            pairs.append((c, candidates[0]))

    random.shuffle(pairs)

    N = len(pairs)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    splits = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:]
    }

    for split, items in splits.items():
        for cover_p, stego_p in items:
            dst_cover = Path(out_dir) / split / 'cover' / cover_p.name
            dst_stego = Path(out_dir) / split / 'stego' / stego_p.name

            dst_cover.parent.mkdir(parents=True, exist_ok=True)
            dst_stego.parent.mkdir(parents=True, exist_ok=True)

            # Windows-compatible copying
            shutil.copy(str(cover_p), str(dst_cover))
            shutil.copy(str(stego_p), str(dst_stego))

    print('Created splits:', {k: len(v) for k, v in splits.items()})

"""
Wrapper script to call the external S-UNIWARD steganography executable.

Usage:
    python generate_stego.py --bdir dataset/BOSSBase --odir dataset/SUNIWARD --bpp 0.2

Note:
    - This script **does NOT implement S-UNIWARD**.
    - It simply calls the external native executable (Linux/Mac) or `.exe` (Windows).
    - You MUST place the executable inside: stego_tools/S-UNIWARD/
"""

import os
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bdir', required=True, help='folder with cover images (.pgm)')
    parser.add_argument('--odir', required=True, help='output folder for stego images')
    parser.add_argument('--bpp', type=float, default=0.2, help='payload in bits per pixel')
    return parser.parse_args()


def find_suniward_executable():
    # Common locations to search
    candidates = [
        Path('stego_tools/S-UNIWARD/S-UNIWARD'),
        Path('stego_tools/S-UNIWARD/S-UNIWARD.exe'),
        Path('stego_tools/suniward/S-UNIWARD'),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main():
    args = parse_args()
    bdir = Path(args.bdir)
    odir = Path(args.odir)
    odir.mkdir(parents=True, exist_ok=True)

    exe = find_suniward_executable()

    if exe is None:
        print("❌ ERROR: S-UNIWARD executable not found.")
        print("Place the executable inside: stego_tools/S-UNIWARD/")
        print("Skipping generation.")
        return

    img_files = sorted([
        p for p in bdir.iterdir()
        if p.suffix.lower() in ['.pgm', '.png', '.jpg']
    ])

    print(f"📁 Found {len(img_files)} images in: {bdir}")

    for i, img in enumerate(img_files, 1):
        out_name = odir / (img.stem + '_stego' + img.suffix)

        # Command to call S-UNIWARD executable
        cmd = [exe, str(img), str(out_name), str(args.bpp)]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {img}: {e}")

        # Progress update
        if i % 100 == 0:
            print(f"✔ Processed {i}/{len(img_files)}")

    print("🎉 Stego image generation complete!")


if __name__ == '__main__':
    main()

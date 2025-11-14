import argparse
from pathlib import Path
from python_suniward import embed_suniward_python


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bdir", required=True, help="folder with cover images")
    p.add_argument("--odir", required=True, help="output stego folder")
    p.add_argument("--bpp", type=float, default=0.2, help="payload bits per pixel")
    return p.parse_args()


def main():
    args = parse_args()

    bdir = Path(args.bdir)
    odir = Path(args.odir)
    odir.mkdir(parents=True, exist_ok=True)

    cover_files = [p for p in bdir.iterdir() if p.suffix.lower() in [".pgm", ".png", ".jpg"]]

    print(f"Found {len(cover_files)} images")

    for i, img in enumerate(cover_files, 1):
        out_path = odir / (img.stem + "_stego.png")
        embed_suniward_python(img, out_path, args.bpp)

        if i % 50 == 0:
            print(f"Processed {i}/{len(cover_files)}")

    print("Done. Stego images generated!")


if __name__ == "__main__":
    main()

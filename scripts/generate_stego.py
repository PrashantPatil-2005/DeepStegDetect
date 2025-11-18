"""
Generate stego images using various steganography algorithms.
Supports S-UNIWARD (via executable or Python implementation), WOW, HILL, and HUGO.

Usage:
    # Generate S-UNIWARD at 0.2 bpp
    python scripts/generate_stego.py --algo suniward --payload 0.2
    
    # Generate all algorithms at 0.4 bpp
    python scripts/generate_stego.py --algo all --payload 0.4
    
    # Generate with custom input/output directories
    python scripts/generate_stego.py --algo suniward --payload 0.2 --input dataset/BOSSBase --output dataset/SUNIWARD
"""

import os
import argparse
import subprocess
from pathlib import Path
import sys

# Add scripts directory to path for python_suniward
sys.path.append(str(Path(__file__).parent))
from python_suniward import embed_suniward_python


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stego images using various algorithms")
    parser.add_argument('--algo', type=str, default='suniward', 
                        choices=['suniward', 'wow', 'hill', 'hugo', 'all'],
                        help='Steganography algorithm to use')
    parser.add_argument('--payload', type=float, default=0.2, 
                        help='Payload in bits per pixel (bpp)')
    parser.add_argument('--input', type=str, default='dataset/BOSSBase',
                        help='Input directory with cover images')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for stego images (auto-generated if not specified)')
    return parser.parse_args()


def find_suniward_executable():
    """Find S-UNIWARD executable"""
    candidates = [
        Path('stego_tools/S-UNIWARD/S-UNIWARD'),
        Path('stego_tools/S-UNIWARD/S-UNIWARD.exe'),
        Path('stego_tools/suniward/S-UNIWARD'),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def generate_suniward(bdir, odir, bpp, use_executable=True):
    """Generate S-UNIWARD stego images"""
    bdir = Path(bdir)
    odir = Path(odir)
    odir.mkdir(parents=True, exist_ok=True)
    
    # Try executable first
    if use_executable:
        exe = find_suniward_executable()
        if exe:
            print(f"🔧 Using S-UNIWARD executable: {exe}")
            img_files = sorted([
                p for p in bdir.iterdir()
                if p.suffix.lower() in ['.pgm', '.png', '.jpg', '.bmp']
            ])
            
            for i, img in enumerate(img_files, 1):
                out_name = odir / (img.stem + '_stego' + img.suffix)
                cmd = [exe, str(img), str(out_name), str(bpp)]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Error with executable, falling back to Python implementation")
                    use_executable = False
                    break
                
                if i % 100 == 0:
                    print(f"✔ Processed {i}/{len(img_files)}")
            
            if use_executable:
                print(f"✅ S-UNIWARD generation complete (executable)")
                return
    
    # Fall back to Python implementation
    print(f"🐍 Using Python S-UNIWARD implementation")
    img_files = sorted([
        p for p in bdir.iterdir()
        if p.suffix.lower() in ['.pgm', '.png', '.jpg', '.bmp']
    ])
    
    for i, img in enumerate(img_files, 1):
        out_name = odir / (img.stem + '_stego' + img.suffix)
        try:
            embed_suniward_python(str(img), str(out_name), bpp)
        except Exception as e:
            print(f"❌ Error processing {img}: {e}")
            continue
        
        if i % 100 == 0:
            print(f"✔ Processed {i}/{len(img_files)}")
    
    print(f"✅ S-UNIWARD generation complete (Python)")


def generate_wow(bdir, odir, bpp):
    """Generate WOW stego images (placeholder - requires implementation)"""
    print(f"⚠️  WOW algorithm not yet implemented. Skipping...")
    print(f"   To implement WOW, you need the WOW executable or Python implementation.")
    print(f"   Place it in: stego_tools/WOW/")


def generate_hill(bdir, odir, bpp):
    """Generate HILL stego images (placeholder - requires implementation)"""
    print(f"⚠️  HILL algorithm not yet implemented. Skipping...")
    print(f"   To implement HILL, you need the HILL executable or Python implementation.")
    print(f"   Place it in: stego_tools/HILL/")


def generate_hugo(bdir, odir, bpp):
    """Generate HUGO stego images (placeholder - requires implementation)"""
    print(f"⚠️  HUGO algorithm not yet implemented. Skipping...")
    print(f"   To implement HUGO, you need the HUGO executable or Python implementation.")
    print(f"   Place it in: stego_tools/HUGO/")


def main():
    args = parse_args()
    
    # Determine output directory
    if args.output:
        base_output = args.output
    else:
        algo_name = args.algo.upper() if args.algo != 'all' else 'ALL'
        base_output = f"dataset/{algo_name}"
    
    bdir = Path(args.input)
    if not bdir.exists():
        print(f"❌ Error: Input directory not found: {bdir}")
        return
    
    print(f"\n{'='*60}")
    print(f"🎨 Stego Image Generation")
    print(f"{'='*60}")
    print(f"📁 Input directory: {bdir}")
    print(f"📊 Payload: {args.payload} bpp")
    print(f"🔧 Algorithm(s): {args.algo}")
    print(f"{'='*60}\n")
    
    algorithms = {
        'suniward': generate_suniward,
        'wow': generate_wow,
        'hill': generate_hill,
        'hugo': generate_hugo
    }
    
    if args.algo == 'all':
        for algo_name, algo_func in algorithms.items():
            print(f"\n🔄 Generating {algo_name.upper()} stego images...")
            odir = Path(base_output) / algo_name.upper()
            algo_func(bdir, odir, args.payload)
    else:
        print(f"\n🔄 Generating {args.algo.upper()} stego images...")
        odir = Path(base_output)
        algorithms[args.algo](bdir, odir, args.payload)
    
    print(f"\n{'='*60}")
    print(f"🎉 Stego image generation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

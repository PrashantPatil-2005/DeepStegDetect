import numpy as np
from PIL import Image
from pathlib import Path


def compute_distortion(img):
    """
    Approximate UNIWARD distortion map:
    - Use high-pass filters to detect textured regions
    - High texture => low cost
    - Smooth areas => high cost
    """

    img = np.array(img).astype(np.float32)

    # High-pass filter
    hp = np.array([[1, -1],
                   [-1, 1]], dtype=np.float32)

    # Convolve (fast 2D correlation)
    from scipy.signal import convolve2d
    residual = np.abs(convolve2d(img, hp, mode="same"))

    # Avoid zero distortion
    residual += 1e-6

    # Cost = 1 / (texture + ε)
    cost = 1.0 / residual

    # Normalize between [0,1]
    cost = cost / cost.max()

    return cost


def embed_bits(img, payload_bits, cost):
    """
    Greedy adaptive embedding:
    - Flatten cost map
    - Sort pixels by lowest cost
    - Flip LSB of selected pixels to embed payload
    """

    img = np.array(img).astype(np.uint8)

    flat_cost = cost.flatten()
    indices = np.argsort(flat_cost)  # lowest cost → highest priority

    flat_img = img.flatten()

    # LSB embedding
    for i, bit in zip(indices, payload_bits):
        flat_img[i] = (flat_img[i] & 0xFE) | bit  # set LSB = bit

    return flat_img.reshape(img.shape)


def generate_payload(size, bpp=0.2):
    """
    bpp = bits per pixel
    total bits = width * height * bpp
    """
    n_bits = int(size * size * bpp)
    return np.random.randint(0, 2, n_bits).astype(np.uint8)


def embed_suniward_python(cover_path, stego_path, bpp=0.2):
    """
    Full pipeline:
    1. Load cover
    2. Compute distortion map
    3. Generate payload
    4. Embed adaptively
    """
    cover = Image.open(cover_path).convert("L")  # grayscale
    size = cover.size[0]

    # Step 1: Distortion
    cost_map = compute_distortion(cover)

    # Step 2: Payload bits
    payload_bits = generate_payload(size, bpp)

    # Step 3: Adaptive embed
    stego = embed_bits(cover, payload_bits, cost_map)

    # Save
    Image.fromarray(stego.astype(np.uint8)).save(stego_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cover", required=True)
    parser.add_argument("--stego", required=True)
    parser.add_argument("--bpp", type=float, default=0.2)
    args = parser.parse_args()

    embed_suniward_python(args.cover, args.stego, args.bpp)

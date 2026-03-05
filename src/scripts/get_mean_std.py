"""
Compute per-channel mean and std of a sampled subset of .tif files in a folder.
Zero-valued pixels are excluded from all statistics.

Usage:
    python src/scripts/get_mean_std.py \
        --input_dir /path/to/tif_folder \
        --n 500 \
        [--seed 42] \
        [--log]
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import rasterio


def compute_mean_std(
    input_dir: str,
    n: int,
    seed: int | None = None,
) -> dict[str, list[float]]:
    """
    Sample `n` .tif files from `input_dir` and compute per-channel mean and std,
    excluding zero-valued pixels.

    Returns a dict with keys "mean" and "std", each a list of floats (one per channel).
    """
    tif_files = sorted(Path(input_dir).glob("*.tif"))
    if len(tif_files) == 0:
        raise ValueError(f"No .tif files found in {input_dir}")

    if n > len(tif_files):
        logging.warning(
            f"Requested n={n} but only {len(tif_files)} files found. Using all files."
        )
        n = len(tif_files)

    rng = random.Random(seed)
    sampled = rng.sample(tif_files, n)
    logging.info(f"Sampled {n} / {len(tif_files)} files.")

    # Determine number of channels from the first file.
    with rasterio.open(sampled[0]) as ds:
        num_channels = ds.count

    # Accumulators: one entry per channel.
    # Using float64 to reduce precision loss when summing many large values.
    px_sum = np.zeros(num_channels, dtype=np.float64)
    px_sum_sq = np.zeros(num_channels, dtype=np.float64)
    px_count = np.zeros(num_channels, dtype=np.int64)

    for i, path in enumerate(sampled):
        logging.info(f"[{i + 1}/{n}] Reading {path.name}")
        with rasterio.open(path) as ds:
            if ds.count != num_channels:
                logging.warning(
                    f"Skipping {path.name}: expected {num_channels} channels, "
                    f"got {ds.count}."
                )
                continue
            data = ds.read().astype(np.float64)  # shape: (C, H, W)

        for c in range(num_channels):
            band = data[c]
            valid = band[band != 0]
            px_sum[c] += valid.sum()
            px_sum_sq[c] += (valid ** 2).sum()
            px_count[c] += valid.size

    if np.any(px_count == 0):
        zero_chs = np.where(px_count == 0)[0].tolist()
        logging.warning(f"Channels {zero_chs} have no valid (non-zero) pixels.")

    # Avoid division by zero for empty channels.
    safe_count = np.where(px_count > 0, px_count, 1)
    mean = px_sum / safe_count
    # Var(X) = E[X^2] - E[X]^2
    variance = (px_sum_sq / safe_count) - (mean ** 2)
    # Numerical noise can push variance slightly negative; clamp to zero.
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    return {"mean": mean.tolist(), "std": std.tolist(), "n_sampled": n, "px_count": px_count.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std over a sampled set of .tif files."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing .tif files.")
    parser.add_argument("--n", type=int, required=True, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--log", action="store_true", help="Enable INFO logging.")
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results = compute_mean_std(
        input_dir=args.input_dir,
        n=args.n,
        seed=args.seed,
    )

    print(f"\nResults over {results['n_sampled']} sampled images (zero pixels excluded):")
    for c, (m, s, cnt) in enumerate(
        zip(results["mean"], results["std"], results["px_count"])
    ):
        print(f"  Channel {c + 1}: mean={m:.6f}  std={s:.6f}  valid_pixels={cnt:,}")

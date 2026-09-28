import os
import json
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build shared Markov segment-wise maskbanks for Beijing windows.")
    p.add_argument("--windows_dir", type=str, required=True,
                   help="Directory containing val_windows.npy and test_windows.npy")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory to save maskbanks")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="Seeds for repeated runs")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7],
                   help="Masked ratios to generate")
    p.add_argument("--lm", type=int, default=5,
                   help="Average masked segment length for Markov/geometric masking")
    return p.parse_args()


def geometric_run_length(rng, mean_len: int) -> int:
    if mean_len <= 1:
        return 1
    p = 1.0 / float(mean_len)
    return max(1, int(rng.geometric(p)))


def make_1d_markov_mask(length: int, masked_ratio: float, lm: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Returns binary mask of shape [length], where:
      1 = masked/evaluation position
      0 = observed/conditioning position
    """
    masked_ratio = float(masked_ratio)
    masked_ratio = min(max(masked_ratio, 1e-6), 1.0 - 1e-6)

    out = np.zeros(length, dtype=np.uint8)

    # Start masked with probability = target masked ratio
    state = 1 if rng.rand() < masked_ratio else 0
    t = 0
    while t < length:
        run = geometric_run_length(rng, lm)
        end = min(length, t + run)
        out[t:end] = state
        state = 1 - state
        t = end

    # small correction so ratio is closer to requested target
    current = out.mean()
    target_count = int(round(masked_ratio * length))
    current_count = int(out.sum())

    if current_count > target_count:
        ones = np.where(out == 1)[0]
        if len(ones) > 0:
            flip = rng.choice(ones, size=(current_count - target_count), replace=False)
            out[flip] = 0
    elif current_count < target_count:
        zeros = np.where(out == 0)[0]
        if len(zeros) > 0:
            flip = rng.choice(zeros, size=(target_count - current_count), replace=False)
            out[flip] = 1

    return out.astype(np.uint8)


def make_maskbank_for_windows(windows: np.ndarray, ratios, lm: int, seed: int):
    """
    windows: [N, L, K]
    returns dict:
      ratio_string -> eval_mask [N, L, K], uint8
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape [N, L, K], got {windows.shape}")

    N, L, K = windows.shape
    rng = np.random.RandomState(seed)
    out = {}

    for ratio in ratios:
        masks = np.zeros((N, L, K), dtype=np.uint8)
        for n in range(N):
            for k in range(K):
                masks[n, :, k] = make_1d_markov_mask(
                    length=L,
                    masked_ratio=ratio,
                    lm=lm,
                    rng=rng
                )
        out[f"{ratio:.2f}"] = masks
    return out


def main():
    args = parse_args()
    windows_dir = Path(args.windows_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = windows_dir / "val_windows.npy"
    test_path = windows_dir / "test_windows.npy"

    if not val_path.exists():
        raise FileNotFoundError(f"Missing: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path}")

    val_windows = np.load(val_path)
    test_windows = np.load(test_path)

    if len(val_windows) == 0:
        raise ValueError("val_windows.npy is empty")
    if len(test_windows) == 0:
        raise ValueError("test_windows.npy is empty")

    summary = {
        "windows_dir": str(windows_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "lm": int(args.lm),
        "ratios": [float(r) for r in args.ratios],
        "seeds": [int(s) for s in args.seeds],
        "val_shape": list(val_windows.shape),
        "test_shape": list(test_windows.shape),
        "files": [],
    }

    for seed in args.seeds:
        val_bank = make_maskbank_for_windows(val_windows, args.ratios, args.lm, seed)
        test_bank = make_maskbank_for_windows(test_windows, args.ratios, args.lm, seed)

        val_out = out_dir / f"val_maskbank_seed{seed}.npz"
        test_out = out_dir / f"test_maskbank_seed{seed}.npz"

        np.savez_compressed(val_out, **val_bank)
        np.savez_compressed(test_out, **test_bank)

        summary["files"].append(str(val_out.name))
        summary["files"].append(str(test_out.name))

        print(f"Saved {val_out}")
        for ratio in sorted(val_bank.keys()):
            print(f"  ratio={ratio} shape={val_bank[ratio].shape} actual_mean={val_bank[ratio].mean():.4f}")

        print(f"Saved {test_out}")
        for ratio in sorted(test_bank.keys()):
            print(f"  ratio={ratio} shape={test_bank[ratio].shape} actual_mean={test_bank[ratio].mean():.4f}")

    with open(out_dir / "maskbank_meta.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved metadata: {out_dir / 'maskbank_meta.json'}")


if __name__ == "__main__":
    main()

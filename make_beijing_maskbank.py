#!/usr/bin/env python3
"""
make_beijing_maskbank.py

Create shared evaluation maskbanks for pre-windowed Beijing data.

Expected input directory:
  train_windows.npy
  val_windows.npy
  test_windows.npy

Outputs:
  val_maskbank_seed{seed}.npz
  test_maskbank_seed{seed}.npz

Each .npz contains arrays keyed by masked ratio strings:
  "0.10", "0.30", "0.50", "0.70"

Mask semantics:
  1 = masked
  0 = observed
"""

import os
import argparse
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def markov_mask_array(
    N: int,
    L: int,
    K: int,
    r_masked: float,
    lm: float,
    seed: int,
) -> np.ndarray:
    """
    Returns eval mask of shape (N, L, K), where:
      1 = masked
      0 = observed
    """
    rng = np.random.default_rng(seed)

    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]

    keep = np.ones((N, K, L), dtype=np.float32)

    for n in range(N):
        for k in range(K):
            state = int(rng.random() < r_keep)  # 1 = keep, 0 = masked
            for t in range(L):
                keep[n, k, t] = state
                if rng.random() < p[state]:
                    state = 1 - state

    keep = np.transpose(keep, (0, 2, 1))   # (N,L,K)
    evalmask = 1.0 - keep                   # 1 = masked
    return evalmask.astype(np.float32)


def _find_first_existing(base_dir, candidates):
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "None of these files were found in "
        f"{base_dir}: {candidates}"
    )


def load_windows(shared_data_dir: str):
    train_path = _find_first_existing(shared_data_dir, [
        "train_windows.npy",
        "beijing_train_windows.npy",
        "train.npy",
        "X_train.npy",
    ])
    val_path = _find_first_existing(shared_data_dir, [
        "val_windows.npy",
        "beijing_val_windows.npy",
        "val.npy",
        "X_val.npy",
    ])
    test_path = _find_first_existing(shared_data_dir, [
        "test_windows.npy",
        "beijing_test_windows.npy",
        "test.npy",
        "X_test.npy",
    ])

    train_w = np.load(train_path)
    val_w = np.load(val_path)
    test_w = np.load(test_path)

    if train_w.ndim != 3 or val_w.ndim != 3 or test_w.ndim != 3:
        raise ValueError(
            f"Expected (N,L,K) arrays, got "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    if train_w.shape[1:] != val_w.shape[1:] or train_w.shape[1:] != test_w.shape[1:]:
        raise ValueError(
            f"Window shape mismatch: "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    print(f"[load] train={train_path}")
    print(f"[load] val={val_path}")
    print(f"[load] test={test_path}")

    return train_w, val_w, test_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_data_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = args.out_dir if args.out_dir is not None else args.shared_data_dir
    os.makedirs(out_dir, exist_ok=True)

    train_w, val_w, test_w = load_windows(args.shared_data_dir)

    N_val, L_val, K_val = val_w.shape
    N_test, L_test, K_test = test_w.shape

    if L_val != args.seq_len or L_test != args.seq_len:
        raise ValueError(
            f"seq_len mismatch: val L={L_val}, test L={L_test}, arg seq_len={args.seq_len}"
        )

    ratios = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]

    val_bank = {}
    test_bank = {}

    for r in ratios:
        key = f"{r:.2f}"

        val_mask = markov_mask_array(
            N=N_val,
            L=L_val,
            K=K_val,
            r_masked=r,
            lm=args.lm,
            seed=args.seed * 1000 + int(round(r * 100)) + 11,
        )
        test_mask = markov_mask_array(
            N=N_test,
            L=L_test,
            K=K_test,
            r_masked=r,
            lm=args.lm,
            seed=args.seed * 1000 + int(round(r * 100)) + 29,
        )

        val_bank[key] = val_mask
        test_bank[key] = test_mask

        print(
            f"[ratio {key}] "
            f"val shape={val_mask.shape} mean_masked={val_mask.mean():.4f} | "
            f"test shape={test_mask.shape} mean_masked={test_mask.mean():.4f}"
        )

    val_path = os.path.join(out_dir, f"val_maskbank_seed{args.seed}.npz")
    test_path = os.path.join(out_dir, f"test_maskbank_seed{args.seed}.npz")

    np.savez(val_path, **val_bank)
    np.savez(test_path, **test_bank)

    print(f"[saved] {val_path}")
    print(f"[saved] {test_path}")
    print(
        f"[summary] train/val/test windows = {len(train_w)}/{len(val_w)}/{len(test_w)}, "
        f"L={args.seq_len}, K={K_val}"
    )


if __name__ == "__main__":
    main()

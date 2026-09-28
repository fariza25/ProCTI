#!/usr/bin/env python3


import os, argparse, random
from typing import Optional
import numpy as np
import pandas as pd


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_weather_windows(csv_path: str, seq_len: int, time_col: Optional[str]):
    df = pd.read_csv(csv_path)

    if time_col is None:
        if "date" in df.columns:
            time_col = "date"
        elif "time" in df.columns:
            time_col = "time"
        else:
            time_col = df.columns[0]

    t = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[~t.isna()].copy()
    df["_t"] = pd.to_datetime(df[time_col])
    df = df.sort_values("_t", kind="mergesort").drop(columns=["_t"])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found.")

    X = df[num_cols].astype(np.float32).to_numpy()  # (T,K)
    T, K = X.shape
    n_win = T // seq_len
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({T}) for seq_len={seq_len}")
    X = X[: n_win * seq_len].reshape(n_win, seq_len, K)
    return X.astype(np.float32), num_cols, time_col


def chrono_split_windows(windows_NLK: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    n = len(windows_NLK)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = windows_NLK[:n_train]
    val = windows_NLK[n_train:n_train + n_val]
    test = windows_NLK[n_train + n_val:]
    return train, val, test


def markov_keep_mask_from_masked_ratio_np(N: int, K: int, L: int, r_masked: float, lm: float, seed: int) -> np.ndarray:
    """
    Segment-wise Markov keep-mask (deterministic via seed).
    state 1 = keep, state 0 = masked
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0

    rng = np.random.RandomState(seed)
    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm
    #p_u = p_m * r_keep / (1.0 - r_keep)
    p_u = p_m *(1.0 - r_keep) /r_keep
    p = [p_m, p_u]  # state 0=masked, 1=unmasked

    out = np.ones((N, L, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            state = int(rng.rand() < r_keep)
            for t in range(L):
                out[n, t, k] = state
                if rng.rand() < p[state]:
                    state = 1 - state
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", type=str, default="shared_weather_maskbank")
    ap.add_argument("--include_val", action="store_true", help="Also generate val keepmasks (recommended).")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    windows, feat_cols, tcol = load_weather_windows(args.csv, args.seq_len, args.time_col)
    train_w, val_w, test_w = chrono_split_windows(windows)

    K = windows.shape[-1]
    L = args.seq_len
    eval_masked = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]

    print(f"[data] windows={len(windows)}  train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={L}  time_col={tcol}")

    splits = [("test", test_w)]
    if args.include_val:
        splits.append(("val", val_w))

    for split_name, split_w in splits:
        N = len(split_w)
        for r_m in eval_masked:
            # stable seed per r and split
            split_id = 2 if split_name == "test" else 1
            seed_mask = args.seed + 999 + int(round(r_m * 1000)) + split_id * 100000
            keep = markov_keep_mask_from_masked_ratio_np(N=N, K=K, L=L, r_masked=r_m, lm=args.lm, seed=seed_mask)

            out = os.path.join(args.out_dir, f"weather_{split_name}_keepmask_r{r_m:.2f}_L{L}_seed{args.seed}.npy")
            np.save(out, keep)
            print(f"[save] {out}  shape={keep.shape}  mean_keep={keep.mean():.3f}")

    meta_path = os.path.join(args.out_dir, f"weather_maskbank_metadata_L{L}_seed{args.seed}.npz")
    np.savez(
        meta_path,
        seq_len=L, lm=args.lm, seed=args.seed, time_col=tcol,
        feature_cols=np.array(feat_cols, dtype=object),
        n_windows=len(windows), n_train=len(train_w), n_val=len(val_w), n_test=len(test_w),
        eval_masked_ratios=np.array(eval_masked, dtype=np.float32),
    )
    print(f"[save] {meta_path}")


if __name__ == "__main__":
    main()

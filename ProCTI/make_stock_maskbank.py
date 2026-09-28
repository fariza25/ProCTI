#!/usr/bin/env python3
"""make_stock_maskbank_aligned.py

Generate deterministic Markov segment *keep-masks* for STOCK-style CSV data
with NO time/date column, using a ROW-WISE chronological split and within-split
non-overlapping windows.

This script mirrors the structure and metadata behavior of
`make_physionet_maskbank_aligned.py`, adapted for stock data.

Key behavior:
- Load CSV in row order; use numeric columns as features.
- Replace +/-inf with NaN.
- ROW-WISE split in original order: 70%/15%/15% on rows (top->bottom).
- Within each split, create NON-OVERLAPPING windows of length L (seq_len).
  Each split is truncated down to a multiple of L.
- Compute TRAIN nanmean/nanstd over windowed TRAIN data and derive a
  valid_feature_mask (finite mean/std and std>eps). This is the *kept-feature*
  space used by sharedmask training/eval pipelines.
- Generate deterministic Markov keep-masks for selected splits (val/test by default).
- Save masks in kept-feature space (default) or raw-feature space (--mask_space raw).

Outputs (.npy float32):
  {out_dir}/stock_{split}_keepmask_r{r:.2f}_L{L}_seed{seed}.npy   shape (N,L,K_out)

Also writes:
  {out_dir}/stock_maskbank_metadata_L{L}_seed{seed}.npz

Mask semantics:
  keepmask[n,t,k] = 1.0  => observed / kept (conditioning input)
  keepmask[n,t,k] = 0.0  => masked (to impute / to score)
  evalmask = 1 - keepmask

"""

import os
import argparse
import random
from typing import List, Tuple

import numpy as np
import pandas as pd


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_stock_numeric(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load numeric feature matrix X (T,K) from CSV in row order."""
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        # fallback for headerless numeric files
        df2 = pd.read_csv(csv_path, header=None)
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            raise ValueError("No numeric columns found in stock CSV.")
        X = df2[num_cols].to_numpy(dtype=np.float32)
        feat_cols = [str(c) for c in num_cols]
    else:
        X = df[num_cols].to_numpy(dtype=np.float32)
        feat_cols = [str(c) for c in num_cols]

    X[~np.isfinite(X)] = np.nan
    return X.astype(np.float32), feat_cols


def split_rows(X_TK: np.ndarray, train_ratio: float, val_ratio: float):
    T = X_TK.shape[0]
    t_train = int(T * train_ratio)
    t_val_end = int(T * (train_ratio + val_ratio))
    return X_TK[:t_train], X_TK[t_train:t_val_end], X_TK[t_val_end:], t_train, t_val_end


def to_windows(seg_TK: np.ndarray, L: int) -> np.ndarray:
    T, K = seg_TK.shape
    n_win = T // L
    if n_win <= 0:
        return np.zeros((0, L, K), dtype=np.float32)
    seg = seg_TK[: n_win * L].reshape(n_win, L, K)
    return seg.astype(np.float32)


def compute_train_stats_and_valid(train_windows: np.ndarray, eps: float = 1e-6):
    flat = train_windows.reshape(-1, train_windows.shape[-1])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)
    if valid.sum() == 0:
        raise ValueError(
            "All features invalid after TRAIN nanmean/nanstd. "
            "Check CSV values (NaN/inf) and eps threshold."
        )
    return mu.astype(np.float32), sd.astype(np.float32), valid


def markov_keep_mask_from_masked_ratio_np(N: int, K: int, L: int, r_masked: float, lm: float, seed: int) -> np.ndarray:
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0

    rng = np.random.RandomState(seed)
    r_keep = 1.0 - r_masked

    # state 0=masked, 1=keep
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / r_keep
    p = [p_m, p_u]

    out = np.ones((N, L, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            state = int(rng.rand() < r_keep)
            for t in range(L):
                out[n, t, k] = float(state)
                if rng.rand() < p[state]:
                    state = 1 - state
    return out


def stable_mask_seed(base_seed: int, split_name: str, r_masked: float) -> int:
    split_id = {"train": 0, "val": 1, "test": 2}.get(split_name, 9)
    return int(base_seed + 999 + round(r_masked * 1000) + split_id * 100000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", type=str, default="shared_stock_maskbank")
    ap.add_argument("--include_val", action="store_true")
    ap.add_argument("--mask_space", type=str, default="kept", choices=["kept", "raw"])
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    args = ap.parse_args()

    set_seed(args.seed)
    _ensure_dir(args.out_dir)

    X, feat_cols = load_stock_numeric(args.csv)
    T, K_raw = X.shape
    L = args.seq_len

    Xtr, Xva, Xte, t_train, t_val_end = split_rows(X, args.train_ratio, args.val_ratio)
    train_w_raw = to_windows(Xtr, L)
    val_w_raw   = to_windows(Xva, L)
    test_w_raw  = to_windows(Xte, L)

    if len(train_w_raw) == 0:
        raise ValueError("No TRAIN windows formed (seq_len too large for train split).")

    mu, sd, valid = compute_train_stats_and_valid(train_w_raw, eps=args.eps)
    K_kept = int(valid.sum())

    ratios = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]
    ratios = [r for r in ratios if 0.0 < r < 1.0]

    print(
        f"[data] rows={T} split_rows train/val/test={t_train}/{t_val_end - t_train}/{T - t_val_end}  "
        f"windows train/val/test={len(train_w_raw)}/{len(val_w_raw)}/{len(test_w_raw)}  "
        f"K_raw={K_raw} K_kept={K_kept} L={L} mask_space={args.mask_space}"
    )
    print(f"[prep] kept {K_kept} / {K_raw} numeric features after TRAIN invalid/zero-std drop.")

    splits: List[Tuple[str, int]] = [("test", len(test_w_raw))]
    if args.include_val:
        splits.append(("val", len(val_w_raw)))

    for split_name, N in splits:
        for r_m in ratios:
            seed_mask = stable_mask_seed(args.seed, split_name, r_m)
            K_out = K_kept if args.mask_space == "kept" else K_raw
            keep = markov_keep_mask_from_masked_ratio_np(N=N, K=K_out, L=L, r_masked=r_m, lm=args.lm, seed=seed_mask)
            out = os.path.join(args.out_dir, f"stock_{split_name}_keepmask_r{r_m:.2f}_L{L}_seed{args.seed}.npy")
            np.save(out, keep)
            print(f"[save] {out}  shape={keep.shape}  mean_keep={keep.mean():.3f}")

    meta_path = os.path.join(args.out_dir, f"stock_maskbank_metadata_L{L}_seed{args.seed}.npz")
    np.savez(
        meta_path,
        csv=args.csv,
        seq_len=L,
        lm=args.lm,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        eps=args.eps,
        mask_space=args.mask_space,
        n_rows_total=T,
        split_row_train_end=t_train,
        split_row_val_end=t_val_end,
        n_windows_train=len(train_w_raw),
        n_windows_val=len(val_w_raw),
        n_windows_test=len(test_w_raw),
        eval_masked_ratios=np.array(ratios, dtype=np.float32),
        valid_feature_mask=valid.astype(np.uint8),
        train_mu_raw=mu,
        train_sd_raw=sd,
        n_features_raw=int(K_raw),
        n_features_kept=int(K_kept),
        feature_cols_raw=np.array(feat_cols, dtype=object),
    )
    print(f"[save] {meta_path}")
    print(
        "\nDone.\n"
        "If you change feature selection / eps / seq_len, regenerate the maskbank.\n"
        "For sharedmask training scripts that operate in kept-feature space, use --mask_space kept (default).\n"
    )


if __name__ == "__main__":
    main()


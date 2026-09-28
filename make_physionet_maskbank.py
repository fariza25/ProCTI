#!/usr/bin/env python3
"""
make_physionet_maskbank.py

Build shared keepmask banks + metadata for PhysioNet 2019.

Outputs:
- physionet_maskbank_metadata_L{seq_len}_seed{seed}.npz
- physionet_val_keepmask_r{ratio}_L{seq_len}_seed{seed}.npy
- physionet_test_keepmask_r{ratio}_L{seq_len}_seed{seed}.npy

Mask convention:
- keepmask = 1 means observed/kept
- keepmask = 0 means masked/evaluated
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


def _to_int_id(x):
    if x is None:
        raise ValueError("patient id is None")
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    s = str(x).strip()
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def load_physionet_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Patient_ID" not in df.columns:
        raise ValueError("physionet2019.csv must contain Patient_ID.")

    df["Patient_ID"] = pd.to_numeric(df["Patient_ID"], errors="coerce")
    df = df.dropna(subset=["Patient_ID"]).copy()
    df["Patient_ID"] = df["Patient_ID"].astype(int)

    return df


def get_time_col(df: pd.DataFrame) -> str:
    if "ICULOS" in df.columns:
        return "ICULOS"
    if "Hour" in df.columns:
        return "Hour"
    raise ValueError("Expected ICULOS or Hour in physionet CSV.")


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_cols = {"Patient_ID"}
    for c in ["SepsisLabel", "Hour", "ICULOS", "HospAdmTime"]:
        if c in df.columns:
            drop_cols.add(c)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns found.")
    return feat_cols


def split_patients(df: pd.DataFrame, seed: int, train_ratio=0.7, val_ratio=0.15):
    pids = sorted(df["Patient_ID"].dropna().unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_p = pids[:n_train]
    val_p = pids[n_train:n_train + n_val]
    test_p = pids[n_train + n_val:]
    return train_p, val_p, test_p


def make_windows_for_patients(
    df: pd.DataFrame,
    patient_ids: List[int],
    feat_cols: List[str],
    time_col: str,
    seq_len: int
) -> np.ndarray:
    patient_ids = set(int(x) for x in patient_ids)
    windows = []

    for pid, g in df.groupby("Patient_ID", sort=False):
        if int(pid) not in patient_ids:
            continue

        g = g.sort_values(time_col, kind="mergesort")
        X = g[feat_cols].astype(np.float32).to_numpy()
        T, K = X.shape
        n_win = T // seq_len
        if n_win <= 0:
            continue

        windows.append(X[: n_win * seq_len].reshape(n_win, seq_len, K))

    if not windows:
        return np.zeros((0, seq_len, len(feat_cols)), dtype=np.float32)

    return np.concatenate(windows, axis=0).astype(np.float32)


def compute_train_stats_and_validmask(train_w: np.ndarray, eps: float = 1e-6):
    train_flat = train_w.reshape(-1, train_w.shape[-1])

    mu_raw = np.nanmean(train_flat, axis=0)
    sd_raw = np.nanstd(train_flat, axis=0)

    valid = np.isfinite(mu_raw) & np.isfinite(sd_raw) & (sd_raw > eps)

    if valid.sum() == 0:
        raise ValueError("All features dropped by train standardization.")

    return mu_raw.astype(np.float32), sd_raw.astype(np.float32), valid.astype(bool)


def markov_keep_mask_array(
    N: int,
    L: int,
    K: int,
    r_masked: float,
    lm: float,
    seed: int
) -> np.ndarray:
    """
    Returns keepmask of shape (N, L, K), where:
      1 = kept/observed
      0 = masked
    """
    rng = np.random.default_rng(seed)
    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]

    out = np.ones((N, K, L), dtype=np.float32)

    for n in range(N):
        for k in range(K):
            state = int(rng.random() < r_keep)
            for t in range(L):
                out[n, k, t] = state
                if rng.random() < p[state]:
                    state = 1 - state

    return np.transpose(out, (0, 2, 1)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="shared_physionet_maskbank")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_physionet_df(args.csv)
    time_col = get_time_col(df)
    feat_cols = get_feature_cols(df)

    train_ids, val_ids, test_ids = split_patients(df, args.seed)

    train_w = make_windows_for_patients(df, train_ids, feat_cols, time_col, args.seq_len)
    val_w = make_windows_for_patients(df, val_ids, feat_cols, time_col, args.seq_len)
    test_w = make_windows_for_patients(df, test_ids, feat_cols, time_col, args.seq_len)

    if len(train_w) == 0 or len(val_w) == 0 or len(test_w) == 0:
        raise ValueError(
            f"Empty split detected: train={len(train_w)}, val={len(val_w)}, test={len(test_w)}"
        )

    mu_raw, sd_raw, valid_feature_mask = compute_train_stats_and_validmask(train_w)

    # metadata
    meta_path = os.path.join(
        args.out_dir,
        f"physionet_maskbank_metadata_L{args.seq_len}_seed{args.seed}.npz"
    )
    np.savez(
        meta_path,
        patients_train=np.array(train_ids, dtype=np.int64),
        patients_val=np.array(val_ids, dtype=np.int64),
        patients_test=np.array(test_ids, dtype=np.int64),
        feature_cols_raw=np.array(feat_cols, dtype=object),
        valid_feature_mask=valid_feature_mask,
        train_mu_raw=mu_raw,
        train_sd_raw=sd_raw,
        eps=np.float32(1e-6),
        seq_len=np.int64(args.seq_len),
        lm=np.float32(args.lm),
        n_train_windows=np.int64(len(train_w)),
        n_val_windows=np.int64(len(val_w)),
        n_test_windows=np.int64(len(test_w)),
    )

    ratios = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]

    K_raw = train_w.shape[-1]

    for r in ratios:
        # different deterministic seeds per split/ratio
        val_keep = markov_keep_mask_array(
            N=len(val_w),
            L=args.seq_len,
            K=K_raw,
            r_masked=r,
            lm=args.lm,
            seed=args.seed * 1000 + int(round(r * 100)) + 11,
        )
        test_keep = markov_keep_mask_array(
            N=len(test_w),
            L=args.seq_len,
            K=K_raw,
            r_masked=r,
            lm=args.lm,
            seed=args.seed * 1000 + int(round(r * 100)) + 29,
        )

        val_path = os.path.join(
            args.out_dir,
            f"physionet_val_keepmask_r{r:.2f}_L{args.seq_len}_seed{args.seed}.npy"
        )
        test_path = os.path.join(
            args.out_dir,
            f"physionet_test_keepmask_r{r:.2f}_L{args.seq_len}_seed{args.seed}.npy"
        )

        np.save(val_path, val_keep)
        np.save(test_path, test_keep)

        print(
            f"[saved] r={r:.2f} "
            f"val_keep={val_keep.shape} mean_keep={val_keep.mean():.4f} "
            f"test_keep={test_keep.shape} mean_keep={test_keep.mean():.4f}"
        )

    print(f"[done] metadata: {meta_path}")
    print(
        f"[splits] patients train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)} | "
        f"windows train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)}"
    )


if __name__ == "__main__":
    main()

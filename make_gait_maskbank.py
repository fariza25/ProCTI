"""t_maskbank.py

Create shared evaluation keepmasks for the OU gait dataset.

Outputs:
  gait_val_keepmask_r{ratio}_L{seq_len}_seed{seed}.npy
  gait_test_keepmask_r{ratio}_L{seq_len}_seed{seed}.npy
  gait_maskbank_meta_L{seq_len}_seed{seed}.npz

Mask semantics:
  keepmask = 1 means observed/kept
  keepmask = 0 means masked/evaluated
"""

import os
import re
import json
import random
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd


USER_RE = re.compile(r"_ID(\d+)_", re.IGNORECASE)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def user_id_from_name(path: str) -> str:
    m = USER_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse user ID from filename: {os.path.basename(path)}")
    return m.group(1)


def list_gait_files(data_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".csv"):
            files.append(os.path.join(data_dir, fn))
    if not files:
        raise ValueError(f"No .csv files found in {data_dir}")
    return sorted(files)


def read_gait_csv(path: str) -> np.ndarray:
    # Matches the gait ablation loader
    df = pd.read_csv(path, skiprows=2, header=None)
    arr = df.values.astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Bad array shape from {path}: {arr.shape}")
    return arr


def split_users(files: List[str], seed: int, train_ratio=0.7, val_ratio=0.15):
    by_user = {}
    for p in files:
        uid = user_id_from_name(p)
        by_user.setdefault(uid, []).append(p)

    users = sorted(by_user.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    n = len(users)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_u = users[:n_train]
    val_u = users[n_train:n_train + n_val]
    test_u = users[n_train + n_val:]

    def gather(uids):
        out = []
        for u in uids:
            out.extend(by_user[u])
        return out

    return gather(train_u), gather(val_u), gather(test_u), (train_u, val_u, test_u)


def windows_from_files(file_list: List[str], seq_len: int) -> np.ndarray:
    windows = []
    for p in file_list:
        arr = read_gait_csv(p)
        T, K = arr.shape
        n_win = T // seq_len
        if n_win <= 0:
            continue
        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, K))
    if not windows:
        return np.zeros((0, seq_len, 0), dtype=np.float32)
    return np.concatenate(windows, axis=0).astype(np.float32)


def markov_keep_mask_array(
    N: int,
    L: int,
    K: int,
    r_masked: float,
    lm: float,
    seed: int,
) -> np.ndarray:
    """
    Returns keepmask of shape (N, L, K):
      1 = observed/kept
      0 = masked/evaluated
    """
    rng = np.random.default_rng(seed)

    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]

    out = np.ones((N, K, L), dtype=np.float32)

    for n in range(N):
        for k in range(K):
            state = int(rng.random() < r_keep)  # 1 = keep
            for t in range(L):
                out[n, k, t] = state
                if rng.random() < p[state]:
                    state = 1 - state

    return np.transpose(out, (0, 2, 1)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="shared_gait_maskbank")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    files = list_gait_files(args.data_dir)
    tr_files, va_files, te_files, (tr_users, va_users, te_users) = split_users(files, seed=args.seed)

    train_w = windows_from_files(tr_files, args.seq_len)
    val_w = windows_from_files(va_files, args.seq_len)
    test_w = windows_from_files(te_files, args.seq_len)

    if len(val_w) == 0 or len(test_w) == 0:
        raise ValueError(
            f"Empty split detected: train={len(train_w)}, val={len(val_w)}, test={len(test_w)}"
        )

    K = train_w.shape[-1]
    if K == 0:
        raise ValueError("No gait features found after windowing.")

    ratios = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]

    for r in ratios:
        val_keep = markov_keep_mask_array(
            N=len(val_w),
            L=args.seq_len,
            K=K,
            r_masked=r,
            lm=args.lm,
            seed=args.seed * 1000 + int(round(r * 100)) + 11,
        )
        test_keep = markov_keep_mask_array(
            N=len(test_w),
            L=args.seq_len,
            K=K,
            r_masked=r,
            lm=args.lm,
            seed=args.seed * 1000 + int(round(r * 100)) + 29,
        )

        val_path = os.path.join(
            args.out_dir,
            f"gait_val_keepmask_r{r:.2f}_L{args.seq_len}_seed{args.seed}.npy"
        )
        test_path = os.path.join(
            args.out_dir,
            f"gait_test_keepmask_r{r:.2f}_L{args.seq_len}_seed{args.seed}.npy"
        )

        np.save(val_path, val_keep)
        np.save(test_path, test_keep)

        print(
            f"[saved] r={r:.2f} "
            f"val_keep={val_keep.shape} mean_keep={val_keep.mean():.4f} "
            f"test_keep={test_keep.shape} mean_keep={test_keep.mean():.4f}"
        )

    meta_npz = os.path.join(
        args.out_dir,
        f"gait_maskbank_meta_L{args.seq_len}_seed{args.seed}.npz"
    )
    np.savez(
        meta_npz,
        train_users=np.array(tr_users, dtype=object),
        val_users=np.array(va_users, dtype=object),
        test_users=np.array(te_users, dtype=object),
        n_train_windows=np.int64(len(train_w)),
        n_val_windows=np.int64(len(val_w)),
        n_test_windows=np.int64(len(test_w)),
        seq_len=np.int64(args.seq_len),
        lm=np.float32(args.lm),
    )

    meta_json = os.path.join(
        args.out_dir,
        f"gait_maskbank_meta_L{args.seq_len}_seed{args.seed}.json"
    )
    with open(meta_json, "w") as f:
        json.dump(
            {
                "train_users": list(tr_users),
                "val_users": list(va_users),
                "test_users": list(te_users),
                "n_train_windows": int(len(train_w)),
                "n_val_windows": int(len(val_w)),
                "n_test_windows": int(len(test_w)),
                "seq_len": int(args.seq_len),
                "lm": float(args.lm),
            },
            f,
            indent=2,
        )

    print(f"[meta] {meta_npz}")
    print(f"[meta] {meta_json}")
    print(
        f"[summary] users train/val/test = {len(tr_users)}/{len(va_users)}/{len(te_users)} | "
        f"windows train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)} | K={K} | L={args.seq_len}"
    )


if __name__ == "__main__":
    main()

import os, re, json, argparse, random
from typing import List, Tuple
import numpy as np
import pandas as pd

USER_RE = re.compile(r"_ID(\d+)_", re.IGNORECASE)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def list_gait_files(data_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".csv"):
            files.append(os.path.join(data_dir, fn))
    if not files:
        raise ValueError(f"No .csv files found in {data_dir}")
    return sorted(files)


def user_id_from_name(path: str) -> str:
    m = USER_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse user ID from filename: {os.path.basename(path)}")
    return m.group(1)


def read_gait_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, skiprows=2, header=None)
    arr = df.values.astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Bad array shape from {path}: {arr.shape}")
    return arr


def windows_from_files(file_list: List[str], seq_len: int) -> np.ndarray:
    windows = []
    K_ref = None
    for p in file_list:
        arr = read_gait_csv(p)
        T, K = arr.shape
        if K_ref is None:
            K_ref = K
        elif K != K_ref:
            raise ValueError(f"Channel count mismatch: {p} has K={K}, expected K={K_ref}")
        n_win = T // seq_len
        if n_win <= 0:
            continue
        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, K))
    if not windows:
        raise ValueError("No windows formed (seq_len may be too large or files too short).")
    return np.concatenate(windows, axis=0).astype(np.float32)


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


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    mu = train.reshape(-1, train.shape[-1]).mean(axis=0)
    sd = train.reshape(-1, train.shape[-1]).std(axis=0)
    sd[sd < eps] = 1.0

    train = (train - mu[None, None, :]) / sd[None, None, :]
    val   = (val   - mu[None, None, :]) / sd[None, None, :]
    test  = (test  - mu[None, None, :]) / sd[None, None, :]

    return train.astype(np.float32), val.astype(np.float32), test.astype(np.float32), mu, sd


def make_channel_drop_maskbank(windows: np.ndarray, n_drop: int, seed: int):
    rng = np.random.default_rng(seed)
    N, L, K = windows.shape

    eval_mask = np.zeros((N, L, K), dtype=np.float32)
    dropped_channels = np.full((N, n_drop), -1, dtype=np.int64)

    for i in range(N):
        ch = rng.choice(K, size=n_drop, replace=False)
        dropped_channels[i] = ch
        eval_mask[i, :, ch] = 1.0

    return eval_mask, dropped_channels


def save_maskbank(path, eval_mask, dropped_channels, metadata):
    np.savez(
        path,
        eval_mask=eval_mask,
        dropped_channels=dropped_channels,
        metadata_json=json.dumps(metadata),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    files = list_gait_files(args.data_dir)
    tr_files, va_files, te_files, (tr_users, va_users, te_users) = split_users(files, seed=args.seed)

    train_w = windows_from_files(tr_files, args.seq_len)
    val_w   = windows_from_files(va_files, args.seq_len)
    test_w  = windows_from_files(te_files, args.seq_len)

    train_w, val_w, test_w, mu, sd = standardize_by_train(train_w, val_w, test_w)

    print(
        f"[data] #users train/val/test={len(tr_users)}/{len(va_users)}/{len(te_users)}  "
        f"#windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  "
        f"K={train_w.shape[-1]} L={args.seq_len}"
    )

    np.save(os.path.join(args.out_dir, f"gait_seq{args.seq_len}_train_windows.npy"), train_w)
    np.save(os.path.join(args.out_dir, f"gait_seq{args.seq_len}_val_windows.npy"), val_w)
    np.save(os.path.join(args.out_dir, f"gait_seq{args.seq_len}_test_windows.npy"), test_w)

    print(f"[save] {os.path.join(args.out_dir, f'gait_seq{args.seq_len}_train_windows.npy')}")
    print(f"[save] {os.path.join(args.out_dir, f'gait_seq{args.seq_len}_val_windows.npy')}")
    print(f"[save] {os.path.join(args.out_dir, f'gait_seq{args.seq_len}_test_windows.npy')}")

    for drop in [1, 2]:
        for split_name, split_w in [("val", val_w), ("test", test_w)]:
            split_id = 2 if split_name == "test" else 1
            seed_mask = args.seed + 1337 + (drop * 1000) + split_id * 100000

            eval_mask, dropped_channels = make_channel_drop_maskbank(
                split_w, n_drop=drop, seed=seed_mask
            )

            out_path = os.path.join(
                args.out_dir,
                f"gait_seq{args.seq_len}_{split_name}_drop{drop}_seed{args.seed}.npz"
            )

            metadata = {
                "dataset": "gait",
                "seq_len": args.seq_len,
                "drop_channels": drop,
                "seed": args.seed,
                "split": split_name,
                "n_users_train": len(tr_users),
                "n_users_val": len(va_users),
                "n_users_test": len(te_users),
                "users_train": list(tr_users),
                "users_val": list(va_users),
                "users_test": list(te_users),
            }

            save_maskbank(out_path, eval_mask, dropped_channels, metadata)
            print(f"[save] {out_path}  shape={eval_mask.shape}")

if __name__ == "__main__":
    main()

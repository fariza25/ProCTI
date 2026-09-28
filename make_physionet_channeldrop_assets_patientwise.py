import os
import json
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

    pid_col = "Patient_ID"
    if pid_col not in df.columns:
        raise ValueError(f"{pid_col} column not found in CSV.")

    df[pid_col] = pd.to_numeric(df[pid_col], errors="coerce")
    df = df.dropna(subset=[pid_col]).copy()
    df[pid_col] = df[pid_col].astype(int)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def get_time_col(df: pd.DataFrame) -> str:
    if "ICULOS" in df.columns:
        return "ICULOS"
    if "Hour" in df.columns:
        return "Hour"
    raise ValueError("Expected a time column ICULOS or Hour in physionet2019.csv.")


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {
        "Patient_ID", "patient_id", "PatientID", "patient", "RecordID", "id", "UID", "stay_id",
        "ICULOS", "Hour",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns found.")
    return cols


def split_patients(df: pd.DataFrame, seed: int, train_ratio=0.70, val_ratio=0.15):
    patient_ids = sorted(_to_int_id(x) for x in df["Patient_ID"].dropna().unique().tolist())
    rng = np.random.default_rng(seed)
    patient_ids = list(rng.permutation(patient_ids))

    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def make_windows_for_patients(df, patient_ids_set, feat_cols, time_col, seq_len):
    patient_ids_set = set(int(x) for x in patient_ids_set)
    windows = []

    for pid, g in df.groupby("Patient_ID", sort=False):
        if int(pid) not in patient_ids_set:
            continue

        if time_col:
            g = g.sort_values(time_col, kind="mergesort")

        X = g[feat_cols].astype(np.float32).ffill().bfill()
        arr = X.to_numpy()
        T = arr.shape[0]
        n_win = T // seq_len
        if n_win <= 0:
            continue

        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, arr.shape[1]))

    if not windows:
        raise ValueError("No windows formed for this split.")

    return np.concatenate(windows, axis=0).astype(np.float32)


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    """
    - compute TRAIN nanmean/nanstd
    - drop features that are non-finite or ~zero-std in TRAIN
    - fill remaining NaNs using TRAIN mean
    - z-score with TRAIN mean/std
    """
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)
    if valid.sum() == 0:
        raise ValueError("All features are non-finite / zero-std after nanmean/nanstd on TRAIN.")

    train = train[:, :, valid]
    val = val[:, :, valid]
    test = test[:, :, valid]
    mu = mu[valid]
    sd = sd[valid]
    sd = np.maximum(sd, eps)

    def fill_and_z(x):
        x = x.copy()
        nanmask = np.isnan(x)
        if nanmask.any():
            x[nanmask] = np.take(mu, np.where(nanmask)[2])
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid


def make_channel_drop_maskbank(windows: np.ndarray, n_drop: int, seed: int):
    rng = np.random.default_rng(seed)
    N, L, K = windows.shape

    eval_mask = np.zeros((N, L, K), dtype=np.float32)
    dropped_channels = np.full((N, n_drop), -1, dtype=np.int64)

    for i in range(N):
        chosen = rng.choice(K, size=n_drop, replace=False)
        dropped_channels[i] = chosen
        eval_mask[i, :, chosen] = 1.0

    return eval_mask, dropped_channels


def save_maskbank_npz(out_path, eval_mask, dropped_channels, metadata):
    np.savez_compressed(
        out_path,
        eval_mask=eval_mask.astype(np.float32),
        dropped_channels=dropped_channels.astype(np.int64),
        metadata_json=json.dumps(metadata, indent=2),
    )
    print(f"[saved] {out_path}")
    print(f"        eval_mask.shape={eval_mask.shape}")
    print(f"        dropped_channels.shape={dropped_channels.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = load_physionet_df(args.csv)
    time_col = get_time_col(df)
    feat_cols_raw = get_feature_cols(df)

    train_ids, val_ids, test_ids = split_patients(
        df, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    train_w = make_windows_for_patients(df, train_ids, feat_cols_raw, time_col, args.seq_len)
    val_w = make_windows_for_patients(df, val_ids, feat_cols_raw, time_col, args.seq_len)
    test_w = make_windows_for_patients(df, test_ids, feat_cols_raw, time_col, args.seq_len)

    train_w, val_w, test_w, mu, sd, valid = standardize_by_train(train_w, val_w, test_w)
    feat_cols_kept = [c for c, keep in zip(feat_cols_raw, valid.tolist()) if keep]

    K = train_w.shape[-1]

    print(
        f"[data] #patients train/val/test={len(set(train_ids))}/{len(set(val_ids))}/{len(set(test_ids))}  "
        f"#windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={args.seq_len}"
    )
    print(f"[prep] kept {int(valid.sum())} / {int(len(valid))} numeric features after dropping all-NaN/zero-std in TRAIN.")

    np.save(os.path.join(args.out_dir, f"physionet_seq{args.seq_len}_train_windows.npy"), train_w)
    np.save(os.path.join(args.out_dir, f"physionet_seq{args.seq_len}_val_windows.npy"), val_w)
    np.save(os.path.join(args.out_dir, f"physionet_seq{args.seq_len}_test_windows.npy"), test_w)
    print("[saved] window arrays")

    meta = {
        "dataset": "physionet",
        "seq_len": int(args.seq_len),
        "seed": int(args.seed),
        "time_col": str(time_col),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "patients_train": [int(x) for x in train_ids],
        "patients_val": [int(x) for x in val_ids],
        "patients_test": [int(x) for x in test_ids],
        "feature_cols_raw": [str(c) for c in feat_cols_raw],
        "feature_cols_kept": [str(c) for c in feat_cols_kept],
        "valid_feature_mask": [int(x) for x in valid.astype(np.uint8).tolist()],
        "mu": [float(x) for x in mu.tolist()],
        "sd": [float(x) for x in sd.tolist()],
    }

    with open(os.path.join(args.out_dir, f"physionet_seq{args.seq_len}_metadata_seed{args.seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[saved] metadata json")

    for n_drop in [1, 2]:
        val_eval_mask, val_dropped = make_channel_drop_maskbank(val_w, n_drop=n_drop, seed=args.seed)
        test_eval_mask, test_dropped = make_channel_drop_maskbank(test_w, n_drop=n_drop, seed=args.seed)

        common_meta = {
            "dataset": "physionet",
            "seq_len": int(args.seq_len),
            "seed": int(args.seed),
            "protocol": f"drop{n_drop}",
            "mask_type": "full_channel_drop",
            "patients_train": [int(x) for x in train_ids],
            "patients_val": [int(x) for x in val_ids],
            "patients_test": [int(x) for x in test_ids],
            "feature_cols_raw": [str(c) for c in feat_cols_raw],
            "feature_cols_kept": [str(c) for c in feat_cols_kept],
            "valid_feature_mask": [int(x) for x in valid.astype(np.uint8).tolist()],
        }

        save_maskbank_npz(
            os.path.join(args.out_dir, f"physionet_seq{args.seq_len}_val_drop{n_drop}_seed{args.seed}.npz"),
            val_eval_mask,
            val_dropped,
            {**common_meta, "split": "val"},
        )
        save_maskbank_npz(
            os.path.join(args.out_dir, f"physionet_seq{args.seq_len}_test_drop{n_drop}_seed{args.seed}.npz"),
            test_eval_mask,
            test_dropped,
            {**common_meta, "split": "test"},
        )

    print("Done.")


if __name__ == "__main__":
    main()

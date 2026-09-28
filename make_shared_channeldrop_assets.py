#!/usr/bin/env python3

"""
Generate shared channel-drop assets for time-series imputation experiments.

Creates:
  - window assets
  - channel-drop maskbanks

Supported datasets:
    weather
    physionet
    gait
    stock
    beijing
"""

import os
import argparse
import numpy as np
import pandas as pd
import json


# ---------------------------------------------------------
# utilities
# ---------------------------------------------------------

def chrono_split_windows(windows, train_ratio=0.7, val_ratio=0.15):

    n = len(windows)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = windows[:n_train]
    val = windows[n_train:n_train+n_val]
    test = windows[n_train+n_val:]

    return train, val, test


def standardize_by_train(train, val, test):

    mu = train.reshape(-1, train.shape[-1]).mean(axis=0)
    sd = train.reshape(-1, train.shape[-1]).std(axis=0)

    sd[sd < 1e-6] = 1

    train = (train - mu) / sd
    val = (val - mu) / sd
    test = (test - mu) / sd

    return train.astype(np.float32), val.astype(np.float32), test.astype(np.float32)


def make_channel_drop_maskbank(N, L, K, drop_channels, seed):

    rng = np.random.default_rng(seed)

    eval_mask = np.zeros((N, L, K), dtype=np.float32)

    dropped_channels = []

    for i in range(N):

        ch = rng.choice(K, size=drop_channels, replace=False)

        dropped_channels.append(ch)

        eval_mask[i,:,ch] = 1

    return eval_mask, np.array(dropped_channels)


def save_maskbank(path, eval_mask, dropped_channels, metadata):

    np.savez(
        path,
        eval_mask=eval_mask,
        dropped_channels=dropped_channels,
        metadata_json=json.dumps(metadata)
    )


# ---------------------------------------------------------
# dataset loaders
# ---------------------------------------------------------

def load_weather(csv, seq_len):

    df = pd.read_csv(csv)

    if "date" in df.columns:
        df = df.sort_values("date")

    X = df.select_dtypes(include=[np.number]).values

    T, K = X.shape

    n_win = T // seq_len

    X = X[:n_win*seq_len].reshape(n_win, seq_len, K)

    return X


def load_physionet(csv, seq_len):

    df = pd.read_csv(csv)

    if "Patient_ID" in df.columns:

        patients = df["Patient_ID"].unique()

        windows = []

        for p in patients:

            d = df[df["Patient_ID"]==p]

            X = d.select_dtypes(include=[np.number]).values

            n = len(X) // seq_len

            if n > 0:
                windows.append(X[:n*seq_len].reshape(n,seq_len,X.shape[1]))

        windows = np.concatenate(windows)

    else:

        X = df.select_dtypes(include=[np.number]).values
        n = len(X) // seq_len
        windows = X[:n*seq_len].reshape(n,seq_len,X.shape[1])

    return windows


def load_stock(csv, seq_len):

    df = pd.read_csv(csv)

    X = df.select_dtypes(include=[np.number]).values

    n = len(X) // seq_len

    return X[:n*seq_len].reshape(n,seq_len,X.shape[1])


def _read_gait_table(path):
    """
    Try to read a gait file as a numeric table.
    Supports comma-separated or whitespace-separated text.
    """
    # try normal csv first
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # then try whitespace / tab separated
    try:
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    return None


def load_gait(data_dir, seq_len):
    windows = []

    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".csv"):
                continue

            path = os.path.join(root, f)

            try:
                # gait files have:
                # line 1: rows: ...
                # line 2: cols: ...
                # then numeric comma-separated data
                df = pd.read_csv(path, skiprows=2, header=None)
            except Exception:
                continue

            # keep only numeric values
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(axis=1, how="all")
            df = df.dropna(axis=0, how="any")

            if df.shape[1] == 0 or len(df) < seq_len:
                continue

            X = df.values.astype(np.float32)
            n = len(X) // seq_len
            if n > 0:
                windows.append(X[: n * seq_len].reshape(n, seq_len, X.shape[1]))

    if len(windows) == 0:
        raise ValueError(
            f"No valid gait windows were loaded from {data_dir}."
        )

    return np.concatenate(windows, axis=0)


def load_beijing(data_dir, seq_len):
    windows = []

    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".csv"):
            continue

        path = os.path.join(data_dir, f)
        df = pd.read_csv(path)

        # Try to sort by a datetime column if present
        time_cols = [c for c in ["date", "datetime", "time", "timestamp"] if c in df.columns]
        if len(time_cols) > 0:
            tcol = time_cols[0]
            try:
                df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
                df = df.loc[~df[tcol].isna()].sort_values(tcol).reset_index(drop=True)
            except Exception:
                pass
        else:
            # Common Beijing benchmark structure: year, month, day, hour
            ymhd = [c for c in ["year", "month", "day", "hour"] if c in df.columns]
            if len(ymhd) == 4:
                try:
                    dt = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
                    df = df.loc[~dt.isna()].copy()
                    df["_dt"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
                    df = df.sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)
                except Exception:
                    pass

        # Keep only numeric columns
        num_df = df.select_dtypes(include=[np.number]).copy()

        # Drop identifier/time-part columns if present
        drop_cols = [c for c in ["No", "year", "month", "day", "hour"] if c in num_df.columns]
        if len(drop_cols) > 0:
            num_df = num_df.drop(columns=drop_cols)

        # Drop all-NaN columns and rows with NaNs
        num_df = num_df.dropna(axis=1, how="all")
        num_df = num_df.dropna(axis=0, how="any")

        if num_df.shape[1] == 0 or len(num_df) < seq_len:
            continue

        X = num_df.values.astype(np.float32)
        n = len(X) // seq_len
        if n > 0:
            windows.append(X[: n * seq_len].reshape(n, seq_len, X.shape[1]))

    if len(windows) == 0:
        raise ValueError(f"No valid Beijing windows were loaded from {data_dir}.")

    return np.concatenate(windows, axis=0)


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True,
                    choices=["weather","physionet","gait","stock","beijing"])

    ap.add_argument("--csv", default=None)
    ap.add_argument("--data_dir", default=None)

    ap.add_argument("--seq_len", default=96, type=int)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", default=1, type=int)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset=="weather":
        windows = load_weather(args.csv, args.seq_len)

    elif args.dataset=="physionet":
        windows = load_physionet(args.csv, args.seq_len)

    elif args.dataset=="stock":
        windows = load_stock(args.csv, args.seq_len)

    elif args.dataset=="gait":
        windows = load_gait(args.data_dir, args.seq_len)

    elif args.dataset=="beijing":
        windows = load_beijing(args.csv, args.seq_len)


    train,val,test = chrono_split_windows(windows)

    train,val,test = standardize_by_train(train,val,test)

    dataset = args.dataset

    np.save(f"{args.out_dir}/{dataset}_seq{args.seq_len}_train_windows.npy",train)
    np.save(f"{args.out_dir}/{dataset}_seq{args.seq_len}_val_windows.npy",val)
    np.save(f"{args.out_dir}/{dataset}_seq{args.seq_len}_test_windows.npy",test)

    print("windows saved:",train.shape,val.shape,test.shape)

    for drop in [1,2]:

        for split,data in [("val",val),("test",test)]:

            eval_mask, dropped = make_channel_drop_maskbank(
                len(data),
                args.seq_len,
                data.shape[-1],
                drop,
                args.seed
            )

            path = f"{args.out_dir}/{dataset}_seq{args.seq_len}_{split}_drop{drop}_seed{args.seed}.npz"

            metadata=dict(
                dataset=dataset,
                seq_len=args.seq_len,
                drop_channels=drop,
                seed=args.seed
            )

            save_maskbank(path,eval_mask,dropped,metadata)

            print("saved:",path)


if __name__=="__main__":
    main()

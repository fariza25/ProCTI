import os
import json
import argparse
import numpy as np
import pandas as pd


BEIJING_FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]


def load_beijing_station_files(data_dir: str):
    station_dfs = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)

        # Build datetime for stable chronological ordering
        if all(c in df.columns for c in ["year", "month", "day", "hour"]):
            dt = pd.to_datetime(
                dict(
                    year=df["year"],
                    month=df["month"],
                    day=df["day"],
                    hour=df["hour"],
                ),
                errors="coerce",
            )
            df = df.loc[~dt.isna()].copy()
            df["_dt"] = pd.to_datetime(
                dict(
                    year=df["year"],
                    month=df["month"],
                    day=df["day"],
                    hour=df["hour"],
                ),
                errors="coerce",
            )
            df = df.sort_values("_dt", kind="mergesort").drop(columns=["_dt"]).reset_index(drop=True)

        missing = [c for c in BEIJING_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"{fname} is missing required columns: {missing}")

        sub = df[BEIJING_FEATURES].copy()

        # numeric coercion
        for c in BEIJING_FEATURES:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        # fill within station
        sub = sub.ffill().bfill()

        # drop rows still containing NaNs
        sub = sub.dropna(axis=0, how="any")

        if len(sub) == 0:
            continue

        station_dfs.append(sub)

    if len(station_dfs) == 0:
        raise ValueError(f"No valid Beijing station files found in {data_dir}")

    return station_dfs


def make_windows_per_station(station_dfs, seq_len: int):
    windows = []

    for df in station_dfs:
        X = df.values.astype(np.float32)
        n_win = len(X) // seq_len
        if n_win <= 0:
            continue
        windows.append(X[: n_win * seq_len].reshape(n_win, seq_len, X.shape[1]))

    if len(windows) == 0:
        raise ValueError(f"No Beijing windows could be formed with seq_len={seq_len}")

    return np.concatenate(windows, axis=0).astype(np.float32)


def chrono_split_windows(windows, train_ratio=0.70, val_ratio=0.15):
    n = len(windows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = windows[:n_train]
    val = windows[n_train:n_train + n_val]
    test = windows[n_train + n_val:]
    return train, val, test


def standardize_by_train(train, val, test, eps=1e-6):
    train_flat = train.reshape(-1, train.shape[-1])

    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)
    if valid.sum() == 0:
        raise ValueError("All Beijing features were dropped during standardization.")

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
            raise ValueError("Non-finite values after Beijing standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid


def make_channel_drop_maskbank(windows, n_drop, seed):
    rng = np.random.default_rng(seed)
    N, L, K = windows.shape

    if K < n_drop:
        raise ValueError(f"Cannot create drop{n_drop} maskbank with only K={K} features.")

    eval_mask = np.zeros((N, L, K), dtype=np.float32)
    dropped = np.zeros((N, n_drop), dtype=np.int64)

    for i in range(N):
        ch = rng.choice(K, size=n_drop, replace=False)
        dropped[i] = ch
        eval_mask[i, :, ch] = 1.0

    return eval_mask, dropped


def save_maskbank(path, eval_mask, dropped_channels, metadata):
    np.savez_compressed(
        path,
        eval_mask=eval_mask.astype(np.float32),
        dropped_channels=dropped_channels.astype(np.int64),
        metadata_json=json.dumps(metadata, indent=2),
    )
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    station_dfs = load_beijing_station_files(args.data_dir)
    windows = make_windows_per_station(station_dfs, args.seq_len)

    train, val, test = chrono_split_windows(windows)
    train, val, test, mu, sd, valid = standardize_by_train(train, val, test)

    feat_cols_kept = [c for c, keep in zip(BEIJING_FEATURES, valid.tolist()) if keep]
    K = train.shape[-1]

    print(f"[data] windows train/val/test={len(train)}/{len(val)}/{len(test)}  K={K} L={args.seq_len}")
    print(f"[features kept] {feat_cols_kept}")

    np.save(os.path.join(args.out_dir, f"beijing_seq{args.seq_len}_train_windows.npy"), train)
    np.save(os.path.join(args.out_dir, f"beijing_seq{args.seq_len}_val_windows.npy"), val)
    np.save(os.path.join(args.out_dir, f"beijing_seq{args.seq_len}_test_windows.npy"), test)

    metadata = {
        "dataset": "beijing",
        "seq_len": int(args.seq_len),
        "seed": int(args.seed),
        "feature_cols_requested": [str(c) for c in BEIJING_FEATURES],
        "feature_cols_kept": [str(c) for c in feat_cols_kept],
        "mu": [float(x) for x in mu.tolist()],
        "sd": [float(x) for x in sd.tolist()],
    }
    with open(os.path.join(args.out_dir, f"beijing_seq{args.seq_len}_metadata_seed{args.seed}.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    for n_drop in [1, 2]:
        val_mask, val_drop = make_channel_drop_maskbank(val, n_drop, args.seed)
        test_mask, test_drop = make_channel_drop_maskbank(test, n_drop, args.seed)

        meta = {
            "dataset": "beijing",
            "seq_len": int(args.seq_len),
            "seed": int(args.seed),
            "protocol": f"drop{n_drop}",
            "feature_cols_kept": [str(c) for c in feat_cols_kept],
        }

        save_maskbank(
            os.path.join(args.out_dir, f"beijing_seq{args.seq_len}_val_drop{n_drop}_seed{args.seed}.npz"),
            val_mask,
            val_drop,
            {**meta, "split": "val"},
        )
        save_maskbank(
            os.path.join(args.out_dir, f"beijing_seq{args.seq_len}_test_drop{n_drop}_seed{args.seed}.npz"),
            test_mask,
            test_drop,
            {**meta, "split": "test"},
        )

    print("Done.")


if __name__ == "__main__":
    main()

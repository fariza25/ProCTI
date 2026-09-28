import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def detect_time_column(df: pd.DataFrame):
    candidates = ["date", "datetime", "time", "timestamp", "Date", "Datetime"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_weather_dataframe(csv_path: str):
    df = pd.read_csv(csv_path)

    time_col = detect_time_column(df)
    if time_col is not None:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col).reset_index(drop=True)
        except Exception:
            # if parsing fails, just keep original order
            pass

    # keep only numeric feature columns except time column
    feature_cols = []
    for c in df.columns:
        if c == time_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found in weather CSV.")

    data_df = df[feature_cols].copy()
    return data_df, feature_cols, time_col


def chronological_row_split(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df


def fit_and_transform(train_df, val_df, test_df):
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df.values),
        columns=train_df.columns,
        index=train_df.index,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df.values),
        columns=val_df.columns,
        index=val_df.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df.values),
        columns=test_df.columns,
        index=test_df.index,
    )
    return scaler, train_scaled, val_scaled, test_scaled


def make_nonoverlapping_windows(arr: np.ndarray, seq_len: int):
    """
    arr: [T, K]
    returns windows: [N, L, K]
    """
    T, K = arr.shape
    n_win = T // seq_len
    trimmed = arr[: n_win * seq_len]
    windows = trimmed.reshape(n_win, seq_len, K)
    return windows


def build_observed_mask(windows: np.ndarray):
    # 1 where finite, else 0
    mask = np.isfinite(windows).astype(np.float32)
    return mask


def impute_for_storage(windows: np.ndarray):
    """
    For storage consistency, replace NaN/Inf with 0.0.
    Models should still rely on observed_mask for true missingness.
    """
    x = np.array(windows, dtype=np.float32, copy=True)
    bad = ~np.isfinite(x)
    x[bad] = 0.0
    return x


def make_channel_drop_maskbank(
    windows: np.ndarray,
    observed_mask: np.ndarray,
    n_drop: int,
    seed: int,
):
    """
    windows: [N, L, K]
    observed_mask: [N, L, K], 1 for originally observed positions
    returns:
      eval_mask: [N, L, K], 1 where hidden for evaluation
      dropped_channels: [N, n_drop]
    """
    rng = np.random.default_rng(seed)
    N, L, K = windows.shape

    eval_mask = np.zeros((N, L, K), dtype=np.float32)
    dropped_channels = np.full((N, n_drop), -1, dtype=np.int64)

    for i in range(N):
        # valid channels = those with at least one observed value in that window
        valid_channels = np.where(observed_mask[i].sum(axis=0) > 0)[0]

        if len(valid_channels) == 0:
            continue

        actual_n_drop = min(n_drop, len(valid_channels))
        chosen = rng.choice(valid_channels, size=actual_n_drop, replace=False)

        dropped_channels[i, :actual_n_drop] = chosen
        eval_mask[i, :, chosen] = 1.0

        # only evaluate on originally observed positions
        eval_mask[i] *= observed_mask[i]

    return eval_mask, dropped_channels


def save_npy_assets(out_dir, prefix, windows, observed_mask):
    np.save(os.path.join(out_dir, f"{prefix}_windows.npy"), windows.astype(np.float32))
    np.save(os.path.join(out_dir, f"{prefix}_observed_mask.npy"), observed_mask.astype(np.float32))
    print(f"[saved] {prefix}_windows.npy       shape={windows.shape}")
    print(f"[saved] {prefix}_observed_mask.npy shape={observed_mask.shape}")


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
    parser = argparse.ArgumentParser(description="Prepare weather window assets + channel-drop maskbanks.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to weather CSV")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save windows/masks/maskbanks")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load dataframe
    # ------------------------------------------------------------------
    df, feature_cols, time_col = load_weather_dataframe(args.csv)
    print(f"[data] rows={len(df)} features={len(feature_cols)} time_col={time_col}")

    # ------------------------------------------------------------------
    # 2) Chronological split (row-wise)
    # ------------------------------------------------------------------
    train_df, val_df, test_df = chronological_row_split(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"[split-rows] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ------------------------------------------------------------------
    # 3) Scale using train only
    # ------------------------------------------------------------------
    scaler, train_scaled, val_scaled, test_scaled = fit_and_transform(train_df, val_df, test_df)

    # Save scaler stats for reproducibility
    scaler_meta = {
        "feature_cols": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "seq_len": args.seq_len,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "time_col": time_col,
    }
    with open(os.path.join(args.out_dir, "scaler_metadata.json"), "w") as f:
        json.dump(scaler_meta, f, indent=2)
    print(f"[saved] {os.path.join(args.out_dir, 'scaler_metadata.json')}")

    # ------------------------------------------------------------------
    # 4) Windowing (non-overlapping)
    # ------------------------------------------------------------------
    train_arr = train_scaled.values.astype(np.float32)
    val_arr = val_scaled.values.astype(np.float32)
    test_arr = test_scaled.values.astype(np.float32)

    train_windows = make_nonoverlapping_windows(train_arr, args.seq_len)
    val_windows = make_nonoverlapping_windows(val_arr, args.seq_len)
    test_windows = make_nonoverlapping_windows(test_arr, args.seq_len)

    train_obs = build_observed_mask(train_windows)
    val_obs = build_observed_mask(val_windows)
    test_obs = build_observed_mask(test_windows)

    # replace NaN/Inf with zeros for saved arrays; masks preserve original missingness
    train_windows = impute_for_storage(train_windows)
    val_windows = impute_for_storage(val_windows)
    test_windows = impute_for_storage(test_windows)

    print(
        f"[windows] train={len(train_windows)} val={len(val_windows)} test={len(test_windows)} "
        f"L={args.seq_len} K={train_windows.shape[-1]}"
    )

    # ------------------------------------------------------------------
    # 5) Save shared windows + observed masks
    # ------------------------------------------------------------------
    save_npy_assets(args.out_dir, f"weather_seq{args.seq_len}_train", train_windows, train_obs)
    save_npy_assets(args.out_dir, f"weather_seq{args.seq_len}_val", val_windows, val_obs)
    save_npy_assets(args.out_dir, f"weather_seq{args.seq_len}_test", test_windows, test_obs)

    # ------------------------------------------------------------------
    # 6) Create shared channel-drop maskbanks for val/test only
    # ------------------------------------------------------------------
    for n_drop in [1, 2]:
        val_eval_mask, val_dropped = make_channel_drop_maskbank(
            windows=val_windows,
            observed_mask=val_obs,
            n_drop=n_drop,
            seed=args.seed,
        )
        test_eval_mask, test_dropped = make_channel_drop_maskbank(
            windows=test_windows,
            observed_mask=test_obs,
            n_drop=n_drop,
            seed=args.seed,
        )

        common_meta = {
            "dataset": "weather",
            "seq_len": args.seq_len,
            "seed": args.seed,
            "protocol": f"drop{n_drop}",
            "mask_type": "full_channel_drop",
            "time_col": time_col,
            "feature_cols": feature_cols,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        }

        save_maskbank_npz(
            os.path.join(args.out_dir, f"weather_seq{args.seq_len}_val_drop{n_drop}_seed{args.seed}.npz"),
            val_eval_mask,
            val_dropped,
            {**common_meta, "split": "val"},
        )
        save_maskbank_npz(
            os.path.join(args.out_dir, f"weather_seq{args.seq_len}_test_drop{n_drop}_seed{args.seed}.npz"),
            test_eval_mask,
            test_dropped,
            {**common_meta, "split": "test"},
        )

    print("\nDone.")
    print("Use the saved val/test maskbanks in all model evaluation scripts with shuffle=False for val/test loaders.")


if __name__ == "__main__":
    main()

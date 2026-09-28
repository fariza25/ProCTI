import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]


def parse_args():
    p = argparse.ArgumentParser(description="Build shared Beijing station train/val/test windows.")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing 12 station CSV files.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for saved windows and metadata.")
    p.add_argument("--seq_len", type=int, default=96,
                   help="Window length.")
    p.add_argument("--stride", type=int, default=None,
                   help="Stride for windowing. Default = seq_len (non-overlapping).")
    p.add_argument("--split_seed", type=int, default=1,
                   help="Seed used only if explicit station split is not provided.")
    p.add_argument("--train_stations", type=str, default="",
                   help="Comma-separated station names (filename stems) for training.")
    p.add_argument("--val_stations", type=str, default="",
                   help="Comma-separated station names (filename stems) for validation.")
    p.add_argument("--test_stations", type=str, default="",
                   help="Comma-separated station names (filename stems) for testing.")
    p.add_argument("--dropna", action="store_true",
                   help="Drop rows with NaNs in selected feature columns before windowing.")
    p.add_argument("--allow_partial_nan_windows", action="store_true",
                   help="If set, do not force windows to be fully observed. By default, windows with any NaN are removed.")
    return p.parse_args()


def maybe_sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    time_cols = ["year", "month", "day", "hour"]
    lower_map = {c.lower(): c for c in df.columns}
    if all(c in lower_map for c in time_cols):
        cols = [lower_map[c] for c in time_cols]
        return df.sort_values(cols).reset_index(drop=True)

    alt_cols = ["date", "datetime", "timestamp", "time"]
    for c in alt_cols:
        if c in lower_map:
            real_c = lower_map[c]
            try:
                tmp = df.copy()
                tmp[real_c] = pd.to_datetime(tmp[real_c], errors="coerce")
                tmp = tmp.sort_values(real_c).reset_index(drop=True)
                return tmp
            except Exception:
                pass
    return df.reset_index(drop=True)


def load_station_csv(csv_path: str, features):
    df = pd.read_csv(csv_path)
    df = maybe_sort_by_time(df)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    x = df[features].copy()
    for c in features:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def split_station_names(station_names, split_seed, train_stations, val_stations, test_stations):
    if train_stations or val_stations or test_stations:
        train = [s.strip() for s in train_stations.split(",") if s.strip()]
        val = [s.strip() for s in val_stations.split(",") if s.strip()]
        test = [s.strip() for s in test_stations.split(",") if s.strip()]

        combined = train + val + test
        if len(combined) != len(set(combined)):
            raise ValueError("Explicit station split contains duplicate station names.")

        missing = sorted(set(combined) - set(station_names))
        if missing:
            raise ValueError(f"These explicit station names were not found: {missing}")

        if len(train) != 8 or len(val) != 2 or len(test) != 2:
            raise ValueError("Explicit split must contain exactly 8 train, 2 val, and 2 test stations.")

        if set(combined) != set(station_names):
            leftover = sorted(set(station_names) - set(combined))
            raise ValueError(f"Explicit split does not cover all station files. Leftover: {leftover}")

        return train, val, test

    rng = np.random.RandomState(split_seed)
    station_names = sorted(station_names)
    perm = list(rng.permutation(station_names))
    train = perm[:8]
    val = perm[8:10]
    test = perm[10:12]
    return train, val, test


def make_windows_from_array(arr: np.ndarray, seq_len: int, stride: int, allow_partial_nan_windows: bool):
    windows = []
    n = len(arr)
    for start in range(0, n - seq_len + 1, stride):
        w = arr[start:start + seq_len]
        if not allow_partial_nan_windows:
            if np.isnan(w).any():
                continue
        windows.append(w)
    if not windows:
        return np.empty((0, seq_len, arr.shape[1]), dtype=np.float32)
    return np.stack(windows).astype(np.float32)


def main():
    args = parse_args()
    stride = args.seq_len if args.stride is None else args.stride
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if len(csv_files) != 12:
        raise ValueError(f"Expected 12 CSV files, found {len(csv_files)} in {args.data_dir}")

    station_to_df = {}
    station_names = []

    for fp in csv_files:
        station = Path(fp).stem
        station_names.append(station)
        df = load_station_csv(fp, DEFAULT_FEATURES)
        if args.dropna:
            df = df.dropna(subset=DEFAULT_FEATURES).reset_index(drop=True)
        station_to_df[station] = df

    train_stations, val_stations, test_stations = split_station_names(
        station_names=station_names,
        split_seed=args.split_seed,
        train_stations=args.train_stations,
        val_stations=args.val_stations,
        test_stations=args.test_stations,
    )

    # Fit scaler on concatenated train rows only
    train_rows = []
    for s in train_stations:
        df = station_to_df[s]
        if len(df) == 0:
            raise ValueError(f"Training station {s} has 0 usable rows.")
        train_rows.append(df[DEFAULT_FEATURES].to_numpy(dtype=np.float32))

    train_concat = np.concatenate(train_rows, axis=0)
    if not args.allow_partial_nan_windows:
        train_concat_no_nan = train_concat[~np.isnan(train_concat).any(axis=1)]
        if len(train_concat_no_nan) == 0:
            raise ValueError("No fully observed training rows remain for fitting the scaler.")
        scaler = StandardScaler().fit(train_concat_no_nan)
    else:
        # fallback: fit on rows with no NaNs
        valid_rows = train_concat[~np.isnan(train_concat).any(axis=1)]
        if len(valid_rows) == 0:
            raise ValueError("No non-NaN training rows available to fit scaler.")
        scaler = StandardScaler().fit(valid_rows)

    split_windows = {}
    station_window_counts = {"train": {}, "val": {}, "test": {}}

    for split_name, stations in [("train", train_stations), ("val", val_stations), ("test", test_stations)]:
        per_station_windows = []

        for station in stations:
            df = station_to_df[station].copy()
            arr = df[DEFAULT_FEATURES].to_numpy(dtype=np.float32)
            arr_scaled = scaler.transform(arr)

            windows = make_windows_from_array(
                arr_scaled,
                seq_len=args.seq_len,
                stride=stride,
                allow_partial_nan_windows=args.allow_partial_nan_windows,
            )

            station_window_counts[split_name][station] = int(len(windows))
            if len(windows) > 0:
                per_station_windows.append(windows)

        if per_station_windows:
            split_windows[split_name] = np.concatenate(per_station_windows, axis=0).astype(np.float32)
        else:
            split_windows[split_name] = np.empty((0, args.seq_len, len(DEFAULT_FEATURES)), dtype=np.float32)

    train_w = split_windows["train"]
    val_w = split_windows["val"]
    test_w = split_windows["test"]

    if len(train_w) == 0:
        raise ValueError("No training windows were created.")
    if len(val_w) == 0:
        raise ValueError("No validation windows were created.")
    if len(test_w) == 0:
        raise ValueError("No test windows were created.")

    np.save(out_dir / "train_windows.npy", train_w)
    np.save(out_dir / "val_windows.npy", val_w)
    np.save(out_dir / "test_windows.npy", test_w)

    np.savez(
        out_dir / "scaler_stats.npz",
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        var=scaler.var_.astype(np.float32),
    )

    meta = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "seq_len": int(args.seq_len),
        "stride": int(stride),
        "features": DEFAULT_FEATURES,
        "num_features": len(DEFAULT_FEATURES),
        "dropna_rows_before_windowing": bool(args.dropna),
        "allow_partial_nan_windows": bool(args.allow_partial_nan_windows),
        "split_seed": int(args.split_seed),
        "train_stations": train_stations,
        "val_stations": val_stations,
        "test_stations": test_stations,
        "shape_train": list(train_w.shape),
        "shape_val": list(val_w.shape),
        "shape_test": list(test_w.shape),
        "station_window_counts": station_window_counts,
    }

    with open(out_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(DEFAULT_FEATURES, f, indent=2)

    print("Saved:")
    print(f"  {out_dir / 'train_windows.npy'}  shape={train_w.shape}")
    print(f"  {out_dir / 'val_windows.npy'}    shape={val_w.shape}")
    print(f"  {out_dir / 'test_windows.npy'}   shape={test_w.shape}")
    print(f"  {out_dir / 'scaler_stats.npz'}")
    print(f"  {out_dir / 'split_meta.json'}")
    print()
    print("Station split:")
    print("  train:", train_stations)
    print("  val  :", val_stations)
    print("  test :", test_stations)


if __name__ == "__main__":
    main()

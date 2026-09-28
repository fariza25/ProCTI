#!/usr/bin/env python3

import os
import argparse
import random
from typing import Optional, Tuple, List, Dict, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.SCINet import SCINet


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_batch_size(requested: int, n_items: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(int(requested), int(n_items)))


def parse_ratios(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# -----------------------------
# Markov keep-mask (segment-based)
# keepmask: 1=observed/kept, 0=masked
# -----------------------------
def markov_keep_mask_from_masked_ratio(
    B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device
) -> torch.Tensor:
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # state 0 = masked, state 1 = keep
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]

    out = np.ones((B, K, L), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < p[state]:
                    state = 1 - state
    return torch.from_numpy(out).to(device=device)


# -----------------------------
# PhysioNet helpers
# -----------------------------
def load_physionet_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    pid_col = "Patient_ID"
    if pid_col not in df.columns:
        raise ValueError("Patient_ID column not found in CSV.")

    df[pid_col] = pd.to_numeric(df[pid_col], errors="coerce")
    df = df.dropna(subset=[pid_col])
    df[pid_col] = df[pid_col].astype(int)

    
    for _alias in ["patient_id", "PatientID", "patient", "RecordID", "id", "UID", "stay_id"]:
        if _alias not in df.columns:
            df[_alias] = df[pid_col]
    return df


def get_time_col(df: pd.DataFrame) -> str:
    if "ICULOS" in df.columns:
        return "ICULOS"
    if "Hour" in df.columns:
        return "Hour"
    raise ValueError("Expected a time column ICULOS or Hour in physionet2019.csv.")


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_cols = {"Patient_ID"}
    for c in ["SepsisLabel", "Hour", "ICULOS", "HospAdmTime"]:
        if c in df.columns:
            drop_cols.add(c)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns found after dropping id/labels/time columns.")
    return feat_cols


def split_patients(df: pd.DataFrame, seed: int, train_ratio=0.7, val_ratio=0.15) -> Tuple[Set[int], Set[int], Set[int]]:
    pids = df["Patient_ID"].dropna().unique().tolist()
    pids = [int(x) for x in pids]
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_p = set(pids[:n_train])
    val_p = set(pids[n_train:n_train + n_val])
    test_p = set(pids[n_train + n_val:])
    return train_p, val_p, test_p


def make_windows_for_patients(
    df: pd.DataFrame,
    patient_ids: Set[int],
    feat_cols: List[str],
    time_col: str,
    seq_len: int,
) -> np.ndarray:
    patient_ids = set(int(x) for x in patient_ids)
    windows = []
    for pid, g in df.groupby("Patient_ID", sort=False):
        pid = int(pid)
        if pid not in patient_ids:
            continue
        g = g.sort_values(time_col, kind="mergesort")
        X = g[feat_cols].astype(np.float32).ffill().bfill()
        arr = X.to_numpy(dtype=np.float32)
        arr[~np.isfinite(arr)] = np.nan

        T, K = arr.shape
        n_win = T // seq_len
        if n_win <= 0:
            continue
        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, K))

    if not windows:
        raise ValueError("No windows formed for this split (check seq_len or patient IDs).")
    return np.concatenate(windows, axis=0).astype(np.float32)


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    """
    Compute train mean/std; drop invalid features; fill NaNs w/ train mean; z-score.

    """
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)
    if valid.sum() == 0:
        raise ValueError("All features are non-finite / zero-std after TRAIN stats. Check CSV.")

    train = train[:, :, valid]
    val = val[:, :, valid]
    test = test[:, :, valid]
    mu = mu[valid].astype(np.float32)
    sd = np.maximum(sd[valid], eps).astype(np.float32)

    def fill_and_z(x):
        x = x.copy().astype(np.float32)
        nanmask = ~np.isfinite(x)
        if nanmask.any():
            ks = np.where(nanmask)[2]
            x[nanmask] = np.take(mu, ks)
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid


# -----------------------------
# Maskbank metadata 
# -----------------------------
def load_maskbank_metadata(mask_dir: str, seq_len: int, seed: int) -> Dict[str, np.ndarray]:
    meta_path = os.path.join(mask_dir, f"physionet_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Maskbank metadata not found: {meta_path}")
    z = np.load(meta_path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def standardize_with_maskbank(train_raw: np.ndarray, val_raw: np.ndarray, test_raw: np.ndarray, meta: Dict[str, np.ndarray]):
    """
    Option-A: use valid_feature_mask + train_mu_raw/train_sd_raw for scaling and feature filtering.
    """
    valid = meta["valid_feature_mask"].astype(bool)
    mu_raw = meta["train_mu_raw"].astype(np.float32)
    sd_raw = meta["train_sd_raw"].astype(np.float32)
    eps = float(meta["eps"]) if "eps" in meta else 1e-6

    if train_raw.shape[-1] != len(valid):
        raise ValueError(
            f"[maskbank] K_raw mismatch: data K={train_raw.shape[-1]} but valid_feature_mask len={len(valid)}. "
            "Ensure you used maskbank feature_cols_raw ordering when loading."
        )

    mu = mu_raw[valid]
    sd = np.maximum(sd_raw[valid], eps)

    def fill_and_z(x):
        x = x[:, :, valid].copy().astype(np.float32)
        nanmask = ~np.isfinite(x)
        if nanmask.any():
            ks = np.where(nanmask)[2]
            x[nanmask] = np.take(mu, ks)
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization (maskbank scaling).")
        return x.astype(np.float32)

    return fill_and_z(train_raw), fill_and_z(val_raw), fill_and_z(test_raw), mu.astype(np.float32), sd.astype(np.float32), valid


def load_shared_keepmask(
    mask_dir: str,
    split: str,
    r_m: float,
    seq_len: int,
    seed: int,
    invert: bool,
    *,
    valid_feature_mask: Optional[np.ndarray] = None,
    expected_K: Optional[int] = None,
) -> np.ndarray:

    path = os.path.join(mask_dir, f"physionet_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if keep.ndim != 3:
        raise ValueError(f"Shared keepmask must have shape (N,L,K), got {keep.shape} from {path}")

    if invert or os.environ.get("PHYSIONET_MASKBANK_INVERT", "0") == "1":
        keep = 1.0 - keep

    if expected_K is None:
        return keep
    if int(keep.shape[2]) == int(expected_K):
        return keep

    if valid_feature_mask is not None:
        valid_feature_mask = np.asarray(valid_feature_mask).astype(bool)
        K_full = int(len(valid_feature_mask))
        K_kept = int(valid_feature_mask.sum())
        if int(expected_K) != K_kept:
            raise ValueError(f"Internal mismatch: expected_K={expected_K} but valid_feature_mask.sum()={K_kept}.")
        if int(keep.shape[2]) == K_full:
            keep2 = keep[:, :, valid_feature_mask]
            if int(keep2.shape[2]) != int(expected_K):
                raise ValueError(f"After slicing keepmask K={keep2.shape[2]} != expected_K={expected_K}. File: {path}")
            return keep2

    raise ValueError(
        f"Shared keepmask K mismatch: maskbank K={keep.shape[2]} but model expects K={expected_K}. "
        f"File: {path}. Fix: regenerate maskbank with matching preprocessing, or store masks in full feature space."
    )


# -----------------------------
# Dataset returning index 
# -----------------------------
class WindowDatasetWithIndex(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), int(i)  # (L,K), idx


# -----------------------------
# SCINet wrapper 
# -----------------------------
class SCINetWrapper(nn.Module):
    def __init__(self, K: int, seq_len: int, num_levels: int = 3, num_stacks: int = 1):
        super().__init__()
        self.model = SCINet(
            output_len=seq_len,
            input_len=seq_len,
            input_dim=K,
            hid_size=1,
            num_stacks=num_stacks,
            num_levels=num_levels,
        )

    def forward(self, x_BLK: torch.Tensor) -> torch.Tensor:
        return self.model(x_BLK)


# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, opt, device, lm: float, r_train_masked: float):
    model.train()
    total = 0.0

    for x, _idx in loader:
        x = x.to(device)  # (B,L,K)
        B, L, K = x.shape

        keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_train_masked, lm, device)  # (B,K,L)
        keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()  # (B,L,K)

        pred = model(x * keep_BLK)

        evalmask = 1.0 - keep_BLK
        denom = evalmask.sum().clamp_min(1.0)
        loss = ((pred - x) ** 2 * evalmask).sum() / denom

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def eval_markov(model, loader, device, lm: float, r_masked: float):
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    denom = 0.0

    for x, _idx in loader:
        x = x.to(device)
        B, L, K = x.shape

        keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_masked, lm, device)
        keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

        pred = model(x * keep_BLK)
        evalmask = 1.0 - keep_BLK
        diff = (pred - x) * evalmask

        sum_abs += diff.abs().sum().item()
        sum_sq += (diff ** 2).sum().item()
        denom += evalmask.sum().item()

    denom = max(1.0, denom)
    mae = sum_abs / denom
    mse = sum_sq / denom
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse


@torch.no_grad()
def eval_shared(model, loader, device, keep_all_NLK: np.ndarray):
    """
    keep_all_NLK: (N,L,K) keepmask, 1=kept 0=masked
    """
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    denom = 0.0

    keep_all = torch.from_numpy(keep_all_NLK).to(device)

    for x, idx in loader:
        x = x.to(device)
        idx = idx.to(device)

        keep_BLK = keep_all.index_select(0, idx)  # (B,L,K)

        pred = model(x * keep_BLK)
        evalmask = 1.0 - keep_BLK
        diff = (pred - x) * evalmask

        sum_abs += diff.abs().sum().item()
        sum_sq += (diff ** 2).sum().item()
        denom += evalmask.sum().item()

    denom = max(1.0, denom)
    mae = sum_abs / denom
    mse = sum_sq / denom
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # masking
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # shared eval masks (Option A)
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_physionet_maskbank")
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    # output
    ap.add_argument("--out_txt", type=str, default="scinet_physionet_5runs_metrics.txt")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # -----------------------------
    # Load + split
    # -----------------------------
    df = load_physionet_df(args.csv)
    time_col = get_time_col(df)

    if args.use_shared_evalmask:
        meta = load_maskbank_metadata(args.shared_evalmask_dir, args.seq_len, args.seed)

        feature_cols_raw = [str(c) for c in list(meta["feature_cols_raw"])]
        # maskbank uses the exact patient splits it saved
        train_p = set(int(x) for x in meta["patients_train"])
        val_p = set(int(x) for x in meta["patients_val"])
        test_p = set(int(x) for x in meta["patients_test"])

        # build windows in RAW feature space using feature_cols_raw ordering
        train_raw = make_windows_for_patients(df, train_p, feature_cols_raw, time_col, args.seq_len)
        val_raw = make_windows_for_patients(df, val_p, feature_cols_raw, time_col, args.seq_len)
        test_raw = make_windows_for_patients(df, test_p, feature_cols_raw, time_col, args.seq_len)

        train, val, test, _mu, _sd, valid_feature_mask = standardize_with_maskbank(train_raw, val_raw, test_raw, meta)

        print(f"[OptionA] windows train/val/test = {len(train)}/{len(val)}/{len(test)}  L={args.seq_len}  K={train.shape[-1]}  time_col={time_col}")

    else:
        feat_cols = get_feature_cols(df)

        train_p, val_p, test_p = split_patients(df, seed=args.seed)

        train_raw = make_windows_for_patients(df, train_p, feat_cols, time_col, args.seq_len)
        val_raw = make_windows_for_patients(df, val_p, feat_cols, time_col, args.seq_len)
        test_raw = make_windows_for_patients(df, test_p, feat_cols, time_col, args.seq_len)

        train, val, test, _mu, _sd, _valid = standardize_by_train(train_raw, val_raw, test_raw)

        print(f"[Split] windows train/val/test = {len(train)}/{len(val)}/{len(test)}  L={args.seq_len}  K={train.shape[-1]}  time_col={time_col}")

    # -----------------------------
    # Loaders
    # -----------------------------
    train_loader = DataLoader(
        WindowDatasetWithIndex(train),
        batch_size=safe_batch_size(args.batch, len(train)),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        WindowDatasetWithIndex(val),
        batch_size=safe_batch_size(args.batch, len(val)),
        shuffle=False,
    )
    test_loader = DataLoader(
        WindowDatasetWithIndex(test),
        batch_size=safe_batch_size(args.batch, len(test)),
        shuffle=False,
    )

    # -----------------------------
    # Model
    # -----------------------------
    K = int(train.shape[-1])
    model = SCINetWrapper(K, args.seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------
    # Train
    # -----------------------------
    for ep in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, device, args.lm, args.r_train_masked)
        print(f"[train] epoch={ep:03d} loss={loss:.6f}")

    # -----------------------------
    # Eval + log
    # -----------------------------
    ratios = parse_ratios(args.eval_masked_ratios)

    file_exists = os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if not file_exists:
            f.write("seed\tsplit\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split, loader, n_expected in [
            ("val", val_loader, len(val)),
            ("test", test_loader, len(test)),
        ]:
            for r in ratios:
                if args.use_shared_evalmask:
                    keep = load_shared_keepmask(
                        args.shared_evalmask_dir,
                        split,
                        r,
                        args.seq_len,
                        args.seed,
                        args.invert_shared_keepmask,
                        valid_feature_mask=valid_feature_mask,
                        expected_K=K,
                    )
                    if keep.shape[0] != n_expected:
                        raise ValueError(f"Shared keepmask N mismatch for {split}: keep.N={keep.shape[0]} {split}.N={n_expected}")

                    mae, mse, rmse = eval_shared(model, loader, device, keep)
                else:
                    mae, mse, rmse = eval_markov(model, loader, device, args.lm, r)

                f.write(f"{args.seed}\t{split}\t{r:.2f}\t{1-r:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{split}] r_masked={r:.2f} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()

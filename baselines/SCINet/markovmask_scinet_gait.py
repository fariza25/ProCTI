#!/usr/bin/env python3

import os
import re
import argparse
import random
from typing import List, Dict, Tuple

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
# GAIT data helpers 
# -----------------------------
USER_RE = re.compile(r"_ID(\d+)_", re.IGNORECASE)


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
    # automatic-ou-gaitdata: csv with two header rows
    df = pd.read_csv(path, skiprows=2, header=None)
    arr = df.values.astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Bad array shape from {path}: {arr.shape}")
    arr[~np.isfinite(arr)] = np.nan
    return arr


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
        raise ValueError("No windows formed (seq_len may be too large or files too short).")
    return np.concatenate(windows, axis=0).astype(np.float32)


def split_users(files: List[str], seed: int, train_ratio=0.7, val_ratio=0.15):
    by_user: Dict[str, List[str]] = {}
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
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.maximum(sd, eps).astype(np.float32)

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

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd


# -----------------------------
# Dataset 
# -----------------------------
class WindowDatasetWithIndex(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), int(idx)  # (L,K), index


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

        x_masked = x * keep_BLK
        pred = model(x_masked)

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


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True, help="automatic-ou-gaitdata directory with per-trial CSVs")
    ap.add_argument("--seq_len", type=int, default=64)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    ap.add_argument("--out_txt", type=str, default="scinet_gait_5runs_metrics.txt")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # -----------------------------
    # Data
    # -----------------------------
    files = list_gait_files(args.data_dir)
    train_files, val_files, test_files, (train_u, val_u, test_u) = split_users(files, seed=args.seed)

    train = windows_from_files(train_files, args.seq_len)
    val = windows_from_files(val_files, args.seq_len)
    test = windows_from_files(test_files, args.seq_len)

    train, val, test, _mu, _sd = standardize_by_train(train, val, test)

    print(f"[GAIT] users train/val/test = {len(train_u)}/{len(val_u)}/{len(test_u)}")
    print(f"[GAIT] windows train/val/test = {len(train)}/{len(val)}/{len(test)}  L={args.seq_len}  K={train.shape[-1]}")

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

        for split, loader in [("val", val_loader), ("test", test_loader)]:
            for r in ratios:
                mae, mse, rmse = eval_markov(model, loader, device, args.lm, r)
                f.write(f"{args.seed}\t{split}\t{r:.2f}\t{1-r:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{split}] r_masked={r:.2f} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()

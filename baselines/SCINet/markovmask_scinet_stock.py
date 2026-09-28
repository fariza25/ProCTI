#!/usr/bin/env python3

import os
import argparse
import random
from typing import Optional, List, Tuple

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
# Markov keep-mask (segment-based) with correct stationary distribution
# keepmask: 1=observed/kept, 0=masked
# -----------------------------
def markov_keep_mask_from_masked_ratio(
    B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device
) -> torch.Tensor:
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # state 0 = masked, state 1 = keep
    p_m = 1.0 / lm  # P(0->1)
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)  # P(1->0) => stationary keep=r_keep
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
# Stock pipeline (row split then window), with optional maskbank metadata
# -----------------------------
def load_maskbank_meta(shared_dir: str, seq_len: int, seed: int):
    meta_path = os.path.join(shared_dir, f"stock_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Maskbank metadata not found: {meta_path}")
    meta = np.load(meta_path, allow_pickle=True)
    return meta_path, meta


def load_stock_csv_ordered(csv_path: str, feature_cols_raw: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.columns = df.columns.astype(str)

    if feature_cols_raw is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols) == 0:
            raise ValueError("No numeric columns found in stock CSV.")
    else:
        cols = [str(c) for c in list(feature_cols_raw)]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing {len(missing)} columns required by maskbank metadata. Example: {missing[:5]}")

    X = df[cols].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X.astype(np.float32), cols


def split_rows_then_window(
    X_TK: np.ndarray, seq_len: int, train_end: int, val_end: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xtr = X_TK[:train_end]
    Xva = X_TK[train_end:val_end]
    Xte = X_TK[val_end:]

    def to_windows(seg: np.ndarray) -> np.ndarray:
        T, K = seg.shape
        n = T // seq_len
        if n <= 0:
            return np.zeros((0, seq_len, K), dtype=np.float32)
        return seg[: n * seq_len].reshape(n, seq_len, K).astype(np.float32)

    return to_windows(Xtr), to_windows(Xva), to_windows(Xte)


def standardize_by_train_simple(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0).astype(np.float32)
    sd = np.nanstd(train_flat, axis=0).astype(np.float32)

    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.maximum(sd, eps).astype(np.float32)

    def fill_and_z(x):
        x = x.copy().astype(np.float32)
        nanmask = ~np.isfinite(x)
        if nanmask.any():
            feat_idx = np.where(nanmask)[2]
            x[nanmask] = mu[feat_idx]
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd


def optionA_standardize_and_keep(train_raw: np.ndarray, val_raw: np.ndarray, test_raw: np.ndarray, meta, eps_default: float = 1e-6):

    valid = meta["valid_feature_mask"].astype(bool)
    mu_raw = meta["train_mu_raw"].astype(np.float32)
    sd_raw = meta["train_sd_raw"].astype(np.float32)
    eps = float(meta["eps"]) if "eps" in meta.files else float(eps_default)

    mu = mu_raw[valid]
    sd = sd_raw[valid]
    sd = np.where(sd > eps, sd, 1.0).astype(np.float32)

    def transform(w: np.ndarray) -> np.ndarray:
        w = w[:, :, valid].astype(np.float32)
        nan = ~np.isfinite(w)
        if nan.any():
            w[nan] = np.take(mu, np.where(nan)[2])
        w = (w - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(w).all():
            raise ValueError("Non-finite after standardization (Option A).")
        return w.astype(np.float32)

    return transform(train_raw), transform(val_raw), transform(test_raw), mu, sd, valid


def load_shared_keepmask(shared_dir: str, split: str, r_masked: float, seq_len: int, seed: int) -> np.ndarray:
    path = os.path.join(shared_dir, f"stock_{split}_keepmask_r{r_masked:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if keep.ndim != 3:
        raise ValueError(f"Shared keepmask must have shape (N,L,K), got {keep.shape}")
    return keep


def align_keepmask_K(
    keep_NLK: np.ndarray,
    expected_K: int,
    valid_feature_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Align shared keepmask channels to expected_K.
    - If keep.K == expected_K: ok
    - Else if valid_feature_mask provided and keep.K == len(valid_feature_mask): slice keep[..., valid]
    """
    K_keep = int(keep_NLK.shape[2])
    if K_keep == int(expected_K):
        return keep_NLK
    if valid_feature_mask is not None:
        valid_feature_mask = np.asarray(valid_feature_mask).astype(bool)
        if K_keep == int(len(valid_feature_mask)):
            keep2 = keep_NLK[:, :, valid_feature_mask]
            if int(keep2.shape[2]) != int(expected_K):
                raise ValueError(f"After slicing keepmask K={keep2.shape[2]} != expected_K={expected_K}")
            return keep2
    raise ValueError(f"Shared keepmask K mismatch: keep.K={K_keep}, expected_K={expected_K}")


# -----------------------------
# Dataset that can return indices (needed for shared maskbank)
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
            num_levels=num_levels
        )

    def forward(self, x_BLK: torch.Tensor) -> torch.Tensor:
        return self.model(x_BLK)


# -----------------------------
# Train / Eval (masked-point loss & metrics)
# -----------------------------
def train_epoch(model, loader, opt, device, lm, r_train_masked: float):
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
def eval_markov(model, loader, device, lm, r_masked: float):
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
    keep_all_NLK: (N,L,K) float32 keepmask, 1=kept, 0=masked
    """
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    denom = 0.0

    keep_all = torch.from_numpy(keep_all_NLK).to(device)  # (N,L,K)

    for x, idx in loader:
        x = x.to(device)  # (B,L,K)
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
    ap.add_argument("--seq_len", type=int, default=48)

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
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_stock_maskbank")
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    # output
    ap.add_argument("--out_txt", type=str, default="scinet_stock_5runs_metrics.txt")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # -----------------------------
    # Load data 
    # -----------------------------
    valid_feature_mask = None
    feature_cols_raw = None

    if args.use_shared_evalmask:
        meta_path, meta = load_maskbank_meta(args.shared_evalmask_dir, args.seq_len, args.seed)

        feature_cols_raw = [str(c) for c in list(meta["feature_cols_raw"])]
        train_end = int(meta["split_row_train_end"])
        val_end = int(meta["split_row_val_end"])

        X_TK, _cols = load_stock_csv_ordered(args.csv, feature_cols_raw=feature_cols_raw)
        train_raw, val_raw, test_raw = split_rows_then_window(X_TK, args.seq_len, train_end, val_end)

        train, val, test, _mu, _sd, valid_feature_mask = optionA_standardize_and_keep(train_raw, val_raw, test_raw, meta)

        print(f"[OptionA] meta={meta_path}")
        print(f"[OptionA] rows={X_TK.shape[0]} train_end={train_end} val_end={val_end} windows={len(train)+len(val)+len(test)} K={train.shape[-1]}")

    else:
        X_TK, _cols = load_stock_csv_ordered(args.csv, feature_cols_raw=None)
        T = X_TK.shape[0]
        train_end = int(T * 0.70)
        val_end = int(T * 0.85)

        train_raw, val_raw, test_raw = split_rows_then_window(X_TK, args.seq_len, train_end, val_end)
        if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
            raise ValueError(f"Not enough data after split+window: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)} (seq_len={args.seq_len})")

        train, val, test, _mu, _sd = standardize_by_train_simple(train_raw, val_raw, test_raw)

        print(f"[RowSplit] rows={T} train_end={train_end} val_end={val_end} windows={len(train)+len(val)+len(test)} K={train.shape[-1]}")

    # -----------------------------
    # Loaders
    # -----------------------------
    train_loader = DataLoader(
        WindowDatasetWithIndex(train),
        batch_size=safe_batch_size(args.batch, len(train)),
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        WindowDatasetWithIndex(val),
        batch_size=safe_batch_size(args.batch, len(val)),
        shuffle=False
    )
    test_loader = DataLoader(
        WindowDatasetWithIndex(test),
        batch_size=safe_batch_size(args.batch, len(test)),
        shuffle=False
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
    # Eval
    # -----------------------------
    ratios = parse_ratios(args.eval_masked_ratios)

    file_exists = os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if not file_exists:
            f.write("seed\tsplit\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split, loader in [("val", val_loader), ("test", test_loader)]:
            for r in ratios:
                if args.use_shared_evalmask:
                    keep = load_shared_keepmask(args.shared_evalmask_dir, split, r, args.seq_len, args.seed)

                    if args.invert_shared_keepmask or os.environ.get("STOCK_MASKBANK_INVERT", "0") == "1":
                        keep = 1.0 - keep

                    # align keepmask channels to model K (Option A)
                    if valid_feature_mask is not None:
                        keep = align_keepmask_K(keep, expected_K=K, valid_feature_mask=valid_feature_mask)
                    else:
                        keep = align_keepmask_K(keep, expected_K=K, valid_feature_mask=None)

                    if keep.shape[0] != len(val) and split == "val":
                        raise ValueError(f"Shared keepmask N mismatch for val: keep.N={keep.shape[0]} val.N={len(val)}")
                    if keep.shape[0] != len(test) and split == "test":
                        raise ValueError(f"Shared keepmask N mismatch for test: keep.N={keep.shape[0]} test.N={len(test)}")

                    mae, mse, rmse = eval_shared(model, loader, device, keep)
                else:
                    mae, mse, rmse = eval_markov(model, loader, device, args.lm, r)

                f.write(f"{args.seed}\t{split}\t{r:.2f}\t{1-r:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{split}] r_masked={r:.2f} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()

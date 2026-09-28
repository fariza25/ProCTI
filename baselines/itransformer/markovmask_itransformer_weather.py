#!/usr/bin/env python3

import os
import sys
import time
import random
import argparse
from types import SimpleNamespace
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------------
# Make repo root importable
# -------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


try:
    from models.iTransformer import Model as iTransformerModel
except Exception:
    from model.iTransformer import Model as iTransformerModel


# -------------------------------------------------------
# Repro
# -------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ratios(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_batch_size(requested: int, n_items: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(int(requested), int(n_items)))


# -------------------------------------------------------
# Markov keep-mask (segment-based) with correct stationary distribution

def markov_keep_mask_from_masked_ratio(
    B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device
) -> torch.Tensor:
    """
    Returns (B,K,L) keepmask with 1=kept, 0=masked.
    r_masked is fraction masked (missingness).
    """
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


# -------------------------------------------------------
# Weather data pipeline 
# -------------------------------------------------------
class WindowedDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L,K)


def load_weather_windows(csv_path: str, seq_len: int, time_col: Optional[str]) -> Tuple[np.ndarray, List[str], str]:
    df = pd.read_csv(csv_path)

    if time_col is None:
        if "date" in df.columns:
            time_col = "date"
        elif "time" in df.columns:
            time_col = "time"
        else:
            time_col = df.columns[0]

    t = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[~t.isna()].copy()
    df["_t"] = pd.to_datetime(df[time_col])
    df = df.sort_values("_t", kind="mergesort").drop(columns=["_t"])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found in weather CSV.")
    X = df[num_cols].astype(np.float32).to_numpy()

    T, K = X.shape
    n_win = T // seq_len
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({T}) for seq_len={seq_len}")
    X = X[: n_win * seq_len].reshape(n_win, seq_len, K)  # (N,L,K)
    return X.astype(np.float32), num_cols, time_col


def chrono_split_windows(windows_NLK: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    n = len(windows_NLK)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = windows_NLK[:n_train]
    val = windows_NLK[n_train:n_train + n_val]
    test = windows_NLK[n_train + n_val:]
    return train, val, test


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.maximum(sd, eps).astype(np.float32)

    def fill_and_z(x):
        x = x.copy().astype(np.float32)
        nanmask = np.isnan(x)
        if nanmask.any():
            feat_idx = np.where(nanmask)[2]
            x[nanmask] = mu[feat_idx]
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd


def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seq_len: int, seed: int, invert: bool) -> np.ndarray:
    path = os.path.join(mask_dir, f"weather_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if invert or os.environ.get("WEATHER_MASKBANK_INVERT", "0") == "1":
        keep = 1.0 - keep
    return keep


# -------------------------------------------------------
# iTransformer helpers
# -------------------------------------------------------
def build_configs(args, K: int):
    return SimpleNamespace(
        task_name="imputation",
        seq_len=int(args.seq_len),
        pred_len=int(args.seq_len),
        enc_in=int(K),
        c_out=int(K),

        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        e_layers=int(args.e_layers),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        factor=int(args.factor),
        activation=str(args.activation),

        embed=str(args.embed),
        freq=str(args.freq),

        # kept for completeness
        num_class=int(args.num_class),
    )


def model_forward_impute(model: nn.Module, x_enc: torch.Tensor) -> torch.Tensor:
    """
    Returns imputed prediction with shape (B,L,K).
    Handles cases where model returns tuple/list.
    """
    out = model(x_enc, None, None, None, None)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# -------------------------------------------------------
# Train / Eval
# -------------------------------------------------------
def train_epoch(model, loader, opt, device, K: int, L: int, r_train_masked: float, lm: float, clip_grad: float):
    model.train()
    losses = []

    for xb in loader:
        xb = xb.to(device).float()  # (B,L,K)

        keep_BKL = markov_keep_mask_from_masked_ratio(B=xb.shape[0], K=K, L=L, r_masked=r_train_masked, lm=lm, device=device)
        keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()  # (B,L,K)

        x_obs = xb * keep_BLK

        pred = model_forward_impute(model, x_obs)

        # train on masked points only (so it learns imputation)
        evalmask = 1.0 - keep_BLK
        denom = evalmask.sum().clamp_min(1.0)
        loss = ((pred - xb).abs() * evalmask).sum() / denom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        opt.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_split(model, loader, split_name: str, ratios: List[float], device, K: int, L: int,
               lm: float, use_shared: bool, shared_dir: Optional[str], seed: int, invert_shared: bool):
    model.eval()
    rows = []

    for r_m in ratios:
        total_abs = 0.0
        total_sq = 0.0
        total_count = 0.0

        total_keep = 0.0
        total_keep_count = 0.0

        keep_NLK = None
        offset = 0
        if use_shared:
            keep_NLK = load_shared_keepmask(shared_dir, split_name, r_m, L, seed, invert_shared)

        for xb in loader:
            xb = xb.to(device).float()
            B = xb.shape[0]

            if keep_NLK is not None:
                keep_BLK = torch.from_numpy(keep_NLK[offset:offset + B]).to(device).float()
                offset += B
            else:
                keep_BKL = markov_keep_mask_from_masked_ratio(B=B, K=K, L=L, r_masked=r_m, lm=lm, device=device)
                keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

            x_obs = xb * keep_BLK
            pred = model_forward_impute(model, x_obs)

            evalmask = 1.0 - keep_BLK
            diff = (pred - xb) * evalmask

            total_abs += float(diff.abs().sum().item())
            total_sq += float((diff ** 2).sum().item())
            total_count += float(evalmask.sum().item())

            total_keep += float(keep_BLK.sum().item())
            total_keep_count += float(keep_BLK.numel())

        mae = total_abs / max(total_count, 1.0)
        mse = total_sq / max(total_count, 1.0)
        rmse = float(np.sqrt(mse))
        r_obs = total_keep / max(total_keep_count, 1.0)

        rows.append((split_name, r_m, r_obs, mae, mse, rmse))

    return rows


@torch.no_grad()
def save_test_arrays(model, loader, ratios: List[float], device, K: int, L: int, lm: float,
                     use_shared: bool, shared_dir: Optional[str], seed: int, invert_shared: bool,
                     save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    ns = 1

    for r_m in ratios:
        gt_all, imp_all, cond_all, eval_all = [], [], [], []
        offset = 0

        keep_NLK = None
        if use_shared:
            keep_NLK = load_shared_keepmask(shared_dir, "test", r_m, L, seed, invert_shared)

        for xb in loader:
            xb = xb.to(device).float()
            B = xb.shape[0]

            if keep_NLK is not None:
                keep_BLK = torch.from_numpy(keep_NLK[offset:offset + B]).to(device).float()
                offset += B
            else:
                keep_BKL = markov_keep_mask_from_masked_ratio(B=B, K=K, L=L, r_masked=r_m, lm=lm, device=device)
                keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

            x_obs = xb * keep_BLK
            pred = model_forward_impute(model, x_obs)

            imputed_full = keep_BLK * xb + (1.0 - keep_BLK) * pred
            evalmask = 1.0 - keep_BLK

            gt_all.append(xb.detach().cpu().numpy())
            imp_all.append(imputed_full.detach().cpu().numpy())
            cond_all.append(keep_BLK.detach().cpu().numpy())
            eval_all.append(evalmask.detach().cpu().numpy())

        gt_all = np.concatenate(gt_all, axis=0)
        imp_all = np.concatenate(imp_all, axis=0)
        cond_all = np.concatenate(cond_all, axis=0)
        eval_all = np.concatenate(eval_all, axis=0)

        tag = f"itransformer_weather_test_r{r_m:.2f}_seed{seed}_ns{ns}_L{L}"
        np.save(os.path.join(save_dir, f"{tag}_gt.npy"), gt_all)
        np.save(os.path.join(save_dir, f"{tag}_imputed.npy"), imp_all)
        np.save(os.path.join(save_dir, f"{tag}_condmask.npy"), cond_all)
        np.save(os.path.join(save_dir, f"{tag}_evalmask.npy"), eval_all)
        print(f"[save] {tag}_*.npy  shapes gt={gt_all.shape} imputed={imp_all.shape}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=96)

    # training
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip_grad", type=float, default=5.0)

    # iTransformer hyperparams (light defaults; tweak later)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--embed", type=str, default="fixed")
    ap.add_argument("--freq", type=str, default="h")
    ap.add_argument("--num_class", type=int, default=2)

    # masking
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # shared eval masks
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_weather_maskbank")
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    # outputs
    ap.add_argument("--out_txt", type=str, default="itransformer_weather_metrics.txt")
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_itransformer_weather")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        raise ValueError("--use_shared_evalmask requires --shared_evalmask_dir")

    # ---- data 
    windows, feat_cols, tcol = load_weather_windows(args.csv, args.seq_len, args.time_col)
    train_w, val_w, test_w = chrono_split_windows(windows)
    train_w, val_w, test_w, mu, sd = standardize_by_train(train_w, val_w, test_w)

    K = int(train_w.shape[-1])
    L = int(args.seq_len)
    print(f"[data] windows={len(windows)} train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={L}  time_col={tcol}")

    train_loader = DataLoader(WindowedDataset(train_w), batch_size=safe_batch_size(args.batch, len(train_w)),
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(WindowedDataset(val_w), batch_size=safe_batch_size(args.batch, len(val_w)),
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(WindowedDataset(test_w), batch_size=safe_batch_size(args.batch, len(test_w)),
                             shuffle=False, drop_last=False)

    # ---- model (faithful init: Model(configs))
    configs = build_configs(args, K)
    model = iTransformerModel(configs).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- train
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(
            model, train_loader, opt, device,
            K=K, L=L, r_train_masked=args.r_train_masked, lm=args.lm, clip_grad=args.clip_grad
        )
        if ep == 1 or ep % 10 == 0:
            print(f"[train] epoch={ep:03d} loss={loss:.6f} time={time.time()-t0:.1f}s")

    ratios = parse_ratios(args.eval_masked_ratios)

    # ---- eval + write metrics
    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    with open(args.out_txt, "a") as f:
        if os.stat(args.out_txt).st_size == 0:
            f.write("seed\tsplit\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            rows = eval_split(
                model, loader, split_name, ratios, device,
                K=K, L=L, lm=args.lm,
                use_shared=args.use_shared_evalmask,
                shared_dir=args.shared_evalmask_dir,
                seed=args.seed,
                invert_shared=args.invert_shared_keepmask
            )
            for (sp, r_m, r_obs, mae, mse, rmse) in rows:
                f.write(f"{args.seed}\t{sp}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{sp}] r_masked={r_m:.2f} r_obs={r_obs:.2f}  MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")

    # ---- save arrays (optional)
    if args.save_test_arrays:
        save_test_arrays(
            model, test_loader, ratios, device,
            K=K, L=L, lm=args.lm,
            use_shared=args.use_shared_evalmask,
            shared_dir=args.shared_evalmask_dir,
            seed=args.seed,
            invert_shared=args.invert_shared_keepmask,
            save_dir=args.save_dir
        )
        # metadata for convenience
        meta = {
            "mu": mu,
            "sd": sd,
            "feature_cols_used": np.array(feat_cols, dtype=object),
            "seq_len": int(args.seq_len),
            "shared_evalmask": bool(args.use_shared_evalmask),
            "shared_evalmask_dir": str(args.shared_evalmask_dir) if args.shared_evalmask_dir else "",
            "invert_shared_keepmask": bool(args.invert_shared_keepmask),
        }
        np.savez(os.path.join(args.save_dir, f"metadata_seed{args.seed}.npz"), **meta)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os, argparse, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from engine.solver import Trainer
from engine.logger import Logger


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


def parse_ratios(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# -----------------------------
# Markov keep-mask (segment-wise) 1=kept, 0=masked
# returns (B,K,L)
# -----------------------------
def markov_keep_mask(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device) -> torch.Tensor:
    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]  # state 0->1 uses p_m, state 1->0 uses p_u

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
# Data pipeline (row-split then window)
# -----------------------------
def load_stock(csv: str):
    df = pd.read_csv(csv)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(cols) == 0:
        raise ValueError("No numeric columns found in stock CSV.")
    X = df[cols].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X.astype(np.float32), cols


def split_rows_then_window(X: np.ndarray, seq_len: int):
    T = X.shape[0]
    train_end = int(0.70 * T)
    val_end   = int(0.85 * T)

    def win(seg: np.ndarray):
        Tseg, K = seg.shape
        n = Tseg // seq_len
        if n <= 0:
            return np.zeros((0, seq_len, K), dtype=np.float32)
        return seg[:n * seq_len].reshape(n, seq_len, K).astype(np.float32)

    return win(X[:train_end]), win(X[train_end:val_end]), win(X[val_end:])


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(flat, axis=0).astype(np.float32)
    sd = np.nanstd(flat, axis=0).astype(np.float32)

    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.maximum(sd, eps).astype(np.float32)

    def fill_and_z(x: np.ndarray):
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


class WindowDataset(Dataset):
    def __init__(self, arr: np.ndarray):
        self.x = arr.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i])  # (L,K)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--seq_len", type=int, default=48)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lm", type=float, default=6.0)

    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--coef", type=float, default=1e-2)
    ap.add_argument("--step_size", type=float, default=5e-2)
    ap.add_argument("--sampling_steps", type=int, default=250)

    # Diffusion-TS logger/trainer expected args
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--log_frequency", type=int, default=100)
    ap.add_argument("--milestone", type=int, default=0)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--name", type=str, default="diffusionts_stock")
    ap.add_argument("--output", type=str, default="OUTPUT")
    ap.add_argument("--out_txt", type=str, default="diffusionts_stock_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save dir like main.py
    args.save_dir = os.path.join(args.output, args.name)
    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------
    # Data
    # -----------------------------
    X, cols = load_stock(args.csv)
    train_raw, val_raw, test_raw = split_rows_then_window(X, args.seq_len)
    if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
        raise ValueError("One of the splits has 0 windows. Check seq_len or dataset length.")

    train_w, val_w, test_w, mu, sd = standardize_by_train(train_raw, val_raw, test_raw)
    K = int(train_w.shape[-1])
    L = int(args.seq_len)

    train_loader = DataLoader(
        WindowDataset(train_w),
        batch_size=safe_batch_size(args.batch, len(train_w)),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        WindowDataset(val_w),
        batch_size=safe_batch_size(args.batch, len(val_w)),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        WindowDataset(test_w),
        batch_size=safe_batch_size(args.batch, len(test_w)),
        shuffle=False,
        drop_last=False,
    )

    # -----------------------------
    # Model (Diffusion-TS)
    # -----------------------------
    model = Diffusion_TS(
        seq_length=L,
        feature_size=K,
        n_layer_enc=4,
        n_layer_dec=4,
        d_model=96,
        timesteps=1000,
        sampling_timesteps=250,
        loss_type="l1",
        beta_schedule="cosine",
        n_heads=4,
        use_ff=True,
        eta=0.0,
    ).to(device)

    config = {
        "solver": {
            "base_lr": args.lr,
            "max_epochs": args.steps,
            "gradient_accumulate_every": 2,
            "save_cycle": 2000,
            "results_folder": os.path.join(args.save_dir, "checkpoints"),
            "ema": {"decay": 0.995, "update_interval": 10},
            "scheduler": {
                "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
                "params": {
                    "factor": 0.5,
                    "patience": 1000,
                    "min_lr": args.lr,
                    "threshold": 1e-1,
                    "threshold_mode": "rel",
                    "warmup_lr": 8e-4,
                    "warmup": 500,
                    "verbose": False,
                },
            },
        }
    }

    logger = Logger(args)
    trainer = Trainer(config, args, model, {"dataloader": train_loader, "dataset": None}, logger)

    print(f"[data] train/val/test windows = {len(train_w)}/{len(val_w)}/{len(test_w)}  K={K}  L={L}")
    print("Training Diffusion-TS on STOCK...")
    trainer.train()

    # -----------------------------
    # Evaluation (Markov segment-wise) + save metrics
    # -----------------------------
    ema_model = trainer.ema.ema_model
    ema_model.eval()

    ratios = parse_ratios(args.eval_masked_ratios)
    rows = []

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        for r in ratios:
            total_abs = 0.0
            total_sq = 0.0
            total_den = 0.0
            total_robs_weighted = 0.0

            for xb in loader:
                xb = xb.to(device)

                keep_BKL = markov_keep_mask(xb.shape[0], K, L, r, args.lm, device)
                keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()  # (B,L,K)

                sample = ema_model.fast_sample_infill(
                    shape=xb.shape,
                    target=xb * keep_BLK,
                    partial_mask=keep_BLK.bool(),
                    model_kwargs={"coef": args.coef, "learning_rate": args.step_size},
                    sampling_timesteps=args.sampling_steps,
                )

                evalmask = 1.0 - keep_BLK
                diff = (sample - xb) * evalmask

                den = float(evalmask.sum().item())
                if den < 1.0:
                    continue

                total_abs += float(diff.abs().sum().item())
                total_sq += float((diff ** 2).sum().item())
                total_den += den
                total_robs_weighted += float(keep_BLK.mean().item()) * den

            total_den = max(total_den, 1.0)
            mae = total_abs / total_den
            mse = total_sq / total_den
            rmse = float(np.sqrt(mse))
            r_obs = float(total_robs_weighted / total_den)

            print(f"[{split_name}] r_masked={r:.2f}  r_obs={r_obs:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}")
            rows.append((split_name, r, r_obs, mae, mse, rmse))

    # write metrics
    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for split_name, r, r_obs, mae, mse, rmse in rows:
            f.write(f"{split_name}\t{r:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"[metrics] appended to {args.out_txt}")


if __name__ == "__main__":
    main()

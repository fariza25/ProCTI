#!/usr/bin/env python3

import os, argparse, random
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from engine.solver import Trainer
from engine.logger import Logger


# -------------------------------------------------
# Repro
# -------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ratios(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# -------------------------------------------------
# Markov keep-mask (segment-wise)
# keep=1 observed, 0 masked
# returns (B,K,L)
# -------------------------------------------------
def markov_keep_mask(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device) -> torch.Tensor:
    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]  # by state: 0->1 uses p_m, 1->0 uses p_u

    out = np.ones((B, K, L), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < p[state]:
                    state = 1 - state
    return torch.from_numpy(out).to(device=device)


# -------------------------------------------------
# Weather loading + chrono windowing (non-overlapping)
# -------------------------------------------------
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

    X = df[num_cols].astype(np.float32).to_numpy()  # (T,K)
    X[~np.isfinite(X)] = np.nan

    T, K = X.shape
    n_win = T // seq_len
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({T}) for seq_len={seq_len}")

    X = X[: n_win * seq_len].reshape(n_win, seq_len, K).astype(np.float32)  # (N,L,K)
    return X, num_cols, time_col


def chrono_split_windows(windows_NLK: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    n = len(windows_NLK)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = windows_NLK[:n_train]
    val = windows_NLK[n_train:n_train + n_val]
    test = windows_NLK[n_train + n_val:]
    return train, val, test


# -------------------------------------------------
# Z-score normalization (train-only), NaN fill by train mean, drop invalid channels
# -------------------------------------------------
def zscore_train_only_drop_invalid(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    """
    train/val/test are (N,L,K) with NaNs allowed.
    Returns standardized arrays with valid-feature filtering applied.
    """
    N, L, K = train.shape
    flat = train.reshape(-1, K)

    mu = np.nanmean(flat, axis=0).astype(np.float32)
    sd = np.nanstd(flat, axis=0).astype(np.float32)

    # valid channels: finite mu/sd and sd > eps
    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)

    if valid.sum() <= 0:
        raise ValueError("No valid features after filtering (all NaN or near-constant).")

    mu_v = mu[valid]
    sd_v = np.maximum(sd[valid], eps).astype(np.float32)

    def fill_and_z(x: np.ndarray):
        x = x[:, :, valid].copy().astype(np.float32)
        nanmask = ~np.isfinite(x)
        if nanmask.any():
            feat_idx = np.where(nanmask)[2]
            x[nanmask] = mu_v[feat_idx]
        x = (x - mu_v[None, None, :]) / sd_v[None, None, :]
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), valid, mu_v, sd_v


# -------------------------------------------------
# Datasets
# -------------------------------------------------
class WindowDataset(Dataset):
    """Returns full standardized windows (L,K)."""
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L,K)


class MaskedTrainDataset(Dataset):
    """
    Returns masked standardized windows for TRAINING:
      x_masked = x * keep, with masked entries set to 0.
    Markov keepmask is generated per-sample, per-channel across time (segment-wise).
    """
    def __init__(self, windows_NLK: np.ndarray, r_train_masked: float, lm_train: float, seed: int):
        self.x = windows_NLK.astype(np.float32)
        self.r = float(r_train_masked)
        self.lm = float(lm_train)
        self.seed = int(seed)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        # deterministic per-idx randomness (so training is reproducible for a given seed)
        rng = np.random.RandomState(self.seed * 1000003 + idx)
        x = self.x[idx]  # (L,K)

        L, K = x.shape
        # generate keepmask in numpy (K,L)
        r_keep = 1.0 - self.r
        p_m = 1.0 / self.lm
        p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
        p = [p_m, p_u]

        keep_KL = np.ones((K, L), dtype=np.float32)
        for k in range(K):
            state = int(rng.rand() < r_keep)
            for t in range(L):
                keep_KL[k, t] = state
                if rng.rand() < p[state]:
                    state = 1 - state

        keep_LK = keep_KL.T  # (L,K)
        x_masked = x * keep_LK  # masked entries become 0
        return torch.from_numpy(x_masked.astype(np.float32))


# -------------------------------------------------
# Shared maskbank loader (optional)
# -------------------------------------------------
def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seq_len: int, seed: int, invert: bool) -> np.ndarray:
    path = os.path.join(mask_dir, f"weather_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if invert:
        keep = 1.0 - keep
    return keep  # (N,L,K), keep=1 observed


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=96)

    # Training
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--save_cycle", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)

    # Align with training masking regime
    ap.add_argument("--r_train_masked", type=float, default=0.10)
    ap.add_argument("--lm_train", type=float, default=6.0)

    # Eval masking
    ap.add_argument("--lm_eval", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # Diffusion-TS restore knobs
    ap.add_argument("--coef", type=float, default=1e-2)
    ap.add_argument("--step_size", type=float, default=5e-2)
    ap.add_argument("--sampling_steps", type=int, default=250)

    # Shared eval masks (optional)
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_weather_maskbank")
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    # Diffusion-TS logger expected args
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--log_frequency", type=int, default=100)
    ap.add_argument("--milestone", type=int, default=0)
    ap.add_argument("--resume", action="store_true")

    # Output
    ap.add_argument("--name", type=str, default="diffusionts_weather_aligned")
    ap.add_argument("--output", type=str, default="OUTPUT")
    ap.add_argument("--out_txt", type=str, default="diffusionts_weather_aligned_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save dir like main.py
    args.save_dir = os.path.join(args.output, args.name)
    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------
    # Data: chrono windowing
    # -----------------------------
    windows, feat_cols, tcol = load_weather_windows(args.csv, args.seq_len, args.time_col)
    train_raw, val_raw, test_raw = chrono_split_windows(windows, 0.7, 0.15)

    # Normalize: z-score on TRAIN
    train_w, val_w, test_w, valid_mask, mu_v, sd_v = zscore_train_only_drop_invalid(train_raw, val_raw, test_raw)

    K = int(train_w.shape[-1])
    L = int(args.seq_len)
    print(f"[data] windows={len(windows)} train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  L={L} K={K} time_col={tcol}")

    # Training dataset uses Markov masking
    train_ds = MaskedTrainDataset(train_w, r_train_masked=args.r_train_masked, lm_train=args.lm_train, seed=args.seed)
    val_ds = WindowDataset(val_w)
    test_ds = WindowDataset(test_w)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

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
        use_ff=True,   # keep True to match common Diffusion-TS configs
        eta=0.0
    ).to(device)

    config = {
        "solver": {
            "base_lr": args.lr,
            "max_epochs": args.steps,
            "gradient_accumulate_every": args.grad_accum,
            "save_cycle": args.save_cycle,
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
                    "verbose": False
                }
            }
        }
    }

    logger = Logger(args)
    trainer = Trainer(config=config, args=args, model=model, dataloader={"dataloader": train_loader, "dataset": None}, logger=logger)

    print("[train] training with Markov-masked inputs")
    trainer.train()

    # -----------------------------
    # Evaluation: Markov segment-wise + save metrics
    # -----------------------------
    ema_model = trainer.ema.ema_model
    ema_model.eval()

    ratios = parse_ratios(args.eval_masked_ratios)
    rows = []

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        keep_NLK = None
        for r_m in ratios:
            total_abs = 0.0
            total_sq = 0.0
            total_den = 0.0
            total_robs_weighted = 0.0

            if args.use_shared_evalmask:
                keep_NLK = load_shared_keepmask(
                    args.shared_evalmask_dir, split=split_name, r_m=r_m,
                    seq_len=L, seed=args.seed, invert=args.invert_shared_keepmask
                )
                offset = 0
            else:
                offset = 0

            for xb in loader:
                xb = xb.to(device).float()  # (B,L,K)
                B = xb.shape[0]

                if keep_NLK is not None:
                    keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device).float()
                    offset += B
                else:
                    keep_BKL = markov_keep_mask(B, K, L, r_m, args.lm_eval, device)
                    keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

                sample = ema_model.fast_sample_infill(
                    shape=xb.shape,
                    target=xb * keep_BLK,
                    partial_mask=keep_BLK.bool(),
                    model_kwargs={"coef": args.coef, "learning_rate": args.step_size},
                    sampling_timesteps=args.sampling_steps
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

            print(f"[{split_name}] r_masked={r_m:.2f}  r_obs={r_obs:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}")
            rows.append((split_name, r_m, r_obs, mae, mse, rmse))

    # save metrics
    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for split, r_m, r_obs, mae, mse, rmse in rows:
            f.write(f"{split}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"[metrics] appended to {args.out_txt}")


if __name__ == "__main__":
    main()

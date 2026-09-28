#!/usr/bin/env python3


import os
import re
import sys
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------
# reproducibility
# -------------------------------------------------

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


# -------------------------------------------------
# Markov keep-mask
# returns (B, K, L), 1=observed/kept, 0=masked
# -------------------------------------------------

def markov_keep_mask(B: int, K: int, L: int, r_masked: float, lm: float, device):
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0

    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)

    out = np.ones((B, K, L), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < (p_m if state == 0 else p_u):
                    state = 1 - state
    return torch.from_numpy(out).to(device)


# -------------------------------------------------
# Dataset
# -------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, windows_nlk: np.ndarray):
        self.x = windows_nlk.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L, K)


# -------------------------------------------------
# Gait preprocessing: matches BRITS gait script
# -------------------------------------------------

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
    df = pd.read_csv(path, skiprows=2, header=None)
    arr = df.values.astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Bad array shape from {path}: {arr.shape}")
    return arr


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
        return np.zeros((0, seq_len, 1), dtype=np.float32)
    return np.concatenate(windows, axis=0).astype(np.float32)


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps: float = 1e-6):
    if len(train) == 0:
        raise ValueError("No training windows available for standardization.")

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
            x[nanmask] = np.take(mu, np.where(nanmask)[2])
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd




def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seq_len: int, seed: int) -> np.ndarray:
    path = os.path.join(mask_dir, f"gait_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if keep.ndim != 3:
        raise ValueError(f"Shared keepmask must have shape (N,L,K) or (N,K,L), got {keep.shape}")
    return keep


def normalize_keepmask_shape(keep: np.ndarray, L: int, K: int) -> np.ndarray:
    if keep.shape[1] == L and keep.shape[2] == K:
        return keep
    if keep.shape[1] == K and keep.shape[2] == L:
        return np.transpose(keep, (0, 2, 1))
    raise ValueError(f"Cannot align keepmask shape {keep.shape} to expected (N,{L},{K})")


# -------------------------------------------------
# PaD-TS imports
# -------------------------------------------------

def import_padts(root: str):
    if root not in sys.path:
        sys.path.insert(0, root)

    from Model import PaD_TS
    from diffmodel_init import create_gaussian_diffusion
    from training import Trainer
    from resample import UniformSampler, Batch_Same_Sampler

    return PaD_TS, create_gaussian_diffusion, Trainer, UniformSampler, Batch_Same_Sampler


# -------------------------------------------------
# diffusion imputation by projection
# -------------------------------------------------

@torch.no_grad()
def diffusion_impute_project(model, diffusion, x0_blk: torch.Tensor, keep_blk: torch.Tensor):
    device = x0_blk.device
    B, L, K = x0_blk.shape

    x_t = torch.randn_like(x0_blk)
    eps_obs = torch.randn_like(x0_blk)

    T = diffusion.num_timesteps

    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        out = diffusion.p_sample(model, x_t, t_tensor, clip_denoised=True)
        x_prev = out["sample"]

        if t > 0:
            t_prev = torch.full((B,), t - 1, device=device, dtype=torch.long)
            x_obs_prev = diffusion.q_sample(x0_blk, t_prev, noise=eps_obs)
        else:
            x_obs_prev = x0_blk

        x_t = keep_blk * x_obs_prev + (1.0 - keep_blk) * x_prev

    return x_t


# -------------------------------------------------
# evaluation
# -------------------------------------------------

@torch.no_grad()
def evaluate_split(
    model,
    diffusion,
    windows_nlk: np.ndarray,
    ratios,
    lm: float,
    device,
    batch_size: int,
    use_shared_evalmask: bool = False,
    shared_evalmask_dir: str = None,
    seq_len: int = None,
    seed: int = None,
    split_name: str = "test",
):
    ds = WindowDataset(windows_nlk)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    results = []

    shared_masks = {}
    if use_shared_evalmask:
        for r in ratios:
            keep = load_shared_keepmask(shared_evalmask_dir, split_name, r, seq_len, seed)
            keep = normalize_keepmask_shape(keep, windows_nlk.shape[1], windows_nlk.shape[2])
            if keep.shape[0] != len(windows_nlk):
                raise ValueError(
                    f"Shared keepmask count mismatch for {split_name}, r={r:.2f}: "
                    f"{keep.shape[0]} masks vs {len(windows_nlk)} windows"
                )
            shared_masks[r] = keep

    for r in ratios:
        sum_abs = 0.0
        sum_sq = 0.0
        sum_den = 0.0
        global_idx = 0

        for x in loader:
            x = x.to(device)
            B, L, K = x.shape

            if use_shared_evalmask:
                keep = torch.from_numpy(shared_masks[r][global_idx:global_idx + B]).to(device)
            else:
                keep = markov_keep_mask(B, K, L, r, lm, device).permute(0, 2, 1)

            xhat = diffusion_impute_project(model, diffusion, x, keep)

            evalmask = 1.0 - keep
            diff = (xhat - x) * evalmask

            sum_abs += diff.abs().sum().item()
            sum_sq += (diff ** 2).sum().item()
            sum_den += evalmask.sum().item()

            global_idx += B

        denom = max(sum_den, 1.0)
        mae = sum_abs / denom
        mse = sum_sq / denom
        rmse = float(np.sqrt(mse))
        r_obs = 1.0 - r

        results.append((r, r_obs, mae, mse, rmse))

    return results


# -------------------------------------------------
# main
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--padts_root", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # PaD-TS model args
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--n_encoder", type=int, default=1)
    ap.add_argument("--n_decoder", type=int, default=3)
    ap.add_argument("--feature_last", action="store_true", default=True)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)

    # diffusion args
    ap.add_argument("--diffusion_steps", type=int, default=250)
    ap.add_argument("--noise_schedule", type=str, default="cosine")
    ap.add_argument("--loss", type=str, default="MSE_MMD")
    ap.add_argument("--predict_xstart", action="store_true", default=True)
    ap.add_argument("--rescale_timesteps", action="store_true", default=False)

    # trainer args
    ap.add_argument("--schedule_sampler", type=str, default="batch", choices=["batch", "uniform"])
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--mmd_alpha", type=float, default=0.0005)
    ap.add_argument("--save_dir", type=str, default="OUTPUT/padts_gait/")

    # evaluation
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # optional shared eval maskbank
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default=None)

    # output
    ap.add_argument("--out_txt", type=str, default="padts_gait_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ratios = parse_ratios(args.eval_ratios)

    print("Loading gait dataset...")
    files = list_gait_files(args.data_dir)
    train_files, val_files, test_files, split_users_meta = split_users(files, args.seed)

    print(f"[GAIT] files total/train/val/test = {len(files)}/{len(train_files)}/{len(val_files)}/{len(test_files)}")
    print(f"[GAIT] users train/val/test = {len(split_users_meta[0])}/{len(split_users_meta[1])}/{len(split_users_meta[2])}")

    train = windows_from_files(train_files, args.seq_len)
    val = windows_from_files(val_files, args.seq_len)
    test = windows_from_files(test_files, args.seq_len)

    if len(train) == 0:
        raise ValueError("No training windows were created. Reduce --seq_len or check gait CSV lengths.")
    if len(val) == 0:
        raise ValueError("No validation windows were created. Reduce --seq_len or check gait CSV lengths.")
    if len(test) == 0:
        raise ValueError("No test windows were created. Reduce --seq_len or check gait CSV lengths.")

    train, val, test, mu, sd = standardize_by_train(train, val, test)

    L = train.shape[1]
    K = train.shape[2]
    print(f"[GAIT] train/val/test windows = {len(train)}/{len(val)}/{len(test)}")
    print(f"[GAIT] sequence length = {L}, features = {K}")

    print("Importing PaD-TS...")
    PaD_TS, create_gaussian_diffusion, Trainer, UniformSampler, Batch_Same_Sampler = import_padts(args.padts_root)

    model = PaD_TS(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        n_encoder=args.n_encoder,
        n_decoder=args.n_decoder,
        feature_last=args.feature_last,
        mlp_ratio=args.mlp_ratio,
        input_shape=(args.seq_len, K),
    )

    diffusion = create_gaussian_diffusion(
        predict_xstart=args.predict_xstart,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        loss=args.loss,
        rescale_timesteps=args.rescale_timesteps,
    )

    train_bs = safe_batch_size(args.batch, len(train))
    train_loader = DataLoader(
        WindowDataset(train),
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[GAIT] train batch size = {train_bs}")
    print(f"[GAIT] train batches = {len(train_loader)}")

    if args.schedule_sampler == "batch":
        schedule_sampler = Batch_Same_Sampler(diffusion)
    else:
        schedule_sampler = UniformSampler(diffusion)

    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=train_bs,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.train_steps,
        save_dir=args.save_dir if args.save_dir.endswith("/") else args.save_dir + "/",
        mmd_alpha=args.mmd_alpha,
    )

    print("====== Training ======")
    trainer.train()
    print("====== Training done ======")

    model.eval()
    model.to(device)

    print("====== Evaluation ======")
    val_res = evaluate_split(
        model=model,
        diffusion=diffusion,
        windows_nlk=val,
        ratios=ratios,
        lm=args.lm,
        device=device,
        batch_size=train_bs,
        use_shared_evalmask=args.use_shared_evalmask,
        shared_evalmask_dir=args.shared_evalmask_dir,
        seq_len=args.seq_len,
        seed=args.seed,
        split_name="val",
    )

    test_res = evaluate_split(
        model=model,
        diffusion=diffusion,
        windows_nlk=test,
        ratios=ratios,
        lm=args.lm,
        device=device,
        batch_size=train_bs,
        use_shared_evalmask=args.use_shared_evalmask,
        shared_evalmask_dir=args.shared_evalmask_dir,
        seq_len=args.seq_len,
        seed=args.seed,
        split_name="test",
    )

    header_needed = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if header_needed:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for r, r_obs, mae, mse, rmse in val_res:
            f.write(f"val\t{r:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        for r, r_obs, mae, mse, rmse in test_res:
            f.write(f"test\t{r:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"[saved] metrics -> {args.out_txt}")


if __name__ == "__main__":
    main()

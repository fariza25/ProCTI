#!/usr/bin/env python3

import os, re, argparse, random, time
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from engine.solver import Trainer
from engine.logger import Logger


# ============================================================
# Repro
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# GAIT Preprocessing 
# ============================================================

USER_RE = re.compile(r"_ID(\d+)_", re.IGNORECASE)

def user_id_from_name(path):
    m = USER_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse user ID from {path}")
    return m.group(1)

def read_gait_csv(path):
    df = pd.read_csv(path, skiprows=2, header=None)
    return df.values.astype(np.float32)

def windows_from_files(file_list, seq_len):
    windows = []
    for p in file_list:
        arr = read_gait_csv(p)
        T, K = arr.shape
        n_win = T // seq_len
        if n_win > 0:
            windows.append(arr[:n_win*seq_len].reshape(n_win, seq_len, K))
    return np.concatenate(windows, axis=0)

def split_users(files, seed):
    by_user: Dict[str, List[str]] = {}
    for p in files:
        uid = user_id_from_name(p)
        by_user.setdefault(uid, []).append(p)

    users = sorted(by_user.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    n = len(users)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)

    train_u = users[:n_train]
    val_u   = users[n_train:n_train+n_val]
    test_u  = users[n_train+n_val:]

    def gather(uids):
        out = []
        for u in uids:
            out.extend(by_user[u])
        return out

    return gather(train_u), gather(val_u), gather(test_u)

def standardize_by_train(train, val, test, eps=1e-6):
    flat = train.reshape(-1, train.shape[-1])
    mu = np.mean(flat, axis=0)
    sd = np.std(flat, axis=0)
    sd = np.maximum(sd, eps)

    def z(x):
        return (x - mu[None,None,:]) / sd[None,None,:]

    return z(train), z(val), z(test)


class WindowDataset(Dataset):
    def __init__(self, arr):
        self.x = arr.astype(np.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


# ============================================================
# Markov Segment Mask
# ============================================================

def markov_mask(B,K,L,r_masked,lm,device):
    r_keep = 1. - r_masked
    p_m = 1./lm
    p_u = p_m*(1-r_keep)/max(r_keep,1e-12)

    out = np.ones((B,K,L),dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand()<r_keep)
            for t in range(L):
                out[b,k,t]=state
                if np.random.rand()<(p_m if state==0 else p_u):
                    state=1-state
    return torch.from_numpy(out).to(device)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=64)

    # Training
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lm", type=float, default=6.0)

    # Diffusion-TS required args
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--log_frequency", type=int, default=100)
    ap.add_argument("--milestone", type=int, default=0)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--name", type=str, default="diffusionts_gait")
    ap.add_argument("--output", type=str, default="OUTPUT")
    ap.add_argument("--out_txt", type=str, default="diffusionts_gait_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare save directory exactly like main.py
    args.save_dir = os.path.join(args.output, args.name)
    os.makedirs(args.save_dir, exist_ok=True)

    # ============================================================
    # Preprocessing
    # ============================================================

    files = [os.path.join(args.data_dir,f) for f in os.listdir(args.data_dir) if f.endswith(".csv")]
    tr_files, va_files, te_files = split_users(files, args.seed)

    train_raw = windows_from_files(tr_files,args.seq_len)
    val_raw   = windows_from_files(va_files,args.seq_len)
    test_raw  = windows_from_files(te_files,args.seq_len)

    train_w,val_w,test_w = standardize_by_train(train_raw,val_raw,test_raw)

    K = train_w.shape[-1]
    L = args.seq_len

    train_loader = DataLoader(WindowDataset(train_w), batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(WindowDataset(val_w), batch_size=args.batch)
    test_loader  = DataLoader(WindowDataset(test_w), batch_size=args.batch)

    # ============================================================
    # Diffusion-TS Model
    # ============================================================

    model = Diffusion_TS(
        seq_length=L,
        feature_size=K,
        n_layer_enc=4,
        n_layer_dec=4,
        d_model=96,
        timesteps=1000,
        sampling_timesteps=250,
        loss_type='l1',
        beta_schedule='cosine',
        n_heads=4,
        use_ff=True,
        eta=0.0
    ).to(device)

    config = {
        "solver":{
            "base_lr":args.lr,
            "max_epochs":args.steps,
            "gradient_accumulate_every":2,
            "save_cycle":2000,
            "results_folder":os.path.join(args.save_dir,"checkpoints"),
            "ema":{"decay":0.995,"update_interval":10},
            "scheduler":{
                "target":"engine.lr_sch.ReduceLROnPlateauWithWarmup",
                "params":{
                    "factor":0.5,
                    "patience":1000,
                    "min_lr":args.lr,
                    "threshold":1e-1,
                    "threshold_mode":"rel",
                    "warmup_lr":8e-4,
                    "warmup":500,
                    "verbose":False
                }
            }
        }
    }

    logger = Logger(args)
    trainer = Trainer(config,args,model,{"dataloader":train_loader,"dataset":None},logger)

    print("Training Diffusion-TS on GAIT...")
    trainer.train()

    # ============================================================
    # Evaluation (Markov segment-wise) + save metrics
    # ============================================================

    model = trainer.ema.ema_model
    model.eval()

    ratios = [0.10, 0.30, 0.50, 0.70]
    rows = []

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        for r in ratios:
            total_abs = 0.0
            total_sq  = 0.0
            total_den = 0.0
            total_robs_weighted = 0.0

            for xb in loader:
                xb = xb.to(device)
                keep_BKL = markov_mask(xb.shape[0], K, L, r, args.lm, device)
                keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()  # (B,L,K)

                sample = model.fast_sample_infill(
                    shape=xb.shape,
                    target=xb * keep_BLK,
                    partial_mask=keep_BLK.bool(),
                    model_kwargs={"coef": 1e-2, "learning_rate": 5e-2},
                    sampling_timesteps=250
                )

                evalmask = 1.0 - keep_BLK
                diff = (sample - xb) * evalmask

                den = float(evalmask.sum().item())
                if den < 1.0:
                    continue

                total_abs += float(diff.abs().sum().item())
                total_sq  += float((diff ** 2).sum().item())
                total_den += den

                # r_observed averaged over masked positions weight (to be comparable across batches)
                total_robs_weighted += float(keep_BLK.mean().item()) * den

            total_den = max(total_den, 1.0)
            mae = total_abs / total_den
            mse = total_sq / total_den
            rmse = float(np.sqrt(mse))
            r_obs = float(total_robs_weighted / total_den)

            print(f"[{split_name}] r_masked={r:.2f}  r_obs={r_obs:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}")
            rows.append((split_name, r, r_obs, mae, mse, rmse))

    # Write / append metrics file
    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for split_name, r, r_obs, mae, mse, rmse in rows:
            f.write(f"{split_name}\t{r:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"[metrics] appended to {args.out_txt}")

if __name__=="__main__":
    main()

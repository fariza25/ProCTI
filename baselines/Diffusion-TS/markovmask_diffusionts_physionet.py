#!/usr/bin/env python3

import os, argparse, random, time
from typing import Set, Tuple, List

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
# Markov Mask
# ============================================================

def markov_keep_mask_from_masked_ratio(B,K,L,r_masked,lm,device):
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
# PhysioNet Preprocessing
# ============================================================

def load_physionet_df(csv):
    df = pd.read_csv(csv)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Patient_ID"] = pd.to_numeric(df["Patient_ID"], errors="coerce")
    df = df.dropna(subset=["Patient_ID"])
    df["Patient_ID"] = df["Patient_ID"].astype(int)
    return df

def split_patients(df, seed):
    pids = df["Patient_ID"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    n = len(pids)
    n_train = int(0.7*n)
    n_val   = int(0.15*n)

    return (
        set(pids[:n_train]),
        set(pids[n_train:n_train+n_val]),
        set(pids[n_train+n_val:])
    )

def make_windows(df, patient_ids, feat_cols, time_col, seq_len):
    windows=[]
    for pid,g in df.groupby("Patient_ID",sort=False):
        if pid not in patient_ids:
            continue
        g=g.sort_values(time_col,kind="mergesort")
        arr=g[feat_cols].astype(np.float32).ffill().bfill().to_numpy()
        T,K=arr.shape
        n_win=T//seq_len
        if n_win>0:
            windows.append(arr[:n_win*seq_len].reshape(n_win,seq_len,K))
    return np.concatenate(windows,axis=0)

def standardize_by_train(train,val,test,eps=1e-6):
    flat=train.reshape(-1,train.shape[-1])
    mu=np.nanmean(flat,axis=0)
    sd=np.nanstd(flat,axis=0)
    valid=np.isfinite(mu)&np.isfinite(sd)&(sd>eps)

    train=train[:,:,valid]
    val=val[:,:,valid]
    test=test[:,:,valid]

    mu=mu[valid]; sd=np.maximum(sd[valid],eps)

    def z(x):
        x=x.copy()
        nanmask=np.isnan(x)
        if nanmask.any():
            ks=np.where(nanmask)[2]
            x[nanmask]=np.take(mu,ks)
        return ((x-mu)/sd).astype(np.float32)

    return z(train),z(val),z(test),valid


class WindowDataset(Dataset):
    def __init__(self,arr):
        self.x=arr.astype(np.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,i):
        return torch.from_numpy(self.x[i])


# ============================================================
# Main
# ============================================================

def main():
    ap=argparse.ArgumentParser()

    ap.add_argument("--csv",required=True)
    ap.add_argument("--seq_len",type=int,default=96)

    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--steps",type=int,default=20000)
    ap.add_argument("--lr",type=float,default=1e-5)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--lm",type=float,default=6.0)

    # Diffusion-TS required
    ap.add_argument("--tensorboard",action="store_true")
    ap.add_argument("--log_frequency",type=int,default=100)
    ap.add_argument("--milestone",type=int,default=0)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--name",type=str,default="diffusionts_physionet")
    ap.add_argument("--output",type=str,default="OUTPUT")
    ap.add_argument("--out_txt", type=str, default="diffusionts_physionet_metrics.txt")

    args=ap.parse_args()
    set_seed(args.seed)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.save_dir=os.path.join(args.output,args.name)
    os.makedirs(args.save_dir,exist_ok=True)

    # ============================================================
    # Data
    # ============================================================

    df=load_physionet_df(args.csv)
    time_col="ICULOS" if "ICULOS" in df.columns else "Hour"

    feat_cols=[c for c in df.select_dtypes(include=[np.number]).columns
               if c not in ["Patient_ID","SepsisLabel","Hour","ICULOS"]]

    train_ids,val_ids,test_ids=split_patients(df,args.seed)

    train_raw=make_windows(df,train_ids,feat_cols,time_col,args.seq_len)
    val_raw  =make_windows(df,val_ids,feat_cols,time_col,args.seq_len)
    test_raw =make_windows(df,test_ids,feat_cols,time_col,args.seq_len)

    train_w,val_w,test_w,valid=standardize_by_train(train_raw,val_raw,test_raw)

    K=train_w.shape[-1]
    L=args.seq_len

    train_loader=DataLoader(WindowDataset(train_w),batch_size=args.batch,shuffle=True,drop_last=True)
    val_loader=DataLoader(WindowDataset(val_w),batch_size=args.batch)
    test_loader=DataLoader(WindowDataset(test_w),batch_size=args.batch)

    # ============================================================
    # Model (faithful to Diffusion-TS)
    # ============================================================

    model=Diffusion_TS(
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

    config={
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

    logger=Logger(args)
    trainer=Trainer(config,args,model,{"dataloader":train_loader,"dataset":None},logger)

    print("Training Diffusion-TS on PhysioNet...")
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

                keep = markov_keep_mask_from_masked_ratio(
                    xb.shape[0], K, L, r, args.lm, device
                ).permute(0, 2, 1).contiguous()  # (B,L,K)

                sample = model.fast_sample_infill(
                    shape=xb.shape,
                    target=xb * keep,
                    partial_mask=keep.bool(),
                    model_kwargs={"coef": 1e-2, "learning_rate": 5e-2},
                    sampling_timesteps=250
                )

                evalmask = 1.0 - keep
                diff = (sample - xb) * evalmask

                den = float(evalmask.sum().item())
                if den < 1.0:
                    continue

                total_abs += float(diff.abs().sum().item())
                total_sq  += float((diff ** 2).sum().item())
                total_den += den

                # weighted r_observed for comparability across batches
                total_robs_weighted += float(keep.mean().item()) * den

            total_den = max(total_den, 1.0)
            mae = total_abs / total_den
            mse = total_sq / total_den
            rmse = float(np.sqrt(mse))
            r_obs = float(total_robs_weighted / total_den)

            print(f"[{split_name}] r_masked={r:.2f}  r_obs={r_obs:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}")
            rows.append((split_name, r, r_obs, mae, mse, rmse))

    # append metrics to out_txt (tab-separated)
    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for split_name, r, r_obs, mae, mse, rmse in rows:
            f.write(f"{split_name}\t{r:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"[metrics] appended to {args.out_txt}")

if __name__=="__main__":
    main()
